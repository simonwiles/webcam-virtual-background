#!/usr/bin/env python3

""" Virtual Background Webcam Proxy """

import cv2
import pyfakewebcam
import requests

import numpy as np

# configure camera for 480p @ 60 FPS
height, width = 480, 640
fps = 60


# shift_img from: https://stackoverflow.com/a/53140617
def shift_img(img, dx, dy):
    img = np.roll(img, dy, axis=0)
    img = np.roll(img, dx, axis=1)
    if dy > 0:
        img[:dy, :] = 0
    elif dy < 0:
        img[dy:, :] = 0
    if dx > 0:
        img[:, :dx] = 0
    elif dx < 0:
        img[:, dx:] = 0
    return img


def starwars_hologram(frame):

    # add a blue tint
    holo = cv2.applyColorMap(frame, cv2.COLORMAP_WINTER)

    # add a halftone effect
    bandLength, bandGap = 1, 2
    for y in range(holo.shape[0]):
        if y % (bandLength + bandGap) < bandLength:
            holo[y, :, :] = holo[y, :, :] * np.random.uniform(0.1, 0.3)

    # add some ghosting
    holo_blur = cv2.addWeighted(holo, 0.2, shift_img(holo.copy(), 5, 5), 0.8, 0)
    holo_blur = cv2.addWeighted(holo_blur, 0.4, shift_img(holo.copy(), -5, -5), 0.6, 0)

    # combine with the original color, oversaturated
    new_frame = cv2.addWeighted(frame, 0.5, holo_blur, 0.6, 0)
    return new_frame


def get_mask(frame, bodypix_url="http://localhost:9000"):
    _, data = cv2.imencode(".jpg", frame)
    r = requests.post(
        url=bodypix_url,
        data=data.tobytes(),
        headers={"Content-Type": "application/octet-stream"},
    )
    mask = np.frombuffer(r.content, dtype=np.uint8)
    mask = mask.reshape((frame.shape[0], frame.shape[1]))
    return mask


def post_process_mask(mask):
    mask = cv2.dilate(mask, np.ones((10, 10), np.uint8), iterations=1)
    # mask = cv2.erode(mask, np.ones((10, 10), np.uint8), iterations=1)
    mask = cv2.blur(mask.astype(float), (30, 30))
    return mask


def init_capture(device="/dev/video0"):
    cap = cv2.VideoCapture(device)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    return cap


def get_frame(cap):
    _success, frame = cap.read()
    return frame


def stream(output_device):
    """ Command-line entry-point. """

    # initialize capture from the device
    cap = init_capture()

    # initialize the fake camera
    fake = pyfakewebcam.FakeWebcam(output_device, width, height)

    background = cv2.imread("/home/simon/Star-Destroyer.640.jpg")
    background_scaled = cv2.resize(background, (width, height))

    print("Initialization complete, waiting for bodypix...")
    while True:
        frame = get_frame(cap)

        # fetch the mask with retries (the app needs to warmup and we're lazy)
        # e v e n t u a l l y c o n s i s t e n t
        mask = None
        while mask is None:
            try:
                mask = get_mask(frame)
            except requests.RequestException:
                print("mask request failed, retrying")
        mask = post_process_mask(mask)
        frame = starwars_hologram(frame)

        # composite the foreground and background
        inv_mask = 1 - mask
        for c in range(frame.shape[2]):
            frame[:, :, c] = (
                frame[:, :, c] * mask + background_scaled[:, :, c] * inv_mask
            )

        # fake webcam expects RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fake.schedule_frame(frame)


if __name__ == "__main__":
    stream("/dev/video20")
