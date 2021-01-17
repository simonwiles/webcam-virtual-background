#!/usr/bin/env python3

""" Virtual Background Webcam Proxy """

import cv2
import pyfakewebcam
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

    while True:
        frame = get_frame(cap)
        frame = starwars_hologram(frame)
        # fake webcam expects RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fake.schedule_frame(frame)


if __name__ == "__main__":
    stream("/dev/video20")
