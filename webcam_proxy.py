#!/usr/bin/env python3

""" Virtual Background Webcam Proxy """

import cv2
import pyfakewebcam

# configure camera for 480p @ 60 FPS
height, width = 480, 640
fps = 60


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
        fake.schedule_frame(frame)


if __name__ == "__main__":
    stream("/dev/video20")
