#!/home/pi/.virtualenvs/owl/bin/python3

from image_sampler import bounding_box_image_sample, square_image_sample, whole_image_save
from utils.blur_algorithms import fft_blur
from greenonbrown import GreenOnBrown
from utils.frame_reader import FrameReader

from datetime import datetime, timezone
from imutils.video import VideoStream, FPS
from queue import Queue
from time import strftime
from threading import Thread

import numpy as np
import argparse
import imutils
import json
import time
import sys
import cv2
import os


def nothing(x):
    pass


class Owl:
    def __init__(self,
                 input_file_or_directory=None,
                 show_display=False,
                 focus=False,
                 nozzleNum=4,
                 exgMin=30,
                 exgMax=180,
                 hueMin=30,
                 hueMax=92,
                 brightnessMin=5,
                 brightnessMax=200,
                 saturationMin=30,
                 saturationMax=255,
                 resolution=(416, 320),
                 framerate=32,
                 exp_mode='sports',
                 awb_mode='auto',
                 sensor_mode=0,
                 exp_compensation=-4,
                 parameters_json=None,
                 image_loop_time=5):

        # different detection parameters
        self.show_display = show_display
        self.focus = focus
        if self.focus:
            self.show_display = True

        self.resolution = resolution
        self.framerate = framerate
        self.exp_mode = exp_mode
        self.awb_mode = awb_mode
        self.sensor_mode = sensor_mode
        self.exp_compensation = exp_compensation

        # threshold parameters for different algorithms
        self.exgMin = exgMin
        self.exgMax = exgMax
        self.hueMin = hueMin
        self.hueMax = hueMax
        self.saturationMin = saturationMin
        self.saturationMax = saturationMax
        self.brightnessMin = brightnessMin
        self.brightnessMax = brightnessMax

        self.thresholdDict = {}
        self.image_loop_time = image_loop_time  # time spent on each image when looping over a directory

        if parameters_json:
            try:
                with open(parameters_json) as f:
                    self.thresholdDict = json.load(f)
                    self.exgMin = self.thresholdDict['exgMin']
                    self.exgMax = self.thresholdDict['exgMax']
                    self.hueMin = self.thresholdDict['hueMin']
                    self.hueMax = self.thresholdDict['hueMax']
                    self.saturationMin = self.thresholdDict['saturationMin']
                    self.saturationMax = self.thresholdDict['saturationMax']
                    self.brightnessMin = self.thresholdDict['brightnessMin']
                    self.brightnessMax = self.thresholdDict['brightnessMax']
                    print('[INFO] Parameters successfully loaded.')

            except FileExistsError:
                print('[ERROR] Parameters file not found. Continuing with default settings.')

            except KeyError:
                print('[ERROR] Parameter key not found. Continuing with default settings.')

        # setup the track bars if show_display is True
        if self.show_display:
            # create trackbars for the threshold calculation
            self.window_name = "Adjust Detection Thresholds"
            cv2.namedWindow("Adjust Detection Thresholds", cv2.WINDOW_AUTOSIZE)
            cv2.createTrackbar("ExG-Min", self.window_name, self.exgMin, 255, nothing)
            cv2.createTrackbar("ExG-Max", self.window_name, self.exgMax, 255, nothing)
            cv2.createTrackbar("Hue-Min", self.window_name, self.hueMin, 179, nothing)
            cv2.createTrackbar("Hue-Max", self.window_name, self.hueMax, 179, nothing)
            cv2.createTrackbar("Sat-Min", self.window_name, self.saturationMin, 255, nothing)
            cv2.createTrackbar("Sat-Max", self.window_name, self.saturationMax, 255, nothing)
            cv2.createTrackbar("Bright-Min", self.window_name, self.brightnessMin, 255, nothing)
            cv2.createTrackbar("Bright-Max", self.window_name, self.brightnessMax, 255, nothing)

        # check that the resolution is not so high it will entirely brick/destroy the OWL.
        total_pixels = resolution[0] * resolution[1]
        if total_pixels > (832 * 640):
            # change here if you want to test higher resolutions, but be warned, backup your current image!
            self.resolution = (416, 320)

        # instantiate the logger
        self.logger = None  # No spraying control, no need for a logger

        # check if test video or videostream from camera
        if input_file_or_directory:
            self.cam = FrameReader(path=input_file_or_directory,
                                   resolution=self.resolution,
                                   loop_time=self.image_loop_time)
            self.frame_width, self.frame_height = self.cam.resolution

        # if no video, start the camera with the provided parameters
        else:
            try:
                from picamera import PiCameraMMALError

            except ImportError:
                PiCameraMMALError = None

            try:
                self.cam = VideoStream(usePiCamera=True,
                                       resolution=self.resolution,
                                       framerate=self.framerate,
                                       exposure_mode=self.exp_mode,
                                       awb_mode=self.awb_mode,
                                       sensor_mode=self.sensor_mode,
                                       exposure_compensation=self.exp_compensation).start()

                self.frame_width = self.resolution[0]
                self.frame_height = self.resolution[1]

            except ModuleNotFoundError as e:
                self.cam = VideoStream(src=0).start()
                self.frame_width = self.cam.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
                self.frame_height = self.cam.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)

            except PiCameraMMALError as e:
                print(f"[CAMERA ERROR] Note, camera is in use by another OWL process.\n"
                      f"To resolve this error, follow these steps:\n1. Use <ps -C owl.py> to find the process\n"
                      f"2. Run <sudo kill PID_NUM>\n"
                      f"This will close the other process using its PID.\n"
                      f"Original error message: {
