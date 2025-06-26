
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0

import argparse
import sys
import time
import csv

import aria.sdk as aria
import cv2
import numpy as np
from common import quit_keypress, update_iptables
from projectaria_tools.core.sensor_data import ImageDataRecord, MotionData

import matplotlib.pyplot as plt
from collections import deque
import threading


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update_iptables",
        default=False,
        action="store_true",
        help="Update iptables to enable receiving the data stream, only for Linux.",
    )
    return parser.parse_args()


# Real-time gyro plotter
class IMUPlotter:
    def __init__(self, max_len=300):
        self.max_len = max_len
        self.timestamps = deque(maxlen=max_len)
        self.gyro_x = deque(maxlen=max_len)
        self.gyro_y = deque(maxlen=max_len)
        self.gyro_z = deque(maxlen=max_len)

        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line_x, = self.ax.plot([], [], label='Gyro X')
        self.line_y, = self.ax.plot([], [], label='Gyro Y')
        self.line_z, = self.ax.plot([], [], label='Gyro Z')
        self.ax.set_ylim(-10, 10)
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Angular Velocity (rad/s)')
        self.ax.legend()
        self.lock = threading.Lock()

    def update(self, timestamp, gyro):
        with self.lock:
            self.timestamps.append(timestamp)
            self.gyro_x.append(gyro[0])
            self.gyro_y.append(gyro[1])
            self.gyro_z.append(gyro[2])

    def redraw(self):
        with self.lock:
            self.line_x.set_data(self.timestamps, self.gyro_x)
            self.line_y.set_data(self.timestamps, self.gyro_y)
            self.line_z.set_data(self.timestamps, self.gyro_z)
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()


def main():
    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    aria.set_log_level(aria.Level.Info)
    streaming_client = aria.StreamingClient()

    config = streaming_client.subscription_config
    config.subscriber_data_type = (
        aria.StreamingDataType.Imu
    )
    config.message_queue_size[aria.StreamingDataType.Imu] = 1000

    options = aria.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    config.security_options = options
    streaming_client.subscription_config = config

    plotter = IMUPlotter()

    # Shared state
    angular_position = [0.0]
    last_imu_time = [None]

    # gyro bias 
    gyro_bias_x = 0.004756912266351248 

    class StreamingClientObserver:
        def __init__(self):
            self.images = {}
            self.latest_Imu = None

        def on_image_received(self, image: np.array, record: ImageDataRecord):
            self.images[record.camera_id] = image

        def on_imu_received(self, data_list: list[MotionData], imu_idx: int):
            if imu_idx == 0:
                for data in data_list:
                    gyro = data.gyro_radsec
                    timestamp = data.capture_timestamp_ns / 1e9  # seconds

                    # Integrate gyro[0]
                    if last_imu_time[0] is not None:
                        dt = timestamp - last_imu_time[0]
                        corrected_gyro_x = gyro[0] - gyro_bias_x
                        delta_angle = corrected_gyro_x * dt
                        angular_position[0] += delta_angle
                    last_imu_time[0] = timestamp

                    plotter.update(timestamp, gyro)
                    self.latest_Imu = gyro

    observer = StreamingClientObserver()
    streaming_client.set_streaming_client_observer(observer)

    print("Start listening to image and Imu data")
    streaming_client.subscribe()

    while not quit_keypress():
        plotter.redraw()

    print("Stop listening to data")
    streaming_client.unsubscribe()

if __name__ == "__main__":
    main()
