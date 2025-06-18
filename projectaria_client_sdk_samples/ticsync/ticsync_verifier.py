# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import argparse

import csv
import signal
import sys

import numpy as np

import rerun as rr

from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import MotionData, TimeDomain, TimeQueryOptions


NS_IN_MS = 1000000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server",
        action="append",
        dest="server_serial_and_vrs_filename",
        nargs="+",
        required=True,
        help="Serial number of the server device and its VRS file to be verified.",
    )
    parser.add_argument(
        "--client",
        action="append",
        dest="client_serial_and_vrs_filenames",
        nargs="+",
        required=True,
        help="Serial number of the client device and its VRS file to be verified.",
    )
    parser.add_argument(
        "--export_to_csv",
        dest="export_to_csv",
        action="store_true",
        help="Optional. Export IMU data to CSV. Default is false.",
    )
    return parser.parse_args()


def visualize_imu(image: np.array, window_name: str) -> None:
    if image is not None:
        rr.log(window_name, image)


def log_imu_data(sensor_name: str, imu: MotionData, timestamp: int):
    rr.set_time_nanos("device_time", timestamp)
    log_accelerometer(sensor_name, imu.accel_msec2)
    log_gyroscope(sensor_name, imu.gyro_radsec)
    log_magnetometer(sensor_name, imu.mag_tesla)


def log_accelerometer(sensor_name: str, accel_msec2: np.array):
    rr.log(sensor_name + "/accel/x", rr.Scalar(accel_msec2[0]))
    rr.log(sensor_name + "/accel/y", rr.Scalar(accel_msec2[1]))
    rr.log(sensor_name + "/accel/z", rr.Scalar(accel_msec2[2]))


def log_gyroscope(sensor_name: str, gyro_radsec: np.array):
    rr.log(sensor_name + "/gyro/x", rr.Scalar(gyro_radsec[0]))
    rr.log(sensor_name + "/gyro/y", rr.Scalar(gyro_radsec[1]))
    rr.log(sensor_name + "/gyro/z", rr.Scalar(gyro_radsec[2]))


def log_magnetometer(sensor_name: str, mag_tesla: np.array):
    rr.log(sensor_name + "/mag/x", rr.Scalar(mag_tesla[0]))
    rr.log(sensor_name + "/mag/y", rr.Scalar(mag_tesla[1]))
    rr.log(sensor_name + "/mag/z", rr.Scalar(mag_tesla[2]))


def main():
    args = parse_args()
    assert (
        len(args.server_serial_and_vrs_filename) == 1
    ), "Only one device can be specified as the server device"

    server_serial = args.server_serial_and_vrs_filename[0][0]
    server_vrs_filename = args.server_serial_and_vrs_filename[0][1]
    server_vrs_data_provider = data_provider.create_vrs_data_provider(
        server_vrs_filename
    )
    if not server_vrs_data_provider:
        print("Couldn't create data provider from vrs file")
        exit(1)

    client_vrs_data_providers = {}
    # Set up the client devices
    for [serial, vrs_filename] in args.client_serial_and_vrs_filenames:
        vrs_data_provider = data_provider.create_vrs_data_provider(vrs_filename)
        if not vrs_data_provider:
            print("Couldn't create data provider from vrs file")
            exit(1)
        client_vrs_data_providers[serial] = vrs_data_provider

    # Stop streaming on ctrl-c
    def signal_handler(sig, frame):
        rr.disconnect()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    rr.init("Aria TicSync Verifier")
    rr.connect()

    imu_stream_id = server_vrs_data_provider.get_stream_id_from_label("imu-right")
    imu_stream_id_str = str(imu_stream_id)
    server_device_timestamps_vec = server_vrs_data_provider.get_timestamps_ns(
        imu_stream_id, TimeDomain.DEVICE_TIME
    )

    csv_fields = [
        "serial",
        "timestamp_ns",
        "accel_x",
        "accel_y",
        "accel_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "mag_x",
        "mag_y",
        "mag_z",
    ]
    csv_server_imu_data = []
    csv_client_imu_data = []

    for timestamp in server_device_timestamps_vec:
        server_imu_data = server_vrs_data_provider.get_imu_data_by_time_ns(
            imu_stream_id, timestamp, TimeDomain.DEVICE_TIME, TimeQueryOptions.BEFORE
        )
        server_sensor_data = server_vrs_data_provider.get_sensor_data_by_time_ns(
            imu_stream_id,
            timestamp,
            TimeDomain.DEVICE_TIME,
            TimeQueryOptions.BEFORE,
        )
        server_imu_timestamp = server_sensor_data.get_time_ns(TimeDomain.DEVICE_TIME)
        log_imu_data(
            imu_stream_id_str + "/server/" + server_serial,
            server_imu_data,
            server_imu_timestamp,
        )
        if args.export_to_csv is True:
            csv_server_imu_data.append(
                [
                    server_serial,
                    server_imu_timestamp,
                    server_imu_data.accel_msec2[0],
                    server_imu_data.accel_msec2[1],
                    server_imu_data.accel_msec2[2],
                    server_imu_data.gyro_radsec[0],
                    server_imu_data.gyro_radsec[1],
                    server_imu_data.gyro_radsec[2],
                    server_imu_data.mag_tesla[0],
                    server_imu_data.mag_tesla[1],
                    server_imu_data.mag_tesla[2],
                ]
            )
        for [serial, client_vrs_data_provider] in client_vrs_data_providers.items():
            client_imu_data = client_vrs_data_provider.get_imu_data_by_time_ns(
                imu_stream_id,
                timestamp,
                TimeDomain.TIC_SYNC,
                TimeQueryOptions.CLOSEST,
            )
            client_sensor_data = client_vrs_data_provider.get_sensor_data_by_time_ns(
                imu_stream_id,
                timestamp,
                TimeDomain.TIC_SYNC,
                TimeQueryOptions.CLOSEST,
            )
            client_imu_timestamp = client_sensor_data.get_time_ns(TimeDomain.TIC_SYNC)

            time_delta = np.abs(server_imu_timestamp - client_imu_timestamp)
            if time_delta < NS_IN_MS:
                log_imu_data(
                    imu_stream_id_str + "/client/" + serial,
                    client_imu_data,
                    client_imu_timestamp,
                )
            if args.export_to_csv is True:
                csv_client_imu_data.append(
                    [
                        serial,
                        client_imu_timestamp,
                        client_imu_data.accel_msec2[0],
                        client_imu_data.accel_msec2[1],
                        client_imu_data.accel_msec2[2],
                        client_imu_data.gyro_radsec[0],
                        client_imu_data.gyro_radsec[1],
                        client_imu_data.gyro_radsec[2],
                        client_imu_data.mag_tesla[0],
                        client_imu_data.mag_tesla[1],
                        client_imu_data.mag_tesla[2],
                    ]
                )

    if args.export_to_csv is True:
        with open("imu_data_server.csv", "w") as csv_server_file:
            csv_writer = csv.writer(csv_server_file)
            csv_writer.writerow(csv_fields)
            csv_writer.writerows(csv_server_imu_data)

        with open("imu_data_client.csv", "w") as csv_client_file:
            csv_writer = csv.writer(csv_client_file)
            csv_writer.writerow(csv_fields)
            csv_writer.writerows(csv_client_imu_data)


if __name__ == "__main__":
    main()
