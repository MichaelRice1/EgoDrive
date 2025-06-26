# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import time

import aria.sdk as aria


def connect_and_stop_recording(device_client, device_client_config, serial):
    device_client_config.device_serial = serial
    device_client.set_client_config(device_client_config)
    device = device_client.connect()
    recording_manager = device.recording_manager
    # Stop recording on device if still recording
    if recording_manager.recording_state == aria.RecordingState.Recording:
        recording_manager.stop_recording()
    return device


def reconnect_devices(
    server_serial, client_serials, device_client, device_client_config
):
    server_device = None
    client_devices = {}
    reconnected_devices = set()
    while len(reconnected_devices) != len(client_serials) + 1:
        # Wait for client devices to be reconnected for cleanup
        for serial in client_serials:
            if serial in reconnected_devices:
                continue
            client_devices[serial] = connect_and_stop_recording(
                device_client, device_client_config, serial
            )
            reconnected_devices.add(serial)
        # Ensure all client devices were reconnected before reconnecting to the server device
        if len(reconnected_devices) != len(client_serials):
            continue

        # Wait for server device to be reconnected for cleanup
        server_device = connect_and_stop_recording(
            device_client, device_client_config, server_serial
        )
        reconnected_devices.add(server_serial)
    return [server_device, client_devices]


def detect_and_reconnect_devices(
    total_num_devices, device_client, device_client_config
):
    server_serial = None
    client_serials = []
    devices = device_client.usb_devices
    print(devices)
    assert (
        len(devices) == total_num_devices
    ), "Number of connected Aria devices is not equal to the total number of devices requested for TicSync cleanup. "
    "Please only plug in the devices that were used in TicSync"

    # Detect the server device and client devices
    retries = 0
    detected_devices = set()
    while retries < 10:
        if len(client_serials) + 1 == total_num_devices:
            break
        if retries > 0:
            time.sleep(1)
        for [serial, _] in devices:
            if serial in detected_devices:
                continue
            device_client_config.device_serial = serial
            device_client.set_client_config(device_client_config)
            device = device_client.connect()
            detected_devices.add(serial)
            dds_rpc_enabled_status = device.dds_rpc_enabled_status
            if dds_rpc_enabled_status.state == aria.DdsRpcState.On:
                server_serial = serial
            else:
                client_serials.append(serial)
        retries += 1
    if retries == 10:
        # Failed to detect the server device
        return [None, None]

    print("Detected server serial", server_serial)
    print("Detected client serials", client_serials)

    return reconnect_devices(
        server_serial, client_serials, device_client, device_client_config
    )


def cleanup_hotspot(device, server_hotspot_ssid):
    wifi_manager = device.wifi_manager
    # Connect client device to server device hotspot
    wifi_status = wifi_manager.wifi_status
    if wifi_status.network.ssid == server_hotspot_ssid:
        wifi_manager.keep_wifi_on(False)
        wifi_manager.forget_wifi(server_hotspot_ssid)


def client_devices_cleanup(client_devices, server_hotspot_ssid):
    for [_, device] in client_devices.items():
        cleanup_hotspot(device, server_hotspot_ssid)


def server_device_cleanup(server_device):
    if server_device.dds_rpc_enabled_status.state == aria.DdsRpcState.On:
        print("DDS RPC enabled, disabling it")
        server_device.set_dds_rpc_enabled(False, aria.StreamingInterface.WifiSoftAp)
    server_wifi_manager = server_device.wifi_manager
    server_wifi_manager.set_device_hotspot_status(False, True)


def generic_cleanup(device):
    recording_manager = device.recording_manager
    if recording_manager.recording_state == aria.RecordingState.Recording:
        recording_manager.stop_recording()
    wifi_manager = device.wifi_manager
    wifi_manager.keep_wifi_on(False)
    if device.dds_rpc_enabled_status.state == aria.DdsRpcState.On:
        print("DDS RPC enabled, disabling it")
        device.set_dds_rpc_enabled(False, aria.StreamingInterface.WifiSoftAp)
    wifi_manager.set_device_hotspot_status(False, True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server",
        dest="server_serial",
        type=str,
        required=False,
        help="Serial number of the server device and the profile it will record/stream with.",
    )
    parser.add_argument(
        "--clients",
        action="append",
        dest="client_serials",
        nargs="+",
        required=False,
        help="Serial number of the client device and the profile it will record/stream with.",
    )
    parser.add_argument(
        "--total_num_devices",
        dest="total_num_devices",
        type=int,
        action="store",
        required=False,
        help="Option to specify total number of devices to start time synchronized recordings."
        "Has no effect if server and client options were also specified.",
    )
    return parser.parse_args()


def main(args):
    #  Optional: Set SDK's log level to Trace or Debug for more verbose logs. Defaults to Info
    aria.set_log_level(aria.Level.Info)

    print(
        "-------- Plug in all devices to your computer again for TicSync cleanup --------"
    )
    input("-------- Then press Enter to start TicSync cleanup --------")

    device_client = aria.DeviceClient()
    device_client_config = aria.DeviceClientConfig()

    assert (args.server_serial is None and args.client_serials is None) or (
        args.server_serial is not None and args.client_serials is not None
    ), "Server and client devices options can only be specified together"

    assert (
        (args.total_num_devices is not None and args.server_serial is None)
        or (args.total_num_devices is None and args.server_serial is not None)
    ), "Please either specify total number of devices or server and client device serials"

    if args.total_num_devices:
        [server_device, client_devices] = detect_and_reconnect_devices(
            args.total_num_devices, device_client, device_client_config
        )
    else:
        [server_device, client_devices] = reconnect_devices(
            args.server_serial,
            args.client_serials[0],
            device_client,
            device_client_config,
        )

    print(
        "-------- All devices reconnected, please keep all devices plugged in. Performing cleanup --------"
    )
    if server_device is None:
        # Failed to detect server device, perform generic cleanup for all devices
        devices = device_client.usb_devices
        for [serial, _] in devices:
            device_client_config.device_serial = serial
            device_client.set_client_config(device_client_config)
            device = device_client.connect()
            generic_cleanup(device)
    else:
        # Perform regular cleanup
        server_wifi_manager = server_device.wifi_manager
        server_wifi_hotspot_status = server_wifi_manager.device_hotspot_status
        client_devices_cleanup(client_devices, server_wifi_hotspot_status.ssid)
        server_device_cleanup(server_device)
    print("-------- Successfully performed cleanup. Exiting --------")


if __name__ == "__main__":
    args = parse_args()
    main(args)
