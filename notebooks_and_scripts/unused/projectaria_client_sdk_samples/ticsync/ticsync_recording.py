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

from ticsync_cleanup import (
    client_devices_cleanup,
    reconnect_devices,
    server_device_cleanup,
)

# Use 5GHz hotspot for tic sync
WIFI_HOTSPOT_5GHZ = True


def get_device_serial_and_profile_names(args, device_client):
    assert (
        args.server_serial_and_profile_name is None
        and args.client_serial_and_profile_names is None
    ) or (
        args.server_serial_and_profile_name is not None
        and args.client_serial_and_profile_names is not None
    ), "Server and client devices options can only be specified together"

    assert (args.total_num_devices is None and args.profile is None) or (
        args.total_num_devices is not None and args.profile is not None
    ), "Total number of devices and recording profile can only be specified together"

    if (
        args.server_serial_and_profile_name is not None
        and args.client_serial_and_profile_names is not None
    ):
        assert (
            len(args.server_serial_and_profile_name) == 1
        ), "Only one device can be specified as the server device"
        return [
            args.server_serial_and_profile_name[0],
            args.client_serial_and_profile_names,
        ]
    elif args.total_num_devices is not None:
        # Try detecting specified number of devices
        ticsync_devices = device_client.usb_devices
        assert len(ticsync_devices) == args.total_num_devices, (
            "Number of connected Aria devices "
            + str(len(ticsync_devices))
            + " is not equal to the total number of devices "
            + str(args.total_num_devices)
            + " requested for TicSync cleanup. "
        )
        print(
            "Detected",
            args.total_num_devices,
            "devices. Using the following devices for TicSync:",
            ticsync_devices,
        )
        server_serial_and_profile_name = [
            ticsync_devices[0][0],
            args.profile,
        ]
        client_serial_and_profile_names = []
        for [serial, _] in ticsync_devices[1:]:
            client_serial_and_profile_names.append([serial, args.profile])
        return [
            server_serial_and_profile_name,
            client_serial_and_profile_names,
        ]

    return [None, None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server",
        action="append",
        dest="server_serial_and_profile_name",
        nargs="+",
        required=False,
        help="Serial number of the server device and the profile it will record/stream with.",
    )
    parser.add_argument(
        "--client",
        action="append",
        dest="client_serial_and_profile_names",
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
    parser.add_argument(
        "--profile",
        dest="profile",
        type=str,
        required=False,
        help="Option to specify the common recording profile to start time synchronized recordings."
        "Only has effect if total_num_devices was specified.",
    )
    return parser.parse_args()


def main(args):
    #  Optional: Set SDK's log level to Trace or Debug for more verbose logs. Defaults to Info
    aria.set_log_level(aria.Level.Info)

    assert (args.total_num_devices is None) or (
        args.total_num_devices > 1
    ), "Total number of devices specified cannot be less than 2"

    device_client = aria.DeviceClient()
    device_client_config = aria.DeviceClientConfig()

    [server_serial_and_profile_name, client_serial_and_profile_names] = (
        get_device_serial_and_profile_names(args, device_client)
    )

    server_serial = server_serial_and_profile_name[0]
    server_profile_name = server_serial_and_profile_name[1]
    # Set up the server device
    device_client_config.device_serial = server_serial
    device_client.set_client_config(device_client_config)
    # Connect to server device
    server_device = device_client.connect()
    # Retrieve wifi_manager of the server device
    server_wifi_manager = server_device.wifi_manager
    # Switch the server device to hotspot mode with a random password
    server_wifi_manager.set_device_hotspot_status(True, WIFI_HOTSPOT_5GHZ)
    if server_device.dds_rpc_enabled_status.state == aria.DdsRpcState.Off:
        print("DDS RPC is not enabled, enabling it")
        server_device.set_dds_rpc_enabled(True, aria.StreamingInterface.WifiSoftAp)
    else:
        # Retrieve a new DDS RPC session ID
        session_id = server_device.dds_rpc_new_session_id()
        print("Retrieved a new DDS RPC session ID", session_id)

    # Retrieve the server device hotspot status. Will be used to connect the client devices to
    # the server device hotspot
    server_wifi_hotspot_status = server_wifi_manager.device_hotspot_status
    # Retrieve recording_manager of the server device
    server_recording_manager = server_device.recording_manager
    # Set time sync mode to TicSyncServer using custom recording config
    recording_config = aria.RecordingConfig()
    recording_config.profile_name = server_profile_name
    recording_config.time_sync_mode = aria.TimeSyncMode.TicSyncServer
    server_recording_manager.recording_config = recording_config

    client_devices = {}
    client_recording_managers = {}
    # Set up the client devices
    for [serial, profile_name] in client_serial_and_profile_names:
        # Reuse the existing DeviceClient instance by setting a new client config
        device_client_config.device_serial = serial
        device_client.set_client_config(device_client_config)
        # Connect to client device
        device = device_client.connect()
        wifi_manager = device.wifi_manager
        # Connect client device to server device hotspot
        device_status = device.status
        # Check if the client device is already connected to the server device hotspot
        if (
            device_status.wifi_connected is False
            or device_status.wifi_ssid != server_wifi_hotspot_status.ssid
        ):
            # If not, connect client device to server device hotspot
            wifi_manager.connect_wifi(
                server_wifi_hotspot_status.ssid,
                server_wifi_hotspot_status.passphrase,
                aria.WifiAuthentication.Wpa,
                False,  # hidden
                "",  # username
                True,  # disable_other_network
                True,  # skip_internet_check
            )
        # Set keep Wi-Fi on as true for the client devices
        # This keeps the client devices connected to the Wi-Fi even when they are disconnected from USB
        wifi_manager.keep_wifi_on(True)
        # Retrieve recording_manager of the client device
        recording_manager = device.recording_manager
        # Set time sync mode to TicSyncClient using custom recording config
        recording_config = aria.RecordingConfig()
        recording_config.profile_name = profile_name
        recording_config.time_sync_mode = aria.TimeSyncMode.TicSyncClient
        recording_manager.recording_config = recording_config
        client_devices[serial] = device
        client_recording_managers[serial] = recording_manager

    # Start recording on the server device
    print(f"Starting to record the server device {server_serial} using {profile_name}")
    server_recording_manager.start_recording()
    # Get recording state of the server device
    server_recording_state = server_recording_manager.recording_state
    print(f"Recording state of server device {server_serial}: {server_recording_state}")

    # Start recording on the client devices
    for manager in client_recording_managers.values():
        manager.start_recording()

    # Wait for for ticsync convergence
    def _is_stable(recording_manager):
        status = recording_manager.tic_sync_status
        return status.synchronization_stability == aria.SynchronizationStability.Stable

    print(
        "-------- Waiting for devices to be ready for time synchronized data collection, this will take around 45 seconds. --------\n"
        "-------- Please keep all devices plugged in. ---------"
    )
    while not all(
        _is_stable(manager) for manager in client_recording_managers.values()
    ):
        time.sleep(5)

    print(
        "-------- All devices are ready for data collection. You can safely unplug all your glasses from USB ---------"
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
