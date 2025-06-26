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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        dest="profile_name",
        type=str,
        default="profile8",
        required=False,
        help="Profile to be used for streaming.",
    )
    parser.add_argument(
        "--duration",
        dest="recording_duration",
        type=int,
        default=10,
        required=False,
        help="Duration of the recording.",
    )
    parser.add_argument(
        "--device-ip", help="IP address to connect to the device over wifi"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    #  Optional: Set SDK's log level to Trace or Debug for more verbose logs. Defaults to Info
    aria.set_log_level(aria.Level.Info)

    # 1. Create DeviceClient instance, setting the IP address if specified
    device_client = aria.DeviceClient()

    client_config = aria.DeviceClientConfig()
    if args.device_ip:
        client_config.ip_v4_address = args.device_ip
    device_client.set_client_config(client_config)

    # 2. Connect to Aria
    device = device_client.connect()

    # 3. Retrieve recording_manager
    recording_manager = device.recording_manager

    # 4. Use custom config for recording
    recording_config = aria.RecordingConfig()
    recording_config.profile_name = args.profile_name
    recording_manager.recording_config = recording_config

    # 3. Start recording
    print(
        f"Starting to record using {args.profile_name} for {args.recording_duration} seconds"
    )
    recording_manager.start_recording()

    # 4. Get recording state
    recording_state = recording_manager.recording_state
    print(f"Recording state: {recording_state}")

    # 5. Record for the duration specified in the arguments
    time.sleep(args.recording_duration)

    # 6. Stop recording
    recording_manager.stop_recording()
    print(f"Took a {args.recording_duration}-second long recording.")

    # 7. Disconnect current device
    device_client.disconnect(device)


if __name__ == "__main__":
    main()
