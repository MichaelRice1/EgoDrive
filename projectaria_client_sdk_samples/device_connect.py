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

import aria.sdk as aria


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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

    # 3. Fetch device status
    status = device.status
    battery_level = status.battery_level
    wifi_ssid = status.wifi_ssid
    wifi_ip = status.wifi_ip_address
    device_mode = status.device_mode

    print(
        "Aria Device Status: battery level {0}, wifi ssid {1}, wifi ip {2}, mode {3}".format(
            battery_level, wifi_ssid, wifi_ip, device_mode
        )
    )

    # 4. Fetch device info
    info = device.info
    model = info.model
    serial = info.serial

    print("Aria Device Info: model {}, serial {}".format(model, serial))

    print("Disconnecting from Aria")

    # 5. Disconnect current device
    device_client.disconnect(device)


if __name__ == "__main__":
    main()
