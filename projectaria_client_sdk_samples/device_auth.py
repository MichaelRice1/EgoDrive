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

import aria.sdk as aria


def main():
    #  Optional: Set SDK's log level to Trace or Debug for more verbose logs. Defaults to Info
    aria.set_log_level(aria.Level.Info)

    # 1. Create DeviceClient instance
    device_client = aria.DeviceClient()

    # 2. Enable Tls v2 auth
    client_config = aria.DeviceClientConfig()
    device_client.set_client_config(client_config)

    # 3. Attempt to authenticate host client: this will connect to the device via adb and assumes
    # you have it installed on your path. Use set_client_config to specify it manually otherwise
    # or connect via IP address.
    try:
        device_client.authenticate()
    except Exception as ex:
        print(f"Could not authenticate host client: {ex}")
        exit(1)

    print(
        "Client authentication pending, please check the companion app and approve the request."
    )


if __name__ == "__main__":
    main()
