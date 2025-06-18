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
import json
import logging
import subprocess

import pkg_resources

adb = pkg_resources.resource_filename("aria", "tools/adb")

from datetime import datetime
from typing import List

ticsync_server_files = {}
ticsync_client_files = {}
date_sorted_ticsync_server_recordings = {}


def run_command(cmd: List[str]) -> subprocess.CompletedProcess:
    logging.debug("Running: %s", " ".join(cmd))
    return subprocess.run(cmd, text=True, capture_output=True)


def date_from_timestamp(timestamp: str) -> datetime:
    return datetime.fromtimestamp(int(timestamp)).strftime("%Y-%m-%d %H:%M:%S")


def populate_files():
    global \
        date_sorted_ticsync_server_recordings, \
        ticsync_server_files, \
        ticsync_client_files
    devices_list = run_command([adb, "devices", "-l"])
    for device in devices_list.stdout.splitlines():
        if "model:Aria" not in device:
            continue
        device_serial = device.split(" ")[0]
        recording_files = run_command(
            [adb, "-s", device_serial, "shell", "ls", "/sdcard/recording"]
        )
        for vrs_json_file in recording_files.stdout.splitlines():
            if "vrs.json" in vrs_json_file:
                recording_uuid = vrs_json_file.split("/")[-1].replace(".vrs.json", "")
                out_json_string = run_command(
                    [
                        adb,
                        "-s",
                        device_serial,
                        "shell",
                        "cat",
                        "/sdcard/recording/" + vrs_json_file,
                    ]
                )
                file_json = json.loads(out_json_string.stdout)
                if "shared_session_id" in file_json:
                    if file_json["ticsync_mode"] == "server":
                        ticsync_server_files[file_json["shared_session_id"]] = [
                            date_from_timestamp(file_json["end_time"]),
                            device_serial,
                            recording_uuid,
                        ]
                    elif file_json["ticsync_mode"] == "client":
                        if file_json["shared_session_id"] not in ticsync_client_files:
                            ticsync_client_files[file_json["shared_session_id"]] = []
                        ticsync_client_files[file_json["shared_session_id"]].append(
                            [
                                device_serial,
                                recording_uuid,
                            ]
                        )


def list_files(verbose: bool) -> bool:
    global \
        date_sorted_ticsync_server_recordings, \
        ticsync_server_files, \
        ticsync_client_files
    populate_files()
    date_sorted_ticsync_server_recordings = dict(
        sorted(ticsync_server_files.items(), key=lambda item: item[1][0], reverse=True)
    )

    client_recordings_not_found_for_shared_session_id = []
    for shared_session_id in date_sorted_ticsync_server_recordings:
        if verbose:
            print(
                date_sorted_ticsync_server_recordings[shared_session_id][0],
                "Shared Session ID:",
                shared_session_id,
                "Server Serial:",
                date_sorted_ticsync_server_recordings[shared_session_id][1],
                "Server Recording UUID:",
                date_sorted_ticsync_server_recordings[shared_session_id][2],
            )
        if shared_session_id not in ticsync_client_files:
            client_recordings_not_found_for_shared_session_id.append(shared_session_id)
            continue
        for file in ticsync_client_files[shared_session_id]:
            if verbose:
                print("\tClient Serial:", file[0], "Client Recording UUID:", file[1])
        if verbose:
            print("\n")


def download_vrs_file(serial: str, uuid: str, output_dir: str):
    print(
        "Downloading the recording",
        "/sdcard/recording/" + uuid + ".vrs from the device",
        serial,
    )
    output_download = run_command(
        [
            adb,
            "-s",
            serial,
            "pull",
            "/sdcard/recording/" + uuid + ".vrs",
            output_dir,
        ]
    )
    print(output_download.stdout)


def download_files(shared_session_id: str, output_dir: str) -> bool:
    run_command(["mkdir", "-p", output_dir])
    if not date_sorted_ticsync_server_recordings:
        list_files(False)
    # Download the server recording
    if shared_session_id not in date_sorted_ticsync_server_recordings:
        print("No server recording found for shared session id:", shared_session_id)
        return False
    download_vrs_file(
        date_sorted_ticsync_server_recordings[shared_session_id][1],
        date_sorted_ticsync_server_recordings[shared_session_id][2],
        output_dir,
    )

    # Download the client recordings
    if shared_session_id not in ticsync_client_files:
        print("No client recordings found for shared session id:", shared_session_id)
        return False
    for file in ticsync_client_files[shared_session_id]:
        download_vrs_file(file[0], file[1], output_dir)
    return True


def main():
    parser = argparse.ArgumentParser(description="TicSync File Manager")
    parser.add_argument(
        "--list",
        dest="list_recordings",
        action="store_true",
        help="List TicSync recordings",
    )
    parser.add_argument(
        "-d",
        "--download",
        dest="shared_session_id",
        type=str,
        help="Download TicSync recordings given a shared session ID",
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        default=".",
        help="Optional. Specifies the directory to save the TicSync recordings in",
    )
    args = parser.parse_args()
    if args.list_recordings:
        list_files(True)
    if args.shared_session_id:
        download_files(args.shared_session_id, args.output_dir)


if __name__ == "__main__":
    main()
