import math
import os
import numpy as np
from PIL import Image
import torch
import random
import cv2
import csv
from time import time
import pandas as pd
import tqdm
from ultralytics import YOLO
from typing import Dict, List, Optional
from filterpy.kalman import KalmanFilter
from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId
import projectaria_tools.core.mps as mps
from projectaria_tools.core.mps.utils import (
    get_nearest_wrist_and_palm_pose,
    get_gaze_vector_reprojection,
    get_nearest_eye_gaze,
    get_nearest_pose,
    get_nearest_hand_tracking_result
)

os.environ['YOLO_VERBOSE'] = 'False'

import sys
sys.path.append('/Users/michaelrice/Documents/GitHub/Thesis/MSc_AI_Thesis/notebooks_and_scripts/unused/')


class VRSDataExtractor():


    def __init__(self, vrs_path: str):
        self.path = vrs_path
        self.provider = data_provider.create_vrs_data_provider(self.path)
        
        self.time_domain = TimeDomain.DEVICE_TIME  # query data based on host time
        self.option = TimeQueryOptions.CLOSEST # get data whose time [in TimeDomain] is CLOSEST to query time
        
        self.provider.set_devignetting(True)
        self.provider.set_devignetting_mask_folder_path('/Users/michaelrice/devignetting_masks')
        
        self.device_calibration = self.provider.get_device_calibration()
        self.rgb_camera_calibration = self.device_calibration.get_camera_calib('camera-rgb')

        
        # self.provider.set_devignetting(True)
        # self.provider.set_devignetting_mask_folder_path('/Users/michaelrice/devignetting_masks')

        self.stream_mappings = {
            "camera-slam-left": StreamId("1201-1"),
            "camera-slam-right":StreamId("1201-2"),
            "camera-rgb":StreamId("214-1"),
            "camera-eyetracking":StreamId("211-1"),
            "mic":StreamId("231-1"),
            "gps":StreamId("281-1"),
            "gps-app":StreamId("281-2"),
            "imu-right":StreamId("1202-1"),
            "imu-left":StreamId("1202-2")
        }

        self.stream_ids: Dict[str, StreamId] = {
                "rgb": StreamId("214-1"),
                "slam-left": StreamId("1201-1"),
                "slam-right": StreamId("1201-2")
                }
        self.stream_labels: Dict[str, str] = {
            key: self.provider.get_label_from_stream_id(stream_id)
            for key, stream_id in self.stream_ids.items()}
        
        self.stream_timestamps_ns: Dict[str, List[int]] = {
            key: self.provider.get_timestamps_ns(stream_id, self.time_domain)
            for key, stream_id in self.stream_ids.items()}

        self.camera_calibrations = {key: self.device_calibration.get_camera_calib(stream_label) for key, stream_label in self.stream_labels.items()}
        

        try:
            self.rgb_start_time = self.provider.get_first_time_ns(self.stream_mappings['camera-rgb'], self.time_domain)
        except:
            self.rgb_start_time = 0
        self.result = {}
        self.face_ego_blur = "models/ego_blur_face.jit"
        self.lp_ego_blur = "models/ego_blur_lp.jit"
        rgblabel = 'camera-rgb'
        self.num_frames_rgb = self.provider.get_num_data(self.provider.get_stream_id_from_label(rgblabel))

    def get_device(self) -> str:
        """
        Return the device type
        """
        return (
            "cpu"
            if not torch.cuda.is_available()
            else f"cuda:{torch.cuda.current_device()}"
        )
    
    def crop_largest_square_inside_circle(self, image):

        """
        Crops the largest square from a circular image centered in a square frame.
        """

        assert image.shape[0] == image.shape[1], "Image must be square"
        size = image.shape[0]
        
        # Compute side of largest inscribed square
        side = int(size / np.sqrt(2))
        
        # Compute starting point to crop centered
        offset = (size - side) // 2
        return image[offset:offset+side, offset:offset+side]
    
    def get_image_data(self, start_index=0, end_index=None, rgb_flag = False, progress_callback=None):

        '''
        Extracts frames from the VRS file based on the index/time domain. The frames are extracted from the following streams:
        - camera-rgb
        - camera-eyetracking
        - camera-slam-left
        - camera-slam-right
        '''
        
        rgb_images, et_images, slam_right_images, slam_left_images = {},{},{},{}


        rgblabel = 'camera-rgb'
        etlabel = 'camera-et'
        left_slam_label  = 'camera-slam-left'
        right_slam_label = 'camera-slam-right'

        num_frames_rgb = self.provider.get_num_data(self.provider.get_stream_id_from_label(rgblabel))
        num_frames_eth = self.provider.get_num_data(self.provider.get_stream_id_from_label(etlabel))
        num_frames_slam_left = self.provider.get_num_data(self.provider.get_stream_id_from_label(left_slam_label))
        num_frames_slam_right = self.provider.get_num_data(self.provider.get_stream_id_from_label(right_slam_label))

        rgb_ts = self.provider.get_timestamps_ns(self.stream_mappings['camera-rgb'], self.time_domain)
        et_ts = self.provider.get_timestamps_ns(self.stream_mappings['camera-eyetracking'], self.time_domain)
        slam_left_ts = self.provider.get_timestamps_ns(self.stream_mappings['camera-slam-left'], self.time_domain)
        slam_right_ts = self.provider.get_timestamps_ns(self.stream_mappings['camera-slam-right'], self.time_domain)



        if end_index is None:
            end_index = num_frames_rgb
            end_index_et = num_frames_eth
            end_index_slam = num_frames_slam_left
        else:
            end_index_et = end_index * 2


        if start_index != 0:
            start_index_et = start_index * 2
        else:
            start_index_et = 0




        for i,index in enumerate(tqdm.tqdm(range(start_index, end_index), desc= "Extracting Images")):
            # buffer = io.BytesIO()

            image_data = self.provider.get_image_data_by_index(self.stream_mappings['camera-rgb'], index)[0].to_numpy_array()
            image_pil = Image.fromarray(image_data)  
            image_pil = image_pil.rotate(-90) 
            image_pil = image_pil.resize((512, 512), Image.LANCZOS)  # Resize to 512x512

            # Save to buffer (optional, only if needed)
            # image_pil.save(buffer, format="PNG")

            # Convert back to NumPy array
            img = np.array(image_pil)  # Convert back to NumPy array


            ## removing circular mask
            # square_crop = self.crop_largest_square_inside_circle(img)


            ## undistorting the images
            # calib = self.provider.get_device_calibration().get_camera_calib('camera-rgb')
            # pinhole = calibration.get_linear_camera_calibration(512, 512, 170)
            # undistorted_image = calibration.distort_by_calibration(img, pinhole, calib)
            # Call progress update

            if progress_callback:
                progress_callback(i + 1, end_index - start_index)


            rgb_images[rgb_ts[index]] = img


        self.result['rgb'] = rgb_images
        print(f"Extracted {len(self.result['rgb'])} images from {rgblabel} stream")

        if not rgb_flag:
            for index in range(start_index_et, end_index_et):
                image_data = self.provider.get_image_data_by_index(self.stream_mappings['camera-eyetracking'], index)[0].to_numpy_array()
                et_images[et_ts[index]] = image_data
                
            self.result['et'] = et_images
            print(f"Extracted {len(self.result['et'])} images from {etlabel} stream")

            try:
                for index in range(start_index, end_index_slam):
                    left_image_data = self.provider.get_image_data_by_index(self.stream_mappings['camera-slam-left'], index)
                    right_image_data = self.provider.get_image_data_by_index(self.stream_mappings['camera-slam-right'], index)
                    left_img = np.array(Image.fromarray(left_image_data[0].to_numpy_array()))
                    right_img = np.array(Image.fromarray(right_image_data[0].to_numpy_array()))
                    slam_left_images[slam_left_ts[index]] = left_img
                    slam_right_images[slam_right_ts[index]] = right_img
                    # break

                self.result['slam_left'] = slam_left_images
                self.result['slam_right'] = slam_right_images

                print(f"Extracted {len(self.result['slam_left'])} images from {left_slam_label} stream")
                print(f"Extracted {len(self.result['slam_right'])} images from {right_slam_label} stream")
            except:
                print("Error extracting slam images, likely not present in VRS file ")

    def create_kalman_filter(self):
        kf = KalmanFilter(dim_x=4, dim_z=2)

        # State vector: [x, y, vx, vy]
        # Measurement: [x, y]
        dt = 1.0  # time step

        # Transition matrix (motion model)
        kf.F = np.array([[1, 0, dt, 0],
                        [0, 1, 0, dt],
                        [0, 0, 1, 0 ],
                        [0, 0, 0, 1 ]])

        # Measurement function
        kf.H = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0]])

        
        kf.P *= 1000. 
        kf.Q *= 0.01
        kf.R *= 5

        return kf
    
    def get_gaze_data(self, gaze_path:str,personalized_gaze_path:str, start_index=0, end_index=None):

        '''
        Extracts eyegaze points from the VRS file based on the index/time domain. The eyegaze points are extracted from the following streams:
        - camera-eyetracking
        '''

        gaze_points = {}
        hw_points = {}
        

        gaze_cpf = mps.read_eyegaze(gaze_path)
        if personalized_gaze_path:
            personalized_gaze_cpf = mps.read_eyegaze(personalized_gaze_path)
        # handwrist_points  = mps.hand_tracking.read_wrist_and_palm_poses(hand_path)
        

        if start_index == 0 and end_index == None:
            num_et = len(gaze_cpf)
            et_ts = [gaze_cpf[i].tracking_timestamp.total_seconds() * 1e9 for i in range(num_et)]
            if personalized_gaze_path:
                p_et_ts = [personalized_gaze_cpf[i].tracking_timestamp.total_seconds() * 1e9 for i in range(num_et)]
        else:
            et_ts = [gaze_cpf[i].tracking_timestamp.total_seconds() * 1e9 for i in range(start_index,end_index)]
            if personalized_gaze_path:
                p_et_ts = [personalized_gaze_cpf[i].tracking_timestamp.total_seconds() * 1e9 for i in range(start_index,end_index)]
            #hw_ts = [handwrist_points[i].tracking_timestamp.total_seconds() * 1e9 for i in range(start_index,end_index)]

            # self.result['gaze_projected'] = project_gaze(gaze_path, self.path)
        for ts in et_ts:
            gaze_point = get_nearest_eye_gaze(gaze_cpf, ts)

            if gaze_point is not None:

                rgb_stream_label = self.provider.get_label_from_stream_id(self.stream_mappings['camera-rgb'])
                device_calibration = self.provider.get_device_calibration()
                rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)

                gaze_projection = get_gaze_vector_reprojection(
                        gaze_point,
                        rgb_stream_label,
                        device_calibration,
                        rgb_camera_calibration,
                        depth_m=gaze_point.depth
                    )
                if gaze_projection is not None:
                    gaze_projection = gaze_projection * (512/1408)
                    gaze_projection = [512 - gaze_projection[1], gaze_projection[0]] 
                
                gaze_points[ts] = {
                    "projection": gaze_projection,
                    'depth': gaze_point.depth,
                }
        self.result['gaze'] = gaze_points
        print(f"Extracted {len(gaze_points)} gaze points")



        p_gaze_points = {}
        # Process personalized gaze data
        if personalized_gaze_path:

            # self.result['pgaze_projected'] = project_gaze(personalized_gaze_path, self.path)


            for ts in p_et_ts:
                personalized_gaze_point = get_nearest_eye_gaze(personalized_gaze_cpf, ts)

                if personalized_gaze_point is not None:

                    rgb_stream_label = self.provider.get_label_from_stream_id(self.stream_mappings['camera-rgb'])
                    device_calibration = self.provider.get_device_calibration()
                    rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)

                    gaze_projection = get_gaze_vector_reprojection(
                            personalized_gaze_point,
                            rgb_stream_label,
                            device_calibration,
                            rgb_camera_calibration,
                            depth_m=personalized_gaze_point.depth
                        )

                    if gaze_projection is not None:
                        gaze_projection = gaze_projection * (512/1408)
                        gaze_projection = [512 - gaze_projection[1], gaze_projection[0]] 
                    
                    p_gaze_points[ts] = {
                        "projection": gaze_projection,
                        'depth': personalized_gaze_point.depth,
                    }

        self.result['personalized_gaze'] = p_gaze_points

        gaze_points = [g['projection'] for g in gaze_points.values()]
        p_gaze_points = [g['projection'] for g in p_gaze_points.values()]
        # p_gaze_points = [point for point in p_gaze_points if point is not None]
        
        last_valid = None
        for i, z in enumerate(p_gaze_points):
            
            if z is None:
                if last_valid is not None:
                    z = last_valid
                    p_gaze_points[i] = z  # Optional: update the source list too
                else:
                    continue  # Skip if no valid previous point exists
            else:
                last_valid = z
        print(f"Extracted {len(p_gaze_points)} personalized gaze points")
        smoothed_gaze = []

        # gaze = [g[0]['projection']*(512/1408) for g in gaze]

        kf = self.create_kalman_filter()

        if len(p_gaze_points) == 0:
            kf.x = np.array([[gaze_points[0][0]], [gaze_points[0][1]], [0], [0]])  # Initialize with first point
        else:
            kf.x = np.array([[p_gaze_points[0][0]], [p_gaze_points[0][1]], [0], [0]])

        for i, z in enumerate(p_gaze_points):
            if z is None:
                z = p_gaze_points[i-1]  # Use the last valid point if current is None
            kf.predict()
            kf.update(np.array(z))
            smoothed_gaze.append((kf.x[0, 0], kf.x[1, 0]))
        print(f"Extracted {len(smoothed_gaze)} smoothed gaze points")   
        self.result['smoothed_gaze'] = smoothed_gaze

    def get_hand_point_reprojection(self, point_position_device: np.array, key: str) -> Optional[np.array]:
        point_position_camera = self.get_T_device_sensor(key).inverse() @ point_position_device
        point_position_pixel = self.camera_calibrations[key].project(point_position_camera)
        return point_position_pixel
    
    def get_T_device_sensor(self, key: str):
        return self.device_calibration.get_transform_device_sensor(self.stream_labels[key])

    def get_landmark_pixels(self, key: str, hand_tracking_result: mps.hand_tracking.HandTrackingResult) -> np.array:
    

        left_wrist = None
        left_palm = None
        left_landmarks = None
        right_wrist = None
        right_palm = None
        right_landmarks = None
        left_wrist_normal_tip = None
        left_palm_normal_tip = None
        right_wrist_normal_tip = None
        right_palm_normal_tip = None


        NORMAL_VIS_LEN = 0.5  # meters
        scale = 512/1408

        
        if hand_tracking_result.left_hand:
            left_landmarks = [
                self.get_hand_point_reprojection(landmark, key)
                for landmark in hand_tracking_result.left_hand.landmark_positions_device
            ]
            left_wrist = self.get_hand_point_reprojection(
                hand_tracking_result.left_hand.landmark_positions_device[
                    int(mps.hand_tracking.HandLandmark.WRIST)
                ],
                key,
            )
            left_palm = self.get_hand_point_reprojection(
                hand_tracking_result.left_hand.landmark_positions_device[
                    int(mps.hand_tracking.HandLandmark.PALM_CENTER)
                ],
                key,
            )
            if hand_tracking_result.left_hand.wrist_and_palm_normal_device is not None:
                left_wrist_normal_tip = self.get_hand_point_reprojection(
                    hand_tracking_result.left_hand.landmark_positions_device[
                        int(mps.hand_tracking.HandLandmark.WRIST)
                    ]
                    + hand_tracking_result.left_hand.wrist_and_palm_normal_device.wrist_normal_device
                    * NORMAL_VIS_LEN,
                    key,
                )
                left_palm_normal_tip = self.get_hand_point_reprojection(
                    hand_tracking_result.left_hand.landmark_positions_device[
                        int(mps.hand_tracking.HandLandmark.PALM_CENTER)
                    ]
                    + hand_tracking_result.left_hand.wrist_and_palm_normal_device.palm_normal_device
                    * NORMAL_VIS_LEN,
                    key,
                )
        if hand_tracking_result.right_hand:
            right_landmarks = [
                self.get_hand_point_reprojection(landmark, key)
                for landmark in hand_tracking_result.right_hand.landmark_positions_device
            ]
            right_wrist = self.get_hand_point_reprojection(
                hand_tracking_result.right_hand.landmark_positions_device[
                    int(mps.hand_tracking.HandLandmark.WRIST)
                ],
                key,
            )
            right_palm = self.get_hand_point_reprojection(
                hand_tracking_result.right_hand.landmark_positions_device[
                    int(mps.hand_tracking.HandLandmark.PALM_CENTER)
                ],
                key,
            )
            if hand_tracking_result.right_hand.wrist_and_palm_normal_device is not None:
                right_wrist_normal_tip = self.get_hand_point_reprojection(
                    hand_tracking_result.right_hand.landmark_positions_device[
                        int(mps.hand_tracking.HandLandmark.WRIST)
                    ]
                    + hand_tracking_result.right_hand.wrist_and_palm_normal_device.wrist_normal_device
                    * NORMAL_VIS_LEN,
                    key,
                )
                right_palm_normal_tip = self.get_hand_point_reprojection(
                    hand_tracking_result.right_hand.landmark_positions_device[
                        int(mps.hand_tracking.HandLandmark.PALM_CENTER)
                    ]
                    + hand_tracking_result.right_hand.wrist_and_palm_normal_device.palm_normal_device
                    * NORMAL_VIS_LEN,
                    key,
                )

        scaled_left_wrist = np.array([left_wrist]) * scale if left_wrist is not None else None
        scaled_left_palm = np.array([left_palm]) * scale if left_palm is not None else None
        scaled_right_wrist = np.array([right_wrist]) * scale if right_wrist is not None else None
        scaled_right_palm = np.array([right_palm]) * scale if right_palm is not None else None
        scaled_left_wrist_normal = np.array([left_wrist_normal_tip]) * scale if left_wrist_normal_tip is not None else None
        scaled_left_palm_normal = np.array([left_palm_normal_tip]) * scale if left_palm_normal_tip is not None else None
        scaled_right_wrist_normal = np.array([right_wrist_normal_tip]) * scale if right_wrist_normal_tip is not None else None
        scaled_right_palm_normal = np.array([right_palm_normal_tip]) * scale if right_palm_normal_tip is not None else None
        scaled_right_landmarks = [[v * scale for v in point] for point in right_landmarks if point is not None] if right_landmarks is not None else None
        scaled_left_landmarks = [[v * scale for v in point] for point in left_landmarks if point is not None] if left_landmarks is not None else None

        
        return (
            scaled_left_wrist,
            scaled_left_palm,
            scaled_right_wrist,
            scaled_right_palm,
            scaled_left_wrist_normal,
            scaled_left_palm_normal,
            scaled_right_wrist_normal,
            scaled_right_palm_normal,
            scaled_left_landmarks,
            scaled_right_landmarks
        )
    
    def get_hand_data(self, hand_path:str, start_index=0, end_index=None):

        '''
        Extracts hand and wrist poses from the VRS file based on the index/time domain. The hand and wrist poses are extracted from the following streams:
        - camera-eyetracking
        '''

        hand_tracking_results = mps.hand_tracking.read_hand_tracking_results(hand_path)


        if start_index == 0 and end_index == None:
            num_hw = len(hand_tracking_results)
            hw_ts = [hand_tracking_results[i].tracking_timestamp.total_seconds() * 1e9 for i in range(num_hw)]
        else:
            hw_ts = [hand_tracking_results[i].tracking_timestamp.total_seconds() * 1e9 for i in range(start_index,end_index)]


        
        hand_landmarks = {}

        for ts in hw_ts:
            hand_point = get_nearest_hand_tracking_result(hand_tracking_results, ts)

            if hand_point is not None:

                (left_wrist,left_palm,right_wrist,right_palm,left_wrist_normal,left_palm_normal,
                 right_wrist_normal,right_palm_normal,left_landmarks,right_landmarks) = self.get_landmark_pixels('rgb', hand_point)
                
                if left_wrist is not None:
                    left_wrist =  [512 - left_wrist[0][1], left_wrist[0][0]]
                if left_palm is not None:
                    left_palm =  [512 - left_palm[0][1], left_palm[0][0]]
                if right_wrist is not None:
                    right_wrist =  [512 - right_wrist[0][1], right_wrist[0][0]]
                if right_palm is not None:
                    right_palm =  [512 - right_palm[0][1], right_palm[0][0]]


        
                hand_landmarks[ts] = {
                    "left_wrist": left_wrist ,
                    "left_palm": left_palm ,
                    "right_wrist": right_wrist , 
                    "right_palm": right_palm ,
                    "left_wrist_normal": left_wrist_normal ,
                    "left_palm_normal": left_palm_normal ,
                    "right_wrist_normal": right_wrist_normal ,
                    "right_palm_normal": right_palm_normal ,
                    "left_landmarks": left_landmarks ,
                    "right_landmarks": right_landmarks 
                    }
        

                
        self.result['hand_landmarks'] = hand_landmarks

    def get_slam_data(self, slam_path:str, start_index=0, end_index=None):

        open_loop_points = {}
        closed_loop_points = {}

        open_loop_path = os.path.normpath(os.path.join(slam_path, "open_loop_trajectory.csv"))
        closed_loop_path = os.path.normpath(os.path.join(slam_path, "closed_loop_trajectory.csv"))

        open_loop_traj = mps.read_open_loop_trajectory(open_loop_path)
        closed_loop_traj = mps.read_closed_loop_trajectory(closed_loop_path)

        open_loop_ts = [open_loop_traj[i].tracking_timestamp.total_seconds() * 1e9 for i in range(len(open_loop_traj))]
        closed_loop_ts = [closed_loop_traj[i].tracking_timestamp.total_seconds() * 1e9 for i in range(len(closed_loop_traj))]

        for ts in open_loop_ts:
            open_loop_point = get_nearest_pose(open_loop_traj, ts)

            if open_loop_point is not None:

                # print(f'open loop point',open_loop_point)

                T_world_device = open_loop_point.transform_odometry_device
                tworld_rotation = T_world_device.rotation().to_matrix()
                tworld_translation = T_world_device.translation()
                
                rgb_stream_label = self.provider.get_label_from_stream_id(self.stream_mappings['camera-rgb'])
                device_calibration = self.provider.get_device_calibration()
                rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)

                T_device_rgb_camera = rgb_camera_calibration.get_transform_device_camera()
                tdevice_rgb_trans = T_device_rgb_camera.translation()
                tdevice_rgb_rotation = T_device_rgb_camera.rotation().to_matrix()


                T_world_rgb_camera = T_world_device @ T_device_rgb_camera
                tworld_rgb_trans = T_world_rgb_camera.translation()
                tworld_rgb_rotation = T_world_rgb_camera.rotation().to_matrix()


                open_loop_point = {
                    "tracking_timestamp": open_loop_point.tracking_timestamp,
                    "quality_score": open_loop_point.quality_score,
                    "linear_velocity": open_loop_point.device_linear_velocity_odometry,
                    "angular_velocity_device": open_loop_point.angular_velocity_device,
                    "t_world_device_rotation": tworld_rotation,
                    "t_world_device_translation": tworld_translation,
                    "gravity": open_loop_point.gravity_odometry,
                    "world_rgb_camera_translation": tworld_rgb_trans,
                    "world_rgb_camera_rotation": tworld_rgb_rotation,
                    "device_rgb_camera_rotation": tdevice_rgb_rotation,
                    "device_rgb_camera_translation": tdevice_rgb_trans,

                }
                
                open_loop_points[ts] = open_loop_point
            
        self.result['open_loop'] = open_loop_points

        for ts in closed_loop_ts:

            closed_loop_point = get_nearest_pose(closed_loop_traj, ts)

            if closed_loop_point is not None:

                T_world_device = closed_loop_point.transform_world_device
                tworld_rotation = T_world_device.rotation().to_matrix()
                tworld_translation = T_world_device.translation()
                
                rgb_stream_label = self.provider.get_label_from_stream_id(self.stream_mappings['camera-rgb'])
                device_calibration = self.provider.get_device_calibration()
                rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)

                T_device_rgb_camera = rgb_camera_calibration.get_transform_device_camera()
                tdevice_rgb_trans = T_device_rgb_camera.translation()
                tdevice_rgb_rotation = T_device_rgb_camera.rotation().to_matrix()
            
                T_world_rgb_camera = T_world_device @ T_device_rgb_camera
                tworld_rgb_trans = T_world_rgb_camera.translation()
                tworld_rgb_rotation = T_world_rgb_camera.rotation().to_matrix()


                closed_loop_point = {
                    "tracking_timestamp": closed_loop_point.tracking_timestamp,
                    "quality_score": closed_loop_point.quality_score,
                    "linear_velocity": closed_loop_point.device_linear_velocity_device,
                    "angular_velocity_device": closed_loop_point.angular_velocity_device,
                    "t_world_device_rotation": tworld_rotation,
                    "t_world_device_translation": tworld_translation,
                    "gravity": closed_loop_point.gravity_world,
                    "world_rgb_camera_translation": tworld_rgb_trans,
                    "world_rgb_camera_rotation": tworld_rgb_rotation,
                    "device_rgb_camera_rotation": tdevice_rgb_rotation,
                    "device_rgb_camera_translation": tdevice_rgb_trans,
                }


                closed_loop_points[ts] = closed_loop_point


        self.result['closed_loop'] = closed_loop_points

    def get_IMU_data(self, start_index=0, end_index=None):

        '''
        Extracts all IMU data from the VRS file
        '''

    
        imu_right, imu_left = {},{}

        imu_right_label = 'imu-right'
        imu_left_label = 'imu-left'

        num_data_imu_right = self.provider.get_num_data(self.provider.get_stream_id_from_label(imu_right_label))
        num_data_imu_left = self.provider.get_num_data(self.provider.get_stream_id_from_label(imu_left_label))

        if end_index is None:
            end_index = num_data_imu_right
    

        imu_right_ts = self.provider.get_timestamps_ns(self.stream_mappings['imu-right'], self.time_domain)
        imu_left_ts = self.provider.get_timestamps_ns(self.stream_mappings['imu-left'], self.time_domain)


        for ind in range(start_index, end_index):
            imu_right_point = self.provider.get_imu_data_by_index(self.stream_mappings['imu-right'], ind)
            right_data = [ imu_right_point.accel_msec2[0],
                            imu_right_point.accel_msec2[1],
                            imu_right_point.accel_msec2[2],
                            imu_right_point.gyro_radsec[0],
                            imu_right_point.gyro_radsec[1],
                            imu_right_point.gyro_radsec[2]]

            if ind < num_data_imu_left:
                imu_left_point = self.provider.get_imu_data_by_index(self.stream_mappings['imu-left'], ind)
                left_data = [ imu_left_point.accel_msec2[0], 
                            imu_left_point.accel_msec2[1],
                            imu_left_point.accel_msec2[2],
                            imu_left_point.gyro_radsec[0],
                            imu_left_point.gyro_radsec[1],
                            imu_left_point.gyro_radsec[2]]
                             

                imu_left[imu_left_ts[ind]] = left_data
                
            imu_right[imu_right_ts[ind]] = right_data
        
        
        self.result['imu_right'] = imu_right
        self.result['imu_left'] = imu_left

        print(f"Extracted {len(self.result['imu_right'])} data points from {imu_right_label} stream")
        print(f"Extracted {len(self.result['imu_left'])} data points from {imu_left_label} stream")


        '''
        Save the extracted data to the output path
        '''

    def get_GPS_data(self, start_index=0, end_index=None):

        num_gps = self.provider.get_num_data(self.provider.get_stream_id_from_label('gps'))
        gps_ts = self.provider.get_timestamps_ns(self.stream_mappings['gps'], self.time_domain)

        if end_index is None:
            end_index = num_gps

        gps_data = {}
        for index in range(start_index, end_index):
            gps_point = self.provider.get_gps_data_by_index(self.stream_mappings['gps'], index)

            p = {
                    "accuracy": gps_point.accuracy,
                    "altitude": gps_point.altitude,
                    "capture_timestamp_ns": gps_point.capture_timestamp_ns,
                    "latitude": gps_point.latitude,
                    "longitude": gps_point.longitude,
                    "provider": gps_point.provider,
                    "raw_data": gps_point.raw_data,
                    "speed": gps_point.speed,
                    "utc_time_ms": gps_point.utc_time_ms,
                    "verticalAccuracy": gps_point.verticalAccuracy,
                }
            gps_data[gps_ts[index]] = p
        
        self.result['gps'] = gps_data
        print(f"Extracted {len(self.result['gps'])} data points from gps stream")

    def ego_blur(self, input_labels:str, frames):

        '''
        Apply ego-blur to the extracted images from the VRS file to make data anonymous
        '''

        egoblur_face_path = self.face_ego_blur
        egoblur_lp_path = self.lp_ego_blur


        face_model_score_threshold = 0.9
        lp_model_score_threshold = 0.9
        nms_iou_threshold = 0.3
        scale_factor_detections = 1.0


        face_detector = torch.jit.load(egoblur_face_path, map_location="cpu").to(
            self.get_device()
        )
        face_detector.eval()

        lp_detector = torch.jit.load(egoblur_lp_path, map_location="cpu").to(
            self.get_device()
        )
        lp_detector.eval()

        data = pd.read_csv(input_labels)



        lp_start = data[data['type'] == 'license_plate']['start_frame'].tolist()
        lp_end = data[data['type'] == 'license_plate']['end_frame'].tolist()
        
        face_start = data[data['type'] == 'face']['start_frame'].tolist()
        face_end = data[data['type'] == 'face']['end_frame'].tolist()

        lp_indices = []

        for start, end in zip(lp_start, lp_end):
            lp_indices.extend(range(start, end + 1))

        face_indices = []
        for start, end in zip(face_start, face_end):
            face_indices.extend(range(start, end + 1))

        indices = sorted(list(set(lp_indices + face_indices)))

        timestamps = list(frames.keys())
        filtered_ts = [timestamps[index] for index in indices]
        filtframes = [frames[fts] for fts in filtered_ts]

        #dict of labels as keys and frames as values
        
        label_frame_map = {}

        for i in sorted(indices):
            if i in lp_indices and i in face_indices:
                label = 'face_license_plate'
            elif i in lp_indices:
                label = 'license_plate'
            elif i in face_indices:
                label = 'face'

            label_frame_map[i] = label

            
        count = 0


        for i,frame in tqdm.tqdm(zip(sorted(indices),filtframes)):

            label = label_frame_map[i]

            print(f'{round(100*(count/len(filtframes)),2)}% of the frames processed, now with label {label} @ frame {i}.')
            
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            image = bgr_frame.copy()

            image_tensor = ego_blur.get_image_tensor(image)
            image_tensor_copy = image_tensor.clone()

            output_image = ego_blur.visualize_image(
                image,
                image_tensor,
                image_tensor_copy,
                face_detector,
                lp_detector,
                face_model_score_threshold,
                lp_model_score_threshold,
                nms_iou_threshold,
                scale_factor_detections,
                label
            )

            rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
            
            #update the frames with the blurred image
            frame_ts = list(frames.keys())[i]
            frames[frame_ts] = rgb

            count += 1
        print(f'Returing {len(frames)} frames with ego-blur applied')
        return frame
    
    def undistort_gaze(self,point_distorted_px, src_calib, dst_calib):

        ray = src_calib.unproject(point_distorted_px)
        point_undistorted = dst_calib.project(ray)
        return point_undistorted
    
    def annotate(self, frames_dict, actions_csv_path, blur_csv_path, fps=15):
        action_label_map = {
            '1': 'checking right wing mirror',
            '2': 'checking left wing mirror',
            '3': 'checking rear view mirror',
            '4': 'mobile phone usage',
            '5': 'driving',
            '6': 'idle'
        }

        blur_label_map = {
            'f': 'face',
            'l': 'license plate'
        }

        LEFT_KEY = 63234 if os.name == 'posix' else 2424832
        RIGHT_KEY = 63235 if os.name == 'posix' else 2555904
        ESC_KEY = 27
        EXIT_KEY = ord('x')
        FRAME_INTERVAL = 1.0 / fps

        sorted_ts = sorted(frames_dict.keys())

        len_pgaze = len(self.result.get('personalized_gaze', {}))
        len_gaze = len(self.result.get('gaze', {}))
        
        if len_pgaze > len_gaze:
            gaze_points = list(self.result['personalized_gaze'].values())[::2]
        else:
            gaze_points = list(self.result['gaze'].values())[::2]
            print(len(gaze_points), "gaze points available for annotation")

        if not sorted_ts:
            print("No frames to label!")
            return

        current_idx = 0
        is_playing = False
        last_frame_time = time()

        num_frames = len(sorted_ts)
        frame_labels = {i: set() for i in range(num_frames)}
        blur_labels = {i: set() for i in range(num_frames)}

        current_actions = set()
        current_blurs = set()

        window_name = "Frame Labeler"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        def propagate_labels(start_idx, label_set, label_dict):
            for i in range(start_idx, num_frames):
                label_dict[i] = set(label_set)

        try:
            while True:
                if is_playing:
                    now = time()
                    if now - last_frame_time >= FRAME_INTERVAL:
                        new_idx = current_idx + 1
                        if new_idx < len(sorted_ts):
                            current_idx = new_idx
                            last_frame_time = now
                        else:
                            is_playing = False

                ts = sorted_ts[current_idx]
                frame = frames_dict[ts].copy()

                if current_idx >= len(gaze_points):
                    projection = None
                    depth = None
                else:
                    projection = gaze_points[current_idx]['projection']
                    depth = gaze_points[current_idx]['depth']

                # Show current labels (inherited)
                active_actions = frame_labels[current_idx]
                active_blurs = blur_labels[current_idx]

                status = []
                if is_playing:
                    status.append(f"[PLAYING {fps}FPS]")
                if active_actions:
                    status.extend(list(active_actions))
                if active_blurs:
                    status.extend([f"blur:{b}" for b in active_blurs])

                cv2.putText(frame, f"ACTIVE: {', '.join(status) or 'None'}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Frame {current_idx + 1}/{len(sorted_ts)} (X: exit, SPACE: play/pause)",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if projection is not None:
                    frame = cv2.circle(frame, (int(projection[0]), int(projection[1])), 6, (255, 0, 0), 3)

                cv2.imshow(window_name, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                key = cv2.waitKeyEx(1)

                if key in (ESC_KEY, EXIT_KEY):
                    break

                if key == LEFT_KEY:
                    current_idx = max(0, current_idx - 1)
                    is_playing = False
                elif key == RIGHT_KEY:
                    current_idx = min(len(sorted_ts) - 1, current_idx + 1)
                    is_playing = False
                elif key == 32:  # Space
                    is_playing = not is_playing
                    last_frame_time = time()

                else:
                    try:
                        char_key = chr(key).lower()
                        # Action label toggle
                        if char_key in action_label_map:
                            label = action_label_map[char_key]
                            if label in current_actions:
                                current_actions.remove(label)
                            else:
                                current_actions.add(label)
                            propagate_labels(current_idx, current_actions, frame_labels)

                        # Blur label toggle
                        elif char_key in blur_label_map:
                            blur = blur_label_map[char_key]
                            if blur in current_blurs:
                                current_blurs.remove(blur)
                            else:
                                current_blurs.add(blur)
                            propagate_labels(current_idx, current_blurs, blur_labels)

                    except ValueError:
                        pass

            # Write to CSV
            def write_contiguous_blocks(label_dict, path, label_type):
                with open(path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['start_frame', 'end_frame', label_type])

                    prev_labels = None
                    start_idx = 0
                    for idx in range(num_frames):
                        current_labels = label_dict[idx]
                        if current_labels != prev_labels:
                            if prev_labels:
                                for lbl in prev_labels:
                                    writer.writerow([start_idx, idx - 1, lbl])
                            start_idx = idx
                            prev_labels = current_labels

                    if prev_labels:
                        for lbl in prev_labels:
                            writer.writerow([start_idx, num_frames - 1, lbl])

            write_contiguous_blocks(frame_labels, actions_csv_path, 'action')
            write_contiguous_blocks(blur_labels, blur_csv_path, 'type')

        finally:
            cv2.destroyAllWindows()

    def _handle_frame_advance(self, sorted_ts, start_idx, count, output_csv, label_face, label_lp):
        """Records labels for all frames passed during advance"""
        labels = []
        if label_face: labels.append('face')
        if label_lp: labels.append('license_plate')
        
        if labels:
            with open(output_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                for i in range(1, count+1):
                    ts = sorted_ts[start_idx + i]
                    for label in labels:
                        writer.writerow([start_idx + i, label])

    def save_data(self,output_path:str):
        '''
        Save the extracted data to the output path
        '''

        np.save(output_path, self.result)
        print(f"Data saved to {output_path}")

    def get_object_dets(self, progress_callback=None):
        '''
        Get automotive object detection results from the VRS file
        '''

        
        classes = {
            0: 'Gear Stick',
            1: 'Infotainment Unit',
            2: 'Left Wing Mirror',
            3: 'Mobile Phone',
            4: 'Rearview Mirror',
            5: 'Right Wing Mirror',
            6: 'Speedometer',
            7: 'Steering Wheel'
        }
        
        model_weight_path = '/Users/michaelrice/Documents/GitHub/Thesis/MSc_AI_Thesis/utilities/InCabinObjectDet.pt'
        model = YOLO(model_weight_path)
        results = []
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        model.to(device)


        rgb_values = list(self.result['rgb'].values())

        for i, image in enumerate(tqdm.tqdm(rgb_values, desc="Object Detection")):
            result = model(image, verbose=False)
            image_dets = []

            if result[0].boxes is not None:
                for j in range(len(result[0].boxes.xyxy)):
                    if result[0].boxes.conf[j] > 0.25:
                        image_dets.append({
                            'class': classes[int(result[0].boxes.cls[j])],
                            'confidence': result[0].boxes.conf[j],
                            'bounding_box': result[0].boxes.xyxy[j]
                        })
                        
                        
                        rand = random.random()
                        if rand < 0.01:
                            rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(f'/Users/michaelrice/Documents/GitHub/Thesis/MSc_AI_Thesis/data/incabin_object_detection_datasets/phone_stand_images/{i}.jpg', rgb)

            results.append(image_dets)

            # Call progress update
            if progress_callback:
                progress_callback(i + 1, len(rgb_values))

        self.result['object_detections'] = results

    def generate_gaussian_mask(self, shape, center, sigma=20):
        x = np.arange(0, shape[1])
        y = np.arange(0, shape[0])
        x, y = np.meshgrid(x, y)
        d = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        g = np.exp(-(d**2 / (2.0 * sigma**2)))
        return g

    def overlay_gaze_heatmap(self, frame, center, angle_error, threshold=0.05):
        h, w = frame.shape[:2]
        fov = 110

        pix_per_deg = w / fov
        radius = int(angle_error * pix_per_deg)
        xmin = max(0, center[0] - radius)
        ymin = max(0, center[1] - radius)
        xmax = min(w - 1, center[0] + radius)
        ymax = min(h - 1, center[1] + radius)

        sigma = radius / 2
        mask = self.generate_gaussian_mask((h, w), center, sigma)

        # Apply a threshold to keep only visible regions
        mask = (mask * 255).astype(np.uint8)
        _, mask_thresh = cv2.threshold(mask, int(threshold * 255), 255, cv2.THRESH_BINARY)

        # Create color heatmap
        heatmap_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)


        # Blend only the gaze region
        mask_bool = mask_thresh > 0
        blended = frame.copy()
        blended[mask_bool] = cv2.addWeighted(frame, 0.5, heatmap_color, 0.5, 0)[mask_bool]

        return blended, xmin, ymin, xmax, ymax

    def evaluate_driving(self,frames, smoothed_gaze, object_dets, imu_gx,imu_gy,imu_gz,hand_lm, video_save_path, progress_callback=None, gaze_predictions = None):
        consecutive_frames_threshold = 8
        overlays = []

        action_tracking = {
            "Left Wing Mirror": [],
            "Right Wing Mirror": [],
            "Rearview Mirror": [],
            "Mobile Phone": []
        }

        action_counters = {
            "Left Wing Mirror": 0,
            "Right Wing Mirror": 0,
            "Rearview Mirror": 0,
            "Mobile Phone": 0,
            "Cross Steering": 0
        }

        direction = "Forward"

        if gaze_predictions is not None:
            smoothed_gaze = gaze_predictions

        crossover_frame_count = 0
        hands_on_wheel = 0
        gear_change_frame_counter = 0
        gear_change_action_counter = 0
        threshold_frames = 10
        one_handed_driving_counter= 0
        bad_steering_counter = 0
        infotainment_distraction_counter = 0
        speedometer_checks = 0

        fixations = []

        for i, (frame, sg, od, hlm) in enumerate(zip(frames, smoothed_gaze, object_dets, hand_lm)):
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if gaze_predictions is None:
                sg = int(sg[0] * (512 / 1408)), int(sg[1] * (512 / 1408))

            if i == 0:
                prev_gaze = sg
                gaze_fixation_duration = 0
            else:
                prev_gaze = smoothed_gaze[i - 1]
                if sg is not None and prev_gaze is not None:
                    distance = math.hypot(sg[0] - prev_gaze[0], sg[1] - prev_gaze[1])
                    if distance < 30:  # Threshold for fixation
                        gaze_fixation_duration += 1
                    else:
                        fixations.append(gaze_fixation_duration)
                        gaze_fixation_duration = 0



            overlay, xmin, ymin, xmax, ymax = self.overlay_gaze_heatmap(frame_rgb, sg, angle_error=4, threshold=0.1)
            gaussian_area = (xmax - xmin) * (ymax - ymin)

            imu_per_frame = 1000 / 15 

            gx_samples = imu_gx[i * int(imu_per_frame):(i + 1) * int(imu_per_frame)]
            gy_samples = imu_gy[i * int(imu_per_frame):(i + 1) * int(imu_per_frame)]
            gz_samples = imu_gz[i * int(imu_per_frame):(i + 1) * int(imu_per_frame)]
            
            max_gx = gx_samples[np.argmax(np.abs(gx_samples))]
            max_gy = gy_samples[np.argmax(np.abs(gy_samples))]
            max_gz = gz_samples[np.argmax(np.abs(gz_samples))]

            left_landmarks = hlm['left_landmarks']
            right_landmarks = hlm['right_landmarks']

            left_wrist = hlm['left_wrist'] if 'left_wrist' in hlm else None
            left_palm = hlm['left_palm'] if 'left_palm' in hlm else None
            right_wrist = hlm['right_wrist'] if 'right_wrist' in hlm else None
            right_palm = hlm['right_palm'] if 'right_palm' in hlm else None


            if all(x is not None for x in [left_wrist, left_palm, right_wrist, right_palm]):
                for b in od:
                    if b['class'] == 'Steering Wheel':
                        x1, y1, x2, y2 = b['bounding_box']
                        wheel_center = ((x1 + x2) / 2, (y1 + y2) / 2)

                        left_dist = min(
                            math.hypot(wheel_center[0] - left_wrist[0], wheel_center[1] - left_wrist[1]),
                            math.hypot(wheel_center[0] - left_palm[0], wheel_center[1] - left_palm[1])
                        )
                        right_dist = min(
                            math.hypot(wheel_center[0] - right_wrist[0], wheel_center[1] - right_wrist[1]),
                            math.hypot(wheel_center[0] - right_palm[0], wheel_center[1] - right_palm[1])
                        )

                        if left_dist < 80 and right_dist < 80:  # Adjust threshold as needed
                            hands_on_wheel += 1

                        cv2.putText(overlay, f"Hands on Wheel: {hands_on_wheel}", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    

                        
            if (right_wrist is not None and right_palm is not None) and (left_wrist is None and left_palm is None):
                gear_change_frame_counter += 1

            if (left_wrist is not None and left_palm is not None) and gear_change_frame_counter >= 0:
                if gear_change_frame_counter >= 1 and gear_change_frame_counter < 30:
                    gear_change_frame_counter = 0  
                    gear_change_action_counter += 1
                    cv2.putText(overlay, "Potential Gear Change Occurred", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


            if ((right_wrist is not None and right_palm is not None) and (left_wrist is None and left_palm is None)) or ((left_wrist is not None and left_palm is not None) and (right_wrist is None and right_palm is None)):
                one_handed_driving_counter += 1
                cv2.putText(overlay, "Potential One-Handed Driving Detected", (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # bad steering detection


            if right_palm is not None and left_palm is not None:
                rightpalm_x = right_palm[0]
                leftpalm_x = left_palm[0]

                if rightpalm_x is not None and leftpalm_x is not None:
                    crossover = (rightpalm_x - leftpalm_x) < 0

                    # Add to a moving window or debounce using a frame counter
                    if crossover:
                        crossover_frame_count += 1
                    else:
                        crossover_frame_count = 0

                    if crossover_frame_count > threshold_frames:
                        bad_steering_counter += 1
                        crossover_frame_count = 0  # reset or keep counting if continuous
                    cv2.putText(overlay, f"Potential Hand Crossover Detected ", (25, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            
            scale = 512 / 1408
            image_width = 512


            if left_landmarks is not None:
                left_landmarks = [
                    (image_width - l[1] * scale, l[0] * scale) if l is not None else None
                    for l in left_landmarks
                ]

            if right_landmarks is not None:
                right_landmarks = [
                    (image_width - r[1] * scale, r[0] * scale) if r is not None else None
                    for r in right_landmarks
                ]

                
            overlay = self.draw_landmarks_and_connections(overlay,left_landmarks,right_landmarks,mps.hand_tracking.kHandJointConnections)
            

            if direction == "Forward":
                if max_gx > 1.0:
                    direction = "Right"
                elif max_gx < -1.0:
                    direction = "Left"
                

            elif direction == "Right":
                if max_gx < -0.5:
                    direction = "Forward"

            elif direction == "Left":
                if max_gx > 0.5:
                    direction = "Forward"

            
            cv2.putText(overlay, f"Head Direction: {direction}", (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            for b in od:
                if b['class'] == 'Infotainment Unit':
                    x1, y1, x2, y2 = b['bounding_box']
                    gx, gy = sg[0], sg[1]
                    if x1 <= gx <= x2 and y1 <= gy <= y2:
                        infotainment_distraction_counter += 1
                        cv2.putText(overlay, "Infotainment Distraction Detected", (25, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                
                if b['class'] == 'Speedometer':
                    x1, y1, x2, y2 = b['bounding_box']
                    gx, gy = sg[0], sg[1]
                    if (x1 <= gx <= x2 and y1 <= gy <= y2) and s == False:
                        s = True
                        speedometer_checks += 1
                        cv2.putText(overlay, "Speedometer Check Detected", (25, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        s = False
                    
            # for b in od:
            #     x1, y1, x2, y2 = b['bounding_box']
            #     bbox_area = (x2 - x1) * (y2 - y1)
            #     label = b['class']

            #     # if "Phone" in label:
            #     #     action_tracking[label].append(i)
            #     #     cv2.putText(overlay, f"Mobile Phone In Use", (25, 75),
            #     #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


            #     if "Mirror" in label:
            #         width, height = x2 - x1, y2 - y1
            #         x1 = x1.clone() - (0.3 * width)
            #         x2 = x2.clone() + (0.3 * width)
            #         y1 = y1.clone() - (0.3 * height)
            #         y2 = y2.clone() + (0.3 * height)

            #         intersection_area = max(0, min(x2, xmax) - max(x1, xmin)) * max(0, min(y2, ymax) - max(y1, ymin))
            #         union_area = bbox_area + gaussian_area - intersection_area
            #         IoU = intersection_area / union_area if union_area > 0 else 0

            #         if IoU > 0.25:
            #             gaze_on_mirror[label] = True

            # for mirror in mirror_counters:
            #     if gaze_on_mirror[mirror]:
            #         mirror_counters[mirror] += 1
            #     else:
            #         mirror_counters[mirror] = 0

            #     if mirror_counters[mirror] >= consecutive_frames_threshold:
            #         if i not in action_tracking[mirror]:
            #             action_tracking[mirror].append(i)

            #         cv2.putText(overlay, f"{mirror} Check Detected", (25, 50 + 30 * list(mirror_counters.keys()).index(mirror)),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



        
            if progress_callback:
                progress_callback(i + 1, len(frames))

            overlays.append(overlay)
        
        # Create an MP4 video from overlays
        if not os.path.exists(video_save_path):
            os.makedirs(video_save_path)
        video_save_path = os.path.join(video_save_path, "driving_evaluation.mp4")
        height, width, _ = overlays[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_save_path, fourcc, 15, (width, height))

        for overlay in overlays:
            video_writer.write(overlay)

        video_writer.release()

        mean_gaze_fixation_duration = np.mean(fixations) if fixations else 0
        self.result['mean_gaze_fixation_duration'] = mean_gaze_fixation_duration
        self.result['one_handed_percent'] = round((one_handed_driving_counter - gear_change_frame_counter) / (len(hand_lm)) * 100 , 4)
        self.result['crossover_percent'] = round(bad_steering_counter / (len(hand_lm)) * 100, 4)
        self.result['infotainment_distraction_percent'] = round(infotainment_distraction_counter / (len(hand_lm)) * 100, 4)
        self.result['speedometer_checks'] = speedometer_checks


        self.result['video_path'] = video_save_path
        self.result['overlays'] = overlays
        self.result['action_tracking'] = action_tracking
        self.result['joined_intervals'] = self.join_action_interval(action_tracking)

    def draw_landmarks_and_connections(self, image, left_landmarks, right_landmarks, connections):
        def draw_point(img, point, color):
            cv2.circle(img, (int(point[0]), int(point[1])), 5, color, -1)

        def draw_line(img, point1, point2, color):
            cv2.line(img, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), color, 2)

        if left_landmarks:
            for left_landmark in left_landmarks:
                if left_landmark is not None:
                    draw_point(image, left_landmark, (255, 0, 0))  # Blue for left landmarks
            for connection in connections:
                if left_landmarks[int(connection[0])] is not None and left_landmarks[int(connection[1])] is not None:
                    draw_line(image, left_landmarks[int(connection[0])], left_landmarks[int(connection[1])], (255, 0, 0))

        if right_landmarks:
            for right_landmark in right_landmarks:
                if right_landmark is not None:
                    draw_point(image, right_landmark, (0, 0, 255))  # Red for right landmarks
            for connection in connections:
                if right_landmarks[int(connection[0])] is not None and right_landmarks[int(connection[1])] is not None:
                    draw_line(image, right_landmarks[int(connection[0])], right_landmarks[int(connection[1])], (0, 0, 255))

        return image

    def plot_wrists_and_palms(self, plt,left_wrist,left_palm,right_wrist,right_palm,left_wrist_normal_tip,left_palm_normal_tip,right_wrist_normal_tip,right_palm_normal_tip,img_height):
        
        def plot_point(point, color):
            plt.plot(img_height - 0.5 - point[1], point[0] + 0.5, ".", c=color, mew=1, ms=15)

        def plot_arrow(point, vector, color):
            plt.arrow(img_height - 0.5 - point[1], point[0] + 0.5, -vector[1], vector[0], color=color)

        if left_wrist is not None:
            plot_point(left_wrist, "blue")
        if left_palm is not None:
            plot_point(left_palm, "blue")
        if right_wrist is not None:
            plot_point(right_wrist, "red")
        if right_palm is not None:
            plot_point(right_palm, "red")
        if left_wrist_normal_tip is not None and left_wrist is not None:
            plot_arrow(left_wrist, left_wrist_normal_tip - left_wrist, "blue")
        if left_palm_normal_tip is not None and left_palm is not None:
            plot_arrow(left_palm, left_palm_normal_tip - left_palm, "blue")
        if right_wrist_normal_tip is not None and right_wrist is not None:
            plot_arrow(right_wrist, right_wrist_normal_tip - right_wrist, "red")
        if right_palm_normal_tip is not None and right_palm is not None:
            plot_arrow(right_palm, right_palm_normal_tip - right_palm, "red")

    def join_action_interval(self, action_tracking):
        joined_actions = {}
        
        for action, frames in action_tracking.items():
            if not frames:
                continue
            
            # Sort frames and join consecutive intervals
            frames = sorted(frames)
            intervals = []
            start = frames[0]
            end = frames[0]
            
            for f in frames[1:]:
                if f == end + 1:  # Consecutive frame
                    end = f
                else:
                    if action == "Mobile Phone":
                        if intervals:
                            if start > intervals[-1][1] + 10:
                                intervals.append([start, end])
                            else:
                                intervals[-1][1] = end 
                        else:
                            intervals.append([start, end])
                    else:
                        if (end - start) > 5:
                            intervals.append([start, end])

                    start = f
                    end = f

             # Extend last interval to include the last frame
            intervals.append((start, end))  # Add last interval
            joined_actions[action] = intervals
        
        return joined_actions        

    def score_driver(self, num_frames, action_intervals, video_save_path):
        """
        Score the driving based on detected actions.
        Returns an overall score between 0 and 1, as well as other statistics.
        """

        score = 100

        rwm = 0
        lwm = 0
        rvm = 0
        mistake_sections = []
        num_mp_frames = 0


        mp_ints = action_intervals.get("Mobile Phone", [])

        if mp_ints:
            for start, end in mp_ints:
                num_mp_frames += (end - start)
                mistake_sections.append((start, end, f'Mobile phone usage, seconds {int(start/15)} to {int(end/15)}'))
            
            

        segment_intervals = {i: [] for i in range(0, num_frames, 30*15)}
        
        for action, intervals in action_intervals.items():
            for start, end in intervals:
                for bin_start in segment_intervals.keys():
                    bin_end = bin_start + 30 * 15
                    if start >= bin_start and end <= bin_end:
                        segment_intervals[bin_start].append((action,start, end))
                        break
        
        for seg, actions in segment_intervals.items():
            if not actions:
                rwm += 1
                lwm += 1
                rvm += 1
                mistake_sections.append((seg, seg + 450, f'No mirror checks, seconds {int(seg/15)} to {int(seg/15) + 30}'))
                continue
            actions = [a[0] for a in actions]

            if "Left Wing Mirror" not in actions:
                lwm += 1
            if "Right Wing Mirror" not in actions:
                rwm += 1
            if "Rearview Mirror" not in actions:
                rvm += 1
        
        lwm_percent = round((lwm / len(segment_intervals)) * 100 , 4)
        rwm_percent = round((rwm / len(segment_intervals)) * 100 , 4)
        rvm_percent = round((rvm / len(segment_intervals)) * 100 , 4)

        mp = int((num_mp_frames / num_frames) * 100 if num_frames > 0 else 0)
        
        score = round((lwm_percent + rwm_percent + rvm_percent) / 3, 4)

        

        self.result['scores'] = (score, lwm_percent, rwm_percent, rvm_percent, mp)
        frames = self.result['rgb']
        frames = list(frames.values())
        mistake_video_paths = []

        for m in mistake_sections:
            start, end, reason = m
            mistake_frames = frames[start:end]
            mistake_video_path = os.path.join(video_save_path, f'{reason}.mp4')
            height, width, _ = mistake_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(mistake_video_path, fourcc, 15, (width, height))
            for frame in mistake_frames:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_writer.write(rgb_frame)
            video_writer.release()
            mistake_video_paths.append(mistake_video_path)
        
        self.result['mistake_video_paths'] = mistake_video_paths
        self.result['mistake_sections'] = mistake_sections
        


        

        


    








        




    
        
    