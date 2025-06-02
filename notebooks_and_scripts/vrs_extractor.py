
import os
import numpy as np
from PIL import Image
import torch
import cv2
import csv
from time import time
import pandas as pd
import tqdm
import io
from ultralytics import YOLO
from typing import Dict, List, Optional
from filterpy.kalman import KalmanFilter

'''
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
'''
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId
import projectaria_tools.core.mps as mps
from projectaria_tools.core.mps.utils import (
    get_nearest_wrist_and_palm_pose,
    get_gaze_vector_reprojection,
    get_nearest_eye_gaze,
    get_nearest_pose,
    filter_points_from_confidence,
    get_nearest_hand_tracking_result
)
from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    distort_by_calibration,
    get_linear_camera_calibration,
)
os.environ['YOLO_VERBOSE'] = 'False'

import sys
sys.path.append('/Users/michaelrice/Documents/GitHub/Thesis/MSc_AI_Thesis/notebooks_and_scripts/unused/')
from OtherModelScripts import ego_blur 


class VRSDataExtractor():


    def __init__(self, vrs_path: str):
        self.path = vrs_path
        self.provider = data_provider.create_vrs_data_provider(self.path)
        
        self.time_domain = TimeDomain.DEVICE_TIME  # query data based on host time
        self.option = TimeQueryOptions.CLOSEST # get data whose time [in TimeDomain] is CLOSEST to query time
        
        self.provider.set_devignetting(True)
        self.provider.set_devignetting_mask_folder_path('/Users/michaelrice/devignetting_masks')
        
        self.device_calibration = self.provider.get_device_calibration()

        
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
        self.mp_hand_landmarker_task_path = 'C:/Users/athen/Desktop/Github/MastersThesis/MSc_AI_Thesis/Coding/other/hand_landmarker.task'
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
        etlabel = 'camera-eyetracking'
        left_slam_label  = 'camera-slam-left'
        right_slam_label = 'camera-slam-right'

        num_frames_rgb = self.provider.get_num_data(self.provider.get_stream_id_from_label(rgblabel))
        # num_frames_eth = self.provider.get_num_data(self.provider.get_stream_id_from_label(etlabel))
        # num_frames_slam_left = self.provider.get_num_data(self.provider.get_stream_id_from_label(left_slam_label))
        # num_frames_slam_right = self.provider.get_num_data(self.provider.get_stream_id_from_label(right_slam_label))

        rgb_ts = self.provider.get_timestamps_ns(self.stream_mappings['camera-rgb'], self.time_domain)
        et_ts = self.provider.get_timestamps_ns(self.stream_mappings['camera-eyetracking'], self.time_domain)
        slam_left_ts = self.provider.get_timestamps_ns(self.stream_mappings['camera-slam-left'], self.time_domain)
        slam_right_ts = self.provider.get_timestamps_ns(self.stream_mappings['camera-slam-right'], self.time_domain)



        if end_index is None:
            end_index = num_frames_rgb


        for i,index in enumerate(tqdm.tqdm(range(start_index, end_index), desc= "Extracting Images")):
            # buffer = io.BytesIO()

            image_data = self.provider.get_image_data_by_index(self.stream_mappings['camera-rgb'], index)[0].to_numpy_array()
            image_pil = Image.fromarray(image_data)  
            image_pil = image_pil.rotate(-90) 
            image_pil = image_pil.resize((512, 512)) 

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
            for index in range(start_index, end_index):
                image_data = self.provider.get_image_data_by_index(self.stream_mappings['camera-eyetracking'], index)[0].to_numpy_array()
                et_images[et_ts[index]] = img
                # break
                
            self.result['et'] = et_images
            print(f"Extracted {len(self.result['et'])} images from {etlabel} stream")

            try:
                for index in range(start_index, end_index):
                    left_image_data = self.provider.get_image_data_by_index(self.stream_mappings['camera-slam-left'], index)
                    right_image_data = self.provider.get_image_data_by_index(self.stream_mappings['camera-slam-right'], index)
                    left_img = np.array(Image.fromarray(left_image_data[0].to_numpy_array()))
                    right_img = np.array(Image.fromarray(right_image_data[0].to_numpy_array()))
                    # print(f'slam storage size {sys.getsizeof(left_img.nbytes)}')
                    # slam_img = Image.fromarray(left_img)
                    # slam_img.save('sampledata/imagetesting/slam_left.png', format='PNG', compress_level=0)
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
    
    def get_gaze_data(self, gaze_path:str,personalized_gaze_path=None, start_index=0, end_index=None):

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
                
                gaze_points[ts] = {
                    "projection": gaze_projection,
                    'depth': gaze_point.depth,
                }
        self.result['gaze'] = gaze_points

        p_gaze_points = {}
        # Process personalized gaze data
        if personalized_gaze_path:
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
                    
                    p_gaze_points[ts] = {
                        "projection": gaze_projection,
                        'depth': personalized_gaze_point.depth,
                    }

        self.result['personalized_gaze'] = p_gaze_points

        if personalized_gaze_path:
            gaze = list(p_gaze_points.values())
        else:
            gaze = list(gaze_points.values())

        smoothed_gaze = []
        gaze = [g['projection']*(512/1408) for g in gaze]

        kf = self.create_kalman_filter()
        kf.x[:2] = np.array([[gaze[0][0]], [gaze[0][1]]])  # Initialize with first point

        smoothed_gaze = []

        for z in gaze:
            kf.predict()
            kf.update(np.array(z))
            smoothed_gaze.append((kf.x[0, 0], kf.x[1, 0]))

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


        NORMAL_VIS_LEN = 0.05  # meters

        
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
        
        return (
            left_wrist,
            left_palm,
            right_wrist,
            right_palm,
            left_wrist_normal_tip,
            left_palm_normal_tip,
            right_wrist_normal_tip,
            right_palm_normal_tip,
            left_landmarks,
            right_landmarks
        )
    
    def get_hand_data(self, hand_path:str, start_index=0, end_index=None):

        '''
        Extracts hand and wrist poses from the VRS file based on the index/time domain. The hand and wrist poses are extracted from the following streams:
        - camera-eyetracking
        '''

        hand_tracking_results = mps.hand_tracking.read_hand_tracking_results(hand_path)

        # frame_timestamps = list(hand_frames_data.keys())

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
                
        
                hand_landmarks[ts] = {
                    "left_wrist": left_wrist,
                    "left_palm": left_palm,
                    "right_wrist": right_wrist,
                    "right_palm": right_palm,
                    "left_wrist_normal": left_wrist_normal,
                    "left_palm_normal": left_palm_normal,
                    "right_wrist_normal": right_wrist_normal,
                    "right_palm_normal": right_palm_normal,
                    "left_landmarks": left_landmarks,
                    "right_landmarks": right_landmarks,
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
                            imu_right_point.gyro_radsec[2],
                            imu_right_point.mag_tesla[0],
                            imu_right_point.mag_tesla[1],
                            imu_right_point.mag_tesla[2]]

            if ind < num_data_imu_left:
                imu_left_point = self.provider.get_imu_data_by_index(self.stream_mappings['imu-left'], ind)
                left_data = [ imu_left_point.accel_msec2[0], 
                            imu_left_point.accel_msec2[1],
                            imu_left_point.accel_msec2[2],
                            imu_left_point.gyro_radsec[0],
                            imu_left_point.gyro_radsec[1],
                            imu_left_point.gyro_radsec[2],
                            imu_left_point.mag_tesla[0],
                            imu_left_point.mag_tesla[1],
                            imu_left_point.mag_tesla[2]]
                             

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
    
    def annotate(self, frames_dict, actions_csv, fps=15):
        """
        Annotate frames with:
        - Blur labels (face, license plate)
        - Driving actions
        - Environment labels (rural, town, city, motorway)
        """

        action_label_map = {
            '0': 'checking left wing mirror',
            '1': 'checking right wing mirror',
            '2': 'checking rear view mirror',
            '3': 'nothing'
        }

        LEFT_KEY = 63234 if os.name == 'posix' else 2424832
        RIGHT_KEY = 63235 if os.name == 'posix' else 2555904
        ESC_KEY = 27
        EXIT_KEY = ord('x')
        FRAME_INTERVAL = 1.0 / fps

        # Initialize CSVs
        for csv_file, headers in [
            (actions_csv, ['start_frame', 'end_frame', 'action'])
        ]:
            if not os.path.exists(csv_file):
                with open(csv_file, 'w', newline='') as f:
                    csv.writer(f).writerow(headers)

        # Load existing actions
        existing_actions = []
        if os.path.exists(actions_csv):
            with open(actions_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_actions.append({
                        'start': int(row['start_frame']),
                        'end': int(row['end_frame']),
                        'action': row['action']
                    })

        sorted_ts = sorted(frames_dict.keys())
        gaze_points = list(self.result['gaze'].values())
        src_calib = calibration.get_linear_camera_calibration(1408, 1408, 608.611)
        dst_calib = calibration.get_linear_camera_calibration(512, 512, 170)

        if not sorted_ts:
            print("No frames to label!")
            return

        current_idx = 0
        is_playing = False
        last_frame_time = time()

        active_actions = {}

        current_env_label = None
        env_label_start_idx = None

        window_name = "Frame Labeler"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        def write_csv_row(csv_file, row):
            with open(csv_file, 'a', newline='') as f:
                csv.writer(f).writerow(row)

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

                projection = gaze_points[current_idx]['projection'] * (512 / 1408)
                depth = gaze_points[current_idx]['depth']

                # Check existing labels
                active_csv_actions = [
                    entry['action'] for entry in existing_actions
                    if entry['start'] <= current_idx <= entry['end']
                ]

                status = []
                if is_playing:
                    status.append(f"[PLAYING {fps}FPS]")
                for action_key in active_actions:
                    status.append(action_label_map[action_key])
                for act in active_csv_actions:
                    if act not in status:
                        status.append(act)

                cv2.putText(frame, f"ACTIVE: {', '.join(status) or 'None'}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Frame {current_idx + 1}/{len(sorted_ts)} (X: exit, SPACE: play/pause)",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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
                        if char_key in action_label_map:
                            if char_key in active_actions:
                                write_csv_row(actions_csv, [active_actions[char_key], current_idx, action_label_map[char_key]])
                                del active_actions[char_key]
                            else:
                                active_actions[char_key] = current_idx
                    except ValueError:
                        pass

            # Final writes on exit
            for action_key, start_frame in active_actions.items():
                write_csv_row(actions_csv, [start_frame, current_idx, action_label_map[action_key]])

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

    def get_object_dets(self, progress_callback=None):
        '''
        Get automotive object detection results from the VRS file
        '''
        classes = {
            0: 'Gear Stick',
            1: 'Left Wing Mirror',
            2: 'Rearview Mirror',
            3: 'Right Wing Mirror',
            4: 'Steering Wheel'
        }

        model_weight_path = '/Users/michaelrice/Documents/GitHub/Thesis/MSc_AI_Thesis/runs/detect/train/weights/best.pt'
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
                    if result[0].boxes.conf[j] > 0.4:
                        image_dets.append({
                            'class': classes[int(result[0].boxes.cls[j])],
                            'confidence': result[0].boxes.conf[j],
                            'bounding_box': result[0].boxes.xyxy[j]
                        })

            results.append(image_dets)

            # Call progress update
            if progress_callback:
                progress_callback(i + 1, len(rgb_values))

        self.result['object_detections'] = results

    def person_detection(self, image):
        '''
        Detect persons in the image using the mediapipe pose detection model
        '''

        model = YOLO("/Users/michaelrice/Documents/GitHub/Thesis/MSc_AI_Thesis/models/yolo11n.pt")
        person_confidences = []

        # Run inference
        results = model(image)[0]  # Get the first result

        person_indices = [results.boxes.cls == 0][0]
        
        
        for i,item in enumerate(person_indices):
            
            if item == True:
                person_confidences.append(results.boxes.conf[i].item())
            else:
                person_confidences.append(0.0)

        
        for i in range(len(person_confidences)):
            if person_confidences[i] < 0.4:
                person_indices[i] = False

        people_bboxes = results.boxes.xyxy[person_indices]

        cropped_images = []

        for bbox in people_bboxes:
            x1, y1, x2, y2 = map(int, bbox[:4])
            cropped_image = image[y1:y2, x1:x2]
            cropped_images.append(cropped_image)

        return cropped_images


        





        

        


    








        




    
        
    