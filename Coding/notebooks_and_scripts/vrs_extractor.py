import sys
import os
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.image import InterpolationMethod
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import RecordableTypeId, StreamId
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import projectaria_tools.core.mps as mps
from projectaria_tools.core.mps.utils import get_nearest_wrist_and_palm_pose
from projectaria_tools.core.mps.utils import (
    get_gaze_vector_reprojection,
    get_nearest_eye_gaze,
    get_nearest_pose
)
import argparse
import torch
from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    distort_by_calibration,
    get_linear_camera_calibration,
)
import cv2

import sys
sys.path.append('C:/Users/athen/Desktop/Github/MastersThesis/MSc_AI_Thesis/Coding/notebooks_and_scripts')
from OtherModelScripts import ego_blur 


class VRSDataExtractor():


    def __init__(self, vrs_path: str):
        self.path = vrs_path
        self.provider = data_provider.create_vrs_data_provider(self.path)

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

        self.time_domain = TimeDomain.DEVICE_TIME  # query data based on host time
        self.option = TimeQueryOptions.CLOSEST # get data whose time [in TimeDomain] is CLOSEST to query time
        self.rgb_start_time = self.provider.get_first_time_ns(self.stream_mappings['camera-rgb'], self.time_domain)
        self.result = {}

    def get_device(self) -> str:
        """
        Return the device type
        """
        return (
            "cpu"
            if not torch.cuda.is_available()
            else f"cuda:{torch.cuda.current_device()}"
        )
    

    def get_image_data(self, start_index = 0, end_index = None):

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


        for index in range(start_index, end_index):
            image_data = self.provider.get_image_data_by_index(self.stream_mappings['camera-rgb'], index)
            img = Image.fromarray(image_data[0].to_numpy_array()).rotate(-90)
            rgb_images[rgb_ts[index]] = img

        self.result['rgb'] = rgb_images
        print(f"Extracted {len(self.result['rgb'])} images from {rgblabel} stream")


        for index in range(start_index, end_index):
            image_data = self.provider.get_image_data_by_index(self.stream_mappings['camera-eyetracking'], index)
            img = Image.fromarray(image_data[0].to_numpy_array())
            et_images[et_ts[index]] = img
             
        self.result['et'] = et_images
        print(f"Extracted {len(self.result['et'])} images from {etlabel} stream")

        try:
            for index in range(start_index, end_index):
                left_image_data = self.provider.get_image_data_by_index(self.stream_mappings['camera-slam-left'], index)
                right_image_data = self.provider.get_image_data_by_index(self.stream_mappings['camera-slam-right'], index)
                left_img = Image.fromarray(left_image_data[0].to_numpy_array())
                right_img = Image.fromarray(right_image_data[0].to_numpy_array())
                slam_left_images[slam_left_ts[index]] = left_img
                slam_right_images[slam_right_ts[index]] = right_img

            self.result['slam_left'] = slam_left_images
            self.result['slam_right'] = slam_right_images

            print(f"Extracted {len(self.result['slam_left'])} images from {left_slam_label} stream")
            print(f"Extracted {len(self.result['slam_right'])} images from {right_slam_label} stream")
        except:
            print("Error extracting slam images, likely not present in VRS file ")
    

    def get_gaze_hand(self, gaze_path, hand_path, start_index = 0, end_index = None):

        '''
        Extracts eyegaze points from the VRS file based on the index/time domain. The eyegaze points are extracted from the following streams:
        - camera-eyetracking
        '''

        gaze_points = {}
        hw_points = {}
        

        # et_start_time = self.provider.get_first_time_ns(self.stream_mappings['camera-eyetracking'], self.time_domain)
        # et_end_time = self.provider.get_last_time_ns(self.stream_mappings['camera-eyetracking'], self.time_domain)


        gaze_cpf = mps.read_eyegaze(gaze_path)
        handwrist_points  = mps.hand_tracking.read_wrist_and_palm_poses(hand_path)
        

        

        num_et = len(gaze_cpf)
        num_hw = len(handwrist_points)

        et_ts = [gaze_cpf[i].tracking_timestamp.total_seconds() * 1e9 for i in range(num_et)]
        hw_ts = [handwrist_points[i].tracking_timestamp.total_seconds() * 1e9 for i in range(num_hw)]


        for ts in hw_ts:
    
            wrist_and_palm_pose = get_nearest_wrist_and_palm_pose(handwrist_points, ts)

            if wrist_and_palm_pose is not None:
                
                left_pose_confidence = wrist_and_palm_pose.left_hand.confidence
                left_wrist_position_device = wrist_and_palm_pose.left_hand.wrist_position_device
                left_palm_position_device = wrist_and_palm_pose.left_hand.palm_position_device
                left_wrist_normal_device = wrist_and_palm_pose.left_hand.wrist_and_palm_normal_device.wrist_normal_device
                left_palm_normal_device = wrist_and_palm_pose.left_hand.wrist_and_palm_normal_device.palm_normal_device

                right_pose_confidence = wrist_and_palm_pose.right_hand.confidence
                right_wrist_position_device = wrist_and_palm_pose.right_hand.wrist_position_device
                right_palm_position_device = wrist_and_palm_pose.right_hand.palm_position_device
                right_wrist_normal_device = wrist_and_palm_pose.right_hand.wrist_and_palm_normal_device.wrist_normal_device
                right_palm_normal_device = wrist_and_palm_pose.right_hand.wrist_and_palm_normal_device.palm_normal_device

                hw_points[ts] = {
                    "left_pose_confidence": left_pose_confidence,
                    "left_wrist_position_device": left_wrist_position_device,
                    "left_palm_position_device": left_palm_position_device,
                    "left_wrist_normal_device": left_wrist_normal_device,
                    "left_palm_normal_device": left_palm_normal_device,
                    "right_pose_confidence": right_pose_confidence,
                    "right_wrist_position_device": right_wrist_position_device,
                    "right_palm_position_device": right_palm_position_device,
                    "right_wrist_normal_device": right_wrist_normal_device,
                    "right_palm_normal_device": right_palm_normal_device
                }
        
        self.result['handwrist'] = hw_points
        print(f"Extracted {len(self.result['handwrist'])} handwrist points from handwrist stream")

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
        print(f"Extracted {len(self.result['gaze'])} gaze points from gaze stream")


    def get_slam_data(self, slam_path, start_index = 0, end_index = None):

        open_loop_points = {}
        closed_loop_points = {}

        open_loop_path = os.path.join(slam_path, "open_loop_trajectory.csv")
        closed_loop_path = os.path.join(slam_path, "closed_loop_trajectory.csv")

        open_loop_traj = mps.read_open_loop_trajectory(open_loop_path)
        closed_loop_traj = mps.read_closed_loop_trajectory(closed_loop_path)

        open_loop_ts = [open_loop_traj[i].tracking_timestamp.total_seconds() * 1e9 for i in range(len(open_loop_traj))]
        closed_loop_ts = [closed_loop_traj[i].tracking_timestamp.total_seconds() * 1e9 for i in range(len(closed_loop_traj))]

        for ts in open_loop_ts:
            open_loop_point = get_nearest_pose(open_loop_traj, ts)

            if open_loop_point is not None:

                # print(f'open loop point',open_loop_point)

                T_world_device = open_loop_point.transform_odometry_device
                
                rgb_stream_label = self.provider.get_label_from_stream_id(self.stream_mappings['camera-rgb'])
                device_calibration = self.provider.get_device_calibration()
                rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)

                T_device_rgb_camera = rgb_camera_calibration.get_transform_device_camera()
                T_world_rgb_camera = T_world_device @ T_device_rgb_camera

                open_loop_point = {
                    "tracking_timestamp": open_loop_point.tracking_timestamp,
                    "quality_score": open_loop_point.quality_score,
                    "linear_velocity": open_loop_point.device_linear_velocity_odometry,
                    "angular_velocity_device": open_loop_point.angular_velocity_device,
                    "t_world_device": T_world_device,
                    "gravity": open_loop_point.gravity_odometry,
                    "world_rgb_camera_translation": T_world_rgb_camera,
                    "device_rgb_camera_rotation": T_device_rgb_camera
                }

                open_loop_points[ts] = open_loop_point
            
        self.result['open_loop'] = open_loop_points

        print(f"Extracted {len(self.result['open_loop'])} open loop points from open loop stream")

        for ts in closed_loop_ts:

            closed_loop_point = get_nearest_pose(closed_loop_traj, ts)

            if closed_loop_point is not None:

                T_world_device = closed_loop_point.transform_world_device
                
                rgb_stream_label = self.provider.get_label_from_stream_id(self.stream_mappings['camera-rgb'])
                device_calibration = self.provider.get_device_calibration()
                rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)

                T_device_rgb_camera = rgb_camera_calibration.get_transform_device_camera()
                T_world_rgb_camera = T_world_device @ T_device_rgb_camera

                closed_loop_point = {
                    "tracking_timestamp": closed_loop_point.tracking_timestamp,
                    "quality_score": closed_loop_point.quality_score,
                    "linear_velocity": closed_loop_point.device_linear_velocity_device,
                    "angular_velocity_device": closed_loop_point.angular_velocity_device,
                    "translation_world": T_world_device,
                    "gravity": closed_loop_point.gravity_world,
                    "world_rgb_camera_translation": T_world_rgb_camera,
                    "device_rgb_camera_rotation": T_device_rgb_camera
                }

                closed_loop_points[ts] = closed_loop_point

        self.result['closed_loop'] = closed_loop_points
        print(f"Extracted {len(self.result['closed_loop'])} closed loop points from closed loop stream")

    #TODO - stop this from loading objects
    def get_IMU_data(self, start_index = 0, end_index = None):

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

            if ind < num_data_imu_left:
                imu_left_point = self.provider.get_imu_data_by_index(self.stream_mappings['imu-left'], ind)
                imu_left[imu_left_ts[ind]] = imu_left_point
                
            imu_right[imu_right_ts[ind]] = imu_right_point
        
        self.result['imu_right'] = imu_right
        self.result['imu_left'] = imu_left

        print(f"Extracted {len(self.result['imu_right'])} data points from {imu_right_label} stream")
        print(f"Extracted {len(self.result['imu_left'])} data points from {imu_left_label} stream")

    def save_data(self, results, output_path):

        '''
        Save the extracted data to the output path
        '''

        pass

    def ego_blur( self,
                egoblur_face_path: str = "MSc_AI_Thesis/Coding/notebooks_and_scripts/OtherModelScripts/EgoBlurModels/ego_blur_face/ego_blur_face.jit",
                egoblur_lp_path: str = "MSc_AI_Thesis/Coding/notebooks_and_scripts/OtherModelScripts/EgoBlurModels/ego_blur_lp/ego_blur_lp.jit",
                input_image_path: str | None = None,
                output_image_path: str | None = None,
                input_video_path: str | None = None,
                output_video_path: str | None = None,
            ) -> None:

        '''
        Apply ego-blur to the extracted images from the VRS file to make data anonymous
        '''


        face_model_score_threshold = 0.9
        lp_model_score_threshold = 0.9
        nms_iou_threshold = 0.3
        scale_factor_detections = 1.0
        output_video_fps = 15


        face_detector = torch.jit.load(egoblur_face_path, map_location="cpu").to(
            self.get_device()
        )
        face_detector.eval()

        lp_detector = torch.jit.load(egoblur_lp_path, map_location="cpu").to(
            self.get_device()
        )
        lp_detector.eval()

        if input_image_path is not None and output_image_path is not None:

            output_image = ego_blur.visualize_image(
                input_image_path,
                face_detector,
                lp_detector,
                face_model_score_threshold,
                lp_model_score_threshold,
                nms_iou_threshold,
                output_image_path,
                scale_factor_detections,
            )

        if input_video_path is not None and output_video_path is not None:
        
            ego_blur.visualize_video(
                input_video_path,
                face_detector,
                lp_detector,
                face_model_score_threshold,
                lp_model_score_threshold,
                nms_iou_threshold,
                output_video_path,
                scale_factor_detections,
                output_video_fps,
            )

    #TODO - unsure if will work
    def rgb_undistort(self, path):

        '''
        Undistort the extracted images from the VRS file
        '''

        # rgb_images = results['rgb']



        samp = cv2.imread(path)

        with open('MSc_AI_Thesis/Coding/other/calibration.json') as f:
            sc = f.read()


        sensors_calib = device_calibration_from_json_string(sc)
        rgb_calib = sensors_calib.get_camera_calib("camera-rgb")
        dst_calib = get_linear_camera_calibration(512, 512, 150, "camera-rgb")

        # for img in rgb_images:

        undistorted_rgb_image = distort_by_calibration(
                        samp, dst_calib, rgb_calib
                    )
        
        plt.imshow(np.array(undistorted_rgb_image))
        plt.show()

        


        


if __name__ == "__main__":
    VRS_DE = VRSDataExtractor('sampledata/sample3/Driving_Profile_Test.vrs')


    # VRS_DE.get_image_data()

    # print(f' sample image entry {list(VRS_DE.result["rgb"].items())[0]}')

    # VRS_DE.get_IMU_data()

    # print(f' sample imu-right entry {list(VRS_DE.result["imu_right"].items())[0][1]}')
    # print(f' sample imu-left entry {list(VRS_DE.result["imu_left"].items())[0][1]}')

    # VRS_DE.get_gaze_hand('C:/Users/athen/Desktop/Github/MastersThesis/sampledata/sample3/mps_Driving_Profile_Test_vrs/eye_gaze/general_eye_gaze.csv', 
    #                'C:/Users/athen/Desktop/Github/MastersThesis/sampledata/sample3/mps_Driving_Profile_Test_vrs/hand_tracking/wrist_and_palm_poses.csv',)
    
    # print(f' sample gaze entry {list(VRS_DE.result["gaze"].items())[0]}')
    # print(f' sample handwrist entry {list(VRS_DE.result["handwrist"].items())[0]}')

    # print(f' sample open loop entry {list(VRS_DE.result["open_loop"].items())[0]}')

    # VRS_DE.ego_blur(input_image_path = 'C:/Users/athen/Desktop/Github/MastersThesis/sampledata/imagetesting/facetest.jpg',
    #                 output_image_path = 'C:/Users/athen/Desktop/Github/MastersThesis/sampledata/imagetesting/facetest_blurred.jpg')

    # VRS_DE.rgb_undistort('C:/Users/athen/Desktop/Github/MastersThesis/sampledata/imagetesting/facetest.jpg')







        




    
        
    