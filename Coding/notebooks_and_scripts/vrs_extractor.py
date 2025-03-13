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
    get_nearest_eye_gaze
)

class VRSDataExtractor():


    def __init__(self, vrs_path):
        self.path = vrs_path
        self.provider = data_provider.create_vrs_data_provider(self.path)

        self.stream_mappings = {
            "camera-slam-left": StreamId("1201-1"),
            "camera-slam-right":StreamId("1201-2"),
            "camera-rgb":StreamId("214-1"),
            "camera-eyetracking":StreamId("211-1"),
            "microphone":StreamId("231-1"),
            "gps":StreamId("281-1"),
            "gps-app":StreamId("281-2"),
            "imu-right":StreamId("1202-1"),
            "imu-left":StreamId("1202-2")
        }

        self.time_domain = TimeDomain.DEVICE_TIME  # query data based on host time
        self.option = TimeQueryOptions.CLOSEST # get data whose time [in TimeDomain] is CLOSEST to query time
        self.rgb_start_time = self.provider.get_first_time_ns(self.stream_mappings['camera-rgb'], self.time_domain)
        self.result = {}



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
    


    def get_mps(self, gaze_path, hand_path, start_index = 0, end_index = None):

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



    

if __name__ == "__main__":
    VRS_DE = VRSDataExtractor('C:/Users/athen/Desktop/Github/MastersThesis/sampledata/sample2/Testing_MPS.vrs')

    VRS_DE.get_image_data()

    print(f' sample image entry {list(VRS_DE.result["rgb"].items())[0]}')

    VRS_DE.get_mps('C:/Users/athen/Desktop/Github/MastersThesis/sampledata/sample3/mps_Driving_Profile_Test_vrs/eye_gaze/general_eye_gaze.csv', 
                   'C:/Users/athen/Desktop/Github/MastersThesis/sampledata/sample3/mps_Driving_Profile_Test_vrs/hand_tracking/wrist_and_palm_poses.csv')
    
    print(f' sample gaze entry {list(VRS_DE.result["gaze"].items())[0]}')
    print(f' sample handwrist entry {list(VRS_DE.result["handwrist"].items())[0]}')







        




    
        
    