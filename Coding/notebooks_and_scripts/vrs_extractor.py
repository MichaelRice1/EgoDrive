import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import torch
import cv2
from moviepy.editor import ImageSequenceClip
import csv
from time import time
import pandas as pd
import tqdm
import io
from projectaria_tools.core import data_provider
from projectaria_tools.core.image import InterpolationMethod
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId
import projectaria_tools.core.mps as mps
from projectaria_tools.core.mps.utils import (
    get_nearest_wrist_and_palm_pose,
    get_gaze_vector_reprojection,
    get_nearest_eye_gaze,
    get_nearest_pose,
    filter_points_from_confidence
)
from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    distort_by_calibration,
    get_linear_camera_calibration,
)


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
        self.face_ego_blur = "MSc_AI_Thesis/Coding/notebooks_and_scripts/OtherModelScripts/EgoBlurModels/ego_blur_face/ego_blur_face.jit"
        self.lp_ego_blur = "MSc_AI_Thesis/Coding/notebooks_and_scripts/OtherModelScripts/EgoBlurModels/ego_blur_lp/ego_blur_lp.jit"

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
            img = np.array(Image.fromarray(image_data[0].to_numpy_array()).rotate(-90))
            rgb_images[rgb_ts[index]] = img

        self.result['rgb'] = rgb_images
        print(f"Extracted {len(self.result['rgb'])} images from {rgblabel} stream")


        for index in range(start_index, end_index):
            image_data = self.provider.get_image_data_by_index(self.stream_mappings['camera-eyetracking'], index)
            img = np.array(Image.fromarray(image_data[0].to_numpy_array()))
            et_images[et_ts[index]] = img
             
        self.result['et'] = et_images
        print(f"Extracted {len(self.result['et'])} images from {etlabel} stream")

        try:
            for index in range(start_index, end_index):
                left_image_data = self.provider.get_image_data_by_index(self.stream_mappings['camera-slam-left'], index)
                right_image_data = self.provider.get_image_data_by_index(self.stream_mappings['camera-slam-right'], index)
                left_img = np.array(Image.fromarray(left_image_data[0].to_numpy_array()))
                right_img = np.array(Image.fromarray(right_image_data[0].to_numpy_array()))
                slam_left_images[slam_left_ts[index]] = left_img
                slam_right_images[slam_right_ts[index]] = right_img

            self.result['slam_left'] = slam_left_images
            self.result['slam_right'] = slam_right_images

            print(f"Extracted {len(self.result['slam_left'])} images from {left_slam_label} stream")
            print(f"Extracted {len(self.result['slam_right'])} images from {right_slam_label} stream")
        except:
            print("Error extracting slam images, likely not present in VRS file ")
    
    def preprocessing(self, image_data):
            
        '''
        Preprocess the extracted images from the VRS file
        '''

        rgb_images = self.result['rgb']
        pass

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


        '''
        Save the extracted data to the output path
        '''

        pass

    def ego_blur( self, input_labels):

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
        indices = data['timestep']
        labels = data['label']
        timestamps = list(self.result['rgb'].keys())
        filtered_ts = [timestamps[index] for index in indices]
        
        frames = [self.result['rgb'][fts] for fts in filtered_ts]

        print(f' Number of frames {len(frames)}')


        for i,frame in tqdm.tqdm(enumerate(frames)):

            label = labels[i]
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
            self.result['rgb'][filtered_ts[i]] = rgb

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
                        samp, dst_calib, rgb_calib,
                        InterpolationMethod.BILINEAR
                    )
        
        plt.imshow(undistorted_rgb_image)
        plt.show()
        
        #interpolation from InterpolationMethod

    def video_from_frames(self,rgb_frames,output_path,fps = 15):
            
        '''
        Create a video from the extracted frames and turn into moviepy VideoClip
        '''
        clip = ImageSequenceClip(rgb_frames, fps=fps)
        clip.write_videofile(f"{output_path}", codec="libx265")


        return clip

    def point_cloud_loading_and_filtering(self, base_path):
        '''
        Load the point cloud data from the VRS file
        '''

        global_points_path = base_path + '/slam_data/global_point_cloud.csv.gz'

        points = mps.read_global_point_cloud(global_points_path)

        # filter the point cloud using thresholds on the inverse depth and distance standard deviation
        inverse_distance_std_threshold = 0.001
        distance_std_threshold = 0.15

        filtered_points = filter_points_from_confidence(points, inverse_distance_std_threshold, distance_std_threshold)



        for point in filtered_points:
            position_world = point.position_world
            position_device = point.uid

            break

        # observations_path = "/path/to/mps/output/trajectory/semidense_observations.csv.gz"
        # observations = mps.read_point_observations(observations_path)

    def label_face_lps(self, frames_dict, output_csv):

        '''
        function to label the rgb frames if they have faces or license plates to save blurring time
        '''

        LEFT_KEY = 81 if os.name == 'posix' else 2424832
        RIGHT_KEY = 83 if os.name == 'posix' else 2555904
        ESC_KEY = 27

        # Initialize CSV and state
        csv_exists = os.path.exists(output_csv)
        with open(output_csv, 'a' if csv_exists else 'w', newline='') as f:
            writer = csv.writer(f)
            if not csv_exists:
                writer.writerow(['timestep', 'label'])

        sorted_ts = sorted(frames_dict.keys())
        if not sorted_ts:
            print("No frames to label!")
            return

        current_idx = 0
        is_playing = False
        label_face = False
        label_lp = False
        last_frame_time = time()
        window_name = "Frame Labeler"

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            ts = sorted_ts[current_idx]
            frame = frames_dict[ts].copy()

            # Add status overlay
            status = []
            if label_face: status.append("FACE")
            if label_lp: status.append("LICENSE")
            cv2.putText(frame, f"TOGGLES: {', '.join(status) or 'None'}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame {current_idx+1}/{len(sorted_ts)}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(window_name, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            key = cv2.waitKeyEx(1)

            # Handle exit
            if key in (ESC_KEY, ord('q'), ord('Q')):
                break

            # Toggle labels
            if key in (ord('f'), ord('F')):
                label_face = not label_face
            if key in (ord('l'), ord('L')):
                label_lp = not label_lp

            # Navigation controls
            if key == LEFT_KEY:
                current_idx = max(0, current_idx - 1)
                is_playing = False
            elif key == RIGHT_KEY:
                self._handle_frame_advance(sorted_ts, current_idx, 1, output_csv, label_face, label_lp)
                current_idx = min(len(sorted_ts)-1, current_idx + 1)
                is_playing = False
            elif key == 32:  # Space
                is_playing = not is_playing

            # Auto-advance with labeling
            if is_playing and time() - last_frame_time >= 1/15:
                new_idx = current_idx + 1
                if new_idx < len(sorted_ts):
                    self._handle_frame_advance(sorted_ts, current_idx, new_idx-current_idx, 
                                            output_csv, label_face, label_lp)
                    current_idx = new_idx
                    last_frame_time = time()
                else:
                    is_playing = False

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

    def save_data(self,output_path):
        '''
        Save the extracted data to the output path
        '''

        # np.save(output_path, self.result)


        pass




if __name__ == "__main__":
    
    base_path = 'sampledata/mps_Lab_Test_vrs/Lab_Test.vrs'
    file_name = base_path.split("/")[-1].split(".")[0]
    gaze_path = "/".join(base_path.split("/")[:-1]) + '/eye_gaze/general_eye_gaze.csv'
    hand_path = "/".join(base_path.split("/")[:-1]) + '/hand_tracking/wrist_and_palm_poses.csv'
    slam_path = "/".join(base_path.split("/")[:-1]) + '/slam_data'
    video_path = "/".join(base_path.split("/")[:-1]) + f'/{file_name}_video.mp4'
    # ego_blur_labels_path = "/".join(base_path.split("/")[:-1]) + f'/{file_name}_labels.csv'

    ego_blur_labels_path = 'sampledata/imagetesting/Lab_test_labels.csv'
    output_path = 'sampledata/imagetesting/Lab_Test_data.npy'




    VRS_DE = VRSDataExtractor(base_path)

    VRS_DE.get_image_data()

    # VRS_DE.label_face_lps(VRS_DE.result['rgb'], 'sampledata/imagetesting/Lab_test_labels.csv')

    VRS_DE.ego_blur(ego_blur_labels_path)
    # VRS_DE.save_data(output_path)
    # VRS_DE.get_IMU_data()
    # VRS_DE.get_gaze_hand(gaze_path, hand_path)
    # VRS_DE.get_slam_data(slam_path)
    #VRS_DE.rgb_undistort('C:/Users/athen/Desktop/Github/MastersThesis/sampledata/imagetesting/facetest.jpg')
    # VRS_DE.video_from_frames(list(VRS_DE.result['rgb'].values()), video_path)

    

    # print(f' sample imu-right entry {list(VRS_DE.result["imu_right"].items())[0][1]}')
    # print(f' sample imu-left entry {list(VRS_DE.result["imu_left"].items())[0][1]}')
    # print(f' sample gaze entry {list(VRS_DE.result["gaze"].items())[0]}')
    # print(f' sample handwrist entry {list(VRS_DE.result["handwrist"].items())[0]}')
    # print(f' sample open loop entry {list(VRS_DE.result["open_loop"].items())[0]}')







        




    
        
    