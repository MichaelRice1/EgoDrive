import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import torch
import cv2
import csv
from time import time
import pandas as pd
import tqdm
import io
'''
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
'''
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
        self.face_ego_blur = "models/ego_blur_face.jit"
        self.lp_ego_blur = "models/ego_blur_lp.jit"
        self.mp_hand_landmarker_task_path = 'C:/Users/athen/Desktop/Github/MastersThesis/MSc_AI_Thesis/Coding/other/hand_landmarker.task'

    def get_device(self) -> str:
        """
        Return the device type
        """
        return (
            "cpu"
            if not torch.cuda.is_available()
            else f"cuda:{torch.cuda.current_device()}"
        )
    
    def get_image_data(self, start_index=0, end_index=None):

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


        for index in tqdm.tqdm(range(start_index, end_index)):
            buffer = io.BytesIO()
            image_data = self.provider.get_image_data_by_index(self.stream_mappings['camera-rgb'], index)

            # Convert to PIL Image, rotate, resize, and then convert back to numpy
            image_pil = Image.fromarray(image_data[0].to_numpy_array())  # Convert to PIL Image
            image_pil = image_pil.rotate(-90)  # Rotate counterclockwise 90 degrees
            image_pil = image_pil.resize((640, 640))  # Resize to 512x512

            # Save to buffer (optional, only if needed)
            # image_pil.save(buffer, format="PNG")

            # Convert back to NumPy array
            img = np.array(image_pil)  # Convert back to NumPy array

            rgb_images[rgb_ts[index]] = img

            # print(f' original bytes {img.nbytes}' )
            # image = Image.fromarray(img)
            # # Save the image to disk
            # image.save('sampledata/imagetesting/pngfull.png', format='PNG', compress_level=0)

            # pimg = Image.fromarray(img)
            # pimg.save(buffer, format="PNG")
            # png_bytes = buffer.getvalue()
            # print(f'rgb storage size {sys.getsizeof(png_bytes)}')
            # resized_img = pimg.resize((512, 512))
            # resized_img.save('sampledata/imagetesting/pngresized.png', format='PNG', compress_level=9)

            # break

        self.result['rgb'] = rgb_images
        print(f"Extracted {len(self.result['rgb'])} images from {rgblabel} stream")


        for index in range(start_index, end_index):
            image_data = self.provider.get_image_data_by_index(self.stream_mappings['camera-eyetracking'], index)
            img = np.array(Image.fromarray(image_data[0].to_numpy_array()))
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
    
    def preprocessing(self, image_data:list):
            
        '''
        Preprocess the extracted images from the VRS file
        '''

        rgb_images = self.result['rgb']
        pass

    def get_gaze_hand(self, gaze_path:str, hand_path:str, start_index=0, end_index=None):

        '''
        Extracts eyegaze points from the VRS file based on the index/time domain. The eyegaze points are extracted from the following streams:
        - camera-eyetracking
        '''

        gaze_points = {}
        hw_points = {}
        

        gaze_cpf = mps.read_eyegaze(gaze_path)
        handwrist_points  = mps.hand_tracking.read_wrist_and_palm_poses(hand_path)
        

        if start_index == 0 and end_index == None:
            num_et = len(gaze_cpf)
            num_hw = len(handwrist_points)
            et_ts = [gaze_cpf[i].tracking_timestamp.total_seconds() * 1e9 for i in range(num_et)]
            hw_ts = [handwrist_points[i].tracking_timestamp.total_seconds() * 1e9 for i in range(num_hw)]
        else:
            et_ts = [gaze_cpf[i].tracking_timestamp.total_seconds() * 1e9 for i in range(start_index,end_index)]
            #hw_ts = [handwrist_points[i].tracking_timestamp.total_seconds() * 1e9 for i in range(start_index,end_index)]

        print(f'length of eye tracking tiemstamps {len(et_ts)}')


        

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

    #TODO - stop this from loading objects
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
            right_data = {
                    "accel_msec2": imu_right_point.accel_msec2,
                    "accel_valid": imu_right_point.accel_valid,
                    "capture_timestamp_ns": imu_right_point.capture_timestamp_ns,
                    "gyro_radsec": imu_right_point.gyro_radsec,
                    "gyro_valid": imu_right_point.gyro_valid,
                    "mag_tesla": imu_right_point.mag_tesla,
                    "mag_valid": imu_right_point.mag_valid,
                    "temperature": imu_right_point.temperature,
                }

            if ind < num_data_imu_left:
                imu_left_point = self.provider.get_imu_data_by_index(self.stream_mappings['imu-left'], ind)
                left_data = {
                    "accel_msec2": imu_left_point.accel_msec2,
                    "accel_valid": imu_left_point.accel_valid,
                    "capture_timestamp_ns": imu_left_point.capture_timestamp_ns,
                    "gyro_radsec": imu_left_point.gyro_radsec,
                    "gyro_valid": imu_left_point.gyro_valid,
                    "mag_tesla": imu_left_point.mag_tesla,
                    "mag_valid": imu_left_point.mag_valid,
                    "temperature": imu_left_point.temperature,
                }

                imu_left[imu_left_ts[ind]] = left_data
                
            imu_right[imu_right_ts[ind]] = right_data
        
        
        self.result['imu_right'] = imu_right
        self.result['imu_left'] = imu_left

        print(f"Extracted {len(self.result['imu_right'])} data points from {imu_right_label} stream")
        print(f"Extracted {len(self.result['imu_left'])} data points from {imu_left_label} stream")


        '''
        Save the extracted data to the output path
        '''

        pass

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
        return frames

    #TODO
    def pc_filter(self, slam_path:str):
        '''
        Load the point cloud data from the VRS file
        '''

        global_points_path = os.path.join(slam_path, "semidense_points.csv.gz")

        points = mps.read_global_point_cloud(global_points_path)

        # filter the point cloud using thresholds on the inverse depth and distance standard deviation
        inverse_distance_std_threshold = 0.001
        distance_std_threshold = 0.15

        filtered_points = filter_points_from_confidence(points, inverse_distance_std_threshold, distance_std_threshold)

        point_cloud_points = []

        for point in filtered_points:
            distance_std = point.distance_std
            graph_uid = point.graph_uid
            inverse_distance_std = point.inverse_distance_std
            position_world = point.position_world
            uid = point.uid

            point = {
                'uid': uid,
                'graph_uid': graph_uid,
                'position_world': position_world,
                'inverse_distance_std': inverse_distance_std,
                'distance_std': distance_std
            }
            point_cloud_points.append(point)

        self.result['Point Cloud List'] = point_cloud_points





        # observations_path = "/path/to/mps/output/trajectory/semidense_observations.csv.gz"
        # observations = mps.read_point_observations(observations_path)

    def annotate(self, frames_dict, labels_csv, actions_csv, fps=15):
        '''
        Label RGB frames with face/license plate regions and driving actions.
        Spacebar plays/pauses video playback at specified FPS (default: 30).
        '''
        action_label_map = {
            '0': 'change gear',
            '1': 'turn steering wheel (two hands)',
            '2': 'turn steering wheel (one hand)',
            '3': 'toggling indicator',
            '4': 'checking left wing mirror',
            '5': 'checking right wing mirror',
            '6': 'checking rear view mirror',
            '7': 'checking left blind spot',
            '8': 'checking right blind spot',
            '9': 'adjusting wipers',
            'q': 'toggle headlights',
            'w': 'toggle hazard lights',
            'e': 'interacting with radio',
            'r': 'interacting with air conditioning',
            't': 'putting on seatbelt',
            'y': 'releasing hand brake',
            'u': 'deploying hand brake',
            'i': 'using mobile phone',
            'o': 'eating/drinking',
            'p': 'looking in vanity mirror',
            'a': 'checking speedometer / fuel gauge',
            's': 'checking navigation system',
            'd': 'idle'
        }

        # Key mappings (cross-platform)
        LEFT_KEY = 63234 if os.name == 'posix' else 2424832
        RIGHT_KEY = 63235 if os.name == 'posix' else 2555904
        ESC_KEY = 27
        EXIT_KEY = ord('x')  # Non-conflicting exit key
        FRAME_INTERVAL = 1.0 / fps

        # Initialize CSVs
        for csv_file, headers in [(labels_csv, ['start_frame', 'end_frame', 'type']), 
                                (actions_csv, ['start_frame', 'end_frame', 'action'])]:
            if not os.path.exists(csv_file):
                with open(csv_file, 'w', newline='') as f:
                    csv.writer(f).writerow(headers)

        sorted_ts = sorted(frames_dict.keys())
        gaze_points = list(self.result['gaze'].values())


        if not sorted_ts:
            print("No frames to label!")
            return

        current_idx = 0
        is_playing = False
        last_frame_time = time()

        # State tracking
        active_blur_labels = {'face': None, 'license_plate': None}
        active_actions = {}

        window_name = "Frame Labeler"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        def write_blur_label(label_type, start_frame, end_frame):
            with open(labels_csv, 'a', newline='') as f:
                csv.writer(f).writerow([start_frame, end_frame, label_type])

        def write_action(action_key, start_frame, end_frame):
            with open(actions_csv, 'a', newline='') as f:
                csv.writer(f).writerow([start_frame, end_frame, action_label_map[action_key]])

        try:
            while True:
                # Auto-advance playback
                if is_playing:
                    current_time = time()
                    if current_time - last_frame_time >= FRAME_INTERVAL:
                        new_idx = current_idx + 1
                        if new_idx < len(sorted_ts):
                            current_idx = new_idx
                            last_frame_time = current_time
                        else:
                            is_playing = False  # Stop at end

                ts = sorted_ts[current_idx]
                frame = frames_dict[ts].copy()

                projection = gaze_points[current_idx]['projection'] / (1408/640)
                depth = gaze_points[current_idx]['depth']

    
                # Display status
                status = []
                if is_playing: status.append(f"[PLAYING {fps}FPS]")
                for label_type, start_frame in active_blur_labels.items():
                    if start_frame is not None: status.append(f"{label_type.upper()}")
                for action_key in active_actions: status.append(action_label_map[action_key])
                
                cv2.putText(frame, f"ACTIVE: {', '.join(status) or 'None'}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Frame {current_idx+1}/{len(sorted_ts)} (X: exit, SPACE: play/pause)", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if projection is not None:
                    frame = cv2.circle(frame, (int(projection[0]), int(projection[1])), 6, (255, 0, 0), 3)

                cv2.imshow(window_name, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                key = cv2.waitKeyEx(1)

                # --- Key Handling ---
                if key in (ESC_KEY, EXIT_KEY):  # Exit
                    break

                # Navigation controls (override playback)
                if key == LEFT_KEY:
                    current_idx = max(0, current_idx - 1)
                    is_playing = False
                elif key == RIGHT_KEY:
                    current_idx = min(len(sorted_ts)-1, current_idx + 1)
                    is_playing = False

                # Play/pause toggle
                elif key == 32:  # Space
                    is_playing = not is_playing
                    last_frame_time = time()  # Reset timer for smooth playback

                # Blur labels
                elif key in (ord('f'), ord('F')):
                    label_type = 'face'
                    if active_blur_labels[label_type] is None:
                        active_blur_labels[label_type] = current_idx
                    else:
                        write_blur_label(label_type, active_blur_labels[label_type], current_idx)
                        active_blur_labels[label_type] = None
                
                elif key in (ord('l'), ord('L')):
                    label_type = 'license_plate'
                    if active_blur_labels[label_type] is None:
                        active_blur_labels[label_type] = current_idx
                    else:
                        write_blur_label(label_type, active_blur_labels[label_type], current_idx)
                        active_blur_labels[label_type] = None

                # Action keys
                else:
                    try:
                        char_key = chr(key).lower()
                        if char_key in action_label_map:
                            action_key = char_key
                            if action_key in active_actions:
                                write_action(action_key, active_actions[action_key], current_idx)
                                del active_actions[action_key]
                            else:
                                active_actions[action_key] = current_idx
                    except ValueError:
                        pass

        finally:
            # Final writes on exit
            for label_type, start_frame in active_blur_labels.items():
                if start_frame is not None:
                    write_blur_label(label_type, start_frame, current_idx)
            for action_key, start_frame in active_actions.items():
                write_action(action_key, start_frame, current_idx)
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

    def get_object_dets(self):
        '''
        Get automotive object detection results from the VRS file
        '''

        classes = ['left hand','right hand','gear stick','steering wheel',
                   'stalk','left wing mirror','right wing mirror',
                   'rear view mirror','infotainment centre','hand brake', 
                   'mobile phone', 'dashboard', 'vanity mirror','navigation system']
        

        for image in self.result['rgb'].values():
            pass
            


    








        




    
        
    