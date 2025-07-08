import numpy as np
import cv2
from torch.utils.data import Dataset
import os 
import pandas as pd 
import matplotlib.pyplot as plt

class EgoDriveAriaDataset():

    def __init__(self, data, image_size, frames_per_clip, annotations_path=None):
        
        
        self.data = data
        self.image_size = image_size
        self.frames_per_clip = frames_per_clip
        self.annotations_path = annotations_path


    def processed_annotations(self, len_frames ,annotations):
        expanded = pd.DataFrame(columns=['start_frame', 'end_frame', 'action'])

        for _, row in annotations.iterrows():
            start_frame = row.iloc[0]
            end_frame = row.iloc[1]
            action = row.iloc[2]
            len = end_frame - start_frame + 1
            rem = self.frames_per_clip - len

            if rem < 0:
                num_clips = len // self.frames_per_clip
                for i in range(num_clips):
                    clip_start = start_frame + i * self.frames_per_clip
                    clip_end = clip_start + self.frames_per_clip - 1
                    if clip_end >= len_frames:
                        break
                    expanded = pd.concat([expanded, pd.DataFrame([{
                        'start_frame': clip_start,
                        'end_frame': clip_end,
                        'action': action
                    }])], ignore_index=True)
            else:
                start_frame = max(0, start_frame - rem // 2)
                end_frame = min((start_frame + self.frames_per_clip - 1 ), len_frames - 1)

                if end_frame - start_frame + 1 < self.frames_per_clip:
                    continue
                
                expanded = pd.concat([expanded, pd.DataFrame([{
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'action': action
                }])], ignore_index=True)
            

        expanded['start_frame'] = expanded['start_frame'].astype(int)
        expanded['end_frame'] = expanded['end_frame'].astype(int)
        expanded['action'] = expanded['action'].astype(str)

        # Remove overlapping sections with actions other than 'driving'
        for i, row in expanded.iterrows():
            overlapping = expanded[
            (expanded['start_frame'] <= row['end_frame']) &
            (expanded['end_frame'] >= row['start_frame']) &
            (expanded['action'] != 'driving') &
            (expanded.index != i)
            ]
            if not overlapping.empty:
                expanded.drop(index=i, inplace=True)

        return expanded 
    
    def evaluate_data_point(self, data, save_path):
        frames = data['frames']
        gaze = data['gaze']
        imu_data = data['imu']
        hands = data['hands']
        object_detections = data['object_detections']

        concat = np.concatenate(imu_data, axis=0)  # Concatenate all IMU data
        imu_data = [i[0:6] for i in concat]  # Extract first 6 channels

        imu_accel_x = [sample[0] for sample in imu_data]
        imu_accel_y = [sample[1] for sample in imu_data]
        imu_accel_z = [sample[2] for sample in imu_data]

        imu_gyro_x = [sample[3] for sample in imu_data]
        imu_gyro_y = [sample[4] for sample in imu_data]
        imu_gyro_z = [sample[5] for sample in imu_data]

        # Save acceleration plot
        plt.figure(figsize=(10, 5))
        plt.plot(imu_accel_x, label='Accel X', color='r')
        plt.plot(imu_accel_y, label='Accel Y', color='g')
        plt.plot(imu_accel_z, label='Accel Z', color='b')
        plt.title('IMU Acceleration')
        plt.xlabel('Sample Index')
        plt.ylabel('Acceleration (m/s²)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{save_path}/imu_acceleration_plot.png')  # Save the plot
        plt.close()  # Close the figure to free memory

        # Save gyroscope plot
        plt.figure(figsize=(10, 5))
        plt.plot(imu_gyro_x, label='Gyro X', linestyle='--', color='r')
        plt.plot(imu_gyro_y, label='Gyro Y', linestyle='--', color='g')
        plt.plot(imu_gyro_z, label='Gyro Z', linestyle='--', color='b')
        plt.title('IMU Gyroscope')
        plt.xlabel('Sample Index')
        plt.ylabel('Angular Velocity (rad/s)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{save_path}/imu_gyroscope_plot.png')  # Save the plot
        plt.close()  # Close the figure to free memory
        

        # Save annotated frames as a video
        output_video_path = f'{save_path}/annotated_video.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 15  # Frames per second
        frame_size = (self.image_size, self.image_size)

        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

        for f, g, h, o in zip(frames, gaze, hands, object_detections):
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)

            gaze_x, gaze_y = g

            gaze_f = cv2.circle(rgb.copy(),
                                (int(gaze_x * self.image_size), int(gaze_y * self.image_size)),
                                radius=5, color=(255, 0, 0), thickness=-1)

            hands_f = gaze_f.copy()
            for i in range(0, len(h), 2):
                if h[i] is not np.nan and h[i + 1] is not np.nan:
                    hands_f = cv2.circle(hands_f,
                                        (int(h[i] * self.image_size), int(h[i + 1] * self.image_size)),
                                        radius=5, color=(0, 255, 0), thickness=-1)


            for i in range(0, len(o), 5):
                if o[i] != 0:
                    x = int(o[i+1] * self.image_size)
                    y = int(o[i + 2] * self.image_size)
                    width = int(o[i + 3] * self.image_size)
                    height = int(o[i + 4] * self.image_size)

                    # Draw bounding box
                    cv2.rectangle(hands_f,
                                (x, y),
                                (x + width, y + height),
                                color=(0, 0, 255), thickness=2)

                    # Draw gaze intersection point
                    # if o[i + 5] == 1.0:
                    #     cv2.circle(hands_f,
                    #             (x + width // 2, y + height // 2),
                    #             radius=5, color=(255, 255, 0), thickness=-1)

            video_writer.write(hands_f)

        video_writer.release()

    def process_object_detections_with_gaze(self, detections, gaze_point):
        """
        Enhanced object features including gaze-relative information
        Features per object: [presence, x, y, width, height, gaze_intersects, gaze_distance]
        """
        target_objects = ['Right Wing Mirror', 'Left Wing Mirror', 'Rearview Mirror', 
                        'Mobile Phone']
        features = np.zeros(20)  # 4 objects × 5 features each
        
        if gaze_point is not None and len(gaze_point) >= 2:
            gx, gy = gaze_point[0], gaze_point[1]
            self.last_valid_gaze = gaze_point

        elif gaze_point is None or len(gaze_point) < 2:
            gx, gy = self.last_valid_gaze[0], self.last_valid_gaze[1]
        
        
        for i, obj_class in enumerate(target_objects):
            start_idx = i * 5

            classes = [d['class'] for d in detections if 'class' in d]

            if obj_class in classes:
                
                entry = next((d for d in detections if d.get('class') == obj_class), None)
                # Standard object features
                bbox = entry.get('bounding_box', {})
                x = bbox[0]
                y = bbox[1]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]

                # Normalize coordinates
                x = x / 512
                y = y / 512
                width = width / 512
                height = height / 512

                det = [x, y, width, height]


                features[start_idx + 0] = 1.0  
                features[start_idx + 1] = x
                features[start_idx + 2] = y
                features[start_idx + 3] = width
                features[start_idx + 4] = height

                
                # Gaze-object interaction features
                # features[start_idx + 5] = 1.0 if self.is_gaze_in_bbox(gx, gy, det) else 0.0
                
                
        return features

    def is_gaze_in_bbox(self,gx, gy, detection):
        """Check if normalized gaze point is within detection bounding box"""
        # Convert detection bbox to normalized coordinates
        x = detection[0]
        y = detection[1]
        width = detection[2]
        height = detection[3]
        
        # Check if gaze point is within bounding box
        return (x <= gx <= x + width) and (y <= gy <= y + height)

    def process_folder(self, data, annotations_path):
        if not os.path.exists(annotations_path):
            print(f"Annotations {annotations_path} does not exist. Skipping.")
            return

        annotations = pd.read_csv(annotations_path)

        frames = data['modalities']['rgb']
        num_frames = len(frames)

        expanded_annotations = self.processed_annotations(num_frames, annotations)
        samples = []

        label_map = {
            'checking left wing mirror': 0,
            'checking rear view mirror': 1,
            'checking right wing mirror': 2,
            'driving': 3,
            'idle': 4,
            'mobile phone usage': 5
        }

        for start, end, action in zip(expanded_annotations['start_frame'], 
                                    expanded_annotations['end_frame'], 
                                    expanded_annotations['action']):
            start = int(start)
            end = int(end)
            action = str(action)

            frames_processed = []
            gaze_processed = []
            hands_processed = []
            imu_processed = []
            hands_processed = []
            object_detections_processed = []

            frames_segment = frames[start:end+1]
            gaze_segment = data['modalities']['gaze'][start:end+1]
            imu_segment = data['modalities']['imu_right'][start:end+1]
            hand_landmarks_segment = data['modalities']['hand_landmarks'][start:end+1]
            object_detections_segment = data['modalities']['object_detections'][start:end+1]


            last_valid_gaze = None  # Initialize outside your loop

            # Resize images and normalize gaze coordinates
            for f, g, h, o, i in zip(frames_segment, gaze_segment, hand_landmarks_segment, object_detections_segment, imu_segment):
                # Resize image

                img_rs = cv2.resize(f, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
                frames_processed.append(img_rs)

                # Process IMU data
                    
                imu_processed.append(i)

                if g is not None and len(g) > 0:
                    projs = []
                    for gaze_point in g:
                        if gaze_point is not None and 'projection' in gaze_point:
                            projs.append(gaze_point['projection'])

                    if len(projs) == 2:
                        projs = [proj for proj in projs if np.shape(proj) == (2,)]
                        if len(projs) >= 1:
                            mean_g = np.mean(projs, axis=0)
                            if mean_g.shape == (2,):
                                mean_g = mean_g * (self.image_size / 512)
                                norm_gx, norm_gy = mean_g[0] / self.image_size, mean_g[1] / self.image_size
                                last_valid_gaze = [norm_gx, norm_gy]
                                gaze_processed.append(last_valid_gaze)
                            else:
                                print(f"[DEBUG] mean_g has unexpected shape: {mean_g.shape}, mean_g: {mean_g}")
                                gaze_processed.append(last_valid_gaze)
                        else:
                            print(f"[DEBUG] All projections filtered out due to shape issues. Original projs: {projs}")
                            gaze_processed.append(last_valid_gaze)

                    elif len(projs) == 1:
                        p = projs[0]
                        p = np.array(p)
                        if p.shape == (2,):
                            p = p * (self.image_size / 512)
                            norm_gx, norm_gy = p[0] / self.image_size, p[1] / self.image_size
                            last_valid_gaze = [norm_gx, norm_gy]
                            gaze_processed.append(last_valid_gaze)
                        else:
                            print(f"[DEBUG] Single proj has unexpected shape: {p.shape}, p: {p}")
                            gaze_processed.append(last_valid_gaze)
                    else:
                        print(f"[DEBUG] No valid projections found in g: {g}")
                        gaze_processed.append(last_valid_gaze)
                else:
                    print(f"[DEBUG] g is None or empty: {g}")
                    gaze_processed.append(last_valid_gaze)


                

                hands = []
                # Process left palm
                if 'left_wrist' in h and h['left_wrist'] is not None:
                    left_wrist_normal = (h['left_wrist'][0], h['left_wrist'][1])
                    left_wrist_normal = np.array(left_wrist_normal) * (self.image_size / 512)
                    norm_lwx, norm_lwy = round(left_wrist_normal[0] / self.image_size, 4), round(left_wrist_normal[1] / self.image_size, 4)
                    hands.append([norm_lwx, norm_lwy])
                else:
                    hands.append([np.nan, np.nan])

                if 'left_palm' in h and h['left_palm'] is not None:
                    left_palm_normal = (h['left_palm'][0], h['left_palm'][1])
                    left_palm_normal = np.array(left_palm_normal) * (self.image_size / 512)
                    norm_lpx, norm_lpy = round(left_palm_normal[0] / self.image_size, 4), round(left_palm_normal[1] / self.image_size, 4)
                    hands.append([norm_lpx, norm_lpy])
                else:
                    hands.append([np.nan, np.nan])
                
                
                if 'right_wrist' in h and h['right_wrist'] is not None:
                    right_palm_normal = (h['right_wrist'][0], h['right_wrist'][1])
                    right_palm_normal = np.array(right_palm_normal) * (self.image_size / 512)
                    norm_rwx, norm_rwy = round(right_palm_normal[0] / self.image_size, 4), round(right_palm_normal[1] / self.image_size, 4)
                    hands.append([norm_rwx, norm_rwy])
                else:
                    hands.append([np.nan, np.nan])
                
                # Process right palm
                if 'right_palm' in h and h['right_palm'] is not None:
                    right_palm_normal = (h['right_palm'][0], h['right_palm'][1])
                    right_palm_normal = np.array(right_palm_normal) * (self.image_size / 512)
                    norm_rpx, norm_rpy = round(right_palm_normal[0] / self.image_size, 4), round(right_palm_normal[1] / self.image_size, 4)
                    hands.append([norm_rpx, norm_rpy])
                else:
                    hands.append([np.nan, np.nan])
                            
                # Process right wrist
                
                
                hands = np.array(hands, dtype=object).flatten().tolist()  # Flatten the list to match expected format
                hands_processed.append(hands)

                # Process object detections
                if o is not None and len(o) > 0:
                    features = self.process_object_detections_with_gaze(o, gaze_processed[-1])
                    object_detections_processed.append(features)

            sample = {'frames': frames_processed,
                    'gaze': gaze_processed,
                    'imu': imu_processed,
                    'hands': hands_processed,
                    'object_detections': object_detections_processed,
                    'label': action,
                    'label_id': label_map.get(action, 0)} # Default to 0 if action not found}
            
            samples.append(sample)
            
            split = annotations_path.split('/')
            video_path = '/'.join(split[:-2])
            action_path = os.path.join(video_path,f'{self.image_size}_reduced_obj', f'{action}')
            video_name = os.path.join(action_path, f'{split[-2]}_{start}_{end}')

            
            if not os.path.exists(action_path):
                os.makedirs(action_path)
            
            if not os.path.exists(video_name):
                os.makedirs(video_name)
            
            self.evaluate_data_point(sample, video_name)
            np.save(os.path.join(video_name, 'data.npy'), sample)
        
    def write_drive(self):

        self.process_folder(self.data, self.annotations_path)

        

