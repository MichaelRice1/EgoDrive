


import os 
import sys
import sys
import cv2
import numpy as np
sys.path.append('/Users/michaelrice/Documents/GitHub/Thesis/MSc_AI_Thesis/notebooks_and_scripts')
from dataset_scripts.vrs_extractor import VRSDataExtractor
from dataset_scripts.Aligner import EgoDriveAriaAligner
import pandas as pd
from projectaria_tools.core.mps import EyeGaze
from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

class DataProcessor:

    def __init__(self, base_data_path = None):
        pass
    
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
    
    def preprocess(self, aligned):
        
        frames_processed = []
        gaze_processed = []
        hands_processed = []
        object_detections_processed = []
        imu_processed = []

        last_valid_gaze = None

        frames = aligned['modalities']['rgb']
        gaze = aligned['modalities']['gaze']
        hands = aligned['modalities']['hand_landmarks']
        obdets = aligned['modalities']['object_detections']
        imu = aligned['modalities']['imu_right']


        for f, g, h, o, i in zip(frames, gaze, hands, obdets, imu):

            img_rs = cv2.resize(f, (512, 512), interpolation=cv2.INTER_LINEAR)
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
                            mean_g = mean_g * (224 / 512)
                            norm_gx, norm_gy = mean_g[0] / 224, mean_g[1] / 224
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
                        p = p * (224/ 512)
                        norm_gx, norm_gy = p[0] / 224, p[1] / 224
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
            if h is not None:
                if 'left_wrist' in h and h['left_wrist'] is not None:
                    left_wrist_normal = (h['left_wrist'][0], h['left_wrist'][1])
                    left_wrist_normal = np.array(left_wrist_normal) * (224 / 512)
                    norm_lwx, norm_lwy = round(left_wrist_normal[0] / 224, 4), round(left_wrist_normal[1] / 224, 4)
                    hands.append([norm_lwx, norm_lwy])
                else:
                    hands.append([np.nan, np.nan])

                if 'left_palm' in h and h['left_palm'] is not None:
                    left_palm_normal = (h['left_palm'][0], h['left_palm'][1])
                    left_palm_normal = np.array(left_palm_normal) * (224 / 512)
                    norm_lpx, norm_lpy = round(left_palm_normal[0] / 224, 4), round(left_palm_normal[1] / 224, 4)
                    hands.append([norm_lpx, norm_lpy])
                else:
                    hands.append([np.nan, np.nan])
                
                
                if 'right_wrist' in h and h['right_wrist'] is not None:
                    right_palm_normal = (h['right_wrist'][0], h['right_wrist'][1])
                    right_palm_normal = np.array(right_palm_normal) * (224/ 512)
                    norm_rwx, norm_rwy = round(right_palm_normal[0] / 224, 4), round(right_palm_normal[1] / 224, 4)
                    hands.append([norm_rwx, norm_rwy])
                else:
                    hands.append([np.nan, np.nan])
                
                # Process right palm
                if 'right_palm' in h and h['right_palm'] is not None:
                    right_palm_normal = (h['right_palm'][0], h['right_palm'][1])
                    right_palm_normal = np.array(right_palm_normal) * (224 / 512)
                    norm_rpx, norm_rpy = round(right_palm_normal[0] / 224, 4), round(right_palm_normal[1] / 224, 4)
                    hands.append([norm_rpx, norm_rpy])
                else:
                    hands.append([np.nan, np.nan])
            else:
                # If hand landmarks are None, append NaN values for both wrists and palms
                hands.append([np.nan, np.nan])  # Process left wrist
                hands.append([np.nan, np.nan])  # Process left palm
                hands.append([np.nan, np.nan])  # Process right wrist
                hands.append([np.nan, np.nan]) # Process right palm
                        
            # Process right wrist
            
            
            hands = np.array(hands, dtype=object).flatten().tolist()  # Flatten the list to match expected format
            hands_processed.append(hands)

            # Process object detections
            if o is not None and len(o) > 0:
                features = self.process_object_detections_with_gaze(o, gaze_processed[-1])
                object_detections_processed.append(features)
            
            processed = {'frames': frames_processed,
                    'gaze': gaze_processed,
                    'imu': imu_processed,
                    'hands': hands_processed,
                    'object_detections': object_detections_processed}
            

        return processed

    def vrs_processing(self, path, preds_path=None, callbacks=None): 
        
        
        if callbacks is None:
            callbacks = {}

        vde = VRSDataExtractor(path)

        if preds_path is not None:
            gaze_preds = []
            gaze = pd.read_csv(preds_path)
            device_calibration = vde.device_calibration
            rgb_camera_calibration = vde.rgb_camera_calibration


            for i, row in gaze.iterrows():
                eye_gaze = EyeGaze
                eye_gaze.yaw = row["yaw_rads_cpf"]
                eye_gaze.pitch = row["pitch_rads_cpf"]
                gaze_projection = get_gaze_vector_reprojection(
                    eye_gaze,
                    'camera-rgb',
                    device_calibration,
                    rgb_camera_calibration,
                    1
                )
                gaze_projection = gaze_projection * (512/1408)
                gaze_projection = [int(512 - gaze_projection[1]), int(gaze_projection[0])] 
                gaze_preds.append(gaze_projection)

            vde.result['gaze_predictions'] = gaze_preds


        vde.get_image_data(rgb_flag=False, progress_callback=callbacks.get('image_extraction'))
        split = path.split('/')[:-1]
        output_path = os.path.join('/',*split, path.split('/')[-1].split('.')[0] + '.npy')


        name = path.split('/')[-1].split('.')[0]

        gaze_path = os.path.join('/',*split, f'mps_{name}_vrs','eye_gaze','general_eye_gaze.csv')
        personalized_gaze_path = os.path.join('/',*split, f'mps_{name}_vrs','eye_gaze','personalized_eye_gaze.csv')


        if os.path.exists(gaze_path) and os.path.exists(personalized_gaze_path):
            vde.get_gaze_data(gaze_path, personalized_gaze_path)
        elif os.path.exists(gaze_path):
            vde.get_gaze_data(gaze_path, None)
        else:
            print(f"Gaze data not found at: {gaze_path}. Skipping gaze data processing.")

        
        
        hand_path = os.path.join('/',*split, f'mps_{name}_vrs','hand_tracking','hand_tracking_results.csv')

        print(f"Processing hand data from: {gaze_path}")
        if os.path.exists(hand_path):
            vde.get_hand_data(hand_path)

        
        vde.get_GPS_data()
        vde.get_IMU_data()




        vde.get_object_dets(progress_callback=callbacks.get('object_detection'))

        aligner = EgoDriveAriaAligner(data=vde.result, primary_visual_modality='rgb', max_time_gap=5e7)
        aligned_data = aligner.align_drive()

        object_dets_raw = aligned_data['modalities']['object_detections']


        processed_data = self.preprocess(aligned_data)



        video_save_path = os.path.join('/',*split, 'video')

        vde.evaluate_driving(processed_data,object_dets_raw, video_save_path, progress_callback=callbacks.get('driving_evaluation'))
        vde.score_driver(vde.result)



        return vde.result

        




        # slam_path = os.path.join('/',*split, 'mps_SensorTest_vrs/slam'
        # vde.get_slam_data(slam_path)

        # vde.save_data(output_path)




        

