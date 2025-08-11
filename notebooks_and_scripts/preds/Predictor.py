import numpy as np
import torch
import sys
sys.path.append('MSc_AI_Thesis/notebooks_and_scripts/')
from model_and_training.EgoDriveMax import EgoDriveMultimodalTransformer
from dataset_scripts.EgoDriveAriaDataset import EgoDriveAriaDataset
import cv2
import time
import tqdm


class Predictor:

    """Predictor class for processing and predicting using EgoDrive data."""

    def __init__(self,data = None, data_path = None):
        if data is not None:
            self.data = data
        elif data_path is not None:
            if not data_path.endswith('.npy'):
                raise ValueError("data_path must point to a .npy file")
            else:
                self.data = np.load(data_path, allow_pickle=True).item()['modalities'] 

    def preprocess(self, frames_segment, gaze_segment, hand_landmarks_segment, object_detections_segment, imu_segment):
        
        """Preprocesses the segments of frames, gaze, hand landmarks, object detections, and IMU data."""

        frames_processed = []
        gaze_processed = []
        hands_processed = []
        object_detections_processed = []
        imu_processed = []

        last_valid_gaze = None


        for f, g, h, o, i in zip(frames_segment, gaze_segment, hand_landmarks_segment, object_detections_segment, imu_segment):
                # Resize image

            img_rs = cv2.resize(f, (224, 224), interpolation=cv2.INTER_LINEAR)
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
            # if o is not None and len(o) > 0:
            #     features = self.process_object_detections_with_gaze(o, gaze_processed[-1])
            #     object_detections_processed.append(features)
            
            sample = {'frames': frames_processed,
                    'gaze': gaze_processed,
                    'imu': imu_processed,
                    'hands': hands_processed,
                    'object_detections': object_detections_processed}
        return sample

    def dict_transform(self,x):

        """Transforms the input dictionary to match the model's expected input format."""

        return {
            'frames': x['frames'],
            'gaze': x['gaze'],
            'hands': x['hands'],
            'imu': x['imu'],
            'objects': x['object_detections']
        }

    def instantiate(self, checkpoint_path = 'RT'):

        """Instantiates the model and loads the checkpoint."""

        if checkpoint_path == 'RT':
            checkpoint_path = 'MSc_AI_Thesis/models/EgoDriveRT.ckpt'
            self.light = True
        else:
            checkpoint_path = 'MSc_AI_Thesis/models/EgoDriveMax.ckpt'
            self.light = False

        self.model = EgoDriveMultimodalTransformer(
        dim_feat=32,
        dropout=0.1,
        num_classes=6,
        num_frames=32,
        transformer_depth=1,
        transformer_heads=2,
        light=self.light 
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)

        state_dict = checkpoint['state_dict']
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if not k.startswith('loss_fn.')}
        state_dict = {k.replace('base_', ''): v for k, v in state_dict.items()}


        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.model.to(device)
    
    def run(self, overlap = 0.5, fpc = 32):
        
        """Runs the prediction on the data with specified overlap and frames per chunk (fpc)."""

        preds = []

        self.instantiate()
        

        if not hasattr(self, 'model'):
            raise ValueError("Model not instantiated. Call instantiate() first.")

        
        start_time = time.time()

        for i in tqdm.tqdm(range(0, len(self.data['frames']), int(fpc * overlap))):
            
            frames = np.array(self.data['frames'][i:i+fpc])
            gaze = np.array(self.data['gaze'][i:i+fpc])
            hands = np.array(self.data['hands'][i:i+fpc])
            imu = np.array(self.data['imu'][i:i+fpc])
            objects = np.array(self.data['object_detections'][i:i+fpc])



            sample = {
            'frames': torch.from_numpy(frames).float(),
            'gaze': torch.from_numpy(gaze).float(),
            'hands': torch.from_numpy(hands).float(),
            'imu': torch.from_numpy(imu).float(),
            'object_detections': torch.from_numpy(objects).float(),
            }

            

            sample = self.dict_transform(sample)

            # Add batch dimension to each tensor in sample (except label)
            for k in sample:
                sample[k] = sample[k].unsqueeze(0)

            if self.light:
                s1 = sample.copy()
                s1.pop('frames')



            with torch.no_grad():
                output = self.model(s1)
                predicted_class = torch.argmax(output['logits'], dim=1)
            

            preds.append({'class':predicted_class.item(), 'start': i, 'end' : i + fpc})
        
        end_time = time.time()
        print(f"Loop execution time: {end_time - start_time:.2f} seconds")
        
        return preds
    

