
import os 
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from notebooks_and_scripts.vrs_extractor import VRSDataExtractor
import numpy as np
import tqdm
import cv2
import csv
from time import time


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

# egoblur appears to work but should be fully checked with better data 
class DataProcessor:

    def __init__(self, base_data_path = None):

        self.path = base_data_path
    
    def yolo_frame_extraction(self, path, totalNumFrames = None):

        names = [n for n in os.listdir(path) if n.endswith('.vrs')]
        
        npypaths = []
        frame_output_path = 'sampledata/yolo/actualframes'


        if totalNumFrames is None:
            for name in names:
                vrspath = os.path.join(path,name)
                fname = name.split('.')[0]

                vde = VRSDataExtractor(vrspath)
                vde.get_image_data(rgb_flag=True)

                output_path = os.path.join(path,'npyframes',(fname + '.npy'))
                npypaths.append(output_path)
                if not os.path.exists(output_path):
                    vde.save_data(output_path)
        
            for i,p in enumerate(npypaths):
                data = np.load(p,allow_pickle=True).item()
                frames = data['rgb']
                frames = list(frames.values())

                filename = p.split('/')[-1].split('.')[0]

                for j, frame in enumerate(tqdm.tqdm(frames, desc=f"Processing file {i+1}/{len(npypaths)}")):
                    
                    if not os.path.exists(os.path.join(frame_output_path,filename)):
                        os.mkdir(os.path.join(frame_output_path,filename))
                    
                    output_filename = f"{frame_output_path}/{filename}/frame_{j:04d}.jpg"
                    cv2.imwrite(output_filename, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
    def annotating_run(self, path, start_frame, end_frame, et_scale = 2):


        names = sorted([p for p in os.listdir(path) if 'Store' or 'failed' not in p])

        for rec_name in names:
            vrs_path = os.path.join(path,rec_name,(rec_name + '.vrs'))
            vde = VRSDataExtractor(vrs_path)
            vde.get_image_data(start_frame,end_frame)
            gaze_path = os.path.join(path, rec_name, 'gaze1.csv')
            hand_path = os.path.join(path, rec_name, 'wap1.csv')

            vde.get_gaze_hand(gaze_path, hand_path, start_frame*et_scale, end_frame*et_scale)

            blur_csv_path = os.path.join(path,rec_name,(rec_name + f'_{start_frame}blur.csv'))
            actions_csv_path = os.path.join(path,rec_name,(rec_name + f'_{start_frame}actions.csv'))
            environment_csv_path = os.path.join(path,rec_name,(rec_name + f'_{start_frame}frame_verification1.csv'))


            vde.annotate(vde.result['rgb'],blur_csv_path,actions_csv_path, environment_csv_path)


            break

    def blurring_run(self, path):

        names = sorted([p for p in os.listdir(path) if 'Store' not in p])

        names = names[1:2]


        for rec_name in tqdm.tqdm(names):
            vrs_path = os.path.join(path,rec_name,(rec_name + '.vrs'))
            vde = VRSDataExtractor(vrs_path)

            curr_res = np.load(os.path.join(path,rec_name,(rec_name + '.npy')), allow_pickle=True).item()
            frames = curr_res['rgb']

            blur_csv_path = os.path.join(path,rec_name,(rec_name + '_blur.csv'))

            blurred_frames = vde.ego_blur(blur_csv_path, frames)
            curr_res['rgb'] = blurred_frames

            np.save(os.path.join(path,rec_name,(rec_name + '.npy')), curr_res)

    def vrs_processing(self, path, start_frame = None, end_frame = None, annotate = False): 

        vde = VRSDataExtractor(path)
        # vde.get_GPS_data()
        if start_frame is None or end_frame is None:
            start_frame = 0
            end_frame = vde.num_frames_rgb

        vde.get_image_data(start_index=start_frame,end_index=end_frame ,rgb_flag=True)
        split = path.split('/')[:-1]
        output_path = os.path.join('/',*split, path.split('/')[-1].split('.')[0] + '.npy')
        gaze_path = os.path.join('/',*split, 'gaze1.csv')
        hand_path = os.path.join('/',*split, 'wap1.csv')
        vde.get_gaze_hand(gaze_path, hand_path)


        # # # vde.get_slam_data(slam_path)
        vde.get_IMU_data()
        
        if annotate:
            vde.annotate(vde.result['rgb'],'actions.csv')

        vde.get_object_dets()



        

        # # slam_path = os.path.join('/',*split, 'mps_SensorTest_vrs/slam')
        

        # vde.save_data(output_path)
        return vde.result


        
if __name__ == "__main__":
    dp = DataProcessor('/Users/michaelrice/Documents/GitHub/Thesis/MSc_AI_Thesis/sampledata/yolo')

    # dp.blurring_run(dp.path)
    # dp.yolo_frame_extraction(dp.path)

    # verification_path = os.path.join('/Volumes/MichaelSSD/dataset/realdata')
    # dp.annotating_run(verification_path,0,100)

    dp.vrs_processing('/Users/michaelrice/Documents/GitHub/Thesis/MSc_AI_Thesis/sampledata/proper_cartesting/1/1.vrs',start_frame=50,end_frame=150)


