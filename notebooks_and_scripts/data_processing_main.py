import os 
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from notebooks_and_scripts.vrs_extractor import VRSDataExtractor
import numpy as np
import tqdm

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

# egoblur appears to work but should be fully checked with better data 
class DataProcessor:

    def __init__(self, base_data_path):

        self.path = base_data_path
        

    def annotating_run(self, path):

        names = sorted([p for p in os.listdir(path) if 'Store' not in p])

        names = names[1:2]

        start_index = 7300
        end_index = 8300

        for rec_name in names:
            vrs_path = os.path.join(path,rec_name,(rec_name + '.vrs'))
            vde = VRSDataExtractor(vrs_path)

            vde.get_image_data(start_index,end_index)
            # vde.videotest_mediapipe()
            gaze_path = os.path.join(path, rec_name, 'general_eye_gaze.csv')
            hand_path = os.path.join(path, rec_name, 'wrist_and_palm_poses.csv')
            vde.get_gaze_hand(gaze_path, hand_path, start_index, end_index)

            blur_csv_path = os.path.join(path,rec_name,(rec_name + '_blur.csv'))
            actions_csv_path = os.path.join(path,rec_name,(rec_name + '_actions.csv'))
            vde.annotate(vde.result['rgb'],blur_csv_path,actions_csv_path )

            vde.get_IMU_data()

            # vde.mediapipe_detection()

            slam_path = os.path.join(path,rec_name,'slam')

            if os.path.exists(slam_path):
                vde.get_slam_data(slam_path)
                vde.pc_filter(slam_path)


            output_path = os.path.join(path,rec_name,(rec_name + '.npy'))
            vde.save_data(output_path)

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

    
if __name__ == "__main__":
    dp = DataProcessor('sampledata/driving_data/')
    # dp.annotating_run(dp.path)
    dp.blurring_run(dp.path)

