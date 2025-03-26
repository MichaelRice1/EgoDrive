import os 
import sys
import matplotlib.pyplot as plt
sys.path.append('C:/Users/athen/Desktop/Github/MastersThesis/MSc_AI_Thesis/notebooks_and_scripts')
from vrs_extractor import VRSDataExtractor
import numpy as np

# egoblur appears to work but should be fully checked with better data 
class DataProcessor:

    def __init__(self, base_data_path):

        self.path = base_data_path
        

    def annotating_run(self, path):

        names = os.listdir(path)

        for rec_name in names:
            vrs_path = os.path.join(path,rec_name,(rec_name + '.vrs'))
            vde = VRSDataExtractor(vrs_path)

            vde.get_image_data()

            blur_csv_path = os.path.join(path,rec_name,(rec_name + '_blur.csv'))
            actions_csv_path = os.path.join(path,rec_name,(rec_name + '_actions.csv'))
            # vde.annotate(vde.result['rgb'],blur_csv_path,actions_csv_path )

            gaze_path = os.path.join(path, rec_name, 'general_eye_gaze.csv')
            hand_path = os.path.join(path, rec_name, 'wrist_and_palm_poses.csv')
            vde.get_gaze_hand(gaze_path, hand_path)
            # vde.mediapipe_detection()

            slam_path = os.path.join(path,rec_name,'slam')
            # vde.get_slam_data(slam_path)

            # vde.get_IMU_data()

            output_path = os.path.join(path,rec_name,(rec_name + '.npy'))
            vde.save_data(output_path)



    def blurring_run(self, path):
        names = os.listdir(path)

        for rec_name in names:
            vrs_path = os.path.join(path,rec_name,(rec_name + '.vrs'))
            vde = VRSDataExtractor(vrs_path)

            curr_res = np.load(os.path.join(path,rec_name,(rec_name + '.npy')), allow_pickle=True).item()
            frames = curr_res['rgb']

            blur_csv_path = os.path.join(path,rec_name,(rec_name + '_blur.csv'))

            blurred_frames = vde.ego_blur(blur_csv_path, frames)
            curr_res['rgb'] = blurred_frames

            samp_frame = curr_res['rgb'][541]
            plt.imshow(samp_frame)
            plt.show()

            np.save(os.path.join(path,rec_name,(rec_name + '.npy')), curr_res)
            break

    

if __name__ == "__main__":
    dp = DataProcessor('sampledata/testfolder')
    # dp.annotating_run(dp.path)
    dp.blurring_run(dp.path)

