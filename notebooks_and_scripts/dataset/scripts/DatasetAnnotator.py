import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from vrs_extractor import VRSDataExtractor


class DatasetAnnotator:
    def __init__(self, base_path):
        self.base_path = base_path

    def process_folder(self, folder):
        actions_csv_path = os.path.join(base_path, folder, 'actions.csv')
        blur_csv_path = os.path.join(base_path, folder, 'blur.csv')
        vrs_path = os.path.join(base_path, folder, f"{folder}.vrs")

        if os.path.exists(actions_csv_path):
            print(f"Annotations file found for {folder}. Skipping...")
            return

        vde = VRSDataExtractor(vrs_path)
        vde.get_image_data(rgb_flag=True)

        full_path = os.path.join(self.base_path, folder)
        name = vrs_path.split('/')[-1].split('.')[0]

        gaze_path = os.path.join(full_path, f'mps_{name}_vrs', 'eye_gaze', 'general_eye_gaze.csv')
        personalized_gaze_path = os.path.join(full_path, f'mps_{name}_vrs', 'eye_gaze', 'personalized_eye_gaze.csv')

        if os.path.exists(gaze_path) and os.path.exists(personalized_gaze_path):
            print('hi1')
            vde.get_gaze_data(gaze_path, personalized_gaze_path)
        elif os.path.exists(gaze_path):
            print('hi2')
            vde.get_gaze_data(gaze_path, None)
        else:
            print(f"Gaze data not found at: {gaze_path}. Skipping gaze data processing.")

        hand_path = os.path.join(full_path, f'mps_{name}_vrs', 'hand_tracking', 'hand_tracking_results.csv')
        vde.get_hand_data(hand_path)
        vde.annotate(vde.result['rgb'], actions_csv_path, blur_csv_path)

    # def process_all_folders(self):
    #     folders = os.listdir(self.base_path)
    #     folders = [folder for folder in folders if 'Drive' in folder]
    #     for folder in sorted(folders):
    #         self.process_folder(folder)


if __name__ == "__main__":
    base_path = '/Users/michaelrice/Documents/GitHub/Thesis/MSc_AI_Thesis/data/newdrive'
    annotator = DatasetAnnotator(base_path)
    annotator.process_folder('Drive1_3')
