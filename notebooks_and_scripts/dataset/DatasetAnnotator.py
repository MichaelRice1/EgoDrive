import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from notebooks_and_scripts.vrs_extractor import VRSDataExtractor


class DatasetAnnotator:
    def __init__(self, base_path):
        self.base_path = base_path

    def process_folder(self, folder):
        actions_csv_path = os.path.join('train', folder, 'actions.csv')
        blur_csv_path = os.path.join('train', folder, 'blur.csv')
        vrs_path = os.path.join('train', folder, f"{folder}.vrs")

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
            vde.get_gaze_data(gaze_path, personalized_gaze_path)
        elif os.path.exists(gaze_path):
            vde.get_gaze_data(gaze_path, None)
        else:
            print(f"Gaze data not found at: {gaze_path}. Skipping gaze data processing.")

        hand_path = os.path.join(full_path, f'mps_{name}_vrs', 'hand_tracking', 'hand_tracking_results.csv')
        vde.get_hand_data(hand_path)

        vde.annotate(vde.result['rgb'], actions_csv_path, blur_csv_path)
        vde.save_data(os.path.join(full_path, f"{folder}.npy"))

    def process_all_folders(self):
        folders = os.listdir(self.base_path)
        for folder in sorted(folders):
            self.process_folder(folder)
            break  # Remove this break if you want to process all folders


if __name__ == "__main__":
    base_path = '/Users/michaelrice/Documents/GitHub/Thesis/MSc_AI_Thesis/train'
    annotator = DatasetAnnotator(base_path)
    annotator.process_all_folders()
