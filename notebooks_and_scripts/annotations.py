
import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from notebooks_and_scripts.vrs_extractor import VRSDataExtractor

path = '/Users/michaelrice/Documents/GitHub/Thesis/MSc_AI_Thesis/train'

folders = os.listdir(path)

for folder in sorted(folders):
    actions_csv_path = os.path.join('train', folder, 'actions.csv')
    blur_csv_path = os.path.join('train', folder, 'blur.csv')
    vrs_path = os.path.join('train', folder, (folder + '.vrs'))
    if os.path.exists(actions_csv_path):
        print(f"Annotations file found for {folder}. Skipping...")
        continue
    
    vde = VRSDataExtractor(vrs_path)
    vde.get_image_data(rgb_flag=True)

    full_path = os.path.join(path, folder)
    
    name = vrs_path.split('/')[-1].split('.')[0]

    gaze_path = os.path.join(full_path ,f'mps_{name}_vrs','eye_gaze','general_eye_gaze.csv')
    personalized_gaze_path = os.path.join(full_path, f'mps_{name}_vrs','eye_gaze','personalized_eye_gaze.csv')
   

    if os.path.exists(gaze_path) and os.path.exists(personalized_gaze_path):
        vde.get_gaze_data(gaze_path, personalized_gaze_path)
    elif os.path.exists(gaze_path):
        vde.get_gaze_data(gaze_path, None)
    else:
        print(f"Gaze data not found at: {gaze_path}. Skipping gaze data processing.")    
    
    hand_path = os.path.join(full_path, f'mps_{name}_vrs','hand_tracking','hand_tracking_results.csv')
    
    vde.get_hand_data(hand_path)
    
    vde.annotate(vde.result['rgb'], actions_csv_path, blur_csv_path)
    break






# for rec_name in names:
#     vrs_path = os.path.join(path,rec_name,(rec_name + '.vrs'))
#     vde = VRSDataExtractor(vrs_path)
#     vde.get_image_data(start_frame,end_frame)
#     # gaze_path = os.path.join(path, rec_name, 'gaze1.csv')
#     # hand_path = os.path.join(path, rec_name, 'wap1.csv')

#     vde.get_gaze_hand(gaze_path, hand_path, start_frame*et_scale, end_frame*et_scale)


    


#     vde.annotate(vde.result['rgb'],blur_csv_path,actions_csv_path, environment_csv_path)


