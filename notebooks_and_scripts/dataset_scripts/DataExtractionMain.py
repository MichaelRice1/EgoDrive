


import os 
import sys
from vrs_extractor import VRSDataExtractor
import pandas as pd
from projectaria_tools.core.mps import EyeGaze
from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

class DataProcessor:

    def __init__(self, base_data_path = None):
        pass


    def vrs_processing(self, path, start_frame = None, end_frame = None, preds_path=None, callbacks=None): 
        
        
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


        # imu_vals =  list(vde.result['imu_right'].values())
        # imu_gyro_x_samples = [g[3] for g in imu_vals]
        # imu_gyro_y_samples = [g[4] for g in imu_vals]
        # imu_gyro_z_samples = [g[5] for g in imu_vals]

        vde.get_object_dets(progress_callback=callbacks.get('object_detection'))
        # video_save_path = os.path.join('/',*split, 'video')
        # hand_data = list(vde.result['hand_landmarks'].values())

        

        # vde.evaluate_driving(list(vde.result['rgb'].values()),vde.result['smoothed_gaze'],vde.result['object_detections'],
        #                      imu_gyro_x_samples,imu_gyro_y_samples,imu_gyro_z_samples,hand_data,video_save_path, progress_callback=callbacks.get('driving_evaluation'), gaze_predictions=vde.result.get('gaze_predictions', None))

        # vde.score_driver(vde.num_frames_rgb, vde.result['joined_intervals'], video_save_path)


        # slam_path = os.path.join('/',*split, 'mps_SensorTest_vrs/slam'
        # vde.get_slam_data(slam_path)

        # vde.save_data(output_path)


        return vde.result


        
if __name__ == "__main__":
    dp = DataProcessor()

    drive_no = 9
    dp.vrs_processing(f'data/newdrive/Drive{drive_no}/Drive{drive_no}.vrs')


