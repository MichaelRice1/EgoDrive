import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import interp1d


class EgoDriveAriaAligner:
    """
    Simple, optimized alignment for Project Aria data for action recognition training.
    
    Single best approach: Visual frames drive alignment, all other modalities 
    aligned using nearest neighbor with temporal bounds.
    """
    
    def __init__(self, data, primary_visual_modality: str = 'rgb', max_time_gap: float = 5e7):
        """
        Args:
            primary_visual_modality: The visual modality that drives frame timing ('rgb', 'slam', 'et')
            max_time_gap: Maximum time difference for alignment (default 50ms)
        """
        self.primary_modality = primary_visual_modality
        self.data = data
        self.max_time_gap = max_time_gap
        self.modalities = {}
        self.frame_timestamps = None
        self.aligned_data = None
    
    def add_modality(self, name: str, data_dict: Dict[float, Any]):
        """Add a modality's data dictionary (timestamp -> data)"""
        if name == 'object_detections':
            timestamps = self.frame_timestamps
            data_dict = {ts: val for ts,val in zip(timestamps, data_dict)}
        # elif name == 'gaze':
        #     timestamps = data_dict['tracking_timestamp_us']
        #     timestamps = np.array([t * 1000 for t in timestamps])  # Convert to nanoseconds
        #     data_dict = {ts: {
        #         'projected_point_2d_x': data_dict.loc[data_dict['tracking_timestamp_us'] == ts // 1000, 'projected_point_2d_x'].values[0],
        #         'projected_point_2d_y': data_dict.loc[data_dict['tracking_timestamp_us'] == ts // 1000, 'projected_point_2d_y'].values[0],
        #         'transformed_gaze_x': data_dict.loc[data_dict['tracking_timestamp_us'] == ts // 1000, 'transformed_gaze_x'].values[0],
        #         'transformed_gaze_y': data_dict.loc[data_dict['tracking_timestamp_us'] == ts // 1000, 'transformed_gaze_y'].values[0],
        #         'transformed_gaze_z': data_dict.loc[data_dict['tracking_timestamp_us'] == ts // 1000, 'transformed_gaze_z'].values[0],
        #         'depth_m': data_dict.loc[data_dict['tracking_timestamp_us'] == ts // 1000, 'depth_m'].values[0]
        #     } for ts in timestamps}
            
        else:
            timestamps = np.array(sorted(data_dict.keys()))
            timestamps = timestamps.astype(np.int64)
            updated_data_dict = {int(ts): value for ts, value in data_dict.items()}
            data_dict.clear()
            data_dict.update(updated_data_dict)
        
        self.modalities[name] = {
            'data': data_dict,
            'timestamps': timestamps
        }
        print(f"Added {name}: {len(timestamps)} samples")
        
        # Set frame timestamps from primary visual modality
        if name == self.primary_modality:
            self.frame_timestamps = timestamps
            print(f"Using {name} as primary frame source: {len(timestamps)} frames")
    
    def align(self) -> Dict[str, Any]:
        """
        Align all modalities to the primary visual frame timeline.
        Returns frame-indexed aligned dataset ready for action recognition.
        """
        if self.frame_timestamps is None:
            raise ValueError(f"Primary modality '{self.primary_modality}' not added yet")
        
        if len(self.modalities) < 2:
            raise ValueError("Need at least 2 modalities")
        
        print(f"Aligning {len(self.modalities)} modalities to {len(self.frame_timestamps)} frames...")
        
        # Initialize aligned data structure
        aligned_modalities = {}
        stats = {}
        
        # Process each modality
        for mod_name, mod_data in self.modalities.items():
            if mod_name == self.primary_modality or mod_name == 'object_detections':
                # Primary modality: direct frame-to-data mapping
                aligned_modalities[mod_name] = [
                    mod_data['data'][timestamp] for timestamp in self.frame_timestamps
                ]
                stats[mod_name] = {
                    'aligned_samples': len(self.frame_timestamps),
                    'completion_rate': 1.0,
                    'mean_time_diff': 0.0
                }
                print(f"  {mod_name}: 100% (primary)")
            else:
                # Other modalities: nearest neighbor alignment
                aligned_data, time_diffs = self._align_to_frames(
                    mod_data['data'], mod_data['timestamps'], self.frame_timestamps, mod_name
                )
                aligned_modalities[mod_name] = aligned_data

                
                # Calculate stats
                valid_samples = sum(1 for x in aligned_data if x is not None)
                valid_diffs = [d for d in time_diffs if d < float('inf')]
                
                stats[mod_name] = {
                    'aligned_samples': valid_samples,
                    'completion_rate': valid_samples / len(self.frame_timestamps),
                    'mean_time_diff': np.mean(valid_diffs) if valid_diffs else 0.0
                }
                
                print(f"  {mod_name}: {stats[mod_name]['completion_rate']:.1%} "
                      f"({valid_samples}/{len(self.frame_timestamps)} frames)")
        
        # Store results
        self.aligned_data = {
            'frame_count': len(self.frame_timestamps),
            'frame_timestamps': self.frame_timestamps,
            'modalities': aligned_modalities,
            'stats': stats
        }
        
        print(f"\nAlignment complete: {len(self.frame_timestamps)} frames ready for training")
        return self.aligned_data
    
    def _align_to_frames(self, data_dict: Dict[float, Any], 
                    timestamps: np.ndarray, 
                    frame_timestamps: np.ndarray,
                    mod_name) -> tuple:
        """Align modality data to frame timestamps """
        aligned_data = []
        time_diffs = []

        if 'imu' in mod_name:
            # IMU-specific alignment with interpolation
            if 'right' in mod_name:
                imu_data = np.array(list(data_dict.values()))
                timestamps = np.array(timestamps, dtype=np.int64)

                samples_per_frame = 67
                half_window = int(1e9 / 15 / 2)  # ~33.33 ms in ns

                # Convert timestamps to seconds once
                timestamps_s = timestamps * 1e-9
                frame_timestamps_s = frame_timestamps * 1e-9

                # Create interpolators for each IMU dimension
                interpolators = [
                    interp1d(timestamps_s, imu_data[:, dim], kind='linear', 
                            bounds_error=False, fill_value='extrapolate')
                    for dim in range(imu_data.shape[1])
                ]

                for t_ns in frame_timestamps:
                    start_ns = t_ns - half_window
                    end_ns = t_ns + half_window
                    window_times_s = np.linspace(start_ns, end_ns, samples_per_frame) * 1e-9

                    frame_samples = np.stack([f(window_times_s) for f in interpolators], axis=1)
                    aligned_data.append(frame_samples)
                    time_diffs.append(0)  

                return np.array(aligned_data), time_diffs

            elif 'left' in mod_name:
                imu_data = np.array(list(data_dict.values()))
                timestamps = np.array(timestamps, dtype=np.int64)

                samples_per_frame = 53
                half_window = int(1e9 / 15 / 2)

                timestamps_s = timestamps * 1e-9
                interpolators = [
                    interp1d(timestamps_s, imu_data[:, dim], kind='linear', 
                            bounds_error=False, fill_value='extrapolate')
                    for dim in range(imu_data.shape[1])
                ]

                for t_ns in frame_timestamps:
                    start_ns = t_ns - half_window
                    end_ns = t_ns + half_window
                    window_times_s = np.linspace(start_ns, end_ns, samples_per_frame) * 1e-9

                    frame_samples = np.stack([f(window_times_s) for f in interpolators], axis=1)
                    aligned_data.append(frame_samples)
                    time_diffs.append(0)

                return np.array(aligned_data), time_diffs

        elif 'gaze' in mod_name or 'et' in mod_name:
            # Gaze/ET: take 2 nearest samples per frame
            for t in frame_timestamps:
                diffs = np.abs(timestamps - t)
                nearest_idxs = np.argpartition(diffs, 2)[:2]
                nearest_tss = [timestamps[idx] for idx in nearest_idxs]
                
                samples = [data_dict[ts] for ts in nearest_tss]
                if 'et' in mod_name:
                    for sample in samples:
                        aligned_data.append(sample)
                else:
                    aligned_data.append(samples)

                time_diffs.append(diffs[nearest_idxs].mean())  # Average time diff
                
            return aligned_data, time_diffs

        else:
            # Default: nearest neighbor alignment for other modalities
            for frame_ts in frame_timestamps:
                time_diff_array = np.abs(timestamps - frame_ts)
                nearest_idx = np.argmin(time_diff_array)
                time_diff = time_diff_array[nearest_idx]
                
                if time_diff <= self.max_time_gap:
                    nearest_ts = timestamps[nearest_idx]
                    aligned_data.append(data_dict[nearest_ts])
                    time_diffs.append(time_diff)
                else:
                    aligned_data.append(None)
                    time_diffs.append(float('inf'))

            return aligned_data, time_diffs
    
    def print_summary(self):
        """Print alignment summary"""
        if self.aligned_data is None:
            print("No alignment performed yet.")
            return
        
        print(f"\n=== ACTION RECOGNITION DATASET SUMMARY ===")
        print(f"Total frames: {self.aligned_data['frame_count']}")
        print(f"Primary modality: {self.primary_modality}")
        print(f"Duration: {self.frame_timestamps[-1] - self.frame_timestamps[0]:.2f}s")
        print(f"Frame rate: {len(self.frame_timestamps) / (self.frame_timestamps[-1] - self.frame_timestamps[0]):.1f} FPS")
        print("\nModality completion rates:")
        
        for mod_name, stats in self.aligned_data['stats'].items():
            print(f"  {mod_name:15}: {stats['completion_rate']:6.1%} "
                  f"({stats['aligned_samples']}/{self.aligned_data['frame_count']} frames)")
        
        avg_completion = np.mean([s['completion_rate'] for s in self.aligned_data['stats'].values()])
        print(f"\nAverage completion: {avg_completion:.1%}")

    def align_drive(self):
        
        aligner = EgoDriveAriaAligner(
            data=self.data,
            primary_visual_modality='rgb',
            max_time_gap=5e7  # 50ms max gap
        )

        rgb_data = aligner.data['rgb']

        len_pgaze = len(aligner.data['personalized_gaze']) if 'personalized_gaze' in aligner.data else 0

        vals = list(aligner.data['personalized_gaze'].values()) if 'personalized_gaze' in aligner.data else []

        null_count = sum(1 for g in vals if g['projection'] is None)
        print(f'Number of null personalized gaze points: {null_count}')        

        
        if len_pgaze > 0 and null_count < 50:
            # If personalized gaze data exists, use it
            gaze_data = aligner.data['personalized_gaze']
        else:
            # Otherwise, use general gaze data
            print("No personalized gaze data found. Using general gaze data instead.")
            if 'gaze' in aligner.data:
                gaze_data = aligner.data['gaze']
            else:
                raise ValueError("No gaze data found in the dataset.")

        
        imu_right_data = aligner.data['imu_right']
        imu_left_data = aligner.data['imu_left']
        hand_landmarks_data = aligner.data['hand_landmarks']
        gps_data = aligner.data['gps']
        et_frames_data = aligner.data['et']
        object_detections_data = aligner.data['object_detections']
        # slam_left_data = aligner.data['slam_left']
        # slam_right_data = aligner.data['slam_right']
        
        
        

        # Add modalities
        aligner.add_modality('rgb', rgb_data)
        aligner.add_modality('gaze', gaze_data)
        aligner.add_modality('imu_right', imu_right_data)
        aligner.add_modality('imu_left', imu_left_data)
        aligner.add_modality('hand_landmarks', hand_landmarks_data)
        aligner.add_modality('object_detections', object_detections_data)
        # aligner.add_modality('gps', gps_data)
        aligner.add_modality('et', et_frames_data)
        # aligner.add_modality('slam_left', slam_left_data)
        # aligner.add_modality('slam_right', slam_right_data)
        
        # # Align data
        aligned_dataset = aligner.align()


        return aligned_dataset
    
