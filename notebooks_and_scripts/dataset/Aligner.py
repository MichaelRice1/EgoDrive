import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
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
        timestamps = np.array(sorted(data_dict.keys()))
        timestamps = timestamps.astype(np.int64)

        # Replace old timestamps in data_dict with int64 ones
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
            if mod_name == self.primary_modality:
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
                    mod_data['data'], mod_data['timestamps'], self.frame_timestamps
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
                        frame_timestamps: np.ndarray) -> tuple:
        """Align modality data to frame timestamps using nearest neighbor"""
        aligned_data = []
        time_diffs = []
        
        for frame_ts in frame_timestamps:
            # Find nearest timestamp

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
    
    def get_window(self, start_frame: int, end_frame: int) -> Dict[str, List[Any]]:
        """
        Extract a window of aligned data for training.
        
        Args:
            start_frame: Starting frame index (inclusive)
            end_frame: Ending frame index (exclusive)
            
        Returns:
            Dictionary with modality data for the specified frame window
        """
        if self.aligned_data is None:
            raise ValueError("No aligned data. Run align() first.")
        
        if start_frame < 0 or end_frame > self.aligned_data['frame_count']:
            raise ValueError(f"Frame range [{start_frame}:{end_frame}] out of bounds "
                           f"[0:{self.aligned_data['frame_count']}]")
        
        window_data = {}
        
        for mod_name, mod_data in self.aligned_data['modalities'].items():
            window_data[mod_name] = mod_data[start_frame:end_frame]
        
        # Add frame metadata
        window_data['frame_indices'] = list(range(start_frame, end_frame))
        window_data['frame_timestamps'] = self.frame_timestamps[start_frame:end_frame].tolist()
        window_data['window_duration'] = (
            self.frame_timestamps[end_frame-1] - self.frame_timestamps[start_frame]
        )
        
        return window_data
    
    def get_training_windows(self, window_size: int, stride: int = 1, 
                           min_valid_ratio: float = 0.8) -> List[Dict[str, List[Any]]]:
        """
        Generate sliding windows for training.
        
        Args:
            window_size: Number of frames per window
            stride: Frame stride between windows
            min_valid_ratio: Minimum ratio of non-None data required per window
            
        Returns:
            List of window dictionaries ready for training
        """
        if self.aligned_data is None:
            raise ValueError("No aligned data. Run align() first.")
        
        windows = []
        frame_count = self.aligned_data['frame_count']
        
        for start_frame in range(0, frame_count - window_size + 1, stride):
            end_frame = start_frame + window_size
            window = self.get_window(start_frame, end_frame)
            
            # Check data quality
            if self._is_window_valid(window, min_valid_ratio):
                windows.append(window)
        
        print(f"Generated {len(windows)} training windows "
              f"(size={window_size}, stride={stride})")
        
        return windows
    
    def _is_window_valid(self, window: Dict[str, List[Any]], 
                        min_valid_ratio: float) -> bool:
        """Check if window has sufficient valid data"""
        for mod_name, mod_data in window.items():
            if mod_name in ['frame_indices', 'frame_timestamps', 'window_duration']:
                continue
                
            valid_count = sum(1 for x in mod_data if x is not None)
            if valid_count / len(mod_data) < min_valid_ratio:
                return False
        
        return True
    
    def export_for_training(self, output_format: str = 'numpy') -> Dict[str, Any]:
        """
        Export aligned data in training-ready format.
        
        Args:
            output_format: 'numpy' or 'list'
        """
        if self.aligned_data is None:
            raise ValueError("No aligned data. Run align() first.")
        
        if output_format == 'numpy':
            # Convert to numpy arrays where possible
            export_data = {}
            for mod_name, mod_data in self.aligned_data['modalities'].items():
                try:
                    # Try to convert to numpy array
                    # Handle None values by masking
                    valid_indices = [i for i, x in enumerate(mod_data) if x is not None]
                    if valid_indices:
                        valid_data = [mod_data[i] for i in valid_indices]
                        if isinstance(valid_data[0], (int, float)):
                            # Scalar data
                            arr = np.full(len(mod_data), np.nan)
                            arr[valid_indices] = valid_data
                            export_data[mod_name] = arr
                        elif isinstance(valid_data[0], (list, tuple, np.ndarray)):
                            # Vector data
                            sample_shape = np.array(valid_data[0]).shape
                            arr = np.full((len(mod_data),) + sample_shape, np.nan)
                            for i, idx in enumerate(valid_indices):
                                arr[idx] = valid_data[i]
                            export_data[mod_name] = arr
                        else:
                            # Non-numeric data, keep as list
                            export_data[mod_name] = mod_data
                    else:
                        export_data[mod_name] = mod_data
                except:
                    # Fall back to list format
                    export_data[mod_name] = mod_data
            
            export_data['frame_timestamps'] = self.frame_timestamps
            return export_data
        
        else:  # list format
            return self.aligned_data['modalities']
    
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



if __name__ == "__main__":
# Create aligner
    data = np.load('/Volumes/MichaelSSD/dataset/Compiled Data (.npy)/Drive8.npy', allow_pickle=True).item()  # Load your data here
    aligner = EgoDriveAriaAligner(
        data=data,
        primary_visual_modality='rgb',
        max_time_gap=5e7  # 50ms max gap
    )

    rgb_data = aligner.data['rgb']
    
    gaze_data = aligner.data['personalized_gaze']
    
    imu_right_data = aligner.data['imu_right']
    
    imu_left_data = aligner.data['imu_left']

    hand_landmarks_data = aligner.data['hand_landmarks']

    # gps_data = aligner.data['gps']
    
    et_frames_data = aligner.data['et']

    slam_left_data = aligner.data['slam_left']

    slam_right_data = aligner.data['slam_right']
    
    
    

    # Add modalities
    aligner.add_modality('rgb', rgb_data)
    aligner.add_modality('gaze', gaze_data)
    aligner.add_modality('imu_right', imu_right_data)
    aligner.add_modality('imu_left', imu_left_data)
    aligner.add_modality('hand_landmarks', hand_landmarks_data)
    # aligner.add_modality('gps', gps_data)
    aligner.add_modality('et', et_frames_data)
    aligner.add_modality('slam_left', slam_left_data)
    aligner.add_modality('slam_right', slam_right_data)
    
    # # Align data
    aligned_dataset = aligner.align()

    np.save('/Volumes/MichaelSSD/dataset/Compiled Data (.npy)/Drive8_aligned.npy', aligned_dataset, allow_pickle=True)

    aligner.print_summary()
    
    # Extract training windows
    # print(f"\n=== TRAINING WINDOW EXAMPLES ===")
    
    
    
    # Example 2: Sliding windows for training
    # training_windows = aligner.get_training_windows(
    #     window_size=16,    # 16 frames per window (~1 second at 15 FPS)
    #     stride=16,          # 50% overlap
    #     min_valid_ratio=0.8  # Require 80% valid data
    # )
    
    # print(f"Generated {len(training_windows)} training windows")
    
    # # Example 3: Export for training
    # numpy_data = aligner.export_for_training('numpy')
    # print(f"\nExported shapes:")
    # for mod_name, data in numpy_data.items():
    #     if isinstance(data, np.ndarray):
    #         print(f"  {mod_name}: {data.shape}")
    #     else:
    #         print(f"  {mod_name}: {len(data)} samples")


    
    # # Show sample window data
    # print(f"\nSample training window:")
    # sample_window = windows[0]
    # for mod_name, mod_data in sample_window.items():
    #     if mod_name not in ['frame_indices', 'frame_timestamps', 'window_duration']:
    #         print(f"  {mod_name}: {len(mod_data)} samples")
    #     else:
    #         print(f"  {mod_name}: {mod_data}")