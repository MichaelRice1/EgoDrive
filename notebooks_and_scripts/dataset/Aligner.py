import numpy as np
from scipy import interpolate

class Aligner:
    def __init__(self, window_size=1.0, reference_modality='rgb'):
        """
        Initialize multimodal aligner
        
        Args:
            window_size: Duration in seconds for temporal windows
            reference_modality: Which modality to use as temporal reference
        """
        self.window_size = window_size
        self.reference_modality = reference_modality
        
    def convert_timestamps_to_seconds(self, timestamps):
        """Convert timestamps to seconds (assuming they're in some time format)"""
        # Convert to numpy array and normalize
        ts_array = np.array([float(ts) for ts in timestamps])
        # If timestamps are in microseconds, divide by 1e6
        if ts_array.max() > 1e9:  # Likely microseconds or nanoseconds
            ts_array = ts_array / 1e6  # Convert to seconds
        return ts_array
    
    def analyze_sampling_rates(self, modality_data):
        """Analyze sampling rates for each modality"""
        rates = {}
        
        for modality, (timestamps, data) in modality_data.items():
            ts_seconds = self.convert_timestamps_to_seconds(timestamps)
            duration = ts_seconds[-1] - ts_seconds[0]
            sample_count = len(timestamps)
            estimated_hz = sample_count / duration if duration > 0 else 0
            
            # Calculate actual intervals
            intervals = np.diff(ts_seconds)
            avg_interval = np.mean(intervals)
            actual_hz = 1.0 / avg_interval if avg_interval > 0 else 0
            
            rates[modality] = {
                'estimated_hz': estimated_hz,
                'actual_hz': actual_hz,
                'sample_count': sample_count,
                'duration': duration,
                'avg_interval': avg_interval,
                'timestamps': ts_seconds
            }
            
            print(f"{modality}: {actual_hz:.2f} Hz ({sample_count} samples over {duration:.2f}s)")
        
        return rates
    
    def create_aligned_windows(self, modality_data, overlap_ratio=0.0):
        """
        Create temporally aligned windows across all modalities
        
        Args:
            modality_data: Dict with {modality: (timestamps, data)}
            overlap_ratio: Overlap between consecutive windows (0.0 = no overlap)
        
        Returns:
            List of aligned windows
        """
        # Convert all timestamps to seconds
        converted_data = {}
        for modality, (timestamps, data) in modality_data.items():
            converted_data[modality] = {
                'timestamps': self.convert_timestamps_to_seconds(timestamps),
                'data': np.array(data)
            }
        
        # Find common time range
        start_times = [converted_data[mod]['timestamps'][0] for mod in converted_data.keys()]
        end_times = [converted_data[mod]['timestamps'][-1] for mod in converted_data.keys()]
        
        common_start = max(start_times)
        common_end = min(end_times)
        
        print(f"Common time range: {common_start:.2f} to {common_end:.2f} seconds")
        print(f"Total duration: {common_end - common_start:.2f} seconds")
        
        # Use reference modality timestamps as anchors
        ref_timestamps = converted_data[self.reference_modality]['timestamps']
        ref_mask = (ref_timestamps >= common_start) & (ref_timestamps <= common_end)
        ref_timestamps = ref_timestamps[ref_mask]
        
        aligned_windows = []
        
        for ref_time in ref_timestamps:
            # Skip if window would extend beyond common range
            if ref_time - self.window_size/2 < common_start or \
               ref_time + self.window_size/2 > common_end:
                continue
                
            window = {
                'reference_time': ref_time,
                'window_start': ref_time - self.window_size/2,
                'window_end': ref_time + self.window_size/2
            }
            
            # Extract data for each modality within the window
            for modality, mod_data in converted_data.items():
                timestamps = mod_data['timestamps']
                data = mod_data['data']
                
                # Find samples within window
                mask = (timestamps >= window['window_start']) & \
                       (timestamps <= window['window_end'])
                
                window_data = data[mask]
                window_timestamps = timestamps[mask]
                
                # Convert to relative timestamps (relative to window center)
                relative_timestamps = window_timestamps - ref_time
                
                window[modality] = {
                    'data': window_data,
                    'timestamps': window_timestamps,
                    'relative_timestamps': relative_timestamps,
                    'count': len(window_data)
                }
            
            aligned_windows.append(window)
        
        print(f"Created {len(aligned_windows)} aligned windows")
        return aligned_windows
    
    def interpolate_to_fixed_rate(self, modality_data, target_hz):
        """
        Interpolate all modalities to a fixed sampling rate
        
        Args:
            modality_data: Dict with {modality: (timestamps, data)}
            target_hz: Target sampling rate in Hz
        
        Returns:
            Dict with interpolated data at fixed rate
        """
        interpolated_data = {}
        
        # Find common time range
        all_timestamps = []
        for modality, (timestamps, data) in modality_data.items():
            ts_seconds = self.convert_timestamps_to_seconds(timestamps)
            all_timestamps.extend([ts_seconds[0], ts_seconds[-1]])
        
        common_start = max([ts[0] for ts in all_timestamps[::2]])
        common_end = min([ts[1] for ts in all_timestamps[1::2]])
        
        # Create target timestamp grid
        duration = common_end - common_start
        num_samples = int(duration * target_hz)
        target_timestamps = np.linspace(common_start, common_end, num_samples)
        
        print(f"Interpolating to {target_hz} Hz ({num_samples} samples)")
        
        for modality, (timestamps, data) in modality_data.items():
            ts_seconds = self.convert_timestamps_to_seconds(timestamps)
            data_array = np.array(data)
            
            # Handle different data shapes
            if len(data_array.shape) == 1:
                # 1D data - direct interpolation
                f = interpolate.interp1d(ts_seconds, data_array, 
                                       kind='linear', bounds_error=False, 
                                       fill_value='extrapolate')
                interpolated_values = f(target_timestamps)
            else:
                # Multi-dimensional data - interpolate each dimension
                interpolated_values = np.zeros((len(target_timestamps), *data_array.shape[1:]))
                
                for i in range(data_array.shape[1]):
                    if len(data_array.shape) == 2:
                        f = interpolate.interp1d(ts_seconds, data_array[:, i], 
                                               kind='linear', bounds_error=False, 
                                               fill_value='extrapolate')
                        interpolated_values[:, i] = f(target_timestamps)
                    # Add more dimensions as needed
            
            interpolated_data[modality] = {
                'data': interpolated_values,
                'timestamps': target_timestamps,
                'original_hz': len(timestamps) / (ts_seconds[-1] - ts_seconds[0]),
                'target_hz': target_hz
            }
        
        return interpolated_data