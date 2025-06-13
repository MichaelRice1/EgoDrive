import torch



class EgoDriveAria(torch.utils.data.Dataset):
    def __init__(self, aligned_windows, transform=None):
        self.aligned_windows = aligned_windows
        self.transform = transform
    
    def __len__(self):
        return len(self.aligned_windows)
    
    def __getitem__(self, idx):
        window = self.aligned_windows[idx]
        
        # Extract data for each modality
        sample = {
            'rgb': torch.FloatTensor(window['rgb']['data']),
            'et': torch.FloatTensor(window['et']['data']) if window['et']['count'] > 0 else torch.zeros(1),
            'gaze': torch.FloatTensor(window['gaze']['data']) if window['gaze']['count'] > 0 else torch.zeros(1),
            'imu': torch.FloatTensor(window['imu']['data']) if window['imu']['count'] > 0 else torch.zeros(1),
            'hand': torch.FloatTensor(window['hand']['data']) if window['hand']['count'] > 0 else torch.zeros(1),
            'timestamps': {
                'rgb': torch.FloatTensor(window['rgb']['relative_timestamps']),
                'et': torch.FloatTensor(window['et']['relative_timestamps']),
                'gaze': torch.FloatTensor(window['gaze']['relative_timestamps']),
                'imu': torch.FloatTensor(window['imu']['relative_timestamps']),
                'hand': torch.FloatTensor(window['hand']['relative_timestamps'])
            },
            'reference_time': window['reference_time']
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample