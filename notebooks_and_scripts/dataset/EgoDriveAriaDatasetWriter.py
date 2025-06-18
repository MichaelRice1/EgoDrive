import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import cv2

@dataclass
class ActionSequence:
    """Single training sequence with multi-modal data and labels"""
    rgb: List[str]  # Frame paths or identifiers
    gaze: List[np.ndarray]  # Gaze coordinates [x, y]
    imu_right: List[np.ndarray]  # IMU readings
    imu_left: List[np.ndarray]  # IMU readings for left hand
    hand_landmarks: List[np.ndarray]  # Hand keypoints
    labels: List[str]  # Multiple action labels for this sequence
    start_frame: int
    end_frame: int
    duration: float
    sequence_id: str

class FixedLengthActionDataset:
    """
    Fixed-length dataset for multi-modal action recognition.
    
    Extracts fixed-size windows based on action boundaries for simpler training.
    Optimized for circular Project Aria glasses images.
    """
    
    def __init__(self, aligned_data_path: str = None, annotations_file: str = None, 
                 window_size: int = 64, image_size: int = 224, 
                 action_taxonomy: Dict[str, List[str]] = None):
        """
        Args:
            aligned_data_path: Path to saved aligned dataset
            annotations_file: CSV with columns [start_frame, end_frame, action]
            window_size: Fixed window size for all sequences
            image_size: Target size for preprocessed circular images (224, 320, or 448)
            action_taxonomy: Optional hierarchical action grouping
        """
        self.window_size = window_size
        self.image_size = image_size
        
        if aligned_data_path is not None:
            # Load from saved aligned data
            self.aligned_data = np.load(aligned_data_path, allow_pickle=True).item()
            self.aligner = None
            print(f"Loaded aligned data from {aligned_data_path}")
        else:
            raise ValueError("Must provide aligned_data_path")
            
        if annotations_file:
            self.annotations = pd.read_csv(annotations_file)
        else:
            raise ValueError("annotations_file is required")
        
        # Build action vocabulary
        self.action_vocab = sorted(self.annotations['action'].unique())
        self.action_to_idx = {action: idx for idx, action in enumerate(self.action_vocab)}
        self.num_classes = len(self.action_vocab)
        
        # Action taxonomy for hierarchical labels (optional)
        self.action_taxonomy = action_taxonomy or {}
        
        # Generate fixed-length training sequences
        self.sequences = self._build_fixed_sequences()
        
        print(f"Dataset created: {len(self.sequences)} sequences, {self.num_classes} action classes")
        print(f"Fixed window size: {self.window_size} frames")
        print(f"Image preprocessing: {self.image_size}x{self.image_size} (optimized for circular Aria frames)")
        self._print_dataset_stats()
    
    def __len__(self):
        """Number of training sequences"""
        return len(self.sequences)
    
    def _build_fixed_sequences(self) -> List[ActionSequence]:
        """Build fixed-length training sequences from annotations"""
        sequences = []
        
        # Group overlapping annotations first
        annotation_groups = self._group_overlapping_annotations()
        
        for group_id, (start_frame, end_frame, actions) in enumerate(annotation_groups):
            # Validate frame range
            if end_frame >= self.aligned_data['frame_count']:
                continue
            
            # Extract fixed window based on action type and length
            window_start, window_end = self._get_action_window(start_frame, end_frame, actions)
            
            # Validate window
            if window_end >= self.aligned_data['frame_count'] or window_start < 0:
                continue
            
            # Extract multi-modal data for this fixed window
            window_data = self._get_window_from_saved_data(window_start, window_end)
            
            # Check data quality
            if not self._is_sequence_valid(window_data):
                continue
            
            # Create sequence object
            sequence = ActionSequence(
                rgb=window_data.get('rgb', []),
                gaze=self.get_gaze(window_data.get('gaze', [])),
                imu_right=self.get_imu(window_data.get('imu_right', [])),
                imu_left=self.get_imu(window_data.get('imu_left', [])),
                hand_landmarks=self.get_hand_landmarks(window_data.get('hand_landmarks', [])),
                labels=actions,
                start_frame=window_start,
                end_frame=window_end,
                duration=window_data['window_duration'],
                sequence_id=f"seq_{group_id:04d}"
            )
            
            sequences.append(sequence)
        
        return sequences
    
    def _get_action_window(self, start_frame: int, end_frame: int, actions: List[str]) -> Tuple[int, int]:
        """Get optimal fixed window based on action type and boundary"""
        action_length = end_frame - start_frame
        
        # Determine extraction strategy based on action type
        primary_action = actions[0]  # Use first action for strategy
        
        if primary_action in ['checking left wing mirror', 'checking right wing mirror', 'checking rear view mirror']:
            # Mirror checks: center on action
            return self._center_window(start_frame, end_frame)
            
        elif primary_action in ['left turn', 'right turn']:
            # Turns: include lead-up and execution
            return self._action_with_context(start_frame, end_frame)
            
        elif primary_action == 'mobile phone usage':
            # Phone usage: sample from middle portion
            return self._sample_from_middle(start_frame, end_frame)
            
        elif primary_action == 'driving':
            # Normal driving: random sample from action
            return self._random_sample(start_frame, end_frame)
            
        else:
            # Default: center on action
            return self._center_window(start_frame, end_frame)
    
    def _center_window(self, start_frame: int, end_frame: int) -> Tuple[int, int]:
        """Center fixed window on action"""
        action_center = (start_frame + end_frame) // 2
        window_start = max(0, action_center - self.window_size // 2)
        window_end = window_start + self.window_size
        
        # Ensure we don't exceed data bounds
        if window_end >= self.aligned_data['frame_count']:
            window_end = self.aligned_data['frame_count'] - 1
            window_start = max(0, window_end - self.window_size)
            
        return window_start, window_end
    
    def _action_with_context(self, start_frame: int, end_frame: int) -> Tuple[int, int]:
        """Include context before action (for turns)"""
        context_frames = self.window_size // 4  # 25% context before action
        window_start = max(0, start_frame - context_frames)
        window_end = window_start + self.window_size
        
        if window_end >= self.aligned_data['frame_count']:
            window_end = self.aligned_data['frame_count'] - 1
            window_start = max(0, window_end - self.window_size)
            
        return window_start, window_end
    
    def _sample_from_middle(self, start_frame: int, end_frame: int) -> Tuple[int, int]:
        """Sample from middle portion of long action"""
        action_length = end_frame - start_frame
        if action_length <= self.window_size:
            return self._center_window(start_frame, end_frame)
        
        # Sample from middle third
        middle_start = start_frame + action_length // 3
        middle_end = end_frame - action_length // 3
        
        sample_center = (middle_start + middle_end) // 2
        window_start = max(0, sample_center - self.window_size // 2)
        window_end = window_start + self.window_size
        
        return window_start, window_end
    
    def _random_sample(self, start_frame: int, end_frame: int) -> Tuple[int, int]:
        """Random sample from action (for driving)"""
        action_length = end_frame - start_frame
        if action_length <= self.window_size:
            return self._center_window(start_frame, end_frame)
        
        # Random start within action
        max_start = end_frame - self.window_size
        window_start = np.random.randint(start_frame, max_start + 1)
        window_end = window_start + self.window_size
        
        return window_start, window_end
    
    def _group_overlapping_annotations(self):
        """Group overlapping annotations into unified sequences"""
        # Sort annotations by start frame
        sorted_annotations = self.annotations.sort_values('start_frame')
        
        groups = []
        current_start = None
        current_end = None
        current_actions = []
        
        for _, row in sorted_annotations.iterrows():
            start, end, action = row['start_frame'], row['end_frame'], row['action']
            
            if current_start is None:
                # First annotation
                current_start = start
                current_end = end
                current_actions = [action]
            elif start <= current_end:
                # Overlapping - extend current group
                current_end = max(current_end, end)
                if action not in current_actions:
                    current_actions.append(action)
            else:
                # No overlap - save current group and start new one
                groups.append((current_start, current_end, current_actions.copy()))
                current_start = start
                current_end = end
                current_actions = [action]
        
        # Add final group
        if current_start is not None:
            groups.append((current_start, current_end, current_actions))
        
        return groups
    
    def _get_window_from_saved_data(self, start_frame: int, end_frame: int):
        """Extract window from saved aligned data"""
        window_data = {}
        
        for mod_name, mod_data in self.aligned_data['modalities'].items():
            window_data[mod_name] = mod_data[start_frame:end_frame]
        
        # Add metadata
        window_data['frame_indices'] = list(range(start_frame, end_frame))
        window_data['frame_timestamps'] = self.aligned_data['frame_timestamps'][start_frame:end_frame].tolist()
        window_data['window_duration'] = (
            self.aligned_data['frame_timestamps'][end_frame-1] - 
            self.aligned_data['frame_timestamps'][start_frame]
        )
        
        return window_data
    
    def get_gaze(self, gaze_data):
        """Extract gaze coordinates from aligned data"""
        gaze_coords = []
        for frame in gaze_data:
            if frame is None:
                gaze_coords.append(np.array([0.0, 0.0], dtype=np.float32))
                continue
            if 'projection' in frame:
                if frame['projection'] is None:
                    gaze_coords.append(np.array([0.0, 0.0], dtype=np.float32))
                    continue
                x, y = frame['projection']
                gaze_coords.append(np.array([x, y], dtype=np.float32))
            else:
                gaze_coords.append(np.array([0.0, 0.0], dtype=np.float32))
        return gaze_coords

    def get_imu(self, imu_data):
        """Extract IMU readings from aligned data"""
        imu_readings = []
        for frame in imu_data:
            if frame is None or len(frame) < 6:
                imu_readings.append(np.zeros(6, dtype=np.float32))
            else:
                imu_readings.append(np.array(frame[0:6], dtype=np.float32))
        return imu_readings
    
    def get_hand_landmarks(self, hand_data: List[np.ndarray]) -> List[np.ndarray]:
        """Extract hand landmarks from aligned data,
        given in list of form {'left_wrist': array([x,y]), 'left_palm': array([x,y]), 
                               'right_wrist': array([x,y]), 'right_palm': array([x,y]),
                               'left_wrist_normal': array([x,y]), 'left_palm_normal': array([x,y]),
                               'right_wrist_normal': array([x,y]), 'right_palm_normal': array([x,y]),
                               'left_landmarks': [array([x,y]), array([x,y]), array([x,y]), 
                               array([x,y]), array([x,y]), array([x,y]), array([x,y]), array([x,y]), array([x,y]), array([x,y]), 
                               array([x,y]), array([x,y]), array([x,y]), array([x,y]), array([x,y]), array([x,y]), array([x,y]), 
                               array([x,y]), array([x,y]), array([x,y]), array([x,y])], 
                               'right_landmarks': [array([x,y]), array([x,y]), array([x,y]), array([x,y]), array([x,y]), array([x,y])
                               , array([x,y]), array([x,y]), array([x,y]), array([x,y]), array([x,y]), array([x,y]), array([x,y]), array([x,y]), 
                               array([x,y]), array([x,y]), array([x,y]), array([x,y]), array([x,y]), array([x,y]), array([x,y])]}
        """

        hand_landmarks = []

        for frame in hand_data:
            if frame is not None:
                # Extract landmarks if frame is valid
                left_wrist = np.array(frame.get('left_wrist', [0.0, 0.0]), dtype=np.float32).flatten()
                left_palm = np.array(frame.get('left_palm', [0.0, 0.0]), dtype=np.float32).flatten()
                right_wrist = np.array(frame.get('right_wrist', [0.0, 0.0]), dtype=np.float32).flatten()
                right_palm = np.array(frame.get('right_palm', [0.0, 0.0]), dtype=np.float32).flatten()

                # Ensure all arrays are 1-dimensional with 2 elements
                if left_wrist.shape == (2,) and left_palm.shape == (2,) and right_wrist.shape == (2,) and right_palm.shape == (2,):
                    # Combine left and right landmarks into one array
                    combined_points = np.concatenate([left_wrist, left_palm, right_wrist, right_palm])
                    hand_landmarks.append(combined_points.astype(np.float32))
                else:
                    # Handle malformed data
                    hand_landmarks.append(np.zeros(8, dtype=np.float32))  # 8 zeros for 4 points (x, y)
            else:
                # Handle missing frame data
                hand_landmarks.append(np.zeros(8, dtype=np.float32))  # 8 zeros for 4 points (x, y)

        return hand_landmarks

    def preprocess_aria_frame(self, image_path):
        """Preprocess circular Project Aria glasses frame"""
        import cv2
        
        try:
            # Load original image
            img = cv2.imread(image_path)
            if img is None:
                return torch.zeros(3, self.image_size, self.image_size)
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            
            # Find circular content
            center = (w // 2, h // 2)
            radius = self._find_circular_radius(img, center)
            
            # Crop to square around circle with padding
            crop_size = int(radius * 2.1)  # 10% padding
            x1 = max(0, center[0] - crop_size // 2)
            y1 = max(0, center[1] - crop_size // 2)
            x2 = min(w, x1 + crop_size)
            y2 = min(h, y1 + crop_size)
            
            # Ensure square crop
            crop_h = y2 - y1
            crop_w = x2 - x1
            if crop_h != crop_w:
                size = min(crop_h, crop_w)
                y2 = y1 + size
                x2 = x1 + size
            
            cropped = img[y1:y2, x1:x2]
            
            # Resize to target size
            resized = cv2.resize(cropped, (self.image_size, self.image_size))
            
            # Convert to tensor and normalize
            tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
            
            # Optional: Apply circular mask to remove corner artifacts
            if hasattr(self, 'apply_circular_mask') and self.apply_circular_mask:
                tensor = self._apply_circular_mask(tensor)
            
            return tensor
            
        except Exception as e:
            print(f"Error processing frame {image_path}: {e}")
            return torch.zeros(3, self.image_size, self.image_size)
    
    def _find_circular_radius(self, img, center):
        """Find radius of circular content in Aria frame"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Find non-black pixels (content)
        mask = gray > 10  # Threshold for non-black content
        
        if not np.any(mask):
            # Fallback if no content found
            return min(img.shape[:2]) // 4
        
        # Find maximum distance from center to content
        y_coords, x_coords = np.where(mask)
        distances = np.sqrt((x_coords - center[0])**2 + (y_coords - center[1])**2)
        
        # Use 95th percentile for robustness against noise
        return int(np.percentile(distances, 95))
    
    def _apply_circular_mask(self, tensor):
        """Apply circular mask to remove corner artifacts"""
        c, h, w = tensor.shape
        center = (h // 2, w // 2)
        radius = min(h, w) // 2
        
        # Create circular mask
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        distance = torch.sqrt((x - center[0])**2 + (y - center[1])**2)
        mask = (distance <= radius).float()
        
        # Apply mask to all channels
        for c_idx in range(c):
            tensor[c_idx] *= mask
        
        return tensor
        
    
    def _is_sequence_valid(self, window_data: Dict, min_valid_ratio: float = 0.7) -> bool:
        """Check if sequence has sufficient valid data"""
        total_frames = len(window_data.get('rgb', []))
        if total_frames < self.window_size // 2:  # At least half the window size
            return False
        
        # Check each modality for sufficient coverage
        for modality in ['gaze', 'imu_right']:
            if modality in window_data:
                valid_count = sum(1 for x in window_data[modality] if x is not None)
                if valid_count / total_frames < min_valid_ratio:
                    return False
        
        return True
    
    def get_sequence(self, idx: int) -> ActionSequence:
        """Get a single sequence by index"""
        return self.sequences[idx]
    
    def get_multi_label_vector(self, labels: List[str]) -> np.ndarray:
        """Convert action labels to multi-hot vector"""
        multi_hot = np.zeros(self.num_classes, dtype=np.float32)
        for label in labels:
            if label in self.action_to_idx:
                multi_hot[self.action_to_idx[label]] = 1.0
        return multi_hot
    
    def create_pytorch_dataset(self) -> 'PyTorchFixedLengthDataset':
        """Create PyTorch-compatible dataset"""
        return PyTorchFixedLengthDataset(self)
    
    def _print_dataset_stats(self):
        """Print dataset statistics"""
        print(f"\nDataset Statistics:")
        print(f"  Total sequences: {len(self.sequences)}")
        print(f"  All sequences have fixed length: {self.window_size} frames")
        
        # Action distribution
        action_counts = {}
        for seq in self.sequences:
            for action in seq.labels:
                action_counts[action] = action_counts.get(action, 0) + 1
        
        print(f"  Action distribution:")
        for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"    {action}: {count}")
        
        # Multi-label statistics
        multi_label_seqs = sum(1 for seq in self.sequences if len(seq.labels) > 1)
        print(f"  Multi-label sequences: {multi_label_seqs}/{len(self.sequences)} ({multi_label_seqs/len(self.sequences):.1%})")

class PyTorchFixedLengthDataset(Dataset):
    """PyTorch Dataset wrapper for fixed-length training"""
    
    def __init__(self, action_dataset: FixedLengthActionDataset):
        self.dataset = action_dataset
        self.window_size = action_dataset.window_size
    
    def __len__(self):
        return len(self.dataset.sequences)
    
    def __getitem__(self, idx):
        sequence = self.dataset.sequences[idx]
        
        # Convert to tensors - all sequences are exactly window_size length
        gaze_tensor = torch.stack([torch.from_numpy(g) for g in sequence.gaze])
        imu_right_tensor = torch.stack([torch.from_numpy(imu) for imu in sequence.imu_right])
        imu_left_tensor = torch.stack([torch.from_numpy(imu) for imu in sequence.imu_left]) if sequence.imu_left else torch.zeros_like(imu_right_tensor)
        hand_tensor = torch.stack([torch.from_numpy(h) for h in sequence.hand_landmarks])
        
        # RGB frames - already preprocessed as tensors
        rgb_tensor = torch.stack(sequence.rgb) if sequence.rgb else torch.zeros(self.window_size, 3, self.dataset.image_size, self.dataset.image_size)
        
        # Pad sequences if they're shorter than window_size (rare case)
        target_length = self.window_size
        
        if len(gaze_tensor) < target_length:
            pad_length = target_length - len(gaze_tensor)
            gaze_tensor = F.pad(gaze_tensor, (0, 0, 0, pad_length))
            imu_right_tensor = F.pad(imu_right_tensor, (0, 0, 0, pad_length))
            imu_left_tensor = F.pad(imu_left_tensor, (0, 0, 0, pad_length))
            hand_tensor = F.pad(hand_tensor, (0, 0, 0, pad_length))
            rgb_tensor = F.pad(rgb_tensor, (0, 0, 0, 0, 0, 0, 0, pad_length))
        
        # Ensure exact length
        gaze_tensor = gaze_tensor[:target_length]
        imu_right_tensor = imu_right_tensor[:target_length]
        imu_left_tensor = imu_left_tensor[:target_length]
        hand_tensor = hand_tensor[:target_length]
        rgb_tensor = rgb_tensor[:target_length]
        
        # Multi-label target
        target = torch.from_numpy(self.dataset.get_multi_label_vector(sequence.labels))
        
        return {
            'gaze': gaze_tensor,           # [window_size, 2]
            'imu_right': imu_right_tensor, # [window_size, 6]
            'imu_left': imu_left_tensor,   # [window_size, 6]
            'hand_landmarks': hand_tensor,  # [window_size, 8]
            'rgb': rgb_tensor,             # [window_size, 3, image_size, image_size]
            'target': target,              # [num_classes]
            'sequence_id': sequence.sequence_id,
            'actions': sequence.labels
        }

def simple_collate_fn(batch):
    """Simple collate function for fixed-length sequences - no padding needed!"""
    
    # Stack all tensors directly since they're all the same size
    gaze_batch = torch.stack([item['gaze'] for item in batch])
    imu_right_batch = torch.stack([item['imu_right'] for item in batch])
    imu_left_batch = torch.stack([item['imu_left'] for item in batch])
    hands_batch = torch.stack([item['hand_landmarks'] for item in batch])
    rgb_batch = torch.stack([item['rgb'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    
    return {
        'gaze': gaze_batch,        # [batch_size, window_size, 2]
        'imu_right': imu_right_batch,  # [batch_size, window_size, 6]
        'imu_left': imu_left_batch,    # [batch_size, window_size, 6]
        'hands': hands_batch,      # [batch_size, window_size, 8]
        'rgb': rgb_batch,          # [batch_size, window_size, 3, image_size, image_size]
        'target': targets,         # [batch_size, num_classes]
        'sequence_ids': [item['sequence_id'] for item in batch],
        'actions': [item['actions'] for item in batch]
    }

def create_fixed_length_training_dataset(aligned_data_path: str, annotations_csv: str, 
                                        window_size: int = 64, image_size: int = 224, 
                                        batch_size: int = 16):
    """Complete pipeline to create fixed-length training dataset"""
    
    # Create fixed-length dataset
    dataset = FixedLengthActionDataset(
        aligned_data_path=aligned_data_path,
        annotations_file=annotations_csv,
        window_size=window_size,
        image_size=image_size
    )
    
    # Convert to PyTorch
    pytorch_dataset = dataset.create_pytorch_dataset()
    print(f"Converted to PyTorch dataset with {len(pytorch_dataset)} sequences")
    
    # Create simple data loader - no complex collate function needed!
    train_loader = DataLoader(
        pytorch_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=simple_collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"Training loader created: {len(train_loader)} batches")
    
    # Example: iterate through one batch to show shapes
    # for batch in train_loader:
    #     print(f"Batch shapes:")
    #     for key, value in batch.items():
    #         if hasattr(value, 'shape'):
    #             print(f"  {key}: {value.shape}")
    #         else:
    #             print(f"  {key}: {len(value)} items")
    #     break
    
    return dataset, train_loader

if __name__ == "__main__":
    print("Fixed-Length Multi-Modal Action Recognition Dataset")
    print("Key features:")
    print("- Fixed-length sequences for simpler training")
    print("- Action-boundary aware window extraction")
    print("- Multi-label classification for overlapping actions")
    print("- No complex padding or collate functions needed")
    print("- Different extraction strategies per action type")
    
    # Example usage with your data
    dataset, train_loader = create_fixed_length_training_dataset(
        aligned_data_path='/Volumes/MichaelSSD/dataset/Compiled Data (.npy)/Drive1_aligned.npy',
        annotations_csv='/Users/michaelrice/Documents/GitHub/Thesis/MSc_AI_Thesis/data/drives/Drive1/actions.csv',
        window_size=64,
        image_size=224,  # Recommended for Aria glasses
        batch_size=8
    )