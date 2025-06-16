import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

@dataclass
class ActionSequence:
    """Single training sequence with multi-modal data and labels"""
    rgb: List[str]  # Frame paths or identifiers
    gaze: List[np.ndarray]  # Gaze coordinates [x, y]
    imu_right: List[Dict[str, np.ndarray]]  # IMU readings
    imu_left: List[Dict[str, np.ndarray]]  # IMU readings for left hand
    hand_landmarks: List[np.ndarray]  # Hand keypoints
    labels: List[str]  # Multiple action labels for this sequence
    start_frame: int
    end_frame: int
    duration: float
    sequence_id: str

class MultiModalActionDataset:
    """
    Research-grade dataset for multi-modal action recognition.
    
    Best singular strategy: Variable-length sequences with multi-label classification
    using natural annotation boundaries and temporal context.
    """
    
    def __init__(self, aligned_data_path: str = None,annotations_file: str = None, action_taxonomy: Dict[str, List[str]] = None):
        """
        Args:
            aligned_data_path: Path to saved aligned dataset (from aligner.export_for_training())
            aligner: Alternative - pass trained ActionRecognitionAligner directly
            annotations_file: CSV with columns [start_frame, end_frame, action]
            action_taxonomy: Optional hierarchical action grouping
        """
        if aligned_data_path is not None:
            # Load from saved aligned data
            import pickle
            with open(aligned_data_path, 'rb') as f:
                self.aligned_data = np.load(f, allow_pickle=True).item()
            self.aligner = None  # No aligner object, just the data
            print(f"Loaded aligned data from {aligned_data_path}")
        elif aligner is not None:
            # Use aligner object directly
            self.aligner = aligner
            self.aligned_data = aligner.aligned_data
            print("Using provided aligner object")
        else:
            raise ValueError("Must provide either aligned_data_path or aligner")
            
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
        
        # Generate training sequences
        self.sequences = self._build_sequences()
        
        print(f"Dataset created: {len(self.sequences)} sequences, {self.num_classes} action classes")
        self._print_dataset_stats()
    
    def _build_sequences(self) -> List[ActionSequence]:
        """Build training sequences from annotations with overlapping action handling"""
        sequences = []
        
        # Group overlapping annotations
        annotation_groups = self._group_overlapping_annotations()
        
        for group_id, (start_frame, end_frame, actions) in enumerate(annotation_groups):
            # Validate frame range
            if end_frame >= self.aligned_data['frame_count']:
                continue
            
            # Extract multi-modal data for this sequence
            window_data = self._get_window_from_saved_data(start_frame, end_frame)
            
            # Check data quality - require minimum valid data
            if not self._is_sequence_valid(window_data):
                continue
            
            # Create sequence object
            sequence = ActionSequence(
                rgb=window_data.get('rgb', []),
                gaze=self.get_gaze(window_data.get('gaze', [])),
                imu_right =self.get_imu(window_data.get('imu_right', [])),
                imu_left = self.get_imu(window_data.get('imu_left', [])),
                hand_landmarks=self.get_hand_landmarks(window_data.get('hand_landmarks', [])),
                labels=actions,
                start_frame=start_frame,
                end_frame=end_frame,
                duration=window_data['window_duration'],
                sequence_id=f"seq_{group_id:04d}"
            )
            
            sequences.append(sequence)
        
        return sequences
    
    def _group_overlapping_annotations(self):
        """Group overlapping annotations into unified sequences and split long segments"""
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
                if current_start is not None:
                    groups.extend(self._split_long_segment(current_start, current_end, current_actions))
                current_start = start
                current_end = end
                current_actions = [action]
        
        # Add final group
        if current_start is not None:
            groups.extend(self._split_long_segment(current_start, current_end, current_actions))
        
        return groups

    def _split_long_segment(self, start_frame, end_frame, actions):
        """Split a segment into smaller chunks if it exceeds 50 frames"""
        segments = []
        while end_frame - start_frame > 50:
            segments.append((start_frame, start_frame + 25, actions.copy()))
            start_frame += 25
        segments.append((start_frame, end_frame, actions.copy()))
        return segments
    
    def _get_window_from_saved_data(self, start_frame: int, end_frame: int):
        """Extract window from saved aligned data (when no aligner object available)"""
        if self.aligner is not None:
            # Use aligner method if available
            return self.aligner.get_window(start_frame, end_frame)
        
        # Extract from saved data directly
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
    
    def get_gaze(self, gaze_data) :
        """Extract gaze coordinates from aligned data,
        given in {'projection': [np.float64(X), np.float64(Y)], 'depth': D} 
        """
        gaze_coords = []

        for frame in gaze_data:
            if frame is None:
                # Handle missing frame data
                gaze_coords.append(np.array([0.0, 0.0], dtype=np.float32))
                continue
            if 'projection' in frame:
                if frame['projection'] is None:
                    # Handle missing projection data
                    gaze_coords.append(np.array([0.0, 0.0], dtype=np.float32))
                    continue
                x, y = frame['projection']
                gaze_coords.append(np.array([x, y], dtype=np.float32))
            else:
                # Handle missing or malformed gaze data
                gaze_coords.append(np.array([0.0, 0.0], dtype=np.float32))
        return gaze_coords

    def get_imu(self, imu_data) :
        """Extract IMU readings from aligned data,
        given in list [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, 0.0, 0.0, 0.0]
        return np array of shape (n,6) for each frame
        """
        imu_readings = []

        for frame in imu_data:
            imu_readings.append(np.array(frame[0:6], dtype=np.float32))  # Extract first 6 values (accel + gyro)

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
    
    def _is_sequence_valid(self, window_data: Dict, min_valid_ratio: float = 0.7) -> bool:
        """Check if sequence has sufficient valid data"""
        total_frames = len(window_data.get('rgb', []))
        if total_frames < 15:  # Minimum sequence length
            return False
        
        # Check each modality for sufficient coverage
        for modality in ['gaze', 'imu']:
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
    
    def create_pytorch_dataset(self, context_frames: int = 1) -> 'PyTorchActionDataset':
        """Create PyTorch-compatible dataset"""
        return PyTorchActionDataset(self, context_frames=context_frames)
    
    def _print_dataset_stats(self):
        """Print dataset statistics"""
        print(f"\nDataset Statistics:")
        print(f"  Total sequences: {len(self.sequences)}")
        
        # Length distribution
        lengths = [seq.end_frame - seq.start_frame for seq in self.sequences]
        print(f"  Sequence lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
        
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

class PyTorchActionDataset(Dataset):
    """PyTorch Dataset wrapper for training"""
    
    def __init__(self, action_dataset: MultiModalActionDataset, context_frames: int = 30):
        self.dataset = action_dataset
        self.context_frames = context_frames
    
    def __len__(self):
        return len(self.dataset.sequences)
    
    def __getitem__(self, idx):
        sequence = self.dataset.sequences[idx]
        
        # Convert to tensors
        gaze_tensor = torch.stack([torch.from_numpy(g) for g in sequence.gaze])

        imu_right_tensor = torch.stack([torch.from_numpy(imu) for imu in sequence.imu_right])
        imu_left_tensor = torch.stack([torch.from_numpy(imu) for imu in sequence.imu_left]) if sequence.imu_left else None
        
        # Hand landmarks
        hand_tensor = torch.stack([torch.from_numpy(h) for h in sequence.hand_landmarks])
        
        # Multi-label target
        target = torch.from_numpy(self.dataset.get_multi_label_vector(sequence.labels))
        
        return {
            'gaze': gaze_tensor,
            'imu_right': imu_right_tensor,  # Use right hand IMU data
            'imu_left': imu_left_tensor,  # Optional: include left hand IMU if needed
            'hand_landmarks': hand_tensor,
            'target': target,
            'length': len(sequence.gaze),
            'sequence_id': sequence.sequence_id,
            'actions': sequence.labels
        }

def collate_variable_length(batch):
    """Custom collate function for variable-length sequences"""
    # Separate each modality
    gaze_batch = [item['gaze'] for item in batch]
    imu_right_batch = [item['imu_right'] for item in batch]
    imu_left_batch = [item['imu_left'] for item in batch]
    hands_batch = [item['hand_landmarks'] for item in batch]
    targets = torch.stack([item['target'] for item in batch])
    lengths = torch.tensor([item['length'] for item in batch])
    
    # Pad sequences to max length in batch
    gaze_padded = pad_sequence(gaze_batch, batch_first=True, padding_value=0)
    imu_right_padded = pad_sequence(imu_right_batch, batch_first=True, padding_value=0)
    imu_left_padded = pad_sequence(imu_left_batch, batch_first=True, padding_value=0)
    hands_padded = pad_sequence(hands_batch, batch_first=True, padding_value=0)
    
    return {
        'gaze': gaze_padded,
        'imu_right': imu_right_padded,
        'imu_left': imu_left_padded, 
        'hands': hands_padded,
        'target': targets,
        'lengths': lengths,
        'sequence_ids': [item['sequence_id'] for item in batch],
        'actions': [item['actions'] for item in batch]
    }

# Example usage
def create_training_dataset(aligned_data_path: str = None,  annotations_csv: str = None):
    """Complete pipeline to create training-ready dataset"""
    
    # Create dataset from saved data or aligner
    dataset = MultiModalActionDataset(
        aligned_data_path=aligned_data_path,
        annotations_file=annotations_csv
    )

    
    # Convert to PyTorch
    pytorch_dataset = dataset.create_pytorch_dataset()
    np.save('/Volumes/MichaelSSD/dataset/Compiled Data (.npy)/Drive1_dataset.npy', pytorch_dataset, allow_pickle=True)

    
    # Create data loader with custom collate function
    train_loader = DataLoader(
        pytorch_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_variable_length,
        num_workers=0
    )

    print(f"Dataset length: {len(pytorch_dataset)}")  # Should be > 0


    # for i, batch in enumerate(train_loader):
    #     print(i)
    #     torch.save(batch, f'/Volumes/MichaelSSD/dataset/Compiled Data (.npy)/Drive1/batch_{i}.pt')
        


    
    return dataset, train_loader

if __name__ == "__main__":
    
    
    print("Multi-Modal Action Recognition Dataset Implementation")
    print("Key features:")
    print("- Variable-length sequences using natural annotation boundaries")
    print("- Multi-label classification for overlapping actions")
    print("- Handles missing modality data gracefully")
    print("- PyTorch-ready with custom collate function")
    print("- Preserves temporal structure for action recognition")
    print("- Works with saved aligned data or live aligner objects")
    

    dataset = create_training_dataset(aligned_data_path='/Volumes/MichaelSSD/dataset/Compiled Data (.npy)/Drive1_aligned.npy', annotations_csv='/Users/michaelrice/Documents/GitHub/Thesis/MSc_AI_Thesis/data/drives/Drive1/actions.csv')


def save_aligned_data_for_training(aligner, save_path: str):
    """Helper function to save aligned data for later use"""
    import pickle
    
    # Export training-ready format
    training_data = aligner.export_for_training('numpy')
    
    # Add metadata needed for dataset creation
    training_data['frame_count'] = aligner.aligned_data['frame_count']
    training_data['stats'] = aligner.aligned_data['stats']
    
    with open(save_path, 'wb') as f:
        pickle.dump(training_data, f)
    
    print(f"Aligned data saved to {save_path}")
    print(f"Load later with: dataset = MultiModalActionDataset(aligned_data_path='{save_path}', annotations_file='annotations.csv')")