import numpy as np
import os
import torch
from torch.utils.data import Dataset, random_split
from typing import Dict, Optional
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import wandb
from trainer import EgoDriveClassifier

class AriaMultimodalTrainDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.actions = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
        self.file_paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.npy'):
                    self.file_paths.append(os.path.join(root, file))
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_name = self.file_paths[idx]
        file_path = os.path.join(self.root_dir, file_name)

        data = np.load(file_path, allow_pickle=True).item()

        # Normalize frames to [0, 1] range
        frames = np.array(data['frames']).astype(np.float32) / 255.0

        sample = {
                    'frames': torch.from_numpy(frames),
                    'gaze': torch.from_numpy(np.array(data['gaze'])).float(),
                    'hands': torch.from_numpy(np.array(data['hands'])).float(),
                    'imu': torch.from_numpy(np.array(data['imu'])).float(),
                    'object_detections': torch.from_numpy(np.array(data['object_detections'])).float(),
                    'label_id': torch.tensor(data['label_id'])
                }

        if self.transform:
            sample = self.transform(sample)

        return sample
    




def dict_transform(x):
    return {
        'frames': x['frames'], 
        'gaze': x['gaze'],
        'hands': x['hands'],
        'imu': x['imu'],
        'objects': x['object_detections'],
        'label': x['label_id']
    }




def calculate_class_weights(dataset, num_classes=6):
    """Calculate class weights for imbalanced dataset"""
    class_counts = torch.zeros(num_classes)
    
    for i in range(len(dataset)):
        sample = dataset[i]
        class_counts[sample['label'].item()] += 1
    
    print(f"Class counts: {class_counts}")

    # Calculate weights (inverse frequency)
    total = class_counts.sum()
    class_weights = total / (num_classes * class_counts)
    
    # Normalize weights
    class_weights = class_weights / class_weights.mean()
    
    return class_weights

def load_data(dataset):
  class_weights = calculate_class_weights(dataset)

  total_size = len(dataset)
  train_size = int(0.6 * total_size)
  val_size = int(0.2 * total_size)
  test_size = total_size - train_size - val_size

  # Split dataset
  train_set, val_set, test_set = random_split(
      dataset, 
      [train_size, val_size, test_size],
      generator=torch.Generator().manual_seed(13)
  )
  
  return train_set, val_set, test_set, class_weights 



class ModelWrapper(nn.Module):
    """
    Simple wrapper to make your existing EgoDriveMultimodalTransformer 
    compatible with the training phases
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, inputs: Dict[str, torch.Tensor], labels: Optional[torch.Tensor] = None, 
                training_phase: str = 'multimodal'):
        """
        Wrapper forward that handles training_phase parameter
        """
        # For now, just ignore training_phase and use standard forward
        # This allows your existing model to work with the new training code
        
        # Call the base model's forward (which expects the dict input)
        outputs = self.base_model(inputs)
        
        # If your model returns a tensor, wrap it in a dict
        if isinstance(outputs, torch.Tensor):
            outputs = {'logits': outputs}
        
        # Add loss computation if labels are provided
        if labels is not None and 'logits' in outputs:
            outputs['loss'] = F.cross_entropy(outputs['logits'], labels)
            
            # Add accuracy
            with torch.no_grad():
                predictions = outputs['logits'].argmax(dim=-1)
                outputs['accuracy'] = (predictions == labels).float().mean()
        
        return outputs
    


def train_single_phase(
    model,
    train_loader,
    val_loader,
    test_loader=None,
    phase='multimodal',
    epochs=10,
    project_name="egodrive-multimodal",
    run_name=None,
    class_weights=None
):
    """
    Train a single phase (useful for debugging or fine-tuning)
    """
    
    
    # Initialize wandb
    wandb.init(
        project=project_name,
        name=run_name or f"{phase}_training",
        config={
            "phase": phase,
            "epochs": epochs,
            "num_classes": 6,
            "batch_size": train_loader.batch_size
        }
    )
    
    # Create logger
    wandb_logger = WandbLogger(project=project_name, log_model=True)
    
    # Create classifier
    classifier = EgoDriveClassifier(
        model=model,
        num_classes=6,
        lr={'multimodal': 1e-4, 'hallucination': 5e-5, 'rgb_only': 1e-5}[phase],
        use_loss_weight=True,
        loss_weight=class_weights.to('cuda' if torch.cuda.is_available() else 'mps'),
        training_phase=phase
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath=f'checkpoints/{phase}',
        filename=f'{phase}_{{epoch:02d}}_{{val_acc:.3f}}',
        save_top_k=1,
        mode='max'
    )

    early_stopping_callback = EarlyStopping(
    monitor='val_loss',  
    patience=5,             
    mode='min',             
    verbose=True
    )

    trainer = L.Trainer(
        max_epochs=epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator='gpu' if torch.cuda.is_available() else 'mps',
        devices=1,
        log_every_n_steps=10
    )

    
    # Train
    trainer.fit(
        model=classifier,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    
    # Test if provided
    if test_loader is not None:
        trainer.test(
            model=classifier,
            dataloaders=test_loader,
            ckpt_path='best'
        )
    
    wandb.finish()
    
    return checkpoint_callback.best_model_path