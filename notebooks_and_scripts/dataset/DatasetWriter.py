import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from notebooks_and_scripts.dataset.EgoDriveAria import EgoDriveAria
from notebooks_and_scripts.dataset.Aligner import Aligner


class DatasetWriter:
    def __init__(self, base_path, batch_size=32, window_size=1.0, reference_modality='rgb'):
        """
        Initialize the DatasetWriter.

        Args:
            base_path (str): Path to the base directory containing the dataset folders.
            batch_size (int): Batch size for the PyTorch DataLoader.
            window_size (float): Window size for data alignment.
            reference_modality (str): Reference modality for alignment.
        """
        self.base_path = base_path
        self.batch_size = batch_size
        self.window_size = window_size
        self.reference_modality = reference_modality
        self.folders = sorted(os.listdir(base_path))
        self.dataloaders = {}

    def align_and_save_dataloader(self, folder):
        """
        Align data and save the dataloader for a specific folder.

        Args:
            folder (str): Folder name to process.

        Returns:
            torch.utils.data.DataLoader: The created dataloader.
        """
        actions_csv_path = os.path.join(self.base_path, folder, 'actions.csv')
        blur_csv_path = os.path.join(self.base_path, folder, 'blur.csv')
        npy_path = os.path.join(self.base_path, folder, f"{folder}.npy")

        if not (os.path.exists(actions_csv_path) and os.path.exists(blur_csv_path) and os.path.exists(npy_path)):
            print(f"Missing files for {folder}. Skipping...")
            return None

        print(f"All files found for {folder}. Processing...")

        # Load data
        data = np.load(npy_path, allow_pickle=True).item()
        modality_data = {
            'rgb': (list(data['rgb'].keys()), list(data['rgb'].values())),
            'et': (list(data['et'].keys()), list(data['et'].values())),
            'gaze': (list(data['personalized_gaze'].keys()), list(data['personalized_gaze'].values())),
            'imu': (list(data['imu'].keys()), list(data['imu'].values())),
            'hand': (list(data['hand'].keys()), list(data['hand'].values()))
        }

        # Align data
        aligner = Aligner(window_size=self.window_size, reference_modality=self.reference_modality)
        aligned_windows = aligner.create_aligned_windows(modality_data)

        # Create PyTorch dataset and dataloader
        dataset = EgoDriveAria(aligned_windows)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Save dataset
        torch.save(dataset, os.path.join(self.base_path, folder, 'aligned_dataset.pth'))
        print(f"Dataloader for folder '{folder}' ready with {len(dataset)} samples.")
        return dataloader

    def process_all_folders(self):
        """
        Process all folders in the base path to create and save dataloaders.
        """
        for folder in self.folders:
            dataloader = self.align_and_save_dataloader(folder)
            if dataloader:
                self.dataloaders[folder] = dataloader

        print("\nAll dataloaders are ready. Ready for training multimodal transformer!")


if __name__ == "__main__":
    base_path = '/Users/michaelrice/Documents/GitHub/Thesis/MSc_AI_Thesis/train'
    dataset_writer = DatasetWriter(base_path)
    dataset_writer.process_all_folders()
