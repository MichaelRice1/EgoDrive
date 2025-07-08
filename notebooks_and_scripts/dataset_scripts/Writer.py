from Aligner import EgoDriveAriaAligner
from EgoDriveAriaDataset import EgoDriveAriaDataset
import os
import sys
import numpy as np
sys.path.append('MSc_AI_Thesis/notebooks_and_scripts/')
from DataExtractionMain import DataProcessor

class Writer:
    def __init__(self,folder_path):

        self.folder_path = folder_path
        

    def write(self, folder_path, completed_folders=None):

        folders = os.listdir(folder_path)
        

        for folder in folders:
            if folder in completed_folders:
                print(f'Skipping {folder} as it has already been processed.')
                continue
            if folder.startswith('Drive'):
            
                folder_path = os.path.join(self.folder_path, folder)
                vrs_path = os.path.join(folder_path, f'{folder}.vrs')
                dp = DataProcessor()
                
                data_path = os.path.join(folder_path, f'{folder}.npy')
                if os.path.exists(data_path):
                    data = np.load(data_path, allow_pickle=True).item()
                else:
                    data = dp.vrs_processing(vrs_path)
                
                aligner = EgoDriveAriaAligner(data=data, primary_visual_modality='rgb', max_time_gap=5e7)
                aligned_data = aligner.align_drive()
                del data

                annotations_path = os.path.join(folder_path, 'actions.csv')
                dataset_creator = EgoDriveAriaDataset(aligned_data, image_size=224, frames_per_clip=32, annotations_path=annotations_path)
                dataset_creator.process_folder(aligned_data, annotations_path)

                # Flush data from mem
                
                del aligned_data
                del dataset_creator
                del aligner




if __name__ == "__main__":
    folder_path = '/Volumes/MichaelSSD/dataset/RawData(VRS&MPS)'
    written_folders = []
    writer = Writer(folder_path)
    writer.write(folder_path, written_folders)

            
        

    