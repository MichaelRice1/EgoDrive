import numpy as np
import cv2
from torch.utils.data import Dataset
import os 
import pandas as pd 


class EgoDriveAriaDataset():

    def __init__(self, data, image_size, frames_per_clip, annotations_path=None):
        
        
        self.data = data
        self.image_size = image_size
        self.frames_per_clip = frames_per_clip
        self.annotations_path = annotations_path

    


    def processed_annotations(self, len_frames ,annotations):
        expanded = pd.DataFrame(columns=['start_frame', 'end_frame', 'action'])

        for _, row in annotations.iterrows():
            start_frame = int(row[0])
            end_frame = int(row[1])
            action = row[2]
            len = end_frame - start_frame + 1
            rem = self.frames_per_clip - len

            start_frame = max(0, start_frame - rem // 2)
            end_frame = min((start_frame + self.frames_per_clip - 1 ), len_frames - 1)

            if end_frame - start_frame + 1 < self.frames_per_clip:
                continue
            
            expanded = pd.concat([expanded, pd.DataFrame([{
                'start_frame': start_frame,
                'end_frame': end_frame,
                'action': action
            }])], ignore_index=True)
            

        expanded['start_frame'] = expanded['start_frame'].astype(int)
        expanded['end_frame'] = expanded['end_frame'].astype(int)
        expanded['action'] = expanded['action'].astype(str)

        # Remove overlapping sections with actions other than 'driving'
        for i, row in expanded.iterrows():
            overlapping = expanded[
            (expanded['start_frame'] <= row['end_frame']) &
            (expanded['end_frame'] >= row['start_frame']) &
            (expanded['action'] != 'driving') &
            (expanded.index != i)
            ]
            if not overlapping.empty:
                expanded.drop(index=i, inplace=True)

        return expanded 
    

    def process(self, data,annotations_path):

        
        if not os.path.exists(annotations_path):
            print(f"Annotations {annotations_path} does not exist. Skipping.")
            return
        
        annotations = pd.read_csv(annotations_path)

        frames = list(data['modalities']['rgb'].values())
        num_frames = len(frames)

        expanded_annotations = self.processed_annotations(annotations,num_frames, self.frames_per_clip)

        for start,end,action in zip(expanded_annotations['start_frame'], 
                                                        expanded_annotations['end_frame'], 
                                                        expanded_annotations['action']):
            start = int(start)
            end = int(end)
            action = str(action)
        

    def write_drive(self):

        dataset = self.process(self.data, self.annotations_path)

        return dataset
        

