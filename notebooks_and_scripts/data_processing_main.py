

import os 
import sys
sys.path.append('C:/Users/athen/Desktop/Github/MastersThesis/MSc_AI_Thesis/notebooks_and_scripts')

from vrs_extractor import VRSDataExtractor

class DataProcessor:

    def human_processing(self):
        pass

    
    def auto_run(self, data_path):
        
        recordings = os.listdir(data_path)

        for r in recordings:
            pass




