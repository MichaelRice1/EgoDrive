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
        
        pass
