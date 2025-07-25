# EgoDrive: An Egocentric, Multimodal Dataset and Methods for Driver Behavior Recognition


<p align="center">
 <img src="images_gifs/output3.gif" width="49%" alt="Annotated Dataset Samples" />
 <img src="images_gifs/final2.gif" width="49%" alt="Predictions" />
</p>

<p align="center">
<img src="images_gifs/dashmp4.gif" alt="Driver Feedback Dashboard" width="100%">
</p>


## 🚗 Overview

EgoDrive is the first egocentric, multimodal dataset for driver behavior analysis captured using Meta's Project Aria glasses. This proof-of-concept work demonstrates the technical feasibility of real-time driver behavior recognition through multimodal sensor fusion, combining RGB video, eye gaze tracking, hand pose estimation, IMU data, and semantic object detection.

**Key Features:**

- 🎥 **138,024 frames** of synchronized multimodal data (2.5 hours)
- 👁️ **Six behavioral classes**: Driving, Left/Right/Rear Mirror Check, Mobile Phone Usage, Idle
- ⚡ **Real-time performance**: Sub-3ms inference with 97.4% accuracy
- 🔬 **Proof-of-concept validation** for multimodal egocentric driver monitoring
- 🏗️ **Two model variants**: EgoDriveMax (accuracy-focused) and EgoDriveRT (efficiency-focused)

## 🔍 Technical Contributions

### Multimodal Architecture

- **Transformer-based fusion** of heterogeneous sensor streams
- **Modality-specific encoders** for RGB, gaze, hands, IMU, and object detection
- **Temporal alignment strategies** for synchronized multimodal processing


### Dataset Methodology

- **Synchronized data collection** using Project Aria glasses
- **Temporal alignment framework** for heterogeneous sensor streams
- **Annotation protocols** for behavioral class identification

## 🤖 Model Architecture

We developed two variants of our Multimodal Transformer architecture:

| Model       | Blocks | Heads | Feature Dim | RGB | Params | Inference Time | Accuracy |
| ----------- | ------ | ----- | ----------- | --- | ------ | -------------- | -------- |
| EgoDriveMax | 2      | 4     | 256         | ✓   | 42M    | 1595ms         | 98.6%    |
| EgoDriveRT  | 1      | 2     | 32          | ✗   | 104K   | 2.65ms         | 97.4%    |

### Architecture Components:

- **RGB Encoder**: Swin-Tiny + ResNet-18 motion stream
- **Gaze Encoder**: Linear projection + 1D convolution
- **Hands Encoder**: Missing value handling + temporal attention
- **Object Detection Encoder**: YOLO v11 features + temporal modeling
- **IMU Encoder**: Stacked 1D CNNs + GRU for dynamics

<!-- ## 🚀 Getting Started

### Prerequisites

```bash
git clone https://github.com/your-username/egodrive.git
cd egodrive
conda create -n egodrive python=3.10
conda activate egodrive
pip install -r requirements.txt
```

**Dependencies**

- PyTorch >= 1.12.0
- Transformers
- OpenCV
- NumPy
- Project Aria Tools (for data processing) -->

### 📦 Dataset

**Data Structure** The EgoDrive dataset, available upon request, is structured as follows.

```
egodrive_dataset/
├── drives/         # all driving sessions
    ├── Drive1    # sample driving session
        ├── vrs_file/   # contains aria native session data
        ├── hand_poses/ # hand landmark data
        ├── gaze_data/  # gaze data
        ├── object_detections/   # YOLO v11 in-cabin detection
        ├── annotations/         # contains the frame level action annotations
```

**Behavioral Classes**

- Driving - Normal forward driving attention
- Left Mirror Check - Checking left side mirror
- Right Mirror Check - Checking right side mirror
- Rear-view Mirror Check - Checking rear-view mirror
- Mobile Phone Usage - Interacting with mobile device
- Idle - Stationary/waiting periods

**Data Characteristics**

- Training Samples: 32-frame sequence length (2.13 seconds)
- Sampling Rates: RGB (15fps), Gaze (30fps), IMU (800Hz-1KHz)
- Annotation: Frame-by-frame behavioral labels
- Scale: Proof-of-concept single-participant dataset

<!-- ## 🏃‍♂️ Training and Inference

### Training

```bash
python train.py --config configs/egodrive_max.yaml
python train.py --config configs/egodrive_rt.yaml
```

### Inference

```bash
python inference.py --model_path checkpoints/egodrive_rt.pth --input_dir data/test/
```

### Evaluation

```bash
python evaluate.py --model_path checkpoints/egodrive_rt.pth --test_data data/test/
``` -->

## 📊 Results

### Overall Performance

| Metric         | EgoDriveMax | EgoDriveRT |
| -------------- | ----------- | ---------- |
| Accuracy       | 98.6%       | 97.4%      |
| F1-Score       | 98.0%       | 96.6%      |
| Parameters     | 42M         | 104K       |
| Inference Time | 1595ms      | 2.65ms     |

### Per-Action Results

| Action       | EgoDriveMax Acc | EgoDriveRT Acc |
| ------------ | --------------- | -------------- |
| Driving      | 99.3%           | 98.7%          |
| Left Mirror  | 96.9%           | 100%           |
| Right Mirror | 97.4%           | 94.9%          |
| Rear Mirror  | 97.9%           | 91.2%          |
| Mobile Phone | 94.1%           | 96.3%          |
| Idle         | 100%            | 97.1%          |

## ⚠️ Limitations and Scope

This work presents a proof-of-concept study with the following limitations:

- **Single Participant**: Dataset collected from one driver in controlled conditions
- **Limited Generalizability**: Results may not extend to diverse populations
- **Controlled Environment**: Laboratory-style data collection setup
- **Scope**: Technical feasibility demonstration rather than production system

**Future Work**: Multi-participant validation, real-world testing, and privacy-preserving deployment strategies.

<!-- ## 📄 Citation

If you use EgoDrive in your research, please cite:

```bibtex
@inproceedings{egodrive2023,
    title={EgoDrive: An Egocentric, Multimodal Dataset and Methods for Driver Behavior Analysis},
    author={Anonymous Authors},
    booktitle={Proceedings of RANLP 2023},
    year={2023}
}
``` -->

## 🔒 Ethics and Privacy

- **Data Collection**: Conducted with informed consent and GDPR compliance
- **Privacy**: Designed for on-device processing to keep PII local
- **Safety**: Non-intrusive data collection ensuring driver safety
- **Future Deployment**: Privacy-preserving principles for real-world applications

## 📝 License

This dataset and code are released under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).

> Note: This is a research dataset intended for academic and non-commercial use only.

<!-- ## 🤝 Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for guidelines. -->

## 📞 Contact

For questions about the dataset or technical issues, please open an issue or contact [[athenrymichael@gmail.com](mailto\:athenrymichael@gmail.com)].

## ⚡ Key Insight

This work demonstrates that effective multimodal driver behavior recognition is feasible, potentially paving the way for practical egocentric driver monitoring systems for use in myriad cases.
