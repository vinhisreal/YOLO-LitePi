# YOLO-LitePi: Lightweight Real-Time Traffic Sign Detection Pipeline Optimized for Raspberry Pi 5

[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%205-C51A4A?logo=raspberry-pi)](https://www.raspberrypi.com/products/raspberry-pi-5/)
[![Framework](https://img.shields.io/badge/Framework-NCNN%20%7C%20ONNX%20%7C%20OpenVINO-blue)](https://github.com/Tencent/ncnn)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Official implementation of the paper:** "YOLO-LitePi: A Lightweight Real-Time Traffic Sign Detection Pipeline Optimized for Raspberry Pi 5".

## 📖 Introduction

**YOLO-LitePi** is an optimized two-stage Traffic Sign Recognition (TSR) pipeline designed specifically for edge devices, targeting the **Raspberry Pi 5**. Real-time TSR on edge hardware is often constrained by limited compute budgets. We introduce **YOLO-LitePi**, a custom detector that applies hardware-aware architectural scaling, coupled with a lightweight **ShuffleNet V2** classifier. By leveraging the **NCNN** inference engine and system-level optimizations (vectorized NMS, CPU threading), our pipeline achieves real-time performance without specialized accelerators.

### Key Features
**Custom Architecture:** Channel-pruned YOLO backbone (1.8M params, 5.2 GFLOPS) optimized for CPU execution.
**Two-Stage Pipeline:** High-speed detection followed by batched classification.
**Edge Optimized:** Fully optimized for Raspberry Pi 5 (Quad-core Cortex-A76 @ 2.4 GHz).
**High Performance:** Achieves **13-24 FPS** on Pi 5 (NCNN backend), significantly outperforming standard YOLOv8n baselines.
**Multi-Backend Support:** Supports ONNX Runtime, OpenVINO, and NCNN.

## System Architecture

The pipeline consists of two main stages:
1. **Detector (YOLO-LitePi):** A lightweight anchor-free detector derived from YOLOv8n, with a 25% channel reduction and removed objectness branch for speed.
2. **Classifier (ShuffleNet V2):** A computationally efficient CNN for recognizing cropped traffic signs.

The workflow involves:
`Image Acquisition` -> `Preprocessing` -> `YOLO-LitePi Detection` -> `Vectorized NMS` -> `ROI Extraction` -> `Batched Classification` -> `Output`.


## 📊 Performance Benchmarks

Tested on **Raspberry Pi 5 (8GB)** running Debian Trixie.

### 1. Detection Model Efficiency (Training Phase)
Comparison of YOLO-LitePi against standard YOLO variants on TT100K dataset.

| Model | Recall | mAP@0.5 | mAP@0.5:0.95 | Training Time Gain |
| :--- | :--- | :--- | :--- | :--- |
| YOLOv5n | 0.838 | 0.912 | 0.657 | - |
| YOLOv8n | 0.839 | 0.916 | 0.662 | - |
| YOLOv11n | 0.846 | 0.916 | 0.661 | - |
| **YOLO-LitePi** | **0.794** | **0.870** | **0.603** | **+6.98%** |

### 2. Inference Backend Comparison (Raspberry Pi 5)
Throughput (FPS) comparison across different inference backends.

| Backend | Dataset | YOLOv8n (FPS) | **YOLO-LitePi (FPS)** | **Gain** |
| :--- | :--- | :--- | :--- | :--- |
| **ONNX Runtime** | TT100K | 6.74 | **9.16** | +35.9% |
| | VN-Signs | 6.85 | **14.30** | +108.7% |
| **OpenVINO** | TT100K | 13.26 | **15.51** | +16.9% |
| | VN-Signs | 13.81 | **22.74** | +64.6% |
| **NCNN** | TT100K | 13.40 | **16.69** | **+24.5%** |
| | VN-Signs | 13.61 | **24.04** | **+74.4%** |

> *Note: NCNN provides the best latency for real-time applications.*

### 3. Classifier Selection (CPU Profiling)
Why we chose **ShuffleNet V2** for the second stage.

| Model | Accuracy (VN-Signs) | FPS (CPU) |
| :--- | :--- | :--- |
| ResNet18 | 99.27% | 196.2 |
| MobileNet V2 | 99.33% | 131.4 |
| EfficientNet-B0 | 99.39% | 143.5 |
| **ShuffleNet V2** | **99.51%** | **279.2** |

## 🛠️ Installation

### Prerequisites
Raspberry Pi 5 (Recommended OS: Raspberry Pi OS Bookworm / Debian Trixie)
* Python 3.11+

### 1. Clone the repository
```bash
git clone [https://github.com/vinhisreal/YOLO-LitePi.git](https://github.com/vinhisreal/YOLO-LitePi.git)
cd YOLO-LitePi
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
Key dependencies include: numpy, opencv-python, ncnn, onnxruntime, openvino.

## 🚀 Usage
Real-time Inference on Raspberry Pi
To run the full pipeline using the NCNN backend (recommended for best speed):
To run the End-to-End (E2E) evaluation system, you need to navigate to the corresponding pipeline directory for each dataset.
### 1. Run on TT100K Dataset
Use the `e2e.py` script located in `src/tt100k/pipeline/`.
```bash
# 1. Navigate to the working directory
cd src/tt100k/pipeline

# 2. Run evaluation (Example: NCNN backend with ShuffleNetV2)
python e2e.py \
  --detector_param "../convert/model/yolo_plus/yolo_plus_ncnn_model/model.ncnn.param" \
  --detector_bin "../convert/model/yolo_plus/yolo_plus_ncnn_model/model.ncnn.bin" \
  --classifier "../weight/shufflenetv2.pth" \
  --clf_arch shufflenetv2 \
  --benchmark_conf 0.25 \
  --detector_threads 4 \
  --save_viz True
```
### 2. Run on VN-Signs Dataset
Similarly, use the script located in src/vntsr/pipeline/.
```bash
# 1. Navigate to the working directory
cd src/vntsr/pipeline

# 2. Run evaluation
python e2e.py \
  --detector_param "../convert/model/yolo_plus/yolo_plus_ncnn_model/model.ncnn.param" \
  --detector_bin "../convert/model/yolo_plus/yolo_plus_ncnn_model/model.ncnn.bin" \
  --classifier "../weight/shufflenetv2_vn.pth" \
  --clf_arch shufflenetv2 \
  --benchmark_conf 0.25 \
  --device cpu
```
## 📂 Datasets
This project utilizes two datasets:

TT100K: A large-scale traffic sign benchmark (Avg resolution 2048x2048).

VN-Signs: A proprietary dataset collected in Vietnam, featuring complex urban backgrounds and localized sign classes (Avg resolution 1198x681).

## 📜 Citation
If you find this project useful in your research, please cite our paper:


@inproceedings{vinh2026yololitepi,
  title={YOLO-LitePi: A Lightweight Real-Time Traffic Sign Detection Pipeline Optimized for Raspberry Pi 5},
  author={Nguyen, Quang Vinh and Nguyen, Quoc Duy and Tran, Tin T.},
  booktitle={ICCIES 2026},
  year={2026},
  organization={Faculty of Information Technology, Ton Duc Thang University}
}
## 👥 Authors
Nguyen Quang Vinh - Ton Duc Thang University 
Nguyen Quoc Duy - Ton Duc Thang University 
Tin T. Tran (Supervisor) - Ton Duc Thang University

## 📄 License
This project is licensed under the MIT License.