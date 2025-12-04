# YOLO-LitePi: Lightweight Real-Time Traffic Sign Detection Pipeline Optimized for Raspberry Pi 5

[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%205-C51A4A?logo=raspberry-pi)](https://www.raspberrypi.com/products/raspberry-pi-5/)
[![Framework](https://img.shields.io/badge/Framework-NCNN%20%7C%20ONNX%20%7C%20OpenVINO-blue)](https://github.com/Tencent/ncnn)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> [cite_start]**Official implementation of the paper:** "YOLO-LitePi: A Lightweight Real-Time Traffic Sign Detection Pipeline Optimized for Raspberry Pi 5"[cite: 1, 2].

## 📖 Introduction

[cite_start]**YOLO-LitePi** is an optimized two-stage Traffic Sign Recognition (TSR) pipeline designed specifically for edge devices, targeting the **Raspberry Pi 5**[cite: 9, 12].

[cite_start]Real-time TSR on edge hardware is often constrained by limited compute budgets[cite: 8]. [cite_start]We introduce **YOLO-LitePi**, a custom detector that applies hardware-aware architectural scaling, coupled with a lightweight **ShuffleNet V2** classifier[cite: 9, 10]. [cite_start]By leveraging the **NCNN** inference engine and system-level optimizations (vectorized NMS, CPU threading), our pipeline achieves real-time performance without specialized accelerators[cite: 9, 14].

### Key Features
* [cite_start]**Custom Architecture:** Channel-pruned YOLO backbone (1.8M params, 5.2 GFLOPS) optimized for CPU execution[cite: 142].
* [cite_start]**Two-Stage Pipeline:** High-speed detection followed by batched classification[cite: 10, 174].
* [cite_start]**Edge Optimized:** Fully optimized for Raspberry Pi 5 (Quad-core Cortex-A76 @ 2.4 GHz)[cite: 39, 175].
* [cite_start]**High Performance:** Achieves **13-24 FPS** on Pi 5 (NCNN backend), significantly outperforming standard YOLOv8n baselines[cite: 228, 275].
* [cite_start]**Multi-Backend Support:** Supports ONNX Runtime, OpenVINO, and NCNN[cite: 43].

## 🏗️ System Architecture

[cite_start]The pipeline consists of two main stages[cite: 73]:
1.  [cite_start]**Detector (YOLO-LitePi):** A lightweight anchor-free detector derived from YOLOv8n, with a 25% channel reduction and removed objectness branch for speed[cite: 79, 137].
2.  [cite_start]**Classifier (ShuffleNet V2):** A computationally efficient CNN for recognizing cropped traffic signs[cite: 81, 235].

The workflow involves:
[cite_start]`Image Acquisition` -> `Preprocessing` -> `YOLO-LitePi Detection` -> `Vectorized NMS` -> `ROI Extraction` -> `Batched Classification` -> `Output`[cite: 77, 78, 79, 80, 81].

## 📊 Performance Benchmarks

[cite_start]Tested on **Raspberry Pi 5 (8GB)** running Debian Trixie[cite: 192].

### Detection Throughput (NCNN Backend)
| Model | Dataset | FPS | mAP@0.5 | Performance Gain |
| :--- | :--- | :--- | :--- | :--- |
| YOLOv8n | TT100K | 13.40 | 0.907 | - |
| **YOLO-LitePi** | **TT100K** | **16.69** | **0.875** | [cite_start]**+24.5%** [cite: 228] |
| YOLOv8n | VN-Signs | 13.61 | 0.978 | - |
| **YOLO-LitePi** | **VN-Signs** | **24.04** | **0.913** | [cite_start]**+74.4%** [cite: 228] |

### End-to-End Latency
[cite_start]The complete pipeline (Detection + Classification) sustains **13.22 - 16.83 FPS** with a total latency of **<60ms** per frame, meeting real-time requirements for ADAS applications[cite: 12, 258].

## 🛠️ Installation

### Prerequisites
* [cite_start]Raspberry Pi 5 (Recommended OS: Raspberry Pi OS Bookworm / Debian Trixie) [cite: 192]
* [cite_start]Python 3.13 

### 1. Clone the repository
```bash
git clone [https://github.com/username/YOLO-LitePi.git](https://github.com/username/YOLO-LitePi.git)
cd YOLO-LitePi
2. Install Dependencies
Bash

pip install -r requirements.txt

Key dependencies include: numpy, opencv-python, ncnn, onnxruntime, openvino.

🚀 Usage
Real-time Inference on Raspberry Pi
To run the full pipeline using the NCNN backend (recommended for best speed):

Bash

python main_edge.py --source 0 --backend ncnn --conf 0.25
Arguments
--source: Video source (0 for webcam, or path to video file).

--backend: Inference backend (ncnn, openvino, onnx, pytorch).

--conf: Confidence threshold for detection.

📂 Datasets
This project utilizes two datasets:

TT100K: A large-scale traffic sign benchmark (Avg resolution 2048x2048).
VN-Signs: A proprietary dataset collected in Vietnam, featuring complex urban backgrounds and localized sign classes (Avg resolution 1198x681).


📜 Citation
If you find this project useful in your research, please cite our paper:


@inproceedings{vinh2026yololitepi,
  title={YOLO-LitePi: A Lightweight Real-Time Traffic Sign Detection Pipeline Optimized for Raspberry Pi 5},
  author={Nguyen, Quang Vinh and Nguyen, Quoc Duy and Tran, Tin T.},
  booktitle={ICCIES 2026},
  year={2026},
  organization={Faculty of Information Technology, Ton Duc Thang University}
}
👥 Authors

Nguyen Quang Vinh - Ton Duc Thang University 


Nguyen Quoc Duy - Ton Duc Thang University 


Tin T. Tran (Supervisor) - Ton Duc Thang University 


📄 License
This project is licensed under the MIT License.