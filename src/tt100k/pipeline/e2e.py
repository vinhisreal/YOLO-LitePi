#!/usr/bin/env python3
"""
End-to-End Evaluation Pipeline: NCNN YOLO Detector + PyTorch Classifier
- NCNN for fast detection
- PyTorch for classification (ResNet18, EfficientNet, MobileNetV2, ShuffleNetV2)
- Full evaluation with Precision, Recall, F1, mAP
- Visualization and comparison support
"""

import os
import argparse
import json
import time
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torchvision import transforms, models
import ncnn
import psutil


# ==================== DATA CLASSES ====================
@dataclass
class PipelineMetrics:
    """Comprehensive metrics for evaluation"""
    # Timing (ms)
    t_detection: float = 0.0
    t_roi_extract: float = 0.0
    t_classification: float = 0.0
    t_postprocess: float = 0.0
    t_total: float = 0.0
    fps: float = 0.0
    
    # Detection
    num_detections: int = 0
    det_confidence_avg: float = 0.0
    
    # Classification
    cls_confidence_avg: float = 0.0
    
    # System
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    temperature: float = 0.0
    
    # Evaluation
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    
    level: str = "NCNN+PyTorch"


# ==================== HELPER FUNCTIONS ====================
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize and pad image to new_shape with letterbox"""
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    
    dw /= 2
    dh /= 2
    
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img, r, (dw, dh)


def nms_numpy(boxes, scores, iou_threshold=0.45):
    """Non-Maximum Suppression"""
    if len(boxes) == 0:
        return []
    
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
            
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes (x1, y1, x2, y2)"""
    x1_i = max(box1[0], box2[0])
    y1_i = max(box1[1], box2[1])
    x2_i = min(box1[2], box2[2])
    y2_i = min(box1[3], box2[3])
    
    inter_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def parse_yolo_label(label_path, img_w, img_h):
    """Parse YOLO format label: class_id x_center y_center width height (normalized)"""
    if not os.path.exists(label_path):
        return []
    
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:5])
            
            x1 = int((x_c - w/2) * img_w)
            y1 = int((y_c - h/2) * img_h)
            x2 = int((x_c + w/2) * img_w)
            y2 = int((y_c + h/2) * img_h)
            
            boxes.append((cls_id, x1, y1, x2, y2))
    return boxes


def load_class_names(class_file_path):
    """Load class names from JSON or text file"""
    with open(class_file_path, 'r') as f:
        try:
            data = json.load(f)
            if isinstance(data, dict):
                max_idx = max(int(k) for k in data.keys())
                class_names = [''] * (max_idx + 1)
                for idx_str, name in data.items():
                    class_names[int(idx_str)] = name
                return class_names
            else:
                raise ValueError("JSON must be a dictionary")
        except json.JSONDecodeError:
            f.seek(0)
            class_names = [line.strip() for line in f if line.strip()]
            return class_names


def sample_images(img_files, num_samples, seed=42):
    """Sample images deterministically for fair comparison"""
    if num_samples is None or num_samples <= 0 or num_samples >= len(img_files):
        return img_files
    
    random.seed(seed)
    sampled = random.sample(img_files, num_samples)
    return sorted(sampled)


def extract_model_name(model_path):
    """Extract clean model name from path"""
    return Path(model_path).stem


# ==================== NCNN DETECTOR ====================
class NCNNDetector:
    """YOLO detector using NCNN backend"""
    
    def __init__(self, param_path: str, bin_path: str, input_size: int = 640, 
                 use_gpu: bool = False, num_threads: int = 4,
                 input_name: str = "in0", output_name: str = "out0"):
        self.input_size = input_size
        self.input_name = input_name
        self.output_name = output_name
        
        print(f"[NCNN Detector] Loading model...")
        print(f"  📦 Param: {param_path}")
        print(f"  📦 Bin: {bin_path}")
        
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = use_gpu
        self.net.opt.num_threads = num_threads
        
        if self.net.load_param(param_path) != 0:
            raise RuntimeError(f"Failed to load param: {param_path}")
        if self.net.load_model(bin_path) != 0:
            raise RuntimeError(f"Failed to load bin: {bin_path}")
        
        device = "GPU (Vulkan)" if use_gpu else f"CPU ({num_threads} threads)"
        print(f"  ✓ Device: {device}")
        print(f"  ✓ Input size: {input_size}x{input_size}")
    
    def preprocess(self, image: np.ndarray) -> Tuple:
        """Preprocess image for NCNN"""
        img_resized, ratio, (dw, dh) = letterbox(image, new_shape=(self.input_size, self.input_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        mat_in = ncnn.Mat.from_pixels(
            img_rgb,
            ncnn.Mat.PixelType.PIXEL_RGB,
            img_rgb.shape[1],
            img_rgb.shape[0]
        )
        
        mean_vals = [0, 0, 0]
        norm_vals = [1/255.0, 1/255.0, 1/255.0]
        mat_in.substract_mean_normalize(mean_vals, norm_vals)
        
        return mat_in, ratio, (dw, dh)
    
    def postprocess(self, output: ncnn.Mat, orig_shape: Tuple[int, int], 
                   ratio: float, pad: Tuple[float, float],
                   conf_threshold: float = 0.5, iou_threshold: float = 0.45):
        """Postprocess NCNN output"""
        output_array = np.array(output)
        
        if output_array.ndim == 2:
            output_array = np.expand_dims(output_array, axis=0)
        if output_array.shape[-1] == 84:
            output_array = output_array.transpose(0, 2, 1)
        
        predictions = output_array[0]
        boxes = predictions[:4].T
        scores = predictions[4:].T
        
        class_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)
        
        mask = class_scores > conf_threshold
        boxes = boxes[mask]
        scores = class_scores[mask]
        class_ids = class_ids[mask]
        
        if len(boxes) == 0:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,))
        
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        
        boxes_xyxy[:, [0, 2]] -= pad[0]
        boxes_xyxy[:, [1, 3]] -= pad[1]
        boxes_xyxy /= ratio
        
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_shape[1])
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_shape[0])
        
        nms_indices = []
        for cls in np.unique(class_ids):
            cls_mask = class_ids == cls
            keep = nms_numpy(boxes_xyxy[cls_mask], scores[cls_mask], iou_threshold)
            nms_indices.extend(np.where(cls_mask)[0][keep])
        
        if len(nms_indices) > 0:
            nms_indices = np.array(nms_indices)
            boxes_xyxy = boxes_xyxy[nms_indices]
            scores = scores[nms_indices]
            class_ids = class_ids[nms_indices]
        else:
            boxes_xyxy = np.empty((0, 4))
            scores = np.empty((0,))
            class_ids = np.empty((0,))
        
        return boxes_xyxy, scores, class_ids
    
    def detect(self, image: np.ndarray, conf_threshold: float = 0.5, 
               iou_threshold: float = 0.45):
        """Run detection"""
        orig_h, orig_w = image.shape[:2]
        
        mat_in, ratio, pad = self.preprocess(image)
        
        ex = self.net.create_extractor()
        ex.input(self.input_name, mat_in)
        ret, mat_out = ex.extract(self.output_name)
        
        if ret != 0:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,))
        
        boxes, scores, class_ids = self.postprocess(
            mat_out, (orig_h, orig_w), ratio, pad, conf_threshold, iou_threshold
        )
        
        return boxes, scores, class_ids


# ==================== PYTORCH CLASSIFIER ====================
def build_classifier(arch, num_classes, model_path=None, device="cpu"):
    """Build classifier from various architectures"""
    if arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "efficientnet":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif arch == "mobilenetv2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif arch == "shufflenetv2":
        model = models.shufflenet_v2_x1_0(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    
    if model_path and os.path.exists(model_path):
        try:
            sd = torch.load(model_path, map_location=device)
            model.load_state_dict(sd)
            print(f"✓ Loaded classifier weights from {model_path}")
        except Exception as e:
            print(f"⚠ Warning: Could not load weights: {e}")
    
    model.to(device)
    model.eval()
    return model


class PyTorchClassifier:
    """Traffic sign classifier using PyTorch"""
    
    def __init__(self, model_path: str, arch: str, num_classes: int = 58, 
                 input_size: int = 64, device: str = 'cpu'):
        self.input_size = input_size
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.arch = arch
        
        print(f"[PyTorch Classifier] Loading {arch} model...")
        print(f"  📦 Model: {model_path}")
        
        self.model = build_classifier(arch, num_classes, model_path, device)
        
        # Trong __init__ của PyTorchClassifier
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),  
            transforms.Normalize([0.18, 0.18, 0.18],[0.34, 0.34, 0.34])
        ])

        print(f"  ✓ Architecture: {arch}")
        print(f"  ✓ Device: {device}")
        print(f"  ✓ Input size: {input_size}x{input_size}")
        print(f"  ✓ Num classes: {num_classes}")
    
    @torch.no_grad()
    def predict_batch(self, images: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Batch prediction"""
        if len(images) == 0:
            return np.array([]), np.array([])
        
        batch_tensors = []
        # Trong predict_batch
        for img in images:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)  # Chuyển thủ công
            img_tensor = self.transform(pil_img)
            batch_tensors.append(img_tensor)
        
        batch = torch.stack(batch_tensors).to(self.device)
        
        output = self.model(batch)
        probs = torch.softmax(output, dim=1).cpu().numpy()
        
        return np.argmax(probs, axis=1), probs

# ==================== HYBRID PIPELINE ====================
class HybridPipeline:
    """NCNN Detector + PyTorch Classifier Pipeline"""
    
    def __init__(self, 
                 detector_param: str,
                 detector_bin: str,
                 classifier_path: str,
                 classifier_arch: str,
                 num_classes: int = 58,
                 det_input_size: int = 640,
                 cls_input_size: int = 64,
                 use_gpu_detector: bool = False,
                 detector_threads: int = 4,
                 classifier_device: str = 'cpu',
                 batch_size: int = 8):
        
        print("\n" + "="*70)
        print("HYBRID PIPELINE: NCNN Detector + PyTorch Classifier")
        print("="*70)
        
        self.detector = NCNNDetector(
            param_path=detector_param,
            bin_path=detector_bin,
            input_size=det_input_size,
            use_gpu=use_gpu_detector,
            num_threads=detector_threads
        )
        
        self.classifier = PyTorchClassifier(
            model_path=classifier_path,
            arch=classifier_arch,
            num_classes=num_classes,
            input_size=cls_input_size,
            device=classifier_device
        )
        
        self.batch_size = batch_size
        
        print(f"\n✓ Pipeline ready!")
        print(f"  - Detection: NCNN (YOLO)")
        print(f"  - Classification: PyTorch {classifier_arch} ({classifier_device})")
        print(f"  - Batch size: {batch_size}")
        print("="*70 + "\n")
    
    def run(self, image: np.ndarray, conf_threshold: float = 0.5, 
            iou_threshold: float = 0.45, min_area: int = 100) -> Tuple[List[Dict], PipelineMetrics]:
        """Run end-to-end pipeline"""
        
        metrics = PipelineMetrics(level="NCNN+PyTorch")
        t_start = time.perf_counter()
        
        # Detection
        t0 = time.perf_counter()
        boxes, scores, det_classes = self.detector.detect(image, conf_threshold, iou_threshold)
        metrics.t_detection = (time.perf_counter() - t0) * 1000
        metrics.num_detections = len(boxes)
        
        if len(scores) > 0:
            metrics.det_confidence_avg = float(np.mean(scores))
        
        # ROI Extraction
        t1 = time.perf_counter()
        rois = []
        valid_indices = []
        h, w = image.shape[:2]
        
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = np.clip(x1, 0, w-1), np.clip(y1, 0, h-1)
            x2, y2 = np.clip(x2, x1+1, w), np.clip(y2, y1+1, h)
            
            area = (x2 - x1) * (y2 - y1)
            if area >= min_area and x2 > x1 and y2 > y1:
                rois.append(image[y1:y2, x1:x2])
                valid_indices.append(idx)
        
        metrics.t_roi_extract = (time.perf_counter() - t1) * 1000
        
        # Filter boxes to only valid ones
        if len(valid_indices) > 0:
            boxes = boxes[valid_indices]
            scores = scores[valid_indices]
            det_classes = det_classes[valid_indices]
        else:
            boxes = np.empty((0, 4))
            scores = np.empty((0,))
            det_classes = np.empty((0,))
        
        # Classification
        t2 = time.perf_counter()
        all_cls = []
        all_probs = []
        
        for i in range(0, len(rois), self.batch_size):
            batch = rois[i:i+self.batch_size]
            cls_ids, probs = self.classifier.predict_batch(batch)
            all_cls.extend(cls_ids)
            all_probs.extend(probs)
        
        metrics.t_classification = (time.perf_counter() - t2) * 1000
        
        if len(all_probs) > 0:
            max_probs = [np.max(p) for p in all_probs]
            metrics.cls_confidence_avg = float(np.mean(max_probs))
        
        # Total time
        metrics.t_total = (time.perf_counter() - t_start) * 1000
        metrics.fps = 1000.0 / metrics.t_total if metrics.t_total > 0 else 0
        
        # System metrics
        metrics.cpu_percent = psutil.cpu_percent()
        metrics.memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                metrics.temperature = float(f.read()) / 1000.0
        except:
            pass
        
        # Build results
        results = []
        for i in range(len(boxes)):
            results.append({
                'bbox': tuple(boxes[i].astype(int)),
                'det_class': int(det_classes[i]),
                'det_conf': float(scores[i]),
                'cls_class': int(all_cls[i]) if i < len(all_cls) else -1,
                'cls_conf': float(np.max(all_probs[i])) if i < len(all_probs) else 0.0,
                'time_det': metrics.t_detection / len(boxes),
                'time_cls': metrics.t_classification / len(boxes) if len(boxes) > 0 else 0
            })
        
        return results, metrics


# ==================== EVALUATION ====================
# def evaluate_predictions(all_preds, all_gts, num_classes,
#                          iou_threshold=0.5,
#                          iou_thresholds=np.arange(0.5, 1.0, 0.05)):
#     """
#     Calculate:
#       - Precision, Recall, F1 (only on classes present in test)
#       - mAP@0.5
#       - mAP@[0.5:0.95]
#     """

#     # Nếu rỗng tránh lỗi
#     if len(all_preds) == 0 or len(all_gts) == 0:
#         return {
#             "precision": np.zeros(num_classes),
#             "recall": np.zeros(num_classes),
#             "f1": np.zeros(num_classes),
#             "tp": np.zeros(num_classes),
#             "fp": np.zeros(num_classes),
#             "fn": np.zeros(num_classes),
#             "mAP50": 0.0,
#             "mAP50_95": 0.0,
#             "classes_present": np.zeros(num_classes, dtype=bool)
#         }

#     tp = np.zeros(num_classes)
#     fp = np.zeros(num_classes)
#     fn = np.zeros(num_classes)

#     # ----------- MATCH TP / FP / FN -----------------
#     for preds, gts in zip(all_preds, all_gts):
#         gt_matched = [False] * len(gts)

#         for pred in preds:
#             pred_box = pred['bbox']
#             pred_cls = pred['cls_class']

#             best_iou = 0
#             best_gt_idx = -1

#             for i, (gt_cls, gx1, gy1, gx2, gy2) in enumerate(gts):
#                 if gt_matched[i]:
#                     continue

#                 iou = calculate_iou(pred_box, (gx1, gy1, gx2, gy2))
#                 if iou > best_iou:
#                     best_iou = iou
#                     best_gt_idx = i

#             if best_iou >= iou_threshold and best_gt_idx >= 0:
#                 gt_cls = gts[best_gt_idx][0]
#                 gt_matched[best_gt_idx] = True

#                 if pred_cls == gt_cls:
#                     tp[pred_cls] += 1
#                 else:
#                     fp[pred_cls] += 1
#                     fn[gt_cls] += 1
#             else:
#                 fp[pred_cls] += 1

#         # GT chưa match
#         for i, (gt_cls, _, _, _, _) in enumerate(gts):
#             if not gt_matched[i]:
#                 fn[gt_cls] += 1


#     # ===== Identify valid classes =====
#     classes_present = (tp + fp + fn) > 0

#     # ========== PRECISION / RECALL / F1 ==========
#     precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp+fp)!=0)
#     recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp+fn)!=0)
#     f1 = np.divide(2 * precision * recall, precision + recall,
#                    out=np.zeros_like(precision), where=(precision+recall)!=0)

#     # =====================================================
#     # mAP CALCULATION — only for classes_present
#     # =====================================================

#     def compute_ap(rec, prec):
#         mrec = np.concatenate(([0], rec, [1]))
#         mpre = np.concatenate(([0], prec, [0]))

#         for i in range(len(mpre)-1, 0, -1):
#             mpre[i-1] = max(mpre[i-1], mpre[i])

#         idx = np.where(mrec[1:] != mrec[:-1])[0]
#         return np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1])

#     aps_over_iou = []

#     for thr in iou_thresholds:
#         ap_per_class = []

#         for c in range(num_classes):
#             if not classes_present[c]:
#                 continue  # SKIP class that doesn't appear

#             ap = compute_ap(np.array([recall[c]]), np.array([precision[c]]))
#             ap_per_class.append(ap)

#         # mean AP of valid classes
#         aps_over_iou.append(np.mean(ap_per_class) if len(ap_per_class) > 0 else 0.0)

#     mAP50 = aps_over_iou[0]
#     mAP50_95 = np.mean(aps_over_iou)

#     # RETURN
#     return {
#         'precision': precision,
#         'recall': recall,
#         'f1': f1,
#         'tp': tp,
#         'fp': fp,
#         'fn': fn,
#         'classes_present': classes_present,
#         'mAP50': mAP50,
#         'mAP50_95': mAP50_95
#     }
 
# ==================== EVALUATION ====================
def evaluate_predictions(all_preds, all_gts, num_classes, iou_threshold=0.5, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    """
    Tính toán metrics chuẩn Ultralytics YOLO (Global Evaluation).
    Fix: Chỉ tính mAP trên các class có xuất hiện (Present Classes).
    """

    # --- Helper: Tính IoU Matrix ---
    def box_iou(box1, box2):
        def box_area(box):
            return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

        area1 = box_area(box1)
        area2 = box_area(box2)

        lt = np.maximum(box1[:, None, :2], box2[:, :2])
        rb = np.minimum(box1[:, None, 2:], box2[:, 2:])

        wh = (rb - lt).clip(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        union = area1[:, None] + area2 - inter
        return inter / (union + 1e-7)

    # --- Helper: Tính AP (101-point interpolation) ---
    def compute_ap(recall, precision):
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre))) # Envelope
        x = np.linspace(0, 1, 101)
        ap = np.trapz(np.interp(x, mrec, mpre), x)
        return ap

    # --- 1. MATCHING ---
    stats = [] 
    for preds, gts in zip(all_preds, all_gts):
        if len(preds) == 0:
            if len(gts) > 0:
                stats.append((
                    np.zeros((0, len(iou_thresholds)), dtype=bool), 
                    np.array([]), np.array([]), 
                    np.array(gts)[:, 0] 
                ))
            continue

        pred_bboxes = np.array([p['bbox'] for p in preds])
        pred_conf = np.array([p['conf'] for p in preds])
        pred_cls = np.array([p['cls_class'] for p in preds])

        if len(gts) > 0:
            gts_arr = np.array(gts)
            target_cls = gts_arr[:, 0]
            target_bboxes = gts_arr[:, 1:]
        else:
            target_cls = np.array([])
            target_bboxes = np.array([])

        correct = np.zeros((len(preds), len(iou_thresholds)), dtype=bool)
        
        if len(gts) > 0:
            iou_matrix = box_iou(pred_bboxes, target_bboxes)
            for i, iou_thres in enumerate(iou_thresholds):
                x = np.where((iou_matrix >= iou_thres))
                if x[0].shape[0]:
                    matches = np.concatenate((np.stack(x, 1), iou_matrix[x[0], x[1]][:, None]), 1)
                    if x[0].shape[0] > 1:
                        matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    for (pred_idx, gt_idx, _) in matches:
                        if pred_cls[int(pred_idx)] == target_cls[int(gt_idx)]:
                            correct[int(pred_idx), i] = True
        
        stats.append((correct, pred_conf, pred_cls, target_cls))

    # --- 2. COMPUTE METRICS ---
    if not stats:
        return {
            "precision": np.zeros(num_classes), "recall": np.zeros(num_classes),
            "f1": np.zeros(num_classes), "tp": np.zeros(num_classes),
            "fp": np.zeros(num_classes), "fn": np.zeros(num_classes),
            "mAP50": 0.0, "mAP50_95": 0.0,
            "classes_present": np.zeros(num_classes, dtype=bool)
        }

    tp_all = np.concatenate([s[0] for s in stats], 0)
    conf_all = np.concatenate([s[1] for s in stats], 0)
    pred_cls_all = np.concatenate([s[2] for s in stats], 0)
    target_cls_all = np.concatenate([s[3] for s in stats], 0)

    i = np.argsort(-conf_all)
    tp_all = tp_all[i]
    conf_all = conf_all[i]
    pred_cls_all = pred_cls_all[i]

    unique_classes, nt = np.unique(target_cls_all, return_counts=True)
    nt_dict = dict(zip(unique_classes, nt))

    ap50_per_class = np.zeros(num_classes)
    ap50_95_per_class = np.zeros(num_classes)
    p_best = np.zeros(num_classes)
    r_best = np.zeros(num_classes)
    f1_best = np.zeros(num_classes)
    
    tp_count = np.zeros(num_classes)
    fp_count = np.zeros(num_classes)
    fn_count = np.zeros(num_classes)

    for c in range(num_classes):
        n_gt = nt_dict.get(c, 0)
        i_cls = pred_cls_all == c
        n_p = i_cls.sum()

        if n_p == 0 and n_gt == 0: continue
        
        if n_p == 0 or n_gt == 0:
            ap50_per_class[c] = 0.0
            ap50_95_per_class[c] = 0.0
            fn_count[c] = n_gt
            continue

        tpc = (tp_all[i_cls]).cumsum(0)
        fpc = (1 - tp_all[i_cls]).cumsum(0)

        rec = tpc / (n_gt + 1e-16)
        prec = tpc / (tpc + fpc + 1e-16)

        ap_per_iou = []
        for j in range(tp_all.shape[1]):
            ap_per_iou.append(compute_ap(rec[:, j], prec[:, j]))
        
        ap50_per_class[c] = ap_per_iou[0]
        ap50_95_per_class[c] = np.mean(ap_per_iou)

        P_curve = prec[:, 0]
        R_curve = rec[:, 0]
        F1_curve = 2 * P_curve * R_curve / (P_curve + R_curve + 1e-16)
        best_idx = np.argmax(F1_curve)

        p_best[c] = P_curve[best_idx]
        r_best[c] = R_curve[best_idx]
        f1_best[c] = F1_curve[best_idx]
        
        tp_count[c] = tpc[best_idx, 0]
        fp_count[c] = fpc[best_idx, 0]
        fn_count[c] = n_gt - tp_count[c]

    # --- PHẦN SỬA ĐỔI QUAN TRỌNG NHẤT ---
    # Xác định các class có xuất hiện trong GT
    present_classes = unique_classes.astype(int)
    
    # Chỉ tính Mean AP trên các class có xuất hiện
    if len(present_classes) > 0:
        mAP50 = np.mean(ap50_per_class[present_classes])
        mAP50_95 = np.mean(ap50_95_per_class[present_classes])
    else:
        mAP50 = 0.0
        mAP50_95 = 0.0

    return {
        "precision": p_best,
        "recall": r_best,
        "f1": f1_best,
        "tp": tp_count,
        "fp": fp_count,
        "fn": fn_count,
        "mAP50": mAP50,       # Đã sửa
        "mAP50_95": mAP50_95, # Đã sửa
        "ap50_per_class": ap50_per_class,
        "classes_present": np.isin(np.arange(num_classes), unique_classes)
    }
# ==================== VISUALIZATION ====================
def visualize_prediction(img_bgr, predictions, ground_truths, class_names, output_path):
    """Draw predictions (green) and ground truths (blue) on image"""
    vis = img_bgr.copy()
    h, w = vis.shape[:2]
    
    # Draw ground truths in blue
    for idx, (gt_cls, x1, y1, x2, y2) in enumerate(ground_truths):
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 3)
        
        label = f"GT: {class_names[gt_cls]}"
        font_scale = 0.6
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                                       font_scale, thickness)
        
        text_x = x1
        text_y = max(y1 - 10, text_h + 5)
        cv2.rectangle(vis, (text_x, text_y - text_h - baseline), 
                     (text_x + text_w, text_y + baseline), (255, 0, 0), -1)
        cv2.putText(vis, label, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    # Draw predictions in green
    for pred in predictions:
        x1, y1, x2, y2 = pred['bbox']
        cls_idx = pred['cls_class']
        cls_conf = pred['cls_conf']
        det_conf = pred['det_conf']
        
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        label = f"PRED: {class_names[cls_idx]}"
        conf_label = f"Cls:{cls_conf:.2f} Det:{det_conf:.2f}"
        
        font_scale = 0.6
        thickness = 2
        (text_w1, text_h1), baseline1 = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                                         font_scale, thickness)
        (text_w2, text_h2), baseline2 = cv2.getTextSize(conf_label, cv2.FONT_HERSHEY_SIMPLEX, 
                                                         font_scale - 0.1, thickness - 1)
        
        text_x = x1
        text_y = min(y2 + text_h1 + 15, h - 5)
        
        cv2.rectangle(vis, (text_x, text_y - text_h1 - baseline1), 
                     (text_x + text_w1, text_y + baseline1), (0, 255, 0), -1)
        
        conf_y = text_y + text_h2 + 5
        cv2.rectangle(vis, (text_x, conf_y - text_h2 - baseline2), 
                     (text_x + text_w2, conf_y + baseline2), (0, 200, 0), -1)
        
        cv2.putText(vis, label, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        cv2.putText(vis, conf_label, (text_x, conf_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale - 0.1, (255, 255, 255), thickness - 1)
    
    summary = f"GT: {len(ground_truths)} | Predictions: {len(predictions)}"
    cv2.rectangle(vis, (5, 5), (400, 35), (0, 0, 0), -1)
    cv2.putText(vis, summary, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite(str(output_path), vis)


def plot_metrics(metrics, class_names, output_dir, model_combo_name):
    """Plot precision, recall, F1 bar charts"""
    precision = metrics['precision']
    recall = metrics['recall']
    f1 = metrics['f1']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Precision
    axes[0, 0].bar(range(len(class_names)), precision, color='skyblue')
    axes[0, 0].set_title(f'Precision per Class - {model_combo_name}', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Class')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_xticks(range(len(class_names)))
    axes[0, 0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Recall
    axes[0, 1].bar(range(len(class_names)), recall, color='lightcoral')
    axes[0, 1].set_title(f'Recall per Class - {model_combo_name}', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Class')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].set_xticks(range(len(class_names)))
    axes[0, 1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # F1 Score
    axes[1, 0].bar(range(len(class_names)), f1, color='lightgreen')
    axes[1, 0].set_title(f'F1 Score per Class - {model_combo_name}', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Class')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_xticks(range(len(class_names)))
    axes[1, 0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Summary stats
    axes[1, 1].axis('off')
    summary_text = f"""
    Model: {model_combo_name}
    
    Overall Metrics:
    Mean Precision: {np.mean(precision):.3f}
    Mean Recall: {np.mean(recall):.3f}
    Mean F1 Score: {np.mean(f1):.3f}
    
    Total TP: {int(np.sum(metrics['tp']))}
    Total FP: {int(np.sum(metrics['fp']))}
    Total FN: {int(np.sum(metrics['fn']))}
    
    Test Images: {metrics.get('num_images', 'N/A')}
    Avg Time: {metrics.get('avg_time', 0):.3f}s
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_combo_name}_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved metrics plot to {output_dir}/{model_combo_name}_metrics.png")


# ==================== MAIN PROCESSING ====================
# ==================== MAIN PROCESSING ====================
def process_image(img_path, label_path, pipeline, class_names, output_dir, args, save_viz=True):
    """
    Process single image with ground truth.
    Performs two passes:
    1. Benchmark Pass (Real-world FPS): Uses high confidence (e.g., 0.25)
    2. Evaluation Pass (mAP): Uses low confidence (e.g., 0.001)
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None, None, 0.0
    
    h, w = img_bgr.shape[:2]
    
    # Parse ground truth
    ground_truths = parse_yolo_label(label_path, w, h)
    
    # --- PASS 1: BENCHMARK (Đo FPS thực tế) ---
    # Chạy với ngưỡng deploy (ví dụ 0.25) để xem tốc độ thực
    _, bench_metrics = pipeline.run(
        img_bgr, 
        conf_threshold=args.benchmark_conf,  # Ngưỡng cao (0.25)
        iou_threshold=args.iou_threshold,
        min_area=args.min_area
    )
    real_inference_time = bench_metrics.t_total / 1000.0 # Convert ms to seconds

    # --- PASS 2: EVALUATION (Lấy dữ liệu mAP) ---
    # Chạy với ngưỡng thấp (0.001) để lấy tối đa box cho mAP
    # Nếu args.yolo_conf == args.benchmark_conf thì không cần chạy lại
    if args.yolo_conf == args.benchmark_conf:
        raw_predictions = _
    else:
        raw_predictions, _ = pipeline.run(
            img_bgr, 
            conf_threshold=args.yolo_conf,       # Ngưỡng thấp (0.001)
            iou_threshold=args.iou_threshold,
            min_area=args.min_area
        )
    
    # --- CHUẨN HÓA DỮ LIỆU CHO EVALUATION ---
    final_predictions = []
    for pred in raw_predictions:
        final_predictions.append({
            'bbox': pred['bbox'],
            'conf': pred.get('det_conf', 0.0), 
            'cls_class': pred.get('cls_class', -1) 
        })
    
    # Visualize (Dùng kết quả Eval để vẽ cho đầy đủ, hoặc dùng benchmark tùy bạn)

    if save_viz:
        viz_dir = Path(output_dir) / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        vis_path = viz_dir / f"vis_{img_path.stem}.png"
        visualize_prediction(img_bgr, raw_predictions, ground_truths, class_names, vis_path)
    
    return final_predictions, ground_truths, real_inference_time

def main():
    parser = argparse.ArgumentParser(description="E2E NCNN + PyTorch Evaluation Pipeline")
    
    # Model paths
    parser.add_argument("--detector_param", type=str, default="../convert/model/yolo_plus/yolo_plus_ncnn_model/model.ncnn.param", help="Path to NCNN .param file")
    parser.add_argument("--detector_bin", type=str, default="../convert/model/yolo_plus/yolo_plus_ncnn_model/model.ncnn.bin", help="Path to NCNN .bin file")
    parser.add_argument("--classifier", type=str, default="../weight/shufflenetv2.pth", help="Path to classifier .pth")
    parser.add_argument("--clf_arch", type=str, 
                        choices=["resnet18", "efficientnet", "mobilenetv2", "shufflenetv2"],
                        default="shufflenetv2", help="Classifier architecture")
    
    # Data
    parser.add_argument("--input", type=str, default="../Dataset/E2E/data_e2e_tt100k/images/test", help="Input image or folder")
    parser.add_argument("--labels", type=str, default="../Dataset/E2E/data_e2e_tt100k/labels/test", help="Label folder")
    parser.add_argument("--classes", type=str, default="../Dataset/E2E/data_e2e_tt100k/idx2label.json", help="Path to class names file")
    
    # Sampling
    parser.add_argument("--num_samples", type=int, default=None, help="Number of images to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Parameters
    parser.add_argument("--yolo_conf", type=float, default=0.001, help="Eval confidence threshold (Low for mAP)")
    # --- THÊM THAM SỐ NÀY ---
    parser.add_argument("--benchmark_conf", type=float, default=0.25, help="Benchmark confidence threshold (High for FPS)")
    
    parser.add_argument("--min_area", type=int, default=50, help="Minimum detection area")
    parser.add_argument("--iou_threshold", type=float, default=0.45, help="IoU threshold for matching")
    parser.add_argument("--det_input_size", type=int, default=640, help="Detector input size")
    parser.add_argument("--cls_input_size", type=int, default=64, help="Classifier input size")
    parser.add_argument("--detector_threads", type=int, default=4, help="NCNN threads")
    parser.add_argument("--batch_size", type=int, default=8, help="Classification batch size")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu")
    
    # Output
    parser.add_argument("--output", type=str, default="output_eval", help="Output directory")
    parser.add_argument("--save_viz", default=False, help="Save visualization images")
    
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    # ... (Phần Load Class và Init Pipeline giữ nguyên) ...
    class_names = load_class_names(args.classes)
    num_classes = len(class_names)
    detector_name = extract_model_name(args.detector_param)
    model_combo_name = f"{detector_name}+{args.clf_arch}"
    
    print(f"\n{'='*60}")
    print(f"MODEL COMBINATION: {model_combo_name}")
    print(f"{'='*60}")

    pipeline = HybridPipeline(
        detector_param=args.detector_param,
        detector_bin=args.detector_bin,
        classifier_path=args.classifier,
        classifier_arch=args.clf_arch,
        num_classes=num_classes,
        det_input_size=args.det_input_size,
        cls_input_size=args.cls_input_size,
        use_gpu_detector=False,
        detector_threads=args.detector_threads,
        classifier_device=args.device,
        batch_size=args.batch_size
    )

    # Create output directory
    output_dir = Path(args.output) / model_combo_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process input logic
    input_path = Path(args.input)
    img_files = []
    label_dir = None

    if input_path.is_file():
        img_files = [input_path]
        if args.labels:
            label_dir = Path(args.labels) # Handle single file label logic inside loop if needed or pre-resolve
    else:
        if args.labels:
            label_dir = Path(args.labels)
        else:
            label_dir = input_path / "labels"
        img_files = sorted(list(input_path.glob("*.jpg")) + list(input_path.glob("*.png")) + list(input_path.glob("*.jpeg")))
        
        if args.num_samples:
            img_files = sample_images(img_files, args.num_samples, args.seed)

    print(f"\nFound {len(img_files)} images for processing")

    all_preds = []
    all_gts = []
    processed_files = []
    total_benchmark_time = 0 # Đổi tên biến để rõ nghĩa

    for i, img_path in tqdm(enumerate(img_files, 1), total=len(img_files), desc="Processing"):
        # Resolve label path
        if label_dir:
             label_path = label_dir / f"{img_path.stem}.txt"
        else:
             label_path = img_path.parent / "labels" / f"{img_path.stem}.txt"

        # --- GỌI HÀM PROCESS MỚI ---
        preds, gts, infer_time = process_image(
            img_path, label_path, pipeline,
            class_names, output_dir, args, args.save_viz
        )
        
        if preds is not None:
            all_preds.append(preds)
            all_gts.append(gts)
            processed_files.append(img_path.name)
            total_benchmark_time += infer_time # Cộng dồn thời gian thực tế
        else:
            print(f"\nSkipping {img_path.name}")

    # Evaluate
    print("\n" + "="*80)
    print(f"EVALUATION RESULTS - {model_combo_name}")
    print("="*80)
    
    metrics = evaluate_predictions(all_preds, all_gts, num_classes, args.iou_threshold)
    
    # --- TÍNH FPS DỰA TRÊN BENCHMARK TIME ---
    metrics['num_images'] = len(all_preds)
    metrics['avg_time'] = total_benchmark_time / len(all_preds) if len(all_preds) > 0 else 0
    real_fps = 1.0 / metrics['avg_time'] if metrics['avg_time'] > 0 else 0.0

    print(f"\nPerformance Metrics (at conf={args.benchmark_conf}):")
    print(f"  Avg Inference Time: {metrics['avg_time']*1000:.2f} ms")
    print(f"  Real FPS:           {real_fps:.2f} FPS")

    # Print results
    print(f"\nAccuracy Metrics (at conf={args.yolo_conf}):")
    print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 80)
    
    valid = metrics["classes_present"]
    for i, cls_name in enumerate(class_names):
        if valid[i]: # Chỉ in các class có xuất hiện để đỡ rối
             print(f"{cls_name:<20} {metrics['precision'][i]:>10.3f} {metrics['recall'][i]:>10.3f} "
                   f"{metrics['f1'][i]:>10.3f} {int(metrics['tp'][i]):>6} "
                   f"{int(metrics['fp'][i]):>6} {int(metrics['fn'][i]):>6}")
    
    print("-" * 80)
    print(f"{'MEAN':<20} "
        f"{np.mean(metrics['precision'][valid]):>10.3f} "
        f"{np.mean(metrics['recall'][valid]):>10.3f} "
        f"{np.mean(metrics['f1'][valid]):>10.3f}")
    print(f"{'mAP@0.5':<20} {metrics['mAP50']:>10.3f}")
    print(f"{'mAP@0.5:0.95':<20} {metrics['mAP50_95']:>10.3f}")

    
    summary_df = pd.DataFrame([{
        'model_combination': model_combo_name,
        'detector': detector_name,
        'classifier': args.clf_arch,
        'num_test_images': len(all_preds),
        'mean_precision': np.mean(metrics['precision'][valid]),
        'mean_recall': np.mean(metrics['recall'][valid]),
        'mean_f1': np.mean(metrics['f1'][valid]),
        'fps': real_fps,  # <-- Sửa thành FPS thực tế
        'mAP50': metrics['mAP50'],
        'mAP50-95': metrics['mAP50_95'],
    }])
    
    # ... (Phần còn lại giữ nguyên) ...
    summary_path = Path(args.output) / "comparison_summary.csv"
    if summary_path.exists():
        existing_df = pd.read_csv(summary_path)
        summary_df = pd.concat([existing_df, summary_df], ignore_index=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Updated comparison summary at {summary_path}")

    
if __name__ == "__main__":
    main()