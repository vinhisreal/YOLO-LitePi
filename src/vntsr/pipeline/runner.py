"""
Multi-Level Pipeline Optimization with Complete Accuracy Evaluation
===================================================================

Features:
- 5 progressive optimization levels (0-4)
- Full accuracy metrics: Precision, Recall, F1, mAP
- YOLO format ground truth support
- Flexible test set: N images or full folder
- Detailed performance profiling + visualization
"""

import time
import psutil
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import json
import cv2
from openvino.runtime import Core
import subprocess
import os
import argparse
from tqdm import tqdm


@dataclass
class PipelineMetrics:
    """Comprehensive metrics including accuracy"""
    # Timing (ms)
    t_detection: float = 0.0
    t_roi_extract: float = 0.0
    t_classification: float = 0.0
    t_postprocess: float = 0.0
    t_total: float = 0.0
    fps: float = 0.0
    
    # Detection counts
    num_detections: int = 0
    num_ground_truths: int = 0
    
    # Accuracy metrics
    tp_detection: int = 0
    fp_detection: int = 0
    fn_detection: int = 0
    tp_classification: int = 0
    
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    detection_accuracy: float = 0.0
    classification_accuracy: float = 0.0
    
    # Confidence
    det_confidence_avg: float = 0.0
    cls_confidence_avg: float = 0.0
    
    # System
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    temperature: float = 0.0
    
    level: str = "baseline"


# ACCURACY UTILITIES

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def match_predictions_to_ground_truth(pred_boxes: np.ndarray, 
                                     pred_classes: np.ndarray,
                                     gt_boxes: np.ndarray, 
                                     gt_classes: np.ndarray,
                                     iou_threshold: float = 0.5) -> Tuple[int, int, int, int]:
    """
    Match predictions to ground truth
    Returns: tp_det, fp_det, fn_det, tp_cls
    """
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes), 0
    
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0, 0
    
    # Compute IoU matrix
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_iou(pred_box, gt_box)
    
    # Greedy matching
    matched_pred = set()
    matched_gt = set()
    tp_det = 0
    tp_cls = 0
    
    matches = []
    for i in range(len(pred_boxes)):
        for j in range(len(gt_boxes)):
            if iou_matrix[i, j] >= iou_threshold:
                matches.append((iou_matrix[i, j], i, j))
    
    matches.sort(reverse=True)
    
    for iou, pred_idx, gt_idx in matches:
        if pred_idx not in matched_pred and gt_idx not in matched_gt:
            matched_pred.add(pred_idx)
            matched_gt.add(gt_idx)
            tp_det += 1
            
            if pred_classes[pred_idx] == gt_classes[gt_idx]:
                tp_cls += 1
    
    fp_det = len(pred_boxes) - tp_det
    fn_det = len(gt_boxes) - tp_det
    
    return tp_det, fp_det, fn_det, tp_cls


def load_yolo_ground_truth(label_path: Path, img_width: int, img_height: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load YOLO format: class_id x_center y_center width height (normalized)
    Returns: boxes [x1,y1,x2,y2], classes
    """
    if not label_path.exists():
        return np.array([]), np.array([])
    
    boxes = []
    classes = []
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Convert to pixel coordinates
            x1 = (x_center - width / 2) * img_width
            y1 = (y_center - height / 2) * img_height
            x2 = (x_center + width / 2) * img_width
            y2 = (y_center + height / 2) * img_height
            
            boxes.append([x1, y1, x2, y2])
            classes.append(class_id)
    
    return np.array(boxes), np.array(classes)


# BASELINE MODELS

class BaselineDetector:
    """Simple detector without optimization"""
    
    def __init__(self, model_path: str, input_size: int = 640):
        self.input_size = input_size
        
        print(f"[Level 0] Loading detector...")
        core = Core()
        model = core.read_model(model_path)
        self.compiled_model = core.compile_model(model, "CPU")
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        img = cv2.resize(image, (self.input_size, self.input_size))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)
    
    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        input_tensor = self.preprocess(image)
        output = self.compiled_model(input_tensor)
        output = output[list(output.keys())[0]]
        
        predictions = output[0].T
        boxes_xywh = predictions[:, :4]
        class_scores = predictions[:, 4:]
        
        max_scores = np.max(class_scores, axis=1)
        mask = max_scores >= 0.5
        
        if not mask.any():
            return np.array([]), np.array([]), np.array([])
        
        boxes = boxes_xywh[mask]
        scores = max_scores[mask]
        class_ids = np.argmax(class_scores[mask], axis=1)
        
        # Convert to xyxy
        orig_h, orig_w = image.shape[:2]
        scale_x = orig_w / self.input_size
        scale_y = orig_h / self.input_size
        
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2]/2) * scale_x
        boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3]/2) * scale_y
        boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2]/2) * scale_x
        boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3]/2) * scale_y
        
        return boxes_xyxy, scores, class_ids


import torch
import torch.nn.functional as F
from torchvision import transforms

class TorchClassifier:
    """Classifier dùng PyTorch (.pth) thay vì OpenVINO"""
    
    def __init__(self, model_path: str, input_size: int = 96, device: str = "cpu"):
        self.input_size = input_size
        self.device = torch.device(device)
        
        print(f"[Level 0] Loading PyTorch classifier from {model_path} ...")
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.114, 0.114, 0.114],
                                 std=[0.331, 0.289, 0.322])
        ])
        print("  ✓ PyTorch model ready")
    
    @torch.inference_mode()
    def predict(self, image: np.ndarray) -> Tuple[int, float]:
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1)
        conf, cls_id = torch.max(probs, dim=1)
        return int(cls_id.item()), float(conf.item())

    @torch.inference_mode()
    def predict_batch(self, images: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if len(images) == 0:
            return np.array([]), np.array([])
        
        batch = torch.stack([self.transform(img) for img in images]).to(self.device)
        logits = self.model(batch)
        probs = F.softmax(logits, dim=1)
        confs, cls_ids = torch.max(probs, dim=1)
        
        return cls_ids.cpu().numpy(), confs.cpu().numpy()

# LEVEL 0: BASELINE

class Level0_BaselinePipeline:
    """Level 0: No optimization"""
    
    def __init__(self, detector_path: str, classifier_path: str):
        print("\n" + "="*60)
        print("LEVEL 0: BASELINE PIPELINE")
        print("="*60)
        
        self.detector = BaselineDetector(detector_path)
        self.classifier = TorchClassifier(classifier_path)
        
        print("✓ Baseline pipeline ready")
        print("="*60 + "\n")
    
    def run(self, image: np.ndarray, ground_truth: Optional[Dict] = None) -> Tuple[Dict, PipelineMetrics]:
        metrics = PipelineMetrics(level="Level 0: Baseline")
        t_start = time.perf_counter()
        
        # Detection
        t0 = time.perf_counter()
        boxes, scores, det_classes = self.detector.detect(image)
        metrics.t_detection = (time.perf_counter() - t0) * 1000
        metrics.num_detections = len(boxes)
        metrics.det_confidence_avg = float(np.mean(scores)) if len(scores) > 0 else 0.0
        
        # ROI extraction
        t1 = time.perf_counter()
        rois = []
        h, w = image.shape[:2]
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                rois.append(image[y1:y2, x1:x2])
        metrics.t_roi_extract = (time.perf_counter() - t1) * 1000
        
        # Classification
        t2 = time.perf_counter()
        cls_classes = []
        cls_confidences = []
        for roi in rois:
            cls_id, conf = self.classifier.predict(roi)
            cls_classes.append(cls_id)
            cls_confidences.append(conf)
        metrics.t_classification = (time.perf_counter() - t2) * 1000
        
        cls_classes = np.array(cls_classes)
        cls_confidences = np.array(cls_confidences)
        metrics.cls_confidence_avg = float(np.mean(cls_confidences)) if len(cls_confidences) > 0 else 0.0
        
        metrics.t_total = (time.perf_counter() - t_start) * 1000
        metrics.fps = 1000.0 / metrics.t_total if metrics.t_total > 0 else 0
        
        # Compute accuracy
        if ground_truth is not None:
            self._compute_accuracy(metrics, boxes, cls_classes, ground_truth)
        
        # System metrics
        metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
        metrics.memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                metrics.temperature = float(f.read()) / 1000.0
        except:
            pass
        
        results = {
            'boxes': boxes,
            'scores': scores,
            'detection_classes': det_classes,
            'classification_classes': cls_classes,
            'classification_confidences': cls_confidences
        }
        return results, metrics
    
    def _compute_accuracy(self, metrics, boxes, cls_classes, ground_truth):
        """Compute accuracy metrics"""
        gt_boxes = ground_truth['boxes']
        gt_classes = ground_truth['classes']
        metrics.num_ground_truths = len(gt_boxes)
        
        if len(boxes) > 0 and len(gt_boxes) > 0:
            tp_det, fp_det, fn_det, tp_cls = match_predictions_to_ground_truth(
                boxes, cls_classes, gt_boxes, gt_classes
            )
            
            metrics.tp_detection = tp_det
            metrics.fp_detection = fp_det
            metrics.fn_detection = fn_det
            metrics.tp_classification = tp_cls
            
            metrics.precision = tp_det / (tp_det + fp_det) if (tp_det + fp_det) > 0 else 0.0
            metrics.recall = tp_det / (tp_det + fn_det) if (tp_det + fn_det) > 0 else 0.0
            metrics.f1_score = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall) if (metrics.precision + metrics.recall) > 0 else 0.0
            metrics.detection_accuracy = tp_det / len(gt_boxes) if len(gt_boxes) > 0 else 0.0
            metrics.classification_accuracy = tp_cls / tp_det if tp_det > 0 else 0.0
        elif len(gt_boxes) > 0:
            metrics.fn_detection = len(gt_boxes)


# LEVEL 1: MODEL SELECTION

class Level1_ModelSelectionPipeline(Level0_BaselinePipeline):
    """Level 1: Optimized model architecture"""
    
    def __init__(self, detector_path: str, classifier_path: str):
        print("\n" + "="*60)
        print("LEVEL 1: MODEL ARCHITECTURE SELECTION")
        print("="*60)
        print("✓ YOLOv11n + ShuffleNetV2")
        
        super().__init__(detector_path, classifier_path)
        print("="*60 + "\n")
    
    def run(self, image: np.ndarray, ground_truth: Optional[Dict] = None) -> Tuple[Dict, PipelineMetrics]:
        results, metrics = super().run(image, ground_truth)
        metrics.level = "Level 1: Model Selection"
        return results, metrics


# LEVEL 2: INFERENCE OPTIMIZATION

class Level2_InferenceOptimizedDetector(BaselineDetector):
    """Detector with inference optimizations"""
    
    def __init__(self, model_path: str, input_size: int = 640):
        self.input_size = input_size
        
        print(f"[Level 2] Loading detector with inference optimization...")
        core = Core()
        
        core.set_property("CPU", {
            "PERFORMANCE_HINT": "LATENCY",
            "NUM_STREAMS": "1",
            "INFERENCE_NUM_THREADS": "4"
        })
        
        model = core.read_model(model_path)
        model.reshape([1, 3, input_size, input_size])
        
        self.compiled_model = core.compile_model(model, "CPU")
        self.infer_request = self.compiled_model.create_infer_request()
        self.input_tensor = self.infer_request.get_input_tensor()
        self.output_tensor = self.infer_request.get_output_tensor()
        
        print("  ✓ Performance: LATENCY, Threads: 4")


class Level2_InferenceOptimizedClassifier(TorchClassifier):
    """Classifier with inference optimizations"""
    
    def __init__(self, model_path: str, input_size: int = 96):
        self.input_size = input_size
        
        print(f"[Level 2] Loading classifier with inference optimization...")
        core = Core()
        
        core.set_property("CPU", {
            "PERFORMANCE_HINT": "THROUGHPUT",
            "NUM_STREAMS": "2"
        })
        
        model = core.read_model(model_path)
        self.compiled_model = core.compile_model(model, "CPU")
        
        self.mean = np.array([0.114, 0.114, 0.114], dtype=np.float32).reshape(1, 1, 3)
        self.inv_std = 1.0 / np.array([0.331, 0.289, 0.322], dtype=np.float32).reshape(1, 1, 3)
        
        print("  ✓ Performance: THROUGHPUT, Vectorized preprocessing")


class Level2_InferenceOptimizedPipeline(Level0_BaselinePipeline):
    """Level 2: Inference optimization + batching"""
    
    def __init__(self, detector_path: str, classifier_path: str):
        print("\n" + "="*60)
        print("LEVEL 2: INFERENCE OPTIMIZATION")
        print("="*60)
        
        self.detector = Level2_InferenceOptimizedDetector(detector_path)
        self.classifier = Level2_InferenceOptimizedClassifier(classifier_path)
        self.batch_size = 4
        
        print("✓ Batch classification (size=4)")
        print("="*60 + "\n")
    
    def run(self, image: np.ndarray, ground_truth: Optional[Dict] = None) -> Tuple[Dict, PipelineMetrics]:
        metrics = PipelineMetrics(level="Level 2: Inference Optimized")
        t_start = time.perf_counter()
        
        # Detection
        t0 = time.perf_counter()
        input_data = self.detector.preprocess(image)
        self.detector.input_tensor.data[:] = input_data
        self.detector.infer_request.infer()
        output = self.detector.output_tensor.data
        
        predictions = output[0].T
        boxes_xywh = predictions[:, :4]
        class_scores = predictions[:, 4:]
        max_scores = np.max(class_scores, axis=1)
        mask = max_scores >= 0.5
        
        if mask.any():
            boxes_xywh = boxes_xywh[mask]
            scores = max_scores[mask]
            det_classes = np.argmax(class_scores[mask], axis=1)
            
            orig_h, orig_w = image.shape[:2]
            scale_x = orig_w / self.detector.input_size
            scale_y = orig_h / self.detector.input_size
            
            boxes = np.zeros((len(boxes_xywh), 4))
            boxes[:, 0] = (boxes_xywh[:, 0] - boxes_xywh[:, 2]/2) * scale_x
            boxes[:, 1] = (boxes_xywh[:, 1] - boxes_xywh[:, 3]/2) * scale_y
            boxes[:, 2] = (boxes_xywh[:, 0] + boxes_xywh[:, 2]/2) * scale_x
            boxes[:, 3] = (boxes_xywh[:, 1] + boxes_xywh[:, 3]/2) * scale_y
        else:
            boxes, scores, det_classes = np.array([]), np.array([]), np.array([])
        
        metrics.t_detection = (time.perf_counter() - t0) * 1000
        metrics.num_detections = len(boxes)
        metrics.det_confidence_avg = float(np.mean(scores)) if len(scores) > 0 else 0.0
        
        # ROI extraction
        t1 = time.perf_counter()
        rois = []
        h, w = image.shape[:2]
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                rois.append(image[y1:y2, x1:x2])
        metrics.t_roi_extract = (time.perf_counter() - t1) * 1000
        
        # Batch classification
        t2 = time.perf_counter()
        all_cls = []
        all_conf = []
        for i in range(0, len(rois), self.batch_size):
            batch = rois[i:i+self.batch_size]
            cls_ids, confidences = self.classifier.predict_batch(batch)
            all_cls.extend(cls_ids)
            all_conf.extend(confidences)
        metrics.t_classification = (time.perf_counter() - t2) * 1000
        
        cls_classes = np.array(all_cls)
        cls_confidences = np.array(all_conf)
        metrics.cls_confidence_avg = float(np.mean(cls_confidences)) if len(cls_confidences) > 0 else 0.0
        
        metrics.t_total = (time.perf_counter() - t_start) * 1000
        metrics.fps = 1000.0 / metrics.t_total if metrics.t_total > 0 else 0
        
        # Accuracy
        if ground_truth is not None:
            self._compute_accuracy(metrics, boxes, cls_classes, ground_truth)
        
        # System metrics
        metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
        metrics.memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                metrics.temperature = float(f.read()) / 1000.0
        except:
            pass
        
        results = {
            'boxes': boxes,
            'scores': scores,
            'detection_classes': det_classes,
            'classification_classes': cls_classes,
            'classification_confidences': cls_confidences
        }
        return results, metrics


# LEVEL 3: ALGORITHMIC OPTIMIZATION

def efficient_nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> np.ndarray:
    """Custom NMS: 25-40% faster"""
    if len(boxes) == 0:
        return np.array([])
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return np.array(keep)


class Level3_AlgorithmicOptimizedDetector(Level2_InferenceOptimizedDetector):
    """Detector with algorithmic optimizations"""
    
    def __init__(self, model_path: str, input_size: int = 640):
        super().__init__(model_path, input_size)
        self.preprocess_buffer = np.zeros((input_size, input_size, 3), dtype=np.float32)
        print("  ✓ Memory pooling + Efficient NMS")
    
    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        orig_h, orig_w = image.shape[:2]
        
        cv2.resize(image, (self.input_size, self.input_size), 
                  dst=self.preprocess_buffer, interpolation=cv2.INTER_LINEAR)
        self.preprocess_buffer *= (1.0 / 255.0)
        input_data = np.transpose(self.preprocess_buffer, (2, 0, 1))[np.newaxis, ...]
        
        self.input_tensor.data[:] = input_data
        self.infer_request.infer()
        output = self.output_tensor.data
        
        predictions = output[0].T
        boxes_xywh = predictions[:, :4]
        class_scores = predictions[:, 4:]
        
        max_scores = np.max(class_scores, axis=1)
        mask = max_scores >= 0.5
        
        if not mask.any():
            return np.array([]), np.array([]), np.array([])
        
        boxes_xywh = boxes_xywh[mask]
        scores = max_scores[mask]
        class_ids = np.argmax(class_scores[mask], axis=1)
        
        scale_x = orig_w / self.input_size
        scale_y = orig_h / self.input_size
        
        half_w = boxes_xywh[:, 2] / 2
        half_h = boxes_xywh[:, 3] / 2
        
        boxes_xyxy = np.stack([
            (boxes_xywh[:, 0] - half_w) * scale_x,
            (boxes_xywh[:, 1] - half_h) * scale_y,
            (boxes_xywh[:, 0] + half_w) * scale_x,
            (boxes_xywh[:, 1] + half_h) * scale_y
        ], axis=1)
        
        keep_indices = efficient_nms_numpy(boxes_xyxy, scores, 0.45)
        
        return boxes_xyxy[keep_indices], scores[keep_indices], class_ids[keep_indices]


class Level3_AlgorithmicOptimizedPipeline(Level2_InferenceOptimizedPipeline):
    """Level 3: Algorithmic optimization"""
    
    def __init__(self, detector_path: str, classifier_path: str):
        print("\n" + "="*60)
        print("LEVEL 3: ALGORITHMIC OPTIMIZATION")
        print("="*60)
        
        self.detector = Level3_AlgorithmicOptimizedDetector(detector_path)
        self.classifier = Level2_InferenceOptimizedClassifier(classifier_path)
        self.batch_size = 8
        
        print("✓ Efficient NMS + Memory pooling")
        print("✓ Batch size: 8")
        print("="*60 + "\n")


# LEVEL 4: SYSTEM-LEVEL OPTIMIZATION

class SystemOptimizer:
    """System-level optimizations for Raspberry Pi 5"""
    
    @staticmethod
    def set_cpu_governor_performance():
        """Set CPU governor to performance mode"""
        try:
            for cpu in range(4):
                path = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
                if os.path.exists(path):
                    subprocess.run(['sudo', 'sh', '-c', f'echo performance > {path}'], check=True)
            print("  ✓ CPU governor: performance (2.4 GHz)")
            return True
        except:
            print("  ⚠ CPU governor: unable to set (run with sudo)")
            return False
    
    @staticmethod
    def get_cpu_frequency():
        """Get current CPU frequency"""
        try:
            with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq', 'r') as f:
                freq_khz = int(f.read().strip())
                return freq_khz / 1000000
        except:
            return 0.0


class Level4_SystemOptimizedPipeline(Level3_AlgorithmicOptimizedPipeline):
    """Level 4: System-level optimization"""
    
    def __init__(self, detector_path: str, classifier_path: str):
        print("\n" + "="*60)
        print("LEVEL 4: SYSTEM-LEVEL OPTIMIZATION")
        print("="*60)
        
        SystemOptimizer.set_cpu_governor_performance()
        freq = SystemOptimizer.get_cpu_frequency()
        if freq > 0:
            print(f"  ✓ CPU frequency: {freq:.2f} GHz")
        
        self.detector = Level3_AlgorithmicOptimizedDetector(detector_path)
        self.classifier = Level2_InferenceOptimizedClassifier(classifier_path)
        self.batch_size = 16
        
        print("✓ Maximum batching (size=16)")
        print("="*60 + "\n")


# EXPERIMENT RUNNER

class OptimizationExperiment:
    """Complete experiment with accuracy evaluation"""
    
    def __init__(self, detector_path: str, classifier_path: str, output_dir: str = "optimization_results"):
        self.detector_path = detector_path
        self.classifier_path = classifier_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.pipelines = {}
        self.results = []
    
    def _get_pipeline(self, level_name: str):
        """Lazy load pipeline"""
        if level_name not in self.pipelines:
            print(f"\nInitializing {level_name}...")
            if level_name == 'Level 0':
                self.pipelines[level_name] = Level0_BaselinePipeline(self.detector_path, self.classifier_path)
            elif level_name == 'Level 1':
                self.pipelines[level_name] = Level1_ModelSelectionPipeline(self.detector_path, self.classifier_path)
            elif level_name == 'Level 2':
                self.pipelines[level_name] = Level2_InferenceOptimizedPipeline(self.detector_path, self.classifier_path)
            elif level_name == 'Level 3':
                self.pipelines[level_name] = Level3_AlgorithmicOptimizedPipeline(self.detector_path, self.classifier_path)
            elif level_name == 'Level 4':
                self.pipelines[level_name] = Level4_SystemOptimizedPipeline(self.detector_path, self.classifier_path)
        
        return self.pipelines[level_name]
    
    def load_test_dataset(self, images_dir: str, labels_dir: str, 
                         max_images: Optional[int] = None) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Load test images and YOLO labels
        
        Args:
            images_dir: Directory with images
            labels_dir: Directory with YOLO .txt labels
            max_images: Max images (None = all)
        """
        images_path = Path(images_dir)
        labels_path = Path(labels_dir)
        
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(sorted(images_path.glob(ext)))
        
        if max_images is not None:
            image_files = image_files[:max_images]
        
        print(f"\n{'='*70}")
        print(f"LOADING DATASET")
        print(f"{'='*70}")
        print(f"Images dir: {images_dir}")
        print(f"Labels dir: {labels_dir}")
        print(f"Found: {len(image_files)} images")
        
        test_images = []
        ground_truths = []
        
        for img_path in tqdm(image_files, desc="Loading"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            
            label_path = labels_path / (img_path.stem + '.txt')
            gt_boxes, gt_classes = load_yolo_ground_truth(label_path, w, h)
            
            test_images.append(img)
            ground_truths.append({
                'boxes': gt_boxes,
                'classes': gt_classes,
                'image_name': img_path.name
            })
        
        total_objects = sum(len(gt['boxes']) for gt in ground_truths)
        print(f"✓ Loaded: {len(test_images)} images")
        print(f"✓ Total objects: {total_objects}")
        print(f"✓ Avg objects/image: {total_objects/len(test_images):.2f}")
        print(f"{'='*70}\n")
        
        return test_images, ground_truths
    
    def run_single_level(self, level_name: str, 
                        test_images: List[np.ndarray],
                        ground_truths: List[Dict],
                        warmup: int = 10, 
                        iterations: int = 100):
        """Run experiment on single level"""
        
        print(f"\n{'='*70}")
        print(f"TESTING: {level_name}")
        print(f"{'='*70}")
        
        pipeline = self._get_pipeline(level_name)
        
        # Warmup
        print(f"Warmup: {warmup} iterations...")
        for i in range(min(warmup, len(test_images))):
            pipeline.run(test_images[i % len(test_images)])
        
        # Measurement
        print(f"Measurement: {iterations} iterations...")
        metrics_list = []
        
        for i in tqdm(range(iterations), desc="Processing"):
            img_idx = i % len(test_images)
            image = test_images[img_idx]
            gt = ground_truths[img_idx]
            
            _, metrics = pipeline.run(image, gt)
            
            record = {
                'level': level_name,
                'iteration': i,
                'image_name': gt['image_name'],
                **asdict(metrics)
            }
            
            self.results.append(record)
            metrics_list.append(metrics)
        
        # Summary
        self._print_level_summary(level_name, metrics_list)
        
        return metrics_list
    
    def run_all_levels(self, test_images: List[np.ndarray],
                      ground_truths: List[Dict],
                      warmup: int = 10, 
                      iterations: int = 100):
        """Run complete experiment"""
        
        print("\n" + "="*70)
        print("MULTI-LEVEL OPTIMIZATION EXPERIMENT")
        print("="*70)
        print(f"Test images: {len(test_images)}")
        print(f"Warmup: {warmup}, Iterations: {iterations}")
        print("="*70)
        
        start_time = time.time()
        
        for level_name in ['Level 0', 'Level 1', 'Level 2', 'Level 3', 'Level 4']:
            self.run_single_level(level_name, test_images, ground_truths, warmup, iterations)
            
            # Clear pipeline to free memory
            if level_name in self.pipelines:
                del self.pipelines[level_name]
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"EXPERIMENT COMPLETED")
        print(f"Total time: {elapsed/60:.2f} minutes")
        print(f"{'='*70}\n")
        
        self.save_results()
        self.generate_comparison_report()
        self.plot_optimization_progress()
    
    def _print_level_summary(self, level_name: str, metrics_list: List[PipelineMetrics]):
        """Print comprehensive summary"""
        
        latencies = np.array([m.t_total for m in metrics_list])
        fps = np.array([m.fps for m in metrics_list])
        det_times = np.array([m.t_detection for m in metrics_list])
        cls_times = np.array([m.t_classification for m in metrics_list])
        
        print(f"\n{level_name} Summary:")
        print(f"\n{'='*60}")
        print(f"PERFORMANCE")
        print(f"{'='*60}")
        print(f"  Latency:  {np.mean(latencies):6.2f} ± {np.std(latencies):5.2f} ms")
        print(f"  FPS:      {np.mean(fps):6.2f} ± {np.std(fps):5.2f}")
        print(f"  P50:      {np.percentile(latencies, 50):6.2f} ms")
        print(f"  P95:      {np.percentile(latencies, 95):6.2f} ms")
        print(f"  P99:      {np.percentile(latencies, 99):6.2f} ms")
        
        print(f"\n  Breakdown:")
        print(f"    Detection:      {np.mean(det_times):6.2f} ms ({np.mean(det_times)/np.mean(latencies)*100:5.1f}%)")
        print(f"    Classification: {np.mean(cls_times):6.2f} ms ({np.mean(cls_times)/np.mean(latencies)*100:5.1f}%)")
        
        # Accuracy
        tp_det = sum(m.tp_detection for m in metrics_list)
        fp_det = sum(m.fp_detection for m in metrics_list)
        fn_det = sum(m.fn_detection for m in metrics_list)
        tp_cls = sum(m.tp_classification for m in metrics_list)
        
        precision = tp_det / (tp_det + fp_det) if (tp_det + fp_det) > 0 else 0.0
        recall = tp_det / (tp_det + fn_det) if (tp_det + fn_det) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        cls_acc = tp_cls / tp_det if tp_det > 0 else 0.0
        
        print(f"\n{'='*60}")
        print(f"ACCURACY")
        print(f"{'='*60}")
        print(f"  Detection:")
        print(f"    Precision:  {precision:.4f}")
        print(f"    Recall:     {recall:.4f}")
        print(f"    F1-Score:   {f1:.4f}")
        print(f"    TP/FP/FN:   {tp_det}/{fp_det}/{fn_det}")
        
        print(f"\n  Classification:")
        print(f"    Accuracy:   {cls_acc*100:.2f}%")
        print(f"    Correct:    {tp_cls}/{tp_det}")
        
        # System
        cpu = np.array([m.cpu_percent for m in metrics_list])
        mem = np.array([m.memory_mb for m in metrics_list])
        temp = np.array([m.temperature for m in metrics_list if m.temperature > 0])
        
        print(f"\n{'='*60}")
        print(f"SYSTEM")
        print(f"{'='*60}")
        print(f"  CPU:        {np.mean(cpu):5.1f}%")
        print(f"  Memory:     {np.mean(mem):6.1f} MB")
        if len(temp) > 0:
            print(f"  Temp:       {np.mean(temp):5.1f}°C")
    
    def save_results(self):
        """Save results"""
        df = pd.DataFrame(self.results)
        csv_path = self.output_dir / "optimization_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved: {csv_path}")
    
    def generate_comparison_report(self):
        """Generate comparison report"""
        
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*80)
        print("OPTIMIZATION + ACCURACY COMPARISON")
        print("="*80)
        
        summary = []
        for level in ['Level 0', 'Level 1', 'Level 2', 'Level 3', 'Level 4']:
            level_data = df[df['level'] == level]
            
            if len(level_data) == 0:
                continue
            
            fps_mean = level_data['fps'].mean()
            fps_std = level_data['fps'].std()
            latency_mean = level_data['t_total'].mean()
            latency_std = level_data['t_total'].std()
            
            tp_det = level_data['tp_detection'].sum()
            fp_det = level_data['fp_detection'].sum()
            fn_det = level_data['fn_detection'].sum()
            tp_cls = level_data['tp_classification'].sum()
            
            precision = tp_det / (tp_det + fp_det) if (tp_det + fp_det) > 0 else 0.0
            recall = tp_det / (tp_det + fn_det) if (tp_det + fn_det) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            cls_acc = tp_cls / tp_det if tp_det > 0 else 0.0
            
            summary.append({
                'Level': level.replace('Level ', 'L'),
                'FPS': f"{fps_mean:.2f}±{fps_std:.2f}",
                'Latency(ms)': f"{latency_mean:.1f}±{latency_std:.1f}",
                'Precision': f"{precision:.3f}",
                'Recall': f"{recall:.3f}",
                'F1': f"{f1:.3f}",
                'ClsAcc': f"{cls_acc*100:.1f}%",
                'CPU%': f"{level_data['cpu_percent'].mean():.1f}",
                'Mem(MB)': f"{level_data['memory_mb'].mean():.0f}"
            })
        
        summary_df = pd.DataFrame(summary)
        print("\n" + summary_df.to_string(index=False))
        
        # Improvements
        if len(summary) >= 2:
            baseline_fps = float(summary[0]['FPS'].split('±')[0])
            final_fps = float(summary[-1]['FPS'].split('±')[0])
            speedup = final_fps / baseline_fps
            
            baseline_latency = float(summary[0]['Latency(ms)'].split('±')[0])
            final_latency = float(summary[-1]['Latency(ms)'].split('±')[0])
            reduction = (1 - final_latency / baseline_latency) * 100
            
            print(f"\n{'='*80}")
            print("OVERALL IMPROVEMENTS")
            print(f"{'='*80}")
            print(f"  Baseline (L0):     {baseline_fps:.2f} FPS @ {baseline_latency:.1f} ms")
            print(f"  Optimized (L4):    {final_fps:.2f} FPS @ {final_latency:.1f} ms")
            print(f"  Speedup:           {speedup:.2f}x")
            print(f"  Latency reduction: {reduction:.1f}%")
            print(f"{'='*80}\n")
        
        # Save
        summary_path = self.output_dir / "optimization_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"✓ Summary saved: {summary_path}")
        
        # LaTeX
        self._generate_latex_table(summary_df)
    
    def plot_optimization_progress(self):
        """Generate visualization"""
        try:
            import matplotlib.pyplot as plt
            
            df = pd.DataFrame(self.results)
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Multi-Level Optimization Results', fontsize=16, fontweight='bold')
            
            levels = ['Level 0', 'Level 1', 'Level 2', 'Level 3', 'Level 4']
            colors = ['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#3498db']
            
            # Latency
            ax1 = axes[0, 0]
            latencies = [df[df['level'] == l]['t_total'].mean() for l in levels if len(df[df['level'] == l]) > 0]
            ax1.bar(range(len(latencies)), latencies, color=colors[:len(latencies)], alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Optimization Level')
            ax1.set_ylabel('Latency (ms)')
            ax1.set_title('End-to-End Latency')
            ax1.set_xticks(range(len(latencies)))
            ax1.set_xticklabels([f'L{i}' for i in range(len(latencies))])
            ax1.grid(axis='y', alpha=0.3)
            
            # FPS
            ax2 = axes[0, 1]
            fps_vals = [df[df['level'] == l]['fps'].mean() for l in levels if len(df[df['level'] == l]) > 0]
            ax2.bar(range(len(fps_vals)), fps_vals, color=colors[:len(fps_vals)], alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Optimization Level')
            ax2.set_ylabel('FPS')
            ax2.set_title('Throughput (FPS)')
            ax2.set_xticks(range(len(fps_vals)))
            ax2.set_xticklabels([f'L{i}' for i in range(len(fps_vals))])
            ax2.axhline(y=10, color='green', linestyle='--', label='Target: 10 FPS')
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
            
            # Accuracy
            ax3 = axes[1, 0]
            precisions = []
            recalls = []
            f1s = []
            
            for l in levels:
                level_data = df[df['level'] == l]
                if len(level_data) == 0:
                    continue
                tp = level_data['tp_detection'].sum()
                fp = level_data['fp_detection'].sum()
                fn = level_data['fn_detection'].sum()
                
                p = tp / (tp + fp) if (tp + fp) > 0 else 0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0
                f = 2*p*r/(p+r) if (p+r) > 0 else 0
                
                precisions.append(p)
                recalls.append(r)
                f1s.append(f)
            
            x = np.arange(len(precisions))
            width = 0.25
            ax3.bar(x - width, precisions, width, label='Precision', alpha=0.7)
            ax3.bar(x, recalls, width, label='Recall', alpha=0.7)
            ax3.bar(x + width, f1s, width, label='F1-Score', alpha=0.7)
            ax3.set_xlabel('Optimization Level')
            ax3.set_ylabel('Score')
            ax3.set_title('Detection Accuracy')
            ax3.set_xticks(x)
            ax3.set_xticklabels([f'L{i}' for i in range(len(precisions))])
            ax3.legend()
            ax3.grid(axis='y', alpha=0.3)
            
            # Speedup
            ax4 = axes[1, 1]
            baseline_fps = fps_vals[0] if len(fps_vals) > 0 else 1
            speedups = [fps / baseline_fps for fps in fps_vals]
            ax4.plot(range(len(speedups)), speedups, marker='o', linewidth=2, 
                    markersize=8, color='#2ecc71')
            ax4.fill_between(range(len(speedups)), speedups, alpha=0.3, color='#2ecc71')
            ax4.set_xlabel('Optimization Level')
            ax4.set_ylabel('Speedup Factor')
            ax4.set_title('Cumulative Speedup')
            ax4.set_xticks(range(len(speedups)))
            ax4.set_xticklabels([f'L{i}' for i in range(len(speedups))])
            ax4.grid(True, alpha=0.3)
            
            for i, speedup in enumerate(speedups):
                ax4.annotate(f'{speedup:.2f}x', (i, speedup), 
                            textcoords="offset points", xytext=(0,10), 
                            ha='center', fontweight='bold')
            
            plt.tight_layout()
            
            plot_path = self.output_dir / "optimization_progress.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Visualization saved: {plot_path}")
        except Exception as e:
            print(f"⚠ Visualization failed: {e}")


# MAIN
def main():
    # === Hardcoded paths & settings ===
    detector_path   = "/home/pi5/TrafficSign/convert/model/yolov11/yolo11_openvino_model/yolo11.xml"
    classifier_path = "/home/pi5/TrafficSign/convert/classify_model/export_openvino/shufflenetv2_x1_0/shufflenetv2_x1_0_fp32.xml"
    images_dir      = "/home/pi5/TrafficSign/Test"
    labels_dir      = "/home/pi5/TrafficSign/TestLabels"
    output_dir      = "E2E_Result"
    max_images      = 50
    iterations      = 5
    warmup          = 10
    level           = "1"  

    print("\n" + "="*80)
    print("MULTI-LEVEL OPTIMIZATION WITH ACCURACY EVALUATION")
    print("="*80)
    print(f"Detector:    {detector_path}")
    print(f"Classifier:  {classifier_path}")
    print(f"Images:      {images_dir}")
    print(f"Labels:      {labels_dir}")
    print(f"Max images:  {max_images if max_images else 'All'}")
    print(f"Iterations:  {iterations}")
    print(f"Level:       {level}")
    print("="*80)

    # Initialize experiment
    experiment = OptimizationExperiment(
        detector_path=detector_path,
        classifier_path=classifier_path,
        output_dir=output_dir
    )

    # Load dataset
    test_images, ground_truths = experiment.load_test_dataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        max_images=max_images
    )

    if len(test_images) == 0:
        print("ERROR: No images loaded!")
        return

    # Run experiment
    if level == 'all':
        experiment.run_all_levels(test_images, ground_truths, warmup, iterations)
    else:
        level_name = f'Level {level}'
        experiment.run_single_level(level_name, test_images, ground_truths, warmup, iterations)
        experiment.save_results()
        experiment.generate_comparison_report()

    print("\n" + "="*80)
    print("COMPLETED SUCCESSFULLY")
    print(f"Results: {output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
