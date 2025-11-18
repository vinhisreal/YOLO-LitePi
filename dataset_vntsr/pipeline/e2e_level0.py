#!/usr/bin/env python3
"""
End-to-End Pipeline: YOLO Detection + Classification
- Supports YOLOv5, YOLOv8, YOLOv11 (.pt)
- Supports ResNet18, EfficientNet, MobileNetV2, ShuffleNetV2 (.pth)
- Process single image or folder with YOLO-format labels
- Output: Metrics (Precision, Recall, F1, mAP) + Visualization
- CSV naming: {yolo_name}+{clf_arch}_results.csv for easy comparison
"""

import os
import argparse
from pathlib import Path
import json
import time
import random

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO

# ==================== MODEL BUILDERS ====================
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

def get_transform():
    """Standard ImageNet normalization"""
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.18, 0.18, 0.18], [0.34, 0.34, 0.34])
    ])

# ==================== LABEL PARSER ====================
def parse_yolo_label(label_path, img_w, img_h):
    """
    Parse YOLO format label: class_id x_center y_center width height (normalized)
    Returns: list of (class_id, x1, y1, x2, y2) in pixel coordinates
    """
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
            
            # Convert to pixel coordinates
            x1 = int((x_c - w/2) * img_w)
            y1 = int((y_c - h/2) * img_h)
            x2 = int((x_c + w/2) * img_w)
            y2 = int((y_c + h/2) * img_h)
            
            boxes.append((cls_id, x1, y1, x2, y2))
    return boxes

# ==================== CLASS NAME LOADER ====================
def load_class_names(class_file_path):
    """
    Load class names from JSON or text file
    JSON format: {"0": "class_name", "1": "class_name", ...}
    Text format: one class name per line
    Returns: list of class names ordered by index
    """
    with open(class_file_path, 'r') as f:
        # Try to load as JSON first
        try:
            data = json.load(f)
            # If it's a dictionary with string keys (indices)
            if isinstance(data, dict):
                # Convert to list, maintaining order
                max_idx = max(int(k) for k in data.keys())
                class_names = [''] * (max_idx + 1)
                for idx_str, name in data.items():
                    class_names[int(idx_str)] = name
                return class_names
            else:
                raise ValueError("JSON must be a dictionary")
        except json.JSONDecodeError:
            # If not JSON, treat as text file (one class per line)
            f.seek(0)  # Reset file pointer
            class_names = [line.strip() for line in f if line.strip()]
            return class_names

# ==================== DETECTION & CLASSIFICATION ====================
def detect_and_classify(img_bgr, yolo_model, classifier, transform, device, 
                        yolo_conf=0.3, min_area=100):
    """
    Run detection + classification on single image
    Returns: list of predictions with bbox, det_class, cls_class, confidences, timing
    """
    h, w = img_bgr.shape[:2]
    results = []
    
    # YOLO Detection with timing
    t_det_start = time.time()
    yolo_results = yolo_model(img_bgr, conf=yolo_conf, verbose=False)[0]
    t_det = time.time() - t_det_start
    
    boxes = yolo_results.boxes
    
    if boxes is None or len(boxes) == 0:
        return []
    
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    
    for (x1, y1, x2, y2), det_conf, det_cls in zip(xyxy, confs, clss):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        area = (x2 - x1) * (y2 - y1)
        if area < min_area:
            continue
        
        # Crop and classify
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_crop = Image.fromarray(crop_rgb)
        
        # Classification with timing
        t_cls_start = time.time()
        x = transform(pil_crop).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = classifier(x)
            probs = torch.softmax(logits, dim=1)
            cls_conf, cls_idx = torch.max(probs, dim=1)
        t_cls = time.time() - t_cls_start
        
        results.append({
            'bbox': (x1, y1, x2, y2),
            'det_class': int(det_cls),
            'det_conf': float(det_conf),
            'cls_class': int(cls_idx.item()),
            'cls_conf': float(cls_conf.item()),
            'time_det': float(t_det),
            'time_cls': float(t_cls)
        })
    
    return results

# ==================== IOU CALCULATION ====================
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

# ==================== EVALUATION ====================
def evaluate_predictions(all_preds, all_gts, num_classes, iou_threshold=0.5):
    """
    Calculate precision, recall, F1 for each class
    all_preds: list of lists of predictions per image
    all_gts: list of lists of ground truth boxes per image
    """
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    
    for preds, gts in zip(all_preds, all_gts):
        gt_matched = [False] * len(gts)
        
        for pred in preds:
            pred_box = pred['bbox']
            pred_cls = pred['cls_class']
            
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching GT box
            for i, (gt_cls, gx1, gy1, gx2, gy2) in enumerate(gts):
                if gt_matched[i]:
                    continue
                iou = calculate_iou(pred_box, (gx1, gy1, gx2, gy2))
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            # Check if match is good enough
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                gt_cls = gts[best_gt_idx][0]
                gt_matched[best_gt_idx] = True
                
                if pred_cls == gt_cls:
                    tp[pred_cls] += 1
                else:
                    fp[pred_cls] += 1
                    fn[gt_cls] += 1
            else:
                fp[pred_cls] += 1
        
        # Count unmatched ground truths as false negatives
        for i, (gt_cls, _, _, _, _) in enumerate(gts):
            if not gt_matched[i]:
                fn[gt_cls] += 1
    
    # Calculate metrics
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp+fp)!=0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp+fn)!=0)
    f1 = np.divide(2 * precision * recall, precision + recall, 
                   out=np.zeros_like(precision), where=(precision+recall)!=0)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }

# ==================== VISUALIZATION ====================
def visualize_prediction(img_bgr, predictions, ground_truths, class_names, output_path):
    """Draw predictions (green) and ground truths (blue) on image with detailed labels"""
    vis = img_bgr.copy()
    h, w = vis.shape[:2]
    
    # Create semi-transparent overlay for better text visibility
    overlay = vis.copy()
    
    # Draw ground truths in blue
    for idx, (gt_cls, x1, y1, x2, y2) in enumerate(ground_truths):
        # Draw bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 3)
        
        # Prepare label with class name
        label = f"GT: {class_names[gt_cls]}"
        
        # Calculate text size for background
        font_scale = 0.6
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                                       font_scale, thickness)
        
        # Draw background rectangle for text (above bbox)
        text_x = x1
        text_y = max(y1 - 10, text_h + 5)
        cv2.rectangle(vis, (text_x, text_y - text_h - baseline), 
                     (text_x + text_w, text_y + baseline), (255, 0, 0), -1)
        
        # Draw text
        cv2.putText(vis, label, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    # Draw predictions in green
    for idx, pred in enumerate(predictions):
        x1, y1, x2, y2 = pred['bbox']
        cls_idx = pred['cls_class']
        cls_conf = pred['cls_conf']
        det_conf = pred['det_conf']
        
        # Draw bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Prepare detailed label
        label = f"PRED: {class_names[cls_idx]}"
        conf_label = f"Cls:{cls_conf:.2f} Det:{det_conf:.2f}"
        
        # Calculate text size
        font_scale = 0.6
        thickness = 2
        (text_w1, text_h1), baseline1 = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                                         font_scale, thickness)
        (text_w2, text_h2), baseline2 = cv2.getTextSize(conf_label, cv2.FONT_HERSHEY_SIMPLEX, 
                                                         font_scale - 0.1, thickness - 1)
        
        # Draw background rectangles (below bbox)
        text_x = x1
        text_y = min(y2 + text_h1 + 15, h - 5)
        
        # Main label background
        cv2.rectangle(vis, (text_x, text_y - text_h1 - baseline1), 
                     (text_x + text_w1, text_y + baseline1), (0, 255, 0), -1)
        
        # Confidence label background
        conf_y = text_y + text_h2 + 5
        cv2.rectangle(vis, (text_x, conf_y - text_h2 - baseline2), 
                     (text_x + text_w2, conf_y + baseline2), (0, 200, 0), -1)
        
        # Draw text
        cv2.putText(vis, label, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        cv2.putText(vis, conf_label, (text_x, conf_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale - 0.1, (255, 255, 255), thickness - 1)
    
    # Add summary info at top
    summary = f"GT: {len(ground_truths)} | Predictions: {len(predictions)}"
    cv2.rectangle(vis, (5, 5), (400, 35), (0, 0, 0), -1)
    cv2.putText(vis, summary, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite(output_path, vis)
    print(f"    ✓ Saved visualization: {output_path}")

def plot_metrics(metrics, class_names, output_dir, model_combo_name):
    """Plot precision, recall, F1 bar charts and confusion-like visualization"""
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
    Avg Time (Det+Cls): {metrics.get('avg_time', 0):.3f}s
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_combo_name}_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved metrics plot to {output_dir}/{model_combo_name}_metrics.png")

# ==================== MAIN PROCESSING ====================
def process_image(img_path, label_path, yolo_model, classifier, transform, 
                  device, class_names, output_dir, args, save_viz=True):
    """Process single image with ground truth"""
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None, None
    
    h, w = img_bgr.shape[:2]
    
    # Parse ground truth
    ground_truths = parse_yolo_label(label_path, w, h)
    
    # Run detection + classification
    predictions = detect_and_classify(
        img_bgr, yolo_model, classifier, transform, device,
        yolo_conf=args.yolo_conf, min_area=args.min_area
    )
    
    # Visualize (optional, can be disabled for large datasets)
    if save_viz:
        viz_dir = Path(output_dir) / "eval" / f"{args.clf_arch}"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Lưu ảnh với format PNG
        vis_path = viz_dir / f"vis_{img_path.stem}.png"
        visualize_prediction(img_bgr, predictions, ground_truths, class_names, vis_path)
    return predictions, ground_truths

def sample_images(img_files, num_samples, seed=42):
    """Sample images deterministically for fair comparison"""
    if num_samples is None or num_samples <= 0 or num_samples >= len(img_files):
        return img_files
    
    random.seed(seed)  # Fixed seed for reproducibility
    sampled = random.sample(img_files, num_samples)
    return sorted(sampled)  # Sort to ensure consistent order

def extract_model_name(model_path):
    """Extract clean model name from path"""
    return Path(model_path).stem

def main():
    parser = argparse.ArgumentParser(description="E2E YOLO + Classifier Evaluation Pipeline")
    
    # Model paths
    parser.add_argument("--yolo", type=str, default="/home/pi5/TrafficSign/WeightDetectionV2/yolo8.pt", help="Path to YOLO .pt model")
    parser.add_argument("--clf", type=str, default="/home/pi5/TrafficSign/WeightClassifyV2/efficientnetb0_best.pth", help="Path to classifier .pth")
    parser.add_argument("--clf_arch", type=str, choices=["resnet18", "efficientnet", "mobilenetv2", "shufflenetv2"],
                        default="efficientnet", help="Classifier architecture")
    
    # Data
    parser.add_argument("--input", type=str, default="/home/pi5/TrafficSign/Dataset/Detect/Test/images", help="Input image or folder")
    parser.add_argument("--labels", type=str, default="/home/pi5/TrafficSign/Dataset/Detect/Test/data_detect_test/labels", help="Label folder (auto-detect if not specified)")
    parser.add_argument("--classes", type=str, default="/home/pi5/TrafficSign/WeightClassifyV2/idx2label.json", help="Path to class names file (JSON or text)")
    
    # Sampling
    parser.add_argument("--num_samples", type=int, default=5, 
                        help="Number of images to sample from test set (None = use all)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for sampling (ensures fair comparison)")
    
    # Parameters
    parser.add_argument("--yolo_conf", type=float, default=0.3, help="YOLO confidence threshold")
    parser.add_argument("--min_area", type=int, default=100, help="Minimum detection area")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold for matching")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "auto"], default="cpu")
    
    # Output
    parser.add_argument("--output", type=str, default="output_eval", help="Output directory")
    parser.add_argument("--save_viz", action="store_true", help="Save visualization images (slower)")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Load class names (supports both JSON and text formats)
    class_names = load_class_names(args.classes)
    num_classes = len(class_names)
    print(f"Loaded {num_classes} classes from {args.classes}")
    print(f"Sample classes: {class_names[:5]}...")
    
    # Extract model names for combo naming
    yolo_name = extract_model_name(args.yolo)
    model_combo_name = f"{yolo_name}+{args.clf_arch}"
    
    # Load models
    print(f"\n{'='*60}")
    print(f"MODEL COMBINATION: {model_combo_name}")
    print(f"{'='*60}")
    print("Loading YOLO model...")
    yolo_model = YOLO(args.yolo)
    print("Loading classifier...")
    classifier = build_classifier(args.clf_arch, num_classes, args.clf, device)
    transform = get_transform()
    
    # Create output directory
    output_dir = Path(args.output) / model_combo_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image mode
        if args.labels:
            label_path = Path(args.labels) / f"{input_path.stem}.txt"
        else:
            label_path = input_path.parent / "labels" / f"{input_path.stem}.txt"
        
        print(f"\nProcessing single image: {input_path.name}")
        preds, gts = process_image(input_path, label_path, yolo_model, classifier, 
                                    transform, device, class_names, output_dir, args, True)
        
        if preds is not None:
            all_preds = [preds]
            all_gts = [gts]
            processed_files = [input_path.name]
        else:
            print("Failed to process image")
            return
    
    else:
        # Folder mode
        if args.labels:
            label_dir = Path(args.labels)
        else:
            label_dir = input_path / "labels"
        
        # Get all image files
        img_files = sorted(list(input_path.glob("*.jpg")) + 
                          list(input_path.glob("*.png")) + 
                          list(input_path.glob("*.jpeg")))
        
        print(f"\nFound {len(img_files)} total images in {input_path}")
        
        # Sample if requested
        if args.num_samples:
            img_files = sample_images(img_files, args.num_samples, args.seed)
            print(f"Sampled {len(img_files)} images (seed={args.seed}) for testing")
        else:
            print("Using all images for testing")
        
        all_preds = []
        all_gts = []
        processed_files = []
        total_time = 0
        
        for i, img_path in enumerate(img_files, 1):
            label_path = label_dir / f"{img_path.stem}.txt"
            print(f"[{i}/{len(img_files)}] Processing {img_path.name}...", end=" ")
            
            t_start = time.time()
            preds, gts = process_image(img_path, label_path, yolo_model, classifier,
                                        transform, device, class_names, output_dir, args, True)
            t_elapsed = time.time() - t_start
            total_time += t_elapsed
            
            if preds is not None:
                all_preds.append(preds)
                all_gts.append(gts)
                processed_files.append(img_path.name)
                print(f"✓ ({len(preds)} det, {len(gts)} GT) - {t_elapsed:.2f}s")
            else:
                print("✗ Failed")
    
    # Evaluate
    print("\n" + "="*80)
    print(f"EVALUATION RESULTS - {model_combo_name}")
    print("="*80)
    
    metrics = evaluate_predictions(all_preds, all_gts, num_classes, args.iou_threshold)
    
    # Add additional info to metrics
    metrics['num_images'] = len(all_preds)
    if len(all_preds) > 1:
        metrics['avg_time'] = total_time / len(all_preds)
    else:
        avg_times = []
        for preds in all_preds:
            if preds:
                avg_times.append(preds[0].get('time_det', 0) + preds[0].get('time_cls', 0))
        metrics['avg_time'] = np.mean(avg_times) if avg_times else 0
    
    # Print results
    print(f"\n{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 80)
    
    for i, cls_name in enumerate(class_names):
        print(f"{cls_name:<20} {metrics['precision'][i]:>10.3f} {metrics['recall'][i]:>10.3f} "
              f"{metrics['f1'][i]:>10.3f} {int(metrics['tp'][i]):>6} "
              f"{int(metrics['fp'][i]):>6} {int(metrics['fn'][i]):>6}")
    
    print("-" * 80)
    print(f"{'MEAN':<20} {np.mean(metrics['precision']):>10.3f} "
          f"{np.mean(metrics['recall']):>10.3f} {np.mean(metrics['f1']):>10.3f}")
    
    # Save detailed results with model combo name in filename
    results_df = pd.DataFrame({
        'class': class_names,
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'tp': metrics['tp'].astype(int),
        'fp': metrics['fp'].astype(int),
        'fn': metrics['fn'].astype(int)
    })
    
    csv_path = output_dir / f"{model_combo_name}_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved detailed results to {csv_path}")
    
    # Save summary for comparison
    summary_df = pd.DataFrame([{
        'model_combination': model_combo_name,
        'yolo_model': yolo_name,
        'classifier': args.clf_arch,
        'num_test_images': len(all_preds),
        'mean_precision': np.mean(metrics['precision']),
        'mean_recall': np.mean(metrics['recall']),
        'mean_f1': np.mean(metrics['f1']),
        'total_tp': int(np.sum(metrics['tp'])),
        'total_fp': int(np.sum(metrics['fp'])),
        'total_fn': int(np.sum(metrics['fn'])),
        'avg_inference_time': metrics['avg_time'],
        'yolo_conf_threshold': args.yolo_conf,
        'iou_threshold': args.iou_threshold,
        'random_seed': args.seed if args.num_samples else 'N/A'
    }])
    
    summary_path = Path(args.output) / "comparison_summary.csv"
    if summary_path.exists():
        # Append to existing summary
        existing_df = pd.read_csv(summary_path)
        summary_df = pd.concat([existing_df, summary_df], ignore_index=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Updated comparison summary at {summary_path}")
    
    # Save test file list for reproducibility
    test_files_path = output_dir / f"{model_combo_name}_test_files.txt"
    with open(test_files_path, 'w') as f:
        for fname in processed_files:
            f.write(f"{fname}\n")
    print(f"✓ Saved test file list to {test_files_path}")
    
    # Plot metrics
    plot_metrics(metrics, class_names, output_dir, model_combo_name)
    
    print(f"\n✓ All outputs saved to {output_dir}")
    print("="*80)
    
    # Print comparison summary if it exists
    if summary_path.exists():
        print("\n" + "="*80)
        print("COMPARISON SUMMARY (All Model Combinations)")
        print("="*80)
        comp_df = pd.read_csv(summary_path)
        print(comp_df[['model_combination', 'mean_precision', 'mean_recall', 'mean_f1', 
                       'avg_inference_time']].to_string(index=False))
        print("="*80)

if __name__ == "__main__":
    main()