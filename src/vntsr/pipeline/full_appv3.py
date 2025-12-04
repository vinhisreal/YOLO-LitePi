#!/usr/bin/env python3
"""
Detect (YOLOv8 ONNX) -> Crop -> Classify (EfficientNet) pipeline
Supports: realtime camera OR process folder of images OR single image OR video file.
Outputs: live window with overlays, and CSV of results + optional saved crops.
Requirements:
  pip install opencv-python pillow torch torchvision pandas onnxruntime
"""

import os
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
import onnxruntime as ort

# -----------------------
# Utilities / transforms
# -----------------------
def get_device(prefer_gpu=False):
    if prefer_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"

def build_classifier(num_classes, model_path=None, device="cpu"):
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    if model_path and os.path.exists(model_path):
        sd = torch.load(model_path, map_location=device)
        model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model

def get_transform():
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

def classify_crop(pil_img, classifier, transform, device="cpu"):
    """pil_img: PIL.Image RGB"""
    t0 = time.time()
    x = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = classifier(x)
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)
        conf_val = float(conf.item())
        idx_val = int(idx.item())
    t1 = time.time()
    return idx_val, conf_val, (t1 - t0)

# -----------------------
# ONNX YOLO utilities
# -----------------------
class YOLOv8ONNX:
    def __init__(self, onnx_path, conf_threshold=0.3, iou_threshold=0.45, img_size=640):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        
        # Load ONNX model
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        # YOLOv8 class names (COCO dataset - adjust if using custom classes)
        self.names = {0: 'traffic_sign'}  # Default, will be overridden if needed
    
    def preprocess(self, img):
        """Preprocess image for ONNX inference"""
        # Resize to model input size
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1] and transpose to CHW
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        # Add batch dimension
        img_batch = np.expand_dims(img_transposed, axis=0)
        return img_batch
    
    def postprocess(self, outputs, orig_shape):
        """Postprocess ONNX outputs to get bounding boxes"""
        # YOLOv8 output shape: [1, 84, 8400] for COCO
        # [1, num_classes + 4, num_predictions]
        predictions = outputs[0]
        
        # Transpose to [8400, 84]
        predictions = np.squeeze(predictions).T
        
        # Extract boxes and scores
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]
        
        # Get class with highest score for each detection
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Filter by confidence threshold
        mask = confidences > self.conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        # Convert from YOLO format (center_x, center_y, w, h) to (x1, y1, x2, y2)
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        
        # Scale boxes back to original image size
        h, w = orig_shape[:2]
        scale_x = w / self.img_size
        scale_y = h / self.img_size
        boxes_xyxy[:, [0, 2]] *= scale_x
        boxes_xyxy[:, [1, 3]] *= scale_y
        
        # Apply NMS
        indices = self.nms(boxes_xyxy, confidences, self.iou_threshold)
        
        return boxes_xyxy[indices], confidences[indices], class_ids[indices]
    
    def nms(self, boxes, scores, iou_threshold):
        """Non-Maximum Suppression"""
        if len(boxes) == 0:
            return np.array([], dtype=np.int32)
        
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
        
        return np.array(keep, dtype=np.int32)
    
    def __call__(self, img, conf=None):
        """Run inference on image"""
        if conf is not None:
            self.conf_threshold = conf
        
        # Preprocess
        input_tensor = self.preprocess(img)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # Postprocess
        boxes, confidences, class_ids = self.postprocess(outputs, img.shape)
        
        # Create result object similar to ultralytics
        class Result:
            def __init__(self, boxes_data):
                self.boxes = boxes_data
        
        class Boxes:
            def __init__(self, xyxy, conf, cls):
                self.xyxy = torch.from_numpy(xyxy) if len(xyxy) > 0 else torch.empty((0, 4))
                self.conf = torch.from_numpy(conf) if len(conf) > 0 else torch.empty(0)
                self.cls = torch.from_numpy(cls.astype(np.float32)) if len(cls) > 0 else torch.empty(0)
        
        boxes_obj = Boxes(boxes, confidences, class_ids)
        return [Result(boxes_obj)]

# -----------------------
# Detection + classification per frame / image
# -----------------------
def detect_and_classify_frame(
    frame_bgr,
    yolo_model,
    classifier,
    class_names,
    yolo_conf=0.3,
    min_area=100,
    device="cpu",
    transform=None,
):
    """
    frame_bgr: OpenCV BGR numpy array
    returns annotated_frame_bgr, detections_list
    """
    results = []
    t_start = time.time()
    
    # Run YOLO ONNX
    t_det0 = time.time()
    yolo_results = yolo_model(frame_bgr, conf=yolo_conf)[0]
    t_det1 = time.time()
    time_det = t_det1 - t_det0

    # Extract boxes
    boxes = yolo_results.boxes
    if boxes is None or len(boxes.xyxy) == 0:
        return frame_bgr, []

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)

    h, w = frame_bgr.shape[:2]
    annotated = frame_bgr.copy()

    for (x1,y1,x2,y2), det_conf, det_cls_idx in zip(xyxy, confs, clss):
        x1i = max(0, int(round(x1)))
        y1i = max(0, int(round(y1)))
        x2i = min(w-1, int(round(x2)))
        y2i = min(h-1, int(round(y2)))
        area = (x2i-x1i) * (y2i-y1i)
        if area < min_area:
            continue

        # Crop, convert to PIL RGB
        crop_bgr = annotated[y1i:y2i, x1i:x2i]
        if crop_bgr.size == 0:
            continue
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_crop = Image.fromarray(crop_rgb)

        # Classify
        cls_idx, cls_conf, time_cls = classify_crop(pil_crop, classifier, transform, device=device)

        # Compose result
        det_name = yolo_model.names.get(int(det_cls_idx), str(det_cls_idx))
        cls_name = class_names[cls_idx] if 0 <= cls_idx < len(class_names) else str(cls_idx)
        total_time = time_det + time_cls

        res = {
            'bbox': (x1i,y1i,x2i,y2i),
            'det_class': det_name,
            'det_conf': float(det_conf),
            'cls_pred': cls_name,
            'cls_conf': float(cls_conf),
            'time_det': float(time_det),
            'time_cls': float(time_cls),
            'total_time': float(total_time)
        }
        if cls_conf >= 0.5:
            results.append(res)

            # Draw on annotated image
            label = f"{cls_name}({cls_conf:.2f})"
            cv2.rectangle(annotated, (x1i,y1i), (x2i,y2i), (0,255,0), 2)
            cv2.putText(annotated, label, (x1i, max(10,y1i-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    return annotated, results

# -----------------------
# Process camera
# -----------------------
def run_camera(args, yolo_model, classifier, class_names, transform, device):
    # Create camera-specific output folder
    output_dir = Path("output_camera")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_csv = output_dir / "results_camera.csv"
    
    cap = cv2.VideoCapture(args.cam_id)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera id", args.cam_id)
        return

    results_rows = []
    fps_smooth = None
    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            annotated, detections = detect_and_classify_frame(
                frame, yolo_model, classifier, class_names,
                yolo_conf=args.yolo_conf, min_area=args.min_area,
                device=device, transform=transform
            )
            t1 = time.time()
            frame_time = t1 - t0
            fps = 1.0 / frame_time if frame_time > 0 else 0.0
            fps_smooth = fps if fps_smooth is None else (0.8*fps_smooth + 0.2*fps)

            # Save results rows for CSV
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            if len(detections) == 0:
                results_rows.append({
                    "timestamp": ts, "frame": "camera",
                    "bbox": "", "det_class": "", "det_conf": "", "cls_pred": "", "cls_conf": "",
                    "time_det": "", "time_cls": "", "total_time": ""
                })
            else:
                for d in detections:
                    results_rows.append({
                        "timestamp": ts, "frame": "camera",
                        "bbox": f"{d['bbox']}", "det_class": d['det_class'], "det_conf": d['det_conf'],
                        "cls_pred": d['cls_pred'], "cls_conf": d['cls_conf'],
                        "time_det": d['time_det'], "time_cls": d['time_cls'], "total_time": d['total_time']
                    })

            # Show FPS & quit key
            cv2.putText(annotated, f"FPS: {fps_smooth:.1f}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.imshow("Detect+Classify", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        # Save CSV
        df = pd.DataFrame(results_rows)
        df.to_csv(output_csv, index=False)
        print(f"[CAMERA] Saved results to {output_csv}")

# -----------------------
# Process video file
# -----------------------
def run_video(args, yolo_model, classifier, class_names, transform, device):
    # Create video-specific output folder
    output_dir = Path("output_video")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_path = Path(args.input)
    if not video_path.exists():
        print(f"[ERROR] Video file not found: {video_path}")
        return
    
    output_csv = output_dir / f"results_{video_path.stem}.csv"
    output_video = output_dir / f"output_{video_path.stem}.mp4"
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return
    
    # Get video properties
    fps_original = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[VIDEO INFO] Resolution: {width}x{height}, Original FPS: {fps_original:.2f}, Total frames: {total_frames}")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(str(output_video), fourcc, fps_original, (width, height))
    
    results_rows = []
    fps_smooth = None
    frame_count = 0
    processing_start = time.time()
    
    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            annotated, detections = detect_and_classify_frame(
                frame, yolo_model, classifier, class_names,
                yolo_conf=args.yolo_conf, min_area=args.min_area,
                device=device, transform=transform
            )
            t1 = time.time()
            frame_time = t1 - t0
            fps = 1.0 / frame_time if frame_time > 0 else 0.0
            fps_smooth = fps if fps_smooth is None else (0.8*fps_smooth + 0.2*fps)
            
            # Save results rows for CSV
            if len(detections) == 0:
                results_rows.append({
                    "frame_number": frame_count,
                    "timestamp_sec": frame_count / fps_original,
                    "bbox": "", "det_class": "", "det_conf": "", 
                    "cls_pred": "", "cls_conf": "",
                    "time_det": "", "time_cls": "", "total_time": "",
                    "processing_fps": fps
                })
            else:
                for d in detections:
                    results_rows.append({
                        "frame_number": frame_count,
                        "timestamp_sec": frame_count / fps_original,
                        "bbox": f"{d['bbox']}", 
                        "det_class": d['det_class'], 
                        "det_conf": d['det_conf'],
                        "cls_pred": d['cls_pred'], 
                        "cls_conf": d['cls_conf'],
                        "time_det": d['time_det'], 
                        "time_cls": d['time_cls'], 
                        "total_time": d['total_time'],
                        "processing_fps": fps
                    })
            
            # Add FPS overlay
            cv2.putText(annotated, f"Processing FPS: {fps_smooth:.1f}", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(annotated, f"Frame: {frame_count}/{total_frames}", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Write frame to output video
            out_writer.write(annotated)
            
            # Show progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"[VIDEO] Processing: {frame_count}/{total_frames} ({progress:.1f}%) | FPS: {fps_smooth:.1f}")
            
            # Optional: show window (press 'q' to stop early)
            if args.show_video:
                cv2.imshow("Video Processing", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[VIDEO] Stopped by user")
                    break
    
    finally:
        processing_end = time.time()
        total_time = processing_end - processing_start
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        cap.release()
        out_writer.release()
        if args.show_video:
            cv2.destroyAllWindows()
        
        # Save CSV
        df = pd.DataFrame(results_rows)
        df.to_csv(output_csv, index=False)
        
        # Print summary
        print(f"\n[VIDEO SUMMARY]")
        print(f"  Processed frames: {frame_count}/{total_frames}")
        print(f"  Total processing time: {total_time:.2f}s")
        print(f"  Average processing FPS: {avg_fps:.2f}")
        print(f"  Output video saved to: {output_video}")
        print(f"  Results CSV saved to: {output_csv}")

# -----------------------
# Process folder of images
# -----------------------
def run_folder(args, yolo_model, classifier, class_names, transform, device):
    # Create folder-specific output folder
    output_dir = Path("output_folder")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_csv = output_dir / "results_folder.csv"
    
    img_dir = Path(args.input)
    rows = []

    img_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png")])
    for img_p in img_files:
        frame = cv2.imread(str(img_p))
        if frame is None:
            continue
        annotated, detections = detect_and_classify_frame(
            frame, yolo_model, classifier, class_names,
            yolo_conf=args.yolo_conf, min_area=args.min_area,
            device=device, transform=transform
        )
        # Save annotated image
        annotated_path = output_dir / f"annot_{img_p.name}"
        cv2.imwrite(str(annotated_path), annotated)

        # Save each crop and rows
        if len(detections) == 0:
            rows.append({
                "image": img_p.name, "bbox":"", "det_class":"", "det_conf":"", "cls_pred":"", "cls_conf":"", "time_det":"", "time_cls":"", "total_time":""
            })
        else:
            for i, d in enumerate(detections):
                # save crop
                x1,y1,x2,y2 = d['bbox']
                crop = frame[y1:y2, x1:x2]
                crop_path = output_dir / f"{img_p.stem}_crop{i}{img_p.suffix}"
                # cv2.imwrite(str(crop_path), crop)
                rows.append({
                    "image": img_p.name, "bbox": f"{d['bbox']}", "det_class": d['det_class'], "det_conf": d['det_conf'],
                    "cls_pred": d['cls_pred'], "cls_conf": d['cls_conf'], "time_det": d['time_det'], "time_cls": d['time_cls'], "total_time": d['total_time']
                })

        print(f"[Processed] {img_p.name} -> dets: {len(detections)} | saved annotated to {annotated_path}")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"[FOLDER] Saved CSV to {output_csv}")

# -----------------------
# Process single image
# -----------------------
def run_single(args, yolo_model, classifier, class_names, transform, device):
    # Create single-specific output folder
    output_dir = Path("output_single")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_csv = output_dir / "results_single.csv"
    
    path = Path(args.input)
    if not path.exists():
        print("[ERROR] Input image not found:", path)
        return
    
    frame = cv2.imread(str(path))
    if frame is None:
        print("[ERROR] Cannot read image:", path)
        return
        
    annotated, detections = detect_and_classify_frame(
        frame, yolo_model, classifier, class_names, 
        yolo_conf=args.yolo_conf, min_area=args.min_area,
        device=device, transform=transform
    )
    
    # Save annotated image
    out_ann = output_dir / f"annot_{path.name}"
    cv2.imwrite(str(out_ann), annotated)
    print(f"[SINGLE] Saved annotated: {out_ann}")
    
    # Save crops
    rows = []
    if len(detections) == 0:
        rows.append({
            "image": path.name, "bbox":"", "det_class":"", "det_conf":"", 
            "cls_pred":"", "cls_conf":"", "time_det":"", "time_cls":"", "total_time":""
        })
    else:
        for i, d in enumerate(detections):
            # Save crop
            x1,y1,x2,y2 = d['bbox']
            crop = frame[y1:y2, x1:x2]
            crop_path = output_dir / f"{path.stem}_crop{i}{path.suffix}"
            # cv2.imwrite(str(crop_path), crop)
            
            rows.append({
                "image": path.name, "bbox":f"{d['bbox']}", "det_class":d['det_class'], 
                "det_conf":d['det_conf'], "cls_pred":d['cls_pred'], "cls_conf":d['cls_conf'],
                "time_det":d['time_det'], "time_cls":d['time_cls'], "total_time":d['total_time']
            })
    
    # Save CSV
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"[SINGLE] Saved CSV: {output_csv}")

# -----------------------
# Argparse + main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="YOLO ONNX detect -> EfficientNet classify pipeline")
    parser.add_argument("--yolo", type=str,
        default=r"./Models/yolov8n_detect.onnx",
        help="path to YOLOv8 ONNX model")

    parser.add_argument("--clf", type=str,
        default=r"./Models/best_efficientnet.pth",
        help="path to classifier .pth (EfficientNet)")

    parser.add_argument("--signnames", type=str,
        default=r"signnames.csv",
        help="CSV with sign_name column")

    parser.add_argument("--mode", type=str, choices=["camera","folder","single","video"], default="camera")
    parser.add_argument("--input", type=str, default=r"./Test", help="input folder/single image/video for mode 'folder', 'single' or 'video'")
    parser.add_argument("--cam_id", type=int, default=0)
    parser.add_argument("--yolo_conf", type=float, default=0.35)
    parser.add_argument("--yolo_iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--img_size", type=int, default=640, help="YOLO input image size")
    parser.add_argument("--min_area", type=int, default=200)
    parser.add_argument("--device", type=str, choices=["cpu","cuda","auto"], default="auto")
    parser.add_argument("--show_video", action="store_true", help="Show video window during processing (video mode only)")
    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        device = get_device(prefer_gpu=True)
    else:
        device = args.device

    # Load sign names
    df = pd.read_csv(args.signnames)
    class_names = df["sign_name"].tolist()

    # Load models
    print("Loading YOLO ONNX model:", args.yolo)
    yolo_model = YOLOv8ONNX(
        args.yolo, 
        conf_threshold=args.yolo_conf,
        iou_threshold=args.yolo_iou,
        img_size=args.img_size
    )
    
    print("Loading classifier (EfficientNet) on device:", device)
    classifier = build_classifier(num_classes=len(class_names), model_path=args.clf, device=device)
    transform = get_transform()

    # Run chosen mode
    if args.mode == "camera":
        print("\n[MODE: CAMERA] Output will be saved to 'output_camera/' folder")
        run_camera(args, yolo_model, classifier, class_names, transform, device)
    elif args.mode == "video":
        print("\n[MODE: VIDEO] Output will be saved to 'output_video/' folder")
        run_video(args, yolo_model, classifier, class_names, transform, device)
    elif args.mode == "folder":
        print("\n[MODE: FOLDER] Output will be saved to 'output_folder/' folder")
        run_folder(args, yolo_model, classifier, class_names, transform, device)
    elif args.mode == "single":
        print("\n[MODE: SINGLE] Output will be saved to 'output_single/' folder")
        run_single(args, yolo_model, classifier, class_names, transform, device)
    else:
        print("Unknown mode")

if __name__ == "__main__":
    main()