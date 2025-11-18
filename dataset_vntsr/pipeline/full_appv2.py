#!/usr/bin/env python3
"""
Detect (YOLO .pt) -> Crop -> Classify (.pth) pipeline for Raspberry Pi 5.
Now with DEBUG LOGS for easy tracking.
"""

import os
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import json

import torch
import torch.nn as nn
from torchvision import transforms

from ultralytics import YOLO


# -----------------------
# Utilities / transforms
# -----------------------
def log(msg):
    print(f"[LOG] {msg}", flush=True)


def get_device(prefer_gpu=False):
    log("Checking device...")
    if prefer_gpu and torch.cuda.is_available():
        log("CUDA is available → using GPU")
        return "cuda"
    log("Using CPU")
    return "cpu"


def build_classifier(num_classes, model_path=None, device="cpu"):
    log("Building classifier model EfficientNet_B0...")
    from torchvision.models import efficientnet_b0
    model = efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if model_path and os.path.exists(model_path):
        log(f"Loading classifier weights from {model_path}")
        sd = torch.load(model_path, map_location=device)
        model.load_state_dict(sd)
    else:
        log("[WARN] Classifier weights not found!")

    model.to(device)
    model.eval()
    log("Classifier loaded and set to eval()")
    return model


def get_transform():
    log("Building image transform (64x64)...")
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.18,0.18,0.18],[0.34,0.34,0.34])
    ])


def classify_crop(pil_img, classifier, transform, device="cpu"):
    log(" Classifying crop...")
    t0 = time.time()
    x = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = classifier(x)
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)
    t1 = time.time()
    log(f"  → Classify done in {t1-t0:.4f}s")
    return int(idx.item()), float(conf.item()), (t1 - t0)


# -----------------------
# Detection + classification
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
    log("Running YOLO inference...")
    det0 = time.time()
    yolo_results = yolo_model(frame_bgr, conf=yolo_conf, verbose=False)[0]
    det1 = time.time()
    time_det = det1 - det0
    log(f"YOLO inference done ({time_det:.4f}s)")

    annotated = frame_bgr.copy()
    h, w = frame_bgr.shape[:2]
    results = []

    log(f"Found {len(yolo_results.boxes)} detections")

    for i, box in enumerate(yolo_results.boxes):
        log(f" Processing box #{i}...")
        x1, y1, x2, y2 = box.xyxy[0]
        det_conf = float(box.conf[0])

        x1i = max(0, int(x1))
        y1i = max(0, int(y1))
        x2i = min(w-1, int(x2))
        y2i = min(h-1, int(y2))
        area = (x2i-x1i) * (y2i-y1i)

        if area < min_area:
            log("  → Box too small, skipping")
            continue

        crop_bgr = annotated[y1i:y2i, x1i:x2i]
        if crop_bgr.size == 0:
            log("  → Empty crop, skipping")
            continue

        crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

        cls_idx, cls_conf, classify_time = classify_crop(
            crop_pil, classifier, transform, device
        )

        cls_name = class_names.get(str(cls_idx), f"{cls_idx}")
        log(f"  → Classified as {cls_name} ({cls_conf:.2f})")

        if cls_conf >= 0.5:
            log("  → Good confidence, drawing bbox")
            results.append({
                'bbox': (x1i, y1i, x2i, y2i),
                'det_conf': det_conf,
                'cls_pred': cls_name,
                'cls_conf': cls_conf,
                'time_det': time_det,
                'time_cls': classify_time,
                'total_time': time_det + classify_time
            })

            cv2.rectangle(annotated, (x1i, y1i), (x2i, y2i), (0,255,0), 2)
            cv2.putText(annotated, f"{cls_name}({cls_conf:.2f})",
                        (x1i, max(10, y1i-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0,255,0), 2)
        else:
            log("  → Confidence too low, skip drawing")

    log("Frame done.\n")
    return annotated, results


# -----------------------
# CAMERA mode
# -----------------------
def run_camera(args, yolo_model, classifier, class_names, transform, device):
    log("Opening camera...")
    cap = cv2.VideoCapture(args.cam_id)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    log("Camera opened successfully")

    fps_smooth = None

    try:
        while True:
            log("Reading frame...")
            ok, frame = cap.read()
            if not ok:
                log("Failed to read frame! Breaking...")
                break

            annotated, dets = detect_and_classify_frame(
                frame, yolo_model, classifier, class_names,
                yolo_conf=args.yolo_conf, min_area=args.min_area,
                device=device, transform=transform
            )

            fps = 1.0 / max(0.0001, (time.time() - time.time()))
            cv2.imshow("TSR Pipeline", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                log("Stop key detected → exiting")
                break

    finally:
        log("Releasing camera...")
        cap.release()
        cv2.destroyAllWindows()
        log("Camera closed.")


# -----------------------
# MAIN
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo", type=str, default="/home/pi5/TrafficSign/WeightDetectionV2/yolo8.pt")
    parser.add_argument("--clf", type=str, default="/home/pi5/TrafficSign/WeightClassifyV2/efficientnetb0_best.pth")
    parser.add_argument("--classnames", type=str, default="/home/pi5/TrafficSign/WeightClassifyV2/idx2label.json")
    parser.add_argument("--cam_id", type=int, default=0)
    parser.add_argument("--yolo_conf", type=float, default=0.35)
    parser.add_argument("--min_area", type=int, default=150)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    log("===== START PIPELINE =====")

    device = get_device(prefer_gpu=True)
    log(f"Using device: {device}")

    log("Loading classnames...")
    with open(args.classnames, "r") as f:
        class_names = json.load(f)
    log(f"Loaded {len(class_names)} classes")

    log("Loading YOLO model...")
    yolo_model = YOLO(args.yolo)
    log("YOLO model loaded.")

    log("Loading classifier...")
    classifier = build_classifier(len(class_names), args.clf, device)
    transform = get_transform()

    log("Starting CAMERA mode...")
    run_camera(args, yolo_model, classifier, class_names, transform, device)


if __name__ == "__main__":
    main()
