#!/usr/bin/env python3
"""
Diagnostic script to test predictions at different confidence thresholds.
"""

import os
import cv2
from pathlib import Path
from ultralytics import YOLO

WORKSPACE_DIR = Path("/Users/kimi/Desktop/j/ ai hand whiteing boaro")
DATASET_DIR = WORKSPACE_DIR / "yolo_dataset"
MODEL_PATH = WORKSPACE_DIR / "yolo_toy_cars_model.pt"

def main():
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        return
        
    model = YOLO(MODEL_PATH)
    
    # Get all validation images
    img_dir = DATASET_DIR / "images" / "val"
    if not img_dir.exists():
        img_dir = DATASET_DIR / "images" / "train"
        
    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.png"))
    if not images:
        print("No images found in dataset directory.")
        return
        
    print(f"Loaded model: {MODEL_PATH}")
    print(f"Scanning {len(images)} images for detections...\n")
    
    thresholds = [0.25, 0.10, 0.05, 0.01]
    
    for img_path in images[:3]:  # test first 3 images
        print(f"--- Image: {img_path.name} ---")
        for thresh in thresholds:
            results = model(str(img_path), conf=thresh, verbose=False)
            boxes = results[0].boxes
            num_det = len(boxes)
            print(f"  Threshold {thresh:.2f}: {num_det} detections")
            if num_det > 0:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    conf_val = float(box.conf[0])
                    print(f"    - Detected {cls_name} with confidence {conf_val:.3f}")
        print()

if __name__ == "__main__":
    main()
