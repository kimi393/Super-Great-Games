#!/usr/bin/env python3
"""
YOLO11 Model Training Script
Converts COCO annotations from Roboflow to YOLO format, splits the dataset,
creates dataset.yaml, and trains the YOLO11 model.
"""

import os
import json
import shutil
import random
from pathlib import Path
import torch
from ultralytics import YOLO

# Set random seed for reproducibility
random.seed(42)

def main():
    # Define paths
    WORKSPACE_DIR = Path("/Users/kimi/Desktop/j/ ai hand whiteing boaro")
    TRAIN_DIR = WORKSPACE_DIR / "train"
    COCO_JSON_PATH = TRAIN_DIR / "_annotations.coco.json"
    DATASET_DIR = WORKSPACE_DIR / "yolo_dataset"

    print("--- STEP 1: Setting up Directory Structure ---")
    # Clean previous dataset if any, to avoid pollution
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
        
    for split in ["train", "val"]:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    print(f"Created YOLO dataset directories in {DATASET_DIR}")

    print("\n--- STEP 2: Loading and Parsing COCO Annotations ---")
    if not COCO_JSON_PATH.exists():
        print(f"Error: COCO annotations file not found at {COCO_JSON_PATH}")
        return
        
    with open(COCO_JSON_PATH, "r") as f:
        coco_data = json.load(f)

    # Extract categories
    categories = coco_data["categories"]
    class_names = {cat["id"]: cat["name"] for cat in categories}
    print(f"Categories found: {class_names}")

    # Extract images and annotations
    images = coco_data["images"]
    annotations = coco_data["annotations"]

    # Group annotations by image_id
    image_ann_map = {img["id"]: [] for img in images}
    for ann in annotations:
        image_ann_map[ann["image_id"]].append(ann)

    # Split images: 10 train, 2 val (83% / 17% split for 12 images)
    # Sort to ensure deterministic split across runs
    images = sorted(images, key=lambda x: x["file_name"])
    random.shuffle(images)
    
    train_images = images[:10]
    val_images = images[10:]
    
    splits = {
        "train": train_images,
        "val": val_images
    }

    print("\n--- STEP 3: Converting Annotations and Copying Images ---")
    for split_name, split_imgs in splits.items():
        print(f"Processing '{split_name}' split ({len(split_imgs)} images)...")
        for img in split_imgs:
            img_id = img["id"]
            file_name = img["file_name"]
            width = img["width"]
            height = img["height"]
            
            # Copy image file
            src_image_path = TRAIN_DIR / file_name
            dest_image_path = DATASET_DIR / "images" / split_name / file_name
            
            if src_image_path.exists():
                shutil.copy(src_image_path, dest_image_path)
            else:
                print(f"Warning: Image file not found: {src_image_path}")
                continue
                
            # Get annotations for this image
            anns = image_ann_map.get(img_id, [])
            
            # Create label text file
            label_file_name = Path(file_name).stem + ".txt"
            label_file_path = DATASET_DIR / "labels" / split_name / label_file_name
            
            with open(label_file_path, "w") as lf:
                for ann in anns:
                    cat_id = ann["category_id"]
                    bbox = ann["bbox"] # [x_min, y_min, width, height]
                    
                    x_min, y_min, w_box, h_box = bbox
                    
                    # Calculate center coordinates
                    x_center = x_min + w_box / 2.0
                    y_center = y_min + h_box / 2.0
                    
                    # Normalize coordinates relative to image width/height
                    x_center_norm = x_center / width
                    y_center_norm = y_center / height
                    w_box_norm = w_box / width
                    h_box_norm = h_box / height
                    
                    # Clip values between 0.0 and 1.0 to satisfy YOLO standards
                    x_center_norm = max(0.0, min(1.0, x_center_norm))
                    y_center_norm = max(0.0, min(1.0, y_center_norm))
                    w_box_norm = max(0.0, min(1.0, w_box_norm))
                    h_box_norm = max(0.0, min(1.0, h_box_norm))
                    
                    lf.write(f"{cat_id} {x_center_norm:.6f} {y_center_norm:.6f} {w_box_norm:.6f} {h_box_norm:.6f}\n")

    print("\n--- STEP 4: Creating dataset.yaml ---")
    # Quote the path in case workspace directory contains spaces
    yaml_content = f"""path: "{DATASET_DIR.as_posix()}"
train: images/train
val: images/val

names:
"""
    for cat_id in sorted(class_names.keys()):
        yaml_content += f"  {cat_id}: {class_names[cat_id]}\n"

    yaml_path = WORKSPACE_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"Created dataset configuration file at {yaml_path}")

    print("\n--- STEP 5: Initializing and Training YOLO11 ---")
    # Check if GPU acceleration is available on Mac (MPS)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Selected hardware accelerator: {device.upper()}")
    
    # Load pretrained YOLO11 nano model
    print("Loading pretrained yolo11n.pt weights...")
    model = YOLO("yolo11n.pt")
    
    # Start training
    epochs = 150
    print(f"Training model for {epochs} epochs...")
    
    # Clear any previous runs with the same project/name to prevent naming conflicts
    project_dir = WORKSPACE_DIR / "runs"
    
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=4,          # Small batch size to get more gradient updates per epoch
        mosaic=0.0,       # Turn off mosaic augmentation (dataset is too small)
        imgsz=640,
        device=device,
        project=str(project_dir),
        name="toy_cars_train",
        exist_ok=True  # Overwrite previous training folder
    )

    print("\n--- STEP 6: Saving Best Model Weights ---")
    best_weights_path = project_dir / "toy_cars_train" / "weights" / "best.pt"
    final_weights_path = WORKSPACE_DIR / "yolo_toy_cars_model.pt"
    
    if best_weights_path.exists():
        shutil.copy(best_weights_path, final_weights_path)
        print(f"Success! Best model weights saved to workspace root: {final_weights_path}")
    else:
        print(f"Error: Best weights file not found at {best_weights_path}")

if __name__ == "__main__":
    main()
