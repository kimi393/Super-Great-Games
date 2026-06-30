#!/usr/bin/env python3
"""
YOLO11 Dataset Inference Viewer
Displays side-by-side Ground Truth annotations vs. YOLO11 Predictions
for the toy car dataset using Matplotlib.
Supports interactive navigation with Arrow Keys.
"""

import os
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

# Define paths
WORKSPACE_DIR = Path("/Users/kimi/Desktop/j/ ai hand whiteing boaro")
DATASET_DIR = WORKSPACE_DIR / "yolo_dataset"
MODEL_PATH = WORKSPACE_DIR / "yolo_toy_cars_model.pt"

# Class names mapping
CLASS_NAMES = {
    0: "toy-cars",
    1: "daihatsu",
    2: "lamborghini",
    3: "mini cooper",
    4: "prius",
    5: "sienta",
    6: "tesla"
}

# Color palette for classes (RGB format for Matplotlib / BGR for OpenCV drawing)
CLASS_COLORS = {
    0: (0, 255, 255),    # Cyan
    1: (255, 0, 255),    # Magenta
    2: (0, 128, 255),    # Orange/Yellow
    3: (255, 255, 0),    # Yellow
    4: (0, 255, 127),    # Neon Green
    5: (255, 128, 0),    # Orange
    6: (128, 0, 255)     # Purple
}

def load_dataset_images():
    images = []
    # Search in both val and train splits
    for split in ["val", "train"]:
        img_dir = DATASET_DIR / "images" / split
        if img_dir.exists():
            for ext in [".jpg", ".jpeg", ".png"]:
                for img_path in sorted(img_dir.glob(f"*{ext}")):
                    label_path = DATASET_DIR / "labels" / split / (img_path.stem + ".txt")
                    images.append({
                        "path": img_path,
                        "label_path": label_path,
                        "split": split
                    })
    return images

def draw_ground_truth(img_path, label_path):
    # Read image and convert to RGB
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    
    if label_path.exists():
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cat_id = int(parts[0])
                    x_center = float(parts[1]) * w
                    y_center = float(parts[2]) * h
                    w_box = float(parts[3]) * w
                    h_box = float(parts[4]) * h
                    
                    x1 = int(x_center - w_box / 2)
                    y1 = int(y_center - h_box / 2)
                    x2 = int(x_center + w_box / 2)
                    y2 = int(y_center + h_box / 2)
                    
                    color = CLASS_COLORS.get(cat_id, (255, 255, 255))
                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw a nice label tag background
                    label_text = CLASS_NAMES.get(cat_id, f"Class {cat_id}")
                    (txt_w, txt_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    
                    # Ensure label is drawn within image boundaries
                    label_y1 = max(y1 - txt_h - 6, 0)
                    cv2.rectangle(img, (x1, label_y1), (x1 + txt_w + 6, label_y1 + txt_h + 6), color, -1)
                    
                    # Draw text label in black on top of color background
                    cv2.putText(img, label_text, (x1 + 3, label_y1 + txt_h + 2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return img

def main():
    if not MODEL_PATH.exists():
        print(f"Error: Custom model weights not found at {MODEL_PATH}")
        print("Please train the model first by running: python train_yolo_model.py")
        return
        
    print("Loading YOLO11 model...")
    model = YOLO(MODEL_PATH)
    
    images = load_dataset_images()
    if not images:
        print(f"Error: No dataset images found in {DATASET_DIR}")
        return
        
    print(f"Found {len(images)} images in dataset.")
    
    current_idx = 0
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Set window title
    fig.canvas.manager.set_window_title("YOLO11 Dataset Inference Viewer")
    
    def display_prediction(idx):
        nonlocal current_idx
        current_idx = idx
        img_info = images[idx]
        img_path = img_info["path"]
        label_path = img_info["label_path"]
        split = img_info["split"]
        
        # 1. Generate Ground Truth image
        gt_img = draw_ground_truth(img_path, label_path)
        
        # 2. Generate Prediction image
        results = model(str(img_path), verbose=False)
        pred_img_bgr = results[0].plot(labels=True, conf=True)
        pred_img = cv2.cvtColor(pred_img_bgr, cv2.COLOR_BGR2RGB)
        
        # Clear and display Ground Truth
        axes[0].clear()
        axes[0].imshow(gt_img)
        axes[0].set_title(f"Ground Truth ({split.upper()})", fontsize=12, fontweight='bold', color='darkblue')
        axes[0].axis("off")
        
        # Clear and display YOLO prediction
        axes[1].clear()
        axes[1].imshow(pred_img)
        axes[1].set_title(f"YOLO11 Prediction ({split.upper()})", fontsize=12, fontweight='bold', color='darkgreen')
        axes[1].axis("off")
        
        plt.suptitle(
            f"Image {idx + 1} / {len(images)}: {img_path.name}\n"
            "Use Left (←) / Right (→) Arrow Keys to Navigate | ESC to Quit", 
            fontsize=12, fontweight='bold', y=0.96
        )
        plt.tight_layout()
        plt.draw()
        
    def on_key(event):
        if event.key == "right":
            next_idx = (current_idx + 1) % len(images)
            display_prediction(next_idx)
        elif event.key == "left":
            prev_idx = (current_idx - 1) % len(images)
            display_prediction(prev_idx)
        elif event.key == "escape":
            plt.close()

    # Register event handler
    fig.canvas.mpl_connect("key_press_event", on_key)
    
    # Display the first image
    display_prediction(current_idx)
    
    print("\nViewer opened successfully!")
    print("Click on the window and use Left (<-) / Right (->) arrow keys to step through images.")
    plt.show()

if __name__ == "__main__":
    main()
