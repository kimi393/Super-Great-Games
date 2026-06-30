#!/usr/bin/env python3
"""
YOLO Toy Car Tracking Script
Tracks the positions of toy cars (and other vehicles) in real-time.
Features a premium sci-fi HUD overlay, fading trailing paths, and data logging.
"""

import argparse
import time
import os
import csv
import cv2
import numpy as np
from ultralytics import YOLO

# Premium Color Palette (BGR format for OpenCV)
COLOR_HUD_BG = (20, 20, 20)          # Dark grey for panels
COLOR_NEON_CYAN = (255, 255, 0)      # Primary text / active box color
COLOR_NEON_MAGENTA = (255, 0, 240)   # Secondary highlight / path end
COLOR_NEON_GREEN = (0, 255, 127)     # Stat indicator / path start
COLOR_NEON_RED = (80, 80, 255)       # Alert indicator
COLOR_TEXT_MUTED = (160, 160, 160)    # Subtitles
COLOR_TEXT_WHITE = (245, 245, 245)    # Regular text

# Default class IDs to filter in COCO dataset
# 2: car, 7: truck, 5: bus, 3: motorcycle, 1: bicycle
DEFAULT_VEHICLE_CLASSES = [1, 2, 3, 5, 7]

def draw_glass_panel(img, pt1, pt2, color=COLOR_HUD_BG, alpha=0.7):
    """Draws a semi-transparent panel with a fine neon-glow border."""
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    # Apply alpha blending
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
    # Draw fine border
    cv2.rectangle(img, pt1, pt2, COLOR_NEON_CYAN, 1)

def draw_sci_fi_corners(img, pt1, pt2, color, thickness=2, length=12):
    """Draws high-tech sci-fi corner brackets instead of a plain box."""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Top-left corner
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
    # Top-right corner
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
    # Bottom-left corner
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
    # Bottom-right corner
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)

def draw_trail(img, points, color_start=COLOR_NEON_GREEN, color_end=COLOR_NEON_MAGENTA):
    """Draws a fading gradient trajectory trail with increasing thickness."""
    n = len(points)
    if n < 2:
        return
    
    for i in range(n - 1):
        # Calculate progression factor (0.0 = oldest, 1.0 = newest)
        factor = i / (n - 1)
        
        # Linearly interpolate color (BGR)
        b = int(color_start[0] + factor * (color_end[0] - color_start[0]))
        g = int(color_start[1] + factor * (color_end[1] - color_start[1]))
        r = int(color_start[2] + factor * (color_end[2] - color_start[2]))
        
        # Calculate dynamic thickness (older segments are thinner)
        thickness = max(1, int(1 + factor * 4))
        
        cv2.line(img, points[i], points[i+1], (b, g, r), thickness)

def draw_header_hud(img, fps, active_count, source_name, filter_label, is_paused):
    """Draws a premium header bar showing system statistics."""
    h, w, _ = img.shape
    header_h = 45
    
    # Draw background panel
    draw_glass_panel(img, (0, 0), (w, header_h), alpha=0.75)
    
    # Title
    cv2.putText(img, "YOLO TOY CAR TRACKER // COGNITIVE SCANNER", (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_NEON_CYAN, 2, cv2.LINE_AA)
    
    # Status Indicators
    status_x = w - 480
    
    # FPS
    cv2.putText(img, f"FPS: {fps:.1f}", (status_x, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
    
    # Active Targets
    cv2.putText(img, f"TARGETS: {active_count:02d}", (status_x + 100, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_NEON_GREEN if active_count > 0 else COLOR_TEXT_MUTED, 2 if active_count > 0 else 1, cv2.LINE_AA)
    
    # Filter state
    cv2.putText(img, f"FILTER: {filter_label}", (status_x + 220, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_NEON_CYAN if filter_label != "ALL" else COLOR_NEON_MAGENTA, 1, cv2.LINE_AA)
    
    # Pause state
    pause_txt = "PAUSED" if is_paused else "RUNNING"
    cv2.putText(img, pause_txt, (w - 80, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_NEON_RED if is_paused else COLOR_NEON_GREEN, 2 if is_paused else 1, cv2.LINE_AA)

def draw_side_logger(img, active_tracks):
    """Draws a scrolling real-time coordinate logger on the right side."""
    h, w, _ = img.shape
    panel_w = 260
    panel_h = 320
    px1, py1 = w - panel_w - 15, 60
    px2, py2 = w - 15, py1 + panel_h
    
    # Panel Background
    draw_glass_panel(img, (px1, py1), (px2, py2), alpha=0.6)
    
    # Panel Header
    cv2.putText(img, "REAL-TIME TRACK LOG", (px1 + 10, py1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_NEON_CYAN, 1, cv2.LINE_AA)
    cv2.line(img, (px1 + 10, py1 + 30), (px2 - 10, py1 + 30), COLOR_NEON_CYAN, 1)
    
    # Log items
    y_offset = py1 + 55
    max_items = 8
    
    # Sort active tracks by ID
    sorted_tracks = sorted(active_tracks.items())[:max_items]
    
    if not sorted_tracks:
        cv2.putText(img, "[ NO TARGETS DETECTED ]", (px1 + 25, py1 + 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    else:
        for track_id, info in sorted_tracks:
            cx, cy = info['center']
            cls_name = info['class']
            
            # Format text
            track_str = f"ID: {track_id:02d} | {cls_name[:7].upper()}"
            coords_str = f"X: {cx:03d} | Y: {cy:03d}"
            
            # Draw target status log
            cv2.putText(img, track_str, (px1 + 10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
            cv2.putText(img, coords_str, (px1 + 130, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_NEON_GREEN, 1, cv2.LINE_AA)
            y_offset += 32

def draw_help_menu(img):
    """Draws a keyboard shortcut menu in the bottom-left corner."""
    h, w, _ = img.shape
    panel_w = 260
    panel_h = 135
    px1, py1 = 15, h - panel_h - 15
    px2, py2 = px1 + panel_w, h - 15
    
    draw_glass_panel(img, (px1, py1), (px2, py2), alpha=0.7)
    
    cv2.putText(img, "CONTROLS & KEYBOARD INFO", (px1 + 10, py1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_NEON_CYAN, 1, cv2.LINE_AA)
    cv2.line(img, (px1 + 10, py1 + 26), (px2 - 10, py1 + 26), COLOR_NEON_CYAN, 1)
    
    shortcuts = [
        ("P", "Pause / Resume Stream"),
        ("C", "Clear Trajectory Trails"),
        ("F", "Toggle Vehicle Classes Only"),
        ("T", "Toggle Trails Display"),
        ("Q", "Exit Application")
    ]
    
    y_offset = py1 + 45
    for key, desc in shortcuts:
        cv2.putText(img, f"[{key}]", (px1 + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_NEON_MAGENTA, 1, cv2.LINE_AA)
        cv2.putText(img, desc, (px1 + 45, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
        y_offset += 18

def main():
    parser = argparse.ArgumentParser(description="YOLO11 Real-Time Toy Car Tracker (Custom Trained)")
    parser.add_argument("--source", type=str, default="0", 
                        help="Camera index (e.g. 0) or path to a video file")
    parser.add_argument("--model", type=str, default="yolo_toy_cars_model.pt", 
                        help="YOLO model name (default: yolo_toy_cars_model.pt)")
    parser.add_argument("--filter", action="store_true", default=True,
                        help="Filter detections to only include target classes (daihatsu, prius, sienta, tesla, etc.)")
    parser.add_argument("--no-filter", action="store_false", dest="filter",
                        help="Disable class filtering and track all detectable objects")
    parser.add_argument("--output", type=str, default="", 
                        help="Optional CSV file path to log tracked positions")
    parser.add_argument("--trail-len", type=int, default=30,
                        help="Maximum length of the tracking trail path in frames")
    args = parser.parse_args()
    
    # Initialize YOLO Model
    print(f"Loading YOLO Model: {args.model}...")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return
    
    # Dynamically determine vehicle/car classes based on model's internal class names.
    # This ensures both default COCO models and custom-trained toy-car models work seamlessly.
    target_vehicle_names = {"car", "truck", "bus", "motorcycle", "bicycle", "toy-cars", "daihatsu", "prius", "sienta", "tesla", "lamborghini", "mini cooper"}
    model_vehicle_classes = [
        cid for cid, name in model.names.items() 
        if name.lower() in target_vehicle_names
    ]
    print(f"Model classes resolved for vehicle filtering: {[model.names[cid] for cid in model_vehicle_classes]}")
    
    # Initialize Video Source
    source = args.source
    # Convert to integer if it's a digit (for webcam index)
    if source.isdigit():
        source = int(source)
        source_name = f"Webcam {source}"
    else:
        source_name = os.path.basename(source)
        
    print(f"Opening Video Source: {source_name}...")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{source}'")
        return
        
    # Setup CSV logger if requested
    csv_file = None
    csv_writer = None
    if args.output:
        print(f"Logging coordinate outputs to: {args.output}")
        try:
            csv_file = open(args.output, mode='w', newline='')
            csv_writer = csv.writer(csv_file)
            # Write Header
            csv_writer.writerow(["Timestamp", "FrameIndex", "TrackID", "Class", "CenterX", "CenterY", "BBoxWidth", "BBoxHeight", "Confidence"])
        except Exception as e:
            print(f"Warning: Could not open output CSV file: {e}")
            csv_writer = None

    # Track trails dictionary: {track_id: list of (cx, cy) tuples}
    trail_history = {}
    
    # UI Toggles
    is_filtering = args.filter
    show_trails = True
    is_paused = False
    
    # Timing and performance stats
    prev_time = time.time()
    fps = 0.0
    frame_idx = 0
    
    print("\nTracking started successfully!")
    print("Press 'q' in the window to quit, or view shortcut menu on screen.")
    
    cv2.namedWindow("YOLO Sci-Fi Tracker HUD", cv2.WINDOW_NORMAL)
    
    while True:
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream or error reading frame.")
                break
                
            frame_idx += 1
            
            # FPS calculation
            curr_time = time.time()
            time_diff = curr_time - prev_time
            if time_diff > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / time_diff)  # Smooth FPS
            prev_time = curr_time
            
            # Create a display copy of the frame to keep drawing operations overlayed nicely
            display_frame = frame.copy()
            
            # Run YOLO Tracking
            # classes parameter filters the classes at prediction time if filtering is enabled
            classes_filter = model_vehicle_classes if is_filtering else None
            
            # Run object tracking with Ultralytics tracker
            results = model.track(
                source=frame, 
                persist=True, 
                verbose=False,
                classes=classes_filter,
                tracker="bytetrack.yaml"  # bytetrack is robust and performs well for tracking small/medium objects
            )
            
            active_tracks_hud = {}
            
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                names = model.names
                
                # Check if tracking IDs are available
                if boxes.id is not None:
                    track_ids = boxes.id.int().cpu().tolist()
                    xyxys = boxes.xyxy.cpu().tolist()
                    clss = boxes.cls.int().cpu().tolist()
                    confs = boxes.conf.cpu().tolist()
                    
                    # Track IDs active in the current frame
                    current_frame_ids = set()
                    
                    for xyxy, track_id, cls, conf in zip(xyxys, track_ids, clss, confs):
                        current_frame_ids.add(track_id)
                        
                        # Extract coordinates
                        x1, y1, x2, y2 = map(int, xyxy)
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        w_box, h_box = x2 - x1, y2 - y1
                        cls_name = names[cls]
                        
                        # Store in active tracks info dictionary
                        active_tracks_hud[track_id] = {
                            'center': (cx, cy),
                            'class': cls_name,
                            'bbox': (x1, y1, x2, y2)
                        }
                        
                        # Update trail history
                        if track_id not in trail_history:
                            trail_history[track_id] = []
                        trail_history[track_id].append((cx, cy))
                        
                        # Cap the trail history length
                        if len(trail_history[track_id]) > args.trail_len:
                            trail_history[track_id].pop(0)
                            
                        # Write coordinate details to CSV if enabled
                        if csv_writer:
                            csv_writer.writerow([
                                time.strftime("%Y-%m-%d %H:%M:%S"),
                                frame_idx,
                                track_id,
                                cls_name,
                                cx,
                                cy,
                                w_box,
                                h_box,
                                round(conf, 3)
                            ])
                            csv_file.flush() # Force write to file
                        
                        # --- DRAW SCENE OVERLAYS ---
                        # Draw high-tech corners
                        draw_sci_fi_corners(display_frame, (x1, y1), (x2, y2), COLOR_NEON_CYAN, thickness=2)
                        
                        # Draw glass box background for target details tag
                        tag_h = 16
                        tag_w = 110
                        draw_glass_panel(display_frame, (x1, y1 - tag_h - 4), (x1 + tag_w, y1 - 2), alpha=0.5)
                        
                        # Draw track ID label
                        label_txt = f"ID:{track_id:02d} | {cls_name.upper()}"
                        cv2.putText(display_frame, label_txt, (x1 + 4, y1 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_NEON_CYAN, 1, cv2.LINE_AA)
                        
                        # Draw current coordinate label at center of bounding box
                        coord_txt = f"({cx},{cy})"
                        cv2.putText(display_frame, coord_txt, (x1, y2 + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_NEON_GREEN, 1, cv2.LINE_AA)
                    
                    # Clean up trail history for tracks that are no longer active
                    inactive_ids = [tid for tid in trail_history if tid not in current_frame_ids]
                    # Keep some history of inactive trails so they don't pop out instantly, 
                    # but clear completely inactive objects that haven't been seen in over 60 frames.
                    # For simplicity, we just keep active trails to prevent clutter.
                    for tid in list(trail_history.keys()):
                        if tid not in current_frame_ids:
                            # Let inactive trails decay by popping their oldest coordinates
                            if len(trail_history[tid]) > 0:
                                trail_history[tid].pop(0)
                            else:
                                del trail_history[tid]
            
            # Draw trails for active objects
            if show_trails:
                for track_id, pts in trail_history.items():
                    # Only draw trails that have at least 2 points
                    draw_trail(display_frame, pts)
            
            # Determine filter status label text
            if is_filtering:
                filter_label = "TOY CARS" if any(name in model.names.values() for name in ["toy-cars", "daihatsu", "prius", "sienta", "tesla", "lamborghini", "mini cooper"]) else "VEHICLES"
            else:
                filter_label = "ALL"
                
            # Draw HUD Overlays
            draw_header_hud(display_frame, fps, len(active_tracks_hud), source_name, filter_label, is_paused)
            draw_side_logger(display_frame, active_tracks_hud)
            draw_help_menu(display_frame)
            
            cv2.imshow("YOLO Sci-Fi Tracker HUD", display_frame)
            
        else:
            # When paused, we just display the same frame copy and handle keystrokes
            # Draw paused overlay banner
            paused_frame = display_frame.copy()
            h_p, w_p, _ = paused_frame.shape
            cv2.rectangle(paused_frame, (0, h_p // 2 - 40), (w_p, h_p // 2 + 40), (0, 0, 0), -1)
            cv2.putText(paused_frame, "SCANNER PAUSED // P TO RESUME", (w_p // 2 - 190, h_p // 2 + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_NEON_RED, 2, cv2.LINE_AA)
            cv2.imshow("YOLO Sci-Fi Tracker HUD", paused_frame)
            
        # Keyboard Input Handler
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Exiting application...")
            break
        elif key == ord('p'):
            is_paused = not is_paused
            print("Tracking Paused" if is_paused else "Tracking Resumed")
        elif key == ord('c'):
            trail_history.clear()
            print("Trajectory trails cleared.")
        elif key == ord('t'):
            show_trails = not show_trails
            print(f"Trails Display: {'ON' if show_trails else 'OFF'}")
        elif key == ord('f'):
            is_filtering = not is_filtering
            print(f"Vehicle Filtering: {'ON (Cars, Trucks, etc.)' if is_filtering else 'OFF (Track all objects)'}")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    if csv_file:
        csv_file.close()
        print("Logged tracking data saved successfully.")

if __name__ == "__main__":
    main()
