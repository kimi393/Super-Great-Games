#!/usr/bin/env python3
"""
YOLO11 Toy Car Distance Game
--------------------------------------------------
Game Rules:
1. Players select 3 cars each.
2. The computer selects a random target spot on the screen.
3. Players place their toy cars and try to get them closest to the target.
4. Press SPACE to lock in and measure. The player with the closest car wins!
"""

import time
import os
import math
import random
import cv2
import numpy as np
import subprocess
import signal
import threading
from pathlib import Path
from ultralytics import YOLO
import time

# Background music process reference
bg_music = None
is_saving_replay = False

def cleanup():
    global bg_music
    if bg_music:
        try:
            os.killpg(os.getpgid(bg_music.pid), signal.SIGTERM)
            bg_music = None
            print("Background music stopped.")
        except Exception as e:
            pass

def save_replay_video(frames, foldername, fps=30.0):
    """Saves recorded frames list as an MP4 video file using OpenCV's VideoWriter."""
    global is_saving_replay
    if not frames:
        print("No frames to record.")
        is_saving_replay = False
        return
    try:
        h, w, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        filename = foldername + "/replay_" + current_time + ".mp4"
        out = cv2.VideoWriter(filename, fourcc, fps, (w, h))
        for frame in frames:
            out.write(frame)
        out.release()
        print(f"Replay saved successfully to: {filename}")
    except Exception as e:
        print(f"Error saving replay video: {e}")
    finally:
        is_saving_replay = False

# Premium Color Palette (BGR format for OpenCV)
COLOR_HUD_BG = (20, 20, 20)           # Dark grey for panels
COLOR_NEON_CYAN = (255, 255, 0)       # Player 1 / Target color
COLOR_NEON_MAGENTA = (255, 0, 240)    # Player 2 / Active highlight
COLOR_NEON_GREEN = (0, 255, 127)      # Success indicator
COLOR_NEON_RED = (80, 80, 255)        # Warning / Alert indicator
COLOR_TEXT_MUTED = (160, 160, 160)     # Subtitles
COLOR_TEXT_WHITE = (245, 245, 245)     # Regular text

# Game States
STATE_SELECTING = 0
STATE_PLAYING = 1
STATE_RESULT = 2
STATE_REPLAY = 3

# Available classes in our custom yolo11_toy_cars.pt model
AVAILABLE_CARS = ["daihatsu", "lamborghini", "mini cooper", "prius", "sienta", "tesla"]

class Particle:
    def __init__(self, x, y, color):
        self.x = float(x)
        self.y = float(y)
        # Velocity radiating outwards in a circle
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(3, 9)
        self.vx = speed * math.cos(angle)
        self.vy = speed * math.sin(angle) - random.uniform(2, 4) # slight upward blast
        self.gravity = 0.20
        self.creation_time = time.time()
        self.lifetime = random.uniform(0.7, 1.4)
        self.color = color
        self.size = random.randint(3, 6)
        
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += self.gravity
        
    def is_alive(self):
        return time.time() - self.creation_time < self.lifetime
        
    def draw(self, img):
        age = time.time() - self.creation_time
        alpha = max(0.0, 1.0 - (age / self.lifetime))
        color = tuple(int(c * alpha) for c in self.color)
        cv2.circle(img, (int(self.x), int(self.y)), self.size, color, -1)

def draw_transparent_png(face_img, overlay_img, x, y, size=32):
    """Draws a BGRA overlay image onto a BGR face image at position (x, y) with a target size."""
    overlay_resized = cv2.resize(overlay_img, (size, size))
    h_face, w_face, _ = face_img.shape
    r = size // 2
    
    x1, y1 = x - r, y - r
    x2, y2 = x + r, y + r
    
    ox1, oy1 = 0, 0
    ox2, oy2 = size, size
    
    if x1 < 0:
        ox1 += -x1
        x1 = 0
    if y1 < 0:
        oy1 += -y1
        y1 = 0
    if x2 > w_face:
        ox2 -= (x2 - w_face)
        x2 = w_face
    if y2 > h_face:
        oy2 -= (y2 - h_face)
        y2 = h_face
        
    if x1 >= x2 or y1 >= y2:
        return
        
    roi = face_img[y1:y2, x1:x2]
    overlay_roi = overlay_resized[oy1:oy2, ox1:ox2]
    
    if overlay_roi.shape[2] == 4:
        b, g, r_ch, a = cv2.split(overlay_roi)
        alpha = a.astype(float) / 255.0
        channels = [b, g, r_ch]
        for c in range(3):
            roi[:, :, c] = (alpha * channels[c] + (1.0 - alpha) * roi[:, :, c]).astype(np.uint8)
    else:
        face_img[y1:y2, x1:x2] = overlay_roi[:, :, :3]

def draw_glass_panel(img, pt1, pt2, color=COLOR_HUD_BG, alpha=0.7, border_color=COLOR_NEON_CYAN):
    """Draws a semi-transparent panel with a fine neon border."""
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
    cv2.rectangle(img, pt1, pt2, border_color, 1)

def draw_sci_fi_corners(img, pt1, pt2, color, thickness=2, length=12):
    """Draws high-tech corner brackets for active targets."""
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)

def calculate_target_points(x1, y1, x2, y2, tx, ty):
    """Calculates point value of a bounding box relative to target centered at tx, ty."""
    # Small square: 3 points. Half-side = 30
    if x1 <= tx + 30 and x2 >= tx - 30 and y1 <= ty + 30 and y2 >= ty - 30:
        return 3
    # Medium square: 2 points. Half-side = 70
    elif x1 <= tx + 70 and x2 >= tx - 70 and y1 <= ty + 70 and y2 >= ty - 70:
        return 2
    # Big square: 1 point. Half-side = 120
    elif x1 <= tx + 120 and x2 >= tx - 120 and y1 <= ty + 120 and y2 >= ty - 120:
        return 1
    return 0

def draw_target_squares(img, tx, ty):
    """Draws three concentric target squares with neon borders and point labels."""
    # Define half-sides and color codes
    sizes = [
        (120, "1 PT", (100, 100, 100)),     # Big: Muted grey
        (70, "2 PTS", COLOR_NEON_MAGENTA),  # Medium: Neon Magenta
        (30, "3 PTS", COLOR_NEON_CYAN)      # Small: Neon Cyan
    ]
    
    # Pulsating factor for visual interest
    pulse = int(3 * math.sin(time.time() * 5))
    
    # Outer glow (semi-transparent panel overlay around the target zone)
    glow = img.copy()
    cv2.rectangle(glow, (tx - 120, ty - 120), (tx + 120, ty + 120), COLOR_NEON_CYAN, -1)
    cv2.addWeighted(glow, 0.08, img, 0.92, 0, img)
    
    for size, label, color in sizes:
        # Add slight pulse to smaller rings to make them dynamic
        s = size if size == 120 else size + pulse
        pt1 = (tx - s, ty - s)
        pt2 = (tx + s, ty + s)
        cv2.rectangle(img, pt1, pt2, color, 1, cv2.LINE_AA)
        
        # Draw small sci-fi corners on each square
        draw_sci_fi_corners(img, pt1, pt2, color, thickness=1, length=10)
        
        # Label each square border
        cv2.putText(img, label, (tx - s + 5, ty - s + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
                    
    # Crosshair ticks at the exact center
    cv2.circle(img, (tx, ty), 3, COLOR_NEON_CYAN, -1)
    cv2.line(img, (tx - 8, ty), (tx + 8, ty), COLOR_NEON_CYAN, 1)
    cv2.line(img, (tx, ty - 8), (tx, ty + 8), COLOR_NEON_CYAN, 1)
    
    # Label text above target
    cv2.putText(img, "TARGET ZONE (SQUARES)", (tx - 65, ty - 128),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_NEON_CYAN, 1, cv2.LINE_AA)

def draw_status_bar(img, state, p1_chosen, p2_chosen, winner_str=""):
    """Draws a game-status bar at the top of the screen."""
    h, w, _ = img.shape
    bar_h = 45
    draw_glass_panel(img, (0, 0), (w, bar_h), alpha=0.8, border_color=(40, 40, 40))
    
    title_text = "YOLO TOY CAR COMPETITION // TACTICAL INFRARED SCANNER"
    cv2.putText(img, title_text, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
    
    state_desc = ""
    color = COLOR_NEON_CYAN
    if state == STATE_SELECTING:
        state_desc = "STAGE: CAR SELECTION (CHOOSE 3 CARS EACH)"
        color = COLOR_NEON_CYAN
    elif state == STATE_PLAYING:
        state_desc = "STAGE: PLAYING (ROLL CARS AND PRESS SPACE TO LOCK IN)"
        color = COLOR_NEON_MAGENTA
    elif state == STATE_RESULT:
        state_desc = f"ROUND COMPLETE: {winner_str.upper()}"
        color = COLOR_NEON_GREEN
        
    cv2.putText(img, state_desc, (w - 480, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

def draw_player_sideboards(img, p1_cars, p2_cars, p1_car_points, p2_car_points, p1_round_points, p2_round_points, p1_score, p2_score, state, p1_coins_collected=0, p2_coins_collected=0):
    """Draws player status boards on the left and right sides of the screen."""
    h, w, _ = img.shape
    panel_w = 230
    panel_h = 240
    
    # Calculate max points for starring
    p1_max_pts = max(p1_car_points.values()) if p1_car_points else 0
    p2_max_pts = max(p2_car_points.values()) if p2_car_points else 0
    
    # Player 1 Panel (Left)
    p1_x1, p1_y1 = 15, 60
    p1_x2, p1_y2 = p1_x1 + panel_w, p1_y1 + panel_h
    draw_glass_panel(img, (p1_x1, p1_y1), (p1_x2, p1_y2), alpha=0.6, border_color=COLOR_NEON_CYAN)
    
    cv2.putText(img, "PLAYER 1 STATUS", (p1_x1 + 15, p1_y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_NEON_CYAN, 1, cv2.LINE_AA)
    cv2.line(img, (p1_x1 + 15, y1_p1 := p1_y1 + 32), (p1_x2 - 15, y1_p1), COLOR_NEON_CYAN, 1)
    
    cv2.putText(img, f"SCORE: {p1_score}", (p1_x1 + 15, p1_y1 + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_NEON_GREEN, 1, cv2.LINE_AA)
    
    y_offset = p1_y1 + 70
    cv2.putText(img, "CHOSEN VEHICLES:", (p1_x1 + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    for car in p1_cars:
        y_offset += 20
        pts = p1_car_points.get(car, 0)
        is_best = (state != STATE_SELECTING and pts > 0 and pts == p1_max_pts)
        car_color = COLOR_NEON_GREEN if pts > 0 else COLOR_TEXT_WHITE
        star = " -> " if is_best else "  "
        pts_str = f" ({pts} PTS)" if state != STATE_SELECTING else ""
        cv2.putText(img, f"{star}{car.upper()}{pts_str}", (p1_x1 + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, car_color, 1, cv2.LINE_AA)
        
    y_offset = p1_y2 - 50
    cv2.line(img, (p1_x1 + 15, y_offset - 10), (p1_x2 - 15, y_offset - 10), (50, 50, 50), 1)
    
    cv2.putText(img, "ROUND POINTS:", (p1_x1 + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    cv2.putText(img, f"{p1_round_points} PTS", (p1_x1 + 15, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_NEON_GREEN if p1_round_points > 0 else COLOR_TEXT_MUTED, 2, cv2.LINE_AA)
    
    # Player 2 Panel (Right)
    p2_x1, p2_y1 = w - panel_w - 15, 60
    p2_x2, p2_y2 = w - 15, p2_y1 + panel_h
    draw_glass_panel(img, (p2_x1, p2_y1), (p2_x2, p2_y2), alpha=0.6, border_color=COLOR_NEON_MAGENTA)
    
    cv2.putText(img, "PLAYER 2 STATUS", (p2_x1 + 15, p2_y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_NEON_MAGENTA, 1, cv2.LINE_AA)
    cv2.line(img, (p2_x1 + 15, y1_p2 := p2_y1 + 32), (p2_x2 - 15, y1_p2), COLOR_NEON_MAGENTA, 1)
    
    cv2.putText(img, f"SCORE: {p2_score}", (p2_x1 + 15, p2_y1 + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_NEON_GREEN, 1, cv2.LINE_AA)
    
    y_offset = p2_y1 + 70
    cv2.putText(img, "CHOSEN VEHICLES:", (p2_x1 + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    for car in p2_cars:
        y_offset += 20
        pts = p2_car_points.get(car, 0)
        is_best = (state != STATE_SELECTING and pts > 0 and pts == p2_max_pts)
        car_color = COLOR_NEON_GREEN if pts > 0 else COLOR_TEXT_WHITE
        star = " -> " if is_best else "  "
        pts_str = f" ({pts} PTS)" if state != STATE_SELECTING else ""
        cv2.putText(img, f"{star}{car.upper()}{pts_str}", (p2_x1 + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, car_color, 1, cv2.LINE_AA)
        
    y_offset = p2_y2 - 50
    cv2.line(img, (p2_x1 + 15, y_offset - 10), (p2_x2 - 15, y_offset - 10), (50, 50, 50), 1)
    
    cv2.putText(img, "ROUND POINTS:", (p2_x1 + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    cv2.putText(img, f"{p2_round_points} PTS", (p2_x1 + 15, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_NEON_GREEN if p2_round_points > 0 else COLOR_TEXT_MUTED, 2, cv2.LINE_AA)



def generate_safe_target_spot(w, h):
    """Generates a random target spot near the center of the table and at least 200px away from the parking bays."""
    # Bay coordinates matching the drawing
    x1_p1, x2_p1 = int(w * 0.15), int(w * 0.25)
    y1_p1, y2_p1 = int(h * 0.28), int(h * 0.42)
    
    x1_p2, x2_p2 = int(w * 0.69), int(w * 0.79)
    y1_p2, y2_p2 = int(h * 0.28), int(h * 0.42)
    
    while True:
        # Pick random target spot restricted to the center area of the table
        tx = random.randint(int(w * 0.40), int(w * 0.60))
        ty = random.randint(int(h * 0.45), int(h * 0.75))
        
        # Distance to P1 Bay (distance to nearest point of rectangle)
        dx1 = max(0, x1_p1 - tx, tx - x2_p1)
        dy1 = max(0, y1_p1 - ty, ty - y2_p1)
        dist_p1 = math.sqrt(dx1**2 + dy1**2)
        
        # Distance to P2 Bay
        dx2 = max(0, x1_p2 - tx, tx - x2_p2)
        dy2 = max(0, y1_p2 - ty, ty - y2_p2)
        dist_p2 = math.sqrt(dx2**2 + dy2**2)
        
        if dist_p1 >= 200 and dist_p2 >= 200:
            return tx, ty

def assign_cars(p1_score, p2_score):
    """Assigns 3 unique cars to each player, giving the leading player a higher probability of getting 'daihatsu'."""
    # Determine probability of P1 getting 'daihatsu'
    if p1_score > p2_score:
        p1_daihatsu_prob = 0.80
    elif p2_score > p1_score:
        p1_daihatsu_prob = 0.20
    else:
        p1_daihatsu_prob = 0.50
        
    p1_gets_daihatsu = random.random() < p1_daihatsu_prob
    
    remaining_cars = [c for c in AVAILABLE_CARS if c != "daihatsu"]
    random.shuffle(remaining_cars)
    
    if p1_gets_daihatsu:
        p1_cars = ["daihatsu"] + remaining_cars[:2]
        p2_cars = remaining_cars[2:]
    else:
        p2_cars = ["daihatsu"] + remaining_cars[:2]
        p1_cars = remaining_cars[2:]
        
    return p1_cars, p2_cars

def get_p1_safe_zone(w, h):
    return 0, int(w * 0.5), int(h * 0.20), int(h * 0.41)

def get_p2_safe_zone(w, h):
    return int(w * 0.5), w, int(h * 0.20), int(h * 0.41)

def generate_safe_coin_spots(w, h, tx, ty, num_coins=5):
    """Generates random coin coordinates near the target (80px to 220px distance) but not too close."""
    x1_p1, x2_p1 = int(w * 0.15), int(w * 0.25)
    y1_p1, y2_p1 = int(h * 0.28), int(h * 0.42)
    
    x1_p2, x2_p2 = int(w * 0.69), int(w * 0.79)
    y1_p2, y2_p2 = int(h * 0.28), int(h * 0.42)
    
    spots = []
    attempts = 0
    while len(spots) < num_coins and attempts < 150:
        attempts += 1
        
        # Spawn near target: distance between 80px and 220px
        angle = random.uniform(0, 2 * math.pi)
        r = random.uniform(80, 220)
        cx = int(tx + r * math.cos(angle))
        cy = int(ty + r * math.sin(angle))
        
        # Ensure within screen bounds
        if not (50 <= cx <= w - 50 and 50 <= cy <= h - 50):
            continue
            
        dx1 = max(0, x1_p1 - cx, cx - x2_p1)
        dy1 = max(0, y1_p1 - cy, cy - y2_p1)
        dist_p1 = math.sqrt(dx1**2 + dy1**2)
        
        dx2 = max(0, x1_p2 - cx, cx - x2_p2)
        dy2 = max(0, y1_p2 - cy, cy - y2_p2)
        dist_p2 = math.sqrt(dx2**2 + dy2**2)
        
        if dist_p1 >= 100 and dist_p2 >= 100:
            too_close = False
            for sx, sy in spots:
                if math.sqrt((sx - cx)**2 + (sy - cy)**2) < 60:
                    too_close = True
                    break
            if not too_close:
                spots.append((cx, cy))
    return spots

def draw_safe_zones(img):
    """Draws the P1 and P2 Safe Zones above the parking lots."""
    h, w, _ = img.shape
    
    # P1 Safe Zone
    x1_p1, x2_p1, y1_p1, y2_p1 = get_p1_safe_zone(w, h)
    draw_glass_panel(img, (x1_p1, y1_p1), (x2_p1, y2_p1), alpha=0.3, border_color=COLOR_NEON_CYAN)
    cv2.putText(img, "P1 SAFE ZONE", (x1_p1 + 10, y1_p1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_NEON_CYAN, 1, cv2.LINE_AA)
    
    # P2 Safe Zone
    x1_p2, x2_p2, y1_p2, y2_p2 = get_p2_safe_zone(w, h)
    draw_glass_panel(img, (x1_p2, y1_p2), (x2_p2, y2_p2), alpha=0.3, border_color=COLOR_NEON_MAGENTA)
    cv2.putText(img, "P2 SAFE ZONE", (x1_p2 + 10, y1_p2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_NEON_MAGENTA, 1, cv2.LINE_AA)

def draw_selection_overlay(img, p1_cars, p2_cars, p1_safe_cars, p2_safe_cars):
    """Draws an elegant, glassmorphic selection window in the center of the screen."""
    h, w, _ = img.shape
    
    # Dim background webcam feed
    dim_overlay = img.copy()
    cv2.rectangle(dim_overlay, (0, 0), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(dim_overlay, 0.6, img, 0.4, 0, img)
    
    card_w, card_h = 580, 440
    x1, y1 = (w - card_w) // 2, (h - card_h) // 2
    x2, y2 = x1 + card_w, y1 + card_h
    
    draw_glass_panel(img, (x1, y1), (x2, y2), alpha=0.85, border_color=COLOR_NEON_CYAN)
    
    # Title
    cv2.putText(img, "SELECT CARS FOR COMPETITION", (x1 + 30, y1 + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_NEON_CYAN, 2, cv2.LINE_AA)
    cv2.line(img, (x1 + 30, y1 + 45), (x2 - 30, y1 + 45), (100, 100, 100), 1)
    
    # P1 selection column
    p1_col_x = x1 + 40
    cv2.putText(img, "PLAYER 1: SELECTED VEHICLES", (p1_col_x, y1 + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_NEON_CYAN, 1, cv2.LINE_AA)
    
    for i, car in enumerate(AVAILABLE_CARS):
        y_pos = y1 + 115 + i * 40
        is_selected = car in p1_cars
        is_safe = car in p1_safe_cars
        
        if is_selected:
            box_color = COLOR_NEON_GREEN if is_safe else COLOR_NEON_RED
        else:
            box_color = (80, 80, 80)
            
        cv2.rectangle(img, (p1_col_x, y_pos - 15), (p1_col_x + 18, y_pos + 3), box_color, -1 if is_selected else 2)
        if is_selected:
            # Draw tiny check mark in box
            cv2.line(img, (p1_col_x + 4, y_pos - 6), (p1_col_x + 8, y_pos - 2), (0, 0, 0), 2)
            cv2.line(img, (p1_col_x + 8, y_pos - 2), (p1_col_x + 14, y_pos - 10), (0, 0, 0), 2)
            
        status_suffix = " (READY)" if (is_selected and is_safe) else (" (MISSING)" if is_selected else "")
        cv2.putText(img, f"{car.upper()}{status_suffix}", (p1_col_x + 30, y_pos - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_WHITE if is_selected else COLOR_TEXT_MUTED, 2 if is_selected else 1, cv2.LINE_AA)
        
    # P2 selection column
    p2_col_x = x1 + 320
    cv2.putText(img, "PLAYER 2: SELECTED VEHICLES", (p2_col_x, y1 + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_NEON_MAGENTA, 1, cv2.LINE_AA)
    
    for i, car in enumerate(AVAILABLE_CARS):
        y_pos = y1 + 115 + i * 40
        is_selected = car in p2_cars
        is_safe = car in p2_safe_cars
        
        if is_selected:
            box_color = COLOR_NEON_GREEN if is_safe else COLOR_NEON_RED
        else:
            box_color = (80, 80, 80)
            
        cv2.rectangle(img, (p2_col_x, y_pos - 15), (p2_col_x + 18, y_pos + 3), box_color, -1 if is_selected else 2)
        if is_selected:
            # Draw tiny check mark
            cv2.line(img, (p2_col_x + 4, y_pos - 6), (p2_col_x + 8, y_pos - 2), (0, 0, 0), 2)
            cv2.line(img, (p2_col_x + 8, y_pos - 2), (p2_col_x + 14, y_pos - 10), (0, 0, 0), 2)
            
        status_suffix = " (READY)" if (is_selected and is_safe) else (" (MISSING)" if is_selected else "")
        cv2.putText(img, f"{car.upper()}{status_suffix}", (p2_col_x + 30, y_pos - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_WHITE if is_selected else COLOR_TEXT_MUTED, 2 if is_selected else 1, cv2.LINE_AA)

    # Info footer
    cv2.line(img, (x1 + 30, y2 - 65), (x2 - 30, y2 - 65), (100, 100, 100), 1)
    
    p1_ready = len(p1_safe_cars) == 3
    p2_ready = len(p2_safe_cars) == 3
    
    if p1_ready and p2_ready:
        cv2.putText(img, "PRESS [SPACE] TO START GAME // [R] TO RE-ROLL CARS", (x1 + 65, y2 - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_NEON_GREEN, 2, cv2.LINE_AA)
    else:
        status_txt = f"PLACE ALL SELECTED CARS IN SAFE ZONES... P1: {len(p1_safe_cars)}/3 | P2: {len(p2_safe_cars)}/3"
        cv2.putText(img, status_txt, (x1 + 55, y2 - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_NEON_RED, 1, cv2.LINE_AA)

def main():
    # Load model
    MODEL_PATH = Path("/Users/kimi/Desktop/j/ ai hand whiteing boaro/yolo_toy_cars_model.pt")
    print(f"Loading YOLO Model: {MODEL_PATH}...")
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}. Train the model first.")
        return
        
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cv2.namedWindow("YOLO Toy Car Game HUD", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("YOLO Toy Car Game HUD", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Game variables
    game_state = STATE_SELECTING
    p1_score = 0
    p2_score = 0
    coins = []
    p1_cars, p2_cars = assign_cars(p1_score, p2_score)
    
    # Load coin image
    coin_img_path = Path("/Users/kimi/Desktop/j/ ai hand whiteing boaro/target_coin.png")
    coin_overlay = None
    if coin_img_path.exists():
        coin_overlay = cv2.imread(str(coin_img_path), cv2.IMREAD_UNCHANGED)
        print("Loaded target_coin.png successfully.")
    
    tx, ty = 0, 0
    winner_str = ""
    winning_pos = (0, 0)
    winning_color = (255, 255, 255)
    
    # Active round scores
    p1_round_points = 0
    p2_round_points = 0
    p1_car_points = {}
    p2_car_points = {}
    p1_best_pos = None
    p2_best_pos = None
    p1_coins_collected = 0
    p2_coins_collected = 0
    
    # Replay recording buffer
    recorded_frames = []
    
    # Safe zone tracked counts
    p1_safe_cars = set()
    p2_safe_cars = set()
    
    # Animation variables
    particles = []
    frozen_frame = None
    
    global bg_music, is_saving_replay
    
    # Start background music loop (using 'afplay' on macOS)
    music_path = Path("/Users/kimi/Desktop/j/ ai hand whiteing boaro/background_music.mp3")
    if music_path.exists():
        try:
            bg_music = subprocess.Popen(
                f"while true; do afplay '{music_path}'; done",
                shell=True,
                preexec_fn=os.setsid
            )
            print("Background music started.")
        except Exception as e:
            print(f"Error starting background music: {e}")
    else:
        print(f"Music file not found at {music_path}")
        
    print("\nGame loaded successfully!")
    print("Cars randomly assigned to players. Press R to re-roll, SPACE to lock in.")
    
    while True:
        if game_state == STATE_SELECTING:
            ret, frame = cap.read()
            if not ret:
                break
                
            display_frame = frame.copy()
            h, w, _ = display_frame.shape
            
            # Predict objects to check if cars are in safe zone
            results = model.track(source=frame, persist=True, verbose=False, tracker="bytetrack.yaml")
            
            p1_safe_cars = set()
            p2_safe_cars = set()
            
            rect_p1 = get_p1_safe_zone(w, h)
            rect_p2 = get_p2_safe_zone(w, h)
            
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                names = model.names
                if boxes.id is not None:
                    clss = boxes.cls.int().cpu().tolist()
                    xyxys = boxes.xyxy.cpu().tolist()
                    for xyxy, cls in zip(xyxys, clss):
                        x1, y1, x2, y2 = map(int, xyxy)
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        cls_name = names[cls]
                        
                        if cls_name in p1_cars and (rect_p1[0] <= cx <= rect_p1[1] and rect_p1[2] <= cy <= rect_p1[3]):
                            p1_safe_cars.add(cls_name)
                            draw_sci_fi_corners(display_frame, (x1, y1), (x2, y2), COLOR_NEON_CYAN, thickness=2)
                            cv2.putText(display_frame, f"{cls_name.upper()} (SAFE)", (x1, y1 - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_NEON_CYAN, 1, cv2.LINE_AA)
                        elif cls_name in p2_cars and (rect_p2[0] <= cx <= rect_p2[1] and rect_p2[2] <= cy <= rect_p2[3]):
                            p2_safe_cars.add(cls_name)
                            draw_sci_fi_corners(display_frame, (x1, y1), (x2, y2), COLOR_NEON_MAGENTA, thickness=2)
                            cv2.putText(display_frame, f"{cls_name.upper()} (SAFE)", (x1, y1 - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_NEON_MAGENTA, 1, cv2.LINE_AA)
            
            draw_safe_zones(display_frame)
            draw_status_bar(display_frame, game_state, p1_cars, p2_cars)
            draw_selection_overlay(display_frame, p1_cars, p2_cars, p1_safe_cars, p2_safe_cars)
            
            # Record selection frame for replay
            recorded_frames.append(display_frame.copy())
            if len(recorded_frames) > 1800:
                recorded_frames.pop(0)
                
            cv2.imshow("YOLO Toy Car Game HUD", display_frame)
            
        elif game_state == STATE_PLAYING:
            ret, frame = cap.read()
            if not ret:
                break
                
            display_frame = frame.copy()
            h, w, _ = display_frame.shape
            
            
            # Predict objects using custom model
            results = model.track(source=frame, persist=True, verbose=False, tracker="bytetrack.yaml")
            
            p1_car_points = {car: 0 for car in p1_cars}
            p2_car_points = {car: 0 for car in p2_cars}
            p1_best_pos = None
            p2_best_pos = None
            p1_max_pts = 0
            p2_max_pts = 0
            
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                names = model.names
                
                if boxes.id is not None:
                    track_ids = boxes.id.int().cpu().tolist()
                    xyxys = boxes.xyxy.cpu().tolist()
                    clss = boxes.cls.int().cpu().tolist()
                    confs = boxes.conf.cpu().tolist()
                    
                    for xyxy, track_id, cls, conf in zip(xyxys, track_ids, clss, confs):
                        x1, y1, x2, y2 = map(int, xyxy)
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        cls_name = names[cls]
                        
                        # Check if this car belongs to P1 or P2
                        belongs_to_p1 = cls_name in p1_cars
                        belongs_to_p2 = cls_name in p2_cars
                        
                        if not belongs_to_p1 and not belongs_to_p2:
                            # Skip untargeted cars
                            continue
                            
                        # Calculate target points using bounding box intersection
                        car_pts = calculate_target_points(x1, y1, x2, y2, tx, ty)
                        
                        # Assign bounding box colors based on player ownership
                        tag_color = COLOR_NEON_CYAN if belongs_to_p1 else COLOR_NEON_MAGENTA
                        
                        # Check collision with coins
                        for coin in coins:
                            if not coin["collected"]:
                                cdist = math.sqrt((cx - coin["x"])**2 + (cy - coin["y"])**2)
                                if cdist < 35:
                                    coin["collected"] = True
                                    os.system("afplay ding_sound_effect.mp3 &")
                                    if belongs_to_p1:
                                        p1_score += 1
                                        p1_coins_collected += 1
                                    elif belongs_to_p2:
                                        p2_score += 1
                                        p2_coins_collected += 1
                                    # Spawn spark particles
                                    for _ in range(15):
                                        particles.append(Particle(coin["x"], coin["y"], (0, 215, 255)))
                                        
                        # Update best scores and positions
                        if belongs_to_p1:
                            p1_car_points[cls_name] = max(p1_car_points.get(cls_name, 0), car_pts)
                            if car_pts >= p1_max_pts:
                                p1_max_pts = car_pts
                                p1_best_pos = (cx, cy)
                        elif belongs_to_p2:
                            p2_car_points[cls_name] = max(p2_car_points.get(cls_name, 0), car_pts)
                            if car_pts >= p2_max_pts:
                                p2_max_pts = car_pts
                                p2_best_pos = (cx, cy)
                            
                        # --- DRAW SCENE OVERLAYS ---
                        # Target corners around car
                        draw_sci_fi_corners(display_frame, (x1, y1), (x2, y2), tag_color, thickness=2)
                        
                        # Connection line to target spot
                        # Color of line based on target tier
                        if car_pts == 3:
                            line_color = COLOR_NEON_GREEN
                        elif car_pts == 2:
                            line_color = COLOR_NEON_CYAN
                        elif car_pts == 1:
                            line_color = (0, 255, 255) # Yellow-cyan
                        else:
                            line_color = (100, 100, 100) # Muted grey
                            
                        cv2.line(display_frame, (cx, cy), (tx, ty), line_color, 1, cv2.LINE_AA)
                        
                        # Draw label tag above car bounding box
                        label_txt = f"{cls_name.upper()} (ID:{track_id})"
                        cv2.putText(display_frame, label_txt, (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, tag_color, 1, cv2.LINE_AA)
                        
                        # Print point text on the center of the line
                        mid_x, mid_y = (cx + tx) // 2, (cy + ty) // 2
                        pts_txt = f"+{car_pts} PTS" if car_pts > 0 else "0 PTS"
                        cv2.putText(display_frame, pts_txt, (mid_x - 15, mid_y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, line_color, 1, cv2.LINE_AA)
            
            # Calculate sum of round points (including coin collection)
            p1_round_points = sum(p1_car_points.values()) + p1_coins_collected
            p2_round_points = sum(p2_car_points.values()) + p2_coins_collected

            # Draw coins
            for coin in coins:
                if not coin["collected"]:
                    if coin_overlay is not None:
                        draw_transparent_png(display_frame, coin_overlay, coin["x"], coin["y"], size=28)
                    else:
                        cv2.circle(display_frame, (coin["x"], coin["y"]), 12, (0, 215, 255), -1, cv2.LINE_AA)
                        cv2.circle(display_frame, (coin["x"], coin["y"]), 12, (0, 255, 255), 1, cv2.LINE_AA)
                        
            # Update and draw active sparks
            particles = [p for p in particles if p.is_alive()]
            for p in particles:
                p.update()
                p.draw(display_frame)
                
            # Draw target and sidebar HUDs
            draw_target_squares(display_frame, tx, ty)
            draw_status_bar(display_frame, game_state, p1_cars, p2_cars)
            draw_player_sideboards(display_frame, p1_cars, p2_cars, p1_car_points, p2_car_points, p1_round_points, p2_round_points, p1_score, p2_score, game_state, p1_coins_collected, p2_coins_collected)
            
            # Quick instructions on bottom
            cv2.putText(display_frame, "PRESS [SPACE] TO LOCK IN & FIND WINNER // [S] TO CHANGE CARS", (w // 2 - 250, h - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
            
            # Record current frame for replay
            recorded_frames.append(display_frame.copy())
            if len(recorded_frames) > 1800:
                recorded_frames.pop(0)
                
            cv2.imshow("YOLO Toy Car Game HUD", display_frame)
            
        elif game_state == STATE_RESULT:
            # Animate particles on top of the frozen locked-in frame
            display_frame = frozen_frame.copy()
            h, w, _ = display_frame.shape
            
            # Update and draw active sparks
            particles = [p for p in particles if p.is_alive()]
            for p in particles:
                p.update()
                p.draw(display_frame)
                
            # Draw game state overlays
            draw_target_squares(display_frame, tx, ty)
            draw_status_bar(display_frame, game_state, p1_cars, p2_cars, winner_str)
            draw_player_sideboards(display_frame, p1_cars, p2_cars, p1_car_points, p2_car_points, p1_round_points, p2_round_points, p1_score, p2_score, game_state, p1_coins_collected, p2_coins_collected)
            
            # Winner celebration pop-up overlay card
            card_w, card_h = 440, 160
            cx1, cy1 = (w - card_w) // 2, (h - card_h) // 2
            cx2, cy2 = cx1 + card_w, cy1 + card_h
            
            # Panel glows with winner color
            draw_glass_panel(display_frame, (cx1, cy1), (cx2, cy2), alpha=0.85, border_color=winning_color)
            
            cv2.putText(display_frame, "MATCH RESULTS LOCK-IN", (cx1 + 25, cy1 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
            
            cv2.putText(display_frame, winner_str, (cx1 + 25, cy1 + 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, winning_color, 2, cv2.LINE_AA)
                        
            # Details description
            details_str = f"P1: {p1_round_points} PTS   vs   P2: {p2_round_points} PTS"
            cv2.putText(display_frame, details_str, (cx1 + 25, cy1 + 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
            
            # Show replay saved/saving indicator
            if is_saving_replay:
                cv2.putText(display_frame, "SAVING REPLAY TO MP4...", (cx1 + 25, cy1 + 122),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1, cv2.LINE_AA) # Orange/amber
            else:
                cv2.putText(display_frame, "REPLAY SAVED TO 'last_replay.mp4'", (cx1 + 25, cy1 + 122),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_NEON_GREEN, 1, cv2.LINE_AA)
                        
            cv2.putText(display_frame, "PRESS [R] NEXT ROUND // [P] REPLAY // [S] RE-SELECT // [Q] QUIT", (cx1 + 10, cy2 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
            
            cv2.imshow("YOLO Toy Car Game HUD", display_frame)
        
        elif game_state == STATE_REPLAY:
            # Play back recorded frames
            for frame_idx, r_frame in enumerate(recorded_frames):
                replay_frame = r_frame.copy()
                h, w, _ = replay_frame.shape
                
                # Draw blinking "REPLAY" watermark in top-center
                dot_visible = (int(time.time() * 3) % 2 == 0)
                dot_color = (0, 0, 255) if dot_visible else (50, 50, 50)
                cv2.circle(replay_frame, (w // 2 - 60, 25), 6, dot_color, -1, cv2.LINE_AA)
                cv2.putText(replay_frame, "REPLAY PLAYBACK", (w // 2 - 45, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2, cv2.LINE_AA)
                
                # Progress bar at the bottom
                bar_y = h - 10
                progress_w = int((frame_idx + 1) / len(recorded_frames) * w)
                cv2.rectangle(replay_frame, (0, bar_y), (progress_w, h), COLOR_NEON_CYAN, -1)
                
                # Show instructions
                cv2.putText(replay_frame, "PRESS [SPACE] OR [P] TO SKIP REPLAY", (20, h - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
                
                cv2.imshow("YOLO Toy Car Game HUD", replay_frame)
                
                # Check for escape/skip key (30ms delay to match 33fps playback)
                key = cv2.waitKey(30) & 0xFF
                if key in [ord(' '), ord('p'), ord('q'), 27]: # space, P, Q, or ESC
                    break
            
            # Replay ended, return to result screen
            game_state = STATE_RESULT

        # Handle Keyboard Inputs
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Exiting game...")
            break
            
        elif key == ord('s'):
            # Return to selection screen
            p1_score = 0
            p2_score = 0
            p1_coins_collected = 0
            p2_coins_collected = 0
            p1_round_points = 0
            p2_round_points = 0
            p1_car_points = {}
            p2_car_points = {}
            p1_best_pos = None
            p2_best_pos = None
            recorded_frames.clear()
            p1_cars, p2_cars = assign_cars(p1_score, p2_score)
            coins.clear()
            particles.clear()
            game_state = STATE_SELECTING
            print("Returned to car selection screen with new randomized cars. Scores reset.")
            
        elif key == ord('p'):
            if game_state == STATE_RESULT and recorded_frames:
                game_state = STATE_REPLAY
                print("Playing round replay...")
            
        elif key == ord('r'):
            if game_state == STATE_RESULT:
                # Trigger next round
                particles.clear()
                p1_coins_collected = 0
                p2_coins_collected = 0
                p1_round_points = 0
                p2_round_points = 0
                p1_car_points = {}
                p2_car_points = {}
                p1_best_pos = None
                p2_best_pos = None
                recorded_frames.clear()
                # Pick a new target spot
                ret, frame = cap.read()
                if ret:
                    h, w, _ = frame.shape
                    tx, ty = generate_safe_target_spot(w, h)
                    # Generate new coins near target
                    coins = [{"x": cx, "y": cy, "collected": False} for cx, cy in generate_safe_coin_spots(w, h, tx, ty, num_coins=5)]
                game_state = STATE_PLAYING
                print(f"New round started! New target coordinates: ({tx}, {ty})")
            elif game_state == STATE_SELECTING:
                # Re-roll randomized cars with Daihatsu bias based on score
                p1_cars, p2_cars = assign_cars(p1_score, p2_score)
                print(f"Re-rolled randomized cars (Daihatsu bias check). P1: {p1_cars}, P2: {p2_cars}")
            
        elif key == ord(' '):
            if game_state == STATE_SELECTING:
                # Lock selection and start playing if both chose exactly 3 and all are in the safe zone
                if len(p1_safe_cars) == 3 and len(p2_safe_cars) == 3:
                    # Select target spot
                    ret, frame = cap.read()
                    if ret:
                        h, w, _ = frame.shape
                        tx, ty = generate_safe_target_spot(w, h)
                        # Generate coins near target
                        coins = [{"x": cx, "y": cy, "collected": False} for cx, cy in generate_safe_coin_spots(w, h, tx, ty, num_coins=5)]
                    p1_coins_collected = 0
                    p2_coins_collected = 0
                    # Do not clear recorded_frames to keep the selection phase in the replay
                    game_state = STATE_PLAYING
                    print(f"Car selection finalized. Target zone locked at ({tx}, {ty}). Ready to play!")
            elif game_state == STATE_PLAYING:
                # Freeze current state and evaluate winner
                frozen_frame = display_frame.copy()
                
                # Evaluate results based on points
                if p1_round_points == p2_round_points:
                    if p1_round_points == 0:
                        winner_str = "DRAW (0 PTS - NO CARS IN TARGET)"
                    else:
                        winner_str = f"DRAW (TIE AT {p1_round_points} PTS)"
                    winning_pos = (tx, ty)
                    winning_color = COLOR_TEXT_WHITE
                elif p1_round_points > p2_round_points:
                    winner_str = f"PLAYER 1 WINS! ({p1_round_points} vs {p2_round_points} PTS)"
                    winning_pos = p1_best_pos if p1_best_pos else (tx, ty)
                    winning_color = COLOR_NEON_CYAN
                else:
                    winner_str = f"PLAYER 2 WINS! ({p2_round_points} vs {p1_round_points} PTS)"
                    winning_pos = p2_best_pos if p2_best_pos else (tx, ty)
                    winning_color = COLOR_NEON_MAGENTA
                
                # Add target points (excluding coin points, which were already added in real-time) to overall scoreboard scores
                p1_score += sum(p1_car_points.values())
                p2_score += sum(p2_car_points.values())
                
                # Save the replay to an MP4 video file asynchronously in a background thread
                is_saving_replay = True
                replay_foldername = "/Users/kimi/Desktop/j/ ai hand whiteing boaro/toy_car_replays"
                # Make a shallow copy of frames list to prevent concurrent modification during background writing
                frames_copy = list(recorded_frames)
                save_thread = threading.Thread(
                    target=save_replay_video,
                    args=(frames_copy, replay_foldername, 30.0),
                    daemon=True
                )
                save_thread.start()
                
                # Spawn physics sparks at the winner's position
                if winning_pos:
                    for _ in range(75):
                        particles.append(Particle(winning_pos[0], winning_pos[1], winning_color))
                        
                game_state = STATE_RESULT
                print(f"Round Locked! Results: {winner_str}")
                
        # Manual selection disabled in favor of computer auto-randomization

    cleanup()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup()
