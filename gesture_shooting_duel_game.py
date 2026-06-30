import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import random
import time
import threading
import numpy as np

# ── Hand skeleton connections (21 landmarks) ──
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# Colors (BGR)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 220, 0)
RED = (0, 0, 220)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
ORANGE = (0, 165, 255)
MAGENTA = (255, 0, 255)
DARK_BG = (40, 40, 40)
GOLD = (0, 215, 255)
FIRE_RED = (0, 60, 255)
DEEP_RED = (0, 0, 180)
LIGHT_BLUE = (255, 200, 100)

# Player colors
P1_COLOR = (0, 200, 255)   # Orange-gold
P2_COLOR = (255, 100, 100) # Light blue


def play_bang_sound():
    """Play a BANG sound using PyAudio in a background thread."""
    try:
        import pyaudio
        RATE = 44100
        duration = 0.4
        t = np.linspace(0, duration, int(RATE * duration), False)
        # Gunshot = white noise burst with fast decay
        noise = np.random.randn(len(t)).astype(np.float32)
        envelope = np.exp(-t * 15)  # Fast decay
        bang = (noise * envelope * 0.7).astype(np.float32)
        # Add a low boom
        boom = np.sin(2 * np.pi * 80 * t) * np.exp(-t * 8) * 0.5
        bang = (bang + boom.astype(np.float32))
        bang = np.clip(bang, -1.0, 1.0).astype(np.float32)

        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, output=True)
        stream.write(bang.tobytes())
        stream.stop_stream()
        stream.close()
        p.terminate()
    except Exception:
        pass  # No sound if pyaudio not available


def play_countdown_beep(freq=800, duration=0.1):
    """Play a short beep for countdown ticks."""
    try:
        import pyaudio
        RATE = 44100
        t = np.linspace(0, duration, int(RATE * duration), False)
        tone = (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)
        envelope = np.exp(-t * 20)
        tone = (tone * envelope).astype(np.float32)

        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, output=True)
        stream.write(tone.tobytes())
        stream.stop_stream()
        stream.close()
        p.terminate()
    except Exception:
        pass


def draw_hand_landmarks(frame, hand_landmarks_list, colors=None):
    """Draw hand skeleton on the frame."""
    h, w, _ = frame.shape
    for i, hand_landmarks in enumerate(hand_landmarks_list):
        points = []
        for lm in hand_landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)
            points.append((px, py))

        conn_color = colors[i] if colors and i < len(colors) else (255, 0, 0)
        dot_color = (0, 255, 0)

        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx], conn_color, 2)
        for pt in points:
            cv2.circle(frame, pt, 4, dot_color, -1)


def is_finger_stretched(hand_landmarks, tip_idx, pip_idx):
    """Check if a finger is extended (tip above PIP joint)."""
    return hand_landmarks[tip_idx].y < hand_landmarks[pip_idx].y


def detect_gun_gesture(hand_landmarks):
    """
    Detect a finger gun: index finger extended, others closed.
    Returns True if the hand is making a gun shape.
    """
    index_open = is_finger_stretched(hand_landmarks, 8, 6)
    middle_open = is_finger_stretched(hand_landmarks, 12, 10)
    ring_open = is_finger_stretched(hand_landmarks, 16, 14)
    pinky_open = is_finger_stretched(hand_landmarks, 20, 18)

    # Gun = only index finger out (others closed)
    if index_open and not middle_open and not ring_open and not pinky_open:
        return True
    return False


def detect_fist(hand_landmarks):
    """Detect a closed fist (all fingers closed)."""
    index_open = is_finger_stretched(hand_landmarks, 8, 6)
    middle_open = is_finger_stretched(hand_landmarks, 12, 10)
    ring_open = is_finger_stretched(hand_landmarks, 16, 14)
    pinky_open = is_finger_stretched(hand_landmarks, 20, 18)
    return not index_open and not middle_open and not ring_open and not pinky_open


def get_hand_center_x(hand_landmarks):
    """Get average x-position of a hand (normalised 0..1)."""
    total_x = sum(lm.x for lm in hand_landmarks)
    return total_x / len(hand_landmarks)


def draw_overlay(frame, alpha=0.7):
    """Draw a dark semi-transparent overlay."""
    h, w, _ = frame.shape
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), BLACK, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_centered_text(frame, text, y, scale=1.0, color=WHITE, thickness=2):
    """Draw text centered horizontally on the frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    h, w, _ = frame.shape
    x = (w - tw) // 2
    # Shadow
    cv2.putText(frame, text, (x + 2, y + 2), font, scale, BLACK, thickness + 2)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness)


def draw_muzzle_flash(frame, cx, cy, size=60):
    """Draw a muzzle flash effect at position."""
    # Bright center
    cv2.circle(frame, (cx, cy), size, YELLOW, -1)
    cv2.circle(frame, (cx, cy), size // 2, WHITE, -1)
    # Rays
    for angle in range(0, 360, 30):
        rad = np.radians(angle)
        ex = int(cx + np.cos(rad) * size * 1.5)
        ey = int(cy + np.sin(rad) * size * 1.5)
        cv2.line(frame, (cx, cy), (ex, ey), ORANGE, 3)


def draw_title_screen(frame):
    """Draw the title/waiting screen."""
    h, w, _ = frame.shape
    draw_overlay(frame, 0.8)

    # Title with glow effect
    draw_centered_text(frame, "QUICK DRAW", h // 4, 2.0, GOLD, 4)
    draw_centered_text(frame, "SHOOTING DUEL", h // 4 + 60, 1.2, FIRE_RED, 3)

    # Instructions
    draw_centered_text(frame, "2 PLAYER GAME", h // 2 - 20, 0.8, CYAN, 2)
    draw_centered_text(frame, "Each player shows a FIST to the camera", h // 2 + 30, 0.6, WHITE, 1)
    draw_centered_text(frame, "P1 = LEFT side    P2 = RIGHT side", h // 2 + 65, 0.6, WHITE, 1)
    draw_centered_text(frame, 'After "BANG!" - first to point finger gun WINS!', h // 2 + 100, 0.6, YELLOW, 1)
    draw_centered_text(frame, "Finger Gun = ONLY index finger extended", h // 2 + 135, 0.6, ORANGE, 1)

    # Pulsing "press space" text
    pulse = abs(int(time.time() * 4) % 2)
    if pulse:
        draw_centered_text(frame, "Press SPACE to start!", h - 80, 0.9, GREEN, 2)

    draw_centered_text(frame, "R=Reset  Q=Quit", h - 30, 0.5, (150, 150, 150), 1)


def draw_game_ui(frame, p1_score, p2_score, game_state, countdown_val,
                 p1_has_gun, p2_has_gun, result_text, p1_detected, p2_detected,
                 flash_timer, winner_side):
    """Draw the main game UI overlay."""
    h, w, _ = frame.shape

    # Dividing line
    cv2.line(frame, (w // 2, 100), (w // 2, h - 50), (80, 80, 80), 2)

    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), DARK_BG, -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    draw_centered_text(frame, "QUICK DRAW DUEL", 38, 0.9, GOLD, 2)

    # Score bar
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, 55), (w, 95), (25, 25, 25), -1)
    cv2.addWeighted(overlay2, 0.8, frame, 0.2, 0, frame)

    cv2.putText(frame, f"P1: {p1_score}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, P1_COLOR, 2)
    cv2.putText(frame, f"P2: {p2_score}", (w - 140, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, P2_COLOR, 2)

    # Player labels
    cv2.putText(frame, "<-- P1", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, P1_COLOR, 2)
    cv2.putText(frame, "P2 -->", (w - 110, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, P2_COLOR, 2)

    # Detection indicators
    p1_status = "FIST" if p1_detected else "No hand"
    p2_status = "FIST" if p2_detected else "No hand"
    if p1_has_gun:
        p1_status = "BANG!"
    if p2_has_gun:
        p2_status = "BANG!"

    # State-specific drawing
    if game_state == "get_ready":
        draw_centered_text(frame, "Show your FISTS!", h // 2, 1.2, YELLOW, 3)
        draw_centered_text(frame, "Both players must show a closed fist", h // 2 + 45, 0.6, WHITE, 1)

        # Show detection status
        p1_col = GREEN if p1_detected else RED
        p2_col = GREEN if p2_detected else RED
        cv2.putText(frame, f"P1: {p1_status}", (30, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, p1_col, 2)
        cv2.putText(frame, f"P2: {p2_status}", (w - 200, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, p2_col, 2)

    elif game_state == "countdown":
        # Big countdown number with dramatic styling
        if countdown_val > 0:
            draw_centered_text(frame, str(countdown_val), h // 2 + 20, 3.0, YELLOW, 5)
        else:
            draw_centered_text(frame, "...", h // 2 + 20, 2.0, RED, 4)

    elif game_state == "bang":
        # BANG! with screen flash
        if flash_timer > 0:
            # Red flash overlay
            flash_overlay = frame.copy()
            cv2.rectangle(flash_overlay, (0, 0), (w, h), FIRE_RED, -1)
            alpha = min(0.5, flash_timer * 2)
            cv2.addWeighted(flash_overlay, alpha, frame, 1 - alpha, 0, frame)

        draw_centered_text(frame, "BANG!", h // 2, 3.5, RED, 6)
        draw_centered_text(frame, "SHOOT NOW!", h // 2 + 60, 1.0, YELLOW, 2)

        # Show who has fired
        if p1_has_gun:
            cv2.putText(frame, "P1: FIRED!", (30, h - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
            draw_muzzle_flash(frame, w // 4, h // 2, 30)
        if p2_has_gun:
            cv2.putText(frame, "P2: FIRED!", (w - 200, h - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
            draw_muzzle_flash(frame, 3 * w // 4, h // 2, 30)

    elif game_state == "too_early":
        draw_centered_text(frame, "TOO EARLY!", h // 2, 2.0, RED, 4)
        draw_centered_text(frame, result_text, h // 2 + 50, 1.0, YELLOW, 2)

    elif game_state == "result":
        # Winner announcement
        if winner_side == "P1":
            result_color = P1_COLOR
        elif winner_side == "P2":
            result_color = P2_COLOR
        elif winner_side == "DRAW":
            result_color = YELLOW
        else:
            result_color = WHITE

        draw_centered_text(frame, result_text, h // 2, 1.5, result_color, 3)

        # Show gun status
        p1_txt = "SHOT!" if p1_has_gun else "MISSED"
        p2_txt = "SHOT!" if p2_has_gun else "MISSED"
        p1_col = GREEN if p1_has_gun else RED
        p2_col = GREEN if p2_has_gun else RED
        cv2.putText(frame, f"P1: {p1_txt}", (30, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, p1_col, 2)
        cv2.putText(frame, f"P2: {p2_txt}", (w - 220, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, p2_col, 2)

    # Bottom bar
    cv2.putText(frame, "SPACE=Start  R=Reset  Q=Quit", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)


def main():
    # Model path
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "hand_landmarker.task")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Download it with:")
        print('  curl -L -o hand_landmarker.task '
              '"https://storage.googleapis.com/mediapipe-models/'
              'hand_landmarker/hand_landmarker/float16/latest/'
              'hand_landmarker.task"')
        return

    # Configure hand landmarker
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("=== QUICK DRAW SHOOTING DUEL ===")
    print()
    print("HOW TO PLAY:")
    print("  1. Two players face the camera (P1=left, P2=right)")
    print("  2. Press SPACE, then both show FISTS")
    print("  3. Wait for the countdown... then BANG!")
    print("  4. First to point a FINGER GUN wins!")
    print()
    print("  Finger Gun = ONLY index finger extended")
    print()
    print("Controls:")
    print("  SPACE - Start round")
    print("  R     - Reset scores")
    print("  Q     - Quit")

    frame_ts = 0

    # Game state
    game_state = "title"   # title, get_ready, countdown, suspense, bang, result, too_early
    p1_score = 0
    p2_score = 0
    countdown_start = 0
    countdown_val = 3
    suspense_duration = 0   # Random delay before BANG
    suspense_start = 0
    bang_time = 0
    result_text = ""
    result_show_time = 0
    winner_side = ""
    p1_has_gun = False
    p2_has_gun = False
    p1_shot_time = 0
    p2_shot_time = 0
    p1_fist_detected = False
    p2_fist_detected = False
    flash_timer = 0
    last_beep_val = -1
    bang_played = False

    BANG_WINDOW = 3.0      # Seconds to detect a shot after BANG
    RESULT_DISPLAY = 3.0   # Seconds to show result

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # MediaPipe detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_ts += 33
        detection = landmarker.detect_for_video(mp_image, frame_ts)

        # Identify hands and gestures
        p1_gun = False
        p2_gun = False
        p1_fist = False
        p2_fist = False
        hands_sorted = []

        if detection.hand_landmarks:
            # Draw landmarks with player colors
            hand_colors = []
            for hand_lm in detection.hand_landmarks:
                cx = get_hand_center_x(hand_lm)
                hands_sorted.append((cx, hand_lm))
            hands_sorted.sort(key=lambda x: x[0])

            # Assign colors based on position
            for i, (cx, lm) in enumerate(hands_sorted):
                hand_colors.append(P1_COLOR if i == 0 else P2_COLOR)

            draw_hand_landmarks(frame, [h[1] for h in hands_sorted], hand_colors)

            if len(hands_sorted) >= 1:
                p1_gun = detect_gun_gesture(hands_sorted[0][1])
                p1_fist = detect_fist(hands_sorted[0][1])
            if len(hands_sorted) >= 2:
                p2_gun = detect_gun_gesture(hands_sorted[-1][1])
                p2_fist = detect_fist(hands_sorted[-1][1])

        p1_fist_detected = p1_fist
        p2_fist_detected = p2_fist

        # ── STATE MACHINE ──

        if game_state == "title":
            draw_title_screen(frame)

        elif game_state == "get_ready":
            # Wait for both players to show fists
            if p1_fist_detected and p2_fist_detected:
                game_state = "countdown"
                countdown_start = time.time()
                countdown_val = 3
                last_beep_val = -1
            draw_game_ui(frame, p1_score, p2_score, "get_ready", 0,
                         False, False, "", p1_fist_detected, p2_fist_detected,
                         0, "")

        elif game_state == "countdown":
            elapsed = time.time() - countdown_start
            countdown_val = 3 - int(elapsed)

            # Play beep on each new number
            if countdown_val != last_beep_val and countdown_val > 0:
                last_beep_val = countdown_val
                threading.Thread(target=play_countdown_beep, daemon=True).start()

            # Check for early shots (cheating!)
            if p1_gun or p2_gun:
                cheater = "P1" if p1_gun else "P2"
                other = "P2" if p1_gun else "P1"
                result_text = f"{cheater} shot too early! {other} WINS!"
                winner_side = other
                if other == "P1":
                    p1_score += 1
                else:
                    p2_score += 1
                game_state = "too_early"
                result_show_time = time.time()

            elif countdown_val <= 0:
                # Move to random suspense phase
                suspense_duration = random.uniform(0.5, 2.5)
                suspense_start = time.time()
                game_state = "suspense"
                countdown_val = 0

            draw_game_ui(frame, p1_score, p2_score, "countdown", countdown_val,
                         False, False, "", True, True, 0, "")

        elif game_state == "suspense":
            # The tense waiting period with "..." before BANG
            elapsed = time.time() - suspense_start

            # Check for early shots
            if p1_gun or p2_gun:
                cheater = "P1" if p1_gun else "P2"
                other = "P2" if p1_gun else "P1"
                result_text = f"{cheater} shot too early! {other} WINS!"
                winner_side = other
                if other == "P1":
                    p1_score += 1
                else:
                    p2_score += 1
                game_state = "too_early"
                result_show_time = time.time()
            elif elapsed >= suspense_duration:
                # BANG!
                game_state = "bang"
                bang_time = time.time()
                p1_has_gun = False
                p2_has_gun = False
                p1_shot_time = 0
                p2_shot_time = 0
                flash_timer = 0.5
                bang_played = False

            draw_game_ui(frame, p1_score, p2_score, "countdown", 0,
                         False, False, "", True, True, 0, "")

        elif game_state == "bang":
            elapsed = time.time() - bang_time
            flash_timer = max(0, 0.5 - elapsed)

            # Play bang sound once
            if not bang_played:
                bang_played = True
                threading.Thread(target=play_bang_sound, daemon=True).start()

            # Detect shots
            if p1_gun and not p1_has_gun:
                p1_has_gun = True
                p1_shot_time = time.time()
            if p2_gun and not p2_has_gun:
                p2_has_gun = True
                p2_shot_time = time.time()

            # Check for winner conditions
            both_shot = p1_has_gun and p2_has_gun
            timeout = elapsed >= BANG_WINDOW

            if both_shot or timeout:
                # Determine winner
                if p1_has_gun and p2_has_gun:
                    # Both shot - faster wins
                    if abs(p1_shot_time - p2_shot_time) < 0.05:
                        result_text = "DRAW! Both fired at the same time!"
                        winner_side = "DRAW"
                    elif p1_shot_time < p2_shot_time:
                        diff = p2_shot_time - p1_shot_time
                        result_text = f"P1 WINS! (faster by {diff:.2f}s)"
                        winner_side = "P1"
                        p1_score += 1
                    else:
                        diff = p1_shot_time - p2_shot_time
                        result_text = f"P2 WINS! (faster by {diff:.2f}s)"
                        winner_side = "P2"
                        p2_score += 1
                elif p1_has_gun:
                    react = p1_shot_time - bang_time
                    result_text = f"P1 WINS! (reaction: {react:.2f}s)"
                    winner_side = "P1"
                    p1_score += 1
                elif p2_has_gun:
                    react = p2_shot_time - bang_time
                    result_text = f"P2 WINS! (reaction: {react:.2f}s)"
                    winner_side = "P2"
                    p2_score += 1
                else:
                    result_text = "DRAW! Nobody fired!"
                    winner_side = "DRAW"

                game_state = "result"
                result_show_time = time.time()

            draw_game_ui(frame, p1_score, p2_score, "bang", 0,
                         p1_has_gun, p2_has_gun, "", True, True,
                         flash_timer, "")

        elif game_state == "too_early":
            if time.time() - result_show_time > RESULT_DISPLAY:
                game_state = "title"
            draw_game_ui(frame, p1_score, p2_score, "too_early", 0,
                         p1_has_gun, p2_has_gun, result_text, False, False,
                         0, winner_side)

        elif game_state == "result":
            if time.time() - result_show_time > RESULT_DISPLAY:
                game_state = "title"
            draw_game_ui(frame, p1_score, p2_score, "result", 0,
                         p1_has_gun, p2_has_gun, result_text, False, False,
                         0, winner_side)

        # Show frame
        cv2.imshow("Quick Draw Duel", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and game_state in ("title", "result", "too_early"):
            game_state = "get_ready"
            p1_has_gun = False
            p2_has_gun = False
            result_text = ""
            winner_side = ""
            bang_played = False
        elif key == ord('r'):
            p1_score = 0
            p2_score = 0
            game_state = "title"
            result_text = ""

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
