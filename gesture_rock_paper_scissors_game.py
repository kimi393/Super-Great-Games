import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import random
import time

# ── Hand skeleton connections (21 landmarks) ──
# Each tuple is (start_landmark_index, end_landmark_index)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm
]

# Colors for drawing (BGR)
LANDMARK_COLOR = (0, 255, 0)    # Green dots
CONNECTION_COLOR = (255, 0, 0)  # Blue lines
LANDMARK_RADIUS = 5
CONNECTION_THICKNESS = 2

# Game colors (BGR)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GREEN = (0, 220, 0)
COLOR_RED = (0, 0, 220)
COLOR_YELLOW = (0, 255, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_ORANGE = (0, 165, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_BG = (40, 40, 40)


def draw_hand_landmarks(frame, hand_landmarks_list):
    """Draw hand skeleton on the frame."""
    h, w, _ = frame.shape

    for hand_landmarks in hand_landmarks_list:
        # Convert normalised landmarks to pixel coordinates
        points = []
        for lm in hand_landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)
            points.append((px, py))

        # Draw connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx],
                     CONNECTION_COLOR, CONNECTION_THICKNESS)

        # Draw landmark dots on top
        for pt in points:
            cv2.circle(frame, pt, LANDMARK_RADIUS, LANDMARK_COLOR, -1)


def is_finger_stretched(hand_landmarks, finger_tip_idx, finger_pip_idx):
    """
    Check if a finger is stretched out (open) by comparing the
    tip's y-position to the PIP joint's y-position.
    A stretched finger has its tip ABOVE (lower y value) the PIP joint.
    """
    tip = hand_landmarks[finger_tip_idx]
    pip = hand_landmarks[finger_pip_idx]
    # In image coords, y increases downward, so tip.y < pip.y means extended
    if tip.y < pip.y:
        return True
    else:
        return False


def detect_gesture(hand_landmarks):
    """
    Detect rock, paper, or scissors using if-else on finger states.

    Landmark indices:
      Index finger:  tip=8,  pip=6
      Middle finger: tip=12, pip=10
      Ring finger:   tip=16, pip=14
      Pinky:         tip=20, pip=18
    """
    # Check each finger
    index_open = is_finger_stretched(hand_landmarks, 8, 6)
    middle_open = is_finger_stretched(hand_landmarks, 12, 10)
    ring_open = is_finger_stretched(hand_landmarks, 16, 14)
    pinky_open = is_finger_stretched(hand_landmarks, 20, 18)

    # --- Use if-else to determine gesture ---

    # PAPER: all fingers are stretched out
    if index_open and middle_open and ring_open and pinky_open:
        return "PAPER"

    # SCISSORS: only index and middle finger are stretched out
    elif index_open and middle_open and not ring_open and not pinky_open:
        return "SCISSORS"

    # ROCK: all fingers are closed
    elif not index_open and not middle_open and not ring_open and not pinky_open:
        return "ROCK"

    # Unrecognised gesture
    else:
        return None


def determine_winner(p1_choice, p2_choice):
    """Determine the winner of a round. Returns result from P1's perspective."""
    if p1_choice == p2_choice:
        return "DRAW"
    elif p1_choice == "ROCK" and p2_choice == "SCISSORS":
        return "P1 WINS!"
    elif p1_choice == "SCISSORS" and p2_choice == "PAPER":
        return "P1 WINS!"
    elif p1_choice == "PAPER" and p2_choice == "ROCK":
        return "P1 WINS!"
    else:
        return "P2 WINS!"


def get_hand_center_x(hand_landmarks):
    """Get the average x-position of a hand (normalised 0..1)."""
    total_x = 0
    for lm in hand_landmarks:
        total_x += lm.x
    return total_x / len(hand_landmarks)


# ═══════════════════════════════════════════════════════════
#  MENU SCREEN
# ═══════════════════════════════════════════════════════════

def draw_menu(frame, selected):
    """Draw the mode selection menu."""
    h, w, _ = frame.shape

    # Darken the camera feed
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), COLOR_BLACK, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Title
    cv2.putText(frame, "ROCK PAPER SCISSORS", (w // 2 - 250, h // 3 - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, COLOR_CYAN, 3)

    # Mode options
    modes = ["1 - Player vs AI", "2 - Player vs Player"]
    for i, mode_text in enumerate(modes):
        y = h // 2 + i * 60
        if i == selected:
            color = COLOR_YELLOW
            cv2.rectangle(frame, (w // 2 - 200, y - 30),
                          (w // 2 + 200, y + 10), (60, 60, 60), -1)
        else:
            color = COLOR_WHITE

        cv2.putText(frame, mode_text, (w // 2 - 150, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Instructions
    cv2.putText(frame, "Press 1 or 2 to select mode   Q=Quit",
                (w // 2 - 230, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)


# ═══════════════════════════════════════════════════════════
#  PLAYER vs AI  UI
# ═══════════════════════════════════════════════════════════

def draw_pvai_ui(frame, gesture, computer_choice, result, player_score,
                 computer_score, countdown, game_state):
    """Draw the Player-vs-AI game UI overlay."""
    h, w, _ = frame.shape

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), COLOR_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Title
    cv2.putText(frame, "PLAYER vs AI", (w // 2 - 120, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_CYAN, 2)

    # Score bar
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, 60), (w, 110), (30, 30, 30), -1)
    cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, f"YOU: {player_score}", (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_GREEN, 2)
    cv2.putText(frame, f"CPU: {computer_score}", (w - 180, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_RED, 2)

    # Current gesture display
    if gesture:
        cv2.putText(frame, f"Your hand: {gesture}", (20, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_YELLOW, 2)
    else:
        cv2.putText(frame, "Show: ROCK / PAPER / SCISSORS", (20, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)

    # Game state messages
    if game_state == "countdown":
        cv2.putText(frame, f"Get ready... {countdown}", (w // 2 - 140, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_YELLOW, 3)

    elif game_state == "result":
        cv2.putText(frame, f"CPU chose: {computer_choice}",
                    (w // 2 - 140, h // 2 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_ORANGE, 2)

        if result == "P1 WINS!":
            color = COLOR_GREEN
            text = "YOU WIN!"
        elif result == "P2 WINS!":
            color = COLOR_RED
            text = "YOU LOSE!"
        else:
            color = COLOR_YELLOW
            text = "DRAW!"

        cv2.putText(frame, text, (w // 2 - 100, h // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    elif game_state == "waiting":
        cv2.putText(frame, "Press SPACE to play!", (w // 2 - 160, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_WHITE, 2)

    # Instructions at the bottom
    cv2.putText(frame, "SPACE=Play  R=Reset  M=Menu  Q=Quit", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)


# ═══════════════════════════════════════════════════════════
#  PLAYER vs PLAYER  UI
# ═══════════════════════════════════════════════════════════

def draw_pvp_ui(frame, p1_gesture, p2_gesture, result, p1_score,
                p2_score, countdown, game_state):
    """Draw the Player-vs-Player game UI overlay."""
    h, w, _ = frame.shape

    # Dividing line down the centre
    cv2.line(frame, (w // 2, 115), (w // 2, h - 60), (100, 100, 100), 2)

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), COLOR_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Title
    cv2.putText(frame, "PLAYER vs PLAYER", (w // 2 - 155, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_CYAN, 2)

    # Score bar
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, 60), (w, 110), (30, 30, 30), -1)
    cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, f"P1: {p1_score}", (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_GREEN, 2)
    cv2.putText(frame, f"P2: {p2_score}", (w - 140, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_MAGENTA, 2)

    # Player labels
    cv2.putText(frame, "<-- P1", (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GREEN, 2)
    cv2.putText(frame, "P2 -->", (w - 120, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_MAGENTA, 2)

    # Gesture displays at bottom
    if p1_gesture:
        cv2.putText(frame, f"P1: {p1_gesture}", (20, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GREEN, 2)
    else:
        cv2.putText(frame, "P1: ???", (20, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

    if p2_gesture:
        cv2.putText(frame, f"P2: {p2_gesture}", (w - 220, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_MAGENTA, 2)
    else:
        cv2.putText(frame, "P2: ???", (w - 220, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

    # Game state messages
    if game_state == "countdown":
        cv2.putText(frame, f"Get ready... {countdown}", (w // 2 - 140, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_YELLOW, 3)

    elif game_state == "result":
        if result == "P1 WINS!":
            color = COLOR_GREEN
        elif result == "P2 WINS!":
            color = COLOR_MAGENTA
        else:
            color = COLOR_YELLOW

        cv2.putText(frame, result, (w // 2 - 100, h // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    elif game_state == "waiting":
        cv2.putText(frame, "Press SPACE to play!", (w // 2 - 160, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_WHITE, 2)

    # Instructions
    cv2.putText(frame, "SPACE=Play  R=Reset  M=Menu  Q=Quit", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    # Path to the downloaded model
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

    # Configure the hand landmarker (VIDEO mode = synchronous, per-frame)
    # Start with 2 hands to support both modes
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    landmarker = vision.HandLandmarker.create_from_options(options)

    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("=== ROCK PAPER SCISSORS ===")
    print("Modes:")
    print("  1 - Player vs AI")
    print("  2 - Player vs Player (2 hands)")
    print()
    print("Controls:")
    print("  SPACE - Start a round")
    print("  R     - Reset scores")
    print("  M     - Back to menu")
    print("  Q     - Quit")
    print()
    print("Gestures:")
    print("  ROCK     = All fingers closed (fist)")
    print("  PAPER    = All fingers stretched out")
    print("  SCISSORS = Only index + middle finger out")

    frame_timestamp_ms = 0

    # App state
    app_mode = "menu"  # "menu", "pvai", "pvp"
    menu_selected = 0

    # Game state (shared by both modes)
    p1_score = 0
    p2_score = 0
    game_state = "waiting"  # "waiting", "countdown", "result"
    countdown_start = 0
    countdown_value = 3
    p1_gesture = None
    p2_gesture = None
    p2_choice = None  # used for CPU choice in pvai mode
    result_text = None
    result_show_time = 0

    choices = ["ROCK", "PAPER", "SCISSORS"]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Flip horizontally for a mirror view
        #frame = cv2.flip(frame, 1)
        #NO YOU!
        # Convert BGR → RGB and wrap in a MediaPipe Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Detect hand landmarks (synchronous in VIDEO mode)
        frame_timestamp_ms += 33  # ~30 fps
        detection_result = landmarker.detect_for_video(mp_image,
                                                       frame_timestamp_ms)

        # Draw the skeleton if hands are detected
        if detection_result.hand_landmarks:
            draw_hand_landmarks(frame, detection_result.hand_landmarks)

        # ══════════════════════════════════════════════
        #  MENU MODE
        # ══════════════════════════════════════════════
        if app_mode == "menu":
            draw_menu(frame, menu_selected)

            cv2.imshow("Rock Paper Scissors", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('1'):
                app_mode = "pvai"
                game_state = "waiting"
                p1_score = 0
                p2_score = 0
            elif key == ord('2'):
                app_mode = "pvp"
                game_state = "waiting"
                p1_score = 0
                p2_score = 0
            continue

        # ══════════════════════════════════════════════
        #  PLAYER vs AI MODE
        # ══════════════════════════════════════════════
        if app_mode == "pvai":
            # Detect gesture from the first hand found
            if detection_result.hand_landmarks:
                p1_gesture = detect_gesture(detection_result.hand_landmarks[0])
            else:
                p1_gesture = None

            # ── Game state machine ──
            if game_state == "countdown":
                elapsed = time.time() - countdown_start
                countdown_value = 3 - int(elapsed)

                if countdown_value <= 0:
                    # Time's up — capture gesture
                    if p1_gesture is not None:
                        p2_choice = random.choice(choices)
                        result_text = determine_winner(p1_gesture, p2_choice)

                        if result_text == "P1 WINS!":
                            p1_score += 1
                        elif result_text == "P2 WINS!":
                            p2_score += 1

                        game_state = "result"
                        result_show_time = time.time()
                    else:
                        game_state = "waiting"

            elif game_state == "result":
                if time.time() - result_show_time > 3.0:
                    game_state = "waiting"
                    p2_choice = None
                    result_text = None

            # Draw UI
            draw_pvai_ui(frame, p1_gesture, p2_choice, result_text,
                         p1_score, p2_score, countdown_value, game_state)

        # ══════════════════════════════════════════════
        #  PLAYER vs PLAYER MODE
        # ══════════════════════════════════════════════
        elif app_mode == "pvp":
            p1_gesture = None
            p2_gesture = None

            if detection_result.hand_landmarks:
                h_frame, w_frame, _ = frame.shape

                # Sort hands by x-position: left side = P1, right side = P2
                hands_with_x = []
                for hand_lm in detection_result.hand_landmarks:
                    cx = get_hand_center_x(hand_lm)
                    hands_with_x.append((cx, hand_lm))

                # Sort by x position (left to right)
                hands_with_x.sort(key=lambda item: item[0])

                if len(hands_with_x) >= 1:
                    # Left-most hand = Player 1
                    p1_gesture = detect_gesture(hands_with_x[0][1])

                if len(hands_with_x) >= 2:
                    # Right-most hand = Player 2
                    p2_gesture = detect_gesture(hands_with_x[-1][1])

            # ── Game state machine ──
            if game_state == "countdown":
                elapsed = time.time() - countdown_start
                countdown_value = 3 - int(elapsed)

                if countdown_value <= 0:
                    # Both players need a valid gesture
                    if p1_gesture is not None and p2_gesture is not None:
                        result_text = determine_winner(p1_gesture, p2_gesture)

                        if result_text == "P1 WINS!":
                            p1_score += 1
                        elif result_text == "P2 WINS!":
                            p2_score += 1

                        game_state = "result"
                        result_show_time = time.time()
                    else:
                        # Not enough hands / gestures — retry
                        game_state = "waiting"

            elif game_state == "result":
                if time.time() - result_show_time > 3.0:
                    game_state = "waiting"
                    result_text = None

            # Draw UI
            draw_pvp_ui(frame, p1_gesture, p2_gesture, result_text,
                        p1_score, p2_score, countdown_value, game_state)

        # ── Show frame and handle keys ──
        cv2.imshow("Rock Paper Scissors", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and game_state == "waiting":
            game_state = "countdown"
            countdown_start = time.time()
            countdown_value = 3
        elif key == ord('r'):
            p1_score = 0
            p2_score = 0
            game_state = "waiting"
            p2_choice = None
            result_text = None
        elif key == ord('m'):
            # Back to menu
            app_mode = "menu"
            game_state = "waiting"
            p1_score = 0
            p2_score = 0
            p2_choice = None
            result_text = None

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
