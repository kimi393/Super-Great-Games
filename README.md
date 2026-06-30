# AI Hand Gesture & Computer Vision Workspace

Welcome to the AI Hand Gesture & Computer Vision workspace. This repository contains various interactive Python games and diagnostics powered by **Mediapipe** (hand gesture recognition), **YOLOv11** (object detection/tracking), **Keras/TensorFlow** (digit classification), and **Whisper** (speech-to-text), along with supporting assets and models.

---

## 🎮 Interactive Games

This repository includes several computer vision and audio-controlled games:

### Mediapipe Gesture Games
*   **[gesture_mosquito_swatter_game.py](file:///Users/kimi/Desktop/j/%20ai%20hand%20whiteing%20boaro/gesture_mosquito_swatter_game.py)**  
    A webcam-based game where players use their hands to swat flying mosquitoes on the screen. It tracks hands using the `hand_landmarker.task` model.
*   **[gesture_rock_paper_scissors_game.py](file:///Users/kimi/Desktop/j/%20ai%20hand%20whiteing%20boaro/gesture_rock_paper_scissors_game.py)**  
    A gesture-controlled Rock-Paper-Scissors game played against either an AI or a second player in real-time via webcam.
*   **[gesture_shooting_duel_game.py](file:///Users/kimi/Desktop/j/%20ai%20hand%20whiteing%20boaro/gesture_shooting_duel_game.py)**  
    A quick-draw shooting duel game where two players face the camera (P1 on the left, P2 on the right) and trigger shooting gestures.

### YOLO Toy Car Games
*   **[yolo_toy_car_distance_game.py](file:///Users/kimi/Desktop/j/%20ai%20hand%20whiteing%20boaro/yolo_toy_car_distance_game.py)**  
    An interactive game where players select 3 toy cars each, and the computer selects a target spot on the webcam feed. Players place their physical toy cars closest to the target, and the custom YOLO model measures the distance.
*   **[yolo_toy_car_tracker.py](file:///Users/kimi/Desktop/j/%20ai%20hand%20whiteing%20boaro/yolo_toy_car_tracker.py)**  
    Tracks the positions of toy cars and vehicles in real-time with a custom sci-fi HUD overlay, fading trailing paths, and logs tracking statistics.

### Voice & Pitch-Controlled Games
*   **[pitch_controlled_parkour_game.py](file:///Users/kimi/Desktop/j/%20ai%20hand%20whiteing%20boaro/pitch_controlled_parkour_game.py)**  
    A Pygame parkour platformer controlled by vocal pitch (sing/hum high pitch to JUMP, normal pitch to WALK, silence to STOP).
*   **[pitch_controlled_parkour_game_tuned.py](file:///Users/kimi/Desktop/j/%20ai%20hand%20whiteing%20boaro/pitch_controlled_parkour_game_tuned.py)**  
    An alternative version of the vocal pitch game featuring customized thresholds tuned for different voice ranges.

### MNIST Digit Drawing Games
*   **[two_player_drawing_math_game.py](file:///Users/kimi/Desktop/j/%20ai%20hand%20whiteing%20boaro/two_player_drawing_math_game.py)**  
    A 2-player match where players solve random math problems and draw the answers (0-9) on 8x8 grids under a timer. It uses `trained_mnist_model.h5` to predict inputs.
*   **[mnist_canvas_drawing.py](file:///Users/kimi/Desktop/j/%20ai%20hand%20whiteing%20boaro/mnist_canvas_drawing.py)**  
    A smooth freehand Pygame drawing canvas. Predicts the drawn digit using `trained_mnist_model.h5` and displays classification confidence bars.
*   **[mnist_grid_drawing.py](file:///Users/kimi/Desktop/j/%20ai%20hand%20whiteing%20boaro/mnist_grid_drawing.py)**  
    An 8x8 grid drawing interface where pixels are set to black or white, which is resized and classified by the MNIST neural network.
*   **[mnist_pixel_averaging.py](file:///Users/kimi/Desktop/j/%20ai%20hand%20whiteing%20boaro/mnist_pixel_averaging.py)**  
    A drawing canvas that classifies digits without a neural network, instead using the mathematical distance to the average pixel maps of each digit.

---

## 🧠 Machine Learning Tools & Diagnostics

Scripts for training models, viewing predictions, and diagnosing accuracy:

*   **[train_mnist_model.py](file:///Users/kimi/Desktop/j/%20ai%20hand%20whiteing%20boaro/train_mnist_model.py)**  
    Downloads the MNIST dataset, builds a Sequential neural network (dense + dropout layers), trains it for 10 epochs, and exports the model as `trained_mnist_model.h5`.
*   **[train_yolo_model.py](file:///Users/kimi/Desktop/j/%20ai%20hand%20whiteing%20boaro/train_yolo_model.py)**  
    Converts COCO format labels to YOLO format, splits datasets, builds `dataset.yaml`, and trains a custom YOLOv11 model.
*   **[yolo_confidence_diagnostic.py](file:///Users/kimi/Desktop/j/%20ai%20hand%20whiteing%20boaro/yolo_confidence_diagnostic.py)**  
    A diagnostic script testing custom model predictions across confidence thresholds (0.25, 0.10, 0.05, 0.01) to identify optimal detection setups.
*   **[yolo_prediction_viewer.py](file:///Users/kimi/Desktop/j/%20ai%20hand%20whiteing%20boaro/yolo_prediction_viewer.py)**  
    Displays side-by-side Ground Truth annotations vs. YOLO11 predictions for validation images using Matplotlib.
*   **[mnist_interactive_evaluator.py](file:///Users/kimi/Desktop/j/%20ai%20hand%20whiteing%20boaro/mnist_interactive_evaluator.py)**  
    Allows interactive exploration of MNIST test samples using arrow keys, printing actual vs predicted labels and plotting confidence bars.
*   **[mnist_average_digits_generator.py](file:///Users/kimi/Desktop/j/%20ai%20hand%20whiteing%20boaro/mnist_average_digits_generator.py)**  
    Calculates average pixel intensities for digits (0-9) from the training set and writes visualizations to `mnist_digits/`.
*   **[wand_particle_effect.py](file:///Users/kimi/Desktop/j/%20ai%20hand%20whiteing%20boaro/wand_particle_effect.py)**  
    Camera utility demonstrating green/blue/red threshold masking to track a magic wand tip and spawn particle bursts.

---

## 🎙️ Audio Utilities & Speech-to-Text

*   **[audio_record_playback_utility.py](file:///Users/kimi/Desktop/j/%20ai%20hand%20whiteing%20boaro/audio_record_playback_utility.py)**  
    A console utility to list audio output/input devices, record voice into `.wav` files, perform passthrough monitoring, and calibrate input gain.
*   **[live_whisper_transcriber.py](file:///Users/kimi/Desktop/j/%20ai%20hand%20whiteing%20boaro/live_whisper_transcriber.py)**  
    Continuously records microphone input, detects speech activity, and outputs transcription chunks in real time using MLX Whisper.
*   **[transcribe_audio_file.py](file:///Users/kimi/Desktop/j/%20ai%20hand%20whiteing%20boaro/transcribe_audio_file.py)**  
    Offline speech-to-text script that reads, resamples (to 16000Hz), and transcribes a target WAV audio file.
*   **[voice_image_search_pygame.py](file:///Users/kimi/Desktop/j/%20ai%20hand%20whiteing%20boaro/voice_image_search_pygame.py)**  
    Listens to microphone input, transcribes the search query, queries DuckDuckGo for matching images, and displays a 2x2 image grid in a Pygame window.

---

## 📂 Models & Data Directories

| Directory / File | Description |
| :--- | :--- |
| **`hand_landmarker.task`** | MediaPipe model used to track hand joints and skeletons in webcam games. |
| **`trained_mnist_model.h5`** | Trained TensorFlow Keras model weights used to classify drawn digits in Pygame. |
| **`yolo_toy_cars_model.pt`** | Custom-trained YOLOv11 model weights used to detect select toy car models. |
| **`yolov8n.pt` / `yolo11n.pt`** | Base pre-trained models from Ultralytics used as training start points. |
| **`yolo_dataset/`** | Contains split images and text file coordinates used to train the YOLO detector. |
| **`train/`** | Training source containing original images and raw COCO JSON annotation coordinates. |
| **`mnist_digits/`** | Destination folder where mean digit templates are saved as images. |
| **`toy_car_replays/`** | Folder where video files (.mp4) of game rounds are written. |

---

## 🎨 Asset Files

*   **`background_music.mp3`**: Loopable background music (originally `1-03.mp3`).
*   **`ding_sound_effect.mp3`**: Triggered when a coin is successfully collected.
*   **`target_coin.png`**: Coin overlay graphics placed over targets (originally `coin.png`).
*   **`ha_ha_victory.png`**: Cartoon overlay image (originally `ha_ha!.png`).
*   **`recorded_music_sample_1.wav`** & **`recorded_music_sample_2.wav`**: Hi-fi music recordings.

---

## 🛠️ Configuration & System Files

*   **`requirements.txt`**: Declares required packages (`tensorflow`, `opencv-python`, `pygame`, etc.).
*   **`dataset.yaml`**: Coordinates dataset paths and labels for YOLOv11 training.
*   **`.gitignore`**: Excludes training checkpoints (`runs/`), local environments (`venv/`), and video replays from Git.
