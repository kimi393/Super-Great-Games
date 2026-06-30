"""
Voice-to-Image Search — Optimized for Mac (Apple Silicon)
=========================================================

Uses Whisper for speech-to-text and DuckDuckGo for image searching.
Displays results in a Pygame window.

Usage:
    python voice_image_search_pygame.py
"""

import argparse
import sys
import time
import threading
import queue
import numpy as np
import pygame
import requests
import io
from PIL import Image
from ddgs import DDGS
import os
import random
# ── Audio Settings ──────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000
CHANNELS = 1

# ── Pygame Settings ─────────────────────────────────────────────────────────
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 800
IMAGE_COUNT = 4
GRID_COLS = 2
GRID_ROWS = 2
CELL_WIDTH = WINDOW_WIDTH // GRID_COLS
CELL_HEIGHT = (WINDOW_HEIGHT - 100) // GRID_ROWS

# ── Colours ──────────────────────────────────────────────────────────────────
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (50, 50, 50)
LIGHT_GRAY = (200, 200, 200)
ACCENT = (0, 150, 255)

# ─────────────────────────────────────────────────────────────────────────────
# Whisper Backend (from live_whisper_transcriber.py)
# ─────────────────────────────────────────────────────────────────────────────
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

def _patch_hf_hub_no_auth():
    try:
        import huggingface_hub as hf
        import huggingface_hub._snapshot_download as _snap
        try:
            import huggingface_hub.utils._auth as _auth
            _auth.get_token = lambda: None
        except: pass
        _original = _snap.snapshot_download
        def _patched(*args, **kwargs):
            kwargs["token"] = False
            return _original(*args, **kwargs)
        _snap.snapshot_download = _patched
        hf.snapshot_download = _patched
        try:
            import mlx_whisper.load_models as lm
            lm.snapshot_download = _patched
        except: pass
    except ImportError: pass

_patch_hf_hub_no_auth()

class MLXWhisperBackend:
    MODEL_MAP = {
        "tiny": "mlx-community/whisper-tiny-mlx",
        "small": "mlx-community/whisper-small-mlx",
        "turbo": "mlx-community/whisper-large-v3-turbo",
    }
    def __init__(self, model_name: str):
        import mlx_whisper
        self._mlx = mlx_whisper
        self.repo = self.MODEL_MAP.get(model_name, model_name)
        print(f"Loading MLX model: {self.repo}...")
        warmup = np.zeros(SAMPLE_RATE, dtype=np.float32)
        self._mlx.transcribe(warmup, path_or_hf_repo=self.repo)

    def transcribe(self, audio: np.ndarray) -> str:
        result = self._mlx.transcribe(audio, path_or_hf_repo=self.repo)
        return result.get("text", "").strip()

class FasterWhisperBackend:
    def __init__(self, model_name: str):
        from faster_whisper import WhisperModel
        print(f"Loading faster-whisper model: {model_name}...")
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(audio, beam_size=1)
        return " ".join(seg.text.strip() for seg in segments).strip()

def get_backend(model_name: str):
    try:
        return MLXWhisperBackend(model_name)
    except ImportError:
        try:
            return FasterWhisperBackend(model_name)
        except ImportError:
            print("ERROR: No Whisper backend found.")
            sys.exit(1)

# ── Game Settings ───────────────────────────────────────────────────────────
GAME_TARGETS = [
    {"en": "Red Apple", "zh": "紅蘋果", "keywords": ["apple", "蘋果"]},
    {"en": "Yellow Banana", "zh": "黃色香蕉", "keywords": ["banana", "香蕉"]},
    {"en": "Blue Car", "zh": "藍色的車", "keywords": ["car", "車", "汽車"]},
    {"en": "Green Frog", "zh": "綠色的青蛙", "keywords": ["frog", "青蛙"]},
    {"en": "Purple Dragon", "zh": "紫色恐龍", "keywords": ["dragon", "恐龍", "dinosaur"]},
    {"en": "Pikachu", "zh": "皮卡丘", "keywords": ["pikachu", "皮卡丘"]},
    {"en": "Golden Crown", "zh": "金皇冠", "keywords": ["crown", "皇冠"]},
    {"en": "Pizza", "zh": "披薩", "keywords": ["pizza", "披薩"]},
]
ROUND_TIME = 25 # Seconds

# ── Image Management ──────────────────────────────────────────────────────────
class ImageManager:
    def __init__(self):
        self.images = []   # List of pygame surfaces
        self.image_titles = [] # To check for game victory
        self.lock = threading.Lock()
        self.current_query = ""
        self.is_searching = False

    def update_query(self, query):
        if not query or query == self.current_query:
            return
        self.current_query = query
        with self.lock:
            self.is_searching = True
        threading.Thread(target=self._search_images, args=(query,), daemon=True).start()

    def _search_images(self, query):
        print(f"Searching for: {query}")
        try:
            with DDGS() as ddgs:
                results = list(ddgs.images(query, max_results=IMAGE_COUNT))
            
            new_surfaces = []
            titles = []
            for r in results:
                url = r.get('image')
                title = r.get('title', '').lower()
                if url:
                    try:
                        resp = requests.get(url, timeout=5)
                        img = Image.open(io.BytesIO(resp.content))
                        img.thumbnail((CELL_WIDTH - 20, CELL_HEIGHT - 20))
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        data = img.tobytes()
                        size = img.size
                        surface = pygame.image.fromstring(data, size, 'RGB')
                        new_surfaces.append(surface)
                        titles.append(title)
                        if len(new_surfaces) >= IMAGE_COUNT:
                            break
                    except Exception as e:
                        print(f"Err downloading {url}: {e}")
            
            with self.lock:
                self.images = new_surfaces
                self.image_titles = titles
                self.is_searching = False
        except Exception as e:
            print(f"Search error: {e}")
            with self.lock:
                self.is_searching = False

# ── Main Application ──────────────────────────────────────────────────────────
class VoiceImageApp:
    def __init__(self, backend):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Voice Scavenger Hunt!")
        self.clock = pygame.time.Clock()
        
        # Select a font that supports Chinese characters
        chinese_fonts = ["pingfanghk", "heititc", "stheitilight", "arielunicodems", "stheititc"]
        found_font = None
        for f in chinese_fonts:
            if pygame.font.match_font(f):
                found_font = pygame.font.match_font(f)
                break
        
        if found_font:
            self.font_large = pygame.font.Font(found_font, 40)
            self.font_mid = pygame.font.Font(found_font, 32)
            self.font_small = pygame.font.Font(found_font, 24)
        else:
            self.font_large = pygame.font.SysFont("Arial", 40, bold=True)
            self.font_mid = pygame.font.SysFont("Arial", 32)
            self.font_small = pygame.font.SysFont("Arial", 24)
        
        self.backend = backend
        self.image_mgr = ImageManager()
        self.transcription = ""
        self.audio_buffer = []
        self.is_recording = False
        self.last_status = "Ready"
        self.lock = threading.Lock()

        # Game State
        self.score = 0
        self.current_target = random.choice(GAME_TARGETS)
        self.time_left = ROUND_TIME
        self.game_state = "PLAYING" # PLAYING, WIN, GAMEOVER
        self.last_tick = time.time()
        
    def _audio_callback(self, indata, frames, time_info, status):
        with self.lock:
            if self.is_recording:
                self.audio_buffer.append(indata[:, 0].copy())

    def toggle_recording(self):
        if self.game_state != "PLAYING":
            self.reset_round()
            return

        with self.lock:
            if not self.is_recording:
                self.audio_buffer = []
                self.is_recording = True
                self.last_status = "Recording..."
            else:
                self.is_recording = False
                self.last_status = "Transcribing..."
                threading.Thread(target=self._process_recorded_audio, daemon=True).start()

    def _process_recorded_audio(self):
        with self.lock:
            if not self.audio_buffer:
                self.last_status = "No audio"
                return
            audio = np.concatenate(self.audio_buffer)
            self.audio_buffer = []

        text = self.backend.transcribe(audio)
        if text:
            self.transcription = text
            self.image_mgr.update_query(text)
            self.last_status = "Done"
            # Give short delay to let images download before check
        else:
            self.last_status = "Try again"

    def reset_round(self):
        self.score = 0 if self.game_state == "GAMEOVER" else self.score
        self.current_target = random.choice(GAME_TARGETS)
        self.time_left = ROUND_TIME
        self.game_state = "PLAYING"
        self.transcription = ""
        with self.image_mgr.lock:
            self.image_mgr.images = []
            self.image_mgr.image_titles = []

    def check_victory(self):
        with self.image_mgr.lock:
            for title in self.image_mgr.image_titles:
                print(title )
                for kw in self.current_target["keywords"]:
                    if kw.lower() in title or kw.lower() in self.transcription.lower():
                        return True
        return False

    def run(self):
        import sounddevice as sd
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=self._audio_callback)
        stream.start()
        
        running = True
        while running:
            # Update Timer
            now = time.time()
            dt = now - self.last_tick
            self.last_tick = now
            
            if self.game_state == "PLAYING":
                self.time_left -= dt
                if self.time_left <= 0:
                    self.game_state = "GAMEOVER"
                
                # Check for victory automatically when results come in
                if self.check_victory():
                    self.game_state = "WIN"
                    self.score += 1

            self.screen.fill(WHITE)
            
            # --- UI HEADER ---
            pygame.draw.rect(self.screen, ACCENT, (0, 0, WINDOW_WIDTH, 120))
            
            if self.game_state == "PLAYING":
                # Show Target
                target_text = f"任務: 搜尋 『{self.current_target['zh']}』 ({self.current_target['en']})"
                target_surf = self.font_large.render(target_text, True, WHITE)
                self.screen.blit(target_surf, (20, 20))
                
                # Show Timer and Score
                timer_color = (255, 100, 100) if self.time_left < 5 else WHITE
                timer_surf = self.font_mid.render(f"時間: {int(self.time_left)}s", True, timer_color)
                self.screen.blit(timer_surf, (WINDOW_WIDTH - 180, 20))
                
                score_surf = self.font_mid.render(f"得分: {self.score}", True, WHITE)
                self.screen.blit(score_surf, (WINDOW_WIDTH - 180, 60))

                # Show Speech
                speech_text = f"你說了: {self.transcription}" if self.transcription else "按 [空白鍵] 開始講話..."
                speech_surf = self.font_small.render(speech_text, True, LIGHT_GRAY)
                self.screen.blit(speech_surf, (20, 75))

            elif self.game_state == "WIN":
                win_surf = self.font_large.render("★ 太強了! 成功找到! ★", True, (50, 200, 50))
                self.screen.blit(win_surf, (WINDOW_WIDTH // 2 - win_surf.get_width() // 2, 30))
                inst_surf = self.font_small.render("按 [空白鍵] 下一個任務", True, WHITE)
                self.screen.blit(inst_surf, (WINDOW_WIDTH // 2 - inst_surf.get_width() // 2, 80))

            elif self.game_state == "GAMEOVER":
                self.screen.fill((50, 0, 0))
                lose_surf = self.font_large.render("時間到! 遊戲結束", True, (255, 50, 50))
                self.screen.blit(lose_surf, (WINDOW_WIDTH // 2 - lose_surf.get_width() // 2, WINDOW_HEIGHT // 2 - 50))
                final_score = self.font_mid.render(f"最終得分: {self.score}", True, WHITE)
                self.screen.blit(final_score, (WINDOW_WIDTH // 2 - final_score.get_width() // 2, WINDOW_HEIGHT // 2 + 20))
                inst_surf = self.font_small.render("按 [空白鍵] 重新開始", True, LIGHT_GRAY)
                self.screen.blit(inst_surf, (WINDOW_WIDTH // 2 - inst_surf.get_width() // 2, WINDOW_HEIGHT // 2 + 80))

            # --- IMAGES ---
            if self.game_state != "GAMEOVER":
                with self.image_mgr.lock:
                    for idx, img in enumerate(self.image_mgr.images):
                        col, row = idx % GRID_COLS, idx // GRID_COLS
                        x, y = col * CELL_WIDTH + 10, row * CELL_HEIGHT + 130
                        inner_x = x + (CELL_WIDTH - 20 - img.get_width()) // 2
                        inner_y = y + (CELL_HEIGHT - 20 - img.get_height()) // 2
                        pygame.draw.rect(self.screen, LIGHT_GRAY, (x, y, CELL_WIDTH - 20, CELL_HEIGHT - 20))
                        self.screen.blit(img, (inner_x, inner_y))

            # --- RECORDING INDICATOR ---
            if self.is_recording:
                pygame.draw.circle(self.screen, (255, 0, 0), (WINDOW_WIDTH - 210, 100), 10)
                rec_surf = self.font_small.render("錄音中...", True, (255, 0, 0))
                self.screen.blit(rec_surf, (WINDOW_WIDTH - 190, 88))

            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.toggle_recording()
            
            pygame.display.flip()
            self.clock.tick(30)
            
        stream.stop(); stream.close(); pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small", help="Whisper model size")
    args = parser.parse_args()
    
    backend = get_backend(args.model)
    app = VoiceImageApp(backend)
    app.run()
