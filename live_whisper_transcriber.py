"""
Live Audio Transcription with Whisper — Optimized for Mac (Apple Silicon)
=========================================================================

Uses mlx-whisper (Apple MLX framework) for GPU/Neural Engine acceleration
on Apple Silicon Macs. Falls back to faster-whisper on Intel Macs or if
mlx-whisper is not installed.

Install dependencies:
    brew install portaudio          # required by sounddevice on macOS
    pip install sounddevice numpy mlx-whisper

For Intel Mac fallback:
    pip install faster-whisper

Usage:
    python live_whisper_transcriber.py                       # default: small model, English
    python live_whisper_transcriber.py --model tiny          # faster, less accurate
    python live_whisper_transcriber.py --model medium        # slower, more accurate
    python live_whisper_transcriber.py --language ja         # Japanese
    python live_whisper_transcriber.py --chunk-sec 3         # 3-second chunks
    python live_whisper_transcriber.py --energy-threshold 0.02  # adjust silence threshold
"""

import argparse
import sys
import time
import threading
import queue
import numpy as np

# ── Audio Settings ──────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000  # Whisper expects 16 kHz mono
CHANNELS = 1

# ── Colours for terminal output ─────────────────────────────────────────────
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


# ─────────────────────────────────────────────────────────────────────────────
# Patch HuggingFace Hub to download WITHOUT authentication
# ─────────────────────────────────────────────────────────────────────────────
import os
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def _patch_hf_hub_no_auth():
    """
    Force huggingface_hub to make all API calls anonymously (token=False).
    This lets us download public models (like mlx-community/*) without
    needing `huggingface-cli login`.

    We patch at multiple levels to be thorough:
    1. Environment variable HF_HUB_DISABLE_IMPLICIT_TOKEN
    2. The built-in token resolution function
    3. snapshot_download kwargs
    4. mlx_whisper's reference to snapshot_download
    """
    try:
        import huggingface_hub as hf
        import huggingface_hub._snapshot_download as _snap

        # ── Patch 1: Override the token resolution to always return None ──
        try:
            import huggingface_hub.utils._auth as _auth
            _auth.get_token = lambda: None
        except (ImportError, AttributeError):
            pass

        # ── Patch 2: Wrap snapshot_download to force token=False ──
        _original = _snap.snapshot_download

        def _patched(*args, **kwargs):
            kwargs["token"] = False
            return _original(*args, **kwargs)

        _snap.snapshot_download = _patched
        hf.snapshot_download = _patched

        # ── Patch 3: Fix mlx_whisper's cached import reference ──
        try:
            import mlx_whisper.load_models as lm
            lm.snapshot_download = _patched
        except ImportError:
            pass

        print(f"{DIM}  ✓ HuggingFace Hub patched for anonymous download{RESET}")
    except ImportError:
        pass


# Apply the patch immediately on import
_patch_hf_hub_no_auth()


# ─────────────────────────────────────────────────────────────────────────────
# Backend abstraction
# ─────────────────────────────────────────────────────────────────────────────
class MLXWhisperBackend:
    """Apple Silicon optimized backend using mlx-whisper."""

    # Map friendly names → HuggingFace repo IDs (public, no login required)
    MODEL_MAP = {
        "tiny":       "mlx-community/whisper-tiny",
        "tiny.en":    "mlx-community/whisper-tiny.en-mlx",
        "base":       "mlx-community/whisper-base-mlx",
        "base.en":    "mlx-community/whisper-base.en-mlx",
        "small":      "mlx-community/whisper-small-mlx",
        "small.en":   "mlx-community/whisper-small.en-mlx",
        "medium":     "mlx-community/whisper-medium-mlx",
        "medium.en":  "mlx-community/whisper-medium.en-mlx",
        "large":      "mlx-community/whisper-large-v3-mlx",
        "large-v3":   "mlx-community/whisper-large-v3-mlx",
        "turbo":      "mlx-community/whisper-large-v3-turbo",
    }

    def __init__(self, model_name: str, language: str | None = None):
        import mlx_whisper  # noqa: F811
        self._mlx = mlx_whisper
        self.repo = self.MODEL_MAP.get(model_name, model_name)
        self.language = language
        # Warm-up: force model download + first inference
        print(f"{DIM}  Loading MLX model: {self.repo} …{RESET}")
        warmup = np.zeros(SAMPLE_RATE, dtype=np.float32)
        self._mlx.transcribe(warmup, path_or_hf_repo=self.repo)
        print(f"{GREEN}  ✓ MLX model ready{RESET}")

    def transcribe(self, audio: np.ndarray) -> str:
        kwargs = {"path_or_hf_repo": self.repo, "task": "translate"}
        if self.language:
            kwargs["language"] = self.language
        result = self._mlx.transcribe(audio, **kwargs)
        return result.get("text", "").strip()


class FasterWhisperBackend:
    """CPU/GPU backend via CTranslate2 (faster-whisper)."""

    def __init__(self, model_name: str, language: str | None = None):
        from faster_whisper import WhisperModel
        self.language = language
        print(f"{DIM}  Loading faster-whisper model: {model_name} …{RESET}")
        # Use int8 for fast CPU inference on Mac
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")
        print(f"{GREEN}  ✓ faster-whisper model ready{RESET}")

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio,
            language=self.language,
            task="translate",      # translate all languages to English
            beam_size=1,           # greedy for speed
            vad_filter=True,       # built-in VAD
        )
        return " ".join(seg.text.strip() for seg in segments).strip()


def get_backend(model_name: str, language: str | None = None):
    """Try MLX first (Apple Silicon), fall back to faster-whisper."""
    try:
        return MLXWhisperBackend(model_name, language)
    except ImportError:
        print(f"{YELLOW}  mlx-whisper not found, trying faster-whisper …{RESET}")
    try:
        return FasterWhisperBackend(model_name, language)
    except ImportError:
        print(f"{RED}  ERROR: No whisper backend found.{RESET}")
        print(f"  Install one of:")
        print(f"    pip install mlx-whisper      (Apple Silicon)")
        print(f"    pip install faster-whisper   (any Mac)")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Boom sound effect (synthesized — no external files needed)
# ─────────────────────────────────────────────────────────────────────────────
def _generate_boom(duration: float = 1.5, sample_rate: int = 44100) -> np.ndarray:
    """
    Synthesize an explosion/boom sound:
      - Low-frequency sine sweep (80→20 Hz)  for the bass rumble
      - White-noise burst with fast decay     for the crack/impact
      - Exponential amplitude envelope        for natural falloff
    """
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

    # Bass rumble: sine swept from 80 Hz → 20 Hz
    freq = np.linspace(80, 20, len(t))
    phase = np.cumsum(freq / sample_rate) * 2 * np.pi
    bass = np.sin(phase).astype(np.float32)

    # Sub-bass punch at the start
    sub = np.sin(2 * np.pi * 40 * t).astype(np.float32)
    sub *= np.exp(-t * 6).astype(np.float32)

    # Noise burst (white noise, fast decay)
    noise = np.random.randn(len(t)).astype(np.float32)
    noise *= np.exp(-t * 8).astype(np.float32)

    # Combine
    boom = 0.5 * bass + 0.3 * sub + 0.4 * noise

    # Master envelope: sharp attack, exponential decay
    envelope = np.exp(-t * 3).astype(np.float32)
    boom *= envelope

    # Normalize to [-0.9, 0.9] to avoid clipping
    peak = np.max(np.abs(boom))
    if peak > 0:
        boom = boom / peak * 0.9

    return boom


def play_boom():
    """Play the boom sound effect and show ASCII explosion."""
    import sounddevice as sd

    boom_audio = _generate_boom()

    # ASCII explosion frames
    frames = [
        f"""
{RED}              . * .{RESET}
{RED}            *  💥  *{RESET}
{RED}              * . *{RESET}
""",
        f"""
{YELLOW}          .  * ' * .  .{RESET}
{RED}        *    💥💥    *{RESET}
{YELLOW}          .  * . * .  .{RESET}
""",
        f"""
{YELLOW}       . '  *  '  *  ' .{RESET}
{RED}     *  '   💥💥💥   '  *{RESET}
{YELLOW}       . '  *  .  *  ' .{RESET}
{DIM}          ░░░░░░░░░{RESET}
""",
        f"""
{YELLOW}    .  '  *  '  *  '  *  .{RESET}
{RED}  *  '    💥💥💥💥💥    '  *{RESET}
{YELLOW}    .  '  *  .  *  '  *  .{RESET}
{DIM}        ░░▒▒▓▓▓▓▒▒░░{RESET}
{DIM}          ░░░░░░░░{RESET}
""",
        f"""
{DIM}        ░░▒▒▒▒▒▒░░{RESET}
{DIM}          ░░░░░░{RESET}
{DIM}            ░░{RESET}
""",
    ]

    # Play sound in background
    sd.play(boom_audio, samplerate=44100)

    # Animate explosion
    for frame in frames:
        print(frame, end="", flush=True)
        time.sleep(0.2)

    sd.wait()  # wait for audio to finish
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Energy-based Voice Activity Detection
# ─────────────────────────────────────────────────────────────────────────────
def rms_energy(audio: np.ndarray) -> float:
    """Root-mean-square energy of an audio buffer."""
    return float(np.sqrt(np.mean(audio ** 2)))


# ─────────────────────────────────────────────────────────────────────────────
# Core: threaded recorder + transcriber
# ─────────────────────────────────────────────────────────────────────────────
class LiveTranscriber:
    def __init__(self, backend, chunk_sec: float = 2.0, energy_threshold: float = 0.01,
                 overlap_sec: float = 0.5):
        self.backend = backend
        self.chunk_sec = chunk_sec
        self.energy_threshold = energy_threshold
        self.overlap_sec = overlap_sec
        self.audio_q: queue.Queue[np.ndarray | None] = queue.Queue()
        self._stop = threading.Event()
        self._chunk_samples = int(SAMPLE_RATE * chunk_sec)
        self._overlap_samples = int(SAMPLE_RATE * overlap_sec)

    # ── Recording thread ────────────────────────────────────────────────────
    def _record_loop(self):
        """Push fixed-size chunks of microphone audio into the queue."""
        import sounddevice as sd

        buffer = np.empty(0, dtype=np.float32)
        block_size = 1024

        def callback(indata, frames, time_info, status):
            if status:
                print(f"{DIM}  ⚠ {status}{RESET}", file=sys.stderr)
            self.audio_q.put(indata[:, 0].copy())

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                            dtype="float32", blocksize=block_size,
                            callback=callback):
            while not self._stop.is_set():
                self._stop.wait(0.1)

    # ── Transcription thread ────────────────────────────────────────────────
    def _transcribe_loop(self):
        """Consume audio from the queue, accumulate, and transcribe chunks."""
        buffer = np.empty(0, dtype=np.float32)
        segment_count = 0
        prev_overlap = np.empty(0, dtype=np.float32)

        while not self._stop.is_set():
            # Drain the queue into the buffer
            try:
                data = self.audio_q.get(timeout=0.3)
            except queue.Empty:
                continue
            buffer = np.concatenate([buffer, data])

            # Wait until we have a full chunk
            if len(buffer) < self._chunk_samples:
                continue

            chunk = buffer[: self._chunk_samples]
            # Keep overlap for context continuity
            buffer = buffer[self._chunk_samples - self._overlap_samples:]

            # Simple energy-based VAD: skip silence
            energy = rms_energy(chunk)
            if energy < self.energy_threshold:
                continue

            # Prepend overlap from previous chunk for context
            if len(prev_overlap) > 0:
                audio_input = np.concatenate([prev_overlap, chunk])
            else:
                audio_input = chunk
            prev_overlap = chunk[-self._overlap_samples:]

            segment_count += 1
            t0 = time.perf_counter()
            text: str = self.backend.transcribe(audio_input)
            elapsed = time.perf_counter() - t0
            
            
            if text:
                # Filter Whisper hallucinations on near-silence
                hallucinations = {
                    "thank you", "thanks for watching", "you", "the end",
                    "bye", "...", ".", "thank you for watching",
                    "thanks for watching!", "subscribe",
                }
                if text.lower().strip(" .!") in hallucinations:
                    continue

                timestamp = time.strftime("%H:%M:%S")
                print(
                    f"{DIM}[{timestamp}]{RESET} "
                    f"{CYAN}#{segment_count}{RESET} "
                    f"{DIM}({elapsed:.2f}s){RESET}  "
                    f"{BOLD}{text}{RESET}"
                )
                text_lower = text.lower()
                if "explode" in text_lower or "boom" in text_lower or "die" in text_lower:
                    print(f"\n{RED}{BOLD}  ⚠  SELF DESTRUCT ACTIVATED  ⚠{RESET}")
                    for n in ["3", "2", "1"]:
                        print(f"  {BOLD}{n}...{RESET}", flush=True)
                        time.sleep(1)
                    play_boom()
                    import os, signal
                    os.kill(os.getpid(), signal.SIGINT)
                    return
                    
                   
    # ── Public API ──────────────────────────────────────────────────────────
    def run(self):
        print()
        print(f"  {BOLD}🎤  Live Transcription{RESET}")
        print(f"  {DIM}────────────────────────────────────────{RESET}")
        print(f"  {DIM}Chunk length : {self.chunk_sec}s | "
              f"Overlap : {self.overlap_sec}s | "
              f"Energy gate : {self.energy_threshold}{RESET}")
        print(f"  {DIM}Press {BOLD}Ctrl+C{RESET}{DIM} to stop{RESET}")
        print()

        rec_thread = threading.Thread(target=self._record_loop, daemon=True)
        tx_thread = threading.Thread(target=self._transcribe_loop, daemon=True)
        rec_thread.start()
        tx_thread.start()

        try:
            while True:
                time.sleep(0.5)
                if not tx_thread.is_alive():
                    sys.exit(1)
        except KeyboardInterrupt:
            print(f"\n{YELLOW}  ■  Stopping …{RESET}")
            self._stop.set()
            rec_thread.join(timeout=2)
            tx_thread.join(timeout=5)
            print(f"{GREEN}  ✓  Done.{RESET}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Live audio transcription with Whisper, optimised for Mac.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m", default="small",
        help="Whisper model size: tiny, base, small, medium, large, turbo "
             "(default: small)",
    )
    parser.add_argument(
        "--language", "-l", default=None,
        help="Language code (e.g. en, ja, zh). Auto-detect if omitted.",
    )
    parser.add_argument(
        "--chunk-sec", type=float, default=3.0,
        help="Seconds of audio per transcription chunk (default: 3.0).",
    )
    parser.add_argument(
        "--overlap-sec", type=float, default=0.5,
        help="Seconds of overlap between chunks for context (default: 0.5).",
    )
    parser.add_argument(
        "--energy-threshold", type=float, default=0.01,
        help="RMS energy threshold to skip silence (default: 0.01).",
    )
    args = parser.parse_args()

    print(f"\n{BOLD}═══  Whisper Live Transcription (Mac Optimised)  ═══{RESET}")

    backend = get_backend(args.model, args.language)

    transcriber = LiveTranscriber(
        backend=backend,
        chunk_sec=args.chunk_sec,
        energy_threshold=args.energy_threshold,
        overlap_sec=args.overlap_sec,
    )
    transcriber.run()


if __name__ == "__main__":
    main()
