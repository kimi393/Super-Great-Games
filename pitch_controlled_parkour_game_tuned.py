"""
Sound-Controlled Parkour Game
=============================
Controls:
  - High pitch voice/sound  → JUMP
  - Normal pitch voice/sound → WALK (run forward)
  - Silence / no pitch       → STOP

Press ESC or close window to quit.
Press R to restart after game over.
"""

import pygame
import numpy as np
import threading
import sys
import math
import random
import time

# Try to import sounddevice for microphone input
try:
    import sounddevice as sd
except ImportError:
    print("=" * 50)
    print("  Missing 'sounddevice' package!")
    print("  Install it with: pip install sounddevice")
    print("=" * 50)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
FPS = 60

# Audio settings
SAMPLE_RATE = 44100
BLOCK_SIZE = 2048  # samples per audio block

# Pitch thresholds (Hz) — tweak these to match your voice range
PITCH_SILENCE_THRESHOLD = 0.02   # RMS below this = silence
PITCH_HIGH_THRESHOLD = 500       # Hz above this = jump
PITCH_LOW_THRESHOLD = 100        # Hz above this (but below high) = walk

# Physics
GRAVITY = 0.8
JUMP_VELOCITY = -15
PLAYER_SPEED = 5
GROUND_Y = 480

# Colors (rich palette)
BG_TOP = (15, 10, 35)
BG_BOTTOM = (45, 20, 80)
PLATFORM_COLOR = (80, 200, 255)
PLATFORM_GLOW = (40, 120, 200)
PLAYER_COLOR = (255, 100, 180)
PLAYER_OUTLINE = (255, 180, 220)
PARTICLE_COLORS = [
    (255, 100, 100), (100, 255, 150), (100, 180, 255),
    (255, 220, 80), (200, 100, 255), (255, 150, 50),
]
TEXT_COLOR = (240, 240, 255)
ACCENT = (0, 255, 200)
WARNING_COLOR = (255, 80, 80)
HUD_BG = (20, 15, 50, 180)

# Pitch state labels
STATE_STOP = 0
STATE_WALK = 1
STATE_JUMP = 2


# ---------------------------------------------------------------------------
# Pitch Detector (runs in its own thread)
# ---------------------------------------------------------------------------
class PitchDetector:
    """Continuously listens to the microphone and estimates pitch."""

    def __init__(self):
        self.pitch = 0.0          # detected pitch in Hz
        self.rms = 0.0            # volume level
        self.state = STATE_STOP   # current control state
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2)

    def get_state(self):
        with self.lock:
            return self.state, self.pitch, self.rms

    # ---- internal --------------------------------------------------------

    def _detect_pitch(self, signal, sr):
        """Autocorrelation-based pitch detection."""
        # Normalize
        signal = signal - np.mean(signal)
        # Autocorrelation via FFT
        n = len(signal)
        fft = np.fft.rfft(signal, n=2 * n)
        acf = np.fft.irfft(fft * np.conj(fft))[:n]
        acf = acf / acf[0] if acf[0] != 0 else acf

        # Find first dip then first peak
        d = np.diff(acf)
        start = 0
        for i in range(len(d)):
            if d[i] > 0:
                start = i
                break

        if start == 0:
            return 0.0

        peak_idx = start + np.argmax(acf[start:])
        if peak_idx == 0 or acf[peak_idx] < 0.2:
            return 0.0

        return sr / peak_idx

    def _run(self):
        """Audio capture loop — runs in background thread."""
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                channels=1,
                dtype="float32",
            ) as stream:
                while self.running:
                    data, _ = stream.read(BLOCK_SIZE)
                    signal = data[:, 0]
                    rms = float(np.sqrt(np.mean(signal ** 2)))
                    if rms < PITCH_SILENCE_THRESHOLD:
                        pitch = 0.0
                        state = STATE_STOP
                    else:
                        pitch = self._detect_pitch(signal, SAMPLE_RATE)
                        if pitch >= PITCH_HIGH_THRESHOLD:
                            state = STATE_JUMP
                        elif pitch >= PITCH_LOW_THRESHOLD:
                            state = STATE_WALK
                        else:
                            state = STATE_STOP

                    with self.lock:
                        self.pitch = pitch
                        self.rms = rms
                        self.state = state
        except Exception as e:
            print(f"[PitchDetector] Error: {e}")
            self.running = False


# ---------------------------------------------------------------------------
# Particle System (visual flair)
# ---------------------------------------------------------------------------
class Particle:
    def __init__(self, x, y, color=None):
        self.x = x
        self.y = y
        self.vx = random.uniform(-3, 3)
        self.vy = random.uniform(-5, -1)
        self.life = random.randint(15, 40)
        self.max_life = self.life
        self.size = random.uniform(2, 5)
        self.color = color or random.choice(PARTICLE_COLORS)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.15
        self.life -= 1

    def draw(self, surface, cam_x):
        alpha = max(0, self.life / self.max_life)
        r = int(self.size * alpha) + 1
        c = tuple(int(ch * alpha) for ch in self.color)
        pygame.draw.circle(surface, c, (int(self.x - cam_x), int(self.y)), r)


# ---------------------------------------------------------------------------
# Platform
# ---------------------------------------------------------------------------
class Platform:
    def __init__(self, x, y, w, h=18):
        self.rect = pygame.Rect(x, y, w, h)

    def draw(self, surface, cam_x):
        r = self.rect.move(-cam_x, 0)
        # Glow under platform
        glow_rect = pygame.Rect(r.x - 4, r.y + 2, r.width + 8, r.height + 6)
        glow_surf = pygame.Surface((glow_rect.width, glow_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*PLATFORM_GLOW, 60), glow_surf.get_rect(), border_radius=8)
        surface.blit(glow_surf, glow_rect.topleft)
        # Main platform
        pygame.draw.rect(surface, PLATFORM_COLOR, r, border_radius=6)
        # Highlight on top
        highlight = pygame.Rect(r.x + 3, r.y + 2, r.width - 6, 4)
        pygame.draw.rect(surface, (180, 230, 255), highlight, border_radius=2)


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------
class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vy = 0
        self.width = 30
        self.height = 44
        self.on_ground = False
        self.moving = False
        self.jump_requested = False
        self.alive = True
        self.run_frame = 0
        self.run_timer = 0

    @property
    def rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def update(self, platforms, state):
        if not self.alive:
            return

        # Horizontal movement
        self.moving = False
        if state == STATE_WALK or state == STATE_JUMP:
            self.x += PLAYER_SPEED
            self.moving = True

        # Jump
        if state == STATE_JUMP and self.on_ground:
            self.vy = JUMP_VELOCITY
            self.on_ground = False

        # Gravity
        self.vy += GRAVITY
        self.y += self.vy

        # Collision with platforms
        self.on_ground = False
        player_rect = self.rect
        for plat in platforms:
            if player_rect.colliderect(plat.rect):
                # Landing on top
                if self.vy >= 0 and player_rect.bottom >= plat.rect.top and (player_rect.bottom - self.vy) <= plat.rect.top + 10:
                    self.y = plat.rect.top - self.height
                    self.vy = 0
                    self.on_ground = True

        # Fell off screen → dead
        if self.y > SCREEN_HEIGHT + 100:
            self.alive = False

        # Animation
        if self.moving and self.on_ground:
            self.run_timer += 1
            if self.run_timer % 6 == 0:
                self.run_frame = (self.run_frame + 1) % 4

    def draw(self, surface, cam_x):
        sx = int(self.x - cam_x)
        sy = int(self.y)

        # Body
        body_rect = pygame.Rect(sx + 5, sy + 10, 20, 24)
        pygame.draw.rect(surface, PLAYER_COLOR, body_rect, border_radius=6)
        pygame.draw.rect(surface, PLAYER_OUTLINE, body_rect, 2, border_radius=6)

        # Head
        head_center = (sx + 15, sy + 6)
        pygame.draw.circle(surface, PLAYER_COLOR, head_center, 9)
        pygame.draw.circle(surface, PLAYER_OUTLINE, head_center, 9, 2)

        # Eyes
        pygame.draw.circle(surface, (255, 255, 255), (sx + 18, sy + 5), 3)
        pygame.draw.circle(surface, (30, 30, 60), (sx + 19, sy + 5), 1)

        # Legs (animated)
        leg_offsets = [(0, 0), (3, -3), (0, 0), (-3, -3)]
        lo = leg_offsets[self.run_frame] if self.moving else (0, 0)
        # Left leg
        pygame.draw.line(surface, PLAYER_OUTLINE, (sx + 10, sy + 34), (sx + 7 + lo[0], sy + 44 + lo[1]), 3)
        # Right leg
        pygame.draw.line(surface, PLAYER_OUTLINE, (sx + 20, sy + 34), (sx + 23 - lo[0], sy + 44 - lo[1]), 3)


# ---------------------------------------------------------------------------
# Background Stars
# ---------------------------------------------------------------------------
class Star:
    def __init__(self):
        self.x = random.randint(0, SCREEN_WIDTH)
        self.y = random.randint(0, SCREEN_HEIGHT - 150)
        self.brightness = random.randint(80, 255)
        self.speed = random.uniform(0.1, 0.5)
        self.size = random.choice([1, 1, 1, 2])
        self.twinkle_speed = random.uniform(0.02, 0.08)
        self.twinkle_phase = random.uniform(0, math.pi * 2)


# ---------------------------------------------------------------------------
# Level Generator
# ---------------------------------------------------------------------------
def generate_platforms(start_x, count=20, is_start=False):
    """Generate a series of parkour platforms."""
    platforms = []
    x = start_x
    y = GROUND_Y
    for i in range(count):
        # First platform is extra wide so player has safe footing
        if is_start and i == 0:
            w = 350
        else:
            w = random.randint(150, 300)
        platforms.append(Platform(x, y, w))
        gap = random.randint(40, 110)
        dy = random.choice([-50, -30, -20, 0, 0, 0, 20, 30])
        x += w + gap
        y = max(250, min(GROUND_Y, y + dy))
    return platforms


# ---------------------------------------------------------------------------
# Draw helpers
# ---------------------------------------------------------------------------
def draw_gradient_bg(surface):
    """Draw a vertical gradient background."""
    for y in range(SCREEN_HEIGHT):
        t = y / SCREEN_HEIGHT
        r = int(BG_TOP[0] * (1 - t) + BG_BOTTOM[0] * t)
        g = int(BG_TOP[1] * (1 - t) + BG_BOTTOM[1] * t)
        b = int(BG_TOP[2] * (1 - t) + BG_BOTTOM[2] * t)
        pygame.draw.line(surface, (r, g, b), (0, y), (SCREEN_WIDTH, y))


def draw_hud(surface, font, small_font, score, state, pitch, rms, alive):
    """Draw heads-up display."""
    # Semi-transparent HUD bar at top
    hud_surf = pygame.Surface((SCREEN_WIDTH, 54), pygame.SRCALPHA)
    hud_surf.fill((10, 8, 30, 170))
    surface.blit(hud_surf, (0, 0))

    # Score
    score_text = font.render(f"SCORE: {score}", True, ACCENT)
    surface.blit(score_text, (20, 12))

    # Pitch state indicator
    state_labels = {STATE_STOP: "STOP", STATE_WALK: "WALK", STATE_JUMP: "JUMP"}
    state_colors = {STATE_STOP: (180, 180, 180), STATE_WALK: ACCENT, STATE_JUMP: (255, 220, 80)}
    label = state_labels.get(state, "?")
    color = state_colors.get(state, TEXT_COLOR)
    state_text = font.render(f"STATE: {label}", True, color)
    surface.blit(state_text, (SCREEN_WIDTH // 2 - state_text.get_width() // 2, 12))

    # Pitch value
    pitch_str = f"{pitch:.0f} Hz" if pitch > 0 else "---"
    pitch_text = small_font.render(f"Pitch: {pitch_str}  |  Vol: {rms:.3f}", True, (160, 160, 200))
    surface.blit(pitch_text, (SCREEN_WIDTH - pitch_text.get_width() - 20, 18))

    # Mic level bar
    bar_x, bar_y, bar_w, bar_h = SCREEN_WIDTH - 220, 40, 200, 6
    pygame.draw.rect(surface, (40, 35, 70), (bar_x, bar_y, bar_w, bar_h), border_radius=3)
    fill_w = min(bar_w, int(bar_w * rms * 20))
    if fill_w > 0:
        bar_color = ACCENT if state != STATE_JUMP else (255, 220, 80)
        pygame.draw.rect(surface, bar_color, (bar_x, bar_y, fill_w, bar_h), border_radius=3)


def draw_game_over(surface, font, big_font, score):
    """Draw game over overlay."""
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 150))
    surface.blit(overlay, (0, 0))

    go_text = big_font.render("GAME OVER", True, WARNING_COLOR)
    surface.blit(go_text, (SCREEN_WIDTH // 2 - go_text.get_width() // 2, 200))

    score_text = font.render(f"Score: {score}", True, ACCENT)
    surface.blit(score_text, (SCREEN_WIDTH // 2 - score_text.get_width() // 2, 280))

    hint_text = font.render("Press R to Restart", True, TEXT_COLOR)
    surface.blit(hint_text, (SCREEN_WIDTH // 2 - hint_text.get_width() // 2, 340))


def draw_instructions(surface, font, alpha):
    """Fade-in instructions at game start."""
    if alpha <= 0:
        return
    a = min(255, int(alpha))
    lines = [
        "🎤  Voice-Controlled Parkour  🎤",
        "",
        "HIGH PITCH  →  Jump",
        "NORMAL PITCH  →  Run",
        "SILENCE  →  Stop",
    ]
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, min(140, a)))
    surface.blit(overlay, (0, 0))
    y = 180
    for line in lines:
        color = (*ACCENT[:3], a) if "→" in line else (*TEXT_COLOR[:3], a)
        txt_surf = font.render(line, True, color[:3])
        txt_surf.set_alpha(a)
        surface.blit(txt_surf, (SCREEN_WIDTH // 2 - txt_surf.get_width() // 2, y))
        y += 36


# ---------------------------------------------------------------------------
# Main Game
# ---------------------------------------------------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("🎤 Voice Parkour — Pitch-Controlled Runner")
    clock = pygame.time.Clock()

    # Fonts
    try:
        font = pygame.font.SysFont("Helvetica", 22, bold=True)
        small_font = pygame.font.SysFont("Helvetica", 16)
        big_font = pygame.font.SysFont("Helvetica", 56, bold=True)
    except Exception:
        font = pygame.font.Font(None, 26)
        small_font = pygame.font.Font(None, 20)
        big_font = pygame.font.Font(None, 60)

    # Pre-render gradient background
    bg_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    draw_gradient_bg(bg_surface)

    # Stars
    stars = [Star() for _ in range(80)]

    # Start pitch detector
    detector = PitchDetector()
    detector.start()

    def reset_game():
        platforms = generate_platforms(0, 40, is_start=True)
        # Spawn player centered on the first platform
        first_plat = platforms[0]
        spawn_x = first_plat.rect.x + first_plat.rect.width // 2 - 15
        player = Player(spawn_x, first_plat.rect.top - 44)
        particles = []
        score = 0
        instruction_alpha = 500  # frames of instruction visibility
        return player, platforms, particles, score, instruction_alpha

    player, platforms, particles, score, instruction_alpha = reset_game()
    cam_x = 0
    furthest_x = player.x

    running = True
    while running:
        dt = clock.tick(FPS)

        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r and not player.alive:
                    player, platforms, particles, score, instruction_alpha = reset_game()
                    cam_x = 0
                    furthest_x = player.x

        # Get pitch state from detector thread
        state, pitch, rms = detector.get_state()

        if player.alive:
            # Update player
            player.update(platforms, state)

            # Score: distance traveled
            if player.x > furthest_x:
                score += int(player.x - furthest_x)
                furthest_x = player.x

            # Camera follows player
            target_cam = player.x - 200
            cam_x += (target_cam - cam_x) * 0.1

            # Spawn particles while running
            if player.moving and player.on_ground and random.random() < 0.4:
                particles.append(Particle(player.x + 10, player.y + player.height, ACCENT))

            # Jump burst particles
            if state == STATE_JUMP and player.on_ground:
                for _ in range(8):
                    particles.append(Particle(player.x + 15, player.y + player.height))

            # Generate more platforms if needed
            last_plat_end = max(p.rect.right for p in platforms) if platforms else 0
            if player.x + SCREEN_WIDTH * 2 > last_plat_end:
                new_plats = generate_platforms(last_plat_end + 60, 20)
                platforms.extend(new_plats)

            # Remove far-behind platforms
            platforms = [p for p in platforms if p.rect.right > cam_x - 400]

            # Instruction fade
            if instruction_alpha > 0:
                instruction_alpha -= 2

        # Update particles
        for p in particles:
            p.update()
        particles = [p for p in particles if p.life > 0]

        # ---- DRAW --------------------------------------------------------
        screen.blit(bg_surface, (0, 0))

        # Stars with parallax + twinkle
        t = time.time()
        for star in stars:
            sx = (star.x - cam_x * star.speed) % SCREEN_WIDTH
            twinkle = math.sin(t * star.twinkle_speed * 60 + star.twinkle_phase)
            brightness = int(star.brightness * (0.6 + 0.4 * twinkle))
            brightness = max(0, min(255, brightness))
            c = (brightness, brightness, int(brightness * 0.9))
            if star.size == 1:
                screen.set_at((int(sx), star.y), c)
            else:
                pygame.draw.circle(screen, c, (int(sx), star.y), star.size)

        # Platforms
        for plat in platforms:
            if -100 < plat.rect.x - cam_x < SCREEN_WIDTH + 100:
                plat.draw(screen, cam_x)

        # Particles
        for p in particles:
            p.draw(screen, cam_x)

        # Player
        player.draw(screen, cam_x)

        # HUD
        draw_hud(screen, font, small_font, score, state, pitch, rms, player.alive)

        # Instructions overlay
        if instruction_alpha > 0:
            draw_instructions(screen, font, instruction_alpha)

        # Game over
        if not player.alive:
            draw_game_over(screen, font, big_font, score)

        pygame.display.flip()

    # Cleanup
    detector.stop()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
