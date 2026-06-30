import pygame
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import time
from enum import Enum

# Initialize Pygame
pygame.init()

# Constants
GRID_SIZE = 8
CELL_SIZE = 40
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 700
FPS = 60
TIMER_DURATION = 30  # seconds per round

pygame.mixer.init()
pygame.mixer.music.load("background_music.mp3")  # Put your music file in the same folder
pygame.mixer.music.play(-1)  # -1 means loop forever

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
BLUE = (100, 150, 255)
RED = (255, 100, 100)
GREEN = (100, 255, 100)
YELLOW = (255, 255, 100)
DARK_BLUE = (50, 100, 200)
DARK_RED = (200, 50, 50)

class GameState(Enum):
    QUESTION = 1
    DRAWING = 2
    SUBMISSION = 3
    RESULT = 4

class Player:
    def __init__(self, player_num, grid_x, grid_y):
        self.player_num = player_num
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        self.predicted_digit = None
        self.confidence = 0
        self.is_correct = False
        self.color = BLUE if player_num == 1 else RED

    def draw_pixel(self, row, col):
        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            self.grid[row, col] = 255

    def erase_pixel(self, row, col):
        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            self.grid[row, col] = 0

    def get_neighbors(self, row, col):
        """Get neighboring cells for line drawing"""
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                    neighbors.append((nr, nc))
        return neighbors

    def draw_line(self, start_row, start_col, end_row, end_col):
        """Draw a line between two points using Bresenham's algorithm"""
        points = []
        dx = abs(end_col - start_col)
        dy = abs(end_row - start_row)
        sx = 1 if start_col < end_col else -1
        sy = 1 if start_row < end_row else -1
        err = dx - dy

        x, y = start_col, start_row

        while True:
            points.append((y, x))
            if x == end_col and y == end_row:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        # Draw line with 1x1 thickness
        for row, col in points:
            self.draw_pixel(row, col)

    def erase_line(self, start_row, start_col, end_row, end_col):
        """Erase a line between two points using Bresenham's algorithm"""
        points = []
        dx = abs(end_col - start_col)
        dy = abs(end_row - start_row)
        sx = 1 if start_col < end_col else -1
        sy = 1 if start_row < end_row else -1
        err = dx - dy

        x, y = start_col, start_row

        while True:
            points.append((y, x))
            if x == end_col and y == end_row:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        # Erase line with 1x1 thickness
        for row, col in points:
            self.erase_pixel(row, col)

    def clear_grid(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

class DrawingGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("2-Player Drawing Game")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Game state
        self.state = GameState.QUESTION
        self.players = [
            Player(1, 150, 150),
            Player(2, WINDOW_WIDTH - 350, 150)
        ]
        self.model = keras.models.load_model("trained_mnist_model.h5")
        
        # Question
        self.question = ""
        self.answer = 0
        self.generate_question()
        
        # Timer
        self.round_start_time = time.time()
        self.question_display_time = 2  # Show question for 2 seconds
        
        # Drawing state
        self.last_pos = {1: None, 2: None}
        self.drawing = {1: False, 2: False}
        self.cursor_pos = {1: [GRID_SIZE // 2, GRID_SIZE // 2], 2: [GRID_SIZE // 2, GRID_SIZE // 2]}
        self.eraser_mode = {1: False, 2: False}
        self.last_keys = {1: {'w': False, 's': False, 'a': False, 'd': False}, 
                          2: {'up': False, 'down': False, 'left': False, 'right': False}}
        
        # Race state
        self.submitted = {1: False, 2: False}
        self.scores = {1: 0, 2: 0}
        self.submission_time = {1: None, 2: None}
        self.submission_message_time = {1: None, 2: None}
        
    def generate_question(self):
        """Generate a random math question with answer 0-9"""
        operations = ['+', '-']
        op = random.choice(operations)
        
        if op == '+':
            a = random.randint(0, 9)
            b = random.randint(0, 9 - a)
            self.answer = a + b
        else:  # subtraction
            a = random.randint(0, 9)
            b = random.randint(0, a)
            self.answer = a - b
        
        self.question = f"{a} {op} {b} = ?"
    
    def predict_digit(self, player_num):
        """Predict digit using the model"""
        player = self.players[player_num - 1]
        
        # Check if grid is empty
        if np.all(player.grid == 0):
            player.predicted_digit = -1  # -1 indicates no drawing
            player.confidence = 0.0
            player.is_correct = False
            return
        
        # Resize grid to 28x28 for the model
        grid_resized = np.zeros((28, 28), dtype=np.float32)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                for di in range(4):
                    for dj in range(4):
                        ni, nj = i * 4 + di, j * 4 + dj
                        if ni < 28 and nj < 28:
                            grid_resized[ni, nj] = max(grid_resized[ni, nj], player.grid[i, j] / 255.0)
        
        # Flatten and predict
        grid_flat = grid_resized.reshape(1, -1)
        prediction = self.model.predict(grid_flat, verbose=0)
        predicted_digit = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][predicted_digit])
        
        player.predicted_digit = predicted_digit
        player.confidence = confidence
        player.is_correct = (predicted_digit == self.answer)
    
    def submit_answer(self, player_num):
        """Player submits their answer"""
        self.predict_digit(player_num)
        self.submission_time[player_num] = time.time() - self.round_start_time
        self.submission_message_time[player_num] = time.time()
        player = self.players[player_num - 1]
        
        # Store prediction before clearing
        predicted_digit = player.predicted_digit
        confidence = player.confidence
        
        if player.is_correct:
            # Correct answer! Award points based on time
            elapsed = time.time() - self.round_start_time
            points = max(100 - int(elapsed * 10), 10)  # Points decrease over time, minimum 10
            self.scores[player_num] += points
            self.submitted[player_num] = True
            # Only move to RESULT if both players have submitted correct answers
            if self.submitted[1] and self.submitted[2]:
                self.state = GameState.RESULT
        else:
            # Wrong answer, erase grid but keep prediction for display
            player.clear_grid()
            self.last_pos[player_num] = None
            # Keep the predicted_digit and confidence for display
            player.predicted_digit = predicted_digit
            player.confidence = confidence
            # Don't set submitted to True - allow player to try again
    
    def handle_input(self):
        """Handle keyboard input for drawing"""
        keys = pygame.key.get_pressed()
        
        # Player 1: WASD for movement, Space to draw
        p1_w_pressed = keys[pygame.K_w]
        p1_s_pressed = keys[pygame.K_s]
        p1_a_pressed = keys[pygame.K_a]
        p1_d_pressed = keys[pygame.K_d]
        
        # Player 2: Arrow keys for movement, Shift to draw
        p2_up_pressed = keys[pygame.K_UP]
        p2_down_pressed = keys[pygame.K_DOWN]
        p2_left_pressed = keys[pygame.K_LEFT]
        p2_right_pressed = keys[pygame.K_RIGHT]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.drawing[1] = True
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    self.drawing[2] = True
                elif event.key == pygame.K_e:  # Toggle eraser for Player 1
                    self.eraser_mode[1] = not self.eraser_mode[1]
                elif event.key == pygame.K_p:  # Toggle eraser for Player 2
                    self.eraser_mode[2] = not self.eraser_mode[2]
                elif event.key == pygame.K_q and self.state == GameState.DRAWING and not self.players[0].is_correct:  # Player 1 submit
                    self.submit_answer(1)
                elif event.key == pygame.K_RETURN and self.state == GameState.DRAWING and not self.players[1].is_correct:  # Player 2 submit
                    self.submit_answer(2)
                elif event.key == pygame.K_r:  # Reset for next round
                    if self.state == GameState.RESULT:
                        self.start_new_round()
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    self.drawing[1] = False
                    self.last_pos[1] = None
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    self.drawing[2] = False
                    self.last_pos[2] = None
        
        # Update cursor positions based on movement (during drawing phase)
        if self.state == GameState.DRAWING:
            # Player 1 - only move on key press transition
            if p1_w_pressed and not self.last_keys[1]['w']:
                self.cursor_pos[1][0] = max(0, self.cursor_pos[1][0] - 1)
            if p1_s_pressed and not self.last_keys[1]['s']:
                self.cursor_pos[1][0] = min(GRID_SIZE - 1, self.cursor_pos[1][0] + 1)
            if p1_a_pressed and not self.last_keys[1]['a']:
                self.cursor_pos[1][1] = max(0, self.cursor_pos[1][1] - 1)
            if p1_d_pressed and not self.last_keys[1]['d']:
                self.cursor_pos[1][1] = min(GRID_SIZE - 1, self.cursor_pos[1][1] + 1)
            
            # Player 2 - only move on key press transition
            if p2_up_pressed and not self.last_keys[2]['up']:
                self.cursor_pos[2][0] = max(0, self.cursor_pos[2][0] - 1)
            if p2_down_pressed and not self.last_keys[2]['down']:
                self.cursor_pos[2][0] = min(GRID_SIZE - 1, self.cursor_pos[2][0] + 1)
            if p2_left_pressed and not self.last_keys[2]['left']:
                self.cursor_pos[2][1] = max(0, self.cursor_pos[2][1] - 1)
            if p2_right_pressed and not self.last_keys[2]['right']:
                self.cursor_pos[2][1] = min(GRID_SIZE - 1, self.cursor_pos[2][1] + 1)
            
            # Draw if space/shift is held
            if self.drawing[1]:
                if self.last_pos[1] is None:
                    if self.eraser_mode[1]:
                        self.players[0].erase_pixel(self.cursor_pos[1][0], self.cursor_pos[1][1])
                    else:
                        self.players[0].draw_pixel(self.cursor_pos[1][0], self.cursor_pos[1][1])
                else:
                    if self.eraser_mode[1]:
                        self.players[0].erase_line(self.last_pos[1][0], self.last_pos[1][1], self.cursor_pos[1][0], self.cursor_pos[1][1])
                    else:
                        self.players[0].draw_line(self.last_pos[1][0], self.last_pos[1][1], self.cursor_pos[1][0], self.cursor_pos[1][1])
                self.last_pos[1] = self.cursor_pos[1].copy()
                # Reset submission message time when player draws again
                if self.submission_message_time[1] is not None:
                    self.submission_message_time[1] = time.time() - 2  # Hide message
            
            if self.drawing[2]:
                if self.last_pos[2] is None:
                    if self.eraser_mode[2]:
                        self.players[1].erase_pixel(self.cursor_pos[2][0], self.cursor_pos[2][1])
                    else:
                        self.players[1].draw_pixel(self.cursor_pos[2][0], self.cursor_pos[2][1])
                else:
                    if self.eraser_mode[2]:
                        self.players[1].erase_line(self.last_pos[2][0], self.last_pos[2][1], self.cursor_pos[2][0], self.cursor_pos[2][1])
                    else:
                        self.players[1].draw_line(self.last_pos[2][0], self.last_pos[2][1], self.cursor_pos[2][0], self.cursor_pos[2][1])
                self.last_pos[2] = self.cursor_pos[2].copy()
                # Reset submission message time when player draws again
                if self.submission_message_time[2] is not None:
                    self.submission_message_time[2] = time.time() - 2  # Hide message
        
        # Store current key states for next frame
        self.last_keys[1] = {'w': p1_w_pressed, 's': p1_s_pressed, 'a': p1_a_pressed, 'd': p1_d_pressed}
        self.last_keys[2] = {'up': p2_up_pressed, 'down': p2_down_pressed, 'left': p2_left_pressed, 'right': p2_right_pressed}
        
        return True
    
    def update(self):
        """Update game state"""
        if self.state == GameState.QUESTION:
            elapsed = time.time() - self.round_start_time
            if elapsed > self.question_display_time:
                self.state = GameState.DRAWING
                self.round_start_time = time.time()
        
        elif self.state == GameState.DRAWING:
            elapsed = time.time() - self.round_start_time
            if elapsed > TIMER_DURATION:
                # Time's up! Both players must submit
                for player_num in [1, 2]:
                    if not self.submitted[player_num]:
                        self.predict_digit(player_num)
                        self.submitted[player_num] = True
                        self.submission_time[player_num] = TIMER_DURATION
                self.state = GameState.RESULT
    
    def draw_grid(self, player):
        """Draw the grid and drawn content"""
        x, y = player.grid_x, player.grid_y
        
        # Draw grid background
        pygame.draw.rect(self.screen, player.color, (x - 5, y - 5, GRID_SIZE * CELL_SIZE + 10, GRID_SIZE * CELL_SIZE + 10), 2)
        pygame.draw.rect(self.screen, WHITE, (x, y, GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE))
        
        # Draw grid lines
        for i in range(GRID_SIZE + 1):
            # Horizontal lines
            pygame.draw.line(self.screen, LIGHT_GRAY, (x, y + i * CELL_SIZE), (x + GRID_SIZE * CELL_SIZE, y + i * CELL_SIZE), 1)
            # Vertical lines
            pygame.draw.line(self.screen, LIGHT_GRAY, (x + i * CELL_SIZE, y), (x + i * CELL_SIZE, y + GRID_SIZE * CELL_SIZE), 1)
        
        # Draw the drawn content
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if player.grid[i, j] > 0:
                    color_val = player.grid[i, j]
                    cell_x = x + j * CELL_SIZE
                    cell_y = y + i * CELL_SIZE
                    pygame.draw.rect(self.screen, (0, 0, color_val), (cell_x, cell_y, CELL_SIZE, CELL_SIZE))
        
        # Draw cursor during drawing phase
        if self.state == GameState.DRAWING:
            cursor_idx = player.player_num
            cursor_row, cursor_col = self.cursor_pos[cursor_idx]
            cursor_x = x + cursor_col * CELL_SIZE
            cursor_y = y + cursor_row * CELL_SIZE
            pygame.draw.rect(self.screen, YELLOW, (cursor_x, cursor_y, CELL_SIZE, CELL_SIZE), 2)
    
    def draw_ui(self):
        """Draw UI elements"""
        # Draw question
        question_text = self.font_medium.render(self.question, True, BLACK)
        self.screen.blit(question_text, (WINDOW_WIDTH // 2 - question_text.get_width() // 2, 30))
        
        # Draw player labels
        p1_label = self.font_medium.render("Player 1", True, BLUE)
        p2_label = self.font_medium.render("Player 2", True, RED)
        self.screen.blit(p1_label, (self.players[0].grid_x, self.players[0].grid_y - 40))
        self.screen.blit(p2_label, (self.players[1].grid_x, self.players[1].grid_y - 40))
        
        # Draw submission messages
        for player_num in [1, 2]:
            if self.submitted[player_num] and self.submission_message_time[player_num] is not None:
                elapsed_since_submission = time.time() - self.submission_message_time[player_num]
                if elapsed_since_submission < 2:  # Show message for 2 seconds
                    player = self.players[player_num - 1]
                    if player.predicted_digit is None:
                        msg = "Predicting..."
                        color = GRAY
                    elif player.predicted_digit == -1:
                        msg = "No drawing!"
                        color = RED
                    else:
                        msg = f"Predicted: {player.predicted_digit}"
                        color = GREEN if player.is_correct else RED
                    
                    msg_text = self.font_large.render(msg, True, color)
                    x = self.players[player_num - 1].grid_x + (GRID_SIZE * CELL_SIZE) // 2
                    y = self.players[player_num - 1].grid_y + (GRID_SIZE * CELL_SIZE) + 50
                    self.screen.blit(msg_text, (x - msg_text.get_width() // 2, y))
        
        # Draw timer
        if self.state == GameState.DRAWING:
            elapsed = time.time() - self.round_start_time
            remaining = max(0, TIMER_DURATION - elapsed)
            timer_text = self.font_large.render(f"{remaining:.1f}s", True, BLACK)
            self.screen.blit(timer_text, (WINDOW_WIDTH // 2 - timer_text.get_width() // 2, WINDOW_HEIGHT - 80))
        
        # Draw controls
        controls_y = WINDOW_HEIGHT - 150
        controls = [
            "Player 1: WASD to move, SPACE to draw, E to erase, Q to submit",
            "Player 2: Arrow keys to move, SHIFT to draw, P to erase, ENTER to submit"
        ]
        for i, control in enumerate(controls):
            control_text = self.font_small.render(control, True, GRAY)
            self.screen.blit(control_text, (20, controls_y + i * 30))
    
    def draw_results(self):
        """Draw result screen"""
        # Title
        title = self.font_large.render("RESULTS", True, BLACK)
        self.screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 50))
        
        # Answer
        answer_text = self.font_medium.render(f"Correct Answer: {self.answer}", True, BLACK)
        self.screen.blit(answer_text, (WINDOW_WIDTH // 2 - answer_text.get_width() // 2, 120))
        
        # Player results
        y_offset = 200
        for i, player in enumerate(self.players):
            player_name = self.font_medium.render(f"Player {i + 1}", True, player.color)
            
            if self.submitted[i + 1]:
                if player.predicted_digit == -1:
                    # Empty grid
                    predicted = self.font_medium.render("Predicted: No drawing", True, BLACK)
                    confidence = self.font_small.render("Confidence: 0%", True, GRAY)
                else:
                    predicted = self.font_medium.render(f"Predicted: {player.predicted_digit}", True, BLACK)
                    confidence = self.font_small.render(f"Confidence: {player.confidence:.1%}", True, GRAY)
                
                result_color = GREEN if player.is_correct else RED
                result_text = self.font_medium.render("✓ CORRECT!" if player.is_correct else "✗ WRONG", True, result_color)
                
                points_earned = max(100 - int(self.submission_time[i + 1] * 10), 10) if player.is_correct else 0
                points_text = self.font_small.render(f"Points: +{points_earned}", True, BLUE)
                
                x = 50 if i == 0 else WINDOW_WIDTH // 2 + 50
                self.screen.blit(player_name, (x, y_offset))
                self.screen.blit(predicted, (x, y_offset + 40))
                self.screen.blit(confidence, (x, y_offset + 80))
                self.screen.blit(result_text, (x, y_offset + 120))
                self.screen.blit(points_text, (x, y_offset + 160))
            else:
                status_text = self.font_medium.render("Did not submit", True, RED)
                x = 50 if i == 0 else WINDOW_WIDTH // 2 + 50
                self.screen.blit(player_name, (x, y_offset))
                self.screen.blit(status_text, (x, y_offset + 40))
        
        # Display total scores
        score_y = WINDOW_HEIGHT - 200
        p1_score_text = self.font_medium.render(f"Player 1 Total: {self.scores[1]}", True, BLUE)
        p2_score_text = self.font_medium.render(f"Player 2 Total: {self.scores[2]}", True, RED)
        self.screen.blit(p1_score_text, (100, score_y))
        self.screen.blit(p2_score_text, (WINDOW_WIDTH // 2 + 100, score_y))
        
        # Instructions
        next_text = self.font_small.render("Press R to start next round", True, GRAY)
        self.screen.blit(next_text, (WINDOW_WIDTH // 2 - next_text.get_width() // 2, WINDOW_HEIGHT - 50))
    
    def start_new_round(self):
        """Start a new round"""
        self.players[0].clear_grid()
        self.players[1].clear_grid()
        self.generate_question()
        self.state = GameState.QUESTION
        self.round_start_time = time.time()
        self.last_pos = {1: None, 2: None}
        self.submitted = {1: False, 2: False}
        self.submission_time = {1: None, 2: None}
        self.submission_message_time = {1: None, 2: None}
    
    def draw(self):
        """Draw everything"""
        self.screen.fill(WHITE)
        
        if self.state == GameState.RESULT:
            self.draw_results()
        else:
            self.draw_grid(self.players[0])
            self.draw_grid(self.players[1])
            self.draw_ui()
        
        pygame.display.flip()
    
    def run(self):
        """Main game loop"""
        running = True
        while running:
            running = self.handle_input()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        
        pygame.quit()

if __name__ == "__main__":
    game = DrawingGame()
    game.run()
