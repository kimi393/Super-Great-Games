import pygame
import numpy as np
from tensorflow.keras.datasets import mnist
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 700
GRID_SIZE = 28
PIXEL_SIZE = 15
CANVAS_WIDTH = GRID_SIZE * PIXEL_SIZE
CANVAS_HEIGHT = GRID_SIZE * PIXEL_SIZE
CANVAS_X = 50
CANVAS_Y = 80
FPS = 60
BRUSH_SIZE = 2  # Brush radius in pixels

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
LIGHT_GRAY = (240, 240, 240)
GREEN = (50, 200, 50)
RED = (200, 50, 50)
DARK_GRAY = (100, 100, 100)

# Load MNIST dataset and calculate average pixel values
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0

# Calculate average pixel values for each digit (0-9)
pixel_scores = {}
for digit in range(10):
    digit_images = x_train[y_train == digit]
    pixel_scores[digit] = np.mean(digit_images, axis=0)
print("Model loaded!")

# Create window
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("MNIST Pixel Averaging Classifier - Draw a Digit!")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 48)
small_font = pygame.font.Font(None, 24)
medium_font = pygame.font.Font(None, 32)

# 8x8 pixel grid
pixel_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

def predict_digit(grid):
    """Predict digit using pixel averaging method"""
    try:
        grid_normalized = grid.astype("float32")
        
        # Check if grid has any drawing
        if grid_normalized.max() < 0.1:
            return None, None, np.zeros(10)
        
        # Calculate scores for each digit (dissimilarity)
        sample_scores = {}
        for digit in range(10):
            score = np.sum(np.abs(grid_normalized - pixel_scores[digit]))
            sample_scores[digit] = score
        
        # Lower score means better match
        predicted_digit = min(sample_scores, key=sample_scores.get)
        
        # Convert scores to confidence (inverse relationship)
        min_score = min(sample_scores.values())
        max_score = max(sample_scores.values())
        
        # Normalize scores to confidence (0-1)
        all_confidences = np.zeros(10)
        for digit in range(10):
            # Invert: lower dissimilarity = higher confidence
            all_confidences[digit] = 1.0 - ((sample_scores[digit] - min_score) / (max_score - min_score + 1e-8))
        
        confidence = all_confidences[predicted_digit]
        
        return predicted_digit, confidence, all_confidences
    except:
        return None, None, np.zeros(10)

def get_grid_position(mouse_pos):
    """Convert mouse position to grid coordinates"""
    rel_x = mouse_pos[0] - CANVAS_X
    rel_y = mouse_pos[1] - CANVAS_Y
    
    if rel_x < 0 or rel_y < 0 or rel_x >= CANVAS_WIDTH or rel_y >= CANVAS_HEIGHT:
        return None
    
    grid_x = int(rel_x / PIXEL_SIZE)
    grid_y = int(rel_y / PIXEL_SIZE)
    
    if grid_x >= GRID_SIZE or grid_y >= GRID_SIZE:
        return None
    
    return grid_x, grid_y

def draw_brush(grid, grid_x, grid_y, brush_size, intensity=0.3):
    """Draw with brush tool - fills pixels in a circular pattern with smooth gradient"""
    for dy in range(-brush_size, brush_size + 1):
        for dx in range(-brush_size, brush_size + 1):
            distance = np.sqrt(dx*dx + dy*dy)
            # Circular brush with smooth gradient falloff
            if distance <= brush_size:
                # Gradient intensity based on distance from center
                gradient_intensity = intensity * (1.0 - (distance / (brush_size + 0.5)))
                nx, ny = grid_x + dx, grid_y + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    grid[ny, nx] = min(1.0, grid[ny, nx] + gradient_intensity)

# Main loop
running = True
drawing = False
prediction_result = None
mouse_pos = (0, 0)

while running:
    clock.tick(FPS)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = get_grid_position(event.pos)
            if pos:
                drawing = True
                grid_x, grid_y = pos
                draw_brush(pixel_grid, grid_x, grid_y, BRUSH_SIZE)
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif event.type == pygame.MOUSEMOTION:
            mouse_pos = event.pos
            if drawing:
                pos = get_grid_position(event.pos)
                if pos:
                    grid_x, grid_y = pos
                    draw_brush(pixel_grid, grid_x, grid_y, BRUSH_SIZE)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:  # Clear canvas
                pixel_grid.fill(0.0)
                prediction_result = None
            elif event.key == pygame.K_ESCAPE:
                running = False
    
    # Real-time prediction
    prediction_result = predict_digit(pixel_grid)
    
    # Draw background
    screen.fill(LIGHT_GRAY)
    
    # Draw title
    title = font.render("MNIST 28x28 Pixel Classifier (Brush Tool)", True, BLACK)
    screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 20))
    
    # Draw grid
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            rect_x = CANVAS_X + x * PIXEL_SIZE
            rect_y = CANVAS_Y + y * PIXEL_SIZE
            
            # Draw pixel background (white or black based on value)
            color = WHITE if pixel_grid[y, x] < 0.5 else BLACK
            pygame.draw.rect(screen, color, (rect_x, rect_y, PIXEL_SIZE, PIXEL_SIZE))
            
            # Draw grid lines
            pygame.draw.rect(screen, DARK_GRAY, (rect_x, rect_y, PIXEL_SIZE, PIXEL_SIZE), 1)
    
    # Draw canvas border
    pygame.draw.rect(screen, BLACK, (CANVAS_X - 2, CANVAS_Y - 2, CANVAS_WIDTH + 4, CANVAS_HEIGHT + 4), 2)
    
    # Draw brush outline
    pos = get_grid_position(mouse_pos)
    if pos:
        grid_x, grid_y = pos
        brush_center_x = CANVAS_X + grid_x * PIXEL_SIZE + PIXEL_SIZE // 2
        brush_center_y = CANVAS_Y + grid_y * PIXEL_SIZE + PIXEL_SIZE // 2
        brush_radius = (BRUSH_SIZE + 0.5) * PIXEL_SIZE
        pygame.draw.circle(screen, GRAY, (brush_center_x, brush_center_y), brush_radius, 2)
    
    # Draw instructions
    instructions = [
        "Click pixels to draw a digit",
        "Press C to clear",
        "Press ESC to exit"
    ]
    for i, instruction in enumerate(instructions):
        text = small_font.render(instruction, True, BLACK)
        screen.blit(text, (CANVAS_X, CANVAS_Y + CANVAS_HEIGHT + 20 + i * 25))
    
    # Draw prediction result on the right
    right_panel_x = CANVAS_X + CANVAS_WIDTH + 100
    
    if prediction_result[0] is not None:
        predicted_digit, confidence, all_pred = prediction_result
        
        # Large predicted digit
        result_text = font.render(f"{predicted_digit}", True, GREEN)
        screen.blit(result_text, (right_panel_x, CANVAS_Y + 50))
        
        label_text = small_font.render("Predicted", True, BLACK)
        screen.blit(label_text, (right_panel_x, CANVAS_Y + 15))
        
        # Confidence percentage
        confidence_text = medium_font.render(f"{confidence:.1%}", True, BLACK)
        screen.blit(confidence_text, (right_panel_x, CANVAS_Y + 130))
        
        confidence_label = small_font.render("Confidence", True, BLACK)
        screen.blit(confidence_label, (right_panel_x, CANVAS_Y + 100))
        
        # Confidence bars for all digits
        bar_y = CANVAS_Y + 200
        bar_label = small_font.render("All Predictions:", True, BLACK)
        screen.blit(bar_label, (right_panel_x, bar_y - 30))
        
        for digit in range(10):
            bar_width = int(all_pred[digit] * 150)
            bar_height = 20
            
            # Background bar
            pygame.draw.rect(screen, GRAY, (right_panel_x + 40, bar_y + digit * 25, 150, bar_height))
            
            # Filled bar
            color = GREEN if digit == predicted_digit else BLACK
            pygame.draw.rect(screen, color, (right_panel_x + 40, bar_y + digit * 25, bar_width, bar_height))
            
            # Digit label
            digit_text = small_font.render(f"{digit}", True, BLACK)
            screen.blit(digit_text, (right_panel_x + 15, bar_y + digit * 25 + 2))
            
            # Percentage
            percent_text = small_font.render(f"{all_pred[digit]:.0%}", True, BLACK)
            screen.blit(percent_text, (right_panel_x + 200, bar_y + digit * 25 + 2))
    else:
        empty_text = small_font.render("Draw a digit", True, GRAY)
        screen.blit(empty_text, (right_panel_x, CANVAS_Y + 100))
    
    pygame.display.flip()

pygame.quit()
