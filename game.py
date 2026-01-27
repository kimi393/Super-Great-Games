import pygame
import numpy as np
from tensorflow import keras
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 700
CANVAS_WIDTH = 280
CANVAS_HEIGHT = 280
CANVAS_X = 50
CANVAS_Y = 150
BRUSH_SIZE = 15
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
LIGHT_GRAY = (240, 240, 240)
GREEN = (50, 200, 50)
RED = (200, 50, 50)

# Load model
model = keras.models.load_model("mnist_model.h5")

# Create window
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("MNIST Digit Classifier - Draw a Digit!")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 48)
small_font = pygame.font.Font(None, 24)
medium_font = pygame.font.Font(None, 32)

# Canvas surface
canvas = pygame.Surface((CANVAS_WIDTH, CANVAS_HEIGHT))
canvas.fill(WHITE)

def predict_digit(canvas_surface):
    """Convert canvas to numpy array and predict digit"""
    try:
        # Convert pygame surface to numpy array
        canvas_array = pygame.surfarray.array3d(canvas_surface)
        canvas_array = np.transpose(canvas_array, (1, 0, 2))
        
        # Convert to grayscale (take red channel)
        gray = canvas_array[:, :, 0]
        
        # Invert (white background = 255, black drawing = 0)
        gray = 255 - gray
        
        # Resize to 28x28
        from scipy import ndimage
        gray_resized = ndimage.zoom(gray, (28/CANVAS_HEIGHT, 28/CANVAS_WIDTH), order=1)
        
        # Normalize
        gray_resized = gray_resized.astype("float32") / 255.0
        
        # Check if canvas has any drawing
        if gray_resized.max() < 0.1:
            return None, None, np.zeros(10)
        
        # Predict
        prediction = model.predict(gray_resized.reshape(1, -1), verbose=0)
        predicted_digit = np.argmax(prediction[0])
        confidence = prediction[0][predicted_digit]
        
        return predicted_digit, confidence, prediction[0]
    except:
        return None, None, np.zeros(10)

# Main loop
running = True
drawing = False
prediction_result = None

while running:
    clock.tick(FPS)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if CANVAS_X <= event.pos[0] <= CANVAS_X + CANVAS_WIDTH and \
               CANVAS_Y <= event.pos[1] <= CANVAS_Y + CANVAS_HEIGHT:
                drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:  # Clear canvas
                canvas.fill(WHITE)
                prediction_result = None
            elif event.key == pygame.K_ESCAPE:
                running = False
    
    # Draw on canvas
    if drawing:
        mouse_pos = pygame.mouse.get_pos()
        if CANVAS_X <= mouse_pos[0] <= CANVAS_X + CANVAS_WIDTH and \
           CANVAS_Y <= mouse_pos[1] <= CANVAS_Y + CANVAS_HEIGHT:
            local_x = mouse_pos[0] - CANVAS_X
            local_y = mouse_pos[1] - CANVAS_Y
            pygame.draw.circle(canvas, BLACK, (local_x, local_y), BRUSH_SIZE)
    
    # Real-time prediction
    prediction_result = predict_digit(canvas)
    
    # Draw background
    screen.fill(LIGHT_GRAY)
    
    # Draw title
    title = font.render("MNIST Digit Classifier", True, BLACK)
    screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 20))
    
    # Draw canvas border and canvas
    pygame.draw.rect(screen, BLACK, (CANVAS_X - 2, CANVAS_Y - 2, CANVAS_WIDTH + 4, CANVAS_HEIGHT + 4), 2)
    screen.blit(canvas, (CANVAS_X, CANVAS_Y))
    
    # Draw instructions
    instructions = [
        "Draw a digit (0-9) in the box",
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