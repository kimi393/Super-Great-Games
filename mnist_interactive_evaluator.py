import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Load MNIST test dataset
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values
x_test = x_test.astype("float32") / 255.0

# Load trained model
model = keras.models.load_model("trained_mnist_model.h5")

# Initialize with random sample
current_idx = np.random.randint(0, len(x_test))

def display_prediction(idx):
    """Display prediction for sample at given index"""
    sample_image = x_test[idx]
    actual_digit = y_test[idx]
    
    # Make prediction
    sample_flat = sample_image.reshape(1, -1)
    prediction = model.predict(sample_flat, verbose=0)
    predicted_digit = np.argmax(prediction[0])
    confidence = prediction[0][predicted_digit]
    
    # Clear previous plot
    plt.clf()
    
    # Create subplots
    axes = plt.subplot(1, 2, 1)
    axes.imshow(sample_image, cmap='gray')
    axes.set_title(f"Actual Digit: {actual_digit}")
    axes.axis('off')
    
    axes2 = plt.subplot(1, 2, 2)
    axes2.bar(range(10), prediction[0])
    axes2.set_xlabel("Digit")
    axes2.set_ylabel("Confidence")
    axes2.set_title(f"Predicted: {predicted_digit} ({confidence:.2%})")
    axes2.set_xticks(range(10))
    
    plt.suptitle(f"Sample {idx+1} / {len(x_test)} | Use ← → arrows to navigate", fontsize=10)
    plt.tight_layout()
    plt.draw()
    
    print(f"Sample {idx+1}: Actual: {actual_digit}, Predicted: {predicted_digit}, Confidence: {confidence:.2%}")
    
    return idx

def on_key(event):
    """Handle keyboard input"""
    global current_idx
    if event.key == 'right':
        current_idx = (current_idx + 1) % len(x_test)
        display_prediction(current_idx)
    elif event.key == 'left':
        current_idx = (current_idx - 1) % len(x_test)
        display_prediction(current_idx)

# Create figure and display first prediction
fig = plt.figure(figsize=(10, 4))
display_prediction(current_idx)

# Connect keyboard event
fig.canvas.mpl_connect('key_press_event', on_key)

plt.show()