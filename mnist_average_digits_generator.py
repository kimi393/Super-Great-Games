import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import os

# 1. Load MNIST dataset and get correct answer samples
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to 0-1 range
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 2. Calculate average pixel values for each digit (0-9)
pixel_scores = {}
for digit in range(10):
    # Get all images of this digit
    digit_images = x_train[y_train == digit]
    # Calculate mean pixel values
    pixel_scores[digit] = np.mean(digit_images, axis=0)

# 3. Load a test sample (using first test image)
test_sample = x_test[0]
test_label = y_test[0]

# 4. Calculate scores for the sample image
sample_scores = {}
for digit in range(10):
    # Sum the absolute differences between sample and average pixels
    score = np.sum(np.abs(test_sample - pixel_scores[digit]))
    sample_scores[digit] = score

# Lower score means better match
best_prediction = min(sample_scores, key=sample_scores.get)
confidence = 1.0 - (sample_scores[best_prediction] / max(sample_scores.values()))

# 5. Display results
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Show sample image
axes[0].imshow(test_sample, cmap='gray')
axes[0].set_title(f'Test Image (Actual: {test_label})')
axes[0].axis('off')

# Show prediction scores
digits = list(range(10))
scores = [sample_scores[d] for d in digits]
colors = ['red' if d == best_prediction else 'blue' for d in digits]
axes[1].bar(digits, scores, color=colors)
axes[1].set_xlabel('Digit')
axes[1].set_ylabel('Dissimilarity Score (lower is better)')
axes[1].set_title('Prediction Scores')
axes[1].set_xticks(digits)

plt.tight_layout()
plt.show()

print(f"Prediction: {best_prediction}")
print(f"Confidence: {confidence:.2%}")
print(f"Actual Label: {test_label}")
print(f"All Scores: {sample_scores}")

# Save average pixel image for each digit
output_dir = '/Users/kimi/Desktop/j/ ai hand whiteing boaro/mnist_digits'
os.makedirs(output_dir, exist_ok=True)

for digit in range(10):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(pixel_scores[digit], cmap='gray')
    ax.set_title(f'Average Image for Digit {digit}')
    ax.axis('off')
    
    filepath = os.path.join(output_dir, f'digit_{digit}_average.png')
    plt.savefig(filepath, bbox_inches='tight', dpi=100)
    plt.close()

print(f"Saved average digit images to: {output_dir}")