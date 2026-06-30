import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import ssl
# import matplotlib.pyplot as plt

ssl._create_default_https_context = ssl._create_unverified_context


# Download MNIST dataset
print("Downloading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten images from 28x28 to 784
x_train_flat = x_train.reshape(-1, 28*28)
x_test_flat = x_test.reshape(-1, 28*28)

# Build simple neural network
model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(10, activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model
print("Training model...")
model.fit(x_train_flat, y_train, epochs=10, batch_size=128, validation_split=0.1)

# Evaluate on test set
print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(x_test_flat, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Inference on sample images
print("\nRunning inference on sample images...")
predictions = model.predict(x_test_flat[:5])
for i in range(5):
    predicted_digit = np.argmax(predictions[i])
    actual_digit = y_test[i]
    confidence = predictions[i][predicted_digit]
    print(f"Sample {i+1}: Predicted={predicted_digit}, Actual={actual_digit}, Confidence={confidence:.2%}")

# Save model
model.save("trained_mnist_model.h5")
print("\nModel saved as trained_mnist_model.h5")