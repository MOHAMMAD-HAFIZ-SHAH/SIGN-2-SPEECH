"""
Debug tool to visualize what the model sees
"""
import cv2
import numpy as np
from tensorflow import keras
import json

# Load model and metadata
model = keras.models.load_model('./trained_model/best_model.h5')
with open('./trained_model/model_metadata.json', 'r') as f:
    metadata = json.load(f)

# Test with a training image
test_image_path = './dataset/A/0.jpg'

print("Loading test image:", test_image_path)
img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
print(f"Original shape: {img.shape}")

# Show original
cv2.imshow('Original Training Image', img)

# Apply same preprocessing as API
img_eq = cv2.equalizeHist(img)
cv2.imshow('After Histogram Equalization', img_eq)

img_blur = cv2.GaussianBlur(img_eq, (5, 5), 0)
cv2.imshow('After Gaussian Blur', img_blur)

img_resized = cv2.resize(img_blur, (64, 64))
cv2.imshow('Resized to 64x64', img_resized)

# Predict
img_normalized = img_resized.astype('float32') / 255.0
img_input = np.expand_dims(img_normalized, axis=-1)
img_input = np.expand_dims(img_input, axis=0)

predictions = model.predict(img_input, verbose=0)
predicted_idx = np.argmax(predictions[0])
predicted_sign = metadata['class_names'][predicted_idx]
confidence = predictions[0][predicted_idx]

print(f"\nPrediction: {predicted_sign} ({confidence:.2%})")
print("\nTop 5 predictions:")
top_5_idx = np.argsort(predictions[0])[-5:][::-1]
for idx in top_5_idx:
    print(f"  {metadata['class_names'][idx]}: {predictions[0][idx]:.2%}")

print("\nPress any key to close windows...")
cv2.waitKey(0)
cv2.destroyAllWindows()
