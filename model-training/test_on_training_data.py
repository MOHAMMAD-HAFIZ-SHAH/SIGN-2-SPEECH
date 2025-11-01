"""
Test to see if the model can correctly predict images from its own training dataset
This will tell us if the issue is with the model or the webcam preprocessing
"""
import cv2
import numpy as np
from tensorflow import keras
import json
import random

# Load model and metadata
print("Loading model...")
model = keras.models.load_model('./trained_model/best_model.h5')
with open('./trained_model/model_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Testing model on training images...")
print("="*60)

# Test a few random images from different classes
test_classes = ['A', 'B', 'C', 'E', 'M', 'Z']

for letter in test_classes:
    img_path = f'./dataset/{letter}/0.jpg'
    
    # Load and preprocess like training
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Could not load {img_path}")
        continue
    
    # Resize
    img_resized = cv2.resize(img, (64, 64))
    
    # Normalize
    img_normalized = img_resized.astype('float32') / 255.0
    
    # Add dimensions
    img_input = np.expand_dims(img_normalized, axis=-1)
    img_input = np.expand_dims(img_input, axis=0)
    
    # Predict
    predictions = model.predict(img_input, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    predicted_letter = metadata['class_names'][predicted_idx]
    confidence = predictions[0][predicted_idx]
    
    # Get top 3
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    
    status = "✓" if predicted_letter == letter else "✗"
    print(f"{status} {letter}/0.jpg => Predicted: {predicted_letter} ({confidence:.1%})")
    print(f"  Top 3: {', '.join([f'{metadata['class_names'][i]}({predictions[0][i]:.1%})' for i in top_3_idx])}")
    print()

print("="*60)
print("\nIf the model predicts training images correctly but webcam images")
print("as 'E', then the issue is preprocessing/image quality difference.")
print("\nIf the model predicts everything as 'E', then the training failed")
print("and the model needs to be retrained with balanced data.")
