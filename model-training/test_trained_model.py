"""
Test the trained sign language recognition model
"""
import numpy as np
import cv2
from tensorflow import keras
import json

# Load model and metadata
print("Loading model...")
model = keras.models.load_model('./trained_model/best_model.h5')

with open('./trained_model/model_metadata.json', 'r') as f:
    metadata = json.load(f)

class_names = metadata['class_names']
img_size = metadata['img_size']

print(f"Model loaded successfully!")
print(f"Classes: {class_names}")
print(f"Validation Accuracy: {metadata['val_accuracy']*100:.2f}%")
print(f"Validation Loss: {metadata['val_loss']:.6f}")

# Test with a sample image
def predict_sign(image_path):
    """Predict sign language letter from image"""
    # Read and preprocess image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    # Resize
    img = cv2.resize(img, (img_size, img_size))
    
    # Normalize
    img = img.astype('float32') / 255.0
    
    # Add batch and channel dimensions
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    
    # Predict
    predictions = model.predict(img, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    predicted_letter = class_names[predicted_class]
    
    print(f"\nPrediction: {predicted_letter}")
    print(f"Confidence: {confidence*100:.2f}%")
    
    # Show top 3 predictions
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    print("\nTop 3 predictions:")
    for idx in top_3_idx:
        print(f"  {class_names[idx]}: {predictions[0][idx]*100:.2f}%")
    
    return predicted_letter, confidence

# Test with sample images from dataset
print("\n" + "="*60)
print("Testing with sample images...")
print("="*60)

# Test a few samples
test_samples = [
    ('./dataset/A/0.jpg', 'A'),
    ('./dataset/B/0.jpg', 'B'),
    ('./dataset/Z/0.jpg', 'Z'),
]

for img_path, expected in test_samples:
    print(f"\nTesting: {img_path} (Expected: {expected})")
    try:
        predicted, confidence = predict_sign(img_path)
        if predicted == expected:
            print("✓ CORRECT!")
        else:
            print(f"✗ INCORRECT (Expected: {expected})")
    except Exception as e:
        print(f"Error: {e}")

print("\n" + "="*60)
print("Model testing complete!")
print("="*60)
