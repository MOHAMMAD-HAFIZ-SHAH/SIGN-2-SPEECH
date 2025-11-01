"""
Quick test script to verify model predictions
Tests the trained model on sample images or webcam
"""

import cv2
import numpy as np
import tensorflow as tf
import json
import os
import sys

def load_model(model_path='trained_model'):
    """Load trained model and metadata"""
    try:
        # Load Keras model
        model = tf.keras.models.load_model(os.path.join(model_path, 'sign_language_model.h5'))
        
        # Load metadata
        with open(os.path.join(model_path, 'model_metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        print("✓ Model loaded successfully")
        print(f"  Classes: {metadata['class_names']}")
        print(f"  Input size: {metadata['img_size']}x{metadata['img_size']}")
        print(f"  Validation accuracy: {metadata['val_accuracy']:.2%}")
        
        return model, metadata
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


def preprocess_image(img, img_size):
    """Preprocess image for model input"""
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize
    img = cv2.resize(img, (img_size, img_size))
    
    # Normalize
    img = img.astype('float32') / 255.0
    
    # Add batch and channel dimensions
    img = np.expand_dims(img, axis=(0, -1))
    
    return img


def predict_image(model, img, metadata, confidence_threshold=0.6):
    """Make prediction on image"""
    # Preprocess
    processed = preprocess_image(img, metadata['img_size'])
    
    # Predict
    predictions = model.predict(processed, verbose=0)[0]
    
    # Get top prediction
    max_conf = np.max(predictions)
    pred_idx = np.argmax(predictions)
    pred_class = metadata['class_names'][pred_idx]
    
    if max_conf >= confidence_threshold:
        return pred_class, max_conf, predictions
    else:
        return None, max_conf, predictions


def test_webcam(model, metadata):
    """Test model on live webcam feed"""
    print("\n" + "="*60)
    print("Testing with webcam")
    print("="*60)
    print("Instructions:")
    print("- Show hand signs to the camera")
    print("- Keep hand centered in the rectangle")
    print("- Press 'q' to quit")
    print()
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("✓ Webcam opened. Press 'q' to quit\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get ROI
        h, w = gray.shape
        center_x, center_y = w // 2, h // 2
        rect_size = 300
        
        roi = gray[center_y - rect_size//2:center_y + rect_size//2,
                   center_x - rect_size//2:center_x + rect_size//2]
        
        # Make prediction
        pred_class, confidence, all_preds = predict_image(model, roi, metadata, 0.5)
        
        # Draw rectangle
        color = (0, 255, 0) if pred_class else (100, 100, 100)
        cv2.rectangle(frame,
                     (center_x - rect_size//2, center_y - rect_size//2),
                     (center_x + rect_size//2, center_y + rect_size//2),
                     color, 2)
        
        # Display prediction
        if pred_class:
            text = f"{pred_class}: {confidence:.2%}"
            cv2.putText(frame, text, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "No confident prediction", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        
        # Show top 3 predictions
        top_3_idx = np.argsort(all_preds)[-3:][::-1]
        y_offset = 80
        for idx in top_3_idx:
            class_name = metadata['class_names'][idx]
            conf = all_preds[idx]
            text = f"  {class_name}: {conf:.1%}"
            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            y_offset += 30
        
        cv2.imshow('Sign Language Detection Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Webcam test complete")


def test_image_file(model, metadata, image_path):
    """Test model on a single image file"""
    print(f"\nTesting on image: {image_path}")
    
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Predict
    pred_class, confidence, all_preds = predict_image(model, img, metadata)
    
    # Display results
    print("\nResults:")
    print(f"  Predicted: {pred_class or 'No confident prediction'}")
    print(f"  Confidence: {confidence:.2%}")
    print("\nTop 5 predictions:")
    top_5_idx = np.argsort(all_preds)[-5:][::-1]
    for idx in top_5_idx:
        class_name = metadata['class_names'][idx]
        conf = all_preds[idx]
        print(f"    {class_name}: {conf:.2%}")


def test_directory(model, metadata, dir_path):
    """Test model on all images in directory"""
    print(f"\nTesting on images in: {dir_path}")
    
    image_files = [f for f in os.listdir(dir_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No images found in directory")
        return
    
    print(f"Found {len(image_files)} images\n")
    
    correct = 0
    total = 0
    
    for img_file in image_files:
        img_path = os.path.join(dir_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            continue
        
        # Get true label from filename or parent directory
        true_label = os.path.basename(dir_path)
        
        # Predict
        pred_class, confidence, _ = predict_image(model, img, metadata)
        
        total += 1
        if pred_class == true_label:
            correct += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"  {status} {img_file}: {pred_class} ({confidence:.2%}) | True: {true_label}")
    
    if total > 0:
        accuracy = correct / total
        print(f"\nAccuracy: {correct}/{total} = {accuracy:.2%}")


def main():
    print("="*60)
    print("Sign Language Model Test")
    print("="*60)
    
    # Load model
    model, metadata = load_model('trained_model')
    if model is None:
        print("\nError: Could not load model. Have you trained it yet?")
        print("Run: python train_sign_model.py <dataset_path>")
        return
    
    # Interactive menu
    while True:
        print("\n" + "="*60)
        print("Test Options:")
        print("  1. Test with webcam (live)")
        print("  2. Test single image file")
        print("  3. Test directory of images")
        print("  4. Exit")
        print("="*60)
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            test_webcam(model, metadata)
        
        elif choice == '2':
            img_path = input("Enter image path: ").strip()
            if os.path.exists(img_path):
                test_image_file(model, metadata, img_path)
            else:
                print(f"Error: File not found: {img_path}")
        
        elif choice == '3':
            dir_path = input("Enter directory path: ").strip()
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                test_directory(model, metadata, dir_path)
            else:
                print(f"Error: Directory not found: {dir_path}")
        
        elif choice == '4':
            print("\nExiting...")
            break
        
        else:
            print("Invalid choice. Please enter 1-4.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted")
        cv2.destroyAllWindows()
