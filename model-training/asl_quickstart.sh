#!/bin/bash

# ASL Training Quick Start Script
# This script helps you train and deploy the ASL sign language model

echo "=========================================="
echo "ASL Sign Language Training & Deployment"
echo "=========================================="
echo ""

# Change to model-training directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found!"
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if required packages are installed
echo "Checking dependencies..."
pip install -q tensorflow opencv-python matplotlib scikit-learn flask flask-cors

echo ""
echo "What would you like to do?"
echo ""
echo "1) Train new ASL model (recommended)"
echo "2) Start ASL API server with new preprocessing"
echo "3) Start API server with old model (fallback)"
echo "4) Test model on training data"
echo "5) View training history"
echo ""

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Starting model training..."
        echo "This will take 15-20 minutes depending on your hardware."
        echo ""
        python3 train_asl_model.py
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ Training completed successfully!"
            echo ""
            read -p "Would you like to start the ASL API server now? (y/n): " start_server
            if [ "$start_server" = "y" ]; then
                echo ""
                echo "🚀 Starting ASL API server..."
                python3 api_server_asl.py
            fi
        else
            echo ""
            echo "❌ Training failed. Please check the error messages above."
        fi
        ;;
        
    2)
        echo ""
        echo "🚀 Starting ASL API server..."
        echo "Using ASL-style preprocessing (adaptive threshold + Otsu)"
        echo ""
        echo "Server will be available at: http://localhost:5001"
        echo "Press Ctrl+C to stop the server"
        echo ""
        python3 api_server_asl.py
        ;;
        
    3)
        echo ""
        echo "🚀 Starting API server with old model..."
        echo ""
        echo "Server will be available at: http://localhost:5001"
        echo "Press Ctrl+C to stop the server"
        echo ""
        python3 api_server.py
        ;;
        
    4)
        echo ""
        echo "🧪 Testing model on training data..."
        echo ""
        
        # Create a quick test script
        python3 - << 'EOF'
import cv2
import numpy as np
from tensorflow import keras
import os
import json

print("Loading ASL model...")
try:
    model = keras.models.load_model('trained_model/asl_best_model.h5')
    with open('trained_model/asl_model_metadata.json', 'r') as f:
        metadata = json.load(f)
    img_size = metadata['img_size']
    print(f"✓ Model loaded (image size: {img_size}x{img_size})")
except:
    print("ASL model not found, using old model...")
    model = keras.models.load_model('trained_model/best_model.h5')
    with open('trained_model/model_metadata.json', 'r') as f:
        metadata = json.load(f)
    img_size = metadata['img_size']
    print(f"✓ Model loaded (image size: {img_size}x{img_size})")

# Test on a few letters
test_letters = ['A', 'B', 'C', 'E', 'M', 'Z']
dataset_path = '../frontend/dataset'

print(f"\nTesting on {len(test_letters)} letters...")
print("-" * 50)

for letter in test_letters:
    letter_path = os.path.join(dataset_path, letter)
    if not os.path.exists(letter_path):
        print(f"⚠️  {letter}: No dataset found")
        continue
    
    img_files = [f for f in os.listdir(letter_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not img_files:
        print(f"⚠️  {letter}: No images found")
        continue
    
    img_path = os.path.join(letter_path, img_files[0])
    img = cv2.imread(img_path)
    
    # ASL-style preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    resized = cv2.resize(res, (img_size, img_size))
    normalized = resized.astype('float32') / 255.0
    final = np.expand_dims(np.expand_dims(normalized, axis=-1), axis=0)
    
    # Predict
    predictions = model.predict(final, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]
    
    classes = metadata.get('class_names', metadata.get('classes', []))
    predicted_letter = classes[predicted_idx]
    
    status = "✅" if predicted_letter == letter else "❌"
    print(f"{status} {letter}: Predicted {predicted_letter} ({confidence:.1%})")

print("-" * 50)
print("\n✓ Testing complete!")
EOF
        ;;
        
    5)
        echo ""
        echo "📊 Viewing training history..."
        
        if [ -f "training_history_asl.png" ]; then
            open training_history_asl.png 2>/dev/null || xdg-open training_history_asl.png 2>/dev/null
            echo "✓ Training history plot opened"
        else
            echo "⚠️  Training history not found. Train the model first (option 1)."
        fi
        ;;
        
    *)
        echo ""
        echo "❌ Invalid choice. Please run the script again and select 1-5."
        ;;
esac

echo ""
