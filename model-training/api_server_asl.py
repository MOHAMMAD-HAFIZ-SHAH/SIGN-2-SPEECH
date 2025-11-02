"""
Flask API Server for ASL Sign Language Recognition
Uses ASL-style preprocessing to match the training pipeline
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
from tensorflow import keras
import json
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global variables for model
model = None
metadata = None

# Preprocessing constants (matching train_asl_model.py)
MIN_THRESHOLD_VALUE = 70

def load_model():
    """Load the trained model and metadata"""
    global model, metadata
    
    try:
        print("Loading ASL sign language model...")
        
        # Try to load the new ASL model first, fall back to old model
        model_path = './trained_model/asl_best_model.h5'
        metadata_path = './trained_model/asl_model_metadata.json'
        
        if not os.path.exists(model_path):
            print("ASL model not found, using previous model...")
            model_path = './trained_model/best_model.h5'
            metadata_path = './trained_model/model_metadata.json'
        
        # Load model
        model = keras.models.load_model(model_path)
        print(f"✓ Model loaded from {model_path}")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Normalize metadata structure
        if 'class_names' not in metadata and 'classes' in metadata:
            metadata['class_names'] = metadata['classes']
        
        print(f"✓ Metadata loaded: {len(metadata['class_names'])} classes")
        print(f"✓ Image size: {metadata['img_size']}x{metadata['img_size']}")
        
        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def preprocess_image_asl_style(img, img_size, min_value=MIN_THRESHOLD_VALUE):
    """
    ASL-style preprocessing pipeline (matching train_asl_model.py):
    1. Convert to grayscale
    2. Gaussian blur
    3. Adaptive thresholding
    4. Otsu's method
    5. Resize to target size
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 2)
        
        # Apply adaptive thresholding
        th3 = cv2.adaptiveThreshold(
            blur, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Apply Otsu's thresholding
        ret, res = cv2.threshold(
            th3, min_value, 255, 
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        # Resize to target size
        resized = cv2.resize(res, (img_size, img_size))
        
        return resized
    except Exception as e:
        print(f'Exception in preprocessing: {e}')
        return None

def preprocess_image(image_data):
    """Preprocess image using ASL-style pipeline"""
    try:
        # Decode base64 image
        img_bytes = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            print("Failed to decode image")
            return None
        
        # Get image size from metadata
        img_size = metadata['img_size']
        
        # Apply ASL-style preprocessing
        processed = preprocess_image_asl_style(img, img_size, MIN_THRESHOLD_VALUE)
        
        if processed is None:
            return None
        
        # Normalize to [0, 1]
        normalized = processed.astype('float32') / 255.0
        
        # Add batch and channel dimensions
        final = np.expand_dims(normalized, axis=-1)
        final = np.expand_dims(final, axis=0)
        
        return final
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes': len(metadata['class_names']) if metadata else 0,
        'preprocessing': 'ASL-style (adaptive threshold + Otsu)'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Get image data from request
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Preprocess image
        processed_image = preprocess_image(data['image'])
        if processed_image is None:
            return jsonify({'error': 'Failed to process image'}), 400
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get class name
        predicted_sign = metadata['class_names'][predicted_class_idx]
        
        # Get top 5 predictions for debugging
        top_5_indices = np.argsort(predictions[0])[-5:][::-1]
        top_5_predictions = [
            {
                'sign': metadata['class_names'][idx],
                'confidence': float(predictions[0][idx])
            }
            for idx in top_5_indices
        ]
        
        # Debug logging
        print(f"Prediction: {predicted_sign} ({confidence:.2%})")
        print(f"Top 3: {', '.join([f'{p['sign']}({p['confidence']:.2%})' for p in top_5_predictions[:3]])}")
        
        # Return prediction
        confidence_threshold = data.get('threshold', 0.5)
        
        if confidence >= confidence_threshold:
            return jsonify({
                'success': True,
                'sign': predicted_sign,
                'confidence': confidence,
                'top_predictions': top_5_predictions
            })
        else:
            return jsonify({
                'success': True,
                'sign': None,
                'confidence': confidence,
                'message': f'Confidence ({confidence:.2f}) below threshold ({confidence_threshold})',
                'top_predictions': top_5_predictions
            })
            
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get list of all classes"""
    if metadata:
        return jsonify({
            'classes': metadata['class_names'],
            'num_classes': len(metadata['class_names']),
            'img_size': metadata['img_size']
        })
    return jsonify({'error': 'Model not loaded'}), 500

if __name__ == '__main__':
    print("="*60)
    print("ASL Sign Language Recognition API Server")
    print("="*60)
    
    # Load model
    if load_model():
        print("\n✓ Server ready!")
        print(f"✓ Serving {len(metadata['class_names'])} sign language classes")
        if 'test_accuracy' in metadata:
            print(f"✓ Model accuracy: {metadata['test_accuracy']*100:.2f}%")
        print(f"✓ Preprocessing: ASL-style (adaptive threshold + Otsu)")
        print("\nStarting Flask server on http://localhost:5001")
        print("API Endpoints:")
        print("  - GET  /health  - Health check")
        print("  - POST /predict - Make prediction")
        print("  - GET  /classes - Get list of classes")
        print("\n" + "="*60)
        
        # Start server on port 5001 (port 5000 is used by AirPlay on macOS)
        app.run(host='0.0.0.0', port=5001, debug=False)
    else:
        print("\n✗ Failed to load model. Server not started.")
