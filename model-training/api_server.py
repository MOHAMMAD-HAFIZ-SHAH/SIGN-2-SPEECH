"""
Flask API Server for Sign Language Recognition
Provides REST API endpoint for real-time sign language prediction
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

def load_model():
    """Load the trained model and metadata"""
    global model, metadata
    
    try:
        print("Loading sign language model...")
        model_path = './trained_model/best_model.h5'
        metadata_path = './trained_model/model_metadata.json'
        
        # Load model
        model = keras.models.load_model(model_path)
        print("✓ Model loaded successfully")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"✓ Metadata loaded: {len(metadata['class_names'])} classes")
        
        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

def preprocess_image(image_data):
    """Preprocess image to match training data characteristics"""
    try:
        # Decode base64 image
        img_bytes = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to grayscale (same as training)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Training images have high brightness (mean ~232)
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Apply slight Gaussian blur to reduce noise (like JPEG compression)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Resize to model input size (training does this)
        img_size = metadata['img_size']
        resized = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1] (same as training)
        normalized = resized.astype('float32') / 255.0
        
        # Add batch and channel dimensions
        processed = np.expand_dims(normalized, axis=-1)
        processed = np.expand_dims(processed, axis=0)
        
        return processed
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes': len(metadata['class_names']) if metadata else 0
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
        confidence_threshold = data.get('threshold', 0.6)
        
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
    print("Sign Language Recognition API Server")
    print("="*60)
    
    # Load model
    if load_model():
        print("\n✓ Server ready!")
        print(f"✓ Serving {len(metadata['class_names'])} sign language classes")
        print(f"✓ Model accuracy: {metadata['val_accuracy']*100:.2f}%")
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
