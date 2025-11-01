"""
Convert trained Keras model to TensorFlow.js format
"""
import tensorflow as tf
from tensorflow import keras
import json
import os

# Load the trained model
model_path = './trained_model/best_model.h5'
output_dir = './trained_model/tfjs_model'

print("Loading model...")
model = keras.models.load_model(model_path)

print("Model loaded successfully!")
print("\nModel Summary:")
model.summary()

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Save in TensorFlow SavedModel format first
saved_model_dir = './trained_model/saved_model'
print(f"\nSaving as TensorFlow SavedModel to {saved_model_dir}...")
tf.saved_model.save(model, saved_model_dir)

# Convert to TensorFlow.js format using Python API
print(f"\nConverting to TensorFlow.js format...")
import tensorflowjs as tfjs

try:
    tfjs.converters.save_keras_model(model, output_dir)
    print(f"✓ Successfully converted model to TensorFlow.js format!")
    print(f"✓ Model saved to: {output_dir}")
    print(f"\nFiles created:")
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        size = os.path.getsize(file_path)
        print(f"  - {file} ({size:,} bytes)")
except Exception as e:
    print(f"Error during conversion: {e}")
    print("\nTrying alternative method...")
    
    # Alternative: Save as SavedModel and use converter
    os.system(f'tensorflowjs_converter --input_format=keras_saved_model {saved_model_dir} {output_dir}')

print("\n" + "="*60)
print("Conversion complete!")
print("="*60)
print(f"\nTo use in your React app:")
print(f"1. Copy the '{os.path.basename(output_dir)}' folder to 'frontend/public/models/'")
print(f"2. Copy 'model_metadata.json' to 'frontend/public/models/'")
print(f"3. Load the model in your React app from '/models/tfjs_model/model.json'")
