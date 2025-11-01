"""
Simple model converter that saves in Keras v3 format then converts to TFJS
"""
import os
import subprocess
import tensorflow as tf
from tensorflow import keras

print("="*60)
print("Converting Model to TensorFlow.js Format")
print("="*60)

# Paths
model_path = './trained_model/best_model.h5'
keras_model_path = './trained_model/model.keras'
tfjs_output_dir = './trained_model/tfjs_model'

# Step 1: Load the H5 model
print("\n1. Loading H5 model...")
model = keras.models.load_model(model_path)
print("✓ Model loaded successfully")

# Step 2: Save as .keras format (Keras v3 native format)
print(f"\n2. Saving as Keras v3 format to {keras_model_path}...")
model.export(keras_model_path)
print("✓ Model saved in Keras format")

# Step 3: Convert using command line tool
print(f"\n3. Converting to TensorFlow.js format...")
print(f"   Output directory: {tfjs_output_dir}")

# Create output directory
os.makedirs(tfjs_output_dir, exist_ok=True)

# Use the command line converter
cmd = f'tensorflowjs_converter --input_format=keras {keras_model_path} {tfjs_output_dir}'
print(f"   Running: {cmd}")

try:
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ Conversion successful!")
    else:
        print("✗ Conversion failed:")
        print(result.stderr)
        # Try alternative method
        print("\nTrying alternative conversion method...")
        # Save weights separately
        model.save_weights('./trained_model/model_weights.h5')
        print("✓ Weights saved, manual conversion needed")
except Exception as e:
    print(f"Error: {e}")

# Step 4: Verify output files
print(f"\n4. Verifying output files...")
if os.path.exists(tfjs_output_dir):
    files = os.listdir(tfjs_output_dir)
    if files:
        print(f"✓ Found {len(files)} files in output directory:")
        for f in files:
            size = os.path.getsize(os.path.join(tfjs_output_dir, f))
            print(f"   - {f} ({size:,} bytes)")
    else:
        print("✗ Output directory is empty")
else:
    print("✗ Output directory not created")

print("\n" + "="*60)
print("Conversion Complete!")
print("="*60)
print("\nNext steps:")
print("1. Copy tfjs_model/ folder to frontend/public/models/")
print("2. Copy model_metadata.json to frontend/public/models/")
print("3. Your React app should now be able to load the model!")
