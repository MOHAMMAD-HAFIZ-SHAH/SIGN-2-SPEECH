"""
Sign Language Recognition Model Training Script
Trains a CNN model on grayscale hand sign images for real-time recognition
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import cv2
import json

# Configuration
IMG_SIZE = 64  # Resize images to 64x64
BATCH_SIZE = 32
EPOCHS = 50
MODEL_NAME = 'sign_language_model'

class SignLanguageDataLoader:
    """Load and preprocess sign language images from directory structure"""
    
    def __init__(self, data_dir, img_size=64):
        """
        Args:
            data_dir: Path to dataset directory with structure:
                      data_dir/
                          A/
                              img1.jpg
                              img2.jpg
                          B/
                              img1.jpg
                          ...
            img_size: Target size for images (will be resized to img_size x img_size)
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.class_names = []
        
    def load_dataset(self):
        """Load all images and labels from directory"""
        images = []
        labels = []
        
        # Get all subdirectories (each represents a sign/letter)
        self.class_names = sorted([d for d in os.listdir(self.data_dir) 
                                   if os.path.isdir(os.path.join(self.data_dir, d))])
        
        print(f"Found {len(self.class_names)} classes: {self.class_names}")
        
        for label_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.data_dir, class_name)
            print(f"Loading class '{class_name}'...")
            
            # Get all image files in this class directory
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                
                # Read image in grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    print(f"Warning: Could not read {img_path}")
                    continue
                
                # Resize to target size
                img = cv2.resize(img, (self.img_size, self.img_size))
                
                # Normalize pixel values to [0, 1]
                img = img.astype('float32') / 255.0
                
                images.append(img)
                labels.append(label_idx)
            
            print(f"  Loaded {len(image_files)} images for class '{class_name}'")
        
        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)
        
        # Add channel dimension (required for CNN)
        images = np.expand_dims(images, axis=-1)
        
        print(f"\nTotal dataset: {len(images)} images")
        print(f"Image shape: {images.shape}")
        print(f"Label shape: {labels.shape}")
        
        return images, labels
    
    def get_class_names(self):
        """Return list of class names"""
        return self.class_names


def create_cnn_model(input_shape, num_classes):
    """
    Create a CNN model for sign language recognition
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(data_dir, output_dir='./trained_model'):
    """
    Complete training pipeline
    
    Args:
        data_dir: Path to dataset directory
        output_dir: Directory to save trained model
    """
    print("="*60)
    print("Sign Language Model Training")
    print("="*60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print("\n1. Loading dataset...")
    data_loader = SignLanguageDataLoader(data_dir, img_size=IMG_SIZE)
    X, y = data_loader.load_dataset()
    class_names = data_loader.get_class_names()
    
    # Split into train and validation sets
    print("\n2. Splitting dataset...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    
    # Create model
    print("\n3. Creating model...")
    input_shape = (IMG_SIZE, IMG_SIZE, 1)
    num_classes = len(class_names)
    model = create_cnn_model(input_shape, num_classes)
    
    print(f"\nModel architecture:")
    model.summary()
    
    # Data augmentation for training
    data_augmentation = keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ])
    
    # Apply augmentation only to training data
    X_train_aug = data_augmentation(X_train, training=True)
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print("\n4. Training model...")
    history = model.fit(
        X_train_aug, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on validation set
    print("\n5. Evaluating model...")
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    # Save model in TensorFlow.js format
    print("\n6. Saving model...")
    
    # Save as Keras model
    model.save(os.path.join(output_dir, f'{MODEL_NAME}.h5'))
    print(f"Saved Keras model to {output_dir}/{MODEL_NAME}.h5")
    
    # Convert to TensorFlow.js format
    import tensorflowjs as tfjs
    tfjs_dir = os.path.join(output_dir, 'tfjs_model')
    tfjs.converters.save_keras_model(model, tfjs_dir)
    print(f"Saved TensorFlow.js model to {tfjs_dir}")
    
    # Save class names and metadata
    metadata = {
        'class_names': class_names,
        'img_size': IMG_SIZE,
        'num_classes': num_classes,
        'input_shape': [IMG_SIZE, IMG_SIZE, 1],
        'val_accuracy': float(val_accuracy),
        'val_loss': float(val_loss)
    }
    
    with open(os.path.join(output_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {output_dir}/model_metadata.json")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print(f"\nTo use this model in your React app:")
    print(f"1. Copy the 'tfjs_model' folder to 'frontend/public/models/'")
    print(f"2. Copy 'model_metadata.json' to 'frontend/public/models/'")
    print(f"3. The model will be loaded at runtime from '/models/tfjs_model/model.json'")
    
    return model, history, metadata


if __name__ == '__main__':
    import sys
    
    # Get dataset path from command line or use default
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        print("Usage: python train_sign_model.py <path_to_dataset>")
        print("\nDataset should be organized as:")
        print("  dataset/")
        print("    A/")
        print("      img1.jpg")
        print("      img2.jpg")
        print("    B/")
        print("      img1.jpg")
        print("    ...")
        sys.exit(1)
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist!")
        sys.exit(1)
    
    # Train model
    model, history, metadata = train_model(dataset_path)
