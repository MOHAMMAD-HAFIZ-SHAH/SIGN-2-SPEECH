"""
ASL Sign Language Model Training Script
Integrates preprocessing from ASL_train.ipynb with improved architecture
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import json

# Configuration
IMG_SIZE = 128  # Using 128x128 as in ASL_train.ipynb
BATCH_SIZE = 32
EPOCHS = 50
MIN_THRESHOLD_VALUE = 70  # For Otsu's thresholding

# Dataset path - adjust based on your structure
DATASET_PATH = '../frontend/dataset'  # A-Z folders with images

def preprocess_image_asl_style(img, img_size=IMG_SIZE, min_value=MIN_THRESHOLD_VALUE):
    """
    Preprocessing pipeline from ASL_train.ipynb:
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

def load_dataset(data_path, img_size=IMG_SIZE):
    """Load and preprocess the dataset"""
    print("Loading dataset...")
    
    # Get all letter folders (A-Z)
    categories = sorted([d for d in os.listdir(data_path) 
                        if os.path.isdir(os.path.join(data_path, d)) and len(d) == 1])
    
    print(f"Found {len(categories)} categories: {categories}")
    
    # Create label dictionary
    label_dict = {category: idx for idx, category in enumerate(categories)}
    print(f"Label mapping: {label_dict}")
    
    data = []
    target = []
    
    for category in categories:
        cat_path = os.path.join(data_path, category)
        img_names = os.listdir(cat_path)
        
        print(f"Processing {category}: {len(img_names)} images")
        
        for img_name in img_names:
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(cat_path, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue
            
            # Apply ASL-style preprocessing
            processed = preprocess_image_asl_style(img, img_size)
            
            if processed is not None:
                data.append(processed)
                target.append(label_dict[category])
    
    print(f"\nTotal images loaded: {len(data)}")
    
    # Convert to numpy arrays
    data = np.array(data) / 255.0  # Normalize to [0, 1]
    data = np.reshape(data, (data.shape[0], img_size, img_size, 1))  # Add channel dimension
    target = np.array(target)
    
    # Convert labels to categorical (one-hot encoding)
    target_categorical = to_categorical(target, num_classes=len(categories))
    
    print(f"Data shape: {data.shape}")
    print(f"Target shape: {target_categorical.shape}")
    
    return data, target_categorical, categories, label_dict

def build_model(img_size=IMG_SIZE, num_classes=26):
    """
    Build CNN model based on ASL_train.ipynb architecture
    with some improvements
    """
    model = Sequential([
        # First convolution block
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Second convolution block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Third convolution block (additional for better feature extraction)
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten
        Flatten(),
        
        # Dense layers with dropout
        Dense(256, activation='relu'),
        Dropout(0.5),
        
        Dense(128, activation='relu'),
        Dropout(0.4),
        
        Dense(96, activation='relu'),
        Dropout(0.3),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history, save_path='training_history_asl.png'):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")
    plt.close()

def main():
    """Main training function"""
    print("=" * 60)
    print("ASL Sign Language Model Training")
    print("=" * 60)
    
    # Load dataset
    data, target, categories, label_dict = load_dataset(DATASET_PATH, IMG_SIZE)
    
    # Split data
    print("\nSplitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2, random_state=42, stratify=target
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Build model
    print("\nBuilding model...")
    model = build_model(img_size=IMG_SIZE, num_classes=len(categories))
    model.summary()
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        'trained_model/asl_best_model.h5',
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Save final model
    model.save('trained_model/asl_final_model.h5')
    print("\nFinal model saved to trained_model/asl_final_model.h5")
    
    # Save metadata
    metadata = {
        'classes': categories,
        'label_dict': label_dict,
        'img_size': IMG_SIZE,
        'num_classes': len(categories),
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'preprocessing': 'ASL-style (grayscale → blur → adaptive threshold → Otsu → resize)'
    }
    
    with open('trained_model/asl_model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("Metadata saved to trained_model/asl_model_metadata.json")
    
    # Plot training history
    plot_training_history(history)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)

if __name__ == '__main__':
    # Create output directory if it doesn't exist
    os.makedirs('trained_model', exist_ok=True)
    main()
