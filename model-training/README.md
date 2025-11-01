# Sign Language Model Training

This directory contains scripts to train a CNN model for real-time sign language recognition.

## Dataset Structure

Organize your grayscale hand sign images in the following structure:

```
dataset/
  A/
    image1.jpg
    image2.jpg
    ...
  B/
    image1.jpg
    image2.jpg
    ...
  C/
    ...
  (continue for all letters/signs)
```

Each folder represents a sign/letter, and contains grayscale images of that hand sign.

## Setup

1. **Install Python dependencies:**

```bash
cd model-training
pip install -r requirements.txt
```

2. **Prepare your dataset:**
   - Download your images from Google Drive
   - Organize them in the folder structure shown above
   - Ensure all images are accessible

## Training

Run the training script with your dataset path:

```bash
python train_sign_model.py /path/to/your/dataset
```

**Example:**
```bash
python train_sign_model.py ./sign_language_dataset
```

The script will:
- Load all images and preprocess them
- Split into training (80%) and validation (20%) sets
- Train a CNN model with data augmentation
- Save the trained model in multiple formats
- Export to TensorFlow.js format for browser use

## Output

After training completes, you'll find:

```
trained_model/
  sign_language_model.h5          # Keras model (for Python use)
  model_metadata.json              # Model info and class names
  tfjs_model/                      # TensorFlow.js format (for React app)
    model.json
    group1-shard1of1.bin
```

## Deploy to React App

1. **Copy the trained model to your frontend:**

```bash
# Create models directory in frontend/public
mkdir -p ../frontend/public/models

# Copy TensorFlow.js model
cp -r trained_model/tfjs_model ../frontend/public/models/

# Copy metadata
cp trained_model/model_metadata.json ../frontend/public/models/
```

2. The React app will automatically load the model from `/models/tfjs_model/model.json`

## Model Architecture

- Input: 64x64 grayscale images
- 3 Convolutional blocks with BatchNorm and Dropout
- Dense layers with regularization
- Softmax output for classification

## Training Tips

- **More data = better results**: Aim for at least 500+ images per sign
- **Variety matters**: Include different lighting, backgrounds, hand positions
- **Balance classes**: Try to have similar number of images for each sign
- **Augmentation**: The script automatically applies rotation, zoom, and translation

## Troubleshooting

**Low accuracy?**
- Check if images are properly labeled
- Ensure enough training data per class
- Increase EPOCHS in the script
- Add more data augmentation

**Out of memory?**
- Reduce BATCH_SIZE in the script
- Reduce IMG_SIZE (currently 64)
- Use fewer images for testing

**Need to retrain?**
- Delete the `trained_model` folder
- Run the training script again
