# Model Files Directory

This directory will contain your trained sign language recognition model.

## Required Files

After training your model (see `model-training/README.md`), copy these files here:

```
models/
├── tfjs_model/
│   ├── model.json
│   └── group1-shard1of1.bin
└── model_metadata.json
```

## How to Get Model Files

### Option 1: Train Your Own Model

1. Prepare your dataset of grayscale hand sign images
2. Navigate to `model-training/` directory
3. Install dependencies: `pip install -r requirements.txt`
4. Train: `python train_sign_model.py /path/to/dataset`
5. Copy files: 
   ```bash
   cp -r ../model-training/trained_model/tfjs_model ./
   cp ../model-training/trained_model/model_metadata.json ./
   ```

### Option 2: Use Pre-trained Model (if available)

If you have a pre-trained model:
1. Download the model files
2. Extract to this directory
3. Ensure the structure matches above

## Model Metadata Format

`model_metadata.json` should contain:
```json
{
  "class_names": ["A", "B", "C", ...],
  "img_size": 64,
  "num_classes": 26,
  "input_shape": [64, 64, 1],
  "val_accuracy": 0.92,
  "val_loss": 0.25
}
```

## Testing Model Loading

After placing files here:
1. Start the React app: `npm start`
2. Open browser console (F12)
3. Navigate to "Sign to Text" tab
4. Look for "Model loaded successfully" message

## Troubleshooting

**Model not found error:**
- Verify files are in correct location
- Check file permissions
- Ensure model.json is valid JSON
- Verify all shard files are present

**Model loading error:**
- Check browser console for specific error
- Ensure TensorFlow.js is installed
- Try clearing browser cache
- Verify model was exported correctly

## Model Size Considerations

- Typical model size: 2-10 MB
- Larger models = more accuracy but slower loading
- Consider compression for production deployment

## Next Steps

1. Train your model using the training scripts
2. Copy trained model files here
3. Refresh the React app
4. Test sign detection!

For detailed instructions, see:
- `SIGN_LANGUAGE_SETUP.md` (root directory)
- `model-training/README.md`
