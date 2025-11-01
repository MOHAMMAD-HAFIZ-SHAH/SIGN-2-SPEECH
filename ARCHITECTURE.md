# System Architecture - SIGN-2-SPEECH

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         SIGN-2-SPEECH                            │
│                    Multi-Feature AI System                       │
└─────────────────────────────────────────────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
┌──────────────┐        ┌──────────────┐        ┌──────────────┐
│ Sign→Text    │        │ Text→Speech  │        │ Speech→Text  │
│ (AI Model)   │        │ (TTS API)    │        │ (STT API)    │
└──────────────┘        └──────────────┘        └──────────────┘
```

## Sign Language to Text - Detailed Architecture

### 1. Training Phase (Python - Offline)

```
┌──────────────────────────────────────────────────────────────┐
│                      TRAINING PIPELINE                        │
└──────────────────────────────────────────────────────────────┘

Google Drive Images
        │
        ▼
┌─────────────────┐
│ Dataset Folder  │  A/, B/, C/, ... (grayscale images)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ train_sign_     │  • Load images
│ model.py        │  • Convert to grayscale
│                 │  • Resize to 64x64
│                 │  • Normalize [0,1]
│                 │  • Split train/val (80/20)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ CNN Model       │  • 3 Conv blocks
│ Architecture    │  • BatchNorm + Dropout
│                 │  • Dense layers
│                 │  • Softmax output
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Training        │  • Data augmentation
│                 │  • 50 epochs
│                 │  • Early stopping
│                 │  • Model checkpoints
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Export          │  • sign_language_model.h5 (Keras)
│                 │  • tfjs_model/ (TensorFlow.js)
│                 │  • model_metadata.json
└─────────────────┘
```

### 2. Inference Phase (JavaScript - Real-time)

```
┌──────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE                         │
└──────────────────────────────────────────────────────────────┘

Webcam Feed (30 FPS)
        │
        ▼
┌─────────────────┐
│ Video Element   │  React useRef
│ <video>         │  640x480 resolution
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Frame Sampling  │  Every 500ms (2 FPS)
│ setInterval     │  Balance speed/accuracy
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Preprocessing   │  1. tf.browser.fromPixels()
│ TensorFlow.js   │  2. Convert to grayscale
│                 │  3. Resize to 64x64
│                 │  4. Normalize [0,1]
│                 │  5. Add batch dimension
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model.predict() │  Forward pass through CNN
│                 │  Returns probability array
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Confidence      │  Check if max_prob > threshold (0.6)
│ Check           │  Filter low-confidence predictions
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Sign            │  Require 3 consecutive detections
│ Stabilization   │  Prevent flickering
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Text Output     │  Append to transcription
│                 │  Display on UI
└─────────────────┘
```

## Component Architecture

### Frontend Component Hierarchy

```
App.js (Main Router)
    │
    ├── Navigation Bar
    │   ├── Sign to Text (tab)
    │   ├── Text to Speech (tab)
    │   ├── Speech to Text (tab)
    │   └── Emotion Detection (tab)
    │
    └── Content Area
        │
        ├── SignToText Component
        │   ├── TitleIconContainer
        │   ├── Model Loading Status
        │   ├── Video Feed
        │   │   ├── <video> element
        │   │   ├── <canvas> overlay
        │   │   └── Detection indicator
        │   ├── PrimaryButton (Start/Stop)
        │   ├── Transcription Area
        │   └── Instructions
        │
        ├── TextToSpeech Component
        │   ├── TitleIconContainer
        │   ├── <textarea> input
        │   ├── Speed slider
        │   ├── PrimaryButton (Speak)
        │   └── Audio controls
        │
        ├── SpeechToText Component
        │   └── [similar structure]
        │
        └── EmotionDetection Component
            └── [similar structure]
```

### Data Flow

```
┌──────────────────────────────────────────────────────────────┐
│                        STATE MANAGEMENT                       │
└──────────────────────────────────────────────────────────────┘

SignToText.js State:
    ├── isCameraActive (boolean)
    ├── detectedText (string)
    ├── currentSign (string | null)
    ├── confidence (number)
    ├── modelLoaded (boolean)
    ├── isModelLoading (boolean)
    └── cameraError (string | null)

Refs:
    ├── videoRef (video element)
    ├── canvasRef (canvas element)
    ├── streamRef (MediaStream)
    ├── predictionIntervalRef (interval ID)
    ├── lastDetectedSignRef (string)
    └── signStabilityCounterRef (number)

External Dependencies:
    └── signLanguageModel (singleton)
        ├── model (TensorFlow.js model)
        ├── metadata (class names, config)
        └── methods (loadModel, predict, preprocess)
```

## File System Structure

```
project/
├── frontend/                        # React App
│   ├── public/
│   │   └── models/                 # Model files (after training)
│   │       ├── tfjs_model/
│   │       │   ├── model.json      # Model architecture
│   │       │   └── *.bin           # Model weights
│   │       └── model_metadata.json # Class names, config
│   ├── src/
│   │   ├── components/
│   │   │   ├── SignToText.js      # Main component
│   │   │   ├── PrimaryButton.js    # Reusable button
│   │   │   └── TitleIconContainer.js
│   │   ├── hooks/
│   │   │   └── useBackendIntegration.js
│   │   ├── utils/
│   │   │   └── signLanguageModel.js # TF.js wrapper
│   │   └── App.js
│   └── package.json
│
└── model-training/                  # Python Scripts
    ├── train_sign_model.py         # Main training
    ├── collect_data.py             # Data collection
    ├── test_model.py               # Testing
    ├── requirements.txt            # Python deps
    └── trained_model/              # Output directory
        ├── sign_language_model.h5
        ├── tfjs_model/
        └── model_metadata.json
```

## Technology Stack Details

### Training Stack
```
Python 3.8+
    ├── TensorFlow 2.15
    ├── Keras (tf.keras)
    ├── OpenCV (image processing)
    ├── NumPy (arrays)
    ├── scikit-learn (train/val split)
    └── TensorFlow.js Converter
```

### Inference Stack
```
React 19
    ├── TensorFlow.js (ML inference)
    ├── Browser APIs
    │   ├── getUserMedia (webcam)
    │   ├── Canvas API (preprocessing)
    │   └── Web Audio API (future)
    ├── Tailwind CSS (styling)
    └── Lucide React (icons)
```

## Performance Characteristics

### Training Phase
- **Time:** 15-30 minutes (dataset dependent)
- **Memory:** ~2-4 GB RAM
- **Storage:** 2-10 MB (trained model)
- **GPU:** Optional (3-5x faster with CUDA)

### Inference Phase
- **Latency:** ~100-200ms per prediction
- **FPS:** 2 predictions per second
- **Memory:** ~50-100 MB (loaded model)
- **CPU:** ~10-20% utilization
- **Battery:** Moderate impact (webcam + ML)

## Scalability Considerations

### Horizontal Scaling
- Deploy model as API service
- Multiple users = multiple browser instances
- Each client runs inference locally

### Vertical Scaling
- Reduce model size (quantization)
- Optimize preprocessing
- Adjust prediction frequency
- Use WebGL backend for TF.js

### Model Updates
1. Retrain with new data
2. Export to TensorFlow.js
3. Replace files in public/models/
4. Users get update on next refresh

## Security & Privacy

### Data Handling
- ✅ All processing client-side (browser)
- ✅ No video uploaded to server
- ✅ Model runs locally
- ✅ No data collection
- ⚠️ Camera permissions required

### Best Practices
- Inform users about camera use
- Allow camera disable
- Clear data on session end
- Secure model files
- HTTPS for production

## Future Enhancements

### Possible Additions
1. **Sequence detection** - Recognize word sequences
2. **Hand tracking** - MediaPipe Hands integration
3. **Multi-hand support** - Two-handed signs
4. **Continuous recognition** - No stabilization delay
5. **Custom vocabulary** - User-defined signs
6. **Backend API** - Optional server processing
7. **Mobile support** - React Native version
8. **Offline mode** - PWA with model caching

---

This architecture provides a complete, scalable solution for real-time sign language recognition with excellent user experience and performance.
