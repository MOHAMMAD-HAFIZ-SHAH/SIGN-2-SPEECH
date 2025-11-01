# SIGN-2-SPEECH 🤟 ➡️ 📝 ➡️ 🔊

A comprehensive React application for real-time sign language recognition, text-to-speech, speech-to-text, and emotion detection.

## 🌟 Features

### 1. **Sign Language to Text** (Real-time AI Detection)
- Uses custom-trained CNN model for sign recognition
- Real-time webcam detection with TensorFlow.js
- Converts hand signs to text in real-time
- Supports grayscale image training
- Customizable confidence thresholds

### 2. **Text to Speech**
- Convert written text to spoken audio
- Adjustable speech speed
- Audio playback controls (play, pause, stop)
- Download generated audio

### 3. **Speech to Text**
- Record voice input
- Real-time transcription
- Copy, download, and clear transcripts

### 4. **Emotion Detection**
- Real-time facial emotion recognition
- Confidence scores for multiple emotions
- Visual feedback with emoji representations

## 🚀 Quick Start

### Prerequisites
- Node.js (v14+)
- Python 3.8+ (for model training)
- Webcam (for camera features)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/MOHAMMAD-HAFIZ-SHAH/SIGN-2-SPEECH.git
cd SIGN-2-SPEECH
```

2. **Install frontend dependencies:**
```bash
cd frontend
npm install
```

3. **Run the application:**
```bash
npm start
```

The app will open at `http://localhost:3000`

## 🧠 Setting Up Sign Language Recognition

### Step 1: Prepare Your Dataset

Organize your grayscale hand sign images:

```
dataset/
  A/
    image1.jpg
    image2.jpg
    ...
  B/
    image1.jpg
    ...
  (one folder per sign/letter)
```

**Tips:**
- 500+ images per sign recommended
- Include variety (different hands, lighting, angles)
- Grayscale images work best

### Step 2: Train the Model

```bash
cd model-training
pip install -r requirements.txt
python train_sign_model.py /path/to/your/dataset
```

Training takes 10-30 minutes depending on dataset size.

### Step 3: Deploy Model

```bash
# Copy trained model to frontend
mkdir -p ../frontend/public/models
cp -r trained_model/tfjs_model ../frontend/public/models/
cp trained_model/model_metadata.json ../frontend/public/models/
```

### Step 4: Test It!

1. Start the frontend app (`npm start`)
2. Navigate to "Sign to Text" tab
3. Click "Start Camera"
4. Show hand signs to the camera
5. Watch real-time detection! ✨

## 📁 Project Structure

```
SIGN-2-SPEECH/
├── frontend/                      # React application
│   ├── public/
│   │   └── models/               # Trained model files (after training)
│   │       ├── tfjs_model/
│   │       └── model_metadata.json
│   ├── src/
│   │   ├── components/           # Feature components
│   │   │   ├── SignToText.js    # Real-time sign detection
│   │   │   ├── TextToSpeech.js
│   │   │   ├── SpeechToText.js
│   │   │   └── EmotionDetection.js
│   │   ├── hooks/
│   │   │   └── useBackendIntegration.js
│   │   ├── utils/
│   │   │   └── signLanguageModel.js  # TensorFlow.js model loader
│   │   └── App.js
│   └── package.json
├── model-training/               # Python training scripts
│   ├── train_sign_model.py      # Main training script
│   ├── collect_data.py          # Data collection helper
│   ├── test_model.py            # Model testing script
│   ├── requirements.txt
│   └── README.md
├── SIGN_LANGUAGE_SETUP.md       # Detailed setup guide
└── README.md                     # This file
```

## 🛠️ Technology Stack

### Frontend
- **React 19** - UI framework
- **TensorFlow.js** - Browser-based ML inference
- **Tailwind CSS** - Styling
- **Lucide React** - Icons

### Model Training
- **TensorFlow/Keras** - Model training
- **OpenCV** - Image processing
- **NumPy** - Numerical operations
- **scikit-learn** - Data splitting

## 📖 Detailed Documentation

- **[Complete Setup Guide](SIGN_LANGUAGE_SETUP.md)** - Step-by-step instructions
- **[Model Training Guide](model-training/README.md)** - Training details
- **[Frontend README](frontend/README.md)** - React app info

## 🎯 How Sign Detection Works

```
Webcam Feed (30 FPS)
    ↓
Frame Sampling (every 500ms)
    ↓
Preprocessing (grayscale, resize to 64x64, normalize)
    ↓
CNN Model Prediction
    ↓
Confidence Check (threshold: 0.6)
    ↓
Sign Stabilization (3 consecutive detections)
    ↓
Add to Text Output
```

## 🔧 Configuration & Customization

### Adjust Detection Speed
In `SignToText.js`, change the interval:
```javascript
setInterval(async () => {
  // prediction code
}, 500);  // milliseconds between predictions
```

### Change Confidence Threshold
```javascript
const prediction = await signLanguageModel.predict(
  videoRef.current, 
  0.6  // 0.0 to 1.0 (higher = more strict)
);
```

### Modify Sign Stabilization
```javascript
if (signStabilityCounterRef.current === 3) {  // consecutive detections needed
  // add to text
}
```

## 🧪 Testing Your Model

```bash
cd model-training

# Test with webcam
python test_model.py

# Choose option 1 for live webcam testing
```

## 📊 Model Performance

Expected accuracy depends on:
- **Dataset size**: 500+ images per sign → 85-90% accuracy
- **Dataset quality**: Clear, varied images → better results
- **Number of classes**: Fewer signs → higher accuracy
- **Training epochs**: 50+ epochs recommended

## 🐛 Troubleshooting

### Model Not Loading
- Verify files exist: `frontend/public/models/tfjs_model/model.json`
- Check browser console for errors
- Ensure training completed successfully

### Low Detection Accuracy
- Add more training images (500+ per sign)
- Improve lighting conditions
- Use plain backgrounds
- Retrain with more epochs

### Camera Not Working
- Grant camera permissions in browser
- Use HTTPS or localhost
- Check if camera is used by another app

### Slow Performance
- Close other tabs/applications
- Reduce prediction frequency
- Check GPU acceleration availability

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- TensorFlow.js team for browser ML capabilities
- Create React App for the foundation
- Sign language community for inspiration

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Made with ❤️ for accessibility and inclusion**

🤟 Happy Signing! 🤟
