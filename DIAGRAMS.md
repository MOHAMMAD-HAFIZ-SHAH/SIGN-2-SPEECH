# 📊 Visual Workflow Diagrams

## 🔄 Complete Workflow Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     COMPLETE SIGN-2-SPEECH WORKFLOW                  │
└─────────────────────────────────────────────────────────────────────┘

PHASE 1: DATA PREPARATION
═══════════════════════════
┌──────────────┐
│ Google Drive │ (Your grayscale images)
└──────┬───────┘
       │ Download
       ▼
┌──────────────┐
│ Organize     │  Create folders: A/, B/, C/, ...
│ Dataset      │  500+ images per sign recommended
└──────┬───────┘
       │
       │

PHASE 2: MODEL TRAINING (Python)
═════════════════════════════════
       │
       ▼
┌──────────────┐
│ Load Images  │  • Read from folders
│              │  • Convert to grayscale
│              │  • Resize to 64x64
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Preprocess   │  • Normalize [0,1]
│              │  • Split train/val 80/20
│              │  • Data augmentation
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Train CNN    │  • 50 epochs
│              │  • Early stopping
│              │  • Save best model
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Export       │  • TensorFlow.js format
│              │  • Metadata JSON
└──────┬───────┘
       │
       │

PHASE 3: DEPLOYMENT
════════════════════
       │
       ▼
┌──────────────┐
│ Copy to      │  cp trained_model/tfjs_model/
│ Frontend     │     → frontend/public/models/
└──────┬───────┘
       │
       │

PHASE 4: REAL-TIME INFERENCE (JavaScript)
═══════════════════════════════════════════
       │
       ▼
┌──────────────┐
│ Start React  │  npm start
│ App          │  → http://localhost:3000
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Load Model   │  TensorFlow.js loads model
│              │  from /public/models/
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Start Camera │  getUserMedia() API
│              │  Video stream at 30 FPS
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Real-Time    │  Every 500ms:
│ Detection    │  1. Capture frame
│              │  2. Preprocess
│              │  3. Predict
│              │  4. Display result
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Text Output  │  Detected signs → Text
│              │  Display on screen
└──────────────┘
```

## 🎯 Training Pipeline Detail

```
INPUT: Organized Dataset
═══════════════════════════
dataset/
├── A/ (500 images)
├── B/ (500 images)
└── ...

        │
        ▼

STEP 1: Data Loading
═════════════════════
for each folder:
  for each image:
    • Read image
    • Check if valid
    • Store with label

Images:  [img1, img2, ..., imgN]
Labels:  [0, 0, ..., 1, 1, ..., 2]
                     │
                     ▼

STEP 2: Preprocessing
═══════════════════════
• Convert to grayscale (if needed)
• Resize to 64x64
• Normalize to [0, 1]
• Add channel dimension

Shape: (N, 64, 64, 1)
                     │
                     ▼

STEP 3: Train/Val Split
═════════════════════════
Train (80%): [========]
Val   (20%): [==]
                     │
                     ▼

STEP 4: Model Architecture
════════════════════════════
Input (64, 64, 1)
        ↓
Conv Block 1 (32 filters)
→ Conv → BatchNorm → MaxPool → Dropout
        ↓
Conv Block 2 (64 filters)
→ Conv → BatchNorm → MaxPool → Dropout
        ↓
Conv Block 3 (128 filters)
→ Conv → BatchNorm → MaxPool → Dropout
        ↓
Flatten
        ↓
Dense (256) → BatchNorm → Dropout
        ↓
Dense (128) → BatchNorm → Dropout
        ↓
Dense (N classes) → Softmax
        ↓
Output: Probabilities [P(A), P(B), ...]
                     │
                     ▼

STEP 5: Training Loop
═══════════════════════
for epoch in 1..50:
  • Forward pass
  • Calculate loss
  • Backward pass
  • Update weights
  • Validate
  • Save if best
                     │
                     ▼

STEP 6: Export
═══════════════
• Save as .h5 (Keras)
• Convert to TensorFlow.js
• Generate metadata
                     │
                     ▼

OUTPUT: Trained Model
══════════════════════
trained_model/
├── sign_language_model.h5
├── tfjs_model/
│   ├── model.json
│   └── *.bin
└── model_metadata.json
```

## 🔍 Inference Pipeline Detail

```
INPUT: Live Webcam Feed
════════════════════════
<video> element (640x480, 30 FPS)
        │
        ▼

STEP 1: Frame Sampling
═══════════════════════
setInterval(500ms) → 2 FPS
Capture current frame
        │
        ▼

STEP 2: Preprocessing
══════════════════════
tf.browser.fromPixels(video)
        ↓
Convert to grayscale:
  gray = (R + G + B) / 3
        ↓
Resize to 64x64:
  tf.image.resizeBilinear()
        ↓
Normalize [0, 1]:
  pixel / 255.0
        ↓
Reshape:
  (1, 64, 64, 1)
        │
        ▼

STEP 3: Model Prediction
═════════════════════════
model.predict(preprocessed)
        ↓
Returns: [0.05, 0.02, 0.89, ...]
                │
                ▼

STEP 4: Confidence Check
═════════════════════════
max_confidence = max(predictions)
predicted_index = argmax(predictions)
predicted_sign = class_names[predicted_index]

if max_confidence >= 0.6:
  ✓ Accept prediction
else:
  ✗ Reject (too uncertain)
        │
        ▼

STEP 5: Sign Stabilization
════════════════════════════
if predicted_sign == last_sign:
  counter++
  if counter >= 3:
    ✓ Add to text output
    counter = 0
else:
  last_sign = predicted_sign
  counter = 1
        │
        ▼

STEP 6: Display Result
═══════════════════════
Update UI:
• Current sign label
• Confidence percentage
• Append to text output
        │
        ▼

LOOP BACK TO STEP 1
```

## 🎨 State Management Flow

```
Component Lifecycle
════════════════════

Mount
  ↓
useEffect(() => {
  loadModel()
}, [])
  ↓
Model Loading...
  ↓
Model Loaded ✓
  ↓
User clicks "Start Camera"
  ↓
handleStartCamera()
  ↓
getUserMedia()
  ↓
Camera Active ✓
  ↓
startRealTimeDetection()
  ↓
setInterval → Prediction Loop
  ↓
Update State:
• currentSign
• confidence
• detectedText
  ↓
UI Re-renders
  ↓
Display Results
  ↓
User clicks "Stop Camera"
  ↓
handleStopCamera()
  ↓
Stop stream
Clear interval
  ↓
Camera Inactive
  ↓
Unmount
  ↓
Cleanup (stop camera, dispose model)
```

## 🧩 Component Interaction

```
┌─────────────────────────────────────────┐
│              App.js                      │
│  ┌─────────────────────────────────┐   │
│  │     Navigation Tabs              │   │
│  │  [Sign2Text][TTS][STT][Emotion] │   │
│  └──────────────┬──────────────────┘   │
│                 │                        │
│                 ▼                        │
│  ┌─────────────────────────────────┐   │
│  │     SignToText Component         │   │
│  │                                   │   │
│  │  ┌─────────────────────────┐    │   │
│  │  │  TitleIconContainer     │    │   │
│  │  └─────────────────────────┘    │   │
│  │                                   │   │
│  │  ┌─────────────────────────┐    │   │
│  │  │  Model Status           │    │   │
│  │  │  [Loading.../Loaded ✓] │    │   │
│  │  └─────────────────────────┘    │   │
│  │                                   │   │
│  │  ┌─────────────────────────┐    │   │
│  │  │  <video> + <canvas>     │    │   │
│  │  │  Camera Feed + Overlay  │    │   │
│  │  │                          │    │   │
│  │  │  ┌──────────────┐       │    │   │
│  │  │  │  Sign: A     │       │    │   │
│  │  │  │  Conf: 92%   │       │    │   │
│  │  │  └──────────────┘       │    │   │
│  │  └─────────────────────────┘    │   │
│  │                                   │   │
│  │  ┌─────────────────────────┐    │   │
│  │  │  PrimaryButton          │    │   │
│  │  │  [Start/Stop Camera]    │    │   │
│  │  └─────────────────────────┘    │   │
│  │                                   │   │
│  │  ┌─────────────────────────┐    │   │
│  │  │  Text Output            │    │   │
│  │  │  "HELLO WORLD"          │    │   │
│  │  └─────────────────────────┘    │   │
│  │                                   │   │
│  │  ┌─────────────────────────┐    │   │
│  │  │  Instructions Panel     │    │   │
│  │  └─────────────────────────┘    │   │
│  └───────────────┬───────────────┘   │
└──────────────────┼─────────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ signLanguageModel.js │
        │ (Utility)            │
        │                      │
        │ • loadModel()        │
        │ • predict()          │
        │ • preprocess()       │
        └──────────────────────┘
```

## 📊 Data Flow Diagram

```
User Action          Component State         External System
═══════════          ════════════════        ════════════════

Click "Start"  →  isCameraActive = true
                         ↓
                  getUserMedia()
                         ↓
                  streamRef.current ← ─── Browser Camera API
                         ↓
                  videoRef.srcObject
                         ↓
                  startRealTimeDetection()
                         ↓
                  setInterval(500ms)
                         │
                         ├─→ Capture frame
                         │   from videoRef
                         │
                         ├─→ preprocess() ─ ─ ─→ TensorFlow.js
                         │         ↓
                         │   Tensor(1,64,64,1)
                         │
                         ├─→ model.predict() ─ ─→ TensorFlow.js
                         │         ↓                  Model
                         │   predictions[]
                         │
                         ├─→ Check confidence
                         │         ↓
                         │   if > 0.6:
                         │     ├─→ currentSign
                         │     ├─→ confidence
                         │     └─→ stabilize
                         │             ↓
                         └──→ detectedText ─→ UI Update
                                     ↓
Show Sign         Display on Screen
on Camera         with Confidence
```

## 🔄 Continuous Detection Loop

```
Time: 0ms          500ms         1000ms        1500ms
      │             │             │             │
      ▼             ▼             ▼             ▼
  ┌───────┐     ┌───────┐     ┌───────┐     ┌───────┐
  │Capture│     │Capture│     │Capture│     │Capture│
  │Frame 1│     │Frame 2│     │Frame 3│     │Frame 4│
  └───┬───┘     └───┬───┘     └───┬───┘     └───┬───┘
      │             │             │             │
      ▼             ▼             ▼             ▼
  ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
  │Preproc.│    │Preproc.│    │Preproc.│    │Preproc.│
  └───┬────┘    └───┬────┘    └───┬────┘    └───┬────┘
      │             │             │             │
      ▼             ▼             ▼             ▼
  ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
  │Predict │    │Predict │    │Predict │    │Predict │
  │  → A   │    │  → A   │    │  → A   │    │  → B   │
  │ 85%    │    │ 88%    │    │ 91%    │    │ 78%    │
  └───┬────┘    └───┬────┘    └───┬────┘    └───┬────┘
      │             │             │             │
      ▼             ▼             ▼             ▼
  Counter=1       Counter=2     Counter=3     Counter=1
  (wait)          (wait)        ✓ Output "A"  (wait)
                                to text       new sign
```

---

These diagrams should help you visualize the complete workflow from start to finish!
