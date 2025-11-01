/**
 * Sign Language Model Loader and Predictor
 * Handles loading TensorFlow.js model and making predictions on webcam frames
 */

import * as tf from '@tensorflow/tfjs';

class SignLanguageModel {
  constructor() {
    this.model = null;
    this.metadata = null;
    this.isLoaded = false;
  }

  /**
   * Load the trained model and metadata
   */
  async loadModel() {
    try {
      console.log('Loading sign language model...');
      
      // Load model
      this.model = await tf.loadLayersModel('/models/tfjs_model/model.json');
      console.log('Model loaded successfully');
      
      // Load metadata (class names, etc.)
      const metadataResponse = await fetch('/models/model_metadata.json');
      this.metadata = await metadataResponse.json();
      console.log('Metadata loaded:', this.metadata);
      
      // Warm up the model with a dummy prediction
      const dummyInput = tf.zeros([1, this.metadata.img_size, this.metadata.img_size, 1]);
      const dummyPrediction = this.model.predict(dummyInput);
      dummyPrediction.dispose();
      dummyInput.dispose();
      
      this.isLoaded = true;
      console.log('Model ready for predictions');
      
      return {
        success: true,
        classNames: this.metadata.class_names,
        imgSize: this.metadata.img_size
      };
    } catch (error) {
      console.error('Error loading model:', error);
      this.isLoaded = false;
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Preprocess video frame for model input
   * Converts to grayscale, resizes, and normalizes
   */
  preprocessFrame(videoElement) {
    return tf.tidy(() => {
      // Capture frame from video element
      const frame = tf.browser.fromPixels(videoElement);
      
      // Convert to grayscale (weighted average of RGB channels)
      const grayscale = frame.mean(2, true);
      
      // Resize to model input size
      const resized = tf.image.resizeBilinear(
        grayscale, 
        [this.metadata.img_size, this.metadata.img_size]
      );
      
      // Normalize to [0, 1]
      const normalized = resized.div(255.0);
      
      // Add batch and channel dimensions
      const batched = normalized.expandDims(0).expandDims(-1);
      
      return batched;
    });
  }

  /**
   * Make prediction on current video frame
   */
  async predict(videoElement, confidenceThreshold = 0.7) {
    if (!this.isLoaded || !this.model) {
      console.warn('Model not loaded yet');
      return null;
    }

    try {
      // Preprocess frame
      const preprocessed = this.preprocessFrame(videoElement);
      
      // Make prediction
      const predictions = this.model.predict(preprocessed);
      const predictionData = await predictions.data();
      
      // Clean up tensors
      preprocessed.dispose();
      predictions.dispose();
      
      // Find class with highest confidence
      const maxConfidence = Math.max(...predictionData);
      const predictedClassIndex = predictionData.indexOf(maxConfidence);
      const predictedClass = this.metadata.class_names[predictedClassIndex];
      
      // Only return prediction if confidence is above threshold
      if (maxConfidence >= confidenceThreshold) {
        return {
          sign: predictedClass,
          confidence: maxConfidence,
          allPredictions: this.metadata.class_names.map((name, idx) => ({
            sign: name,
            confidence: predictionData[idx]
          }))
        };
      }
      
      return null;
    } catch (error) {
      console.error('Prediction error:', error);
      return null;
    }
  }

  /**
   * Get all class names
   */
  getClassNames() {
    return this.metadata?.class_names || [];
  }

  /**
   * Check if model is loaded
   */
  isModelLoaded() {
    return this.isLoaded;
  }

  /**
   * Dispose of model and free memory
   */
  dispose() {
    if (this.model) {
      this.model.dispose();
      this.model = null;
      this.isLoaded = false;
      console.log('Model disposed');
    }
  }
}

// Export singleton instance
export const signLanguageModel = new SignLanguageModel();
export default signLanguageModel;
