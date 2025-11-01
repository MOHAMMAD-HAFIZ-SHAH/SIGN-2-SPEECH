/**
 * Sign Language Model API Client
 * Communicates with Flask backend API for predictions
 */

class SignLanguageModel {
  constructor() {
    this.apiUrl = 'http://localhost:5001';
    this.metadata = null;
    this.isLoaded = false;
  }

  /**
   * Check API health and load metadata
   */
  async loadModel() {
    try {
      console.log('Connecting to sign language API...');
      
      // Check if API is running
      const healthResponse = await fetch(`${this.apiUrl}/health`);
      const healthData = await healthResponse.json();
      
      if (!healthData.model_loaded) {
        throw new Error('Model not loaded on server');
      }
      
      // Load metadata (class names, etc.)
      const classesResponse = await fetch(`${this.apiUrl}/classes`);
      this.metadata = await classesResponse.json();
      console.log('Metadata loaded:', this.metadata);
      
      this.isLoaded = true;
      console.log('API ready for predictions');
      
      return {
        success: true,
        classNames: this.metadata.classes,
        imgSize: this.metadata.img_size
      };
    } catch (error) {
      console.error('Error connecting to API:', error);
      this.isLoaded = false;
      return {
        success: false,
        error: error.message + ' - Make sure the Flask server is running (python api_server.py)'
      };
    }
  }

  /**
   * Capture frame from video element as base64
   */
  captureFrame(videoElement) {
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoElement, 0, 0);
    return canvas.toDataURL('image/jpeg', 0.8);
  }

  /**
   * Make prediction on current video frame
   */
  async predict(videoElement, confidenceThreshold = 0.7) {
    if (!this.isLoaded) {
      console.warn('API not connected yet');
      return null;
    }

    try {
      // Capture frame as base64
      const imageData = this.captureFrame(videoElement);
      
      // Send to API
      const response = await fetch(`${this.apiUrl}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageData,
          threshold: confidenceThreshold
        })
      });
      
      const result = await response.json();
      
      if (result.success && result.sign) {
        return {
          sign: result.sign,
          confidence: result.confidence,
          allPredictions: result.top_predictions
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
    return this.metadata?.classes || [];
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
