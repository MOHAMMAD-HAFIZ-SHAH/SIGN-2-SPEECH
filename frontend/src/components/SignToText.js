import React, { useState, useRef, useEffect } from 'react';
import { Type, StopCircle, VideoOff, AlertCircle, Loader } from 'lucide-react';
import PrimaryButton from './PrimaryButton';
import TitleIconContainer from './TitleIconContainer';
import signLanguageModel from '../utils/signLanguageModel';

const SignToText = () => {
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [detectedText, setDetectedText] = useState('Sign language interpretation will appear here...');
  const [cameraError, setCameraError] = useState(null);
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [currentSign, setCurrentSign] = useState(null);
  const [confidence, setConfidence] = useState(0);
  
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const predictionIntervalRef = useRef(null);
  const canvasRef = useRef(null);
  const lastDetectedSignRef = useRef(null);
  const signStabilityCounterRef = useRef(0);

  // Load model on component mount
  useEffect(() => {
    loadModel();
    return () => {
      // Cleanup on unmount
      if (signLanguageModel.isModelLoaded()) {
        signLanguageModel.dispose();
      }
    };
  }, []);

  const loadModel = async () => {
    setIsModelLoading(true);
    try {
      const result = await signLanguageModel.loadModel();
      if (result.success) {
        setModelLoaded(true);
        console.log('Model loaded with classes:', result.classNames);
      } else {
        setCameraError(`Failed to load model: ${result.error}`);
      }
    } catch (error) {
      console.error('Model loading error:', error);
      setCameraError('Failed to load sign language model. Please ensure the model files are in /public/models/');
    } finally {
      setIsModelLoading(false);
    }
  };
  
  const startRealTimeDetection = () => {
    if (!modelLoaded || !videoRef.current) return;

    setDetectedText('');
    lastDetectedSignRef.current = null;
    signStabilityCounterRef.current = 0;
    
    // Run predictions every 500ms
    predictionIntervalRef.current = setInterval(async () => {
      if (!videoRef.current || videoRef.current.paused || videoRef.current.ended) {
        return;
      }

      try {
        // Make prediction on current frame (lowered threshold to 0.4 for better detection)
        const prediction = await signLanguageModel.predict(videoRef.current, 0.4);
        
        if (prediction) {
          const { sign, confidence: conf } = prediction;
          
          setCurrentSign(sign);
          setConfidence(conf);
          
          // Sign stabilization: only add to text if same sign detected multiple times
          if (sign === lastDetectedSignRef.current) {
            signStabilityCounterRef.current += 1;
            
            // Add to text after 3 consecutive detections (1.5 seconds)
            if (signStabilityCounterRef.current === 3) {
              setDetectedText(prev => {
                const words = prev.trim().split(' ').filter(w => w);
                // Don't add duplicate consecutive signs
                if (words[words.length - 1] !== sign) {
                  return (prev + ' ' + sign).trim();
                }
                return prev;
              });
              signStabilityCounterRef.current = 0;
            }
          } else {
            // New sign detected, reset counter
            lastDetectedSignRef.current = sign;
            signStabilityCounterRef.current = 1;
          }
        } else {
          // No confident prediction
          setCurrentSign(null);
          setConfidence(0);
          signStabilityCounterRef.current = 0;
        }
      } catch (error) {
        console.error('Prediction error:', error);
      }
    }, 500);
  };
  
  const handleStartCamera = async () => {
    if (!modelLoaded) {
      setCameraError('Please wait for the model to load before starting the camera.');
      return;
    }

    setCameraError(null);
    try {
      // Check for MediaStream support
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Your browser does not support camera access.");
      }
      
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        } 
      });
      streamRef.current = stream;
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setIsCameraActive(true);
        
        // Start real-time detection
        startRealTimeDetection();
      }
    } catch (err) {
      console.error('Camera access denied or failed:', err);
      setCameraError(err.message || 'Camera access denied or failed. Please check permissions.');
      setIsCameraActive(false);
    }
  };
  
  const handleStopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (predictionIntervalRef.current) {
        clearInterval(predictionIntervalRef.current);
        predictionIntervalRef.current = null;
    }
    setIsCameraActive(false);
    setCurrentSign(null);
    setConfidence(0);
    setDetectedText('Camera stopped. Ready to start new session.');
  };
  
  // Cleanup effect
  useEffect(() => {
    return () => {
      handleStopCamera();
    };
  }, []);

  return (
    <div className="p-8">
      <TitleIconContainer icon={Type} colorClass="text-blue-600" title="Sign Language to Text" />

      {/* Model Loading Status */}
      {isModelLoading && (
        <div className="mb-4 p-4 bg-blue-50 border border-blue-200 rounded-lg flex items-center">
          <Loader className="w-5 h-5 text-blue-600 mr-3 animate-spin" />
          <span className="text-blue-800 font-medium">Loading AI model...</span>
        </div>
      )}

      {/* Model Error */}
      {!isModelLoading && !modelLoaded && (
        <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg flex items-center">
          <AlertCircle className="w-5 h-5 text-red-600 mr-3" />
          <div className="text-red-800">
            <p className="font-medium">Model not loaded</p>
            <p className="text-sm">Please train and deploy the model first. See model-training/README.md</p>
          </div>
        </div>
      )}

      {/* Camera Feed Area */}
      <div className="bg-gray-800 p-2 h-96 rounded-xl shadow-inner mb-4 flex flex-col items-center justify-center relative overflow-hidden">
        
        {/* Video Element (mirrored to feel natural) */}
        <video 
          ref={videoRef} 
          className={`w-full h-full object-cover rounded-lg transform scale-x-[-1] transition-opacity duration-500 
            ${isCameraActive ? 'opacity-100' : 'opacity-0 absolute'}`}
          autoPlay 
          playsInline 
          muted 
        />

        {/* Canvas for overlays (optional - can be used for hand detection visualization) */}
        <canvas 
          ref={canvasRef}
          className={`absolute inset-0 w-full h-full pointer-events-none ${isCameraActive ? '' : 'hidden'}`}
        />

        {/* Current Detection Overlay */}
        {isCameraActive && currentSign && (
          <div className="absolute top-4 right-4 bg-black bg-opacity-70 text-white px-4 py-2 rounded-lg">
            <p className="text-2xl font-bold">{currentSign}</p>
            <p className="text-xs text-gray-300">Confidence: {(confidence * 100).toFixed(1)}%</p>
          </div>
        )}
        
        {/* Placeholder / Error Message */}
        {!isCameraActive && (
            <div className={`absolute inset-0 flex flex-col items-center justify-center p-4 rounded-xl text-center bg-gray-100 ${isCameraActive ? 'hidden' : ''}`}>
                <VideoOff className="w-12 h-12 text-gray-400 mb-4" />
                {cameraError ? (
                    <p className="text-red-500 font-medium">{cameraError}</p>
                ) : (
                    <p className="text-gray-600">Click 'Start Camera' to enable live sign language detection.</p>
                )}
            </div>
        )}
        
      </div>

      {/* Status Info */}
      {isCameraActive && (
        <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded-lg">
          <p className="text-sm text-green-800">
            ✓ Camera active • AI detection running • Hold each sign steady for best results
          </p>
        </div>
      )}
      
      <div className="flex justify-center space-x-4 mb-8">
          {!isCameraActive ? (
              <PrimaryButton 
                onClick={handleStartCamera} 
                icon={Type} 
                className="w-48"
                disabled={!modelLoaded || isModelLoading}
              >
                Start Camera
              </PrimaryButton>
          ) : (
              <PrimaryButton onClick={handleStopCamera} icon={StopCircle} className="w-48 bg-red-600 hover:bg-red-700">
                Stop Camera
              </PrimaryButton>
          )}
      </div>

      {/* Detected Text Area */}
      <h2 className="text-lg font-semibold text-gray-700 mb-3">Live Transcription:</h2>
      <div className="bg-white p-4 rounded-xl border border-gray-200 shadow-inner">
        <p className="w-full h-32 bg-transparent resize-none overflow-y-auto text-gray-800 p-1">
            {detectedText || 'Detected signs will appear here...'}
        </p>
      </div>

      {/* Instructions */}
      <div className="mt-6 p-4 bg-gray-50 rounded-lg">
        <h3 className="font-semibold text-gray-700 mb-2">Tips for best results:</h3>
        <ul className="text-sm text-gray-600 space-y-1">
          <li>• Ensure good lighting on your hands</li>
          <li>• Keep your hand centered in the frame</li>
          <li>• Hold each sign steady for 1-2 seconds</li>
          <li>• Use a plain background if possible</li>
        </ul>
      </div>
    </div>
  );
};

export default SignToText;
