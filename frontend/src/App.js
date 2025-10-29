import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import { Type, Volume2, Mic, Smile, Download, Copy, Trash2, Pause, StopCircle, VideoOff } from 'lucide-react';

// --- Global Constants and Styling Helpers ---

// Define tab configuration for navigation and icon mapping
const tabs = [
  { id: 'sign_to_text', name: 'Sign to Text', icon: Type, title: 'Sign Language to Text' },
  { id: 'text_to_speech', name: 'Text to Speech', icon: Volume2, title: 'Text to Speech' },
  { id: 'speech_to_text', name: 'Speech to Text', icon: Mic, title: 'Speech to Text' },
  { id: 'emotion_detection', name: 'Emotion Detection', icon: Smile, title: 'Emotion Detection' },
];

// Reusable Button Component with Tailwind styling
const PrimaryButton = ({ children, onClick, icon: Icon, className = '', disabled = false }) => (
  <button
    onClick={onClick}
    disabled={disabled}
    className={`flex items-center justify-center px-6 py-3 text-white rounded-lg font-semibold shadow-lg transition duration-150 active:scale-[0.98] focus:outline-none focus:ring-4 
      ${disabled 
        ? 'bg-gray-400 cursor-not-allowed' 
        : 'bg-blue-600 hover:bg-blue-700 focus:ring-blue-300'
      } ${className}`}
  >
    {Icon && <Icon className="w-5 h-5 mr-2" />}
    {children}
  </button>
);

// Reusable Icon Container for titles
const TitleIconContainer = ({ icon: Icon, colorClass, title }) => (
  <div className="flex items-center space-x-3 mb-6">
    <div className={`p-2 rounded-full ${colorClass} bg-opacity-20`}>
      <Icon className={`w-6 h-6 ${colorClass}`} />
    </div>
    <h1 className="text-2xl font-bold text-gray-800">{title}</h1>
  </div>
);

// --- Backend Integration Placeholders (Used by other features) ---
const useBackendIntegration = () => {
  const [loading, setLoading] = useState(false);

  // General function for API interaction (mocked)
  const callApi = useCallback(async (endpoint, data) => {
    console.log(`[Backend Call] Attempting to hit: /api/${endpoint}`);
    setLoading(true);
    try {
      // Simulate network delay
      await new Promise(resolve => setTimeout(resolve, 1500)); 
      console.log(`[Backend Call] Success for ${endpoint}.`);
      return { success: true, mockData: `Processed result for ${endpoint}` };
    } catch (error) {
      console.error(`[Backend Error] Failed to process ${endpoint}:`, error);
      return { success: false, error: error.message };
    } finally {
      setLoading(false);
    }
  }, []);

  return { callApi, loading };
};

// --- Feature Content Components ---

// 1. Sign to Text (Integrated with live camera and simulated transcription)
const SignToTextContent = () => {
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [detectedText, setDetectedText] = useState('Sign language interpretation will appear here...');
  const [cameraError, setCameraError] = useState(null);
  
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);

  // Sign simulation sequence
  const signs = ["HELLO", "WORLD", "HOW ARE YOU", "I AM GOOD", "THANK YOU"];
  let signIndex = 0;
  
  const startSimulation = () => {
    setDetectedText('');
    signIndex = 0;
    
    intervalRef.current = setInterval(() => {
        const nextSign = signs[signIndex % signs.length];
        setDetectedText(prev => (prev + ' ' + nextSign).trim());
        console.log(`Simulated Sign Detected: ${nextSign}`);
        signIndex++;
    }, 3000); // Simulate detecting a new sign every 3 seconds
  };
  
  const handleStartCamera = async () => {
    setCameraError(null);
    try {
      // Check for MediaStream support
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Your browser does not support camera access.");
      }
      
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        setIsCameraActive(true);
        startSimulation();
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
    if (intervalRef.current) {
        clearInterval(intervalRef.current);
    }
    setIsCameraActive(false);
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
      <TitleIconContainer icon={Type} colorClass="text-blue-600" title={tabs[0].title} />

      {/* Camera Feed Area */}
      <div className="bg-gray-800 p-2 h-96 rounded-xl shadow-inner mb-8 flex flex-col items-center justify-center relative overflow-hidden">
        
        {/* Video Element (mirrored to feel natural) */}
        <video 
          ref={videoRef} 
          className={`w-full h-full object-cover rounded-lg transform scale-x-[-1] transition-opacity duration-500 
            ${isCameraActive ? 'opacity-100' : 'opacity-0 absolute'}`}
          autoPlay 
          playsInline 
          muted 
        />
        
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
      
      <div className="flex justify-center space-x-4 mb-8">
          {!isCameraActive ? (
              <PrimaryButton onClick={handleStartCamera} icon={Type} className="w-48">
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
            {detectedText}
        </p>
      </div>
    </div>
  );
};

// 2. Text to Speech
const TextToSpeechContent = ({ api }) => {
  const [text, setText] = useState('');
  const [speed, setSpeed] = useState(1); // Placeholder for speed control
  
  // Audio state placeholders
  const [audioUrl, setAudioUrl] = useState(null);
  const audioRef = useRef(null);

  // UseEffect hook to handle audio loading and playback
  useEffect(() => {
    const audio = audioRef.current;
    if (audio && audioUrl) {
      // 1. Explicitly load the new source to ensure it's ready
      audio.load();

      const handleLoadedData = () => {
        // 2. Play only when data is confirmed to be loaded
        audio.play().catch(e => {
            console.error("Audio playback error (likely autoplay restriction):", e);
        });
      };
      
      audio.addEventListener('loadeddata', handleLoadedData);

      // Cleanup listener when component unmounts or audioUrl changes
      return () => {
        audio.removeEventListener('loadeddata', handleLoadedData);
      };
    }
  }, [audioUrl]); // Dependency on audioUrl

  const handleSpeak = async () => {
    if (!text.trim()) return;
    // Placeholder for backend TTS call
    const result = await api.callApi('text-to-speech', { text, speed });

    if (result.success) {
        // Mocking an audio URL for UI demonstration
        // Add cache-busting parameter (?t=...) to force the audio element to reload the source
        const mockAudioUrl = `https://mock.audio.url/generated.mp3?t=${Date.now()}`; 
        setAudioUrl(mockAudioUrl);
    }
  };

  const handlePause = () => audioRef.current?.pause();
  const handleStop = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
  };
  const handleDownload = () => {
    if (audioUrl) {
      // Logic to trigger a download (e.g., creating an anchor tag and clicking it)
      console.log('Downloading audio from:', audioUrl);
      // Simulate download feedback without using alert():
      const downloadMessage = document.getElementById('download-message');
      if (downloadMessage) {
        downloadMessage.textContent = 'Download simulated. Audio file ready!';
        setTimeout(() => { downloadMessage.textContent = ''; }, 3000);
      }
    }
  };

  return (
    <div className="p-8">
      <TitleIconContainer icon={Volume2} colorClass="text-blue-600" title={tabs[1].title} />

      <h2 className="text-gray-600 mb-2">Enter text to convert to speech:</h2>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Type your text here..."
        className="w-full h-40 p-4 mb-6 rounded-xl border border-gray-300 focus:ring-blue-500 focus:border-blue-500 shadow-sm transition"
      />

      <div className="flex items-center justify-between mb-8">
        <PrimaryButton onClick={handleSpeak} icon={Volume2} disabled={api.loading || !text.trim()}>
          {api.loading ? 'Generating...' : 'Speak'}
        </PrimaryButton>
        <div className="flex items-center space-x-3 text-gray-600">
          <label htmlFor="speed-slider" className="font-medium">Speed:</label>
          <input
            id="speed-slider"
            type="range"
            min="0.5"
            max="2"
            step="0.1"
            value={speed}
            onChange={(e) => setSpeed(parseFloat(e.target.value))}
            className="w-40 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer range-lg"
          />
          <span className="text-sm font-mono">{speed.toFixed(1)}x</span>
        </div>
      </div>

      <h2 className="text-lg font-semibold text-gray-700 mb-3">Audio Controls:</h2>
      <div className="bg-white p-4 rounded-xl border border-gray-200 shadow-md flex space-x-4">
        {/* Hidden audio element for control */}
        <audio ref={audioRef} src={audioUrl || undefined} hidden /> 
        
        <button onClick={handlePause} className="px-4 py-2 text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition">
          <Pause className="w-5 h-5 inline-block mr-1" /> Pause
        </button>
        <button onClick={handleStop} className="px-4 py-2 text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition">
          <StopCircle className="w-5 h-5 inline-block mr-1" /> Stop
        </button>
        <button onClick={handleDownload} className="px-4 py-2 text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition" disabled={!audioUrl}>
          <Download className="w-5 h-5 inline-block mr-1" /> Download Audio
        </button>
      </div>
      {/* Visual Feedback for Download */}
      <p id="download-message" className="text-sm text-green-600 mt-2 h-5"></p>
    </div>
  );
};

// 3. Speech to Text
const SpeechToTextContent = ({ api }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [transcription, setTranscription] = useState('Your speech will be transcribed here...');

  const handleStartRecording = async () => {
    if (isRecording) {
      setIsRecording(false);
      // Placeholder for backend call to stop recording and process audio
      const result = await api.callApi('stop-speech-to-text', { action: 'stop' });
      if (result.success) {
        setTranscription('The quick brown fox jumps over the lazy dog. This is a transcribed output from the speech recognition service.');
      }
    } else {
      setIsRecording(true);
      setTranscription('...Listening...');
      // Placeholder for backend call to start recording
      await api.callApi('start-speech-to-text', { action: 'start' });
    }
  };

  const handleCopy = () => {
    // Basic clipboard copy
    navigator.clipboard.writeText(transcription).then(() => {
      console.log('Transcription copied!');
    }).catch(err => {
      console.error('Could not copy text: ', err);
    });
  };

  const handleClear = () => setTranscription('');
  
  const handleDownload = () => {
      if (transcription && transcription.trim()) {
          console.log('Simulating download of transcription:', transcription.substring(0, 50) + '...');
          // Mock action: create a blob and download it
          const blob = new Blob([transcription], { type: 'text/plain' });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = 'transcription.txt';
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(url);
      } else {
          console.log('No content to download.');
      }
  };

  return (
    <div className="p-8">
      <TitleIconContainer icon={Mic} colorClass="text-green-600" title={tabs[2].title} />

      {/* Recording Area */}
      <div className="bg-gray-50 p-8 h-80 rounded-xl flex flex-col items-center justify-center mb-8 border border-gray-200"
           style={{ background: 'linear-gradient(180deg, #f0fff0, #f5f5f5)' }}>
        
        {/* Animated Mic Icon */}
        <div className={`p-6 rounded-full ${isRecording ? 'bg-green-100 animate-pulse' : 'bg-green-100'}`} style={{ transition: 'all 0.3s' }}>
          <Mic className={`w-10 h-10 ${isRecording ? 'text-red-500' : 'text-green-600'}`} />
        </div>

        <PrimaryButton 
          onClick={handleStartRecording} 
          className={`mt-6 w-52 ${isRecording ? 'bg-red-500 hover:bg-red-600' : 'bg-green-600 hover:bg-green-700'}`}
          disabled={api.loading && !isRecording} // Allow stopping while API is loading the stop result
        >
          {api.loading && isRecording ? 'Processing...' : isRecording ? 'Stop Recording' : 'Start Recording'}
        </PrimaryButton>
        <p className="mt-2 text-sm text-gray-500">
          Click to {isRecording ? 'stop and transcribe' : 'start recording'} your voice
        </p>
      </div>

      {/* Transcription Area */}
      <h2 className="text-lg font-semibold text-gray-700 mb-3">Transcription:</h2>
      <div className="bg-white p-4 rounded-xl border border-gray-200 shadow-inner">
        <textarea
          readOnly
          value={transcription}
          className="w-full h-32 bg-transparent resize-none focus:outline-none text-gray-800"
        />
        <div className="mt-3 flex space-x-3">
          <PrimaryButton onClick={handleCopy} className="bg-blue-600 hover:bg-blue-700 px-4 py-2 text-sm">
            <Copy className="w-4 h-4 mr-1" /> Copy Text
          </PrimaryButton>
          <PrimaryButton onClick={handleDownload} className="bg-gray-600 hover:bg-gray-700 px-4 py-2 text-sm">
            <Download className="w-4 h-4 mr-1" /> Download
          </PrimaryButton>
          <PrimaryButton onClick={handleClear} className="bg-red-600 hover:bg-red-700 px-4 py-2 text-sm">
            <Trash2 className="w-4 h-4 mr-1" /> Clear
          </PrimaryButton>
        </div>
      </div>
    </div>
  );
};

// 4. Emotion Detection
const EmotionDetectionContent = ({ api }) => {
  const [isDetecting, setIsDetecting] = useState(false);
  const [emotions, setEmotions] = useState({
    Happy: 0, Sad: 0, Angry: 0, Surprised: 0, Neutral: 100
  });

  const handleStartDetection = async () => {
    setIsDetecting(true);
    // Placeholder for backend call to start camera/model for emotion detection
    await api.callApi('start-emotion-detection', { action: 'start' });

    // Mock data update after delay
    setTimeout(() => {
        setEmotions({
            Happy: 15, Sad: 5, Angry: 5, Surprised: 5, Neutral: 70
        });
    }, 2500);
  };

  const primaryEmotion = useMemo(() => {
    const sorted = Object.entries(emotions).sort(([, a], [, b]) => b - a);
    return sorted[0];
  }, [emotions]);

  return (
    <div className="p-8">
      <TitleIconContainer icon={Smile} colorClass="text-orange-600" title={tabs[3].title} />

      {/* Camera Feed Area (Emotion View) */}
      <div className="bg-gray-50 p-6 h-96 rounded-xl border-4 border-dashed border-orange-300 flex flex-col items-center justify-center mb-8"
           style={{ background: 'linear-gradient(180deg, #fff8f0, #f5f5f5)' }}>
        <div className="text-6xl text-orange-400 p-4 rounded-full bg-white mb-4">
          <span role="img" aria-label="smiley face">😊</span>
        </div>
        <p className="text-gray-600 mb-4">Camera feed for emotion detection</p>
        <PrimaryButton 
          onClick={handleStartDetection} 
          className="bg-orange-600 hover:bg-orange-700 w-56"
          disabled={isDetecting || api.loading}
        >
          {api.loading ? 'Connecting...' : isDetecting ? 'Detection Active' : 'Start Detection'}
        </PrimaryButton>
      </div>

      {/* Results Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Detected Emotions Panel */}
        <div className="bg-white p-6 rounded-xl shadow-lg border border-gray-200">
          <h2 className="text-lg font-bold text-gray-700 mb-4">Detected Emotions:</h2>
          <div className="space-y-3">
            {Object.entries(emotions).map(([name, percent]) => (
              <div key={name} className="flex flex-col">
                <div className="flex justify-between text-sm font-medium text-gray-600 mb-1">
                  <span>{name}</span>
                  <span>{percent.toFixed(0)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2.5">
                  <div 
                    className="bg-blue-500 h-2.5 rounded-full transition-all duration-500" 
                    style={{ width: `${percent}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Primary Emotion Panel */}
        <div className="bg-white p-6 rounded-xl shadow-lg border border-gray-200 text-center flex flex-col items-center justify-center">
          <h2 className="text-lg font-bold text-gray-700 mb-4">Primary Emotion:</h2>
          <div className="text-8xl mb-3">
            {/* Simple icon representation of primary emotion for visualization */}
            {primaryEmotion[0] === 'Happy' && <span role="img" aria-label="Happy">😄</span>}
            {primaryEmotion[0] === 'Sad' && <span role="img" aria-label="Sad">😞</span>}
            {primaryEmotion[0] === 'Angry' && <span role="img" aria-label="Angry">😡</span>}
            {primaryEmotion[0] === 'Surprised' && <span role="img" aria-label="Surprised">😮</span>}
            {primaryEmotion[0] === 'Neutral' && <span role="img" aria-label="Neutral">🙂</span>}
          </div>
          <p className="text-2xl font-extrabold text-gray-800">{primaryEmotion[0]}</p>
          <p className="text-sm text-gray-500">Confidence: {primaryEmotion[1].toFixed(0)}%</p>
        </div>
      </div>
    </div>
  );
};


// --- Main Application Component ---

const App = () => {
  const [activeTab, setActiveTab] = useState(tabs[0].id);
  const { callApi, loading } = useBackendIntegration();

  // Function to render the correct content component based on the active tab
  const renderContent = () => {
    const apiProps = { api: { callApi, loading } };
    switch (activeTab) {
      // SignToTextContent handles its own camera integration
      case 'sign_to_text':
        return <SignToTextContent />;
      case 'text_to_speech':
        return <TextToSpeechContent {...apiProps} />;
      case 'speech_to_text':
        return <SpeechToTextContent {...apiProps} />;
      case 'emotion_detection':
        return <EmotionDetectionContent {...apiProps} />;
      default:
        return <div className="p-8 text-gray-500">Select a feature from the navigation bar.</div>;
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 font-sans antialiased">
      {/* Top Navigation Bar (Header) */}
      <nav className="bg-white shadow-md sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            
            {/* Logo/Title */}
            <div className="flex-shrink-0 text-xl font-extrabold text-blue-600">
              Sign2Speech
            </div>

            {/* Navigation Tabs */}
            <div className="flex space-x-2 md:space-x-4">
              {tabs.map((tab) => {
                const isActive = activeTab === tab.id;
                const Icon = tab.icon;
                
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`
                      flex items-center px-3 py-2 rounded-lg text-sm font-medium transition duration-150
                      ${isActive
                        ? 'bg-blue-600 text-white shadow-lg'
                        : 'text-gray-600 hover:bg-gray-100 hover:text-blue-600'
                      }
                    `}
                  >
                    <Icon className="w-5 h-5 mr-1" />
                    <span className="hidden sm:inline">{tab.name}</span>
                  </button>
                );
              })}
            </div>

          </div>
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="max-w-7xl mx-auto py-8 sm:px-6 lg:px-8">
        <div className="bg-white rounded-2xl shadow-xl overflow-hidden min-h-[70vh]">
          {renderContent()}
        </div>
        
        {/* Footer for extra spacing */}
        <div className="h-16"></div>
      </main>

      {/* Loading Indicator for API Calls */}
      {loading && (
        <div className="fixed inset-0 bg-black bg-opacity-30 flex items-center justify-center z-50">
          <div className="flex items-center p-4 bg-white rounded-xl shadow-2xl">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600 mr-3"></div>
            <span className="text-gray-700 font-medium">Communicating with backend...</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
