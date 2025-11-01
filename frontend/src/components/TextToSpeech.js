import React, { useState, useRef, useEffect } from 'react';
import { Volume2, Pause, StopCircle, Download } from 'lucide-react';
import PrimaryButton from './PrimaryButton';
import TitleIconContainer from './TitleIconContainer';

const TextToSpeech = ({ api }) => {
  const [text, setText] = useState('');
  const [speed, setSpeed] = useState(1);
  const [audioUrl, setAudioUrl] = useState(null);
  const audioRef = useRef(null);

  // UseEffect hook to handle audio loading and playback
  useEffect(() => {
    const audio = audioRef.current;
    if (audio && audioUrl) {
      audio.load();

      const handleLoadedData = () => {
        audio.play().catch(e => {
            console.error("Audio playback error (likely autoplay restriction):", e);
        });
      };
      
      audio.addEventListener('loadeddata', handleLoadedData);

      return () => {
        audio.removeEventListener('loadeddata', handleLoadedData);
      };
    }
  }, [audioUrl]);

  const handleSpeak = async () => {
    if (!text.trim()) return;
    const result = await api.callApi('text-to-speech', { text, speed });

    if (result.success) {
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
      console.log('Downloading audio from:', audioUrl);
      const downloadMessage = document.getElementById('download-message');
      if (downloadMessage) {
        downloadMessage.textContent = 'Download simulated. Audio file ready!';
        setTimeout(() => { downloadMessage.textContent = ''; }, 3000);
      }
    }
  };

  return (
    <div className="p-8">
      <TitleIconContainer icon={Volume2} colorClass="text-blue-600" title="Text to Speech" />

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
      <p id="download-message" className="text-sm text-green-600 mt-2 h-5"></p>
    </div>
  );
};

export default TextToSpeech;
