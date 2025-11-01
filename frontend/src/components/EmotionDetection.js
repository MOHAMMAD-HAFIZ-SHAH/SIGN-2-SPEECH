import React, { useState, useMemo } from 'react';
import { Smile } from 'lucide-react';
import PrimaryButton from './PrimaryButton';
import TitleIconContainer from './TitleIconContainer';

const EmotionDetection = ({ api }) => {
  const [isDetecting, setIsDetecting] = useState(false);
  const [emotions, setEmotions] = useState({
    Happy: 0, Sad: 0, Angry: 0, Surprised: 0, Neutral: 100
  });

  const handleStartDetection = async () => {
    setIsDetecting(true);
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
      <TitleIconContainer icon={Smile} colorClass="text-orange-600" title="Emotion Detection" />

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

export default EmotionDetection;
