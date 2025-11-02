import React, { useState } from 'react';
import Navigation from './components/common/Navigation';
import LoadingIndicator from './components/common/LoadingIndicator';
import SignToText from './components/features/SignToText';
import TextToSpeech from './components/features/TextToSpeech';
import SpeechToText from './components/features/SpeechToText';
import EmotionDetection from './components/features/EmotionDetection';
import { useBackendIntegration } from './hooks/useBackendIntegration';
import { tabs } from './constants/tabs';

// Main Application Component
const App = () => {
  const [activeTab, setActiveTab] = useState(tabs[0].id);
  const { callApi, loading } = useBackendIntegration();

  // Function to render the correct content component based on the active tab
  const renderContent = () => {
    const apiProps = { api: { callApi, loading } };
    
    switch (activeTab) {
      case 'sign_to_text':
        return <SignToText />;
      case 'text_to_speech':
        return <TextToSpeech {...apiProps} />;
      case 'speech_to_text':
        return <SpeechToText {...apiProps} />;
      case 'emotion_detection':
        return <EmotionDetection {...apiProps} />;
      default:
        return <div className="p-8 text-gray-500">Select a feature from the navigation bar.</div>;
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 font-sans antialiased">
      <Navigation activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Main Content Area */}
      <main className="max-w-7xl mx-auto py-8 sm:px-6 lg:px-8">
        <div className="bg-white rounded-2xl shadow-xl overflow-hidden min-h-[70vh]">
          {renderContent()}
        </div>
        
        {/* Footer for extra spacing */}
        <div className="h-16"></div>
      </main>

      <LoadingIndicator loading={loading} />
    </div>
  );
};

export default App;
