import React, { useState } from 'react';
import { Type, Volume2, Mic, Smile } from 'lucide-react';
import useBackendIntegration from './hooks/useBackendIntegration';
import SignToText from './components/SignToText';
import TextToSpeech from './components/TextToSpeech';
import SpeechToText from './components/SpeechToText';
import EmotionDetection from './components/EmotionDetection';

// Define tab configuration for navigation and icon mapping
const tabs = [
  { id: 'sign_to_text', name: 'Sign to Text', icon: Type, title: 'Sign Language to Text' },
  { id: 'text_to_speech', name: 'Text to Speech', icon: Volume2, title: 'Text to Speech' },
  { id: 'speech_to_text', name: 'Speech to Text', icon: Mic, title: 'Speech to Text' },
  { id: 'emotion_detection', name: 'Emotion Detection', icon: Smile, title: 'Emotion Detection' },
];


// --- Main Application Component ---

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
