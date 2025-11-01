import React, { useState } from 'react';
import { Mic, Copy, Download, Trash2 } from 'lucide-react';
import PrimaryButton from './PrimaryButton';
import TitleIconContainer from './TitleIconContainer';

const SpeechToText = ({ api }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [transcription, setTranscription] = useState('Your speech will be transcribed here...');

  const handleStartRecording = async () => {
    if (isRecording) {
      setIsRecording(false);
      const result = await api.callApi('stop-speech-to-text', { action: 'stop' });
      if (result.success) {
        setTranscription('The quick brown fox jumps over the lazy dog. This is a transcribed output from the speech recognition service.');
      }
    } else {
      setIsRecording(true);
      setTranscription('...Listening...');
      await api.callApi('start-speech-to-text', { action: 'start' });
    }
  };

  const handleCopy = () => {
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
      <TitleIconContainer icon={Mic} colorClass="text-green-600" title="Speech to Text" />

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
          disabled={api.loading && !isRecording}
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

export default SpeechToText;
