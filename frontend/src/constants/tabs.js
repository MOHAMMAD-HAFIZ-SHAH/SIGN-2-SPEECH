import { Type, Volume2, Mic, Smile } from 'lucide-react';

// Define tab configuration for navigation and icon mapping
export const tabs = [
  { id: 'sign_to_text', name: 'Sign to Text', icon: Type, title: 'Sign Language to Text' },
  { id: 'text_to_speech', name: 'Text to Speech', icon: Volume2, title: 'Text to Speech' },
  { id: 'speech_to_text', name: 'Speech to Text', icon: Mic, title: 'Speech to Text' },
  { id: 'emotion_detection', name: 'Emotion Detection', icon: Smile, title: 'Emotion Detection' },
];
