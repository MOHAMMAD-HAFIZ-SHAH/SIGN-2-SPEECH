import React from 'react';

// Loading Indicator Component
const LoadingIndicator = ({ loading }) => {
  if (!loading) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-30 flex items-center justify-center z-50">
      <div className="flex items-center p-4 bg-white rounded-xl shadow-2xl">
        <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600 mr-3"></div>
        <span className="text-gray-700 font-medium">Communicating with backend...</span>
      </div>
    </div>
  );
};

export default LoadingIndicator;
