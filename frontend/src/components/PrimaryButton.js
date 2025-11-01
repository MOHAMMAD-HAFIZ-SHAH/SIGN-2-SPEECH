import React from 'react';

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

export default PrimaryButton;
