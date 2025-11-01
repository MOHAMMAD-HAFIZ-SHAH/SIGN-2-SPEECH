import React from 'react';

const TitleIconContainer = ({ icon: Icon, colorClass, title }) => (
  <div className="flex items-center space-x-3 mb-6">
    <div className={`p-2 rounded-full ${colorClass} bg-opacity-20`}>
      <Icon className={`w-6 h-6 ${colorClass}`} />
    </div>
    <h1 className="text-2xl font-bold text-gray-800">{title}</h1>
  </div>
);

export default TitleIconContainer;
