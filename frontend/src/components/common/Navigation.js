import React from 'react';
import { tabs } from '../../constants/tabs';

// Navigation Bar Component
const Navigation = ({ activeTab, setActiveTab }) => {
  return (
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
  );
};

export default Navigation;
