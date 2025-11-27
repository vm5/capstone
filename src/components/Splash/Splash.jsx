import React from 'react';
import './Splash.css';

function Splash() {
  return (
    <div className="splash-container">
      <div className="splash-stars"></div>
      <div className="splash-content glass-bg neon-border">
        <div className="splash-logo-container">
          <img src="/pesu.png" alt="PESU" className="splash-logo" />
          <div className="splash-glow"></div>
        </div>
        
        <div className="splash-brand">
          PESU<span>Prep</span>
        </div>
        
        <div className="splash-tagline">
          Your AI Learning Companion
        </div>
        
        <div className="loading-container">
          <div className="loading-bar">
            <div className="loading-progress"></div>
          </div>
          <div className="loading-text">Loading your personalized experience...</div>
        </div>
      </div>
    </div>
  );
}

export default Splash; 