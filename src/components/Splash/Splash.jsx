import React from 'react';
import './Splash.css';

function Splash() {
  return (
    <div className="splash-container">
      <div className="splash-content">
        <img src="/pesu.png" alt="PESU" className="splash-logo" />
        <div className="splash-brand">
          PESU<span>Prep</span>
        </div>
        <div className="loading-bar"></div>
      </div>
    </div>
  );
}

export default Splash; 