import React from 'react';
import './Footer.css';

const Footer = () => {
  return (
    <footer className="footer">
      <div className="footer-content">
        <div className="footer-logo">
          <img src="/pesu.png" alt="PESU" />
        </div>
        
        <div className="footer-brand">
          PESU<span>Prep</span>
        </div>
        
        <div className="footer-location">
          Bengaluru, India
        </div>
      </div>
    </footer>
  );
};

export default Footer; 