import React from 'react';
import Lottie from 'react-lottie';
import PropTypes from 'prop-types';

// Import animations
import confettiAnimation from '../../assets/animations/confetti.json';
import robotAnimation from '../../assets/animations/robot.json';
import rocketAnimation from '../../assets/animations/rocket.json';

const FeedbackAnimation = ({ type, message }) => {
  const getAnimationData = () => {
    switch (type) {
      case 'success':
        return confettiAnimation;
      case 'error':
        return robotAnimation;
      case 'info':
      default:
        return rocketAnimation;
    }
  };

  const defaultOptions = {
    loop: true,
    autoplay: true,
    animationData: getAnimationData(),
    rendererSettings: {
      preserveAspectRatio: 'xMidYMid slice'
    }
  };

  const getMessageStyle = () => {
    switch (type) {
      case 'success':
        return { color: '#28a745' };
      case 'error':
        return { color: '#dc3545' };
      case 'info':
      default:
        return { color: '#17a2b8' };
    }
  };

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
      <div style={{ width: '50px', height: '50px' }}>
        <Lottie options={defaultOptions} />
      </div>
      <p style={{ ...getMessageStyle(), margin: 0 }}>{message}</p>
    </div>
  );
};

FeedbackAnimation.propTypes = {
  type: PropTypes.oneOf(['success', 'error', 'info']).isRequired,
  message: PropTypes.string.isRequired
};

export default FeedbackAnimation; 