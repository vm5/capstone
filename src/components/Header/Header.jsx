import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import Particles from 'react-particles';
import { loadSlim } from "tsparticles-slim";
import { initiateGoogleAuth, handleGoogleCallback } from '../services/api';
import Splash from '../Splash/Splash';
import './Header.css';

function Header() {
  const navigate = useNavigate();
  const [messages, setMessages] = useState([]);
  const [selectedRole, setSelectedRole] = useState(null);
  const [showSplash, setShowSplash] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const messagesRef = useRef(messages); // Ref to track current messages

  // Update the ref whenever messages change
  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  const particlesInit = async (engine) => {
    await loadSlim(engine);
  };

  const handleGoogleSignIn = (role) => {
    initiateGoogleAuth(role);
  };

  const handleAuthCallback = async (code) => {
    try {
      setIsLoading(true);
      const response = await handleGoogleCallback(code);
      console.log('Auth response:', response);

      if (!response || !response.user) {
        throw new Error('Invalid response from server');
      }

      // Store user data
      localStorage.setItem('user', JSON.stringify(response.user));
      localStorage.setItem('token', response.token);

      // Force a hard redirect based on role
      if (response.user.role === 'teacher') {
        window.location.replace('/teacher-dashboard');
      } else {
        window.location.replace('/dashboard');
      }

    } catch (error) {
      console.error('Auth error:', error);
      addMessage({
        content: error.message || "Authentication failed. Please try again.",
        delay: 0
      });
      setTimeout(() => {
        window.location.replace('/');
      }, 3000);
    } finally {
      setIsLoading(false);
    }
  };

  const addMessage = (msg) => {
    setMessages((prev) => [
      ...prev,
      {
        ...msg,
        content: (
          <div className={`message-content ${isLoading ? 'loading' : ''}`}>
            {msg.content}
          </div>
        ),
      },
    ]);
  };

  const handleRoleSelect = (role) => {
    setSelectedRole(role);
    // Clear existing messages
    setMessages([]);
    
    // Add initial message
    addMessage({
      content: `Great choice! Opening up ${role} registration...`,
      delay: 0
    });

    // After a short delay, refresh and store role
    setTimeout(() => {
      localStorage.setItem('selectedRole', role);
      window.location.reload();
    }, 1500);
  };

  useEffect(() => {
    // Check if this is a post-refresh load with a stored role
    const storedRole = localStorage.getItem('selectedRole');
    if (storedRole) {
      // Clear the stored role immediately
      localStorage.removeItem('selectedRole');
      
      // Show post-refresh messages
      const newMessages = [
        {
          content: storedRole === 'student'
            ? 'Please sign in with your PES University email address to continue.'
            : 'Please sign in with your institutional or personal email address to continue.',
          delay: 1000
        },
        {
          content: (
            <div className="auth-section">
              <button
                className="google-auth-button"
                onClick={() => handleGoogleSignIn(storedRole)}
              >
                <img src="https://www.google.com/favicon.ico" alt="Google" />
                Sign in with Google
              </button>
              <p className="email-note">
                {storedRole === 'student'
                  ? '*Only @pes.edu email addresses are accepted'
                  : '*Institutional or personal email addresses are accepted'}
              </p>
            </div>
          ),
          delay: 2000
        }
      ];

      newMessages.forEach((msg, index) => {
        setTimeout(() => {
          addMessage(msg);
        }, msg.delay);
      });
      
      return; // Skip initial messages if we're showing post-refresh messages
    }

    // Only show initial messages if we're on the home route and no stored role
    if (window.location.pathname === '/') {
      const initialMessages = [
        {
          content: 'Hey there! 👋 Welcome to PESUprep!',
          delay: 1000,
        },
        {
          content: "I'm your personal AI assistant, here to help you excel in your academic journey.",
          delay: 2000,
        },
        {
          content: "Let's get started by choosing your role:",
          delay: 3000,
        },
        {
          content: (
            <div className="role-selection-container">
              <div className="role-cards">
                <button
                  className={`role-card ${selectedRole === 'student' ? 'selected' : ''}`}
                  onClick={() => handleRoleSelect('student')}
                >
                  <div className="role-icon">👨‍🎓</div>
                  <h3>Student</h3>
                  <p>Access study materials, practice tests, and personalized learning paths</p>
                </button>
                <button
                  className={`role-card ${selectedRole === 'teacher' ? 'selected' : ''}`}
                  onClick={() => handleRoleSelect('teacher')}
                >
                  <div className="role-icon">👨‍🏫</div>
                  <h3>Teacher</h3>
                  <p>Create content, manage classes, and track student progress</p>
                </button>
              </div>
            </div>
          ),
          delay: 4000,
        },
      ];

      initialMessages.forEach((msg, index) => {
        setTimeout(() => {
          addMessage(msg);
        }, msg.delay);
      });
    }

    // Check if we're on the callback route
    if (window.location.pathname === '/auth/callback') {
      const urlParams = new URLSearchParams(window.location.search);
      const code = urlParams.get('code');
      console.log('Found auth code:', code);

      if (code) {
        handleAuthCallback(code);
      }
    }
  }, [navigate]);

  return (
    <>
      {showSplash && <Splash />}
      <Particles
        id="tsparticles"
        init={particlesInit}
        options={{
          background: {
            color: {
              value: '#030711',
            },
          },
          fpsLimit: 120,
          particles: {
            color: {
              value: '#ffffff',
            },
            links: {
              enable: false,
            },
            move: {
              enable: true,
              direction: 'none',
              outModes: {
                default: 'out',
              },
              random: true,
              speed: 0.3,
              straight: false,
            },
            number: {
              density: {
                enable: true,
                area: 800,
              },
              value: 100,
            },
            opacity: {
              value: 0.5,
              animation: {
                enable: true,
                speed: 0.5,
                minimumValue: 0.1,
              },
            },
            size: {
              value: { min: 1, max: 3 },
            },
          },
          detectRetina: true,
        }}
      />
      <header className="app-header">
        <div className="logo-section">
          <img src="/pesu.png" alt="PESU" className="logo" />
          <div className="brand-name">
            PESU<span>Prep</span>
          </div>
        </div>
        <div className="tagline">
          Designed and tailored carefully for the students of PES University
        </div>
      </header>

      <div className="chat-container">
        <div className="messages-wrapper">
          {messages.map((msg, index) => (
            <div key={index} className="message">
              <div className="avatar">
                <img src="/bot.png" alt="AI Assistant" className="bot-avatar" />
              </div>
              <div className={`message-content ${isLoading ? 'loading' : ''}`}>
                {msg.content}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="loading-indicator">
              <div className="loading-dots">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          )}
        </div>
      </div>
      <footer className="app-footer">
        <div className="footer-content">
          <div className="footer-brand">
            PESU<span>Prep</span>
          </div>
          <div className="footer-location">Bengaluru, India</div>
          <div className="footer-year">© {new Date().getFullYear()}</div>
        </div>
      </footer>
    </>
  );
}

export default Header;