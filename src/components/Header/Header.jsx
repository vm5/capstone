import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Particles from 'react-particles';
import { loadSlim } from "tsparticles-slim";
import { initiateGoogleAuth, handleGoogleCallback } from '../../services/api';
import Splash from '../Splash/Splash';
import './Header.css';

function Header() {
  const navigate = useNavigate();
  const [showSplash, setShowSplash] = useState(true);
  const [messages, setMessages] = useState([]);
  const [selectedRole, setSelectedRole] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  // Initialize particles
  const particlesInit = async (engine) => {
    await loadSlim(engine);
  };

  // Add this function at the beginning of the Header component
  const checkAuthAndRedirect = () => {
    const userData = localStorage.getItem('user');
    const token = localStorage.getItem('token');
    
    if (userData && token) {
      try {
        const user = JSON.parse(userData);
        if (user.role === 'teacher') {
          navigate('/teacher-dashboard', { replace: true });
          return true;
        } else if (user.role === 'student') {
          navigate('/dashboard', { replace: true });
          return true;
        }
      } catch (error) {
        console.error('Error parsing user data:', error);
        localStorage.clear();
      }
    }
    return false;
  };

  // Handle auth flow
  useEffect(() => {
    // Check if we're on the callback route
    if (window.location.pathname === '/auth/callback') {
      handleAuthCallback();
      return;
    }

    // Check for existing auth on home page
    if (window.location.pathname === '/') {
      const isAuthenticated = checkAuthAndRedirect();
      if (isAuthenticated) {
        return;
      }

      // If no auth, show welcome sequence
      setMessages([]);
      
      const timer1 = setTimeout(() => {
        setMessages([{
          id: 'welcome-1',
          content: 'Hey there! 👋 Welcome to PESUprep!'
        }]);
      }, 1000);

      const timer2 = setTimeout(() => {
        setMessages(prev => [...prev, {
          id: 'welcome-2',
          content: "I'm your personal AI assistant, here to help you excel in your academic journey."
        }]);
      }, 2000);

      const timer3 = setTimeout(() => {
        setMessages(prev => [...prev, {
          id: 'welcome-3',
          content: "Let's get started by choosing your role:"
        }]);
      }, 3000);

      const timer4 = setTimeout(() => {
        setMessages(prev => [...prev, {
          id: 'role-selection',
          content: (
            <div className="role-selection-container">
              <div className="role-cards">
                <button
                  className="role-card"
                  onClick={() => handleRoleSelect('student')}
                >
                  <div className="role-icon">👨‍🎓</div>
                  <h3>Student</h3>
                  <p>Access study materials, practice tests, and personalized learning paths</p>
                </button>
                <button
                  className="role-card"
                  onClick={() => handleRoleSelect('teacher')}
                >
                  <div className="role-icon">👨‍🏫</div>
                  <h3>Teacher</h3>
                  <p>Create content, manage classes, and track student progress</p>
                </button>
              </div>
            </div>
          )
        }]);
      }, 4000);

      // Hide splash screen after a delay
      const splashTimer = setTimeout(() => setShowSplash(false), 2000);

      return () => {
        clearTimeout(timer1);
        clearTimeout(timer2);
        clearTimeout(timer3);
        clearTimeout(timer4);
        clearTimeout(splashTimer);
      };
    }
  }, [navigate]);

  const handleRoleSelect = (role) => {
    if (isLoading) return;
    
    setSelectedRole(role);
    setMessages([
      {
        id: 'role-confirm',
        content: `Great choice! You've selected ${role} access.`
      },
      {
        id: 'email-info',
        content: role === 'student' ? (
          <div className="email-info">
            <p><strong>Important:</strong> Students must use their PES University email address:</p>
            <ul>
              <li>@pes.edu</li>
              <li>@pesu.pes.edu</li>
            </ul>
          </div>
        ) : (
          <div className="email-info">
            <p><strong>Note:</strong> Teachers can use:</p>
            <ul>
              <li>PES University email (@pes.edu, @pesu.pes.edu)</li>
              <li>Personal email (Gmail, Yahoo, etc.)</li>
            </ul>
          </div>
        )
      },
      {
        id: 'signin-prompt',
        content: (
          <div className="auth-section">
            <button
              className="google-auth-button"
              onClick={() => handleGoogleSignIn(role)}
              disabled={isLoading}
            >
              <img src="https://www.google.com/favicon.ico" alt="Google" className="google-icon" />
              Sign in with Google
            </button>
          </div>
        )
      }
    ]);
  };

  const handleGoogleSignIn = async (role) => {
    if (isLoading) return;
    setIsLoading(true);
    try {
      await initiateGoogleAuth(role);
    } catch (error) {
      console.error('Auth error:', error);
      setIsLoading(false);
      setMessages([{
        id: 'auth-error',
        content: 'Failed to initialize authentication. Please try again.'
      }]);
    }
  };

  const handleAuthCallback = async () => {
    if (isLoading) return;
    setIsLoading(true);
    
    try {
      const urlParams = new URLSearchParams(window.location.search);
      const error = urlParams.get('error');
      
      if (error) {
        throw new Error(decodeURIComponent(error));
      }
      
      const response = await handleGoogleCallback();
      if (!response?.user || !response?.token) {
        throw new Error('Invalid authentication response');
      }

      // Store auth data
      localStorage.setItem('user', JSON.stringify(response.user));
      localStorage.setItem('token', response.token);

      // Show success message
      setMessages([{
        id: 'auth-success',
        content: 'Authentication successful! Redirecting to dashboard...'
      }]);

      // Redirect after a short delay
      setTimeout(() => {
        if (response.user.role === 'teacher') {
          navigate('/teacher-dashboard', { replace: true });
        } else {
          navigate('/dashboard', { replace: true });
        }
      }, 1500);

    } catch (error) {
      console.error('Auth error:', error);
      localStorage.clear();
      
      // Show specific error message
      setMessages([{
        id: 'auth-error',
        content: `Authentication failed: ${error.message}`
      }]);
      
      // Add a retry button after error
      setTimeout(() => {
        setMessages(prev => [...prev, {
          id: 'retry-auth',
          content: (
            <div className="auth-retry">
              <p>Please try again with the correct account.</p>
              <button
                className="retry-button"
                onClick={() => navigate('/', { replace: true })}
              >
                Return to Login
              </button>
            </div>
          )
        }]);
      }, 1000);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      {showSplash && <Splash />}
      
      <Particles
        id="tsparticles"
        init={particlesInit}
        options={{
          background: { color: { value: "#030711" } },
          fpsLimit: 120,
          particles: {
            color: { value: "#ffffff" },
            links: { enable: false },
            move: {
              enable: true,
              direction: "none",
              outModes: { default: "out" },
              random: true,
              speed: 0.3,
              straight: false
            },
            number: {
              density: { enable: true, area: 800 },
              value: 100
            },
            opacity: {
              value: 0.5,
              animation: {
                enable: true,
                speed: 0.5,
                minimumValue: 0.1
              }
            },
            size: { value: { min: 1, max: 3 } }
          },
          detectRetina: true
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
          {messages.map((msg) => (
            <div key={msg.id} className="message">
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
    </>
  );
}

export default Header;