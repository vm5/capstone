import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Particles from 'react-particles';
import { loadSlim } from "tsparticles-slim";
import Splash from '../Splash/Splash';
import './Dashboard.css';

function Dashboard() {
  const navigate = useNavigate();
  const [user, setUser] = useState(null);
  const [messages, setMessages] = useState([]);
  const [selectedCourse, setSelectedCourse] = useState(null);
  const [showSplash, setShowSplash] = useState(true);

  const courses = [
    { 
      id: 'UE22CS352A', 
      name: 'Machine Learning',
      sections: {
        explore: [
          { title: 'Introduction to ML', type: 'video' },
          { title: 'Supervised Learning', type: 'document' },
          { title: 'Neural Networks', type: 'interactive' }
        ],
        mockQuizzes: [
          { title: 'Introduction to Machine Learning quiz', status: 'available' },
          { title: 'Supervised Learning quiz', status: 'available' },
          { title: 'Neural Networks Practice', status: 'available' }
        ],
        isa1: { status: 'not-scheduled', date: null }
      }
    }
  ];

  const particlesInit = async (engine) => {
    await loadSlim(engine);
  };

  const handleLogout = () => {
    localStorage.removeItem('user');
    localStorage.removeItem('token');
    navigate('/', { replace: true });
  };

  useEffect(() => {
    const userData = localStorage.getItem('user');
    if (!userData) {
      navigate('/', { replace: true });
      return;
    }

    const parsedUser = JSON.parse(userData);
    setUser(parsedUser);

    setTimeout(() => {
      setShowSplash(false);
    }, 2000);

    const initialMessages = [
      {
        content: `Welcome back, ${parsedUser.name}! 👋`,
        delay: 2500
      },
      {
        content: "Select which course you want to explore:",
        delay: 3500
      }
    ];

    initialMessages.forEach((msg, index) => {
      setTimeout(() => {
        addMessage(msg);
      }, msg.delay);
    });
  }, [navigate]);

  const handleCourseSelect = async (course) => {
    if (selectedCourse?.id === course.id) {
      setSelectedCourse(null);
      return;
    }
    
    setSelectedCourse(course);
    try {
      const response = await fetch('http://localhost:5000/api/enroll', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({ courseId: course.id, courseName: course.name })
      });

      if (response.ok) {
        addMessage({
          content: `You're now viewing ${course.name}. Here's what you can explore:`,
          delay: 0
        });
      }
    } catch (error) {
      console.error('Enrollment error:', error);
    }
  };

  const addMessage = (msg) => {
    setMessages(prev => [...prev, msg]);
  };

  return (
    <div className="dashboard">
      {showSplash && <Splash />}
      
      <Particles
        id="tsparticles"
        init={particlesInit}
        options={{
          background: {
            color: {
              value: "#030711",
            },
          },
          fpsLimit: 120,
          particles: {
            color: {
              value: "#ffffff",
            },
            links: {
              enable: false,
            },
            move: {
              enable: true,
              direction: "none",
              outModes: {
                default: "out",
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
      
      <header className="dashboard-header">
        <div className="header-content">
          <div className="logo-section">
            <img src="/pesu.png" alt="PESU" className="logo" />
            <div className="brand-name">
              PESU<span>Prep</span>
            </div>
          </div>
          {user && (
            <div className="user-section">
              <div className="user-info">
                <img src={user.picture} alt={user.name} className="user-avatar" />
                <div className="user-details">
                  <h2>{user.name}</h2>
                  <p>{user.email}</p>
                </div>
              </div>
              <button className="logout-button" onClick={handleLogout}>
                <span className="logout-icon">↪</span>
                Logout
              </button>
            </div>
          )}
        </div>
      </header>

      <main className="dashboard-main">
        <div className="chat-container">
          <div className="messages-wrapper">
            {messages.map((msg, index) => (
              <div key={index} className="message">
                <div className="avatar">
                  <img src="/bot.png" alt="AI Assistant" className="bot-avatar" />
                </div>
                <div className="message-content">
                  {msg.content}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="courses-grid">
          {courses.map(course => (
            <div 
              key={course.id} 
              className={`course-card ${selectedCourse?.id === course.id ? 'selected' : ''}`}
              onClick={() => handleCourseSelect(course)}
            >
              <div className="course-header">
                <h3>{course.name}</h3>
                <p className="course-id">{course.id}</p>
              </div>
              
              {selectedCourse?.id === course.id && (
                <div className="course-sections">
                  <div className="section">
                    <h4>Explore</h4>
                    <ul>
                      {course.sections.explore.map((item, index) => (
                        <li key={index} className={`resource ${item.type}`}>
                          {item.title}
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div className="section">
                    <h4>Mock Quizzes</h4>
                    <ul>
                      {course.sections.mockQuizzes.map((quiz, index) => (
                        <li key={index} className={`quiz ${quiz.status}`}>
                          {quiz.title}
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div className="section">
                    <h4>ISA 1</h4>
                    <div className="isa-status">
                      {course.sections.isa1.status === 'not-scheduled' ? (
                        <p className="not-scheduled">Test not scheduled yet</p>
                      ) : (
                        <p className="scheduled">
                          Scheduled for: {new Date(course.sections.isa1.date).toLocaleDateString()}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </main>

      <footer className="dashboard-footer">
        <div className="footer-content">
          <div className="footer-brand">
            PESU<span>Prep</span>
          </div>
          <div className="footer-location">Bengaluru, India</div>
          <div className="footer-year">© {new Date().getFullYear()}</div>
        </div>
      </footer>
    </div>
  );
}

export default Dashboard; 