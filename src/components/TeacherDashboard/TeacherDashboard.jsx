import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { FaUser } from 'react-icons/fa';
import Particles from 'react-particles';
import { loadSlim } from "tsparticles-slim";
import Splash from '../Splash/Splash';
import './TeacherDashboard.css';
import QuestionReview from './QuestionReview';
import QuestionView from './QuestionView';

function TeacherDashboard() {
  const navigate = useNavigate();
  const [user, setUser] = useState(null);
  const [showSplash, setShowSplash] = useState(true);
  const [activeTab, setActiveTab] = useState('content');
  const [selectedCourse, setSelectedCourse] = useState(null);
  const [messages, setMessages] = useState([]);
  const [initialized, setInitialized] = useState(false);
  const [generatedQuestions, setGeneratedQuestions] = useState({ MCQs: [], Descriptive: [] });
  const [showQuestionReview, setShowQuestionReview] = useState(false);
  const [uploadedContent, setUploadedContent] = useState('');
  const [promptSettings, setPromptSettings] = useState({
    style: 'academic',
    difficulty: 'moderate',
    questionTypes: ['multiple-choice', 'short-answer'],
    includeExplanations: true,
    topicFocus: '',
    customInstructions: ''
  });
  const [uploadProgress, setUploadProgress] = useState(null);
  const [isaSchedule, setIsaSchedule] = useState({
    date: '',
    time: '',
    duration: 60, // default duration in minutes
  });
  const [selectedTopic, setSelectedTopic] = useState('');
  const [text, setText] = useState('');
  const [questions, setQuestions] = useState([]);
  const [loadingQuestions, setLoadingQuestions] = useState(false);
  const [prompt, setPrompt] = useState('');
  const [modelResponse, setModelResponse] = useState('');

  const courses = [
    { 
      id: 'UE22CS352A', 
      name: 'Machine Learning',
      topics: ['Introduction to ML', 'Supervised Learning', 'Neural Networks']
    }
  ];

  const addMessage = (msg) => {
    setMessages(prev => [...prev, msg]);
  };

  useEffect(() => {
    const userData = localStorage.getItem('user');
    if (!userData) {
      navigate('/', { replace: true });
      return;
    }

    const parsedUser = JSON.parse(userData);
    if (parsedUser.role !== 'teacher') {
      navigate('/dashboard', { replace: true });
      return;
    }

    setUser(parsedUser);

    // Only initialize messages once
    if (!initialized && messages.length === 0) {
      const currentHour = new Date().getHours();
      const greeting = currentHour < 12 ? 'Good morning' : 
                      currentHour < 17 ? 'Good afternoon' : 
                      'Good evening';
      
      addMessage({ content: `${greeting}, Professor ${parsedUser.name}! 👋` });
      addMessage({ content: "Your dedication to shaping young minds is truly appreciated." });
      addMessage({ content: "Here are the courses that PESUprep is currently offering. You can manage content, create assessments, and track student progress." });
      addMessage({ content: "Select a course to get started. I'm here to help you every step of the way!" });
      
      setInitialized(true);
    }

    setTimeout(() => setShowSplash(false), 2000);
  }, [navigate, initialized, messages.length]);

  const handleLogout = () => {
    localStorage.removeItem('teacherMessages');
    localStorage.removeItem('user');
    localStorage.removeItem('token');
    navigate('/', { replace: true });
  };

  const particlesInit = async (engine) => {
    await loadSlim(engine);
  };

  const handleTabClick = (tab) => {
    setActiveTab(tab);
    const tabMessages = {
      content: "Let's manage your course content! Select a course to begin.",
      questions: "Ready to create and review questions? Choose a course to start generating questions.",
      progress: "Let's check how your students are progressing! Select a course to view detailed analytics."
    };
    addMessage({ content: tabMessages[tab] });
  };

  const handleCourseSelect = (course) => {
    setSelectedCourse(course);
    addMessage({ content: `You've selected ${course.name}. What would you like to do with this course?` });
    addMessage({ content: "You can paste text to generate questions or manage ISA schedules." });
  };

  const handleGenerateQuestions = async () => {
    setLoadingQuestions(true);
    try {
      const response = await fetch('http://localhost:8000/generate-questions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate questions');
      }

      const data = await response.json();
      console.log("Backend response:", data); // Debug log

      // Check if data has the expected structure
      if (data && Array.isArray(data.MCQs) && Array.isArray(data.Descriptive)) {
        setGeneratedQuestions({
          MCQs: data.MCQs,
          Descriptive: data.Descriptive,
        });
        
        if (data.MCQs.length === 0 && data.Descriptive.length === 0) {
          addMessage({ content: "No questions could be generated. Please try with different text.", type: 'warning' });
        } else {
          addMessage({ content: "Questions generated successfully!" });
        }
      } else {
        console.error("Invalid response structure:", data);
        throw new Error('Invalid response structure from backend');
      }
    } catch (error) {
      console.error('Error generating questions:', error);
      addMessage({ 
        content: `Error: ${error.message}. Please try again with different text.`, 
        type: 'error' 
      });
      setGeneratedQuestions({ MCQs: [], Descriptive: [] });
    } finally {
      setLoadingQuestions(false);
    }
  };

  const handleDatasetUpload = async (files) => {
    const formData = new FormData();
    for (let file of files) {
      formData.append('file', file);
    }

    try {
      const response = await fetch('http://localhost:8000/upload-dataset', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to upload dataset');
      }

      const data = await response.json();
      console.log(data.message);
    } catch (error) {
      console.error('Error uploading dataset:', error);
    }
  };

  const PromptSettings = ({ settings, onUpdate }) => {
    return (
      <div className="prompt-settings">
        <h4>Question Generation Settings</h4>
        
        <div className="setting-group">
          <label>Style:</label>
          <select 
            value={settings.style} 
            onChange={e => onUpdate({ ...settings, style: e.target.value })}
          >
            <option value="academic">Academic</option>
            <option value="conversational">Conversational</option>
            <option value="socratic">Socratic</option>
          </select>
        </div>

        <div className="setting-group">
          <label>Difficulty:</label>
          <select 
            value={settings.difficulty}
            onChange={e => onUpdate({ ...settings, difficulty: e.target.value })}
          >
            <option value="easy">Easy</option>
            <option value="moderate">Moderate</option>
            <option value="challenging">Challenging</option>
          </select>
        </div>

        <div className="setting-group">
          <label>Question Types:</label>
          <div className="checkbox-group">
            {['multiple-choice', 'short-answer', 'true-false', 'fill-in-blank'].map(type => (
              <label key={type}>
                <input
                  type="checkbox"
                  checked={settings.questionTypes.includes(type)}
                  onChange={e => {
                    const types = e.target.checked 
                      ? [...settings.questionTypes, type]
                      : settings.questionTypes.filter(t => t !== type);
                    onUpdate({ ...settings, questionTypes: types });
                  }}
                />
                {type.split('-').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
              </label>
            ))}
          </div>
        </div>

        <div className="setting-group">
          <label>Topic Focus:</label>
          <input
            type="text"
            value={settings.topicFocus}
            onChange={e => onUpdate({ ...settings, topicFocus: e.target.value })}
            placeholder="E.g., Neural Networks, Linear Regression"
          />
        </div>

        <div className="setting-group">
          <label>Custom Instructions:</label>
          <textarea
            value={settings.customInstructions}
            onChange={e => onUpdate({ ...settings, customInstructions: e.target.value })}
            placeholder="Any specific requirements or focus areas..."
          />
        </div>
      </div>
    );
  };

  const generatePrompt = (content, settings) => {
    const difficultyMap = {
      easy: "basic understanding",
      moderate: "intermediate comprehension",
      challenging: "advanced analysis"
    };

    return `
      Generate questions from the following content that test ${difficultyMap[settings.difficulty]}.
      Style: ${settings.style}
      ${settings.topicFocus ? `Focus on aspects related to: ${settings.topicFocus}` : ''}
      Question types to include: ${settings.questionTypes.join(', ')}
      ${settings.includeExplanations ? 'Include detailed explanations for each answer' : ''}
      ${settings.customInstructions ? `Additional instructions: ${settings.customInstructions}` : ''}
      
      Content: ${content}
    `.trim();
  };

  const getDifficultyRange = (difficulty) => {
    switch (difficulty) {
      case 'easy':
        return [0.3, 0.5];
      case 'moderate':
        return [0.5, 0.7];
      case 'challenging':
        return [0.7, 0.9];
      default:
        return [0.5, 0.7];
    }
  };

  const handleQuizGeneration = async (courseId, topic, type = 'quiz') => {
    try {
      setUploadProgress({
        status: 'Processing content...',
        progress: 50
      });

      // Get prompt settings
      const prompt = `Generate ${type === 'isa' ? 'difficult' : 'standard'} questions 
                     with ${promptSettings.style} style, 
                     focusing on ${promptSettings.topicFocus || topic}
                     ${promptSettings.customInstructions}`.trim();

      // Create FormData object
      const formData = new FormData();
      formData.append('prompt', prompt);
      formData.append('topic', topic);
      formData.append('type', type);
      
      // If we have uploaded content, add it
      if (uploadedContent) {
        formData.append('content', uploadedContent);
      }

      const response = await fetch('http://localhost:5001/upload-files', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('Failed to process content');
      }

      const data = await response.json();
      console.log('Received data:', data); // Debug log

      if (data.questions && data.questions.length > 0) {
        setGeneratedQuestions(data.questions);
        setShowQuestionReview(true);
        setUploadProgress(null);
        
        addMessage({
          content: `Generated ${data.questions.length} questions successfully`
        });
      } else {
        throw new Error('No questions were generated');
      }

    } catch (error) {
      console.error('Error generating questions:', error);
      addMessage({
        content: `Error: ${error.message}`,
        type: 'error'
      });
      setUploadProgress(null);
    }
  };

  const handleIsaSchedule = async (courseId) => {
    try {
      // Validate date and time
      if (!isaSchedule.date || !isaSchedule.time) {
        throw new Error('Please select both date and time');
      }

      // Save ISA schedule
      const response = await fetch('http://localhost:5000/api/teacher/isa/schedule', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          courseId,
          ...isaSchedule
        })
      });

      if (response.ok) {
        addMessage({
          content: `ISA scheduled successfully for ${isaSchedule.date} at ${isaSchedule.time}`
        });
      }
    } catch (error) {
      console.error('ISA scheduling error:', error);
      addMessage({
        content: `Error scheduling ISA: ${error.message}`,
        type: 'error'
      });
    }
  };

  const handleEditIsa = (isa) => {
    setIsaSchedule({
      date: isa.date,
      time: isa.time,
      duration: isa.duration
    });
    // You can also set an editingIsa state if needed
  };

  const handleDeleteIsa = async (isaId) => {
    try {
      const response = await fetch(`http://localhost:5000/api/teacher/isa/${isaId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });

      if (response.ok) {
        addMessage({
          content: 'ISA deleted successfully'
        });
        // Refresh course data
      }
    } catch (error) {
      console.error('Error deleting ISA:', error);
    }
  };

  const handleApprove = (questionId) => {
    console.log(`Approved question with ID: ${questionId}`);
  };

  const handleDelete = (questionId) => {
    console.log(`Deleted question with ID: ${questionId}`);
  };

  // Function to handle question review
  const handleQuestionReview = (questions) => {
    setGeneratedQuestions(questions);
    setShowQuestionReview(true);
  };

  // Function to handle feedback submission
  const handleFeedback = async (questionId, feedback) => {
    try {
      const response = await fetch('/api/questions/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ questionId, feedback }),
      });

      if (!response.ok) {
        throw new Error('Failed to submit feedback');
      }

      const data = await response.json();
      console.log('Feedback submitted:', data.message);
    } catch (error) {
      console.error('Error submitting feedback:', error);
    }
  };

  return (
    <div className="teacher-dashboard">
      {showSplash && <Splash />}
      
      <Particles
        id="tsparticles"
        init={particlesInit}
        options={{
          background: {
            color: { value: "#030711" }
          },
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
      
      <header className="dashboard-header">
        <div className="header-content">
          <div className="logo-section">
            <div className="brand-name">
              <img src="/pesu.png" alt="PESU" className="logo" /> 
              PESU<span>Prep</span>
            </div>
          </div>
          {user && (
            <div className="user-section">
              <div className="user-info">
                <div className="user-details">
                  <h2>{user.name}</h2>
                  <p>{user.email}</p>
                  <span className="role-badge">Teacher</span>
                </div>
              </div>
              <button className="logout-button" onClick={handleLogout}>
                Logout
              </button>
            </div>
          )}
        </div>
      </header>

      <main className="dashboard-main">
        <div className="chat-container">
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

        <div className="dashboard-tabs">
          <button 
            className={`tab ${activeTab === 'content' ? 'active' : ''}`}
            onClick={() => handleTabClick('content')}
          >
            Course Content
          </button>
          <button 
            className={`tab ${activeTab === 'questions' ? 'active' : ''}`}
            onClick={() => handleTabClick('questions')}
          >
            Question Bank
          </button>
          <button 
            className={`tab ${activeTab === 'progress' ? 'active' : ''}`}
            onClick={() => handleTabClick('progress')}
          >
            Student Progress
          </button>
        </div>

        <div className="dashboard-content">
          {activeTab === 'content' && (
            <div className="content-section">
              <h2>Course Content Management</h2>
              <div className="courses-grid">
                {courses.map(course => (
                  <div key={course.id} 
                       className={`course-card ${selectedCourse?.id === course.id ? 'selected' : ''}`}
                       onClick={() => handleCourseSelect(course)}>
                    <div className="course-header">
                      <h3>{course.name}</h3>
                      <p className="course-id">{course.id}</p>
                    </div>
                    
                    {selectedCourse?.id === course.id && (
                      <div className="course-content-manager">
                        <div className="content-types">
                          <div className="section-grid">
                            <div className="section-card">
                              <h4>Generate Questions</h4>
                              <textarea
                                value={text}
                                onChange={(e) => setText(e.target.value)}
                                placeholder="Paste text here to generate questions..."
                                rows={6}
                                className="text-input"
                              />
                              <select 
                                value={selectedTopic} 
                                onChange={(e) => setSelectedTopic(e.target.value)}
                                className="topic-select"
                              >
                                <option value="">Select Topic</option>
                                {course.topics.map(topic => (
                                  <option key={topic} value={topic}>{topic}</option>
                                ))}
                              </select>
                              <button 
                                className="generate-btn"
                                onClick={handleGenerateQuestions}
                                disabled={!selectedTopic || !text.trim()}
                              >
                                Generate Questions
                              </button>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'questions' && (
            <div className="questions-section">
              <h2>Question Generation & Review</h2>
              {loadingQuestions ? (
                <p>Generating questions...</p>
              ) : (
                <div className="generated-questions">
                  <h3>Prompt Used</h3>
                  <pre>{prompt}</pre>
                  <h3>Model Response</h3>
                  <pre>{modelResponse}</pre>
                  {generatedQuestions.MCQs.length > 0 || generatedQuestions.Descriptive.length > 0 ? (
                    <>
                      <h3>Generated Questions</h3>
                      {generatedQuestions.MCQs.map((question, index) => (
                        <div key={index} className="question-review">
                          <p><strong>Question:</strong> {question.question}</p>
                          <ul>
                            {question.options.map((option, i) => (
                              <li key={i}>
                                <strong>{String.fromCharCode(65 + i)})</strong> {option}
                              </li>
                            ))}
                          </ul>
                          <p><strong>Correct Answer:</strong> {question.answer}</p>
                          <div className="question-actions">
                            <button onClick={() => handleApprove(index)}>Approve</button>
                            <button onClick={() => handleDelete(index)}>Reject</button>
                          </div>
                        </div>
                      ))}
                      {generatedQuestions.Descriptive.map((question, index) => (
                        <div key={`desc-${index}`} className="question-review">
                          <p>{question}</p>
                          <div className="question-actions">
                            <button onClick={() => handleApprove(index)}>Approve</button>
                            <button onClick={() => handleDelete(index)}>Reject</button>
                          </div>
                        </div>
                      ))}
                    </>
                  ) : (
                    <p>No questions generated yet. Select a course and topic to generate questions.</p>
                  )}
                </div>
              )}
            </div>
          )}

          {activeTab === 'progress' && (
            <div className="progress-section">
              <h2>Student Progress Tracking</h2>
              {/* Progress tracking interface */}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default TeacherDashboard;