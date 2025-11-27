import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import Particles from 'react-particles';
import { loadSlim } from "tsparticles-slim";
import Splash from '../Splash/Splash';
import './TeacherDashboard.css';
import QuestionReview from './QuestionReview';
import QuestionView from './QuestionView';
import ErrorBoundary from '../ErrorBoundary';
import { 
  uploadFiles, 
  createSchedule, 
  getSchedules, 
  updateSchedule, 
  deleteSchedule,
  createPaper,
  getPapers,
  getPaperById,
  updatePaper,
  deletePaper,
  getPapersByCourse,
  downloadPaper
} from '../../services/api';
import QuestionDisplay from './QuestionDisplay';
import { FaPlus, FaEdit, FaTrash } from 'react-icons/fa';
import ExamCreator from './ExamCreator';
import PaperView from './PaperView';

const modelLogStyles = {
  processLog: {
    background: '#1a1a1a',
    padding: '15px',
    borderRadius: '8px',
    marginBottom: '20px',
    fontFamily: 'monospace',
    color: '#fff'  // Adding this to ensure text is visible
  },
  logEntry: {
    margin: '8px 0',
    padding: '8px',
    borderLeft: '3px solid #2196f3',
    backgroundColor: '#252525'
  }
};

function TeacherDashboard() {
  const navigate = useNavigate();
  const [user, setUser] = useState(null);
  const [showSplash, setShowSplash] = useState(true);
  const [activeTab, setActiveTab] = useState('content');
  const [selectedCourse, setSelectedCourse] = useState({
    courseId: 'UE22CS352A',
    name: 'Machine Learning',
    description: 'Machine Learning course for 5th semester'
  });
  const [messages, setMessages] = useState([]);
  const [generatedQuestions, setGeneratedQuestions] = useState({
    MCQs: [],
    Descriptive: [],
    TrueFalse: [],
    FillBlank: []
  });
  const [processedContent, setProcessedContent] = useState({ results: [] });
  const [isGenerating, setIsGenerating] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(null);
  const [mcqCount, setMcqCount] = useState(4);
  const [descriptiveCount, setDescriptiveCount] = useState(4);
  const [trueFalseCount, setTrueFalseCount] = useState(4);
  const [fillBlanksCount, setFillBlanksCount] = useState(4);
  const [papers, setPapers] = useState([]);
  const [showPaperForm, setShowPaperForm] = useState(false);
  const [loadingQuestions, setLoadingQuestions] = useState(false);
  const [contentText, setContentText] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [feedbackHistory, setFeedbackHistory] = useState([]);
  const [generationStep, setGenerationStep] = useState('');
  const [showQuestionReview, setShowQuestionReview] = useState(false);
  const [text, setText] = useState('');
  const [schedules, setSchedules] = useState([]);
  const [showScheduleForm, setShowScheduleForm] = useState(false);
  const [selectedSchedule, setSelectedSchedule] = useState(null);
  const [scheduleFormData, setScheduleFormData] = useState({
    title: '',
    type: 'ISA',
    date: '',
    time: '',
    duration: 60,
    courseId: '',
    description: '',
    paperId: '' // Add paperId to the form data
  });
  const [showExamCreator, setShowExamCreator] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Add missing state variables
  const [promptSettings, setPromptSettings] = useState({
    style: 'academic',
    difficulty: 'moderate',
    questionTypes: ['multiple-choice', 'descriptive'],
    topicFocus: '',
    customInstructions: '',
    includeExplanations: true
  });
  const [uploadedContent, setUploadedContent] = useState(null);
  const [isaDate, setIsaDate] = useState('');
  const [isaTime, setIsaTime] = useState('');
  const [isaSchedule, setIsaSchedule] = useState({
    date: '',
    time: '',
    duration: 60
  });
  const [modelLearning, setModelLearning] = useState(null);
  const [quizTitle, setQuizTitle] = useState('');
  const [quizDate, setQuizDate] = useState('');
  const [selectedPaper, setSelectedPaper] = useState(null);
  const [showPaperView, setShowPaperView] = useState(false);

  const courses = [
    { 
      id: 'UE22CS352A',
      courseId: 'UE22CS352A',
      name: 'Machine Learning',
      description: 'Machine Learning course for 5th semester',
      topics: [
        { title: 'Introduction to ML', type: 'video' },
        { title: 'Supervised Learning', type: 'document' },
        { title: 'Neural Networks', type: 'interactive' }
      ]
    },
    {
      id: 'UE22CS351A',
      courseId: 'UE22CS351A',
      name: 'DBMS',
      description: 'Databases: SQL, normalization, transactions',
      topics: [
        { title: 'Relational Model', type: 'document' },
        { title: 'SQL Joins', type: 'interactive' },
        { title: 'Transactions & ACID', type: 'video' }
      ]
    }
  ];

  // Initialize user and course
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
    setSelectedCourse(courses[0]);
    setTimeout(() => setShowSplash(false), 2000);
  }, [navigate]);

  // Add message helper
  const addMessage = useCallback((msg) => {
    const timestamp = Date.now();
    const messageId = `msg-${timestamp}-${Math.random().toString(36).substr(2, 9)}`;
    setMessages(prev => [...prev, { 
      ...msg, 
      id: messageId,
      timestamp 
    }]);
  }, []);

  const particlesInit = async (engine) => {
    await loadSlim(engine);
  };

  const handleTabClick = (tab) => {
    setActiveTab(tab);
  };

  const handleCourseSelect = (course) => {
    setSelectedCourse(course);
    // Reset course-scoped UI where applicable
    setGeneratedQuestions({ MCQs: [], Descriptive: [], TrueFalse: [], FillBlank: [] });
    setPapers([]);
    setSchedules([]);
    setActiveTab('content');
    setScheduleFormData({
      title: '',
      type: 'ISA',
      date: '',
      time: '',
      duration: 60,
      courseId: '',
      description: '',
      paperId: ''
    });
  };

  const handleGenerateQuestions = async () => {
    try {
        setIsGenerating(true);
        addMessage({
            content: "Loading model and generating questions...",
            type: 'info'
        });

        // First, load the model
        const modelResponse = await fetch('http://localhost:8000/load-model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (!modelResponse.ok) {
            const errorData = await modelResponse.json();
            throw new Error(errorData.error || 'Failed to load model');
        }

        const modelData = await modelResponse.json();
        if (!modelData.success) {
            throw new Error(modelData.error || 'Failed to load model');
        }

      // Now generate questions
        const response = await fetch('http://localhost:8000/generate-questions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                counts: {
                    mcq: mcqCount,
                    descriptive: descriptiveCount,
                    true_false: trueFalseCount,
                    fill_in_blanks: fillBlanksCount
                },
                courseId: selectedCourse.courseId
            })
        });

        if (!response.ok) {
        throw new Error('Failed to generate questions');
        }

        const data = await response.json();
      if (!data.success) {
        throw new Error(data.error || 'Failed to generate questions');
      }

      // Ensure questions object has the correct structure
      const formattedQuestions = {
        MCQs: Array.isArray(data.questions?.MCQs) ? data.questions.MCQs : [],
        Descriptive: Array.isArray(data.questions?.Descriptive) ? data.questions.Descriptive : [],
        TrueFalse: Array.isArray(data.questions?.TrueFalse) ? data.questions.TrueFalse : [],
        FillBlank: Array.isArray(data.questions?.FillBlank) ? data.questions.FillBlank : []
      };

      setGeneratedQuestions(formattedQuestions);
            setActiveTab('questions');
            
            // Show success message with counts
      const counts = Object.entries(formattedQuestions)
                .map(([type, questions]) => `${questions.length} ${type}`)
                .join(', ');
            
            addMessage({
                content: `Generated questions successfully: ${counts}. Check them in the Question Bank tab!`,
                type: 'success'
            });
    } catch (error) {
        console.error('Error generating questions:', error);
        addMessage({
            content: `Error generating questions: ${error.message}. Please try uploading your content again.`,
            type: 'error'
        });
    } finally {
        setIsGenerating(false);
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

  const getDifficultyLabel = (difficulty) => {
    if (typeof difficulty === 'string') {
      return difficulty; // If already a string label, return as is
    }
    
    // Convert numeric difficulty to label
    if (difficulty <= 0.4) {
      return 'easy';
    } else if (difficulty <= 0.7) {
      return 'moderate';
    } else {
      return 'challenging';
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

  const handleScheduleIsa = async () => {
    if (!selectedCourse) {
      addMessage({
        content: "Please select a course first",
        type: 'error'
      });
      return;
    }

    if (!isaDate || !isaTime) {
      addMessage({
        content: "Please select both date and time",
        type: 'error'
      });
      return;
    }

    try {
      const response = await fetch('http://localhost:5000/api/schedule-isa', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          courseId: selectedCourse.courseId,
          date: isaDate,
          time: isaTime,
          createdBy: user._id
        })
      });

      if (response.ok) {
        const data = await response.json();
        // Add success message
        addMessage({
          content: `ISA scheduled successfully for ${isaDate} at ${isaTime}`,
          type: 'success'
        });

        // Clear the form
        setIsaDate('');
        setIsaTime('');
      } else {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to schedule ISA');
      }
    } catch (error) {
      console.error('Error scheduling ISA:', error);
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

  const handleApprove = async (questionId) => {
    try {
      if (!questionId) {
        addMessage({ 
          content: <FeedbackAnimation 
            type="error" 
            message="Question ID is missing. Cannot approve question." 
          />,
          type: 'error'
        });
        return;
      }

      // Show processing notification
      addMessage({ 
        content: <FeedbackAnimation 
          type="info" 
          message="Saving question to database..." 
        />,
        type: 'info'
      });

      // Find the question from all question types
      const question = [
        ...(generatedQuestions.MCQs || []), 
        ...(generatedQuestions.Descriptive || []),
        ...(generatedQuestions.TrueFalse || []),
        ...(generatedQuestions.FillBlank || [])
      ].find(q => q.id === questionId || q._id === questionId);

      if (!question) {
        throw new Error('Question not found');
      }

      // Format question data according to the required MongoDB schema
      // Normalize question type
      let normalizedType = question.type || (question.options ? 'mcq' : 'descriptive');
      if (normalizedType === 'TrueFalse') normalizedType = 'trueFalse';
      if (normalizedType === 'FillBlank') normalizedType = 'fillInBlanks';
      
      const formattedQuestion = {
        courseId: selectedCourse?.id || selectedCourse?.courseId,
        questionText: question.question || question.text || question.statement,
        type: normalizedType,
        difficulty: question.difficulty || 0.5,
        status: 'approved',
        options: question.options ? 
          question.options.map((opt, idx) => ({
            text: typeof opt === 'string' ? opt : opt.text,
            isCorrect: question.correct_answer === idx || question.answer === opt || opt.isCorrect
          })) : [],
        explanation: question.explanation || '',
        metadata: {
          generatedAt: new Date().toISOString(),
          conceptsCovered: [question.concept || question.topic || ''],
          timeToSolve: question.type === 'mcq' ? 2 : 5
        }
      };

      // For newly generated questions (with UUID-style IDs)
      if (questionId.startsWith('mcq-') || questionId.startsWith('desc-') || 
          questionId.startsWith('tf-') || questionId.startsWith('fb-')) {
        const saveResponse = await fetch('http://localhost:5000/api/questions/save', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(formattedQuestion)
        });

        if (!saveResponse.ok) {
          const errorData = await saveResponse.json();
          throw new Error(errorData.message || 'Failed to save question');
        }

        // Update local state to remove the approved question
        setGeneratedQuestions(prev => ({
          MCQs: prev.MCQs?.filter(q => q.id !== questionId) || [],
          Descriptive: prev.Descriptive?.filter(q => q.id !== questionId) || [],
          TrueFalse: prev.TrueFalse?.filter(q => q.id !== questionId) || [],
          FillBlank: prev.FillBlank?.filter(q => q.id !== questionId) || []
        }));

        // Show success notification
        addMessage({ 
          content: <FeedbackAnimation 
            type="success" 
            message="Question approved and saved successfully!" 
          />,
          type: 'success'
        });
        return;
      }

      // For existing questions in the database
      const reviewResponse = await fetch('http://localhost:5000/api/questions/review', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          questionId,
          status: 'approved',
          timestamp: new Date().toISOString()
        })
      });

      if (!reviewResponse.ok) {
        const errorData = await reviewResponse.json();
        throw new Error(errorData.message || 'Failed to update review status');
      }

      // Update local state
      setGeneratedQuestions(prev => ({
        MCQs: prev.MCQs?.filter(q => q.id !== questionId) || [],
        Descriptive: prev.Descriptive?.filter(q => q.id !== questionId) || [],
        TrueFalse: prev.TrueFalse?.filter(q => q.id !== questionId) || [],
        FillBlank: prev.FillBlank?.filter(q => q.id !== questionId) || []
      }));

      // Show success notification
      addMessage({ 
        content: <FeedbackAnimation 
          type="success" 
          message="Question approved successfully!" 
        />,
        type: 'success'
      });

    } catch (error) {
      console.error('Error approving question:', error);
      addMessage({ 
        content: <FeedbackAnimation 
          type="error" 
          message={`Error approving question: ${error.message}`} 
        />,
        type: 'error' 
      });
    }
  };

  const trackFeedback = (questionId, feedback, status) => {
    const feedbackEntry = {
      questionId,
      feedback,
      status,
      timestamp: new Date().toISOString()
    };
    setFeedbackHistory(prev => [feedbackEntry, ...prev]);
  };

  const handleDelete = async (questionId) => {
    try {
      // Show processing notification
      addMessage({ 
        content: <FeedbackAnimation 
          type="info" 
          message="Processing question rejection..." 
        />,
        type: 'info'
      });

        // Find the question
      const question = [
        ...(generatedQuestions.MCQs || []),
        ...(generatedQuestions.Descriptive || []),
        ...(generatedQuestions.TrueFalse || []),
        ...(generatedQuestions.FillBlank || [])
      ].find(q => q.id === questionId || q._id === questionId);

        if (!question) {
          throw new Error('Question not found');
        }

      // Format question data with feedback
        const formattedQuestion = {
          courseId: selectedCourse?.id,
        questionText: question.question || question.text || question.statement,
        type: question.type || (question.options ? 'mcq' : 'descriptive'),
          difficulty: question.difficulty || 0.5,
          status: 'rejected',
        options: question.options ? 
            question.options.map((opt, idx) => ({
            text: typeof opt === 'string' ? opt : opt.text,
            isCorrect: question.correct_answer === idx || question.answer === opt || opt.isCorrect
            })) : [],
          metadata: {
            generatedAt: new Date().toISOString(),
          conceptsCovered: [question.concept || question.topic || ''],
            timeToSolve: question.type === 'mcq' ? 2 : 5,
          reviewHistory: []
        }
      };

      // Save rejected question with feedback
        const saveResponse = await fetch('http://localhost:5000/api/questions/save', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(formattedQuestion)
        });

        if (!saveResponse.ok) {
          throw new Error('Failed to save rejected question');
        }

        // Update local state
        setGeneratedQuestions(prev => ({
        MCQs: prev.MCQs?.filter(q => q.id !== questionId) || [],
        Descriptive: prev.Descriptive?.filter(q => q.id !== questionId) || [],
        TrueFalse: prev.TrueFalse?.filter(q => q.id !== questionId) || [],
        FillBlank: prev.FillBlank?.filter(q => q.id !== questionId) || []
      }));

        // Show success message
        addMessage({ 
          content: <FeedbackAnimation 
            type="success" 
          message="Question rejected and feedback processed successfully!" 
          />,
          type: 'success'
        });

    } catch (error) {
      console.error('Error rejecting question:', error);
      addMessage({ 
        content: <FeedbackAnimation 
          type="error" 
          message={`Error rejecting question: ${error.message}`} 
        />,
        type: 'error' 
      });
    }
  };

  const submitFeedbackToML = async (questionId, feedback) => {
    try {
      const response = await fetch('http://localhost:8000/learn-from-feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          questionId,
          feedback,
          timestamp: new Date().toISOString()
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (!data.success) {
        throw new Error(data.message || 'Failed to process feedback');
      }

      // Show success animation
      addMessage({ 
        content: <FeedbackAnimation 
          type="success" 
          message="Feedback sent to ML model successfully!" 
        />,
        type: 'success'
      });

      return data;
    } catch (error) {
      console.error('Error submitting feedback:', error);
      // Show error animation
      addMessage({ 
        content: <FeedbackAnimation 
          type="error" 
          message={`Failed to send feedback to ML model: ${error.message}`} 
        />,
        type: 'error'
      });
      return null;
    }
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

  // Add paper generation form component
  const PaperGenerationForm = ({ onSubmit, onCancel }) => {
    const [paperConfig, setPaperConfig] = useState({
      title: '',
      totalMarks: 10,
      duration: 10,
      description: '',
      instructions: '',
      questionTypes: {
        mcq: false,
        descriptive: false,
        trueFalse: false,
        fillInBlanks: false
      }
    });

    const handleSubmit = (e) => {
      e.preventDefault();
      // Validate that at least one question type is selected
      if (!Object.values(paperConfig.questionTypes).some(type => type)) {
        addMessage({
          content: "Please select at least one question type",
          type: 'error'
        });
        return;
      }
      onSubmit(paperConfig);
    };

    return (
      <div className="paper-generation-form">
        <h3>Generate Question Paper</h3>
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label>Paper Title</label>
            <input
              type="text"
              value={paperConfig.title}
              onChange={(e) => setPaperConfig({...paperConfig, title: e.target.value})}
              placeholder="Enter paper title"
              required
            />
          </div>

          <div className="form-group">
            <label>Total Marks (Max 10)</label>
            <input
              type="number"
              value={paperConfig.totalMarks}
              onChange={(e) => setPaperConfig({...paperConfig, totalMarks: Math.min(10, parseInt(e.target.value) || 0)})}
              max={10}
              min={1}
              required
            />
          </div>

          <div className="form-group">
            <label>Duration (Minutes, Max 10)</label>
            <input
              type="number"
              value={paperConfig.duration}
              onChange={(e) => setPaperConfig({...paperConfig, duration: Math.min(10, parseInt(e.target.value) || 0)})}
              max={10}
              min={1}
              required
            />
          </div>

          <div className="form-group">
            <label>Description</label>
            <textarea
              value={paperConfig.description}
              onChange={(e) => setPaperConfig({...paperConfig, description: e.target.value})}
              placeholder="Enter paper description"
              rows={3}
            />
          </div>

          <div className="form-group">
            <label>Instructions</label>
            <textarea
              value={paperConfig.instructions}
              onChange={(e) => setPaperConfig({...paperConfig, instructions: e.target.value})}
              placeholder="Enter instructions for students"
              rows={3}
            />
          </div>

          <div className="form-group">
            <label>Question Types</label>
            <div className="checkbox-group">
              <label>
                <input
                  type="checkbox"
                  checked={paperConfig.questionTypes.mcq}
                  onChange={(e) => setPaperConfig({
                    ...paperConfig,
                    questionTypes: {...paperConfig.questionTypes, mcq: e.target.checked}
                  })}
                />
                Multiple Choice Questions
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={paperConfig.questionTypes.descriptive}
                  onChange={(e) => setPaperConfig({
                    ...paperConfig,
                    questionTypes: {...paperConfig.questionTypes, descriptive: e.target.checked}
                  })}
                />
                Descriptive Questions
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={paperConfig.questionTypes.trueFalse}
                  onChange={(e) => setPaperConfig({
                    ...paperConfig,
                    questionTypes: {...paperConfig.questionTypes, trueFalse: e.target.checked}
                  })}
                />
                True/False Questions
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={paperConfig.questionTypes.fillInBlanks}
                  onChange={(e) => setPaperConfig({
                    ...paperConfig,
                    questionTypes: {...paperConfig.questionTypes, fillInBlanks: e.target.checked}
                  })}
                />
                Fill in the Blanks
              </label>
            </div>
          </div>

          <div className="form-actions">
            <button type="submit" className="generate-btn">Generate Paper</button>
            <button type="button" onClick={onCancel} className="cancel-btn">Cancel</button>
          </div>
        </form>
      </div>
    );
  };

  // Update paper generation handler
  const handleGeneratePaper = async (paperConfig) => {
    try {
      if (!selectedCourse) {
        addMessage({
          content: 'Please select a course first',
          type: 'error'
        });
        return;
      }

      // Validate that at least one question type is selected
      if (!Object.values(paperConfig.questionTypes).some(type => type)) {
        addMessage({
          content: 'Please select at least one question type',
          type: 'error'
        });
        return;
      }

      setLoadingQuestions(true);
      addMessage({
        content: 'Generating paper...',
        type: 'info'
      });

      // Calculate total questions based on selected types
      const questionCounts = {
        mcq: paperConfig.questionTypes.mcq ? 3 : 0,
        descriptive: paperConfig.questionTypes.descriptive ? 2 : 0,
        trueFalse: paperConfig.questionTypes.trueFalse ? 2 : 0,
        fillInBlanks: paperConfig.questionTypes.fillInBlanks ? 2 : 0
      };

      // Ensure at least one type has a non-zero count
      if (Object.values(questionCounts).every(count => count === 0)) {
        addMessage({
          content: 'Please select at least one question type',
          type: 'error'
        });
        return;
      }

      const totalQuestions = Object.values(questionCounts).reduce((a, b) => a + b, 0);

      // Create paper object with proper structure
      const paperData = {
        courseId: selectedCourse.courseId,
        config: {
          title: paperConfig.title || 'Untitled Paper',
          totalMarks: paperConfig.totalMarks || 10,
          duration: paperConfig.duration || 10,
          description: paperConfig.description || '',
          instructions: paperConfig.instructions || '',
          type: 'exam',
          difficulty: {
            easy: 0.6,    // 60% easy questions
            moderate: 0.3, // 30% moderate questions
            challenging: 0.1  // 10% challenging questions
          },
          questionTypes: questionCounts
        }
      };

      console.log('Sending paper data:', paperData);  // Debug log
      const response = await createPaper(paperData);
      console.log('Paper creation response:', response);  // Debug log
      
      // Refresh papers list
      await fetchPapers();
      setShowPaperForm(false);
      
      addMessage({
        content: `Paper generated successfully with ${response.questionCount} questions!`,
        type: 'success'
      });
    } catch (error) {
      console.error('Error generating paper:', error);
      addMessage({
        content: error.message || 'Failed to generate paper',
        type: 'error'
      });
    } finally {
      setLoadingQuestions(false);
    }
  };

  // Update GeneratedPapersList component
  const GeneratedPapersList = ({ papers, onView, onDelete }) => {
    const handleDownload = async (paper) => {
      if (paper.source === 'file') {
        try {
          const blob = await downloadPaper(paper._id);
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = paper._id;
          document.body.appendChild(a);
          a.click();
          window.URL.revokeObjectURL(url);
          document.body.removeChild(a);
        } catch (error) {
          console.error('Error downloading paper:', error);
          addMessage({
            content: `Error downloading paper: ${error.message}`,
            type: 'error'
          });
        }
      }
    };

    if (!papers || papers.length === 0) {
      return (
        <div className="papers-list">
          <div className="no-papers">
            <p>No papers available</p>
          </div>
        </div>
      );
    }

    return (
      <div className="papers-list">
        {papers.map((paper) => (
          <div key={paper._id} className="paper-item">
            <div className="paper-info">
              <h4>{paper.title || 'Untitled Paper'}</h4>
              <div className="paper-details">
                {paper.courseId && (
                  <div className="paper-detail-row">
                    <span className="paper-label">Course:</span>
                    <span className="paper-value">{paper.courseId}</span>
                  </div>
                )}
                {paper.totalMarks && (
                  <div className="paper-detail-row">
                    <span className="paper-label">Total Marks:</span>
                    <span className="paper-value">{paper.totalMarks}</span>
                  </div>
                )}
                {paper.duration && (
                  <div className="paper-detail-row">
                    <span className="paper-label">Duration:</span>
                    <span className="paper-value">{paper.duration} minutes</span>
                  </div>
                )}
                <div className="paper-detail-row">
                  <span className="paper-label">Source:</span>
                  <span className="paper-value">{paper.source || 'database'}</span>
                </div>
                <div className="paper-detail-row">
                  <span className="paper-label">Created:</span>
                  <span className="paper-value">{new Date(paper.createdAt).toLocaleDateString()}</span>
                </div>
              </div>
            </div>
            <div className="paper-actions">
              {paper.source === 'file' ? (
                <button onClick={() => handleDownload(paper)} className="btn-primary">
                  Download
                </button>
              ) : (
                <button onClick={() => onView(paper)} className="btn-primary">
                  View
                </button>
              )}
              {paper.source !== 'file' && (
                <button onClick={() => onDelete(paper._id)} className="btn-danger">
                  Delete
                </button>
              )}
            </div>
          </div>
        ))}
      </div>
    );
  };

  const handleFileUpload = async (event) => {
    try {
        const files = Array.from(event.target.files);
        if (files.length === 0) {
            addMessage({
                content: "Please select at least one file to upload",
                type: 'error'
            });
            return;
        }

        // Check file sizes
        const maxSize = 50 * 1024 * 1024; // 50MB
        const oversizedFiles = files.filter(file => file.size > maxSize);
        if (oversizedFiles.length > 0) {
            addMessage({
                content: `Some files exceed the 50MB limit: ${oversizedFiles.map(f => f.name).join(', ')}`,
                type: 'error'
            });
            return;
        }

        setUploadProgress({
            status: 'Processing files...',
            progress: 0
        });

        const formData = new FormData();
        files.forEach(file => formData.append('files[]', file));
        
        // Add question counts to form data
        formData.append('mcqCount', mcqCount);
        formData.append('descriptiveCount', descriptiveCount);
        formData.append('trueFalseCount', trueFalseCount);
        formData.append('fillBlanksCount', fillBlanksCount);
        formData.append('caseStudyCount', 1); // Default to 1 case study
        formData.append('numericalCount', 1); // Default to 1 numerical question
        formData.append('courseId', selectedCourse.courseId);

        const response = await fetch('http://localhost:8000/process-content', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        if (data.success && data.processed_files && data.processed_files.length > 0) {
            const firstFile = data.processed_files[0];
            setText(firstFile.text);
            
            // Store questions but don't show them yet
            if (firstFile.questions) {
                setGeneratedQuestions(firstFile.questions);
            }

            setProcessedContent({
                results: data.processed_files.map(file => ({
                    filename: file.filename,
                    content: file.text,
                    summary: file.summary
                }))
            });

            addMessage({
                content: "Content processed successfully! You can now generate questions.",
                type: 'success'
            });
            
            // Stay in the current tab (Course Content)
        } else {
            throw new Error('No files were processed');
        }

    } catch (error) {
        console.error('Error processing files:', error);
        addMessage({
            content: `Error processing files: ${error.message}`,
            type: 'error'
        });
        setProcessedContent({ results: [] });
    } finally {
        setUploadProgress(null);
    }
};

  const handleContentSubmit = async () => {
    try {
      if (!contentText.trim()) {
        addMessage({
          content: "Please enter some content first",
          type: 'error'
        });
        return;
      }

      setIsProcessing(true);
      addMessage({
        content: "Processing content...",
        type: 'info'
      });

      const response = await fetch('http://localhost:8000/process-content', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          text: contentText
        })
      });

      const data = await response.json();
      
      if (data.success && data.processed_files && data.processed_files.length > 0) {
        setText(data.processed_files[0].text);
        
        setModelLearning({
          topic: "Machine Learning",
          what_i_learned: data.processed_files[0].summary,
          key_concepts: {},
          content_analysis: {
            topic_confidence: "high",
            main_topics: []
          }
        });

        await handleGenerateQuestions();

        addMessage({
          content: "Content processed and questions generated successfully!",
          type: 'success'
        });
      } else {
        throw new Error(data.error || 'Failed to process content');
      }

    } catch (error) {
      console.error('Error processing content:', error);
      addMessage({
        content: `Error processing content: ${error.message}`,
        type: 'error'
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const renderContentInput = () => (
    <div className="content-input-section">
      <h3>Enter Content for Question Generation</h3>
      <div className="content-input-container">
        <textarea
          className="content-textarea"
          value={contentText}
          onChange={(e) => setContentText(e.target.value)}
          placeholder="Paste your content here (e.g., lecture notes, study material, etc.)"
          rows={10}
          style={{
            width: '100%',
            padding: '12px',
            marginBottom: '16px',
            borderRadius: '8px',
            border: '1px solid #ccc',
            fontSize: '16px',
            resize: 'vertical'
          }}
        />
        <button
          className="process-button"
          onClick={handleContentSubmit}
          disabled={isProcessing || !contentText.trim()}
          style={{
            padding: '12px 24px',
            fontSize: '16px',
            backgroundColor: '#2196f3',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: isProcessing || !contentText.trim() ? 'not-allowed' : 'pointer',
            opacity: isProcessing || !contentText.trim() ? 0.7 : 1
          }}
        >
          {isProcessing ? 'Processing...' : 'Generate Questions'}
        </button>
      </div>
    </div>
  );

  const FeedbackHistory = () => {
    if (feedbackHistory.length === 0) return null;

    return (
      <div className="feedback-history">
        <h4>Recent Feedback History</h4>
        {feedbackHistory.map((entry, index) => (
          <div key={index} className={`feedback-entry feedback-${entry.status}`}>
            <div className="feedback-time">
              {new Date(entry.timestamp).toLocaleTimeString()}
            </div>
            <div className="feedback-content">
              <strong>Status:</strong> {entry.status}
              <br />
              <strong>Feedback:</strong> {entry.feedback}
            </div>
          </div>
        ))}
      </div>
    );
  };

  const handleCancelPaperForm = () => {
    setShowPaperForm(false);
  };

  const LoadingSpinner = () => {
    return (
      <div className="loading-spinner">
        <div className="spinner-container">
          <div className="spinner"></div>
        </div>
        <p>
          {generationStep === 'receiving' && "Receiving content..."}
          {generationStep === 'analyzing' && "Analyzing content..."}
          {generationStep === 'generating' && "Generating questions..."}
          {generationStep === 'complete' && "Questions generated successfully!"}
        </p>
      </div>
    );
  };

  // Add this new function to handle entire paper approval/rejection
  const handlePaperAction = async (action, paperId) => {
    try {
      switch (action) {
        case 'view':
          window.open(`/exam/${paperId}`, '_blank');
          break;
        case 'delete':
          await deletePaper(paperId);
          fetchPapers();
          break;
        default:
          console.error('Unknown action:', action);
      }
    } catch (error) {
      console.error('Error handling paper action:', error);
      setError(error.message);
    }
  };

  const handleSaveQuestions = async () => {
    try {
      if (!selectedCourse) {
        addMessage({
          content: "Please select a course first",
          type: 'error'
        });
        return;
      }

      const allQuestions = [...generatedQuestions.MCQs, ...generatedQuestions.Descriptive];
      if (allQuestions.length === 0) {
        addMessage({
          content: "No questions to save",
          type: 'error'
        });
        return;
      }

      setLoadingQuestions(true);

      // Save questions to the database
      const response = await fetch('http://localhost:5000/api/questions/save-batch', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          courseId: selectedCourse.courseId,
          questions: allQuestions.map(q => ({
            ...q,
            status: 'approved',
            type: q.type || (q.options ? 'mcq' : 'descriptive')
          }))
        })
      });

      if (!response.ok) {
        throw new Error('Failed to save questions');
      }

      const data = await response.json();

      // Clear the generated questions after saving
      setGeneratedQuestions({
        MCQs: [],
        Descriptive: [],
        FillInBlanks: [],
        TrueFalse: []
      });

      addMessage({
        content: `Successfully saved ${data.savedCount} questions!`,
        type: 'success'
      });

    } catch (error) {
      console.error('Error saving questions:', error);
      addMessage({
        content: `Error saving questions: ${error.message}`,
        type: 'error'
      });
    } finally {
      setLoadingQuestions(false);
    }
  };

  const handleClearQuestions = () => {
    // Clear all generated questions
    setGeneratedQuestions({
      MCQs: [],
      Descriptive: [],
      FillInBlanks: [],
      TrueFalse: []
    });

    addMessage({
      content: "All generated questions have been cleared.",
      type: 'info'
    });
  };

  const QuestionDisplay = ({ questions = {}, onApprove, onReject }) => {
    const [showFeedbackModal, setShowFeedbackModal] = useState(false);
    const [selectedQuestionId, setSelectedQuestionId] = useState(null);

    const handleRejectClick = (questionId) => {
      setSelectedQuestionId(questionId);
      setShowFeedbackModal(true);
    };

    const handleFeedbackSubmit = (feedback) => {
      onReject(selectedQuestionId, feedback);
      setShowFeedbackModal(false);
      setSelectedQuestionId(null);
    };

    const handleFeedbackCancel = () => {
      setShowFeedbackModal(false);
      setSelectedQuestionId(null);
    };

    // Ensure questions is an object and each question list is an array
    const safeQuestions = {
      MCQs: Array.isArray(questions.MCQs) ? questions.MCQs : [],
      Descriptive: Array.isArray(questions.Descriptive) ? questions.Descriptive : [],
      TrueFalse: Array.isArray(questions.TrueFalse) ? questions.TrueFalse : [],
      FillBlank: Array.isArray(questions.FillBlank) ? questions.FillBlank : []
    };

    return (
      <div className="question-display">
        {showFeedbackModal && (
          <FeedbackModal
            onSubmit={handleFeedbackSubmit}
            onCancel={handleFeedbackCancel}
          />
        )}

        {Object.entries(safeQuestions).map(([type, questionList]) => (
          questionList.length > 0 && (
            <div key={type} className="question-type-section">
              <h3>{type}</h3>
              {questionList.map((question, index) => (
                <div key={`${type}-${index}`} className="question-item">
                  <div className="question-text">
                    {question.question || question.statement}
                    <span className="difficulty-badge">
                      {question.difficulty ? 
                        typeof question.difficulty === 'number' ? 
                          question.difficulty > 0.7 ? 'Hard' : 
                          question.difficulty > 0.4 ? 'Moderate' : 'Easy'
                        : question.difficulty
                      : 'Moderate'}
                    </span>
                  </div>
                  
                  {question.options && Array.isArray(question.options) && (
                    <div className="options-list">
                      {question.options.map((option, optIndex) => (
                        <div 
                          key={`${type}-${index}-option-${optIndex}`}
                          className={`option-item ${option === question.correct_answer ? 'correct' : ''}`}
                          data-option={String.fromCharCode(65 + optIndex)}
                        >
                          {typeof option === 'string' ? option : option.text}
                        </div>
                      ))}
                    </div>
                  )}

                  {question.explanation && (
                    <div className="explanation-section">
                      <div className="explanation-title">Explanation:</div>
                      <div className="explanation-text">{question.explanation}</div>
                    </div>
                  )}

                  <div className="question-actions">
                    <button 
                      className="approve-btn"
                      onClick={() => onApprove(question.id)}
                    >
                      Approve
                    </button>
                    <button 
                      className="reject-btn"
                      onClick={() => handleRejectClick(question.id)}
                    >
                      Reject
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )
        ))}
      </div>
    );
  };

  // Add this component for feedback animations
  const FeedbackAnimation = ({ type, message }) => {
    return (
      <div className={`feedback-animation ${type}`}>
        <div className="feedback-icon">
          {type === 'success' && '✓'}
          {type === 'error' && '✕'}
          {type === 'info' && 'ℹ'}
        </div>
        <div className="feedback-message">{message}</div>
      </div>
    );
  };

  const FeedbackModal = ({ onSubmit, onCancel }) => {
    const [feedback, setFeedback] = useState('');

    return (
      <div className="feedback-modal-overlay">
        <div className="feedback-modal">
          <div className="feedback-modal-header">
            <h3>Question Rejection Feedback</h3>
            <p>Please provide detailed feedback to help improve the question generation</p>
          </div>
          
          <div className="feedback-modal-content">
            <div className="feedback-input-group">
              <label>What needs improvement?</label>
          <textarea
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
                placeholder="Please explain what's wrong with this question (e.g., unclear wording, incorrect options, too easy/difficult...)"
                rows={5}
            autoFocus
          />
            </div>
            
            <div className="feedback-categories">
              <label>Common Issues:</label>
              <div className="feedback-tags">
                <button 
                  className={`tag-button ${feedback.includes('Unclear wording') ? 'active' : ''}`}
                  onClick={() => setFeedback(prev => prev + ' Unclear wording.')}
                >
                  Unclear wording
            </button>
            <button 
                  className={`tag-button ${feedback.includes('Incorrect options') ? 'active' : ''}`}
                  onClick={() => setFeedback(prev => prev + ' Incorrect options.')}
                >
                  Incorrect options
                </button>
                <button 
                  className={`tag-button ${feedback.includes('Wrong difficulty') ? 'active' : ''}`}
                  onClick={() => setFeedback(prev => prev + ' Wrong difficulty level.')}
                >
                  Wrong difficulty
                </button>
                <button 
                  className={`tag-button ${feedback.includes('Off-topic') ? 'active' : ''}`}
                  onClick={() => setFeedback(prev => prev + ' Off-topic content.')}
                >
                  Off-topic
            </button>
          </div>
        </div>
      </div>
          
          <div className="feedback-modal-actions">
            <button className="cancel-button" onClick={onCancel}>
              Cancel
            </button>
            <button 
              className="submit-button" 
              onClick={() => onSubmit(feedback)}
              disabled={!feedback.trim()}
            >
              Submit Feedback
            </button>
        </div>
            </div>
      </div>
    );
  };

  const ModelTrainingInterface = () => null;

  // Add this new component for displaying the model's learning
  const ModelLearningSection = ({ modelLearning }) => {
    if (!modelLearning) {
      return null;
    }

    const { stats = {}, sample_content = '' } = modelLearning;
    const { word_count = 0, sentence_count = 0, file_count = 0 } = stats;

    return (
      <div className="model-learning-section">
        <h3>Model Learning Summary</h3>
        <div className="learning-content">
          <div className="confidence-level">
            <span className="label">Content Processed:</span>
            <div className="progress-bar">
              <div 
                className="progress" 
                style={{ width: `${Math.min(100, (word_count / 1000) * 100)}%` }}
              />
            </div>
            <span className="value">{word_count} words</span>
          </div>
          
          <div className="learned-topics">
            <h4>Content Statistics</h4>
            <ul>
              <li>
                <span className="topic-name">Sentences Processed</span>
                <span className="topic-confidence">{sentence_count}</span>
              </li>
              <li>
                <span className="topic-name">Files Analyzed</span>
                <span className="topic-confidence">{file_count}</span>
              </li>
            </ul>
          </div>

          {sample_content && (
            <div className="sample-content">
              <h4>Sample Content</h4>
              <p>{sample_content}</p>
            </div>
          )}
        </div>
      </div>
    );
  };

  const handleCreateQuiz = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/create-quiz', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          courseId: selectedCourse.courseId,
          title: quizTitle,
          date: quizDate,
          createdBy: user._id
        })
      });

      if (response.ok) {
        const data = await response.json();
        addMessage({
          content: `Quiz "${quizTitle}" created successfully`,
          type: 'success'
        });
      }
    } catch (error) {
      console.error('Error creating quiz:', error);
      addMessage({
        content: `Error creating quiz: ${error.message}`,
        type: 'error'
      });
    }
  };

  const fetchPapers = async () => {
    try {
      const papers = await getPapersByCourse(selectedCourse.courseId);
      const filtered = Array.isArray(papers) 
        ? papers.filter(p => p.courseId === selectedCourse.courseId)
        : [];
      setPapers(filtered);
    } catch (error) {
      console.error('Error fetching papers:', error);
      addMessage({
        content: `Error fetching papers: ${error.message}`,
        type: 'error'
      });
    }
  };

  useEffect(() => {
    if (activeTab === 'paper') {
      fetchPapers();
    } else if (activeTab === 'schedules') {
      fetchSchedules();
    }
  }, [activeTab, selectedCourse]);

  // Fetch schedules on component mount
  useEffect(() => {
    fetchSchedules();
  }, [selectedCourse]);

  const fetchSchedules = async () => {
    try {
      setLoading(true);
      const data = await getSchedules();
      if (!Array.isArray(data)) {
        throw new Error('Invalid response format');
      }
      setSchedules(data.filter(s => s.courseId === selectedCourse.courseId));
      setError(null);
    } catch (error) {
      setError('Failed to fetch schedules');
      setSchedules(null);
      console.error('Error fetching schedules:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleScheduleSubmit = async (e) => {
    e.preventDefault();
    try {
      // Create date in IST
      const startTime = new Date(`${scheduleFormData.date}T${scheduleFormData.time}+05:30`);
      
      // Calculate end time in IST
      const endTime = new Date(startTime.getTime() + scheduleFormData.duration * 60000);

      // Log date information for debugging
      console.log('Creating schedule with dates (IST):', {
        inputDate: scheduleFormData.date,
        inputTime: scheduleFormData.time,
        startTimeIST: startTime.toLocaleString('en-US', { timeZone: 'Asia/Kolkata' }),
        endTimeIST: endTime.toLocaleString('en-US', { timeZone: 'Asia/Kolkata' })
      });

      const scheduleData = {
        title: scheduleFormData.title,
        type: scheduleFormData.type,
        startTime: startTime.toISOString(),
        endTime: endTime.toISOString(),
        duration: scheduleFormData.duration,
        courseId: selectedCourse.courseId,
        description: scheduleFormData.description,
        paperId: scheduleFormData.paperId,
        createdBy: user._id
      };

      if (selectedSchedule) {
        await updateSchedule(selectedSchedule._id, scheduleData);
        setSchedules(prevSchedules =>
          prevSchedules.map(s => s._id === selectedSchedule._id ? { ...s, ...scheduleData } : s)
        );
        addMessage({
          content: 'Schedule updated successfully',
          type: 'success'
        });
      } else {
        const newSchedule = await createSchedule(scheduleData);
        setSchedules(prevSchedules => [...prevSchedules, newSchedule]);
        addMessage({
          content: 'Schedule created successfully',
          type: 'success'
        });
      }

      setShowScheduleForm(false);
      setSelectedSchedule(null);
      setScheduleFormData({
        title: '',
        type: 'ISA',
        date: '',
        time: '',
        duration: 60,
        courseId: '',
        description: '',
        paperId: ''
      });
    } catch (error) {
      console.error('Error saving schedule:', error);
      addMessage({
        content: `Error saving schedule: ${error.message}`,
        type: 'error'
      });
    }
  };

  const handleEditSchedule = (schedule) => {
    setSelectedSchedule(schedule);
    setScheduleFormData({
      title: schedule.title,
      type: schedule.type,
      date: schedule.date,
      time: schedule.time,
      duration: schedule.duration,
      courseId: schedule.courseId,
      description: schedule.description,
      paperId: schedule.paperId // Add paperId to the form data
    });
    setShowScheduleForm(true);
  };

  const handleDeleteSchedule = async (scheduleId) => {
    if (!window.confirm('Are you sure you want to delete this exam?')) {
      return;
    }

    try {
      setLoading(true);
      const response = await fetch(`/api/schedules/${scheduleId}`, {
        method: 'DELETE'
      });

      if (!response.ok) {
        throw new Error('Failed to delete exam');
      }

      await fetchSchedules();
    } catch (error) {
      setError('Failed to delete exam');
      console.error('Error deleting exam:', error);
    } finally {
      setLoading(false);
    }
  };

  const ScheduleForm = () => {
    const [availablePapers, setAvailablePapers] = useState([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
      const loadPapers = async () => {
        try {
          setLoading(true);
          const papers = await getPapersByCourse(selectedCourse.courseId);
          const filtered = Array.isArray(papers)
            ? papers.filter(p => p.courseId === selectedCourse.courseId)
            : [];
          console.log('Loaded papers (filtered):', filtered);
          setAvailablePapers(filtered);
        } catch (error) {
          console.error('Error loading papers:', error);
          addMessage({
            content: `Error loading papers: ${error.message}`,
            type: 'error'
          });
        } finally {
          setLoading(false);
        }
      };

      loadPapers();
    }, []);

    const handleDateChange = (e) => {
      const selectedDate = e.target.value;
      // Ensure the selected date is not in the past
      const today = new Date().toISOString().split('T')[0];
      if (selectedDate < today) {
        addMessage({
          content: 'Please select a future date',
          type: 'error'
        });
        return;
      }
      setScheduleFormData(prev => ({ ...prev, date: selectedDate }));
    };

    const handleTimeChange = (e) => {
      const selectedTime = e.target.value;
      setScheduleFormData(prev => ({ ...prev, time: selectedTime }));
    };

    return (
      <div className="schedule-form">
        <h3>{selectedSchedule ? 'Edit Schedule' : 'Create New Schedule'}</h3>
        <form onSubmit={handleScheduleSubmit}>
          <div className="form-group">
            <label>Title:</label>
            <input
              type="text"
              value={scheduleFormData.title}
              onChange={(e) => setScheduleFormData(prev => ({ ...prev, title: e.target.value }))}
              required
            />
          </div>

          <div className="form-group">
            <label>Type:</label>
            <select
              value={scheduleFormData.type}
              onChange={(e) => setScheduleFormData(prev => ({ ...prev, type: e.target.value }))}
              required
            >
              <option value="ISA">ISA</option>
              <option value="QUIZ">Quiz</option>
            </select>
          </div>

          {scheduleFormData.type === 'ISA' && (
            <div className="form-group">
              <label>Select Paper:</label>
              {loading ? (
                <div>Loading papers...</div>
              ) : availablePapers && availablePapers.length > 0 ? (
                <select
                  value={scheduleFormData.paperId}
                  onChange={(e) => setScheduleFormData(prev => ({ ...prev, paperId: e.target.value }))}
                  required
                >
                  <option value="">Select a paper</option>
                  {availablePapers.map(paper => (
                    <option key={paper._id} value={paper._id}>
                      {paper.title} ({paper.totalMarks} marks, {paper.duration} mins) - Course: {paper.courseId}
                    </option>
                  ))}
                </select>
              ) : (
                <div>No papers available</div>
              )}
            </div>
          )}

          <div className="form-group">
            <label>Date:</label>
            <input
              type="date"
              value={scheduleFormData.date}
              onChange={handleDateChange}
              min={new Date().toISOString().split('T')[0]}
              required
            />
          </div>

          <div className="form-group">
            <label>Time:</label>
            <input
              type="time"
              value={scheduleFormData.time}
              onChange={handleTimeChange}
              required
            />
          </div>

          <div className="form-group">
            <label>Duration (minutes):</label>
            <input
              type="number"
              value={scheduleFormData.duration}
              onChange={(e) => setScheduleFormData(prev => ({ ...prev, duration: parseInt(e.target.value) }))}
              min="1"
              required
            />
          </div>

          <div className="form-group">
            <label>Description:</label>
            <textarea
              value={scheduleFormData.description}
              onChange={(e) => setScheduleFormData(prev => ({ ...prev, description: e.target.value }))}
            />
          </div>

          <div className="form-actions">
            <button type="submit" className="btn-primary">
              {selectedSchedule ? 'Update Schedule' : 'Create Schedule'}
            </button>
            <button type="button" className="btn-secondary" onClick={() => {
              setShowScheduleForm(false);
              setSelectedSchedule(null);
            }}>
              Cancel
            </button>
          </div>
        </form>
      </div>
    );
  };

  const ScheduleList = () => (
    <div className="schedule-list">
      <div className="schedule-header">
        <h3>Schedules</h3>
        <button className="btn-primary" onClick={() => setShowScheduleForm(true)}>
          <i className="fas fa-plus"></i>
          Add New Schedule
        </button>
      </div>
      {loading ? (
        <div className="loading-state">Loading schedules...</div>
      ) : !schedules ? (
        <div className="error-state">Error loading schedules</div>
      ) : schedules.length === 0 ? (
        <div className="empty-state">
          <i className="fas fa-calendar-times"></i>
          <p>No schedules found.</p>
        </div>
      ) : (
        <div className="schedule-items">
          {schedules.map((schedule) => (
            <div key={schedule._id} className="schedule-item">
              <div className="schedule-info">
                <h4>{schedule.title}</h4>
                <p>
                  <i className="fas fa-tag"></i>
                  Type: {schedule.type}
                </p>
                <p>
                  <i className="fas fa-calendar"></i>
                  Start: {new Date(schedule.startTime).toLocaleString()}
                </p>
                <p>
                  <i className="fas fa-clock"></i>
                  End: {new Date(schedule.endTime).toLocaleString()}
                </p>
                <p>
                  <i className="fas fa-hourglass-half"></i>
                  Duration: {schedule.duration} minutes
                </p>
                {schedule.description && (
                  <p>
                    <i className="fas fa-info-circle"></i>
                    Description: {schedule.description}
                  </p>
                )}
              </div>
              <div className="schedule-actions">
                <button onClick={() => handleEditSchedule(schedule)}>
                  <i className="fas fa-edit"></i>
                  Edit
                </button>
                <button onClick={() => handleDeleteSchedule(schedule._id)}>
                  <i className="fas fa-trash-alt"></i>
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  const handleSaveExam = async (examData) => {
    try {
      setLoading(true);
      const response = await fetch('/api/schedules', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(examData)
      });

      if (!response.ok) {
        throw new Error('Failed to create exam');
      }

      await fetchSchedules();
      setShowExamCreator(false);
    } catch (error) {
      setError('Failed to save exam');
      console.error('Error saving exam:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="teacher-dashboard">
      {showSplash ? (
        <Splash />
      ) : (
        <>
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
                      <div className="user-header">
                        <div className="user-name-email">
                          <h2>{user.name}</h2>
                          <p>{user.email}</p>
                        </div>
                        <span className="role-badge">Teacher</span>
                      </div>
                      <div className="course-selection">
                        <p className="selected-course-label">Selected Course: {selectedCourse.name} ({selectedCourse.courseId})</p>
                        <p className="course-description">{selectedCourse.description}</p>
                        <div className="course-buttons">
                          {courses.map(c => (
                            <button
                              key={c.courseId}
                              className={`course-tab ${selectedCourse.courseId === c.courseId ? 'active' : ''}`}
                              onClick={() => handleCourseSelect(c)}
                            >
                              {c.name}
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                  <button className="logout-button" onClick={() => {
                    localStorage.clear();
                    navigate('/', { replace: true });
                  }}>
                    Logout
                  </button>
                </div>
              )}
            </div>
          </header>

          <main className="dashboard-main">
            <div className="chat-container">
              <div className="messages-wrapper">
                {messages.map((msg, index) => {
                  // Generate a unique key that works for both string and component content
                  const contentKey = typeof msg.content === 'string' 
                    ? msg.content.substring(0, 10) 
                    : `component-${index}`;
                  
                  // Handle the content rendering
                  let messageContent;
                  if (typeof msg.content === 'string') {
                    messageContent = msg.content;
                  } else if (React.isValidElement(msg.content)) {
                    messageContent = msg.content;
                  } else if (msg.content && typeof msg.content === 'object') {
                    // If it's an object with type and message properties (for FeedbackAnimation)
                    messageContent = (
                      <FeedbackAnimation 
                        type={msg.content.type} 
                        message={msg.content.message} 
                      />
                    );
                  } else {
                    messageContent = String(msg.content);
                  }

    return (
                    <div 
                      key={`msg-${msg.id || msg.timestamp || index}-${contentKey}`} 
                      className="message"
                    >
                  <div className="avatar">
                    <img src="/bot.png" alt="AI Assistant" className="bot-avatar" />
                  </div>
                  <div className="message-content">
                        {messageContent}
                  </div>
                </div>
                  );
                })}
              </div>
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
                className={`tab ${activeTab === 'paper' ? 'active' : ''}`}
                onClick={() => handleTabClick('paper')}
              >
                Question Paper
              </button>
              <button 
                className={`tab ${activeTab === 'schedules' ? 'active' : ''}`}
                onClick={() => handleTabClick('schedules')}
              >
                Schedules
              </button>
            </div>

            <div className="dashboard-content">
              {activeTab === 'content' && (
                <div className="content-section">
                  <h2>Course Content Management</h2>
                  <div className="file-upload-section">
                <input
                      type="file"
                      id="file-upload"
                      multiple
                      accept=".pdf,.doc,.docx,.txt,.jpg,.png"
                      onChange={handleFileUpload}
                      className="file-input"
                    />
                    <label htmlFor="file-upload" className="upload-button">
                      Choose Files
                    </label>
                    <p className="upload-info">
                      Upload any learning materials (PDF, Word, Text, or Images)
                    </p>
                  </div>

                  {uploadProgress && (
                    <div className="upload-progress">
                      <p>{uploadProgress.status}</p>
                      <div className="progress-bar">
                        <div 
                          className="progress" 
                          style={{width: `${uploadProgress.progress}%`}}
                        ></div>
                      </div>
                    </div>
                  )}

                  {processedContent && processedContent.results && processedContent.results.length > 0 && (
                    <div className="processed-content">
                        <h3>Generated Summary</h3>
                      {processedContent.results.map((result) => (
                        <div key={`file-${result.filename}-${result.timestamp || Date.now()}`} className="file-item">
                                <div className="file-info">
                                    <div className="file-name">{result.filename}</div>
                                    <div className="content-summary">
                                        <div className="summary-section">
                                            {result.summary.split('\n').map((line, i) => {
                          const lineKey = `${result.filename}-line-${i}-${line.substring(0, 10)}`;
                                                if (line.startsWith('#')) {
                                                    const level = line.match(/^#+/)[0].length;
                                                    const text = line.replace(/^#+\s*/, '');
                            return <h3 key={lineKey} style={{fontSize: `${1.5 - (level-1)*0.2}rem`}}>{text}</h3>;
                                                } else if (line.startsWith('-')) {
                            return <p key={lineKey} className="summary-bullet">{line}</p>;
                                                } else if (line.trim()) {
                            return <p key={lineKey} className="summary-text">{line}</p>;
                                                }
                                                return null;
                                            })}
                                        </div>
                                        <div className="thank-you-message">
                                            Thank you for providing the content! I've generated a summary above. Now, let's generate some questions.
                                        </div>
                                        
                                        <div className="question-generation-panel">
                                            <h4>Choose Number of Questions (Max 4 per type):</h4>
                                            <div className="question-count-controls">
                                                <div className="count-control">
                                                    <label>MCQs:</label>
                                                    <input 
                                                        type="number" 
                                                        min="1" 
                                                        max="4"
                                                        value={mcqCount}
                                                        onChange={(e) => setMcqCount(Math.min(4, parseInt(e.target.value) || 1))}
                                                    />
                                                    <span className="max-indicator">/ 4</span>
                                                </div>
                                                <div className="count-control">
                                                    <label>Descriptive:</label>
                                                    <input 
                                                        type="number"
                                                        min="1" 
                                                        max="4"
                                                        value={descriptiveCount}
                                                        onChange={(e) => setDescriptiveCount(Math.min(4, parseInt(e.target.value) || 1))}
                                                    />
                                                    <span className="max-indicator">/ 4</span>
                                                </div>
                                                <div className="count-control">
                                                    <label>True/False:</label>
                                                    <input 
                                                        type="number"
                                                        min="1" 
                                                        max="4"
                                                        value={trueFalseCount}
                                                        onChange={(e) => setTrueFalseCount(Math.min(4, parseInt(e.target.value) || 1))}
                                                    />
                                                    <span className="max-indicator">/ 4</span>
                                                </div>
                                                <div className="count-control">
                                                    <label>Fill in Blanks:</label>
                                                    <input 
                                                        type="number"
                                                        min="1" 
                                                        max="4"
                                                        value={fillBlanksCount}
                                                        onChange={(e) => setFillBlanksCount(Math.min(4, parseInt(e.target.value) || 1))}
                                                    />
                                                    <span className="max-indicator">/ 4</span>
                                                </div>
                                            </div>
                                        </div>

                                        <button 
                                            className="generate-questions-btn primary-btn"
                                            onClick={handleGenerateQuestions}
                                            disabled={isGenerating}
                                        >
                                            {isGenerating ? (
                                                <div className="button-content">
                                                    <div className="loading-spinner-small"></div>
                                                    <span>Generating Questions...</span>
                                                </div>
                                            ) : (
                                                'Generate Questions'
                                            )}
                                        </button>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                  )}
                </div>
              )}

              {activeTab === 'questions' && (
                <div className="questions-section">
                  <div className="section-header">
                    <h2>Question Bank</h2>
                  </div>
                  
                  {isGenerating ? (
                    <div className="loading-spinner">
                      <div className="spinner"></div>
                      <p>Generating questions...</p>
                      <p>This might take a minute.</p>
                    </div>
                  ) : (
                    <div className="generated-questions">
                      <QuestionDisplay 
                        questions={generatedQuestions}
                        onApprove={handleApprove}
                        onReject={handleDelete}
                      />
                    </div>
                  )}
                </div>
              )}

              {activeTab === 'paper' && (
                <div className="paper-section">
                  <div className="section-header">
                    <h2>Question Paper Generation</h2>
                    <button 
                      className="create-paper-btn"
                      onClick={() => setShowPaperForm(true)}
                    >
                      Create New Paper
                    </button>
                  </div>
                  
                  {showPaperForm ? (
                    <PaperGenerationForm 
                      onSubmit={handleGeneratePaper}
                      onCancel={() => setShowPaperForm(false)}
                    />
                  ) : (
                    <GeneratedPapersList
                      papers={papers}
                      onView={(paper) => {
                        // Open paper in new tab using Python backend URL
                        const paperId = paper._id.$oid || paper._id;
                        window.open(`http://localhost:5000/papers/${paperId}`, '_blank');
                      }}
                      onDelete={async (paperId) => {
                        try {
                          await handlePaperAction('delete', paperId);
                        } catch (error) {
                          console.error('Error deleting paper:', error);
                          addMessage({
                            content: `Error deleting paper: ${error.message}`,
                            type: 'error'
                          });
                        }
                      }}
                    />
                  )}
                </div>
              )}

              {activeTab === 'schedules' && (
                <div className="schedules-container">
                  {showScheduleForm ? <ScheduleForm /> : <ScheduleList />}
                </div>
              )}
            </div>
          </main>
          
          {showPaperView && selectedPaper && (
            <div className="modal-overlay">
              <div className="modal-content">
                <button className="close-btn" onClick={() => setShowPaperView(false)}>×</button>
                <PaperView paper={selectedPaper} />
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default TeacherDashboard; 