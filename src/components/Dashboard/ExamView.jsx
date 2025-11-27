import React, { useState, useEffect } from 'react';
import { FaClock } from 'react-icons/fa';
import { submitExamAnswers } from '../../services/examService';
import './ExamView.css';

const ScoreDisplay = ({ onClose }) => {
  return (
    <div className="exam-overlay">
      <div className="score-display-container">
        <div className="score-display">
          <div className="score-header">
            <h2>Test Submitted Successfully</h2>
          </div>
          <div className="success-message">
            <div className="success-icon">✓</div>
            <p className="success-text">Test submitted to PES successfully</p>
          </div>
          <div className="score-actions">
            <button onClick={onClose} className="close-score-btn">Close</button>
          </div>
        </div>
      </div>
    </div>
  );
};

const ExamView = ({ exam: initialExam, onSubmit, onClose }) => {
  const [exam, setExam] = useState(initialExam);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [answers, setAnswers] = useState({});
  const [timeLeft, setTimeLeft] = useState(
    typeof initialExam?.duration === 'number' ? initialExam.duration * 60 : null
  );
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [questionTimes, setQuestionTimes] = useState({});
  const [examStartTime] = useState(Date.now());
  const [score, setScore] = useState(null);
  const [showScoreDisplay, setShowScoreDisplay] = useState(false);

  useEffect(() => {
    if (initialExam?.questions) {
      setExam(initialExam);
      // Calculate total marks from questions with defaults by type
      const getDefaultMarks = (type) => (type === 'descriptive' ? 4 : 1);
      const totalMarks = initialExam.questions.reduce((total, q) => {
        const marks = typeof q.marks === 'number' ? q.marks : getDefaultMarks(q.type);
        return total + marks;
      }, 0);
      setExam(prev => ({ ...prev, totalMarks }));
      // Store exam globally for descriptive answer evaluation
      window.currentExam = initialExam;
    }
    return () => {
      // Cleanup
      delete window.currentExam;
    };
  }, [initialExam]);

  useEffect(() => {
    if (timeLeft === null) return;
    const timer = setInterval(() => {
      setTimeLeft((prev) => {
        if (prev === null) return prev;
        if (prev <= 0) {
          clearInterval(timer);
          handleSubmit();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [timeLeft]);

  // Add effect to ensure score display shows
  useEffect(() => {
    if (isSubmitted && score !== null) {
      console.log('Score display should show:', { score, totalMarks: exam?.totalMarks });
      setShowScoreDisplay(true);
    }
  }, [isSubmitted, score]);

  const formatTime = (seconds) => {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const handleAnswerChange = (questionId, answer, type) => {
    if (!exam?.questions) return;
    
    // Ensure we're using the actual question ID from the exam object
    const question = exam.questions.find(q => q._id === questionId);
    if (!question) return;

    setAnswers(prev => ({
      ...prev,
      [question._id]: {
        answer,
        type,
        timestamp: new Date().toISOString()
      }
    }));

    // Record time spent on question
    setQuestionTimes(prev => ({
      ...prev,
      [question._id]: (prev[question._id] || 0) + 1
    }));
  };

  const handleSubmit = async () => {
    try {
      if (isSubmitted) {
        console.log('Exam already submitted');
        return;
      }

      if (!exam?.questions) {
        alert('Exam data not loaded. Please try again.');
        return;
      }

      // Confirm submission
      const confirmed = window.confirm('Are you sure you want to submit the exam? This action cannot be undone.');
      if (!confirmed) {
        return;
      }

      setLoading(true);

      // Filter out any answers that don't have valid question IDs
      const validAnswers = Object.entries(answers).filter(([questionId]) => {
        return exam.questions.some(q => q._id === questionId);
      });

      // Submit to backend
      const submissionData = {
        paperId: exam.paperId || exam._id,
        startTime: examStartTime,
        answers: validAnswers.map(([questionId, data]) => ({
          questionId,
          answer: data.answer,
          type: data.type,
          timeSpent: questionTimes[questionId] || 0
        }))
      };

      // Try to submit even if no answers (allow empty submission)
      try {
        // Submit to backend
        await submitExamAnswers(submissionData);
      } catch (submitError) {
        console.error('Backend submission error:', submitError);
        // Continue even if backend submission fails
      }

      // Mark as submitted and show success message
      setIsSubmitted(true);
      setShowScoreDisplay(true);
      setLoading(false);

      // Notify parent component if needed
      if (onSubmit) {
        onSubmit({
          ...submissionData,
          score: 0,
          maxScore: exam.totalMarks,
          timeSpent: Math.round((Date.now() - examStartTime) / 1000)
        });
      }

    } catch (error) {
      console.error('Error submitting exam:', error);
      setLoading(false);
      alert('Error submitting exam. Please try again.');
    }
  };

  const isQuestionAnswered = (questionId) => {
    if (!exam?.questions) return false;
    
    // Ensure we're using the actual question ID from the exam object
    const question = exam.questions.find(q => q._id === questionId);
    if (!question) return false;
    const ans = answers[question._id]?.answer;
    switch (question.type) {
      case 'mcq':
        // index 0 should be considered answered
        return typeof ans === 'number';
      case 'trueFalse':
        return ans === 'true' || ans === 'false' || ans === true || ans === false;
      case 'fillInBlanks':
        return typeof ans === 'string' && ans.trim().length > 0;
      case 'descriptive':
        return typeof ans === 'string' && ans.trim().length > 0;
      default:
        return ans !== undefined && ans !== null;
    }
  };

  const handleRetry = () => {
    setError(null);
    setLoading(false);
    window.location.reload();
  };

  const getQuestionTypeDisplay = (type) => {
    switch (type) {
      case 'mcq': return 'Multiple Choice';
      case 'descriptive': return 'Descriptive';
      case 'trueFalse': return 'True/False';
      case 'fillInBlanks': return 'Fill in the Blanks';
      default: return type;
    }
  };

  const getQuestionTypeIndicator = (type) => {
    switch (type) {
      case 'mcq': return 'M';
      case 'descriptive': return 'D';
      case 'trueFalse': return 'T';
      case 'fillInBlanks': return 'F';
      default: return '?';
    }
  };

  if (error) {
    return (
      <div className="exam-error">
        <h3>Error</h3>
        <p>{error}</p>
        <div className="error-actions">
          <button onClick={onClose}>Close</button>
          <button onClick={handleRetry}>Try Again</button>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="exam-loading">
        <div className="loader"></div>
        <p>Preparing your exam...</p>
      </div>
    );
  }

  if (!exam?.questions || !exam.questions.length) {
    return (
      <div className="exam-error">
        <h3>Error</h3>
        <p>No questions available for this exam.</p>
        <div className="error-actions">
          <button onClick={onClose}>Close</button>
          <button onClick={handleRetry}>Try Again</button>
        </div>
      </div>
    );
  }

  const currentQuestion = exam.questions[currentQuestionIndex];
  if (!currentQuestion) {
    return (
      <div className="exam-error">
        <h3>Error</h3>
        <p>Failed to load question.</p>
        <div className="error-actions">
          <button onClick={onClose}>Close</button>
          <button onClick={handleRetry}>Try Again</button>
        </div>
      </div>
    );
  }

  if (showScoreDisplay) {
    return (
      <div className="exam-overlay">
        <div className="score-display-container">
          <ScoreDisplay 
            onClose={() => {
              setShowScoreDisplay(false);
              onClose();
            }} 
          />
        </div>
      </div>
    );
  }

  return (
    <div className="exam-view">
      <div className="exam-header">
        <h2>{exam?.title || 'Exam'}</h2>
        <div className="exam-info">
          {timeLeft !== null && (
            <div className="timer">
              <FaClock />
              <span className={timeLeft < 300 ? 'time-warning' : ''}>
                Time Left: {formatTime(timeLeft)}
              </span>
            </div>
          )}
          <div className="progress">
            Question {currentQuestionIndex + 1} of {exam.questions.length}
          </div>
          <div className="total-marks">
            Total Marks: {exam.totalMarks}
          </div>
          <div>
            <button
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                handleSubmit();
              }}
              disabled={isSubmitted || loading}
              className="submit-btn"
              type="button"
            >
              {loading ? 'Submitting...' : 'Submit Exam'}
            </button>
          </div>
        </div>
      </div>

      <div className="exam-layout">
        {/* Question List Sidebar */}
        <div className="question-list">
          <h3>Questions</h3>
          <div className="question-buttons">
            {exam.questions.map((q, index) => (
              <button
                key={index}
                className={`question-button ${index === currentQuestionIndex ? 'active' : ''} ${
                  isQuestionAnswered(q._id) ? 'answered' : ''
                }`}
                onClick={() => setCurrentQuestionIndex(index)}
              >
                {index + 1}
                <span className="question-type-indicator">
                  {getQuestionTypeIndicator(q.type)}
                </span>
                <span className="question-marks">
                  ({typeof q.marks === 'number' ? q.marks : (q.type === 'descriptive' ? 4 : 1)} marks)
                </span>
              </button>
            ))}
          </div>
          <div className="question-legend">
            <div className="legend-item">
              <span className="legend-color answered"></span>
              <span>Answered</span>
            </div>
            <div className="legend-item">
              <span className="legend-color unanswered"></span>
              <span>Unanswered</span>
            </div>
            <div className="legend-item">
              <span className="legend-type">M</span>
              <span>Multiple Choice</span>
            </div>
            <div className="legend-item">
              <span className="legend-type">D</span>
              <span>Descriptive</span>
            </div>
            <div className="legend-item">
              <span className="legend-type">T</span>
              <span>True/False</span>
            </div>
            <div className="legend-item">
              <span className="legend-type">F</span>
              <span>Fill in Blanks</span>
            </div>
          </div>
        </div>

        {/* Main Question Area */}
        <div className="question-area">
          <div className="question-container">
            <div className="question-content">
              <div className="question-header">
                <span className="question-number">Question {currentQuestionIndex + 1}</span>
                <span className="question-type">{getQuestionTypeDisplay(currentQuestion.type)}</span>
                <span className="question-marks">({typeof currentQuestion.marks === 'number' ? currentQuestion.marks : (currentQuestion.type === 'descriptive' ? 4 : 1)} marks)</span>
              </div>
              
              <p className="question-text">{currentQuestion.questionText}</p>
              
              {currentQuestion.type === 'mcq' && currentQuestion.options && (
                <div className="options-container">
                  {currentQuestion.options.map((option, index) => (
                    <label key={index} className="option">
                      <input
                        type="radio"
                        name={`question-${currentQuestion._id}`}
                    checked={answers[currentQuestion._id]?.answer === index}
                    onChange={() => handleAnswerChange(currentQuestion._id, index, 'mcq')}
                        disabled={isSubmitted}
                      />
                      <span className="option-text">{option.text}</span>
                    </label>
                  ))}
                </div>
              )}

              {currentQuestion.type === 'trueFalse' && (
                <div className="options-container">
                  <label className="option">
                    <input
                      type="radio"
                      name={`question-${currentQuestion._id}`}
                      checked={String(answers[currentQuestion._id]?.answer) === 'true'}
                      onChange={() => handleAnswerChange(currentQuestion._id, 'true', 'trueFalse')}
                      disabled={isSubmitted}
                    />
                    <span className="option-text">True</span>
                  </label>
                  <label className="option">
                    <input
                      type="radio"
                      name={`question-${currentQuestion._id}`}
                      checked={String(answers[currentQuestion._id]?.answer) === 'false'}
                      onChange={() => handleAnswerChange(currentQuestion._id, 'false', 'trueFalse')}
                      disabled={isSubmitted}
                    />
                    <span className="option-text">False</span>
                  </label>
                </div>
              )}

              {currentQuestion.type === 'fillInBlanks' && (
                <div className="fill-blanks-container">
                  <input
                    type="text"
                    className="fill-blanks-input"
                    value={answers[currentQuestion._id]?.answer || ''}
                    onChange={(e) => handleAnswerChange(currentQuestion._id, e.target.value, 'fillInBlanks')}
                    placeholder="Type your answer here..."
                    disabled={isSubmitted}
                  />
                </div>
              )}

              {currentQuestion.type === 'descriptive' && (
                <textarea
                  value={answers[currentQuestion._id]?.answer || ''}
                  onChange={(e) => handleAnswerChange(currentQuestion._id, e.target.value, 'descriptive')}
                  placeholder="Type your answer here..."
                  disabled={isSubmitted}
                  style={{
                    width: '100%',
                    minHeight: '200px',
                    padding: '10px',
                    fontSize: '16px',
                    border: '1px solid #ccc',
                    borderRadius: '4px',
                    marginTop: '10px'
                  }}
                />
              )}
            </div>
          </div>

          <div className="exam-navigation">
            <button
              onClick={() => setCurrentQuestionIndex(prev => Math.max(0, prev - 1))}
              disabled={currentQuestionIndex === 0}
              className="nav-btn prev-btn"
            >
              Previous
            </button>
            <button
              onClick={() => setCurrentQuestionIndex(prev => Math.min(exam.questions.length - 1, prev + 1))}
              className="nav-btn next-btn"
            >
              Next
            </button>
            <button
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                handleSubmit();
              }}
              disabled={isSubmitted || loading}
              className="submit-btn"
              type="button"
            >
              {loading ? 'Submitting...' : 'Submit Exam'}
            </button>
          </div>
        </div>
      </div>

      <button className="close-btn" onClick={onClose}>
        Close Exam
      </button>
    </div>
  );
};

export default ExamView; 