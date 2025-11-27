import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import './Dashboard.css';

const DiagnosticTest = () => {
  const [questions, setQuestions] = useState([]);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [answers, setAnswers] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [submitted, setSubmitted] = useState(false);
  const [results, setResults] = useState(null);
  const [startTime, setStartTime] = useState(null);
  const [confidenceScores, setConfidenceScores] = useState({});
  const navigate = useNavigate();

  useEffect(() => {
    // Check if user is authenticated
    const token = localStorage.getItem('token');
    if (!token) {
      navigate('/login');
      return;
    }
    fetchQuestions();
  }, [navigate]);

  useEffect(() => {
    // Set start time when question changes
    setStartTime(Date.now());
  }, [currentQuestionIndex]);

  const fetchQuestions = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get('/api/diagnostic/questions', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      setQuestions(response.data);
      setStartTime(Date.now()); // Set initial start time
      setLoading(false);
    } catch (err) {
      console.error('Error fetching questions:', err);
      if (err.response?.status === 401) {
        localStorage.removeItem('token');
        navigate('/login');
      } else {
        setError('Failed to load questions. Please try again.');
      }
      setLoading(false);
    }
  };

  const handleAnswer = (answer) => {
    const timeSpent = Math.round((Date.now() - startTime) / 1000); // Convert to seconds
    const currentQuestionId = questions[currentQuestionIndex]._id;
    
    // Get current confidence score or default to 5 (middle value)
    const currentConfidence = confidenceScores[currentQuestionId] || 5;
    
    // Update confidence scores if not already set
    if (!confidenceScores[currentQuestionId]) {
      setConfidenceScores(prev => ({
        ...prev,
        [currentQuestionId]: currentConfidence
      }));
    }
    
    setAnswers({
      ...answers,
      [currentQuestionId]: {
        questionId: currentQuestionId,
        questionText: questions[currentQuestionIndex].questionText,
        type: questions[currentQuestionIndex].type,
        answer: answer,
        correctAnswer: questions[currentQuestionIndex].correctAnswer,
        timeSpent: timeSpent,
        confidenceScore: currentConfidence
      }
    });
  };

  const handleConfidenceChange = (score) => {
    const currentQuestionId = questions[currentQuestionIndex]._id;
    
    // Update confidence scores
    setConfidenceScores(prev => ({
      ...prev,
      [currentQuestionId]: score
    }));
    
    // Update answer if it exists
    if (answers[currentQuestionId]) {
      setAnswers(prev => ({
        ...prev,
        [currentQuestionId]: {
          ...prev[currentQuestionId],
          confidenceScore: score
        }
      }));
    }
  };

  const handleSubmit = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await axios.post('/api/diagnostic/submit', 
        { answers: Object.values(answers) },
        {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        }
      );
      setResults(response.data);
      setSubmitted(true);
    } catch (err) {
      console.error('Error submitting test:', err);
      if (err.response?.status === 401) {
        localStorage.removeItem('token');
        navigate('/login');
      } else {
        setError('Failed to submit test. Please try again.');
      }
    }
  };

  const handleStartActualTest = () => {
    // Calculate average confidence score
    const confidenceValues = Object.values(answers).map(a => a.confidenceScore || 0);
    const avgConfidence = confidenceValues.length > 0 ? 
      confidenceValues.reduce((a, b) => a + b, 0) / confidenceValues.length : 0;

    // Calculate score (percentage of correct answers)
    const correctAnswers = Object.values(answers).filter(a => {
      if (!a || !a.type) return false;
      
      if (a.type === 'mcq' || a.type === 'trueFalse') {
        return a.answer === a.correctAnswer;
      } else if (a.type === 'fillInBlanks') {
        return a.answer && a.correctAnswer && 
               a.answer.toString().toLowerCase().trim() === a.correctAnswer.toString().toLowerCase().trim();
      }
      // For descriptive questions, we'll count them as correct if they provided an answer
      return a.answer && a.answer.toString().trim().length > 0;
    }).length;
    
    const score = questions.length > 0 ? (correctAnswers / questions.length) * 10 : 0; // Convert to 0-10 scale

    // Store diagnostic results in localStorage for exam page
    localStorage.setItem('diagnosticResults', JSON.stringify({
      confidence: Math.round(avgConfidence),
      score: Math.round(score),
      performance: results?.performance || {
        overallAccuracy: 0,
        averageConfidence: 0,
        totalTimeSpent: 0
      }
    }));
    
    // Navigate to exam page
    navigate('/exam');
  };

  const renderResults = () => {
    if (!results) return null;

    const getZoneColor = (zone) => {
      switch (zone) {
        case 'EASY': return '#4CAF50';
        case 'MEDIUM': return '#FF9800';
        case 'HARD': return '#f44336';
        default: return '#4CAF50';
      }
    };

    // Calculate the score percentage for the circle
    const scorePercentage = (results.baselineScore || 0) * 3.6; // Convert to degrees (360/100)

    return (
      <div className="results-container">
        <div className="results-header">
          <h2>Diagnostic Test Results</h2>
          <div className="score-summary">
            <div className="score-circle" style={{ 
              background: `conic-gradient(${getZoneColor(results.learningZone)} ${scorePercentage}deg, #2a2a2a 0deg)`
            }}>
              <div className="score-value">{results.baselineScore || 0}%</div>
            </div>
            <div className="zone-badge" style={{ backgroundColor: getZoneColor(results.learningZone) }}>
              {results.learningZone || 'MEDIUM'} ZONE
            </div>
          </div>
        </div>

        <div className="results-grid">
          <div className="results-card performance">
            <h3>Performance Details</h3>
            <div className="stat-row">
              <div className="stat-item">
                <span className="stat-label">Accuracy</span>
                <span className="stat-value">{results.performance?.overallAccuracy?.toFixed(1) || '0'}%</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Confidence</span>
                <span className="stat-value">{results.performance?.averageConfidence?.toFixed(1) || '0'}/10</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Time</span>
                <span className="stat-value">{((results.performance?.totalTimeSpent || 0) / 60).toFixed(1)} min</span>
              </div>
            </div>
          </div>

          <div className="results-card evaluation">
            <h3>Evaluation Summary</h3>
            <div className="evaluation-content">
              <div className="evaluation-item">
                <span className="eval-label">Learning Zone:</span>
                <span className="eval-value" style={{ color: getZoneColor(results.learningZone) }}>
                  {results.learningZone || 'MEDIUM'} Level
                </span>
              </div>
              <div className="evaluation-item">
                <span className="eval-label">Knowledge Level:</span>
                <span className="eval-value">
                  {results.baselineScore >= 80 ? 'Advanced' :
                   results.baselineScore >= 60 ? 'Intermediate' :
                   results.baselineScore >= 40 ? 'Basic' : 'Beginner'}
                </span>
              </div>
              <div className="evaluation-item">
                <span className="eval-label">Readiness:</span>
                <span className="eval-value">
                  {results.baselineScore >= 70 ? 'Ready for Advanced Topics' :
                   results.baselineScore >= 50 ? 'Ready for Intermediate Topics' :
                   'Focus on Fundamentals Recommended'}
                </span>
              </div>
            </div>
          </div>

          {results.recommendations && (
            <div className="results-card recommendations">
        <h3>Recommendations</h3>
              <div className="recommendations-content">
                {results.recommendations.strongAreas && results.recommendations.strongAreas.length > 0 && (
                  <div className="strengths">
                    <h4>💪 Strong Areas</h4>
          <ul>
            {results.recommendations.strongAreas.map((area, index) => (
              <li key={index}>{area}</li>
            ))}
          </ul>
                  </div>
                )}
                {results.recommendations.weakAreas && results.recommendations.weakAreas.length > 0 && (
                  <div className="improvements">
                    <h4>🎯 Areas for Growth</h4>
          <ul>
            {results.recommendations.weakAreas.map((area, index) => (
              <li key={index}>{area}</li>
            ))}
          </ul>
                  </div>
                )}
                {results.recommendations.recommendedTopics && results.recommendations.recommendedTopics.length > 0 && (
                  <div className="next-steps">
                    <h4>🚀 Next Steps</h4>
          <ul>
            {results.recommendations.recommendedTopics.map((topic, index) => (
              <li key={index}>{topic}</li>
            ))}
          </ul>
        </div>
                )}
              </div>
            </div>
          )}
        </div>
        
        <div className="start-test-container">
          <button 
            onClick={handleStartActualTest}
            className="start-test-button"
          >
            Start Actual Test
          </button>
          <p className="test-info">
            You will have 45 minutes to complete the actual test. The test consists of 11 questions worth 17 marks total.
          </p>
        </div>
      </div>
    );
  };

  if (loading) return <div className="diagnostic-test-container">Loading...</div>;
  if (error) return <div className="diagnostic-test-container">{error}</div>;
  if (submitted) return <div className="diagnostic-test-container">{renderResults()}</div>;

  return (
    <div className="diagnostic-test-container">
      <div className="test-header">
        <h1>Machine Learning Diagnostic Test</h1>
        <p>This test will help us understand your current knowledge level in Machine Learning concepts.</p>
        <p>Please answer all questions to the best of your ability.</p>
      </div>

      <div className="progress-bar">
        <div 
          className="progress-fill" 
          style={{ width: `${((currentQuestionIndex + 1) / questions.length) * 100}%` }}
        ></div>
      </div>

      {loading ? (
        <div className="loading-state">Loading questions...</div>
      ) : error ? (
        <div className="error-state">{error}</div>
      ) : submitted ? (
        renderResults()
      ) : (
        <>
          <div className="question-container">
            <div className="question-header">
              <h2 className="question-number">Question {currentQuestionIndex + 1} of {questions.length}</h2>
              <div className="question-type">{questions[currentQuestionIndex]?.type === 'mcq' ? 'Multiple Choice' : 
                questions[currentQuestionIndex]?.type === 'descriptive' ? 'Descriptive Answer' :
                questions[currentQuestionIndex]?.type === 'trueFalse' ? 'True/False' : 'Fill in the Blank'}</div>
            </div>

            <div className="question-text">
              {questions[currentQuestionIndex]?.questionText}
            </div>

            {/* Force show textarea for this question */}
            <div className="descriptive-container" style={{
              marginTop: '20px',
              marginBottom: '20px',
              padding: '20px',
              background: '#2a2a2a',
              border: '2px solid #4ecdc4',
              borderRadius: '8px',
              boxShadow: '0 0 10px rgba(78, 205, 196, 0.2)'
            }}>
              <div style={{
                color: '#4ecdc4',
                marginBottom: '10px',
                fontSize: '0.9rem',
                fontWeight: 'bold'
              }}>
                Write your answer below:
              </div>
              <textarea
                  value={answers[questions[currentQuestionIndex]?._id]?.answer || ''}
                  onChange={(e) => handleAnswer(e.target.value)}
                placeholder="Explain the concept of backpropagation in neural networks and its importance in deep learning..."
                style={{
                  width: '100%',
                  minHeight: '200px',
                  padding: '15px',
                  background: '#1a1a1a',
                  border: '1px solid #3a3a3a',
                  borderRadius: '4px',
                  color: '#fff',
                  fontSize: '1rem',
                  lineHeight: '1.5',
                  resize: 'vertical'
                }}
              />
            </div>

            <div className="confidence-selector">
              <h3>How confident are you in your answer?</h3>
              <div className="confidence-buttons">
                {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((score) => {
                  const currentQuestionId = questions[currentQuestionIndex]?._id;
                  const currentConfidence = answers[currentQuestionId]?.confidenceScore || confidenceScores[currentQuestionId];
                  return (
                    <button
                      key={score}
                      className={`confidence-button ${currentConfidence === score ? 'selected' : ''}`}
                      onClick={() => handleConfidenceChange(score)}
                    >
                      {score}
                    </button>
                  );
                })}
              </div>
              <div className="confidence-labels">
                <span>Not Confident</span>
                <span>Very Confident</span>
              </div>
            </div>
          </div>

          <div className="navigation-buttons">
            <button
              onClick={() => setCurrentQuestionIndex(prev => prev - 1)}
              disabled={currentQuestionIndex === 0}
              className="nav-button"
            >
              Previous
            </button>

            {currentQuestionIndex === questions.length - 1 ? (
              <button
                onClick={handleSubmit}
                disabled={!answers[questions[currentQuestionIndex]?._id]}
                className="submit-button"
              >
                Submit Test
              </button>
            ) : (
              <button
                onClick={() => setCurrentQuestionIndex(prev => prev + 1)}
                disabled={!answers[questions[currentQuestionIndex]?._id]}
                className="nav-button"
              >
                Next
              </button>
            )}
          </div>
        </>
      )}
    </div>
  );
};

export default DiagnosticTest; 