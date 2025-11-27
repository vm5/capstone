import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { generatePersonalizedExam, submitExamAnswers } from '../../services/examService';
import ExamView from './ExamView';
import './ExamView.css';

const ExamPage = () => {
  const navigate = useNavigate();
  const [exam, setExam] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadExam = async () => {
      try {
        setLoading(true);
        
        // Get diagnostic results
        const diagnosticResults = localStorage.getItem('diagnosticResults');
        if (!diagnosticResults) {
          navigate('/dashboard');
          return;
        }

        // Load the adaptive exam
        const response = await generatePersonalizedExam();
        
        if (!response.questions || response.questions.length === 0) {
          throw new Error('No questions available for this exam');
        }

        setExam({
          questions: response.questions,
          duration: response.duration,
          totalMarks: response.totalMarks,
          metadata: response.metadata,
          paperId: response.paperId
        });
      } catch (err) {
        console.error('Error loading exam:', err);
        setError(err.message || 'Failed to load exam');
      } finally {
        setLoading(false);
      }
    };

    loadExam();
  }, [navigate]);

  const handleSubmit = async (submissionData) => {
    try {
      setLoading(true);
      setError(null);
      
      // Submit to backend
      const result = await submitExamAnswers(submissionData);
      
      // Show result
      setLoading(false);
      alert(`Exam submitted successfully! Score: ${result.totalScore}/${result.maxScore}`);
      
      // Navigate back to dashboard after 3 seconds
      setTimeout(() => {
        navigate('/dashboard');
      }, 3000);
    } catch (error) {
      console.error('Error submitting exam:', error);
      setLoading(false);
      setError(error.message || 'Failed to submit exam. Please try again.');
      
      // Clear error after 5 seconds
      setTimeout(() => {
        setError(null);
      }, 5000);
    }
  };

  const handleClose = () => {
    if (window.confirm('Are you sure you want to leave the exam? Your progress will be lost.')) {
      navigate('/dashboard');
    }
  };

  if (loading) {
    return (
      <div className="exam-loading">
        <div className="loader"></div>
        <p>Loading exam...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="exam-error">
        <h2>Error Loading Exam</h2>
        <p>{error}</p>
        <button onClick={() => navigate('/dashboard')}>Back to Dashboard</button>
      </div>
    );
  }

  if (!exam) {
    return (
      <div className="exam-error">
        <h2>No Exam Found</h2>
        <p>The requested exam could not be found.</p>
        <button onClick={() => navigate('/dashboard')}>Back to Dashboard</button>
      </div>
    );
  }

  return (
    <div className="exam-page">
      <ExamView 
        exam={exam}
        onSubmit={handleSubmit}
        onClose={handleClose}
      />
    </div>
  );
};

export default ExamPage; 