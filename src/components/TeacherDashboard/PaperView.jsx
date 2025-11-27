import React from 'react';
import './PaperView.css';

const PaperView = ({ paper }) => {
  if (!paper) return null;

  const getTotalMarks = () => {
    return paper.questions.reduce((total, q) => total + (q.marks || 2), 0);
  };

  return (
    <div className="paper-view">
      <div className="paper-header">
        <h1>{paper.title}</h1>
        <h2>Course: {paper.courseId}</h2>
        <div className="paper-info">
          <p>Total Marks: {paper.totalMarks || getTotalMarks()}</p>
          <p>Duration: {paper.duration} minutes</p>
          <p>Type: {paper.type}</p>
          <p>Status: {paper.status}</p>
        </div>
      </div>

      <div className="questions-container">
        {paper.questions.map((question, index) => (
          <div key={question._id} className="question-box">
            <div className="question-header">
              <span className="question-number">Question {index + 1}</span>
              <div className="question-meta">
                <span className="question-type">{question.type.toUpperCase()}</span>
                <span className="question-marks">{question.marks || 2} marks</span>
                <span className="question-difficulty">Difficulty: {question.difficulty}/10</span>
              </div>
            </div>
            
            <div className="question-content">
              <p className="question-text">{question.questionText}</p>
              
              {question.type === 'mcq' && (
                <div className="options-container">
                  {question.options.map((option, optIndex) => (
                    <div 
                      key={option._id} 
                      className={`option ${option.isCorrect ? 'correct' : ''}`}
                    >
                      <span className="option-label">{String.fromCharCode(65 + optIndex)}.</span>
                      <span className="option-text">{option.text}</span>
                      {option.isCorrect && <span className="correct-marker">✓</span>}
                    </div>
                  ))}
                </div>
              )}

              {question.type === 'descriptive' && (
                <div className="descriptive-answer">
                  <p className="answer-label">Sample Answer:</p>
                  <p className="answer-text">{question.correctAnswer}</p>
                </div>
              )}
            </div>

            {question.explanation && (
              <div className="explanation">
                <h4>Explanation:</h4>
                <p>{question.explanation}</p>
              </div>
            )}

            <div className="question-metadata">
              <p>Topic: {question.topic}</p>
              {question.conceptsCovered && question.conceptsCovered.length > 0 && (
                <p>Concepts: {question.conceptsCovered.join(', ')}</p>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default PaperView; 