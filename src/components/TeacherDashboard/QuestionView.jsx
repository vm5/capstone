// src/components/TeacherDashboard/QuestionView.jsx
import React from 'react';
import './QuestionView.css';

const QuestionView = ({ question, onApprove, onDelete }) => {
  return (
    <div className="question-card">
      <h3 className="question-text">{question.text}</h3>
      {question.options && (
        <div className="options">
          {question.options.map((option, index) => (
            <div key={index} className={`option ${index === question.correct_answer ? 'correct' : ''}`}>
              {option}
            </div>
          ))}
        </div>
      )}
      <div className="actions">
        <button className="approve-btn" onClick={() => onApprove(question.id)}>Approve</button>
        <button className="delete-btn" onClick={() => onDelete(question.id)}>Delete</button>
      </div>
    </div>
  );
};

export default QuestionView;