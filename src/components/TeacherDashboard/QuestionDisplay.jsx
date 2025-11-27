import React from 'react';

const QuestionDisplay = ({ questions = {}, onApprove, onReject }) => {
  return (
    <div className="question-display">
      {Object.entries(questions).map(([type, questionList]) => (
        questionList && questionList.length > 0 && (
          <div key={type}>
            {questionList.map((question, index) => (
              <div key={index}>
                <div className="question-text">
                  {question.question || question.statement}
                  <span className="difficulty-badge">
                    Moderate
                  </span>
                </div>
                
                <div className="options-list">
                  {question.options && question.options.map((option, optIndex) => (
                    <div 
                      key={optIndex} 
                      className={`option-item ${option === question.correct_answer ? 'correct' : ''}`}
                      data-option={String.fromCharCode(65 + optIndex)}
                    >
                      {option}
                    </div>
                  ))}
                </div>

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
                    onClick={() => onReject(question.id)}
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

export default QuestionDisplay; 