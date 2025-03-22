import React, { useState } from 'react';
import './QuestionReview.css';

const QuestionReview = ({ 
  questions, 
  onClose, 
  onSave,
  selectedCourse,
  selectedTopic,
  addMessage 
}) => {
  const [reviewedQuestions, setReviewedQuestions] = useState(
    questions.map(q => ({ ...q, approved: false, edited: false }))
  );

  const handleApprove = (index) => {
    setReviewedQuestions(prev => prev.map((q, i) => 
      i === index ? { ...q, approved: !q.approved } : q
    ));
  };

  const handleEdit = (index, newText) => {
    setReviewedQuestions(prev => prev.map((q, i) => 
      i === index ? { ...q, text: newText, edited: true } : q
    ));
  };

  const handleOptionEdit = (questionIndex, optionIndex, newText) => {
    setReviewedQuestions(prev => prev.map((q, i) => 
      i === questionIndex ? {
        ...q,
        options: q.options.map((opt, j) => 
          j === optionIndex ? newText : opt
        ),
        edited: true
      } : q
    ));
  };

  const handleCorrectAnswer = (questionIndex, optionIndex) => {
    setReviewedQuestions(prev => prev.map((q, i) => 
      i === questionIndex ? {
        ...q,
        correct_answer: optionIndex,
        edited: true
      } : q
    ));
  };

  const handleExplanationEdit = (index, newText) => {
    setReviewedQuestions(prev => prev.map((q, i) => 
      i === index ? { ...q, explanation: newText, edited: true } : q
    ));
  };

  const handleDifficultyChange = (index, value) => {
    setReviewedQuestions(prev => prev.map((q, i) => 
      i === index ? { ...q, difficulty: parseFloat(value), edited: true } : q
    ));
  };

  const handleDelete = (index) => {
    setReviewedQuestions(prev => prev.filter((_, i) => i !== index));
  };

  const handleSave = async () => {
    const approvedQuestions = reviewedQuestions.filter(q => q.approved);
    try {
      const response = await fetch(`http://localhost:5000/api/courses/${selectedCourse.id}/save-questions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          questions: approvedQuestions,
          topic: selectedTopic
        })
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Failed to save questions');
      }

      const data = await response.json();
      console.log('Questions saved successfully:', data);
      onClose();
      addMessage({ 
        content: `Successfully saved ${approvedQuestions.length} questions for ${selectedTopic}` 
      });

    } catch (error) {
      console.error('Error saving questions:', error);
      addMessage({ 
        content: `Error saving questions: ${error.message}`, 
        type: 'error' 
      });
    }
  };

  return (
    <div className="question-review-overlay">
      <div className="question-review">
        <div className="review-header">
          <h3>Review Generated Questions</h3>
          <div className="review-stats">
            <span>{reviewedQuestions.filter(q => q.approved).length} of {questions.length} Approved</span>
          </div>
        </div>

        <div className="questions-list">
          {reviewedQuestions.map((question, index) => (
            <div key={index} className={`question-item ${question.approved ? 'approved' : ''}`}>
              <div className="question-number">Question {index + 1}</div>
              
              <div className="question-content">
                <textarea
                  value={question.text}
                  onChange={(e) => handleEdit(index, e.target.value)}
                  placeholder="Question text"
                  className="question-text"
                />
                
                {question.options && (
                  <div className="options-grid">
                    {question.options.map((option, optIndex) => (
                      <div key={optIndex} className="option-item">
                        <input
                          type="text"
                          value={option}
                          onChange={(e) => handleOptionEdit(index, optIndex, e.target.value)}
                          placeholder={`Option ${optIndex + 1}`}
                        />
                        <input
                          type="radio"
                          name={`correct-${index}`}
                          checked={question.correct_answer === optIndex}
                          onChange={() => handleCorrectAnswer(index, optIndex)}
                        />
                      </div>
                    ))}
                  </div>
                )}

                {question.explanation && (
                  <textarea
                    value={question.explanation}
                    onChange={(e) => handleExplanationEdit(index, e.target.value)}
                    placeholder="Explanation"
                    className="question-explanation"
                  />
                )}
              </div>

              <div className="question-meta">
                <span className="meta-item">Type: {question.type}</span>
                <span className="meta-item">
                  Difficulty: 
                  <input 
                    type="range" 
                    min="0" 
                    max="1" 
                    step="0.1"
                    value={question.difficulty}
                    onChange={(e) => handleDifficultyChange(index, e.target.value)}
                  />
                  {(question.difficulty * 10).toFixed(1)}
                </span>
                <span className="meta-item">Source: {question.source}</span>
                {question.edited && <span className="edited-badge">Edited</span>}
              </div>

              <div className="question-actions">
                <button 
                  className={`approve-btn ${question.approved ? 'approved' : ''}`}
                  onClick={() => handleApprove(index)}
                >
                  {question.approved ? 'Approved ✓' : 'Approve'}
                </button>
                <button 
                  className="delete-btn"
                  onClick={() => handleDelete(index)}
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>

        <div className="review-actions">
          <button onClick={onClose} className="cancel-btn">Cancel</button>
          <button 
            onClick={handleSave}
            className="save-btn"
            disabled={!reviewedQuestions.some(q => q.approved)}
          >
            Save {reviewedQuestions.filter(q => q.approved).length} Questions
          </button>
        </div>
      </div>
    </div>
  );
};

export default QuestionReview; 