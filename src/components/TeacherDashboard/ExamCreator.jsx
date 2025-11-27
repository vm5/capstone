import React, { useState } from 'react';
import { FaPlus, FaTrash, FaClock, FaCheck } from 'react-icons/fa';
import './ExamCreator.css';

const ExamCreator = ({ onSave, initialData }) => {
  const [examData, setExamData] = useState(initialData || {
    title: '',
    type: 'ISA',
    duration: 60,
    startTime: '',
    endTime: '',
    description: '',
    questionPool: {
      easy: [],
      medium: [],
      hard: []
    }
  });

  const [currentQuestion, setCurrentQuestion] = useState({
    text: '',
    type: 'mcq',
    options: ['', '', '', ''],
    correctAnswer: '',
    marks: 1,
    difficultyLevel: 'easy'
  });

  const handleExamDataChange = (e) => {
    const { name, value } = e.target;
    setExamData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleQuestionChange = (e) => {
    const { name, value } = e.target;
    setCurrentQuestion(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleOptionChange = (index, value) => {
    setCurrentQuestion(prev => ({
      ...prev,
      options: prev.options.map((opt, i) => i === index ? value : opt)
    }));
  };

  const addQuestion = () => {
    if (!currentQuestion.text || 
        (currentQuestion.type === 'mcq' && !currentQuestion.options.every(opt => opt)) ||
        !currentQuestion.correctAnswer) {
      alert('Please fill in all question fields');
      return;
    }

    const difficulty = currentQuestion.difficultyLevel;
    setExamData(prev => ({
      ...prev,
      questionPool: {
        ...prev.questionPool,
        [difficulty]: [...prev.questionPool[difficulty], { ...currentQuestion }]
      }
    }));

    // Reset current question
    setCurrentQuestion({
      text: '',
      type: 'mcq',
      options: ['', '', '', ''],
      correctAnswer: '',
      marks: 1,
      difficultyLevel: 'easy'
    });
  };

  const removeQuestion = (difficulty, index) => {
    setExamData(prev => ({
      ...prev,
      questionPool: {
        ...prev.questionPool,
        [difficulty]: prev.questionPool[difficulty].filter((_, i) => i !== index)
      }
    }));
  };

  const handleSubmit = () => {
    // Validate exam data
    if (!examData.title || !examData.startTime || !examData.endTime) {
      alert('Please fill in all exam details');
      return;
    }

    // Ensure there are questions in each difficulty level
    if (!examData.questionPool.easy.length || 
        !examData.questionPool.medium.length || 
        !examData.questionPool.hard.length) {
      alert('Please add questions for all difficulty levels');
      return;
    }

    onSave({
      ...examData,
      paperType: 'mongodb' // Ensure it's set as mongodb type
    });
  };

  return (
    <div className="exam-creator">
      <div className="exam-details">
        <h2>Create New Exam</h2>
        
        <div className="form-group">
          <label>Title</label>
          <input
            type="text"
            name="title"
            value={examData.title}
            onChange={handleExamDataChange}
            placeholder="Enter exam title"
          />
        </div>

        <div className="form-row">
          <div className="form-group">
            <label>Start Time</label>
            <input
              type="datetime-local"
              name="startTime"
              value={examData.startTime}
              onChange={handleExamDataChange}
            />
          </div>

          <div className="form-group">
            <label>End Time</label>
            <input
              type="datetime-local"
              name="endTime"
              value={examData.endTime}
              onChange={handleExamDataChange}
            />
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label>Duration (minutes)</label>
            <input
              type="number"
              name="duration"
              value={examData.duration}
              onChange={handleExamDataChange}
              min="1"
            />
          </div>

          <div className="form-group">
            <label>Type</label>
            <select name="type" value={examData.type} onChange={handleExamDataChange}>
              <option value="ISA">ISA</option>
              <option value="QUIZ">Quiz</option>
            </select>
          </div>
        </div>

        <div className="form-group">
          <label>Description</label>
          <textarea
            name="description"
            value={examData.description}
            onChange={handleExamDataChange}
            placeholder="Enter exam description"
          />
        </div>
      </div>

      <div className="question-creator">
        <h3>Add Question</h3>
        
        <div className="form-group">
          <label>Question Text</label>
          <textarea
            name="text"
            value={currentQuestion.text}
            onChange={handleQuestionChange}
            placeholder="Enter question text"
          />
        </div>

        <div className="form-row">
          <div className="form-group">
            <label>Question Type</label>
            <select name="type" value={currentQuestion.type} onChange={handleQuestionChange}>
              <option value="mcq">Multiple Choice</option>
              <option value="descriptive">Descriptive</option>
            </select>
          </div>

          <div className="form-group">
            <label>Difficulty Level</label>
            <select name="difficultyLevel" value={currentQuestion.difficultyLevel} onChange={handleQuestionChange}>
              <option value="easy">Easy</option>
              <option value="medium">Medium</option>
              <option value="hard">Hard</option>
            </select>
          </div>

          <div className="form-group">
            <label>Marks</label>
            <input
              type="number"
              name="marks"
              value={currentQuestion.marks}
              onChange={handleQuestionChange}
              min="1"
            />
          </div>
        </div>

        {currentQuestion.type === 'mcq' && (
          <div className="options-section">
            <label>Options</label>
            {currentQuestion.options.map((option, index) => (
              <input
                key={index}
                type="text"
                value={option}
                onChange={(e) => handleOptionChange(index, e.target.value)}
                placeholder={`Option ${index + 1}`}
              />
            ))}
          </div>
        )}

        <div className="form-group">
          <label>Correct Answer</label>
          {currentQuestion.type === 'mcq' ? (
            <select
              name="correctAnswer"
              value={currentQuestion.correctAnswer}
              onChange={handleQuestionChange}
            >
              <option value="">Select correct option</option>
              {currentQuestion.options.map((option, index) => (
                <option key={index} value={index}>{option || `Option ${index + 1}`}</option>
              ))}
            </select>
          ) : (
            <textarea
              name="correctAnswer"
              value={currentQuestion.correctAnswer}
              onChange={handleQuestionChange}
              placeholder="Enter correct answer"
            />
          )}
        </div>

        <button className="add-question-btn" onClick={addQuestion}>
          <FaPlus /> Add Question
        </button>
      </div>

      <div className="question-pool">
        <h3>Question Pool</h3>
        
        {['easy', 'medium', 'hard'].map(difficulty => (
          <div key={difficulty} className={`difficulty-section ${difficulty}`}>
            <h4>{difficulty.charAt(0).toUpperCase() + difficulty.slice(1)} Questions</h4>
            {examData.questionPool[difficulty].map((question, index) => (
              <div key={index} className="question-item">
                <p>{question.text}</p>
                {question.type === 'mcq' && (
                  <ul>
                    {question.options.map((opt, i) => (
                      <li key={i} className={i === parseInt(question.correctAnswer) ? 'correct' : ''}>
                        {opt}
                      </li>
                    ))}
                  </ul>
                )}
                <div className="question-meta">
                  <span>{question.marks} marks</span>
                  <button onClick={() => removeQuestion(difficulty, index)}>
                    <FaTrash />
                  </button>
                </div>
              </div>
            ))}
          </div>
        ))}
      </div>

      <div className="exam-actions">
        <button className="save-exam-btn" onClick={handleSubmit}>
          <FaCheck /> Save Exam
        </button>
      </div>
    </div>
  );
};

export default ExamCreator; 