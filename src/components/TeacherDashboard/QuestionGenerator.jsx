import React, { useState } from 'react';

const QuestionGenerator = () => {
    const [prompt, setPrompt] = useState('');
    const [generatedQuestions, setGeneratedQuestions] = useState([]);
    const [loading, setLoading] = useState(false);

    const generateQuestions = async () => {
        setLoading(true);
        try {
            const response = await fetch('/api/questions/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ prompt, topic: 'General' }),
            });
            const data = await response.json();
            setGeneratedQuestions(data.questions);
        } catch (error) {
            console.error('Error generating questions:', error);
        }
        setLoading(false);
    };

    return (
        <div className="question-generator">
            <h2>AI Question Generator</h2>
            <div className="prompt-section">
                <textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="Describe the types of questions you want to generate..."
                    rows={4}
                    className="prompt-input"
                />
                <button 
                    onClick={generateQuestions}
                    disabled={loading}
                    className="generate-btn"
                >
                    {loading ? 'Generating...' : 'Generate Questions'}
                </button>
            </div>
            {generatedQuestions.length > 0 && (
                <div className="generated-questions">
                    {generatedQuestions.map((question, i) => (
                        <div key={i} className="question-card">
                            <h3>Question {i + 1}</h3>
                            <p>{question.text}</p>
                            {question.type === 'mcq' && (
                                <div className="options">
                                    {question.options.map((option, j) => (
                                        <div key={j} className={`option ${j === question.correct_answer ? 'correct' : ''}`}>
                                            {option}
                                        </div>
                                    ))}
                                </div>
                            )}
                            <div className="explanation">
                                <strong>Explanation:</strong> {question.suggested_answer}
                            </div>
                            <div className="metadata">
                                <span>Difficulty: {question.difficulty}</span>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default QuestionGenerator;