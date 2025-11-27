import React, { useState } from 'react';
import './QuestionGenerator.css';

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
        <div className="question-paper-section">
            <div className="section-header">
                <h2>Question Paper Generation</h2>
                <button className="create-paper-btn">
                    <i className="fas fa-plus"></i>
                    Create New Paper
                </button>
            </div>

            <div className="papers-grid">
                {/* Example Paper Card */}
                <div className="paper-card">
                    <div className="paper-id">6807c2154c9d3ff128853629</div>
                    <div className="paper-info">
                        <h3>UE22CS352A</h3>
                        <div className="paper-meta">
                            <div className="meta-item">
                                <i className="fas fa-clock"></i>
                                Duration: 10 minutes
                            </div>
                            <div className="meta-item">
                                <i className="fas fa-star"></i>
                                Total Marks: 10
                            </div>
                            <div className="meta-item">
                                <i className="fas fa-database"></i>
                                Source: database
                            </div>
                        </div>
                    </div>

                    <div className="paper-actions">
                        <button className="action-btn view-btn">
                            <i className="fas fa-eye"></i>
                            View
                        </button>
                        <button className="action-btn delete-btn">
                            <i className="fas fa-trash"></i>
                            Delete
                        </button>
                    </div>

                    <div className="source-info">
                        <span>Created: 7/2/2025</span>
                        <span>Source: file</span>
                    </div>
                </div>

                {/* Example Paper Card with Download */}
                <div className="paper-card">
                    <div className="paper-id">6807c2154c9d3ff128853629</div>
                    <div className="paper-info">
                        <h3>UE22CS352A</h3>
                        <div className="paper-meta">
                            <div className="meta-item">
                                <i className="fas fa-clock"></i>
                                Duration: 10 minutes
                            </div>
                            <div className="meta-item">
                                <i className="fas fa-star"></i>
                                Total Marks: 10
                            </div>
                        </div>
                    </div>

                    <button className="download-btn">
                        <i className="fas fa-download"></i>
                        Download Paper
                    </button>

                    <div className="source-info">
                        <span>Created: 7/2/2025</span>
                        <span>Source: file</span>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default QuestionGenerator;