import React, { useState, useRef, useEffect } from 'react';
import './Chatbot.css';

const FIXED_PROMPTS = [
  "What is PESUprep?",
  "How can PESUprep help me learn?",
  "What courses are available?",
  "How does the AI assistant work?",
  "What are the learning features?",
  "How do I get started?"
];

const INITIAL_MESSAGE = {
  id: 'welcome',
  content: "👋 Hi! I'm your PESUprep assistant, part of PESUAcademy (pesuacademy.com/Academy). I'm here to help you excel in your academic journey!",
  sender: 'bot'
};

const Chatbot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([INITIAL_MESSAGE]);
  const [isLoading, setIsLoading] = useState(false);
  const [hasBeenOpened, setHasBeenOpened] = useState(false);
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);

  const scrollToBottom = () => {
    if (messagesContainerRef.current) {
      const scrollHeight = messagesContainerRef.current.scrollHeight;
      const height = messagesContainerRef.current.clientHeight;
      const maxScrollTop = scrollHeight - height;
      messagesContainerRef.current.scrollTop = maxScrollTop > 0 ? maxScrollTop : 0;
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Auto-open effect
  useEffect(() => {
    if (!hasBeenOpened) {
      const timer = setTimeout(() => {
        setIsOpen(true);
        setHasBeenOpened(true);
      }, 3000); // Open after 3 seconds

      return () => clearTimeout(timer);
    }
  }, [hasBeenOpened]);

  const handleClose = (e) => {
    e.stopPropagation();
    setIsOpen(false);
  };

  const handlePromptClick = async (prompt) => {
    // Add user message
    setMessages(prev => [...prev, {
      id: Date.now(),
      content: prompt,
      sender: 'user'
    }]);

    setIsLoading(true);

    try {
      // Predefined responses for each prompt
      const responses = {
        "What is PESUprep?": "PESUprep is an AI-powered learning companion integrated with PESUAcademy (pesuacademy.com/Academy). It's designed specifically for PES University students to help you master Machine Learning concepts through interactive learning, personalized quizzes, and real-time assistance! 🚀",
        "How can PESUprep help me learn?": "As part of the PESUAcademy ecosystem, I offer several ways to enhance your learning:\n• Interactive lessons with real-time feedback\n• Practice questions and quizzes\n• Personalized learning paths\n• AI-powered explanations of complex topics\n• Progress tracking and gamified learning\n• Direct integration with your PESUAcademy account 📚",
        "What courses are available?": "Through PESUAcademy (pesuacademy.com/Academy), we're currently focused on Machine Learning (UE22CS352A) with topics including:\n• Neural Networks\n• Support Vector Machines\n• Decision Trees\n• Clustering Algorithms\n• And many more! We're constantly adding new content. 🎓",
        "How does the AI assistant work?": "I use advanced AI to enhance your PESUAcademy experience:\n• Answer your questions about ML concepts\n• Generate practice problems\n• Provide detailed explanations\n• Adapt to your learning style\n• Offer real-time feedback on your progress\n• Seamlessly integrate with your PESUAcademy coursework 🤖",
        "What are the learning features?": "PESUprep enhances your PESUAcademy experience with:\n• Interactive tutorials\n• Mini-games for concept practice\n• Progress tracking\n• XP and achievement system\n• Personalized learning paths\n• AI-powered chat assistance\n• Direct access to course materials 🌟",
        "How do I get started?": "Getting started is easy:\n1. Sign in with your PESUAcademy account (pesuacademy.com/Academy)\n2. Choose your role (Student/Teacher)\n3. Select your course\n4. Start learning with our interactive lessons\n5. Track your progress and earn achievements!\n\nMake sure you're logged into PESUAcademy to access all features! 🎯"
      };

      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Add bot response
      setMessages(prev => [...prev, {
        id: Date.now(),
        content: responses[prompt] || "I'm sorry, I don't have information about that yet. Please try another question!",
        sender: 'bot'
      }]);
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        id: Date.now(),
        content: "I'm sorry, I encountered an error. Please try again.",
        sender: 'bot'
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={`chatbot-widget ${isOpen ? 'open' : ''} ${!hasBeenOpened ? 'shine' : ''}`}>
      <button 
        className="chatbot-toggle"
        onClick={() => setIsOpen(!isOpen)}
      >
        {isOpen ? '💬' : '💬'}
      </button>

      <div className="chatbot-container">
        <div className="chatbot-header">
          <img src="/bot.png" alt="Bot" className="bot-avatar" />
          <div className="header-text">
            <h3>PESUprep Assistant</h3>
            <p>Ask me anything!</p>
          </div>
          <button className="close-button" onClick={handleClose}>✕</button>
        </div>

        <div className="messages-area" ref={messagesContainerRef}>
          <div className="messages-wrapper">
            {messages.map((msg) => (
              <div key={msg.id} className={`message ${msg.sender}`}>
                {msg.sender === 'bot' && (
                  <div className="avatar">
                    <img src="/bot.png" alt="Bot" className="bot-avatar" />
                  </div>
                )}
                <div className="message-content">
                  {msg.content}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="message bot">
                <div className="avatar">
                  <img src="/bot.png" alt="Bot" className="bot-avatar" />
                </div>
                <div className="message-content">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
        
        <div className="fixed-prompts">
          {FIXED_PROMPTS.map((prompt) => (
            <button
              key={prompt}
              className="prompt-button"
              onClick={() => handlePromptClick(prompt)}
              disabled={isLoading}
            >
              {prompt}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Chatbot; 