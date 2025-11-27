import React, { useState, useEffect } from 'react';
import { Player } from '@lottiefiles/react-lottie-player';
import robotAnimation from '../../assets/animations/robot.json';
import './Dashboard.css';

const MiniGame = ({ game, onComplete, onClose }) => {
  const [timeLeft, setTimeLeft] = useState(game.timeLimit);
  const [score, setScore] = useState(0);
  const [gameState, setGameState] = useState('ready'); // ready, playing, completed
  const [gameData, setGameData] = useState(null);
  const [selectedCards, setSelectedCards] = useState([]);
  const [matchedPairs, setMatchedPairs] = useState([]);
  const [canClick, setCanClick] = useState(true);

  useEffect(() => {
    if (gameState === 'playing') {
      const timer = setInterval(() => {
        setTimeLeft(prev => {
          if (prev <= 1) {
            clearInterval(timer);
            endGame();
            return 0;
          }
          return prev - 1;
        });
      }, 1000);

      return () => clearInterval(timer);
    }
  }, [gameState]);

  useEffect(() => {
    // Initialize game data based on game type
    switch (game.type) {
      case 'matching':
        const pairs = [
          { id: 1, content: 'Neural Network', type: 'term', matches: 6 },
          { id: 2, content: 'Decision Tree', type: 'term', matches: 7 },
          { id: 3, content: 'K-Means', type: 'term', matches: 8 },
          { id: 4, content: 'Linear Regression', type: 'term', matches: 9 },
          { id: 5, content: 'SVM', type: 'term', matches: 10 },
          { id: 6, content: 'Deep Learning', type: 'definition', matches: 1 },
          { id: 7, content: 'Classification', type: 'definition', matches: 2 },
          { id: 8, content: 'Clustering', type: 'definition', matches: 3 },
          { id: 9, content: 'Prediction', type: 'definition', matches: 4 },
          { id: 10, content: 'Support Vector Classification', type: 'definition', matches: 5 }
        ].sort(() => Math.random() - 0.5);

        setGameData({ cards: pairs });
        break;

      case 'pattern':
        setGameData({
          patterns: [
            { sequence: [2, 4, 6, 8], answer: 10, hint: "What's the pattern?" },
            { sequence: [1, 3, 9, 27], answer: 81, hint: "Think exponentially!" },
            { sequence: [1, 1, 2, 3, 5], answer: 8, hint: "Famous sequence!" }
          ],
          currentPattern: 0,
          attempts: 0
        });
        break;

      case 'building':
        if (game.title === 'Network Builder') {
          setGameData({
            layers: [{ neurons: 2, type: 'input' }],
            maxLayers: 4,
            targetArchitecture: [2, 4, 4, 1],
            message: 'Build a neural network for binary classification!'
          });
        } else { // Tree Builder
          setGameData({
            nodes: [{
              id: 'root',
              question: 'Start your decision tree',
              children: [],
              depth: 0
            }],
            maxDepth: 3,
            targetFeatures: ['age', 'income', 'credit_score'],
            message: 'Build a decision tree for loan approval!'
          });
        }
        break;

      case 'racing':
        setGameData({
          position: 0,
          speed: 0,
          obstacles: [
            { pos: 20, type: 'local_minimum' },
            { pos: 40, type: 'vanishing_gradient' },
            { pos: 60, type: 'overfitting' },
            { pos: 80, type: 'underfitting' }
          ],
          powerups: [
            { pos: 30, type: 'momentum' },
            { pos: 50, type: 'learning_rate_boost' },
            { pos: 70, type: 'dropout_shield' }
          ],
          finish: 100
        });
        break;

      case 'quiz':
        setGameData({
          questions: [
            {
              question: "What's the main purpose of a Decision Tree?",
              options: [
                "Classification and Regression",
                "Only Classification",
                "Image Processing",
                "Natural Language Processing"
              ],
              correct: 0,
              explanation: "Decision trees can handle both classification and regression tasks!"
            },
            {
              question: "Which metric is NOT used in Decision Trees?",
              options: [
                "Gini Index",
                "Information Gain",
                "Gradient Descent",
                "Entropy"
              ],
              correct: 2,
              explanation: "Gradient Descent is used in neural networks, not decision trees!"
            },
            {
              question: "What's the maximum depth recommended to avoid overfitting?",
              options: [
                "No limit",
                "3-10 levels",
                "100 levels",
                "1 level"
              ],
              correct: 1,
              explanation: "3-10 levels usually provides good balance between complexity and accuracy!"
            }
          ],
          currentQuestion: 0,
          answered: [],
          showExplanation: false
        });
        break;
      default:
        break;
    }
  }, [game.type, game.title]);

  const startGame = () => {
    setGameState('playing');
  };

  const endGame = () => {
    setGameState('completed');
    onComplete(score);
  };

  const handleCardClick = (card) => {
    if (!canClick || card.matched || selectedCards.includes(card)) return;

    const newSelected = [...selectedCards, card];
    setSelectedCards(newSelected);

    if (newSelected.length === 2) {
      setCanClick(false);
      const [first, second] = newSelected;
      
      if (first.matches === second.id || second.matches === first.id) {
        // Matched!
        setScore(prev => prev + 10);
        setMatchedPairs(prev => [...prev, first.id, second.id]);
        setSelectedCards([]);
        setCanClick(true);

        if (matchedPairs.length + 2 === gameData.cards.length) {
          // All pairs matched!
          endGame();
        }
      } else {
        // No match
        setTimeout(() => {
          setSelectedCards([]);
          setCanClick(true);
        }, 1000);
      }
    }
  };

  const handlePatternGuess = (guess) => {
    const currentPattern = gameData.patterns[gameData.currentPattern];
    if (parseInt(guess) === currentPattern.answer) {
      setScore(prev => prev + 10);
      if (gameData.currentPattern < gameData.patterns.length - 1) {
        setGameData({
          ...gameData,
          currentPattern: gameData.currentPattern + 1,
          attempts: 0
        });
      } else {
        endGame();
      }
    } else {
      setGameData({
        ...gameData,
        attempts: gameData.attempts + 1
      });
    }
  };

  const handleNetworkBuild = (layerConfig) => {
    const { neurons, activation } = layerConfig;
    const newLayers = [...gameData.layers, { neurons, activation }];
    setGameData({
      ...gameData,
      layers: newLayers,
      message: newLayers.length === gameData.maxLayers ? 
        'Network complete! Training...' : 
        'Add the next layer!'
    });

    if (newLayers.length === gameData.maxLayers) {
      const isCorrect = newLayers.every((layer, i) => 
        layer.neurons === gameData.targetArchitecture[i]
      );
      setScore(prev => prev + (isCorrect ? 50 : 20));
      setTimeout(endGame, 1500);
    } else {
      setScore(prev => prev + 5);
    }
  };

  const handleTreeBuild = (decision) => {
    if (!decision.feature || !decision.threshold) return;

    const addNode = (nodes, parentId, newNode) => {
      return nodes.map(node => {
        if (node.id === parentId) {
          return {
            ...node,
            children: [...(node.children || []), newNode]
          };
        }
        if (node.children) {
          return {
            ...node,
            children: addNode(node.children, parentId, newNode)
          };
        }
        return node;
      });
    };

    const newNode = {
      id: `node-${Date.now()}`,
      feature: decision.feature,
      threshold: decision.threshold,
      children: [],
      depth: decision.depth
    };

    const newNodes = addNode(gameData.nodes, decision.parentId, newNode);
    setGameData({
      ...gameData,
      nodes: newNodes
    });

    // Check if we used all target features
    const usedFeatures = new Set(gameData.nodes.map(n => n.feature));
    if (gameData.targetFeatures.every(f => usedFeatures.has(f))) {
      setScore(prev => prev + 50);
      setTimeout(endGame, 1500);
    } else {
      setScore(prev => prev + 5);
    }
  };

  const handleRacing = (action) => {
    const newPosition = Math.max(0, Math.min(100, 
      gameData.position + (action === 'accelerate' ? 5 : -2)
    ));

    // Check collisions with obstacles
    const hitObstacle = gameData.obstacles.some(obs => 
      Math.abs(obs.pos - newPosition) < 5
    );

    // Check powerups
    const powerup = gameData.powerups.find(p => 
      Math.abs(p.pos - newPosition) < 5 && !p.collected
    );

    if (powerup) {
      setScore(prev => prev + 10);
      setGameData({
        ...gameData,
        position: newPosition,
        powerups: gameData.powerups.map(p => 
          p === powerup ? { ...p, collected: true } : p
        )
      });
    } else if (!hitObstacle) {
      setGameData({
        ...gameData,
        position: newPosition
      });
      setScore(prev => prev + 1);

      if (newPosition >= gameData.finish) {
        endGame();
      }
    }
  };

  const handleQuizAnswer = (answerIndex) => {
    const currentQ = gameData.questions[gameData.currentQuestion];
    const isCorrect = answerIndex === currentQ.correct;
    
    if (isCorrect) {
      setScore(prev => prev + 10);
    }

    setGameData({
      ...gameData,
      showExplanation: true
    });

    setTimeout(() => {
      if (gameData.currentQuestion < gameData.questions.length - 1) {
        setGameData({
          ...gameData,
          currentQuestion: gameData.currentQuestion + 1,
          answered: [...gameData.answered, answerIndex],
          showExplanation: false
        });
      } else {
        endGame();
      }
    }, 2000);
  };

  const renderGame = () => {
    switch (game.type) {
      case 'matching':
        return (
          <div className="matching-game">
            {gameData?.cards.map(card => (
              <div
                key={card.id}
                className={`pair-card ${
                  matchedPairs.includes(card.id) ? 'matched' : ''
                } ${selectedCards.includes(card) ? 'selected' : ''}`}
                onClick={() => handleCardClick(card)}
              >
                <div className="card-content">
                  {card.content}
                </div>
              </div>
            ))}
          </div>
        );

      case 'pattern':
        return (
          <div className="pattern-game">
            <div className="pattern-display">
              <h3>Find the next number:</h3>
              <div className="sequence">
                {gameData?.patterns[gameData.currentPattern].sequence.join(' → ')} → ?
              </div>
              <div className="hint">
                {gameData?.patterns[gameData.currentPattern].hint}
              </div>
            </div>
            <div className="pattern-input">
              <input
                type="number"
                placeholder="Enter your answer"
                onChange={(e) => handlePatternGuess(e.target.value)}
              />
              {gameData?.attempts > 0 && (
                <div className="feedback">Try again! Attempts: {gameData.attempts}</div>
              )}
            </div>
          </div>
        );

      case 'building':
        return game.title === 'Network Builder' ? (
          <div className="network-builder">
            <div className="network-message">{gameData?.message}</div>
            <div className="network-visualization">
              {gameData?.layers.map((layer, i) => (
                <div key={i} className="layer">
                  {Array(layer.neurons).fill(0).map((_, j) => (
                    <div key={j} className="neuron" />
                  ))}
                </div>
              ))}
            </div>
            <div className="network-controls">
              <button onClick={() => handleNetworkBuild({ neurons: 2, activation: 'relu' })}>
                Add 2 Neurons
              </button>
              <button onClick={() => handleNetworkBuild({ neurons: 4, activation: 'relu' })}>
                Add 4 Neurons
              </button>
              <button onClick={() => handleNetworkBuild({ neurons: 1, activation: 'sigmoid' })}>
                Add Output (1 Neuron)
              </button>
            </div>
          </div>
        ) : (
          <div className="tree-builder">
            <div className="tree-message">{gameData?.message}</div>
            <div className="tree-visualization">
              {gameData?.nodes.map(node => (
                <div key={node.id} className="tree-node" style={{ marginLeft: node.depth * 40 }}>
                  {node.feature ? `${node.feature} > ${node.threshold}` : 'Start here'}
                </div>
              ))}
            </div>
            <div className="tree-controls">
              <select onChange={(e) => setGameData({
                ...gameData,
                selectedFeature: e.target.value
              })}>
                <option value="">Select Feature</option>
                {gameData?.targetFeatures.map(f => (
                  <option key={f} value={f}>{f}</option>
                ))}
              </select>
              <input
                type="number"
                placeholder="Threshold"
                onChange={(e) => setGameData({
                  ...gameData,
                  threshold: e.target.value
                })}
              />
              <button onClick={() => handleTreeBuild({
                feature: gameData.selectedFeature,
                threshold: gameData.threshold,
                parentId: 'root',
                depth: 1
              })}>
                Add Decision
              </button>
            </div>
          </div>
        );

      case 'racing':
        return (
          <div className="gradient-race">
            <div className="race-track">
              {gameData?.obstacles.map((obs, i) => (
                <div
                  key={i}
                  className={`obstacle ${obs.type}`}
                  style={{ left: `${obs.pos}%` }}
                />
              ))}
              {gameData?.powerups.map((powerup, i) => (
                <div
                  key={i}
                  className={`powerup ${powerup.type} ${powerup.collected ? 'collected' : ''}`}
                  style={{ left: `${powerup.pos}%` }}
                />
              ))}
              <div
                className="player"
                style={{ left: `${gameData?.position}%` }}
              />
              <div className="finish-line" style={{ left: '95%' }} />
            </div>
            <div className="race-controls">
              <button onClick={() => handleRacing('brake')}>← Brake</button>
              <button onClick={() => handleRacing('accelerate')}>Accelerate →</button>
            </div>
          </div>
        );

      case 'quiz':
        const currentQ = gameData?.questions[gameData.currentQuestion];
        return (
          <div className="quiz-game">
            <div className="question">
              <h3>Question {gameData.currentQuestion + 1} of {gameData.questions.length}</h3>
              <p>{currentQ?.question}</p>
            </div>
            <div className="options">
              {currentQ?.options.map((option, i) => (
                <button
                  key={i}
                  onClick={() => handleQuizAnswer(i)}
                  className={`option ${
                    gameData.showExplanation ? 
                      (i === currentQ.correct ? 'correct' : 'incorrect') : ''
                  }`}
                  disabled={gameData.showExplanation}
                >
                  {option}
                </button>
              ))}
            </div>
            {gameData.showExplanation && (
              <div className="explanation">
                {currentQ.explanation}
              </div>
            )}
          </div>
        );

      default:
        return <div>Game type not supported</div>;
    }
  };

  return (
    <div className="mini-game-container">
      <div className="game-header">
        <h3>{game.title}</h3>
        <div className="game-stats">
          <span>Time: {timeLeft}s</span>
          <span>Score: {score}</span>
        </div>
        <button onClick={onClose} className="close-button">×</button>
      </div>

      {gameState === 'ready' && (
        <div className="game-start">
          <Player
            autoplay
            loop
            src={robotAnimation}
            style={{ height: '150px', width: '150px' }}
          />
          <h2>Ready to slay this {game.title}?</h2>
          <p>{game.description}</p>
          <button onClick={startGame} className="start-button">
            Start Game
          </button>
        </div>
      )}

      {gameState === 'playing' && renderGame()}

      {gameState === 'completed' && (
        <div className="game-completed">
          <h2>Slay! You finished with {score} points!</h2>
          <button onClick={onClose}>Close</button>
        </div>
      )}
    </div>
  );
};

export default MiniGame;