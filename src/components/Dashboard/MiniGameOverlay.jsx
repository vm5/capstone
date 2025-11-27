import React, { useMemo, useEffect, useState } from 'react';

// Minimal mini-game loader that immediately shows a playable view.
// Emits XP via onComplete so Dashboard can award and persist.
export default function MiniGameOverlay({ gameId, topicTitle, onClose, onComplete }) {
  const title = useMemo(() => gameId || 'Mini Game', [gameId]);

  // GAME: Matching Pairs (works for any topic, no assets needed)
  // Cards will be simple algorithm icons/labels
  const [cards, setCards] = useState([]);
  const [flippedIds, setFlippedIds] = useState([]);
  const [matchedIds, setMatchedIds] = useState(new Set());
  const [moves, setMoves] = useState(0);

  useEffect(() => {
    // Build 3 pairs → 6 cards
    const base = [
      { id: 'A', label: 'KNN' },
      { id: 'B', label: 'SVM' },
      { id: 'C', label: 'DT' }
    ];
    const duplicated = base.flatMap((b, i) => ([
      { key: `${b.id}-1`, pair: b.id, label: b.label },
      { key: `${b.id}-2`, pair: b.id, label: b.label }
    ]));
    // Shuffle
    for (let i = duplicated.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [duplicated[i], duplicated[j]] = [duplicated[j], duplicated[i]];
    }
    setCards(duplicated);
    setFlippedIds([]);
    setMatchedIds(new Set());
    setMoves(0);
  }, [gameId]);

  // Auto-submit when player is stuck: time limit OR move limit
  const [timeLeft, setTimeLeft] = useState(90); // seconds
  const [completed, setCompleted] = useState(false);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  useEffect(() => {
    if (completed) return;
    const timer = setInterval(() => setTimeLeft((t) => t - 1), 1000);
    return () => clearInterval(timer);
  }, [completed]);

  useEffect(() => {
    if (completed) return;
    const exceededMoves = moves >= 20;
    const outOfTime = timeLeft <= 0;
    const allDone = matchedIds.size === cards.length && cards.length > 0;
    if (allDone) {
      setCompleted(true);
      const base = 150;
      const bonus = Math.max(0, 100 - (moves * 10));
      onComplete({ xp: base + bonus });
      return;
    }
    if (exceededMoves || outOfTime) {
      setCompleted(true);
      onComplete({ xp: 0 }); // no XP if failed
      return;
    }
  }, [moves, timeLeft, matchedIds, cards, completed, onComplete]);

  const handleFlip = (key) => {
    if (matchedIds.has(key)) return;
    if (flippedIds.includes(key)) return;
    if (flippedIds.length === 2) return;

    const next = [...flippedIds, key];
    setFlippedIds(next);
    if (next.length === 2) {
      setMoves(m => m + 1);
      const [k1, k2] = next;
      const c1 = cards.find(c => c.key === k1);
      const c2 = cards.find(c => c.key === k2);
      if (c1 && c2 && c1.pair === c2.pair) {
        const newMatched = new Set(matchedIds);
        newMatched.add(k1);
        newMatched.add(k2);
        setTimeout(() => {
          setMatchedIds(newMatched);
          setFlippedIds([]);
          // Completed all pairs → award XP based on efficiency
          if (newMatched.size === cards.length) {
            const base = 150;
            const bonus = Math.max(0, 100 - (moves * 10));
            onComplete({ xp: base + bonus });
          }
        }, 300);
      } else {
        setTimeout(() => setFlippedIds([]), 600);
      }
    }
  };

  const renderMatchingGame = () => (
    <div>
      <div className="question" style={{ marginBottom: '0.75rem' }}>
        Match the pairs (moves: {moves}, time: {Math.max(0, timeLeft)}s). Tip: Click two cards to flip. No drag & drop.
      </div>
      <div style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
        <button
          className="game-button"
          onClick={() => {
            // reset the board
            const triggerReset = () => {
              const base = [
                { id: 'A', label: 'KNN' },
                { id: 'B', label: 'SVM' },
                { id: 'C', label: 'DT' }
              ];
              const duplicated = base.flatMap((b) => ([
                { key: `${b.id}-1`, pair: b.id, label: b.label },
                { key: `${b.id}-2`, pair: b.id, label: b.label }
              ]));
              for (let i = duplicated.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [duplicated[i], duplicated[j]] = [duplicated[j], duplicated[i]];
              }
              setCards(duplicated);
              setFlippedIds([]);
              setMatchedIds(new Set());
              setMoves(0);
            };
            triggerReset();
          }}
        >
          Reset Game
        </button>
        <button className="close-button" onClick={onClose}>Close</button>
      </div>
      <div className="matching-game">
        {cards.map(card => {
          const isFlipped = flippedIds.includes(card.key) || matchedIds.has(card.key);
          return (
            <div
              key={card.key}
              className={`pair-card ${isFlipped ? 'selected' : ''}`}
              style={{ cursor: 'pointer', userSelect: 'none' }}
              onClick={() => handleFlip(card.key)}
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  handleFlip(card.key);
                }
              }}
            >
              <span>{isFlipped ? card.label : '❓'}</span>
            </div>
          );
        })}
      </div>
    </div>
  );

  // GAME 2: Reaction — click as fast as possible after signal
  const [reactionState, setReactionState] = useState('idle'); // idle|waiting|go|done|toosoon
  const [reactionMsg, setReactionMsg] = useState('');
  const [startTs, setStartTs] = useState(null);
  const [reactionTime, setReactionTime] = useState(null);
  useEffect(() => {
    if (gameId !== 'reaction') return;
    setReactionState('waiting');
    setReactionMsg('Wait for green...');
    const delay = Math.floor(Math.random() * 2000) + 800;
    const t = setTimeout(() => {
      setReactionState('go');
      setReactionMsg('GO! Click now!');
      setStartTs(performance.now());
    }, delay);
    return () => clearTimeout(t);
  }, [gameId]);

  const handleReactionClick = () => {
    if (reactionState === 'waiting') {
      setReactionState('toosoon');
      setReactionMsg('Too soon!');
      setTimeout(() => onComplete({ xp: 30 }), 600);
      return;
    }
    if (reactionState === 'go') {
      const rt = Math.floor(performance.now() - startTs);
      setReactionTime(rt);
      setReactionState('done');
      setReactionMsg(`Your time: ${rt} ms`);
      const xp = Math.max(40, 180 - Math.floor(rt / 2));
      setTimeout(() => onComplete({ xp }), 800);
    }
  };

  const renderReactionGame = () => (
    <div>
      <div className="question" style={{ marginBottom: '0.75rem' }}>
        Reaction Test. Click as soon as the panel turns green.
      </div>
      <div style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
        <button className="close-button" onClick={onClose}>Close</button>
      </div>
      <div
        onClick={handleReactionClick}
        style={{
          height: 160,
          borderRadius: 12,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontWeight: 700,
          cursor: 'pointer',
          userSelect: 'none',
          background: reactionState === 'go' ? '#22c55e' : '#f43f5e',
        }}
      >
        {reactionMsg}
      </div>
      {reactionTime !== null && (
        <div style={{ marginTop: 10 }}>Final: {reactionTime} ms</div>
      )}
    </div>
  );

  // GAME 3: Sequence — remember and repeat a short pattern
  const [sequence, setSequence] = useState([]);
  const [userSeq, setUserSeq] = useState([]);
  const [seqMsg, setSeqMsg] = useState('');
  const pads = ['A', 'B', 'C', 'D'];
  useEffect(() => {
    if (gameId !== 'sequence') return;
    const build = [];
    for (let i = 0; i < 4; i++) {
      build.push(pads[Math.floor(Math.random() * pads.length)]);
    }
    setSequence(build);
    setUserSeq([]);
    setSeqMsg('Watch the sequence...');
    let idx = 0;
    const show = setInterval(() => {
      setSeqMsg(`Step ${idx + 1}: ${build[idx]}`);
      idx += 1;
      if (idx >= build.length) {
        clearInterval(show);
        setSeqMsg('Now repeat it by clicking the pads');
      }
    }, 700);
    return () => clearInterval(show);
  }, [gameId]);

  const pressPad = (p) => {
    if (seqMsg.startsWith('Watch')) return;
    const next = [...userSeq, p];
    setUserSeq(next);
    const i = next.length - 1;
    if (next[i] !== sequence[i]) {
      setSeqMsg('Oops! Incorrect.');
      setTimeout(() => onComplete({ xp: 40 }), 600);
      return;
    }
    if (next.length === sequence.length) {
      setSeqMsg('Great job!');
      const xp = 150 + 50; // small bonus
      setTimeout(() => onComplete({ xp }), 600);
      return;
    }
  };

  const renderSequenceGame = () => (
    <div>
      <div className="question" style={{ marginBottom: '0.75rem' }}>
        Memory Sequence. Repeat the pattern.
      </div>
      <div style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
        <button className="close-button" onClick={onClose}>Close</button>
      </div>
      <div style={{ marginBottom: 10 }}>{seqMsg}</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 120px)', gap: 12 }}>
        {pads.map(p => (
          <button
            key={p}
            className="game-button"
            style={{ height: 80 }}
            onClick={() => pressPad(p)}
          >
            {p}
          </button>
        ))}
      </div>
    </div>
  );

  // GAME 4: ClassifierChoice — pick the best algorithm for a scenario
  const classifierQuestions = [
    {
      q: 'High-dimensional text classification with sparse features',
      options: ['KNN', 'Naive Bayes', 'K-Means', 'Linear Regression'],
      correct: 1
    },
    {
      q: 'Non-linear boundaries with medium data and kernels',
      options: ['SVM (RBF)', 'Naive Bayes', 'K-Means', 'Logistic Regression'],
      correct: 0
    }
  ];
  const [clfIdx, setClfIdx] = useState(0);
  const [clfSel, setClfSel] = useState(null);
  const renderClassifierGame = () => {
    const item = classifierQuestions[clfIdx];
    return (
      <div>
        <div className="question" style={{ marginBottom: '0.75rem' }}>
          Choose the most suitable algorithm:
        </div>
        <div style={{ marginBottom: 8, fontWeight: 600 }}>{item.q}</div>
        <div className="options" style={{ marginBottom: 12 }}>
          {item.options.map((opt, i) => (
            <button
              key={i}
              className="game-button"
              style={{ textAlign: 'left' }}
              onClick={() => setClfSel(i)}
            >
              {opt}
            </button>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="close-button" onClick={onClose}>Close</button>
          <button
            className="game-button"
            disabled={clfSel === null}
            onClick={() => {
              const correct = clfSel === item.correct;
              const xp = correct ? 180 : 0;
              if (correct && clfIdx < classifierQuestions.length - 1) {
                setClfIdx(clfIdx + 1);
                setClfSel(null);
              } else {
                onComplete({ xp });
              }
            }}
          >
            {clfIdx < classifierQuestions.length - 1 ? 'Next' : 'Submit'}
          </button>
        </div>
      </div>
    );
  };

  // GAME 5: HyperparamTuning — pick a stable learning rate
  const [lrSel, setLrSel] = useState(null);
  const renderHyperparamGame = () => (
    <div>
      <div className="question" style={{ marginBottom: '0.75rem' }}>
        Choose a learning rate that converges stably for a simple MLP on MNIST.
      </div>
      <div className="options" style={{ marginBottom: 12 }}>
        {[0.0001, 0.001, 0.01, 0.5].map((lr) => (
          <button
            key={lr}
            className="game-button"
            onClick={() => setLrSel(lr)}
          >
            lr = {lr}
          </button>
        ))}
      </div>
      <div style={{ display: 'flex', gap: 8 }}>
        <button className="close-button" onClick={onClose}>Close</button>
        <button
          className="game-button"
          disabled={lrSel === null}
          onClick={() => {
            // Treat 0.001 as optimal, 0.0001/0.01 acceptable, 0.5 bad
            const xp = lrSel === 0.001 ? 200 : 0; // only correct gets XP
            onComplete({ xp });
          }}
        >
          Submit
        </button>
      </div>
    </div>
  );

  // GAME 6: Short Quiz (validated MCQs with explanations)
  const quizQs = [
    {
      q: 'Which algorithm is unsupervised?',
      options: ['Linear Regression', 'K-Means', 'Logistic Regression', 'Naive Bayes'],
      correct: 1,
      expl: 'K-Means is a clustering algorithm and does not require labels.'
    },
    {
      q: 'Which metric is best for imbalanced classification?',
      options: ['Accuracy', 'Precision/Recall', 'MSE', 'R-Squared'],
      correct: 1,
      expl: 'Precision/Recall (or F1) reflect performance on the minority class.'
    },
    {
      q: 'Regularization that drives weights to exactly 0:',
      options: ['L1 (Lasso)', 'L2 (Ridge)', 'Dropout', 'BatchNorm'],
      correct: 0,
      expl: 'L1 induces sparsity by pushing some weights to zero.'
    }
  ];
  const [quizIdx, setQuizIdx] = useState(0);
  const [quizSel, setQuizSel] = useState(null);
  const [quizScore, setQuizScore] = useState(0);
  const [quizMsg, setQuizMsg] = useState('');
  const answerQuiz = () => {
    const item = quizQs[quizIdx];
    if (quizSel === null) return;
    const correct = quizSel === item.correct;
    setQuizMsg(correct ? 'Correct!' : `Incorrect. ${item.expl}`);
    if (correct) setQuizScore(s => s + 1);
    setTimeout(() => {
      if (quizIdx < quizQs.length - 1) {
        setQuizIdx(quizIdx + 1);
        setQuizSel(null);
        setQuizMsg('');
      } else {
        // award XP: 100 per correct
        onComplete({ xp: quizScore * 100 + (correct ? 100 : 0) });
      }
    }, 800);
  };
  const renderQuiz = () => {
    const item = quizQs[quizIdx];
    return (
      <div>
        <div className="question" style={{ marginBottom: '0.75rem' }}>
          Quiz ({quizIdx + 1}/{quizQs.length}) — Score: {quizScore}
        </div>
        <div style={{ marginBottom: 8, fontWeight: 600 }}>{item.q}</div>
        <div className="options" style={{ display: 'grid', gap: 8, marginBottom: 12 }}>
          {item.options.map((opt, i) => (
            <button
              key={i}
              className="game-button"
              style={{ textAlign: 'left', border: quizSel === i ? '2px solid #4caf50' : '1px solid rgba(255,255,255,0.15)' }}
              onClick={() => setQuizSel(i)}
            >
              {opt}
            </button>
          ))}
        </div>
        {quizMsg && <div style={{ marginBottom: 8 }}>{quizMsg}</div>}
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="close-button" onClick={onClose}>Close</button>
          <button className="game-button" disabled={quizSel === null} onClick={answerQuiz}>Submit</button>
        </div>
      </div>
    );
  };

  return (
    <div className="mini-game-container" style={{ maxHeight: '80vh', overflowY: 'auto' }}>
      <div className="game-header">
        <h2>{title} — {topicTitle}</h2>
        <button className="close-button" onClick={onClose}>✕</button>
      </div>

      {gameId === 'classifier' ? renderClassifierGame()
        : gameId === 'hyperparams' ? renderHyperparamGame()
        : gameId === 'reaction' ? renderReactionGame()
        : gameId === 'sequence' ? renderSequenceGame()
        : gameId === 'quiz' ? renderQuiz()
        : renderMatchingGame()}
    </div>
  );
}


