import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
// Removed background particles to declutter student view
import Splash from '../Splash/Splash';
import { getSchedules, loadPlayerStats, savePlayerStats } from '../../services/api';
import { generatePersonalizedExam } from '../../services/examService';
import './Dashboard.css';
import { Player } from '@lottiefiles/react-lottie-player';
import robotAnimation from '../../assets/animations/robot.json';
import rocketAnimation from '../../assets/animations/rocket.json';
import confettiAnimation from '../../assets/animations/confetti.json';
import { FaStar, FaLock, FaTrophy, FaGem, FaClock } from 'react-icons/fa';
import ExamView from './ExamView';
import GamificationPanel from './GamificationPanel';
import Leaderboard from './Leaderboard';
import MiniGameOverlay from './MiniGameOverlay';

function Dashboard() {
  const navigate = useNavigate();
  const location = useLocation();
  const [user, setUser] = useState(null);
  const [messages, setMessages] = useState([]);
  const [selectedCourse, setSelectedCourse] = useState(null);
  const [showSplash, setShowSplash] = useState(true);
  const [activeTab, setActiveTab] = useState('overview'); // 'overview' | 'exams' | 'resources'
  const [xp, setXp] = useState(0);
  const [data, setData] = useState(null);
  const [schedules, setSchedules] = useState([]);
  // Helper function to get user-specific stats key
  const getPlayerStatsKey = (userEmail) => {
    return userEmail ? `playerStats_${userEmail}` : 'playerStats';
  };

  const [playerStats, setPlayerStats] = useState({
    level: 1,
    xp: 0,
    nextLevelXp: 1000,
    completedTopics: [],
    achievements: [],
    inventory: [],
    streak: 0,
    coins: 0
  });

  const [activeQuests, setActiveQuests] = useState([
    {
      id: 1,
      title: "Absolute Unit of Knowledge",
      description: "Complete ML Basics without flopping! You got this bestie!",
      reward: 500,
      type: "learning",
      emoji: "🔥"
    },
    {
      id: 2,
      title: "Neural Network Girlboss/Boyboss",
      description: "Build a neural net that passes the vibe check!",
      reward: 750,
      type: "challenge",
      emoji: "💅"
    }
  ]);

  const [showConfetti, setShowConfetti] = useState(false);
  const [selectedTopic, setSelectedTopic] = useState(null);
  const [isMiniGameOpen, setIsMiniGameOpen] = useState(false);
  const [activeGameId, setActiveGameId] = useState(null);
  // Removed mini-game UI from student dashboard
  const [showPrerequisiteAlert, setShowPrerequisiteAlert] = useState(false);
  const [prerequisiteMessage, setPrerequisiteMessage] = useState('');
  const [selectedExam, setSelectedExam] = useState(null);
  const [examError, setExamError] = useState(null);
  const [loading, setLoading] = useState(false);
  
  const [error, setError] = useState(null);

  const miniGames = {
    "Match the Algorithm": {
      title: "Match the Algorithm",
      description: "Match these algorithms faster than your bestie can say 'slay'!",
      reward: 50,
      timeLimit: 60,
      type: "matching"
    },
    "Spot the Pattern": {
      title: "Spot the Pattern",
      description: "Find the pattern before the time runs out! No cap!",
      reward: 75,
      timeLimit: 45,
      type: "pattern"
    },
    "Network Builder": {
      title: "Network Builder",
      description: "Build that neural net and make it fire! It's giving genius vibes!",
      reward: 100,
      timeLimit: 120,
      type: "building"
    },
    "Gradient Rush": {
      title: "Gradient Rush",
      description: "Race through the gradients! Don't let that loss function catch you!",
      reward: 80,
      timeLimit: 60,
      type: "racing"
    },
    "Tree Builder": {
      title: "Tree Builder",
      description: "Grow your decision tree! Make it the main character!",
      reward: 90,
      timeLimit: 90,
      type: "building"
    },
    "Decision Master": {
      title: "Decision Master",
      description: "Choose your path! Every decision is a slay moment!",
      reward: 85,
      timeLimit: 75,
      type: "quiz"
    }
  };

  const mlTopics = [
    {
      id: 1,
      title: "Machine Learning",
      subtitle: "UE22CS352A",
      description: "Machine Learning Crash Course",
      xpReward: 300,
      level: 1,
      difficulty: "Noob Friendly",
      prerequisites: [],
      progress: 0,
      animation: "robot",
      catchPhrase: "fr fr, this is bussin!",
      minigames: ["Match the Algorithm", "Spot the Pattern"]
    },
    {
      id: 2,
      title: "Neural Networks: Big Brain Time 🧠",
      description: "Straight bussin! Build networks that think like your bestie.",
      xpReward: 500,
      level: 2,
      difficulty: "Kinda Sus",
      prerequisites: [1],
      progress: 0,
      animation: "brain",
      catchPhrase: "no cap, these networks be vibin!",
      minigames: ["Network Builder", "Gradient Rush"]
    },
    {
      id: 3,
      title: "Decision Trees: Choose Your Path, bestie! 🌳",
      description: "Slay these decisions like you're the main character!",
      xpReward: 400,
      level: 2,
      difficulty: "Mid Difficulty",
      prerequisites: [1],
      progress: 0,
      animation: "tree",
      catchPhrase: "let's get this bread!",
      minigames: ["Tree Builder", "Decision Master"]
    }
  ];

  // Centralized XP/level logic to avoid missing level-ups
  const awardXp = (amount) => {
    if (!amount || amount <= 0) return;
    setPlayerStats(prev => {
      let newXp = (prev.xp || 0) + amount;
      let newLevel = prev.level || 1;
      let nextLevelXp = prev.nextLevelXp || 1000;
      // Promote levels until XP falls below next threshold
      while (newXp >= nextLevelXp) {
        newLevel += 1;
        nextLevelXp = Math.floor(nextLevelXp * 1.5);
      }
      return { ...prev, xp: newXp, level: newLevel, nextLevelXp };
    });
  };

  const achievements = [
    {
      id: 1,
      title: "Main Character Energy",
      description: "Completed your first ML topic! You're so valid!",
      icon: "👑",
      unlocked: false,
      xpBonus: 100
    },
    {
      id: 2,
      title: "Bestie Behavior",
      description: "Helped another student understand ML! We stan!",
      icon: "💫",
      unlocked: false,
      xpBonus: 150
    }
  ];

  const course = { 
    id: 'UE22CS352A', 
    name: 'Machine Learning',
    link: 'https://developers.google.com/machine-learning/crash-course'
  };

  // particles removed

  const handleLogout = () => {
    // Only remove auth data, preserve playerStats (user-specific)
    localStorage.removeItem('user');
    localStorage.removeItem('token');
    navigate('/', { replace: true });
  };

  useEffect(() => {
    const userData = localStorage.getItem('user');
    if (!userData) {
      navigate('/', { replace: true });
      return;
    }

    const parsedUser = JSON.parse(userData);
    setUser(parsedUser);

    // Load user-specific playerStats
    const statsKey = getPlayerStatsKey(parsedUser.email);
    let savedStats = localStorage.getItem(statsKey);
    if (!savedStats) {
      // Fallback to legacy key if user-specific key missing
      savedStats = localStorage.getItem('playerStats');
    }
    if (savedStats) {
      try {
        const stats = JSON.parse(savedStats);
        setPlayerStats(stats);
      } catch (e) {
        console.error('Error loading player stats:', e);
      }
    }

    // Also try to load from backend
    (async () => {
      try {
        const remote = await loadPlayerStats(parsedUser.email);
        if (remote) {
          setPlayerStats(prev => ({ ...prev, ...remote }));
        }
      } catch (_) {}
    })();

    setTimeout(() => {
      setShowSplash(false);
    }, 2000);

    // Add regular HTTP fetch for data
    const fetchData = async () => {
      try {
        const response = await fetch('http://localhost:8000/');
        const data = await response.json();
        setData(data);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };
    
    fetchData();
    
    return () => {
      // Remove WebSocket cleanup
    };
  }, [navigate]);

  useEffect(() => {
    const fetchSchedules = async () => {
      try {
        const data = await getSchedules();
        console.log('Fetched schedules:', data.map(schedule => ({
          id: schedule._id,
          type: schedule.type,
          title: schedule.title,
          paperType: schedule.paperType,
          paperId: schedule.paperId?._id || schedule.paperId,
          hasQuestionPool: !!schedule.questionPool,
          questionPoolCounts: schedule.questionPool ? {
            easy: schedule.questionPool.easy?.length || 0,
            medium: schedule.questionPool.medium?.length || 0,
            hard: schedule.questionPool.hard?.length || 0
          } : null
        })));
        setSchedules(data);
      } catch (error) {
        console.error('Error fetching schedules:', error);
      }
    };

    fetchSchedules();
  }, []);

  const handleUpdate = async (updateData) => {
    try {
      const response = await fetch('http://localhost:8000/update', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updateData),
      });
      const data = await response.json();
      setData(data);
    } catch (error) {
      console.error('Error updating data:', error);
    }
  };

  const addMessage = (msg) => {
    setMessages(prev => [...prev, msg]);
  };

  const handleTopicClick = (topic) => {
    // Check if previous topics are completed
    const currentLevel = playerStats.level;
    const topicLevel = topic.level || 1; // Default to level 1 if not specified

    if (topicLevel > currentLevel) {
      setPrerequisiteMessage(`Bestie, you need to complete level ${currentLevel} first! No skips! 💅`);
      setShowPrerequisiteAlert(true);
      return;
    }

    // Check if specific prerequisites are met
    if (topic.prerequisites && topic.prerequisites.length > 0) {
      const missingPrereqs = topic.prerequisites.filter(
        prereq => !playerStats.completedTopics.includes(prereq)
      );
      
      if (missingPrereqs.length > 0) {
        const prettyNames = missingPrereqs.map(id => mlTopics.find(t => t.id === id)?.title || id);
        setPrerequisiteMessage(`You need to complete ${prettyNames.join(', ')} first! Keep slaying! ✨`);
        setShowPrerequisiteAlert(true);
        return;
      }
    }
    
    setSelectedTopic(topic);
    // Auto-launch ML-centric games per level
    // L1: classifier, L2: hyperparams, L3+: reaction (or sequence)
    const gameForLevel = 'quiz';
    setActiveGameId(gameForLevel);
    setIsMiniGameOpen(true);
  };

  // Update XP only after completing content
  const handleContentComplete = (topicId) => {
    const topic = mlTopics.find(t => t.id === topicId);
    if (!topic) return;

    awardXp(topic.xpReward);
    setPlayerStats(prev => ({
      ...prev,
      completedTopics: [...prev.completedTopics, topicId]
    }));
  };

  // minigame removed

  const completeQuest = (quest) => {
    handleContentComplete(quest.id);
    setPlayerStats(prev => ({
      ...prev,
      coins: prev.coins + quest.reward / 2
    }));
  };

  const renderPlayerStats = () => (
    <div className="player-stats">
      <div className="player-level">
        Level {playerStats.level}
        <div className="streak">🔥 {playerStats.streak} day streak</div>
      </div>
      <div className="xp-bar">
        <div 
          className="xp-progress" 
          style={{width: `${(playerStats.xp / playerStats.nextLevelXp) * 100}%`}}
        />
      </div>
      <div className="stats-details">
        <div className="xp-text">
          {playerStats.xp} / {playerStats.nextLevelXp} XP
        </div>
        <div className="coins">
          🪙 {playerStats.coins} coins
        </div>
      </div>
    </div>
  );

  // Simplified: hide skill tree from student main view
  const renderSkillTree = () => null;

  const renderTopicContent = () => null;

  const renderQuests = () => null;

  const renderAchievements = () => null;

  const renderBossChallenge = () => null;

  // Removed secondary mock XP structure to avoid redundancy

  // Save playerStats to localStorage whenever it changes (user-specific + legacy backup)
  useEffect(() => {
    if (user && user.email) {
      const statsKey = getPlayerStatsKey(user.email);
      localStorage.setItem(statsKey, JSON.stringify(playerStats));
      // Also keep a generic backup for compatibility
      localStorage.setItem('playerStats', JSON.stringify(playerStats));
      // Persist to backend
      savePlayerStats(user.email, playerStats);
    }
  }, [playerStats, user]);

  const resources = [
    {
      type: 'documentation',
      title: 'Scikit-learn Documentation',
      description: 'Official documentation for scikit-learn, a powerful ML library for Python.',
      link: 'https://scikit-learn.org/stable/documentation.html',
      readTime: '20 min',
      difficulty: 'Intermediate'
    },
    {
      type: 'tutorial',
      title: 'Neural Networks from Scratch',
      description: 'Build neural networks from the ground up to understand the fundamentals.',
      link: 'https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html',
      readTime: '60 min',
      difficulty: 'Advanced'
    },
    {
      type: 'course',
      title: 'Machine Learning Basics',
      description: 'Introduction to core ML concepts with hands-on examples.',
      link: 'https://www.coursera.org/learn/machine-learning',
      readTime: '40 min',
      difficulty: 'Beginner'
    },
    {
      type: 'documentation',
      title: 'TensorFlow Guide',
      description: 'Comprehensive guide to TensorFlow for deep learning applications.',
      link: 'https://www.tensorflow.org/guide',
      readTime: '30 min',
      difficulty: 'Intermediate'
    },
    {
      type: 'tutorial',
      title: 'Practical ML Projects',
      description: 'Real-world machine learning projects with step-by-step tutorials.',
      link: 'https://www.kaggle.com/learn/intro-to-machine-learning',
      readTime: '45 min',
      difficulty: 'Intermediate'
    },
    {
      type: 'course',
      title: 'Deep Learning Specialization',
      description: 'Advanced deep learning concepts and architectures explained.',
      link: 'https://www.deeplearning.ai/',
      readTime: '50 min',
      difficulty: 'Advanced'
    }
  ];

  const handleTakeExam = async (scheduleId) => {
    try {
      setLoading(true);
      setError(null);

      const schedule = schedules.find(s => s._id === scheduleId);
      if (!schedule) {
        throw new Error('Schedule not found');
      }

      if (!isExamAvailable(schedule)) {
        throw new Error('Exam is not currently available');
      }

      const paperId = schedule.paperId?._id || schedule.paperId;
      if (!paperId) {
        throw new Error('Paper not linked to this schedule');
      }

      const response = await generatePersonalizedExam(paperId);
      if (!response || !response.questions || response.questions.length === 0) {
        throw new Error('No questions available for this exam');
      }

      setSelectedExam({
        ...response,
        id: paperId,
        title: schedule.title || response.title,
        duration: schedule.duration || response.duration
      });
      setLoading(false);
      
    } catch (err) {
      console.error('Error taking exam:', err);
      setError(err.message || 'Failed to start exam');
      setLoading(false);
    }
  };

  const handleExamSubmit = async (submissionData) => {
    try {
      // Add XP for completing exam
      const baseXP = 500;
      const timeBonus = submissionData.timeSpent < selectedExam.duration * 30 ? 100 : 0;
      const totalXP = baseXP + timeBonus;
      awardXp(totalXP);
      setPlayerStats(prev => ({
        ...prev,
        completedTopics: [...prev.completedTopics, selectedExam.topic]
      }));

      // Show confetti animation
      setShowConfetti(true);
      setTimeout(() => setShowConfetti(false), 3000);

      // Add encouraging message
      addMessage({
        type: 'success',
        text: `Slay! You just earned ${totalXP} XP for completing the exam! Keep slaying bestie! 💅✨`
      });
    } catch (error) {
      console.error('Error submitting exam:', error);
      setExamError('Failed to submit exam. Please try again.');
    }
  };

  const handleCloseExam = () => {
    setSelectedExam(null);
    setExamError(null);
  };

  const isExamAvailable = (schedule) => {
    const now = new Date();
    const startTime = new Date(schedule.startTime);
    const endTime = new Date(schedule.endTime);
    return now >= startTime && now <= endTime;
  };

  

  return (
    <div className="dashboard">
      {showSplash && <Splash />}
      {/* MiniGame overlay removed */}
      
      {selectedExam && (
        <div className="exam-overlay">
          <ExamView
            exam={selectedExam}
            onSubmit={handleExamSubmit}
            onClose={handleCloseExam}
          />
        </div>
      )}

      {isMiniGameOpen && (
        <MiniGameOverlay
          gameId={activeGameId}
          topicTitle={selectedTopic?.title || 'Topic'}
          onClose={() => setIsMiniGameOpen(false)}
          onComplete={({ xp }) => {
            setIsMiniGameOpen(false);
            if (xp && xp > 0) {
              awardXp(xp);
              addMessage({ type: 'success', text: `Nice! You earned ${xp} XP from the mini-game.` });
            }
          }}
        />
      )}
      
      {/* Background particles removed for clarity */}
      
      <header className="dashboard-header">
        <div className="header-content">
          <div className="logo-section">
            <img src="/pesu.png" alt="PESU" className="logo" />
            <div className="brand-name">
              PESU<span>Prep</span>
            </div>
          </div>
          {user && (
            <div className="user-section">
              <div className="user-info">
                <img src={user.picture} alt={user.name} className="user-avatar" />
                <div className="user-details">
                  <h2>{user.name}</h2>
                  <p>{user.email}</p>
                </div>
                <div className="xp-counter">
                  ✨ {playerStats.xp} XP
                </div>
              </div>
              <button className="logout-button" onClick={handleLogout}>
                Logout
              </button>
            </div>
          )}
        </div>
      </header>

      <main className="dashboard-main">
        <div className="course-header">
          <div className="available-course-badge">Currently Available</div>
          <h1>Machine Learning</h1>
          <p className="course-id">UE22CS352A</p>
          
            <GamificationPanel
              playerStats={playerStats}
              setPlayerStats={setPlayerStats}
              userEmail={user?.email}
            />
        </div>

        {/* Tabs */}
        <div className="tabs" style={{ margin: '20px 0', display: 'flex', gap: '12px' }}>
          <button className={`tab-btn ${activeTab === 'overview' ? 'active' : ''}`} onClick={() => setActiveTab('overview')}>Overview</button>
          <button className={`tab-btn ${activeTab === 'exams' ? 'active' : ''}`} onClick={() => setActiveTab('exams')}>Exams</button>
          <button className={`tab-btn ${activeTab === 'resources' ? 'active' : ''}`} onClick={() => setActiveTab('resources')}>Resources</button>
          <button className={`tab-btn ${activeTab === 'leaderboard' ? 'active' : ''}`} onClick={() => setActiveTab('leaderboard')}>Leaderboard</button>
        </div>

        {/* Tab Panels */}
        {activeTab === 'overview' && (
          <div className="dashboard-content">
            <div className="welcome-banner">
              <h1>Welcome back</h1>
              <p>Track your progress and prepare for upcoming exams.</p>
            </div>
            {renderSkillTree()}
          </div>
        )}

        {activeTab === 'exams' && (
          <div className="upcoming-container">
            <div className="upcoming-box">
              <h3><i className="fas fa-calendar-alt"></i> Upcoming ISAs</h3>
              {schedules.filter(schedule => schedule.type === 'ISA').length === 0 ? (
                <div className="event-item isa">
                  <div className="event-date">No ISA scheduled yet</div>
                  <div className="event-title">Machine Learning(UE222CS352A)</div>
                </div>
              ) : (
                schedules
                  .filter(schedule => schedule.type === 'ISA')
                  .sort((a, b) => new Date(a.startTime) - new Date(b.startTime))
                  .map(schedule => (
                    <div key={schedule._id} className="event-item isa">
                      <div className="event-date">
                        {new Date(schedule.startTime).toLocaleString()}
                      </div>
                      <div className="event-title">
                        {schedule.title}
                      </div>
                      <div className="event-details">
                        Duration: {schedule.duration} minutes
                        {schedule.description && (
                          <p className="event-description">{schedule.description}</p>
                        )}
                        <button 
                          className="take-exam-btn"
                          onClick={() => handleTakeExam(schedule._id)}
                          disabled={!isExamAvailable(schedule)}
                        >
                          {isExamAvailable(schedule) ? 'Take Exam' : 'Not Available'}
                        </button>
                      </div>
                    </div>
                  ))
              )}
            </div>

            <div className="upcoming-box">
              <h3><i className="fas fa-pen"></i> Upcoming Quizzes</h3>
              {schedules.filter(schedule => schedule.type === 'QUIZ').length === 0 ? (
                <div className="event-item quiz">
                  <div className="event-date">No quiz scheduled yet</div>
                  <div className="event-title">Machine Learning(UE222CS352A)</div>
                </div>
              ) : (
                schedules
                  .filter(schedule => schedule.type === 'QUIZ')
                  .sort((a, b) => new Date(a.startTime) - new Date(b.startTime))
                  .map(schedule => (
                    <div key={schedule._id} className="event-item quiz">
                      <div className="event-date">
                        {new Date(schedule.startTime).toLocaleString()}
                      </div>
                      <div className="event-title">
                        {schedule.title}
                      </div>
                      <div className="event-details">
                        Duration: {schedule.duration} minutes
                        {schedule.description && (
                          <p className="event-description">{schedule.description}</p>
                        )}
                      </div>
                    </div>
                  ))
              )}
            </div>
          </div>
        )}

        {activeTab === 'resources' && (
          <div className="resources-section">
            <div className="resources-title">
              <h2>Learning Resources</h2>
              <p>Level up your ML game with these carefully curated resources! 🚀</p>
            </div>
            <div className="resources-grid">
              {resources.map((resource, index) => (
                <a 
                  href={resource.link} 
                  target="_blank" 
                  rel="noopener noreferrer" 
                  key={index}
                  className="resource-card"
                >
                  <span className={`resource-type ${resource.type}`}>
                    {resource.type.charAt(0).toUpperCase() + resource.type.slice(1)}
                  </span>
                  <h3>{resource.title}</h3>
                  <p>{resource.description}</p>
                  <div className="resource-meta">
                    <span>
                      <FaClock /> {resource.readTime}
                    </span>
                    <span>
                      <FaStar /> {resource.difficulty}
                    </span>
                  </div>
                </a>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'leaderboard' && (
          <Leaderboard />
        )}

        <div className="topics-section">
          <div className="topics-grid">
            {mlTopics.map(topic => (
              <div 
                key={topic.id}
                className="topic-card"
                onClick={() => handleTopicClick(topic)}
              >
                <div className={`topic-difficulty ${topic.difficulty.toLowerCase().includes('noob') ? 'noob' : 
                  topic.difficulty.toLowerCase().includes('sus') ? 'sus' : 'mid'}`}>
                  {topic.difficulty}
                </div>
                <h3>{topic.title}</h3>
                <p>{topic.description}</p>
                <div className="topic-rewards">
                  <span>✨ {topic.xpReward} XP</span>
                </div>
                {topic.prerequisites.length > 0 && (
                  <div className="topic-prerequisites">
                    Prerequisites: {topic.prerequisites.join(', ')}
                  </div>
                )}
                {topic.level > playerStats.level && (
                  <div className="topic-locked">
                    <FaLock /> Unlock at Level {topic.level}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </main>

        {/* Chat container removed for a cleaner student dashboard */}

        <div className="courses-grid">
          <div 
            className={`course-card ${selectedCourse?.id === course.id ? 'selected' : ''}`}
            onClick={() => setSelectedCourse(course)}
          >
            <div className="course-header">
              <h3>{course.name}</h3>
              <p className="course-id">{course.id}</p>
            </div>
            
            {selectedCourse?.id === course.id && (
              <div className="course-sections">
                <div className="section">
                  <h4>Explore</h4>
                  <ul>
                    <li className="resource">
                      Machine Learning Crash Course
                    </li>
                  </ul>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="upcoming-container">
          <div className="upcoming-box">
            <h3><i className="fas fa-calendar-alt"></i> Upcoming ISAs</h3>
            {schedules.filter(schedule => schedule.type === 'ISA').length === 0 ? (
              <div className="event-item isa">
                <div className="event-date">No ISA scheduled yet</div>
                <div className="event-title">Machine Learning(UE222CS352A)</div>
              </div>
            ) : (
              schedules
                .filter(schedule => schedule.type === 'ISA')
                .sort((a, b) => new Date(a.startTime) - new Date(b.startTime))
                .map(schedule => (
                  <div key={schedule._id} className="event-item isa">
                    <div className="event-date">
                      {new Date(schedule.startTime).toLocaleString()}
                    </div>
                    <div className="event-title">
                      {schedule.title}
                    </div>
                    <div className="event-details">
                      Duration: {schedule.duration} minutes
                      {schedule.description && (
                        <p className="event-description">{schedule.description}</p>
                      )}
                      <button 
                        className="take-exam-btn"
                        onClick={() => handleTakeExam(schedule._id)}
                        disabled={!isExamAvailable(schedule)}
                      >
                        Take Exam
                      </button>
                    </div>
                  </div>
                ))
            )}
          </div>

          <div className="upcoming-box">
            <h3><i className="fas fa-pen"></i> Upcoming Quizzes</h3>
            {schedules.filter(schedule => schedule.type === 'QUIZ').length === 0 ? (
              <div className="event-item quiz">
                <div className="event-date">No quiz scheduled yet</div>
                <div className="event-title">Machine Learning(UE222CS352A)</div>
              </div>
            ) : (
              schedules
                .filter(schedule => schedule.type === 'QUIZ')
                .sort((a, b) => new Date(a.startTime) - new Date(b.startTime))
                .map(schedule => (
                  <div key={schedule._id} className="event-item quiz">
                    <div className="event-date">
                      {new Date(schedule.startTime).toLocaleString()}
                    </div>
                    <div className="event-title">
                      {schedule.title}
                    </div>
                    <div className="event-details">
                      Duration: {schedule.duration} minutes
                      {schedule.description && (
                        <p className="event-description">{schedule.description}</p>
                      )}
                    </div>
                  </div>
                ))
            )}
          </div>
        </div>

        <div className="dashboard-content">
          <div className="welcome-banner">
          <h1>Ready to slay your ML journey? </h1>
            <p>Choose a topic below and let's make this learning journey hit different!</p>
          </div>
          
          {/* Removed detailed topic content from main view for simplicity */}
        </div>

      {showPrerequisiteAlert && (
        <div className="prerequisite-alert">
          <div className="alert-content">
            <div className="alert-title">Hold up, bestie! 💅</div>
            <div className="alert-message">{prerequisiteMessage}</div>
          </div>
          <button 
            className="alert-button"
            onClick={() => setShowPrerequisiteAlert(false)}
          >
            Got it!
          </button>
        </div>
      )}

      {showConfetti && (
        <Player
          autoplay
          keepLastFrame
          src={confettiAnimation}
          style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
        />
      )}

      {/* Resources are shown under the Resources tab */}

      <footer className="dashboard-footer">
        <div className="footer-content">
          <div className="footer-brand">
            PESU<span>Prep</span>
          </div>
          <div className="footer-location">Bengaluru, India</div>
          <div className="footer-year">© {new Date().getFullYear()}</div>
        </div>
      </footer>

      {examError && (
        <div className="error-popup">
          <p>{examError}</p>
          <button onClick={() => setExamError(null)}>Close</button>
        </div>
      )}
    </div>
  );
}

export default Dashboard; 