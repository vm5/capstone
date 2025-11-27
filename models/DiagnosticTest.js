const mongoose = require('mongoose');

const diagnosticTestSchema = new mongoose.Schema({
  userId: {
    type: String,  // Changed from ObjectId to String to handle Google IDs
    required: true
  },
  answers: [{
    questionId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Question',
      required: true
    },
    answer: {
      type: String,
      required: true
    },
    isCorrect: {
      type: Boolean,
      default: false
    },
    timeSpent: {
      type: Number,
      required: true
    },
    confidenceScore: {
      type: Number,
      min: 1,
      max: 5,
      required: true
    },
    difficulty: {
      type: Number,
      min: 1,
      max: 10,
      required: true
    }
  }],
  learningZone: {
    type: String,
    enum: ['EASY', 'MEDIUM', 'HARD'],
    required: true
  },
  baselineScore: {
    type: Number,
    required: true
  },
  metadata: {
    totalTimeSpent: Number,
    averageConfidence: Number,
    accuracyByDifficulty: {
      easy: Number,
      medium: Number,
      hard: Number
    },
    timeByDifficulty: {
      easy: Number,
      medium: Number,
      hard: Number
    },
    recommendedTopics: [String],
    weakAreas: [String],
    strongAreas: [String],
    completedAt: {
      type: Date,
      default: Date.now
    }
  },
  isValid: {
    type: Boolean,
    default: true
  },
  validUntil: {
    type: Date,
    required: true
  }
}, {
  timestamps: true
});

// Evaluate a single answer
diagnosticTestSchema.methods.evaluateAnswer = function(answer, correctAnswer, questionType) {
  if (questionType === 'mcq') {
    return answer === correctAnswer;
  } else {
    // For descriptive answers, use similarity check
    const similarity = this.calculateTextSimilarity(answer, correctAnswer);
    return similarity >= 0.7; // 70% similarity threshold
  }
};

// Calculate text similarity for descriptive answers
diagnosticTestSchema.methods.calculateTextSimilarity = function(answer, correctAnswer) {
  // Convert both texts to lowercase and remove punctuation
  const cleanText = (text) => text.toLowerCase().replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g, "");
  const text1 = cleanText(answer).split(' ');
  const text2 = cleanText(correctAnswer).split(' ');

  // Calculate Jaccard similarity
  const intersection = new Set(text1.filter(word => text2.includes(word)));
  const union = new Set([...text1, ...text2]);
  return intersection.size / union.size;
};

// Calculate learning zone based on comprehensive evaluation
diagnosticTestSchema.methods.calculateLearningZone = function() {
  const scores = {
    easy: this.calculateScoreByDifficulty('easy'),
    medium: this.calculateScoreByDifficulty('medium'),
    hard: this.calculateScoreByDifficulty('hard')
  };

  const avgConfidence = this.calculateAverageConfidence();
  const avgTime = this.calculateAverageTimeSpent();

  // Weight factors
  const weights = {
    performance: 0.5,
    confidence: 0.3,
    time: 0.2
  };

  // Calculate weighted scores
  const performanceScore = (
    (scores.easy * 0.3) +
    (scores.medium * 0.4) +
    (scores.hard * 0.3)
  ) * weights.performance;

  const confidenceScore = (avgConfidence / 5) * weights.confidence;
  
  // Normalize time score (lower is better)
  const expectedTime = 120; // 2 minutes per question
  const timeScore = Math.max(0, 1 - (avgTime / expectedTime)) * weights.time;

  const totalScore = performanceScore + confidenceScore + timeScore;

  // Determine zone based on total score
  if (totalScore >= 0.8) return 'HARD';
  if (totalScore >= 0.5) return 'MEDIUM';
  return 'EASY';
};

// Calculate score for a specific difficulty level
diagnosticTestSchema.methods.calculateScoreByDifficulty = function(difficulty) {
  const difficultyAnswers = this.answers.filter(a => a.difficulty === difficulty);
  if (difficultyAnswers.length === 0) return 0;

  const correctAnswers = difficultyAnswers.filter(a => a.isCorrect).length;
  return correctAnswers / difficultyAnswers.length;
};

// Calculate average confidence score
diagnosticTestSchema.methods.calculateAverageConfidence = function() {
  return this.answers.reduce((sum, a) => sum + a.confidenceScore, 0) / this.answers.length;
};

// Calculate average time spent per question
diagnosticTestSchema.methods.calculateAverageTimeSpent = function() {
  return this.answers.reduce((sum, a) => sum + a.timeSpent, 0) / this.answers.length;
};

// Evaluate the entire test and update metadata
diagnosticTestSchema.methods.evaluateTest = async function() {
  // Calculate scores by difficulty
  const accuracyByDifficulty = {
    easy: this.calculateScoreByDifficulty('easy'),
    medium: this.calculateScoreByDifficulty('medium'),
    hard: this.calculateScoreByDifficulty('hard')
  };

  // Calculate time spent by difficulty
  const timeByDifficulty = {
    easy: 0,
    medium: 0,
    hard: 0
  };

  this.answers.forEach(answer => {
    timeByDifficulty[answer.difficulty] += answer.timeSpent;
  });

  // Identify weak and strong areas
  const weakAreas = [];
  const strongAreas = [];
  
  Object.entries(accuracyByDifficulty).forEach(([difficulty, score]) => {
    if (score < 0.6) {
      weakAreas.push(difficulty);
    } else if (score >= 0.8) {
      strongAreas.push(difficulty);
    }
  });

  // Generate recommended topics based on performance
  const recommendedTopics = this.generateRecommendedTopics(weakAreas);

  // Update metadata
  this.metadata = {
    totalTimeSpent: this.answers.reduce((sum, a) => sum + a.timeSpent, 0),
    averageConfidence: this.calculateAverageConfidence(),
    accuracyByDifficulty,
    timeByDifficulty,
    weakAreas,
    strongAreas,
    recommendedTopics,
    completedAt: new Date()
  };

  // Calculate overall baseline score (weighted average)
  this.baselineScore = Math.round(
    (accuracyByDifficulty.easy * 0.3 +
     accuracyByDifficulty.medium * 0.4 +
     accuracyByDifficulty.hard * 0.3) * 100
  );

  // Set learning zone
  this.learningZone = this.calculateLearningZone();

  // Set validity period (30 days from completion)
  this.validUntil = new Date(Date.now() + 30 * 24 * 60 * 60 * 1000);

  await this.save();
  return {
    learningZone: this.learningZone,
    baselineScore: this.baselineScore,
    metadata: this.metadata
  };
};

// Generate recommended topics based on weak areas
diagnosticTestSchema.methods.generateRecommendedTopics = function(weakAreas) {
  const topicsByDifficulty = {
    easy: [
      'Basic Concepts Review',
      'Fundamental Principles',
      'Practice Problems'
    ],
    medium: [
      'Advanced Concepts',
      'Problem-Solving Techniques',
      'Application Examples'
    ],
    hard: [
      'Complex Problem Solving',
      'Advanced Applications',
      'Integration of Concepts'
    ]
  };

  return weakAreas.reduce((topics, area) => {
    return [...topics, ...topicsByDifficulty[area]];
  }, []);
};

// Check if diagnostic test is still valid
diagnosticTestSchema.methods.isStillValid = function() {
  return this.isValid && new Date() < this.validUntil;
};

const DiagnosticTest = mongoose.model('DiagnosticTest', diagnosticTestSchema);

module.exports = DiagnosticTest; 