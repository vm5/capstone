const mongoose = require('mongoose');

const studentProgressSchema = new mongoose.Schema({
  studentId: {
    type: String,
    required: true
  },
  courseId: {
    type: String,
    required: true
  },
  diagnosticScore: {
    score: {
      type: Number,
      min: 1,
      max: 10,
      required: true
    },
    completedAt: {
      type: Date,
      default: Date.now
    }
  },
  learningZone: {
    type: String,
    enum: ['EASY', 'MEDIUM', 'HARD'],
    required: true
  },
  examHistory: [{
    examId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Schedule'
    },
    scores: {
      easy: {
        scored: Number,
        total: Number,
        weightedScore: Number
      },
      medium: {
        scored: Number,
        total: Number,
        weightedScore: Number
      },
      hard: {
        scored: Number,
        total: Number,
        weightedScore: Number
      }
    },
    totalScore: Number,
    weightedTotalScore: Number,
    completedAt: {
      type: Date,
      default: Date.now
    }
  }],
  progressHistory: [{
    timestamp: {
      type: Date,
      default: Date.now
    },
    capabilityScore: {
      type: Number,
      min: 1,
      max: 10
    },
    learningZone: {
      type: String,
      enum: ['EASY', 'MEDIUM', 'HARD']
    }
  }],
  quizzes: [{
    quizId: { type: mongoose.Schema.Types.ObjectId, ref: 'QuestionSet' },
    score: Number,
    timeTaken: Number,
    completedAt: Date,
    answers: [{
      questionId: { type: mongoose.Schema.Types.ObjectId, ref: 'Question' },
      selectedOption: Number,
      isCorrect: Boolean,
      timeSpent: Number  // time spent on this question
    }]
  }],
  isas: [{
    isaId: { type: mongoose.Schema.Types.ObjectId, ref: 'QuestionSet' },
    score: Number,
    setNumber: Number,
    completedAt: Date
  }],
  overallProgress: {
    topicsCompleted: [String],
    averageQuizScore: Number,
    averageIsaScore: Number,
    totalTimeSpent: Number,
    strengthTopics: [String],
    weaknessTopics: [String],
    lastUpdated: Date
  }
}, {
  timestamps: true
});

// Calculate weighted score based on zone performance
studentProgressSchema.methods.calculateWeightedScore = function(scores) {
  const weights = {
    easy: 10,    // Easy questions weighted to 10 points
    medium: 15,  // Medium questions weighted to 15 points
    hard: 15     // Hard questions weighted to 15 points
  };

  const weightedScores = {
    easy: (scores.easy.scored / scores.easy.total) * weights.easy,
    medium: (scores.medium.scored / scores.medium.total) * weights.medium,
    hard: (scores.hard.scored / scores.hard.total) * weights.hard
  };

  return {
    scores: {
      easy: {
        ...scores.easy,
        weightedScore: weightedScores.easy
      },
      medium: {
        ...scores.medium,
        weightedScore: weightedScores.medium
      },
      hard: {
        ...scores.hard,
        weightedScore: weightedScores.hard
      }
    },
    totalScore: scores.easy.scored + scores.medium.scored + scores.hard.scored,
    weightedTotalScore: Math.round(weightedScores.easy + weightedScores.medium + weightedScores.hard)
  };
};

// Update student's learning zone based on performance
studentProgressSchema.methods.updateLearningZone = function(weightedScore) {
  if (weightedScore >= 30) { // 75% of max weighted score (40)
    this.learningZone = 'HARD';
  } else if (weightedScore >= 20) { // 50% of max weighted score
    this.learningZone = 'MEDIUM';
  } else {
    this.learningZone = 'EASY';
  }
  
  // Add to progress history
  this.progressHistory.push({
    capabilityScore: Math.round((weightedScore / 40) * 10), // Convert to 1-10 scale
    learningZone: this.learningZone
  });
};

const StudentProgress = mongoose.model('StudentProgress', studentProgressSchema);

module.exports = StudentProgress; 