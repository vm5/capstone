const mongoose = require('mongoose');

const questionSchema = new mongoose.Schema({
  courseId: {
    type: String,
    required: true
  },
  topic: {
    type: String,
    required: true
  },
  questionText: {
    type: String,
    required: true
  },
  options: [{
    text: {
      type: String,
      required: true
    },
    isCorrect: {
      type: Boolean,
      required: true
    }
  }],
  correctAnswer: {
    type: mongoose.Schema.Types.Mixed,  // Can be number (for MCQ) or string (for descriptive)
    required: true
  },
  difficulty: {
    type: Number,  // 1-10 scale
    required: true,
    min: 1,
    max: 10
  },
  type: {
    type: String,
    enum: ['mcq', 'descriptive', 'trueFalse', 'fillInBlanks'],
    required: true
  },
  status: {
    type: String,
    enum: ['pending', 'approved', 'rejected'],
    default: 'pending'
  },
  feedback: String,  // Teacher's feedback if rejected
  generatedAt: {
    type: Date,
    default: Date.now
  },
  approvedBy: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User'
  },
  usedInSets: [{
    type: String,
    ref: 'Paper'
  }],
  metadata: {
    topicDifficulty: {
      type: Number,
      min: 1,
      max: 10
    },
    conceptsCovered: [String],
    timeToSolve: Number,  // estimated time in minutes
    similarityFeatures: {
      topicVector: [Number],  // For topic similarity calculation
      conceptVector: [Number]  // For concept overlap calculation
    },
    performance: {
      totalAttempts: {
        type: Number,
        default: 0
      },
      correctAttempts: {
        type: Number,
        default: 0
      },
      averageTime: {
        type: Number,
        default: 0
      }
    }
  }
});

// Method to update performance metrics
questionSchema.methods.updatePerformance = function(isCorrect, timeSpent) {
  this.metadata.performance.totalAttempts++;
  if (isCorrect) {
    this.metadata.performance.correctAttempts++;
  }
  
  // Update average time
  const currentTotal = this.metadata.performance.averageTime * (this.metadata.performance.totalAttempts - 1);
  this.metadata.performance.averageTime = (currentTotal + timeSpent) / this.metadata.performance.totalAttempts;
};

// Method to calculate similarity with another question
questionSchema.methods.calculateSimilarity = function(otherQuestion) {
  const topicSimilarity = this.calculateVectorSimilarity(
    this.metadata.similarityFeatures.topicVector,
    otherQuestion.metadata.similarityFeatures.topicVector
  );

  const conceptSimilarity = this.calculateVectorSimilarity(
    this.metadata.similarityFeatures.conceptVector,
    otherQuestion.metadata.similarityFeatures.conceptVector
  );

  const difficultyDifference = Math.abs(this.difficulty - otherQuestion.difficulty) / 9; // Normalize to 0-1

  const performanceCorrelation = this.calculatePerformanceCorrelation(otherQuestion);

  return {
    topicSimilarity,
    conceptSimilarity,
    difficultyDifference,
    performanceCorrelation,
    overall: (
      topicSimilarity * 0.3 +
      (1 - difficultyDifference) * 0.2 +
      conceptSimilarity * 0.3 +
      performanceCorrelation * 0.2
    )
  };
};

// Helper method to calculate cosine similarity between vectors
questionSchema.methods.calculateVectorSimilarity = function(vec1, vec2) {
  if (!vec1 || !vec2 || vec1.length !== vec2.length) return 0;

  const dotProduct = vec1.reduce((sum, val, i) => sum + val * vec2[i], 0);
  const mag1 = Math.sqrt(vec1.reduce((sum, val) => sum + val * val, 0));
  const mag2 = Math.sqrt(vec2.reduce((sum, val) => sum + val * val, 0));

  return dotProduct / (mag1 * mag2) || 0;
};

// Helper method to calculate performance correlation
questionSchema.methods.calculatePerformanceCorrelation = function(otherQuestion) {
  const thisSuccess = this.metadata.performance.correctAttempts / Math.max(1, this.metadata.performance.totalAttempts);
  const otherSuccess = otherQuestion.metadata.performance.correctAttempts / Math.max(1, otherQuestion.metadata.performance.totalAttempts);
  
  // Simple correlation based on success rates
  return 1 - Math.abs(thisSuccess - otherSuccess);
};

// Method to check answers based on question type
questionSchema.methods.checkAnswer = function(submittedAnswer) {
  switch (this.type) {
    case 'mcq':
    case 'trueFalse':
      // For MCQ and True/False, check both ways of storing correct answers
      // First try using isCorrect flag in options
      if (this.options[submittedAnswer]?.isCorrect) {
        return true;
      }
      // Then try using correctAnswer field
      if (typeof this.correctAnswer === 'number') {
        return submittedAnswer === this.correctAnswer;
      }
      // If neither method works, assume incorrect
      return false;
      
    case 'fillInBlanks':
      // For fill in blanks, do case-insensitive comparison
      return typeof submittedAnswer === 'string' && 
             submittedAnswer.toLowerCase().trim() === this.correctAnswer.toLowerCase().trim();
      
    case 'descriptive':
      // For descriptive questions, we'll need manual grading
      // For now, store the answer and return null to indicate manual grading needed
      return null;
      
    default:
      throw new Error(`Unsupported question type: ${this.type}`);
  }
};

const Question = mongoose.model('Question', questionSchema);

module.exports = Question; 