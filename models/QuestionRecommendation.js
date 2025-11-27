const mongoose = require('mongoose');

const questionRecommendationSchema = new mongoose.Schema({
  questionId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Question',
    required: true
  },
  similarQuestions: [{
    questionId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Question'
    },
    similarityScore: {
      type: Number,
      min: 0,
      max: 1
    },
    features: {
      topicSimilarity: Number,
      difficultyDifference: Number,
      conceptOverlap: Number,
      performanceCorrelation: Number
    }
  }],
  metadata: {
    lastUpdated: {
      type: Date,
      default: Date.now
    },
    totalUsage: {
      type: Number,
      default: 0
    },
    averagePerformance: {
      type: Number,
      min: 0,
      max: 1
    }
  }
});

// Method to find similar questions within a difficulty range
questionRecommendationSchema.methods.getSimilarQuestions = function(difficultyRange, count = 1) {
  return this.similarQuestions
    .filter(q => {
      const question = Question.findById(q.questionId);
      return question && 
             question.difficulty >= difficultyRange.min && 
             question.difficulty <= difficultyRange.max;
    })
    .sort((a, b) => b.similarityScore - a.similarityScore)
    .slice(0, count);
};

// Method to update similarity scores based on student performance
questionRecommendationSchema.methods.updateSimilarityScores = async function(studentResponses) {
  for (const similar of this.similarQuestions) {
    // Calculate performance correlation
    const originalPerformance = studentResponses
      .filter(r => r.questionId.equals(this.questionId))
      .map(r => r.isCorrect ? 1 : 0);
    
    const similarPerformance = studentResponses
      .filter(r => r.questionId.equals(similar.questionId))
      .map(r => r.isCorrect ? 1 : 0);

    if (originalPerformance.length > 0 && similarPerformance.length > 0) {
      const correlation = calculateCorrelation(originalPerformance, similarPerformance);
      similar.features.performanceCorrelation = correlation;
      
      // Update overall similarity score with weighted features
      similar.similarityScore = (
        similar.features.topicSimilarity * 0.3 +
        (1 - similar.features.difficultyDifference) * 0.2 +
        similar.features.conceptOverlap * 0.3 +
        similar.features.performanceCorrelation * 0.2
      );
    }
  }
  
  this.metadata.lastUpdated = new Date();
  await this.save();
};

function calculateCorrelation(array1, array2) {
  // Pearson correlation implementation
  const mean1 = array1.reduce((a, b) => a + b) / array1.length;
  const mean2 = array2.reduce((a, b) => a + b) / array2.length;
  
  const variance1 = array1.reduce((a, b) => a + Math.pow(b - mean1, 2), 0);
  const variance2 = array2.reduce((a, b) => a + Math.pow(b - mean2, 2), 0);
  
  const covariance = array1.reduce((a, b, i) => a + (b - mean1) * (array2[i] - mean2), 0);
  
  return covariance / Math.sqrt(variance1 * variance2);
}

const QuestionRecommendation = mongoose.model('QuestionRecommendation', questionRecommendationSchema);

module.exports = QuestionRecommendation; 