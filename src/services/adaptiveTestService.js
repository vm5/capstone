import { api } from './api';

// Get diagnostic test status
export const getDiagnosticStatus = async () => {
  try {
    const response = await api.get('/diagnostic/status');
    return response.data;
  } catch (error) {
    console.error('Error fetching diagnostic status:', error);
    throw error;
  }
};

// Get diagnostic test questions
export const getDiagnosticTest = async () => {
  try {
    const response = await api.get('/diagnostic/questions');
    return response.data;
  } catch (error) {
    console.error('Error fetching diagnostic test:', error);
    throw error;
  }
};

// Submit diagnostic test and get learning zone
export const submitDiagnosticTest = async (submissionData) => {
  try {
    // Format the submission data according to the server's expected structure
    const formattedData = {
      answers: submissionData.answers.map(answer => ({
        questionId: answer.questionId,
        answer: answer.answer || '',
        timeSpent: answer.timeSpent || 0,
        confidenceScore: answer.confidenceScore || 3,
        difficulty: answer.difficulty || 'medium'
      })),
      metadata: {
        totalTimeSpent: submissionData.behaviorAnalysis.totalTestTime,
        behaviorAnalysis: {
          averageTimePerQuestion: submissionData.behaviorAnalysis.averageTimePerQuestion,
          averageConfidence: submissionData.behaviorAnalysis.averageConfidence,
          answerChangePattern: submissionData.behaviorAnalysis.answerChangePattern
        }
      }
    };

    const response = await api.post('/diagnostic/submit', formattedData);
    return response.data;
  } catch (error) {
    console.error('Error submitting diagnostic test:', error);
    throw error;
  }
};

// Get personalized questions based on learning zone
export const getPersonalizedQuestions = async (paperId, learningZone, baselineScore) => {
  try {
    const response = await api.get('/diagnostic/personalized-questions', {
      params: {
        paperId,
        learningZone,
        baselineScore
      }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching personalized questions:', error);
    throw error;
  }
};

// Calculate weighted score
export const calculateWeightedScore = (scores) => {
  const {
    easyScore, // out of 16
    mediumScore, // out of 8
    hardScore, // out of 16
  } = scores;

  // Weighted scoring formula
  const weightedScore = Math.round(
    (easyScore / 16) * 10 + // Easy zone weight: 10 points
    (mediumScore / 8) * 15 + // Medium zone weight: 15 points
    (hardScore / 16) * 15 // Hard zone weight: 15 points
  );

  return {
    weightedScore,
    breakdown: {
      easy: (easyScore / 16) * 10,
      medium: (mediumScore / 8) * 15,
      hard: (hardScore / 16) * 15
    }
  };
};

// Submit exam with adaptive scoring
export const submitAdaptiveExam = async (paperId, submissionData) => {
  try {
    const response = await api.post(`/diagnostic/submit-adaptive/${paperId}`, submissionData);
    return response.data;
  } catch (error) {
    console.error('Error submitting adaptive exam:', error);
    throw error;
  }
};

// Get student's learning progress
export const getLearningProgress = async () => {
  try {
    const response = await api.get('/diagnostic/learning-progress');
    return response.data;
  } catch (error) {
    console.error('Error fetching learning progress:', error);
    throw error;
  }
};

export const getRecommendations = async () => {
  try {
    const response = await api.get('/diagnostic/recommendations');
    return response.data;
  } catch (error) {
    console.error('Error fetching recommendations:', error);
    throw error;
  }
}; 