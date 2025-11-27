import { api } from './api';

// Load an exam directly from a generated paper
export const generatePersonalizedExam = async (paperId) => {
  try {
    if (!paperId) {
      throw new Error('Paper ID is required');
    }

    const response = await api.get(`/papers/${paperId}`);

    const paper = response.data?.paper || response.data;
    if (!paper || !paper.questions || paper.questions.length === 0) {
      throw new Error('No questions available for this paper');
    }

    return {
      questions: paper.questions,
      duration: paper.duration,
      totalMarks: paper.totalMarks,
      paperId: paper._id,
      title: paper.title,
      courseId: paper.courseId
    };
  } catch (error) {
    console.error('Error loading exam from paper:', error);
    throw new Error('Failed to load exam: ' + (error.response?.data?.message || error.message));
  }
};

// Submit exam answers
export const submitExamAnswers = async (submissionData) => {
  try {
    // Validate submission data
    if (!submissionData.answers || !Array.isArray(submissionData.answers)) {
      throw new Error('Invalid answers format');
    }

    // Validate each answer has required fields
    const isValidAnswer = answer => 
      answer.questionId && 
      answer.answer !== undefined &&
      answer.type &&
      typeof answer.timeSpent === 'number';

    if (!submissionData.answers.every(isValidAnswer)) {
      throw new Error('Invalid answer format - missing required fields');
    }

    // Get paper ID from exam metadata
    const paperId = submissionData.paperId;
    if (!paperId) {
      throw new Error('Paper ID is required');
    }

    // Pre-process descriptive answers before sending to backend
    const processedAnswers = submissionData.answers.map(answer => {
      if (answer.type === 'descriptive') {
        // Get the question from the exam
        const question = window.currentExam?.questions.find(q => q._id === answer.questionId);
        if (question?.keywords) {
          // Calculate score based on keyword matching
          const answerText = answer.answer.toLowerCase();
          const keywordMatches = question.keywords.filter(keyword => 
            answerText.includes(keyword.toLowerCase())
          );
          const keywordScore = keywordMatches.length / question.keywords.length;
          
          // Consider answer length and coherence
          const minLength = 50; // Minimum expected length
          const lengthScore = Math.min(answer.answer.length / minLength, 1);
          
          // Calculate final score (70% keywords, 30% length)
          const score = (keywordScore * 0.7 + lengthScore * 0.3) * (question.marks || 1);
          
          return {
            ...answer,
            autoScore: Math.round(score * 100) / 100,
            keywordMatches
          };
        }
      }
      return answer;
    });

    // Backend expects an object keyed by questionId for answers
    const answersById = processedAnswers.reduce((acc, ans) => {
      acc[ans.questionId] = {
        answer: ans.answer,
        type: ans.type,
        timeSpent: ans.timeSpent || 0,
        autoScore: ans.autoScore,
        keywordMatches: ans.keywordMatches
      };
      return acc;
    }, {});

    const response = await api.post(`/diagnostic/submit-adaptive/${paperId}`, {
      answers: answersById,
      timeTaken: Math.round((Date.now() - submissionData.startTime) / 1000)
    });
    
    if (!response.data) {
      throw new Error('No response from server');
    }
    
    return response.data;
  } catch (error) {
    console.error('Error submitting exam:', error);
    if (error.response?.data?.message) {
      throw new Error(error.response.data.message);
    } else if (error.message) {
      throw new Error(error.message);
    } else {
      throw new Error('Failed to submit exam. Please try again.');
    }
  }
};

// Get student's exam history
export const getExamHistory = async () => {
  try {
    const response = await api.get('/student/exam-history');
    return response.data;
  } catch (error) {
    console.error('Error fetching exam history:', error);
    throw new Error('Failed to fetch exam history: ' + (error.response?.data?.message || error.message));
  }
};

// Get student's current learning zone and progress
export const getStudentProgress = async () => {
  try {
    const response = await api.get('/student/progress');
    return response.data;
  } catch (error) {
    console.error('Error fetching student progress:', error);
    throw new Error('Failed to fetch progress: ' + (error.response?.data?.message || error.message));
  }
}; 