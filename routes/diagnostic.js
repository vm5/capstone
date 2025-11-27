const express = require('express');
const router = express.Router();
const DiagnosticTest = require('../models/DiagnosticTest');
const StudentProgress = require('../models/StudentProgress');
const Paper = require('../models/Paper');
const Question = require('../models/Question');
const auth = require('../middleware/auth');
const mongoose = require('mongoose');
const path = require('path');
const adaptiveTestService = require('../services/adaptiveTestService');
const axios = require('axios');

// Configure ML service URL
const ML_SERVICE_URL = 'http://localhost:8000';  // Updated port to match ML service

// Helper function to check if a string is a valid MongoDB ObjectId
const isValidObjectId = (id) => {
  try {
    return mongoose.Types.ObjectId.isValid(id);
  } catch (error) {
    return false;
  }
};

// Helper function to clean paper ID
const cleanPaperId = (paperId) => {
  // Remove .pdf extension if present
  return paperId.replace(/\.pdf$/, '');
};

// Get diagnostic questions
router.get('/questions', auth, async (req, res) => {
  try {
    // Get one question of each type
    const mcqQuestion = await Question.findOne({ type: 'mcq', status: 'approved' });
    const descriptiveQuestion = await Question.findOne({ type: 'descriptive', status: 'approved' });
    const trueFalseQuestion = await Question.findOne({ type: 'trueFalse', status: 'approved' });
    const fillInBlanksQuestion = await Question.findOne({ type: 'fillInBlanks', status: 'approved' });

    const questions = [mcqQuestion, descriptiveQuestion, trueFalseQuestion, fillInBlanksQuestion].filter(q => q);

    if (!questions || questions.length === 0) {
      return res.status(404).json({ message: 'No diagnostic questions found' });
    }

    if (questions.length < 4) {
      return res.status(404).json({ message: 'Not all question types are available' });
    }

    // Format questions for the frontend - remove isCorrect flag for security
    const formattedQuestions = questions.map(q => ({
      _id: q._id,
      questionText: q.questionText || q.text, // Support both formats
      type: q.type,
      options: q.options ? q.options.map(opt => ({
        text: typeof opt === 'string' ? opt : opt.text
      })) : [],
      difficulty: q.difficulty
    }));

    res.json(formattedQuestions);
  } catch (error) {
    console.error('Error fetching diagnostic questions:', error);
    res.status(500).json({ message: error.message });
  }
});

// Create a diagnostic test
router.post('/create', auth, async (req, res) => {
  try {
    const {
      courseId,
      title,
      description,
      duration,
      questions,
      passingScore,
      difficultyDistribution
    } = req.body;

    const diagnosticTest = new DiagnosticTest({
      courseId,
      title,
      description,
      duration,
      questions,
      passingScore,
      difficultyDistribution
    });

    await diagnosticTest.save();
    res.status(201).json(diagnosticTest);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// Get diagnostic test by ID
router.get('/test/:id', auth, async (req, res) => {
  try {
    const diagnosticTest = await DiagnosticTest.findById(req.params.id);
    if (!diagnosticTest) {
      return res.status(404).json({ message: 'Diagnostic test not found' });
    }
    res.json(diagnosticTest);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// Get diagnostic test for a student
router.get('/course/:courseId', auth, async (req, res) => {
  try {
    const test = await DiagnosticTest.findOne({ courseId: req.params.courseId });
    if (!test) {
      return res.status(404).json({ message: 'Diagnostic test not found' });
    }
    res.json(test);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// Generate personalized test from template
router.get('/generate/:paperId', auth, async (req, res) => {
  try {
    const { paperId } = req.params;
    let paperTemplate;
    const cleanedPaperId = cleanPaperId(paperId);

    // Try to find paper by MongoDB ID first
    if (isValidObjectId(cleanedPaperId)) {
      paperTemplate = await Paper.findById(cleanedPaperId).populate('questions');
    }

    // If not found, try to find by file ID
    if (!paperTemplate) {
      paperTemplate = await Paper.findOne({ _id: cleanedPaperId }).populate('questions');
    }

    if (!paperTemplate) {
      console.error('Paper not found:', paperId);
      return res.status(404).json({ message: 'Paper template not found' });
    }

    // If the paper has questions, just use them directly
    if (paperTemplate.questions && paperTemplate.questions.length > 0) {
      return res.json({
        questions: paperTemplate.questions,
        totalMarks: paperTemplate.totalMarks,
        duration: paperTemplate.duration,
        paperId: paperTemplate._id
      });
    }

    // If no questions in paper, try to generate them (fallback to old logic)
    const allQuestions = await Question.find({
      courseId: paperTemplate.courseId,
      status: 'approved'
    });

    if (!allQuestions || allQuestions.length === 0) {
      console.error('No questions found for course:', paperTemplate.courseId);
      return res.status(404).json({ message: 'No approved questions found for this course' });
    }

    // Group questions by difficulty
    const easyQuestions = allQuestions.filter(q => q.difficulty <= 3);
    const mediumQuestions = allQuestions.filter(q => q.difficulty > 3 && q.difficulty <= 6);
    const hardQuestions = allQuestions.filter(q => q.difficulty > 6);

    // Generate questions based on what's available
    const personalizedQuestions = [];
    
    // Add all available questions, up to 24 total
    const addQuestions = (questions, maxCount) => {
      const available = questions.filter(q => 
        !personalizedQuestions.some(pq => pq._id.toString() === q._id.toString())
      );
      const toAdd = available.slice(0, maxCount);
      personalizedQuestions.push(...toAdd);
      return toAdd.length;
    };

    // Try to maintain a rough distribution but be flexible
    const easyCount = addQuestions(easyQuestions, 12);
    const mediumCount = addQuestions(mediumQuestions, 8);
    const hardCount = addQuestions(hardQuestions, 4);

    // If we don't have enough questions, add more from any difficulty
    const totalNeeded = Math.max(10, paperTemplate.questions?.length || 10);
    if (personalizedQuestions.length < totalNeeded) {
      const remaining = allQuestions.filter(q => 
        !personalizedQuestions.some(pq => pq._id.toString() === q._id.toString())
      );
      const additionalCount = Math.min(remaining.length, totalNeeded - personalizedQuestions.length);
      personalizedQuestions.push(...remaining.slice(0, additionalCount));
    }

    if (personalizedQuestions.length === 0) {
      return res.status(400).json({ message: 'Not enough questions available to generate exam' });
    }

    return res.json({
      questions: personalizedQuestions,
      totalMarks: paperTemplate.totalMarks || personalizedQuestions.length * 2,
      duration: paperTemplate.duration || 30,
      paperId: paperTemplate._id
    });

  } catch (error) {
    console.error('Error generating personalized test:', error);
    res.status(500).json({ message: error.message });
  }
});

// Submit diagnostic test
router.post('/submit', auth, async (req, res) => {
  try {
    const { answers } = req.body;

    // Validate answers
    if (!answers || !Array.isArray(answers)) {
      return res.status(400).json({ message: 'Invalid answers format' });
    }

    // Get questions for validation
    const questionIds = answers.map(a => a.questionId);
    const questions = await Question.find({ _id: { $in: questionIds } });

    if (questions.length !== answers.length) {
      return res.status(400).json({ message: 'Some questions not found' });
    }

    // Get the course ID from the first question (assuming all questions are from the same course)
    const courseId = questions[0].courseId;

    // Prepare answers for ML evaluation
    const answersForEvaluation = answers.map(answer => {
      const question = questions.find(q => q._id.toString() === answer.questionId);
      return {
        questionId: question._id.toString(),
        questionText: question.questionText,
        type: question.type,
        answer: answer.answer,
        correctAnswer: question.correctAnswer || question.options?.find(o => o.isCorrect)?.text,
        timeSpent: answer.timeSpent || 0,
        confidenceScore: answer.confidenceScore || 5
      };
    });

    // Send to ML service for evaluation
    let mlEvaluation = { results: [] };
    try {
      const mlResponse = await axios.post(ML_SERVICE_URL + '/evaluate-answers', {
        answers: answersForEvaluation
      }, {
        timeout: 10000  // 10 second timeout
      });
      mlEvaluation = mlResponse.data;
    } catch (error) {
      console.error('ML service evaluation failed:', error);
      // Continue with basic evaluation if ML service fails
    }

    // Process and evaluate answers
    const processedAnswers = answers.map(answer => {
      const question = questions.find(q => q._id.toString() === answer.questionId);
      const mlResult = mlEvaluation.results.find(r => r.questionId === answer.questionId);
      
      let isCorrect = false;
      if (question.type === 'mcq' || question.type === 'trueFalse') {
        // For MCQ and True/False, check if the selected option is marked as correct
        const selectedOption = question.options[answer.answer];
        isCorrect = selectedOption && selectedOption.isCorrect;
      } else if (mlResult) {
        isCorrect = mlResult.isCorrect;
      }
      
      // Update question performance metrics
      question.updatePerformance(isCorrect, answer.timeSpent || 0);
      question.save(); // Save the updated metrics

      return {
        questionId: question._id,
        answer: answer.answer,
        timeSpent: answer.timeSpent || 0,
        confidenceScore: answer.confidenceScore || 5,
        difficulty: parseInt(question.difficulty),
        isCorrect: isCorrect,
        type: question.type,
        needsManualGrading: question.type === 'descriptive' && !mlResult,
        evaluation: mlResult?.evaluation,
        score: mlResult?.score
      };
    });

    // Calculate scores
    const totalQuestions = processedAnswers.length;
    const autoGradedQuestions = processedAnswers.filter(a => !a.needsManualGrading);
    const correctAnswers = autoGradedQuestions.filter(a => a.isCorrect).length;
    const baselineScore = (correctAnswers / autoGradedQuestions.length) * 100;
    
    // Determine learning zone based on score
    let learningZone;
    if (baselineScore >= 80) learningZone = 'HARD';
    else if (baselineScore >= 60) learningZone = 'MEDIUM';
    else learningZone = 'EASY';

    // Create new diagnostic test
    const diagnosticTest = new DiagnosticTest({
      userId: req.user.googleId || req.user._id.toString(),
      answers: processedAnswers,
      validUntil: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000), // 30 days validity
      baselineScore,
      learningZone
    });

    await diagnosticTest.save();

    // Normalize baseline score to be between 1 and 10
    const normalizedScore = Math.min(Math.max(Math.round(baselineScore / 10), 1), 10);

    // Create or update student progress
    let studentProgress = await StudentProgress.findOne({
      studentId: req.user._id,
      courseId: courseId
    });

    if (!studentProgress) {
      studentProgress = new StudentProgress({
        studentId: req.user._id,
        courseId: courseId,
        diagnosticScore: {
          score: normalizedScore,
          completedAt: new Date()
        },
        learningZone: learningZone
      });
    } else {
      studentProgress.diagnosticScore = {
        score: normalizedScore,
        completedAt: new Date()
      };
      studentProgress.learningZone = learningZone;
    }

    await studentProgress.save();

    // Analyze performance
    const performanceByType = {};
    processedAnswers.forEach(answer => {
      if (!answer.needsManualGrading) {
        performanceByType[answer.type] = performanceByType[answer.type] || { total: 0, correct: 0 };
        performanceByType[answer.type].total++;
        if (answer.isCorrect) performanceByType[answer.type].correct++;
      }
    });

    // Generate recommendations
    const weakAreas = Object.entries(performanceByType)
      .filter(([_, stats]) => (stats.correct / stats.total) < 0.6)
      .map(([type]) => type);

    const strongAreas = Object.entries(performanceByType)
      .filter(([_, stats]) => (stats.correct / stats.total) >= 0.8)
      .map(([type]) => type);

    // Return results
    res.json({
      message: 'Diagnostic test submitted successfully',
      learningZone: learningZone,
      baselineScore: normalizedScore,
      recommendations: {
        weakAreas,
        strongAreas,
        recommendedTopics: weakAreas.length > 0 ? 
          ['Review basic concepts', 'Practice more ' + weakAreas.join(', ') + ' questions'] :
          ['Move on to advanced topics', 'Try harder questions']
      },
      performance: {
        byQuestionType: performanceByType,
        overallAccuracy: (correctAnswers / autoGradedQuestions.length) * 100,
        averageConfidence: processedAnswers.reduce((sum, a) => sum + a.confidenceScore, 0) / totalQuestions,
        totalTimeSpent: processedAnswers.reduce((sum, a) => sum + a.timeSpent, 0)
      }
    });
  } catch (error) {
    console.error('Error submitting diagnostic test:', error);
    res.status(500).json({ message: error.message });
  }
});

// Submit adaptive test answers
router.post('/submit-adaptive/:paperId', auth, async (req, res) => {
  try {
    const { answers, timeTaken } = req.body;
    
    // Get the paper first
    const paper = await Paper.findById(req.params.paperId).populate('questions');
    if (!paper) {
      return res.status(404).json({ message: 'Paper not found' });
    }

    // Calculate scores
    let totalScore = 0;
    let maxScore = 0;
    const questionScores = {};

    // Process each question's answer
    for (const question of paper.questions) {
      const questionId = question._id.toString();
      const answerData = answers[questionId];
      const marks = question.marks || 2;
      maxScore += marks;

      if (!answerData) {
        questionScores[questionId] = {
          scored: 0,
          possible: marks,
          isCorrect: false
        };
        continue;
      }

      let isCorrect = false;
      if (question.type === 'mcq') {
        // For MCQ, check if the answer matches the correct option index
        const correctOptionIndex = question.options.findIndex(opt => opt.isCorrect);
        isCorrect = answerData.answer === correctOptionIndex;
      } else {
        // For descriptive questions, store the answer for manual evaluation
        isCorrect = false; // Needs manual evaluation
      }

      const scored = isCorrect ? marks : 0;
      totalScore += scored;
      
      questionScores[questionId] = {
        scored,
        possible: marks,
        isCorrect,
        timeSpent: answerData.timeSpent || 0
      };
    }

    // Calculate percentage score
    const percentageScore = (totalScore / maxScore) * 100;

    // Update student progress if needed
    let studentProgress = await StudentProgress.findOne({
      studentId: req.user._id,
      courseId: paper.courseId
    });

    if (studentProgress) {
      studentProgress.examHistory.push({
        examId: paper._id,
        score: totalScore,
        maxScore,
        percentageScore,
        timeTaken,
        completedAt: new Date()
      });
      await studentProgress.save();
    }

    // Return detailed results
    res.json({
      totalScore,
      maxScore,
      percentageScore,
      questionScores,
      timeTaken,
      paperTitle: paper.title,
      courseId: paper.courseId
    });

  } catch (error) {
    console.error('Error submitting adaptive test:', error);
    res.status(500).json({ message: error.message });
  }
});

// Get student's diagnostic history
router.get('/student/history', auth, async (req, res) => {
  try {
    const studentProgress = await StudentProgress.findOne({
      studentId: req.user._id
    });

    if (!studentProgress) {
      return res.json({
        diagnosticScore: null,
        learningZone: null,
        progressHistory: []
      });
    }

    res.json({
      diagnosticScore: studentProgress.diagnosticScore,
      learningZone: studentProgress.learningZone,
      progressHistory: studentProgress.progressHistory
    });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// Get diagnostic test questions
router.get('/get-diagnostic', auth, async (req, res) => {
  try {
    const questions = await adaptiveTestService.getDiagnosticQuestions();
    if (!questions || questions.length === 0) {
      return res.status(404).json({ message: 'No diagnostic questions found' });
    }
    res.json(questions);
  } catch (error) {
    console.error('Error in get-diagnostic:', error);
    res.status(500).json({ message: error.message });
  }
});

// Get detailed diagnostic results
router.get('/results', auth, async (req, res) => {
  try {
    const latestTest = await DiagnosticTest.findOne({
      userId: req.user.googleId || req.user._id.toString()
    })
    .sort({ createdAt: -1 })
    .populate('answers.questionId');

    if (!latestTest) {
      return res.status(404).json({ message: 'No diagnostic test found' });
    }

    res.json({
      learningZone: latestTest.learningZone,
      baselineScore: latestTest.baselineScore,
      metadata: latestTest.metadata,
      isValid: latestTest.isStillValid(),
      validUntil: latestTest.validUntil,
      answers: latestTest.answers.map(answer => ({
        question: answer.questionId,
        yourAnswer: answer.answer,
        isCorrect: answer.isCorrect,
        timeSpent: answer.timeSpent,
        confidenceScore: answer.confidenceScore,
        difficulty: answer.difficulty
      }))
    });
  } catch (error) {
    console.error('Error fetching diagnostic results:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

// Get personalized questions for a paper
router.get('/get-personalized/:paperId', auth, async (req, res) => {
  try {
    const questions = await adaptiveTestService.getPersonalizedQuestions(
      req.params.paperId,
      req.query.learningZone,
      parseInt(req.query.baselineScore)
    );
    res.json(questions);
  } catch (error) {
    console.error('Error in get-personalized:', error);
    res.status(500).json({ message: error.message });
  }
});

// Submit adaptive exam
router.post('/submit-adaptive/:paperId', auth, async (req, res) => {
  try {
    const score = await adaptiveTestService.calculateAdaptiveScore(
      req.body.answers,
      req.body.metadata.learningZone
    );
    res.json(score);
  } catch (error) {
    console.error('Error in submit-adaptive:', error);
    res.status(500).json({ message: error.message });
  }
});

// Get learning progress
router.get('/learning-progress', auth, async (req, res) => {
  try {
    const progress = await DiagnosticTest.find({ student: req.user.id })
      .sort({ createdAt: -1 })
      .limit(10);
    res.json(progress);
  } catch (error) {
    console.error('Error in learning-progress:', error);
    res.status(500).json({ message: error.message });
  }
});

// Get diagnostic test status
router.get('/status', auth, async (req, res) => {
  try {
    // Use googleId instead of _id for finding the test
    const latestTest = await DiagnosticTest.findOne({ 
      userId: req.user.googleId || req.user._id.toString()
    }).sort({ createdAt: -1 });

    if (!latestTest) {
      return res.json({
        hasTakenDiagnostic: false,
        isValid: false
      });
    }

    const isValid = latestTest.isStillValid();
    
    res.json({
      hasTakenDiagnostic: true,
      isValid,
      learningZone: isValid ? latestTest.learningZone : null,
      baselineScore: isValid ? latestTest.baselineScore : null
    });
  } catch (error) {
    console.error('Error checking diagnostic status:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

// Get personalized questions
router.get('/personalized-questions', auth, async (req, res) => {
  try {
    const { paperId, learningZone, baselineScore } = req.query;

    // Get questions based on learning zone
    const questions = await Question.aggregate([
      {
        $match: {
          paperId,
          isActive: true,
          difficulty: learningZone.toLowerCase()
        }
      },
      { $sample: { size: 10 } } // Get 10 random questions
    ]);

    // Add some questions from adjacent zones based on baseline score
    if (baselineScore > 70) {
      // Add some harder questions
      const harderQuestions = await Question.aggregate([
        {
          $match: {
            paperId,
            isActive: true,
            difficulty: learningZone === 'Easy' ? 'medium' : 'hard'
          }
        },
        { $sample: { size: 5 } }
      ]);
      questions.push(...harderQuestions);
    } else if (baselineScore < 40) {
      // Add some easier questions
      const easierQuestions = await Question.aggregate([
        {
          $match: {
            paperId,
            isActive: true,
            difficulty: learningZone === 'Hard' ? 'medium' : 'easy'
          }
        },
        { $sample: { size: 5 } }
      ]);
      questions.push(...easierQuestions);
    }

    // Shuffle and return questions
    res.json(questions.sort(() => Math.random() - 0.5));
  } catch (error) {
    console.error('Error fetching personalized questions:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

// Get adaptive test questions
router.get('/adaptive-test', auth, async (req, res) => {
  try {
    // Get student's diagnostic score
    const studentProgress = await StudentProgress.findOne({
      studentId: req.user._id
    });

    if (!studentProgress || !studentProgress.diagnosticScore) {
      return res.status(400).json({ message: 'Please complete the diagnostic test first' });
    }

    // Get adaptive questions based on diagnostic score
    const adaptiveQuestions = await adaptiveTestService.getAdaptiveQuestions(
      req.user._id,
      studentProgress.diagnosticScore.score
    );

    res.json(adaptiveQuestions);
  } catch (error) {
    console.error('Error getting adaptive test:', error);
    res.status(500).json({ message: error.message });
  }
});

// Submit adaptive test
router.post('/adaptive-test/submit', auth, async (req, res) => {
  try {
    const { answers } = req.body;

    if (!answers || !Array.isArray(answers)) {
      return res.status(400).json({ message: 'Invalid answers format' });
    }

    // Get student's diagnostic score
    const studentProgress = await StudentProgress.findOne({
      studentId: req.user._id
    });

    if (!studentProgress || !studentProgress.diagnosticScore) {
      return res.status(400).json({ message: 'No diagnostic score found' });
    }

    // Process answers and calculate weighted score
    const result = await adaptiveTestService.calculateAdaptiveScore(answers);

    // Update student progress
    studentProgress.adaptiveTestResults = {
      score: result.finalScore,
      completedAt: new Date(),
      zoneScores: result.zoneScores
    };

    await studentProgress.save();

    // Return results
    res.json({
      message: 'Adaptive test submitted successfully',
      score: result.finalScore,
      maxScore: result.maxScore,
      zoneScores: result.zoneScores,
      performance: {
        byZone: {
          easy: (result.zoneScores.easy.score / result.zoneScores.easy.total) * 100,
          medium: (result.zoneScores.medium.score / result.zoneScores.medium.total) * 100,
          hard: (result.zoneScores.hard.score / result.zoneScores.hard.total) * 100
        },
        overallScore: (result.finalScore / result.maxScore) * 100
      }
    });
  } catch (error) {
    console.error('Error submitting adaptive test:', error);
    res.status(500).json({ message: error.message });
  }
});

// Get actual test questions after diagnostic
router.get('/actual-test', auth, async (req, res) => {
  try {
    // Get student's diagnostic score
    const studentProgress = await StudentProgress.findOne({
      studentId: req.user._id
    });

    if (!studentProgress || !studentProgress.diagnosticScore) {
      return res.status(400).json({ message: 'Please complete the diagnostic test first' });
    }

    // Get questions based on diagnostic score
    const questions = await adaptiveTestService.getAdaptiveQuestions(
      req.user._id,
      studentProgress.diagnosticScore.score
    );

    res.json(questions);
  } catch (error) {
    console.error('Error getting actual test:', error);
    res.status(500).json({ message: error.message });
  }
});

// Submit actual test
router.post('/actual-test/submit', auth, async (req, res) => {
  try {
    const { answers } = req.body;

    if (!answers || !Array.isArray(answers)) {
      return res.status(400).json({ message: 'Invalid answers format' });
    }

    // Get student's diagnostic score
    const studentProgress = await StudentProgress.findOne({
      studentId: req.user._id
    });

    if (!studentProgress || !studentProgress.diagnosticScore) {
      return res.status(400).json({ message: 'No diagnostic score found' });
    }

    // Process answers and calculate score
    let totalScore = 0;
    let maxScore = 0;
    const scoresByType = {
      descriptive: { score: 0, total: 0 },
      mcq: { score: 0, total: 0 },
      trueFalse: { score: 0, total: 0 },
      fillInBlanks: { score: 0, total: 0 }
    };

    // Get all questions
    const questionIds = answers.map(a => a.questionId);
    const questions = await Question.find({ _id: { $in: questionIds } });
    const questionsMap = new Map(questions.map(q => [q._id.toString(), q]));

    // Calculate scores
    for (const answer of answers) {
      const question = questionsMap.get(answer.questionId);
      if (!question) continue;

      const marks = question.type === 'descriptive' ? 4 : 1;
      maxScore += marks;
      scoresByType[question.type].total += marks;

      let isCorrect = false;
      switch (question.type) {
        case 'mcq':
        case 'trueFalse':
          isCorrect = question.options[answer.answer]?.isCorrect || false;
          break;
        case 'fillInBlanks':
          isCorrect = answer.answer.toLowerCase().trim() === question.correctAnswer.toLowerCase().trim();
          break;
        case 'descriptive':
          // For descriptive questions, we'll count them as partially correct if they provided an answer
          isCorrect = answer.answer && answer.answer.trim().length > 0;
          break;
      }

      if (isCorrect) {
        totalScore += marks;
        scoresByType[question.type].score += marks;
      }
    }

    // Calculate performance metrics
    const performance = {
      overall: (totalScore / maxScore) * 100,
      byType: Object.entries(scoresByType).reduce((acc, [type, data]) => ({
        ...acc,
        [type]: data.total > 0 ? (data.score / data.total) * 100 : 0
      }), {})
    };

    // Update student progress
    studentProgress.actualTestResults = {
      score: totalScore,
      maxScore,
      completedAt: new Date(),
      performance,
      scoresByType
    };

    await studentProgress.save();

    // Return results
    res.json({
      message: 'Test submitted successfully',
      score: totalScore,
      maxScore,
      performance,
      scoresByType
    });
  } catch (error) {
    console.error('Error submitting actual test:', error);
    res.status(500).json({ message: error.message });
  }
});

// Generate adaptive test
router.get('/generate-adaptive', auth, async (req, res) => {
  try {
    const { confidence, score } = req.query;
    
    if (!score) {
      return res.status(400).json({ message: 'Score is required' });
    }

    // Get adaptive questions
    const adaptiveTest = await adaptiveTestService.getAdaptiveQuestions(
      req.user._id.toString(), // Convert ObjectId to string
      parseFloat(score)
    );

    if (!adaptiveTest || !adaptiveTest.questions || adaptiveTest.questions.length === 0) {
      return res.status(404).json({ message: 'No questions available' });
    }

    res.json(adaptiveTest);
  } catch (error) {
    console.error('Error generating adaptive test:', error);
    res.status(500).json({ message: error.message });
  }
});

module.exports = router; 