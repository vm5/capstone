const DiagnosticTest = require('../models/DiagnosticTest');
const Question = require('../models/Question');
const { spawn } = require('child_process');
const path = require('path');
const mongoose = require('mongoose');
const Paper = require('../models/Paper');

// GPT-2 model path
const MODEL_PATH = path.join(__dirname, '../../ml/fine_tuned_gpt2_large');

class AdaptiveTestService {
  // Get diagnostic test questions
  async getDiagnosticQuestions() {
    // Return hardcoded aptitude questions
    return [
      {
        _id: new mongoose.Types.ObjectId(),
        courseId: 'DIAG101',
        topic: 'Number Series',
        questionText: 'What comes next in the series: 3, 9, 27, 81, __?',
        type: 'mcq',
        options: [
          { text: '243', isCorrect: true },
          { text: '162', isCorrect: false },
          { text: '210', isCorrect: false },
          { text: '189', isCorrect: false }
        ],
        difficulty: 3,
        metadata: {
          skillsAssessed: ['pattern recognition', 'mathematical reasoning'],
          expectedTime: 45
        }
      },
      {
        _id: new mongoose.Types.ObjectId(),
        courseId: 'DIAG101',
        topic: 'Time and Work',
        questionText: 'A can do a piece of work in 20 days and B in 30 days. They work together for 4 days and then A leaves. How many more days will B take to complete the remaining work? Show your steps.',
        type: 'descriptive',
        correctAnswer: 'Solution:\n1. Work done in 1 day by A = 1/20\n2. Work done in 1 day by B = 1/30\n3. Work done in 1 day together = 1/20 + 1/30 = 5/60\n4. Work done in 4 days = 4 × 5/60 = 1/3\n5. Remaining work = 1 - 1/3 = 2/3\n6. Days B will take = 2/3 × 30 = 20 days',
        difficulty: 4,
        metadata: {
          skillsAssessed: ['fraction calculation', 'logical reasoning'],
          expectedTime: 180
        }
      },
      {
        _id: new mongoose.Types.ObjectId(),
        courseId: 'DIAG101',
        topic: 'Percentage',
        questionText: 'If the price of a book increases from Rs. 250 to Rs. 300, the percentage increase is _______ %.',
        type: 'fillInBlanks',
        correctAnswer: '20',
        difficulty: 3,
        metadata: {
          skillsAssessed: ['percentage calculation'],
          expectedTime: 45
        }
      },
      {
        _id: new mongoose.Types.ObjectId(),
        courseId: 'DIAG101',
        topic: 'Logical Deduction',
        questionText: 'Statement: All cats are animals. Some animals are black. Therefore, some cats must be black.',
        type: 'trueFalse',
        options: [
          { text: 'True', isCorrect: false },
          { text: 'False', isCorrect: true }
        ],
        difficulty: 4,
        metadata: {
          skillsAssessed: ['logical reasoning'],
          expectedTime: 45
        }
      }
    ];
  }

  // Process diagnostic test submission
  async processDiagnosticSubmission(studentId, answers) {
    try {
      // Calculate accuracy by difficulty
      const accuracyByDifficulty = {
        easy: this._calculateAccuracy(answers.filter(a => a.difficultyLevel <= 3)),
        medium: this._calculateAccuracy(answers.filter(a => a.difficultyLevel > 3 && a.difficultyLevel <= 6)),
        hard: this._calculateAccuracy(answers.filter(a => a.difficultyLevel > 6))
      };

      // Create diagnostic test record
      const diagnosticTest = new DiagnosticTest({
        student: studentId,
        diagnosticAnswers: answers,
        metadata: {
          totalTimeSpent: answers.reduce((total, a) => total + a.timeSpent, 0),
          averageConfidence: answers.reduce((total, a) => total + a.confidenceScore, 0) / answers.length,
          accuracyByDifficulty,
          recommendedTopics: await this._getRecommendedTopics(answers)
        }
      });

      // Calculate learning zone and baseline score
      diagnosticTest.learningZone = diagnosticTest.calculateLearningZone();
      diagnosticTest.baselineScore = diagnosticTest.calculateBaselineScore();

      await diagnosticTest.save();

      return {
        learningZone: diagnosticTest.learningZone,
        baselineScore: diagnosticTest.baselineScore,
        recommendedTopics: diagnosticTest.metadata.recommendedTopics
      };
    } catch (error) {
      console.error('Error processing diagnostic submission:', error);
      throw error;
    }
  }

  // Get personalized questions using collaborative filtering
  async getPersonalizedQuestions(paperId, learningZone, baselineScore) {
    try {
      // Get paper questions
      const paper = await Question.find({ paperId });
      
      // Filter questions based on learning zone
      const difficultyRanges = {
        'E': { min: 1, max: 4 },
        'M': { min: 3, max: 7 },
        'H': { min: 6, max: 10 }
      };

      const range = difficultyRanges[learningZone];
      
      // Use GPT-2 to analyze and select questions
      const selectedQuestions = await this._useGPT2ForQuestionSelection(
        paper,
        range,
        baselineScore
      );

      return this._structureQuestions(selectedQuestions);
    } catch (error) {
      console.error('Error getting personalized questions:', error);
      throw error;
    }
  }

  // Get questions for a student based on their diagnostic results
  async getAdaptiveQuestions(studentId, diagnosticScore) {
    try {
      // Define question structure with specific types and marks
      const questionStructure = {
        descriptive: { count: 2, marks: 4, difficultyRange: [7, 10] },
        trueFalse: { count: 2, marks: 1, difficultyRange: [4, 6] },
        fillInBlanks: { count: 2, marks: 1, difficultyRange: [4, 6] },
        mcq: { count: 5, marks: 1, difficultyRange: [4, 6] }
      };

      const selectedQuestions = [];
      let totalMarks = 0;

      // Get questions for each type
      for (const [type, config] of Object.entries(questionStructure)) {
        const adjustedRange = this._adjustDifficultyRange(config.difficultyRange, diagnosticScore);
        
        // Get questions from database
        const typeQuestions = await Question.aggregate([
          {
            $match: {
              type: type,
              difficulty: { $gte: adjustedRange[0], $lte: adjustedRange[1] },
              status: 'approved'
            }
          },
          { $sample: { size: config.count } }
        ]);

        // Add marks information
        typeQuestions.forEach(q => {
          q.marks = config.marks;
          totalMarks += config.marks;
        });

        selectedQuestions.push(...typeQuestions);
      }

      // Create a new paper record
      const paper = new Paper({
        title: `Adaptive Test - ${new Date().toISOString()}`,
        questions: selectedQuestions.map(q => q._id),
        totalMarks,
        duration: 60, // 1 hour default duration
        type: 'practice', // Use 'practice' type for adaptive tests
        courseId: 'ADAPTIVE_COURSE', // Default course ID for adaptive tests
        createdBy: studentId, // Use student ID as creator
        status: 'active',
        metadata: {
          studentId,
          diagnosticScore,
          generatedAt: new Date()
        }
      });

      await paper.save();
      
      // Return structured response with paper ID
      return {
        ...this._structureQuestions(selectedQuestions),
        paperId: paper._id,
        totalMarks,
        duration: 60
      };
    } catch (error) {
      console.error('Error getting adaptive questions:', error);
      throw error;
    }
  }

  // Helper function to adjust difficulty range based on diagnostic score
  _adjustDifficultyRange(range, score) {
    const [min, max] = range;
    const adjustment = Math.min(2, Math.max(-2, (score - 5) / 2));
    return [
      Math.max(1, Math.min(10, min + adjustment)),
      Math.max(1, Math.min(10, max + adjustment))
    ];
  }

  // Calculate final score
  async calculateAdaptiveScore(answers) {
    try {
      let totalScore = 0;
      const scoresByType = {
        descriptive: { total: 8, score: 0 },  // 2 questions × 4 marks
        trueFalse: { total: 2, score: 0 },    // 2 questions × 1 mark
        fillInBlanks: { total: 2, score: 0 }, // 2 questions × 1 mark
        mcq: { total: 5, score: 0 }           // 5 questions × 1 mark
      };

      // Calculate scores by type
      answers.forEach(answer => {
        if (answer.isCorrect) {
          scoresByType[answer.type].score += answer.marks;
          totalScore += answer.marks;
        }
      });

      return {
        scoresByType,
        totalScore,
        maxScore: 17, // Total possible marks: 8 + 2 + 2 + 5 = 17
        performance: {
          descriptive: (scoresByType.descriptive.score / scoresByType.descriptive.total) * 100,
          trueFalse: (scoresByType.trueFalse.score / scoresByType.trueFalse.total) * 100,
          fillInBlanks: (scoresByType.fillInBlanks.score / scoresByType.fillInBlanks.total) * 100,
          mcq: (scoresByType.mcq.score / scoresByType.mcq.total) * 100,
          overall: (totalScore / 17) * 100
        }
      };
    } catch (error) {
      console.error('Error calculating adaptive score:', error);
      throw error;
    }
  }

  // Private helper methods
  _calculateAccuracy(answers) {
    if (!answers.length) return 0;
    return answers.filter(a => a.isCorrect).length / answers.length;
  }

  async _useGPT2ForQuestionSelection(questions, difficultyRange, baselineScore) {
    return new Promise((resolve, reject) => {
      const pythonProcess = spawn('python', [
        path.join(__dirname, '../../ml/question_generator.py'),
        '--mode', 'select',
        '--questions', JSON.stringify(questions),
        '--difficulty_range', JSON.stringify(difficultyRange),
        '--baseline_score', baselineScore.toString()
      ]);

      let result = '';

      pythonProcess.stdout.on('data', (data) => {
        result += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        console.error(`GPT-2 Error: ${data}`);
      });

      pythonProcess.on('close', (code) => {
        if (code !== 0) {
          reject(new Error('Failed to select questions using GPT-2'));
        } else {
          resolve(JSON.parse(result));
        }
      });
    });
  }

  async _getRecommendedTopics(answers) {
    // Analyze incorrect answers to identify weak topics
    const incorrectAnswers = answers.filter(a => !a.isCorrect);
    const topics = incorrectAnswers.map(a => a.question.topic);
    
    // Count topic frequencies
    const topicCounts = topics.reduce((acc, topic) => {
      acc[topic] = (acc[topic] || 0) + 1;
      return acc;
    }, {});

    // Return top 3 most frequent topics
    return Object.entries(topicCounts)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 3)
      .map(([topic]) => topic);
  }

  _structureQuestions(questions) {
    return {
      questions,
      metadata: {
        totalQuestions: questions.length,
        distribution: {
          descriptive: questions.filter(q => q.type === 'descriptive').length,
          trueFalse: questions.filter(q => q.type === 'trueFalse').length,
          fillInBlanks: questions.filter(q => q.type === 'fillInBlanks').length,
          mcq: questions.filter(q => q.type === 'mcq').length
        },
        totalMarks: questions.reduce((sum, q) => sum + q.marks, 0),
        duration: 45 // 45 minutes for the actual test
      }
    };
  }
}

// Export an instance of the service
module.exports = new AdaptiveTestService(); 