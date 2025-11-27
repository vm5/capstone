const mongoose = require('mongoose');
const Question = require('../models/Question');

// MongoDB connection string
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/pesuprep';

// Helper function to convert difficulty string to number
function difficultyToNumber(difficulty) {
  if (typeof difficulty === 'number') return difficulty;
  
  switch(difficulty.toLowerCase()) {
    case 'easy':
      return Math.floor(Math.random() * 3) + 1; // 1-3
    case 'moderate':
      return Math.floor(Math.random() * 3) + 4; // 4-6
    case 'challenging':
    case 'hard':
      return Math.floor(Math.random() * 4) + 7; // 7-10
    default:
      return Math.floor(Math.random() * 10) + 1; // 1-10
  }
}

// Helper function to extract topic from question text
function extractTopic(questionText, conceptsCovered) {
  if (conceptsCovered && conceptsCovered.length > 0) {
    return conceptsCovered[0]; // Use the first concept as the topic
  }
  
  // Extract topic from question text (simple heuristic)
  const topicKeywords = [
    'hidden states', 'hidden markov model', 'hmm', 
    'bagging', 'random forest', 'decision tree',
    'cnn', 'convolutional neural network',
    'em', 'expectation maximization',
    'ensemble learning',
    'map', 'maximum a posteriori',
    'svm', 'support vector machine',
    'neural network', 'regression', 'classification',
    'optimization'
  ];
  
  const lowerText = questionText.toLowerCase();
  for (const keyword of topicKeywords) {
    if (lowerText.includes(keyword.toLowerCase())) {
      return keyword;
    }
  }
  
  return 'General ML'; // Default topic
}

async function fixQuestions() {
  try {
    // Connect to MongoDB
    await mongoose.connect(MONGODB_URI);
    console.log('Connected to MongoDB');

    // Get all questions for the course
    const questions = await Question.find({ courseId: 'UE22CS352A' });
    console.log(`Found ${questions.length} questions`);

    // Update each question
    let updated = 0;
    for (const question of questions) {
      let wasUpdated = false;

      // Set status to approved if not set
      if (!question.status || question.status === 'pending') {
        question.status = 'approved';
        wasUpdated = true;
      }

      // Set type if not set correctly
      if (!question.type || !['mcq', 'descriptive', 'trueFalse', 'fillInBlanks'].includes(question.type)) {
        // Determine type based on question structure
        if (question.options && question.options.length > 0) {
          question.type = 'mcq';
        } else if (question.questionText.toLowerCase().includes('true or false')) {
          question.type = 'trueFalse';
        } else if (question.questionText.includes('___') || question.questionText.includes('...')) {
          question.type = 'fillInBlanks';
        } else {
          question.type = 'descriptive';
        }
        wasUpdated = true;
      }

      // Convert difficulty string to number
      if (typeof question.difficulty === 'string' || !question.difficulty) {
        question.difficulty = difficultyToNumber(question.difficulty || 'moderate');
        wasUpdated = true;
      }

      // Set topic if not set
      if (!question.topic) {
        question.topic = extractTopic(question.questionText, question.metadata?.conceptsCovered);
        wasUpdated = true;
      }

      // Set correctAnswer if not set
      if (!question.correctAnswer) {
        if (question.type === 'mcq' && question.options && question.options.length > 0) {
          // For MCQ, find the correct option index
          const correctIndex = question.options.findIndex(opt => opt.isCorrect);
          question.correctAnswer = correctIndex >= 0 ? correctIndex : 0;
        } else if (question.type === 'trueFalse') {
          // For true/false, set a default
          question.correctAnswer = "true";
        } else if (question.type === 'fillInBlanks') {
          // For fill in blanks, use a placeholder
          question.correctAnswer = "[Answer to be provided]";
        } else {
          // For descriptive, use a placeholder
          question.correctAnswer = "Sample answer to be provided by instructor";
        }
        wasUpdated = true;
      }

      // Fix options array
      if (question.options && question.options.length > 0) {
        let needsOptionsFix = false;
        
        // Check if any option is missing isCorrect
        for (const opt of question.options) {
          if (typeof opt.isCorrect === 'undefined') {
            needsOptionsFix = true;
            break;
          }
        }

        if (needsOptionsFix) {
          // For existing options, set the first one as correct
          question.options = question.options.map((opt, idx) => ({
            text: opt.text,
            isCorrect: idx === 0
          }));
          wasUpdated = true;
        }
      } else if (question.type === 'mcq') {
        // If no options exist for MCQ
        question.options = [
          { text: 'Option A', isCorrect: true },
          { text: 'Option B', isCorrect: false },
          { text: 'Option C', isCorrect: false },
          { text: 'Option D', isCorrect: false }
        ];
        wasUpdated = true;
      } else if (question.type === 'trueFalse') {
        // If no options exist for true/false
        question.options = [
          { text: 'True', isCorrect: question.correctAnswer === 'true' },
          { text: 'False', isCorrect: question.correctAnswer === 'false' }
        ];
        wasUpdated = true;
      }

      // Save if any changes were made
      if (wasUpdated) {
        try {
          await question.save();
          updated++;
        } catch (err) {
          console.error('Error saving question:', err.message);
          console.error('Question data:', JSON.stringify(question, null, 2));
        }
      }
    }

    console.log(`Updated ${updated} questions successfully`);

    // Get updated stats
    const updatedQuestions = await Question.find({ courseId: 'UE22CS352A' });
    const stats = {
      total: updatedQuestions.length,
      byStatus: {
        approved: updatedQuestions.filter(q => q.status === 'approved').length,
        pending: updatedQuestions.filter(q => q.status === 'pending').length,
        rejected: updatedQuestions.filter(q => q.status === 'rejected').length
      },
      byType: {
        mcq: updatedQuestions.filter(q => q.type === 'mcq').length,
        descriptive: updatedQuestions.filter(q => q.type === 'descriptive').length,
        trueFalse: updatedQuestions.filter(q => q.type === 'trueFalse').length,
        fillInBlanks: updatedQuestions.filter(q => q.type === 'fillInBlanks').length
      },
      byDifficulty: {
        easy: updatedQuestions.filter(q => q.difficulty <= 3).length,
        moderate: updatedQuestions.filter(q => q.difficulty > 3 && q.difficulty <= 6).length,
        challenging: updatedQuestions.filter(q => q.difficulty > 6).length
      }
    };

    console.log('Updated stats:', stats);

  } catch (error) {
    console.error('Error updating questions:', error);
  } finally {
    // Close MongoDB connection
    await mongoose.connection.close();
    console.log('Disconnected from MongoDB');
  }
}

// Run the script
fixQuestions(); 