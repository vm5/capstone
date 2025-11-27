const mongoose = require('mongoose');
const Paper = require('../models/Paper');
const Question = require('../models/Question');

// MongoDB connection string
const MONGODB_URI = 'mongodb://localhost:27017/pesuprep';

async function addSamplePaper() {
  try {
    // Connect to MongoDB
    await mongoose.connect(MONGODB_URI, {
      useNewUrlParser: true,
      useUnifiedTopology: true
    });

    console.log('Connected to MongoDB');

    // Get all diagnostic questions
    const questions = await Question.find({ courseId: 'DIAG101' });
    
    if (questions.length < 4) {
      throw new Error('Not enough questions found. Please run addSampleQuestions.js first.');
    }

    // Create sample paper template
    const paperTemplate = {
      title: 'Diagnostic Test - Machine Learning Fundamentals',
      courseId: 'DIAG101',
      questions: questions.map(q => q._id), // Just the question IDs
      totalMarks: 20,
      duration: 20, // 20 minutes
      type: 'quiz',
      createdBy: 'system',
      status: 'active',
      difficulty: {
        easy: 2,
        moderate: 1,
        challenging: 1
      },
      questionTypes: {
        mcq: 1,
        descriptive: 3 // Including fillInBlanks and trueFalse here
      }
    };

    // Clear existing diagnostic papers
    await Paper.deleteMany({ courseId: 'DIAG101', type: 'quiz' });
    console.log('Cleared existing diagnostic papers');

    // Create new paper
    const paper = await Paper.create(paperTemplate);
    console.log('Created diagnostic paper with ID:', paper._id);

  } catch (error) {
    console.error('Error:', error);
  } finally {
    await mongoose.disconnect();
    console.log('Disconnected from MongoDB');
  }
}

// Run the script
addSamplePaper(); 