const mongoose = require('mongoose');

const dbmsQuestionSchema = new mongoose.Schema({
  courseId: {
    type: String,
    required: true
  },
  questionText: {
    type: String,
    required: true
  },
  type: {
    type: String,
    enum: ['mcq', 'descriptive', 'trueFalse', 'fillInBlanks'],
    required: true
  },
  difficulty: {
    type: Number,
    default: 5,
    min: 1,
    max: 10
  },
  topic: {
    type: String,
    required: false,
    default: 'DBMS'
  },
  correctAnswer: {
    type: mongoose.Schema.Types.Mixed,
    required: false
  },
  status: {
    type: String,
    enum: ['pending', 'approved', 'rejected'],
    default: 'approved'
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
  explanation: {
    type: String,
    default: ''
  },
  answer: {
    type: mongoose.Schema.Types.Mixed
  },
  correct_answer: {
    type: mongoose.Schema.Types.Mixed
  },
  metadata: {
    generatedAt: {
      type: Date,
      default: Date.now
    },
    conceptsCovered: [String],
    timeToSolve: Number
  },
  createdAt: {
    type: Date,
    default: Date.now
  },
  usedInPapers: [{
    type: String
  }]
}, {
  collection: 'dbmsquestions'
});

const DBMSQuestion = mongoose.model('DBMSQuestion', dbmsQuestionSchema);

module.exports = DBMSQuestion;

