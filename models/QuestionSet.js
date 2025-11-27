const mongoose = require('mongoose');

const questionSetSchema = new mongoose.Schema({
  courseId: String,
  type: String,  // 'quiz' or 'isa'
  title: String,
  paperNumber: Number,
  questions: [{
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Question'
  }],
  difficulty: Number,
  totalMarks: Number,
  duration: Number,
  scheduledDate: Date,
  scheduledTime: String,
  createdAt: { type: Date, default: Date.now },
  createdBy: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
  status: {
    type: String,
    enum: ['draft', 'scheduled', 'approved', 'rejected', 'active', 'completed'],
    default: 'draft'
  },
  questionTypes: {
    mcq: Number,
    descriptive: Number
  },
  difficulty: {
    easy: Number,
    moderate: Number,
    challenging: Number
  }
});

const QuestionSet = mongoose.model('QuestionSet', questionSetSchema);

module.exports = QuestionSet; 