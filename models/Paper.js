const mongoose = require('mongoose');

const optionSchema = new mongoose.Schema({
  text: String,
  isCorrect: Boolean
}, { _id: false });

const questionSchema = new mongoose.Schema({
  _id: mongoose.Schema.Types.ObjectId,
  type: {
    type: String,
    enum: ['mcq', 'descriptive', 'trueFalse', 'fillInBlanks'],
    required: true
  },
  questionText: {
    type: String,
    required: true
  },
  options: [optionSchema],
  correctAnswer: mongoose.Schema.Types.Mixed,
  difficulty: {
    type: Number,
    required: true
  },
  marks: {
    type: Number,
    required: true
  }
}, { _id: false });

const paperSchema = new mongoose.Schema({
  title: {
    type: String,
    required: true
  },
  courseId: {
    type: String,
    required: true
  },
  questions: [questionSchema],
  questionTypes: {
    model: {
      type: String,
      required: true,
      enum: ['Question', 'QuestionTemplate'],
      default: 'Question'
    },
    mcq: {
      type: Number,
      default: 0
    },
    descriptive: {
      type: Number,
      default: 0
    }
  },
  totalMarks: {
    type: Number,
    required: true
  },
  duration: {
    type: Number,
    required: true
  },
  type: {
    type: String,
    enum: ['quiz', 'exam', 'practice'],
    required: true
  },
  createdBy: {
    type: String,
    required: true
  },
  status: {
    type: String,
    enum: ['draft', 'published', 'archived', 'active'],
    default: 'draft'
  },
  difficulty: {
    easy: {
      type: Number,
      default: 0
    },
    moderate: {
      type: Number,
      default: 0
    },
    challenging: {
      type: Number,
      default: 0
    }
  },
  metadata: {
    difficulty: {
      easy: Number,
      moderate: Number,
      challenging: Number
    },
    questionTypes: {
      mcq: Number,
      descriptive: Number,
      trueFalse: Number,
      fillInBlanks: Number
    }
  },
  createdAt: {
    type: Date,
    default: Date.now
  }
}, {
  collection: 'papers'  // Explicitly set the collection name
});

// Pre-save middleware to populate difficulty and questionType counts
paperSchema.pre('save', async function(next) {
  if (this.isModified('questions')) {
    // Reset counts
    this.difficulty = {
      easy: 0,
      moderate: 0,
      challenging: 0
    };

    this.questionTypes = {
      model: this.questionTypes.model || 'Question',
      mcq: 0,
      descriptive: 0
    };

    // Count questions by difficulty and type
    this.questions.forEach(question => {
      // Normalize difficulty to 1-10 scale if it's 0-1
      let normalizedDifficulty = question.difficulty;
      if (normalizedDifficulty <= 1) {
        normalizedDifficulty = Math.round(normalizedDifficulty * 10);
      }

      // Difficulty count
      if (normalizedDifficulty <= 4) {
        this.difficulty.easy++;
      } else if (normalizedDifficulty <= 7) {
        this.difficulty.moderate++;
      } else {
        this.difficulty.challenging++;
      }

      // Type count
      if (question.type === 'mcq') {
        this.questionTypes.mcq++;
      } else if (question.type === 'descriptive') {
        this.questionTypes.descriptive++;
      }
    });
  }
  next();
});

const Paper = mongoose.model('Paper', paperSchema);

module.exports = Paper; 