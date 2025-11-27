const mongoose = require('mongoose');

const scheduleSchema = new mongoose.Schema({
  title: {
    type: String,
    required: true
  },
  type: {
    type: String,
    enum: ['ISA', 'QUIZ', 'DIAGNOSTIC'],
    required: true
  },
  startTime: {
    type: Date,
    required: true
  },
  endTime: {
    type: Date,
    required: true
  },
  duration: {
    type: Number,
    required: true,
    min: 1
  },
  courseId: {
    type: String,
    required: true
  },
  description: String,
  createdBy: {
    type: String,
    required: true
  },
  paperId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Paper',
    required: function() { return this.type === 'ISA'; }  // Only required for ISA type
  },
  paperType: {  // Added to distinguish between MongoDB papers and PDF files
    type: String,
    enum: ['mongodb', 'pdf'],
    required: function() { return this.type === 'ISA'; }  // Only required for ISA type
  },
  questionPool: {
    easy: [{
      text: {
        type: String,
        required: true
      },
      type: {
        type: String,
        enum: ['mcq', 'descriptive'],
        required: true
      },
      options: [String], // For MCQs
      correctAnswer: {
        type: mongoose.Schema.Types.Mixed, // Can be number (for MCQ) or string (for descriptive)
        required: true
      },
      marks: {
        type: Number,
        required: true,
        default: 1
      },
      difficultyLevel: {
        type: Number,
        min: 1,
        max: 3,
        required: true
      }
    }],
    medium: [{
      text: {
        type: String,
        required: true
      },
      type: {
        type: String,
        enum: ['mcq', 'descriptive'],
        required: true
      },
      options: [String],
      correctAnswer: {
        type: mongoose.Schema.Types.Mixed,
        required: true
      },
      marks: {
        type: Number,
        required: true,
        default: 2
      },
      difficultyLevel: {
        type: Number,
        min: 4,
        max: 6,
        required: true
      }
    }],
    hard: [{
      text: {
        type: String,
        required: true
      },
      type: {
        type: String,
        enum: ['mcq', 'descriptive'],
        required: true
      },
      options: [String],
      correctAnswer: {
        type: mongoose.Schema.Types.Mixed,
        required: true
      },
      marks: {
        type: Number,
        required: true,
        default: 4
      },
      difficultyLevel: {
        type: Number,
        min: 7,
        max: 10,
        required: true
      }
    }]
  },
  questionDistribution: {
    easy: {
      count: {
        type: Number,
        default: 16 // Default number of easy questions
      },
      totalMarks: {
        type: Number,
        default: 16
      }
    },
    medium: {
      count: {
        type: Number,
        default: 4 // Default number of medium questions
      },
      totalMarks: {
        type: Number,
        default: 8
      }
    },
    hard: {
      count: {
        type: Number,
        default: 4 // Default number of hard questions
      },
      totalMarks: {
        type: Number,
        default: 16
      }
    }
  }
}, {
  timestamps: true
});

// Add index for faster queries
scheduleSchema.index({ startTime: 1 });
scheduleSchema.index({ courseId: 1 });
scheduleSchema.index({ createdBy: 1 });

// Method to generate exam paper based on student's learning zone
scheduleSchema.methods.generateExamPaper = function(learningZone) {
  const getRandomQuestions = (pool, count) => {
    const shuffled = [...pool].sort(() => 0.5 - Math.random());
    return shuffled.slice(0, count);
  };

  const distribution = this.questionDistribution;
  let examPaper = {
    easy: [],
    medium: [],
    hard: []
  };

  // Base distribution
  examPaper.easy = getRandomQuestions(this.questionPool.easy, distribution.easy.count);
  examPaper.medium = getRandomQuestions(this.questionPool.medium, distribution.medium.count);
  examPaper.hard = getRandomQuestions(this.questionPool.hard, distribution.hard.count);

  // Adjust based on learning zone
  if (learningZone === 'MEDIUM') {
    // Increase medium questions, decrease easy ones
    const additionalMedium = Math.min(4, distribution.easy.count);
    examPaper.medium = [
      ...examPaper.medium,
      ...getRandomQuestions(
        this.questionPool.medium.filter(q => !examPaper.medium.includes(q)),
        additionalMedium
      )
    ];
    examPaper.easy = examPaper.easy.slice(additionalMedium);
  } else if (learningZone === 'HARD') {
    // Increase hard questions, decrease easy ones
    const additionalHard = Math.min(4, distribution.easy.count);
    examPaper.hard = [
      ...examPaper.hard,
      ...getRandomQuestions(
        this.questionPool.hard.filter(q => !examPaper.hard.includes(q)),
        additionalHard
      )
    ];
    examPaper.easy = examPaper.easy.slice(additionalHard);
  }

  return {
    questions: [
      ...examPaper.easy,
      ...examPaper.medium,
      ...examPaper.hard
    ],
    distribution: {
      easy: examPaper.easy.length,
      medium: examPaper.medium.length,
      hard: examPaper.hard.length
    }
  };
};

const Schedule = mongoose.model('Schedule', scheduleSchema);

module.exports = Schedule; 