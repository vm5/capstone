const mongoose = require('mongoose');
const Question = require('../models/Question');

const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/pesuprep';

const sampleQuestions = [
  // Descriptive Questions (Hard)
  {
    courseId: 'ML101',
    topic: 'Neural Networks',
    questionText: "Explain the concept of backpropagation in neural networks and its importance in deep learning.",
    type: "descriptive",
    difficulty: 8,
    status: "approved",
    correctAnswer: "Backpropagation is an algorithm used to train neural networks by calculating gradients of the loss function with respect to the network weights, enabling efficient weight updates during training. It works by propagating the error backwards through the network layers.",
    options: []
  },
  {
    courseId: 'ML101',
    topic: 'Machine Learning Fundamentals',
    questionText: "Compare and contrast supervised and unsupervised learning with examples.",
    type: "descriptive",
    difficulty: 7,
    status: "approved",
    correctAnswer: "Supervised learning uses labeled data (e.g., classification, regression) while unsupervised learning works with unlabeled data (e.g., clustering, dimensionality reduction). Examples: Supervised - spam detection, house price prediction; Unsupervised - customer segmentation, anomaly detection.",
    options: []
  },
  
  // True/False Questions (Medium)
  {
    courseId: 'ML101',
    topic: 'Decision Trees',
    questionText: "Decision trees are a type of supervised learning algorithm.",
    type: "trueFalse",
    options: [
      { text: "true", isCorrect: true },
      { text: "false", isCorrect: false }
    ],
    difficulty: 5,
    status: "approved",
    correctAnswer: 0
  },
  {
    courseId: 'ML101',
    topic: 'Clustering',
    questionText: "K-means clustering is an example of unsupervised learning.",
    type: "trueFalse",
    options: [
      { text: "true", isCorrect: true },
      { text: "false", isCorrect: false }
    ],
    difficulty: 4,
    status: "approved",
    correctAnswer: 0
  },
  
  // Fill in the Blanks (Medium)
  {
    courseId: 'ML101',
    topic: 'Dimensionality Reduction',
    questionText: "The process of reducing the dimensionality of input data is called _____ reduction.",
    type: "fillInBlanks",
    correctAnswer: "dimensionality",
    difficulty: 5,
    status: "approved",
    options: [{ text: "dimensionality", isCorrect: true }]
  },
  {
    courseId: 'ML101',
    topic: 'Neural Networks',
    questionText: "In a neural network, the _____ function determines the output of a node given an input or set of inputs.",
    type: "fillInBlanks",
    correctAnswer: "activation",
    difficulty: 4,
    status: "approved",
    options: [{ text: "activation", isCorrect: true }]
  },
  
  // MCQs (Medium)
  {
    courseId: 'ML101',
    topic: 'Optimization',
    questionText: "Which of the following is NOT a type of gradient descent optimization?",
    type: "mcq",
    options: [
      { text: "Batch gradient descent", isCorrect: false },
      { text: "Stochastic gradient descent", isCorrect: false },
      { text: "Linear gradient descent", isCorrect: true },
      { text: "Mini-batch gradient descent", isCorrect: false }
    ],
    difficulty: 5,
    status: "approved",
    correctAnswer: 2
  },
  {
    courseId: 'ML101',
    topic: 'Neural Networks',
    questionText: "What is the purpose of the activation function in a neural network?",
    type: "mcq",
    options: [
      { text: "To add non-linearity to the network", isCorrect: true },
      { text: "To reduce the number of parameters", isCorrect: false },
      { text: "To increase training speed", isCorrect: false },
      { text: "To prevent overfitting", isCorrect: false }
    ],
    difficulty: 4,
    status: "approved",
    correctAnswer: 0
  },
  {
    courseId: 'ML101',
    topic: 'Neural Networks',
    questionText: "Which of the following is a common activation function?",
    type: "mcq",
    options: [
      { text: "ReLU", isCorrect: true },
      { text: "JPEG", isCorrect: false },
      { text: "HTTP", isCorrect: false },
      { text: "ASCII", isCorrect: false }
    ],
    difficulty: 4,
    status: "approved",
    correctAnswer: 0
  },
  {
    courseId: 'ML101',
    topic: 'Convolutional Neural Networks',
    questionText: "What is the main advantage of using CNN for image processing?",
    type: "mcq",
    options: [
      { text: "Faster training time", isCorrect: false },
      { text: "Less memory usage", isCorrect: false },
      { text: "Feature extraction capabilities", isCorrect: true },
      { text: "Simpler architecture", isCorrect: false }
    ],
    difficulty: 5,
    status: "approved",
    correctAnswer: 2
  },
  {
    courseId: 'ML101',
    topic: 'Convolutional Neural Networks',
    questionText: "Which layer in a CNN performs the convolution operation?",
    type: "mcq",
    options: [
      { text: "Pooling layer", isCorrect: false },
      { text: "Convolutional layer", isCorrect: true },
      { text: "Fully connected layer", isCorrect: false },
      { text: "Output layer", isCorrect: false }
    ],
    difficulty: 4,
    status: "approved",
    correctAnswer: 1
  }
];

async function addSampleQuestions() {
  try {
    await mongoose.connect(MONGODB_URI, {
      useNewUrlParser: true,
      useUnifiedTopology: true
    });
    console.log('Connected to MongoDB');

    // Clear existing questions
    await Question.deleteMany({});
    console.log('Cleared existing questions');
    
    // Add new questions
    const result = await Question.insertMany(sampleQuestions);
    console.log(`Added ${result.length} sample questions`);

    console.log('Sample questions added successfully');
    process.exit(0);
  } catch (error) {
    console.error('Error:', error);
    process.exit(1);
  }
}

addSampleQuestions();