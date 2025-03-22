const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const axios = require('axios');
const querystring = require('querystring');
const { google } = require('googleapis');
const { OAuth2Client } = require('google-auth-library');
const multer = require('multer');
const mammoth = require('mammoth');
const pdf = require('pdf-parse');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const { AutoModelForCausalLM, AutoTokenizer } = require('transformers');
const torch = require('torch');

const app = express();

// Middleware
app.use(cors({
  origin: 'http://localhost:3000',
  credentials: true
}));
app.use(express.json());

// Set Mongoose strictQuery option
mongoose.set('strictQuery', true);

// MongoDB Connection
mongoose.connect('mongodb://localhost:27017/pesuprep', {
  useNewUrlParser: true,
  useUnifiedTopology: true,
  serverSelectionTimeoutMS: 5000, // Increase timeout to 5 seconds
  socketTimeoutMS: 45000, // Increase socket timeout to 45 seconds
})
.then(() => console.log('Connected to MongoDB'))
.catch(err => console.error('MongoDB connection error:', err));

// User Schema
const userSchema = new mongoose.Schema({
  email: { type: String, required: true, unique: true },
  name: String,
  role: { type: String, enum: ['student', 'teacher'], default: 'student' },
  googleId: String,
  picture: String,
  courses: [{
    courseId: String,
    courseName: String,
    enrolledAt: { type: Date, default: Date.now },
    progress: {
      quizzesTaken: Number,
      averageScore: Number,
      lastActivity: Date,
      completedTopics: [String]
    }
  }],
  teachingCourses: [{
    courseId: String,
    courseName: String,
    addedAt: { type: Date, default: Date.now },
    content: [{
      title: String,
      type: String,  // video, document, interactive
      url: String,
      uploadedAt: Date
    }]
  }],
  createdAt: { type: Date, default: Date.now },
  lastLogin: { type: Date },
  isActive: { type: Boolean, default: true }
});

// Question Schema for ML-generated questions
const questionSchema = new mongoose.Schema({
  courseId: String,
  topic: String,
  questionText: String,
  options: [{
    text: String,
    isCorrect: Boolean
  }],
  difficulty: Number,  // 1-10 scale
  type: String,  // quiz or isa
  status: {
    type: String,
    enum: ['pending', 'approved', 'rejected'],
    default: 'pending'
  },
  feedback: String,  // Teacher's feedback if rejected
  generatedAt: { type: Date, default: Date.now },
  approvedBy: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
  usedInSets: [String],  // Question set IDs where this question was used
  metadata: {
    topicDifficulty: Number,
    conceptsCovered: [String],
    timeToSolve: Number  // estimated time in minutes
  }
});

// Question Set Schema for quizzes and ISAs
const questionSetSchema = new mongoose.Schema({
  courseId: String,
  type: String,  // 'quiz' or 'isa'
  title: String,
  questions: [{
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Question'
  }],
  difficulty: Number,  // average difficulty of questions
  totalMarks: Number,
  duration: Number,  // in minutes
  createdAt: { type: Date, default: Date.now },
  createdBy: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
  status: {
    type: String,
    enum: ['draft', 'approved', 'active', 'completed'],
    default: 'draft'
  }
});

// Student Progress Schema
const studentProgressSchema = new mongoose.Schema({
  studentId: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
  courseId: String,
  quizzes: [{
    quizId: { type: mongoose.Schema.Types.ObjectId, ref: 'QuestionSet' },
    score: Number,
    timeTaken: Number,
    completedAt: Date,
    answers: [{
      questionId: { type: mongoose.Schema.Types.ObjectId, ref: 'Question' },
      selectedOption: Number,
      isCorrect: Boolean,
      timeSpent: Number  // time spent on this question
    }]
  }],
  isas: [{
    isaId: { type: mongoose.Schema.Types.ObjectId, ref: 'QuestionSet' },
    score: Number,
    setNumber: Number,
    completedAt: Date
  }],
  overallProgress: {
    topicsCompleted: [String],
    averageQuizScore: Number,
    averageIsaScore: Number,
    totalTimeSpent: Number,
    strengthTopics: [String],
    weaknessTopics: [String],
    lastUpdated: Date
  }
});

const User = mongoose.model('User', userSchema);
const Question = mongoose.model('Question', questionSchema);
const QuestionSet = mongoose.model('QuestionSet', questionSetSchema);
const StudentProgress = mongoose.model('StudentProgress', studentProgressSchema);

// Helper function to check if email is from PES domains or teacher's personal email
const isValidTeacherEmail = (email) => {
  const pesEmailPatterns = [
    /@pes\.edu$/,
    /@pesu\.pes\.edu$/,
    // Add personal email domains that teachers commonly use
    /@gmail\.com$/,
    /@yahoo\.com$/,
    /@outlook\.com$/,
    /@hotmail\.com$/
  ];
  return pesEmailPatterns.some(pattern => pattern.test(email));
};

// Helper function to check if email is a student email
const isValidStudentEmail = (email) => {
  return email.endsWith('@pes.edu') || email.endsWith('@pesu.pes.edu');
};

// Routes
app.get('/', (req, res) => {
  res.json({ message: 'Welcome to PESUprep API!' });
});

// Add this route to handle OAuth registration
app.post('/api/oauth/register', async (req, res) => {
  try {
    const { email, name, role } = req.body;
    
    // Validate email based on role
    if (role === 'student') {
      if (!isValidStudentEmail(email)) {
        return res.status(400).json({ 
          message: 'Students must use their PES University email address'
        });
      }
    } else if (role === 'teacher') {
      if (!isValidTeacherEmail(email)) {
        return res.status(400).json({ 
          message: 'Please use a valid email address (PES University or personal email)'
        });
      }
    }

    // Check if user exists
    let user = await User.findOne({ email });
    
    if (user) {
      // Update existing user's role if needed
      if (user.role !== role) {
        user.role = role;
        await user.save();
      }
    } else {
      // Create new user
      user = new User({
        email,
        name,
        role,
        password: Math.random().toString(36), // Generate random password for OAuth users
      });
      await user.save();
    }

    // Create token
    const token = jwt.sign(
      { userId: user._id },
      'your-secret-key',
      { expiresIn: '24h' }
    );

    res.status(201).json({ token, user: { email, name, role } });
  } catch (error) {
    console.error('Registration error:', error);
    res.status(500).json({ message: 'Error registering user' });
  }
});

// Update the OAuth configuration
const GOOGLE_CLIENT_ID = '245320520839-rl82ksu9ic4s9skadnei2tdmlbhlocvf.apps.googleusercontent.com';
const GOOGLE_CLIENT_SECRET = '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';
const REDIRECT_URI = 'http://localhost:3000/auth/callback';  // Updated to match Google's expected URI

// Add this route to handle OAuth callback
app.post('/api/auth/google/callback', async (req, res) => {
  try {
    const { code, role } = req.body;
    
    // Create OAuth2 client
    const oauth2Client = new google.auth.OAuth2(
      GOOGLE_CLIENT_ID,
      GOOGLE_CLIENT_SECRET,
      REDIRECT_URI
    );

    // Get tokens
    const { tokens } = await oauth2Client.getToken(code);
    oauth2Client.setCredentials(tokens);

    // Get user info
    const oauth2 = google.oauth2('v2');
    const userInfo = await oauth2.userinfo.get({ auth: oauth2Client });

    // Create or update user with role
    const user = await User.findOneAndUpdate(
      { email: userInfo.data.email },
      {
        email: userInfo.data.email,
        name: userInfo.data.name,
        picture: userInfo.data.picture,
        role: role || 'student',
        googleId: userInfo.data.id
      },
      { upsert: true, new: true }
    );

    // Generate JWT token
    const token = jwt.sign(
      { userId: user._id, role: user.role },
      'your-secret-key',
      { expiresIn: '24h' }
    );

    res.json({ user, token });
  } catch (error) {
    console.error('Auth callback error:', error);
    res.status(500).json({ message: 'Authentication failed', error: error.message });
  }
});

// Add course enrollment endpoint
app.post('/api/enroll', async (req, res) => {
  try {
    const { courseId, courseName } = req.body;
    const token = req.headers.authorization?.split(' ')[1];
    
    if (!token) {
      return res.status(401).json({ message: 'No token provided' });
    }

    const decoded = jwt.verify(token, 'your-secret-key');
    const user = await User.findById(decoded.userId);

    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }

    // Check if already enrolled
    const isEnrolled = user.courses.some(course => course.courseId === courseId);
    if (!isEnrolled) {
      user.courses.push({ courseId, courseName });
      await user.save();
    }

    res.json({ message: 'Enrollment successful', courses: user.courses });
  } catch (error) {
    console.error('Enrollment error:', error);
    res.status(500).json({ message: 'Error enrolling in course' });
  }
});

// Configure multer for file upload
const storage = multer.memoryStorage();
const upload = multer({ 
  storage: storage,
  limits: {
    fileSize: 10 * 1024 * 1024 // 10MB limit
  }
});

// Upload course content
app.post('/api/teacher/content/upload', upload.single('file'), async (req, res) => {
  try {
    const { courseId } = req.body;
    const token = req.headers.authorization?.split(' ')[1];
    
    if (!token) {
      return res.status(401).json({ message: 'No token provided' });
    }

    const decoded = jwt.verify(token, 'your-secret-key');
    const teacher = await User.findById(decoded.userId);

    if (!teacher || teacher.role !== 'teacher') {
      return res.status(403).json({ message: 'Not authorized' });
    }

    // Process uploaded file
    let text;
    if (req.file.mimetype === 'application/pdf') {
      const data = await pdf(req.file.buffer);
      text = data.text;
    } else {
      return res.status(400).json({ message: 'Unsupported file type' });
    }

    // Generate questions from the uploaded content
    const generatedQuestions = questionGenerator.generate_questions(text, 'General', 5);

    // Save questions to the database
    const questions = await Question.insertMany(generatedQuestions.map(q => ({
      ...q,
      courseId,
      status: 'pending'
    })));

    res.json({ questions });
  } catch (error) {
    console.error('Upload error:', error);
    res.status(500).json({ message: 'Error uploading content' });
  }
});

// Load MPT-7B model and tokenizer
// const tokenizer = AutoTokenizer.from_pretrained("mosaicml/mpt-7b");
// const model = AutoModelForCausalLM.from_pretrained("mosaicml/mpt-7b");
// model.to('cpu');  // Use 'cuda' if you have a GPU

// Generate questions using MPT-7B
app.post('/api/teacher/questions/generate', async (req, res) => {
    try {
        const { text, topic, count } = req.body;

        // Generate questions using MPT-7B
        const questions = [];
        for (let i = 0; i < count; i++) {
            const prompt = `Generate a question about ${topic} based on the following text:\n${text}\nQuestion:`;
            const inputs = tokenizer(prompt, return_tensors="pt").to('cpu');
            const outputs = model.generate(inputs.input_ids, max_length=150, num_return_sequences=1);
            const question_text = tokenizer.decode(outputs[0], skip_special_tokens=True);

            // Parse question and options
            const [question, ...options] = question_text.split('\n');
            questions.push({
                text: question,
                options: options.slice(0, 4),  // Assume first 4 lines are options
                correct_answer: 0,  // Assume first option is correct
                type: 'mcq',
                difficulty: 0.5
            });
        }

        res.json({ questions });
    } catch (error) {
        console.error('Question generation error:', error);
        res.status(500).json({ message: 'Error generating questions' });
    }
});

// Review and approve/reject questions
app.post('/api/teacher/questions/review', async (req, res) => {
  try {
    const { questionId, status, feedback } = req.body;
    const token = req.headers.authorization?.split(' ')[1];
    
    const decoded = jwt.verify(token, 'your-secret-key');
    const teacher = await User.findById(decoded.userId);

    if (!teacher || teacher.role !== 'teacher') {
      return res.status(403).json({ message: 'Not authorized' });
    }

    const question = await Question.findById(questionId);
    question.status = status;
    question.feedback = feedback;
    question.approvedBy = teacher._id;
    await question.save();

    res.json({ message: 'Question reviewed successfully' });
  } catch (error) {
    console.error('Review error:', error);
    res.status(500).json({ message: 'Error reviewing question' });
  }
});

// Get student progress
app.get('/api/teacher/students/progress', async (req, res) => {
  try {
    const { courseId } = req.query;
    const progress = await StudentProgress.find({ courseId })
      .populate('studentId', 'name email')
      .sort('-lastUpdated');

    res.json({ progress });
  } catch (error) {
    console.error('Progress fetch error:', error);
    res.status(500).json({ message: 'Error fetching student progress' });
  }
});

// Get teacher's courses
app.get('/api/teacher/courses', async (req, res) => {
  try {
    const token = req.headers.authorization?.split(' ')[1];
    if (!token) {
      return res.status(401).json({ message: 'No token provided' });
    }

    const decoded = jwt.verify(token, 'your-secret-key');
    const teacher = await User.findById(decoded.userId);

    if (!teacher || teacher.role !== 'teacher') {
      return res.status(403).json({ message: 'Not authorized' });
    }

    // Return only ML and Linear Algebra courses
    const courses = [
      { 
        id: 'UE22CS352A', 
        name: 'Machine Learning',
        topics: ['Introduction to ML', 'Supervised Learning', 'Neural Networks']
      },
      { 
        id: 'UE22MA241B', 
        name: 'Linear Algebra',
        topics: ['Matrices & Determinants', 'Vector Spaces', 'Eigenvalues']
      }
    ];

    res.json({ courses });
  } catch (error) {
    console.error('Error fetching courses:', error);
    res.status(500).json({ message: 'Error fetching courses' });
  }
});

// Add this endpoint to handle doc/docx conversion
app.post('/api/convert-doc', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ message: 'No file uploaded' });
    }

    let text;
    if (req.file.mimetype === 'application/msword' || 
        req.file.mimetype === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') {
      const result = await mammoth.extractRawText({ buffer: req.file.buffer });
      text = result.value;
    } else {
      return res.status(400).json({ message: 'Unsupported file type' });
    }

    res.json({ text });
  } catch (error) {
    console.error('Document conversion error:', error);
    res.status(500).json({ message: 'Error converting document' });
  }
});

// Add this endpoint to save questions
app.post('/api/courses/:courseId/questions', async (req, res) => {
  try {
    const { courseId } = req.params;
    const { questions } = req.body;
    
    // Save questions to database
    const savedQuestions = await Question.insertMany(
      questions.map(q => ({
        ...q,
        courseId,
        createdAt: new Date()
      }))
    );
    
    res.json({ questions: savedQuestions });
  } catch (error) {
    console.error('Error saving questions:', error);
    res.status(500).json({ message: 'Error saving questions' });
  }
});

// Add content processing endpoint with proper error handling
app.post('/api/process-content', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ message: 'No file uploaded' });
    }

    console.log('Processing file:', req.file.originalname, 'Type:', req.file.mimetype);

    let content = '';
    const fileType = req.file.mimetype;

    switch (fileType) {
      case 'text/plain':
        content = req.file.buffer.toString('utf-8');
        break;
      
      case 'application/pdf':
        try {
          const pdfData = await pdf(req.file.buffer);
          content = pdfData.text;
        } catch (pdfError) {
          console.error('PDF processing error:', pdfError);
          return res.status(500).json({ 
            message: 'Error processing PDF file. Please ensure it is not corrupted.' 
          });
        }
        break;
      
      case 'application/msword':
      case 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        try {
          const result = await mammoth.extractRawText({ buffer: req.file.buffer });
          content = result.value;
        } catch (docError) {
          console.error('DOC processing error:', docError);
          return res.status(500).json({ 
            message: 'Error processing DOC/DOCX file. Please ensure it is not corrupted.' 
          });
        }
        break;
      
      default:
        return res.status(400).json({ 
          message: `Unsupported file type: ${fileType}. Please upload a PDF, DOC, DOCX, or TXT file.` 
        });
    }

    if (!content.trim()) {
      return res.status(400).json({ 
        message: 'The file appears to be empty or could not be processed.' 
      });
    }

    console.log('Content processed successfully');
    res.json({ content });

  } catch (error) {
    console.error('Content processing error:', error);
    res.status(500).json({ 
      message: 'Error processing file: ' + (error.message || 'Unknown error') 
    });
  }
});

// Add dataset upload endpoint
app.post('/api/upload-dataset', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      console.error('No file uploaded');
      return res.status(400).json({ message: 'No file uploaded' });
    }

    console.log('File uploaded:', req.file.originalname, 'Type:', req.file.mimetype);

    // Process the uploaded file
    const fileType = req.file.mimetype;
    let content = '';

    switch (fileType) {
      case 'application/pdf':
        try {
          const pdfData = await pdf(req.file.buffer);
          content = pdfData.text;
        } catch (pdfError) {
          console.error('PDF processing error:', pdfError);
          return res.status(500).json({ message: 'Error processing PDF file. Please ensure it is not corrupted.' });
        }
        break;
      case 'text/plain':
        content = req.file.buffer.toString('utf-8');
        break;
      default:
        console.error('Unsupported file type:', fileType);
        return res.status(400).json({ message: 'Unsupported file type' });
    }

    if (!content.trim()) {
      console.error('File appears to be empty or could not be processed');
      return res.status(400).json({ message: 'The file appears to be empty or could not be processed.' });
    }

    console.log('Content processed successfully');
    res.json({ message: 'File uploaded successfully' });
  } catch (error) {
    console.error('Error processing file:', error);
    res.status(500).json({ message: 'Error processing file: ' + (error.message || 'Unknown error') });
  }
});

// Add this route to save approved questions
app.post('/api/courses/:courseId/save-questions', async (req, res) => {
  try {
    const { courseId } = req.params;
    const { questions, topic } = req.body;

    if (!questions || !Array.isArray(questions)) {
      return res.status(400).json({ message: 'Invalid questions data' });
    }

    // Map the questions to match our Question schema
    const questionDocs = questions.map(q => ({
      courseId,
      topic,
      questionText: q.text,
      options: q.type === 'mcq' ? q.options.map((opt, index) => ({
        text: opt,
        isCorrect: index === q.correct_answer
      })) : [],
      difficulty: q.difficulty || 0.5,
      type: q.type,
      status: 'approved',
      metadata: {
        topicDifficulty: q.difficulty || 0.5,
        conceptsCovered: [topic],
        timeToSolve: 2 // default 2 minutes per question
      }
    }));

    // Save questions to database
    const savedQuestions = await Question.insertMany(questionDocs);

    // Create a new question set
    const questionSet = new QuestionSet({
      courseId,
      type: 'quiz',
      title: `${topic} Quiz`,
      questions: savedQuestions.map(q => q._id),
      difficulty: savedQuestions.reduce((acc, q) => acc + q.difficulty, 0) / savedQuestions.length,
      totalMarks: savedQuestions.length,
      duration: savedQuestions.length * 2, // 2 minutes per question
      status: 'approved'
    });

    await questionSet.save();

    res.json({ 
      message: 'Questions saved successfully',
      questions: savedQuestions,
      questionSet
    });

  } catch (error) {
    console.error('Error saving approved questions:', error);
    res.status(500).json({ 
      message: 'Error saving questions',
      error: error.message 
    });
  }
});
app.post('/api/generate-questions', async (req, res) => {
  try {
    const { text } = req.body;
    const response = await axios.post('http://localhost:8000/generate-questions', { text });
    res.json(response.data);
  } catch (error) {
    console.error('Error generating questions:', error);
    res.status(500).json({ message: 'Error generating questions' });
  }
});
app.post('/api/upload-dataset', async (req, res) => {
  try {
    const formData = new FormData();
    req.files.forEach(file => {
      formData.append('file', file.buffer, file.originalname);
    });

    const response = await axios.post('http://localhost:8000/upload-dataset', formData, {
      headers: formData.getHeaders()
    });

    res.json(response.data);
  } catch (error) {
    console.error('Error uploading dataset:', error);
    res.status(500).json({ message: 'Error uploading dataset' });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
}); 