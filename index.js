require('dotenv').config();
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
const tesseract = require('node-tesseract-ocr');
const natural = require('natural');
const tokenizer = new natural.WordTokenizer();
const passport = require('passport');
const session = require('express-session');
const authRoutes = require('./routes/auth.routes');
const paperRoutes = require('./routes/papers');
const scheduleRoutes = require('./routes/schedules');
const diagnosticRoutes = require('./routes/diagnostic');
const Schedule = require('./models/Schedule');
const StudentProgress = require('./models/StudentProgress');
const DBMSQuestion = require('./models/DBMSQuestion');
const MLQuestion = require('./models/MLQuestion');
const PlayerStats = require('./models/PlayerStats');

const app = express();

// Middleware
app.use(cors({
  origin: 'http://localhost:3000',
  credentials: true
}));
// Increase JSON and urlencoded body limits to handle larger payloads (e.g., uploads/answers)
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));
app.use(session({
  secret: process.env.SESSION_SECRET || 'your-session-secret',
  resave: false,
  saveUninitialized: false
}));

// Initialize Passport
app.use(passport.initialize());
app.use(passport.session());

// Mount routes
app.use('/api/auth', authRoutes);
app.use('/api/papers', paperRoutes);
app.use('/api/schedules', scheduleRoutes);
app.use('/api/diagnostic', diagnosticRoutes);

// Set Mongoose strictQuery option
mongoose.set('strictQuery', true);

// MongoDB Connection
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/pesuprep';

// Initialize collections
let db;

mongoose.connect(MONGODB_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
  serverSelectionTimeoutMS: 30000,  // Increased from 5000
  socketTimeoutMS: 90000,  // Increased from 45000
  retryWrites: true,
  retryReads: true,
  w: 'majority'
})
.then(async () => {
  console.log('Connected to MongoDB successfully');
  db = mongoose.connection.db;
  
  // Ensure the questions collections exist
  const collections = ['questions', 'dbmsquestions', 'mlquestions'];
  for (const collectionName of collections) {
    try {
      await db.createCollection(collectionName);
      console.log(`${collectionName} collection initialized`);
    } catch (e) {
      if (e.code !== 48) { // 48 is collection already exists error
        console.error(`Error creating ${collectionName} collection:`, e);
      } else {
        console.log(`${collectionName} collection already exists`);
      }
    }
  }
})
.catch(err => {
  console.error('MongoDB connection error:', err);
  process.exit(1);
});

// Add this right after MongoDB connection setup
mongoose.connection.on('error', (err) => {
  console.error('MongoDB connection error:', err);
});

mongoose.connection.on('connected', () => {
  console.log('MongoDB connected successfully');
});

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

// Add this schema for storing processed content
const processedContentSchema = new mongoose.Schema({
  courseId: String,
  title: String,
  content: String,
  summary: String,
  keyTerms: [String],
  topics: [String],
  processedAt: { type: Date, default: Date.now },
  fileType: String,
  originalFileName: String
});

const ProcessedContent = mongoose.model('ProcessedContent', processedContentSchema);

// Add this schema for paper templates
const paperTemplateSchema = new mongoose.Schema({
  courseId: String,
  title: String,
  totalMarks: Number,
  duration: Number,
  sections: [{
    topic: String,
    numQuestions: Number,
    marks: Number,
    difficulty: String,
    questionTypes: [String],
    cognitiveLevel: String
  }],
  createdBy: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
  createdAt: { type: Date, default: Date.now }
});

const PaperTemplate = mongoose.model('PaperTemplate', paperTemplateSchema);

// Update QuestionTemplate Schema
const questionTemplateSchema = new mongoose.Schema({
  courseId: String,
  questionText: String,
  options: [{
    text: String,
    isCorrect: Boolean
  }],
  type: String,  // 'mcq' or 'descriptive'
  difficulty: {
    type: String,
    enum: ['easy', 'moderate', 'hard'],
    default: 'moderate'
  },
  status: {
    type: String,
    enum: ['approved', 'rejected'],
    default: 'approved'
  },
  suggestedPoints: [String],  // For descriptive questions
  explanation: String,
  createdAt: { type: Date, default: Date.now },
  metadata: {
    topicDifficulty: String,
    conceptsCovered: [String],
    timeToSolve: Number
  }
});

const QuestionTemplate = mongoose.model('QuestionTemplate', questionTemplateSchema);

// Course Schema
const courseSchema = new mongoose.Schema({
  id: String,
  name: String,
  isas: [{
    id: mongoose.Schema.Types.ObjectId,
    date: Date,
    time: String
  }],
  quizzes: [{
    id: mongoose.Schema.Types.ObjectId,
    title: String,
    date: Date
  }],
  papers: [{
    id: mongoose.Schema.Types.ObjectId,
    name: String,
    uploadedAt: Date
  }]
});

const Course = mongoose.model('Course', courseSchema);

const User = mongoose.model('User', userSchema);
const Question = require('./models/Question');
const QuestionSet = require('./models/QuestionSet');

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

// Mount routes
app.use('/auth', authRoutes);
app.use('/papers', paperRoutes);
app.use('/schedules', scheduleRoutes);
app.use('/diagnostic', diagnosticRoutes);

// Player stats persistence
app.get('/player-stats/:email', async (req, res) => {
  try {
    const email = (req.params.email || '').toLowerCase();
    if (!email) return res.status(400).json({ error: 'email required' });
    const doc = await PlayerStats.findOne({ email });
    return res.json({ stats: doc?.stats || null });
  } catch (e) {
    console.error('get player stats error', e);
    return res.status(500).json({ error: 'failed to get player stats' });
  }
});

app.post('/player-stats', async (req, res) => {
  try {
    const { email, stats } = req.body || {};
    if (!email || !stats) return res.status(400).json({ error: 'email and stats required' });
    const key = email.toLowerCase();
    const doc = await PlayerStats.findOneAndUpdate(
      { email: key },
      { email: key, stats, updatedAt: new Date() },
      { new: true, upsert: true }
    );
    return res.json({ ok: true, stats: doc.stats });
  } catch (e) {
    console.error('save player stats error', e);
    return res.status(500).json({ error: 'failed to save player stats' });
  }
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

// Google OAuth Configuration
const GOOGLE_CLIENT_ID = '245320520839-rl82ksu9ic4s9skadnei2tdmlbhlocvf.apps.googleusercontent.com';
const GOOGLE_CLIENT_SECRET = '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';
const REDIRECT_URI = 'http://localhost:5000/auth/callback';

// Create OAuth2 client
const oauth2Client = new OAuth2Client(
  GOOGLE_CLIENT_ID,
  GOOGLE_CLIENT_SECRET,
  REDIRECT_URI
);

// Configure Multer for file uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    // Make sure the uploads directory exists
    if (!fs.existsSync('./uploads')) {
      fs.mkdirSync('./uploads', { recursive: true });
    }
    cb(null, './uploads/');
  },
  filename: function (req, file, cb) {
    // Create a safe filename
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, uniqueSuffix + '-' + file.originalname);
  }
});

const fileFilter = (req, file, cb) => {
  console.log('Received file:', file.originalname, 'Type:', file.mimetype);
  
  // Accept these file types
  const allowedTypes = [
    'application/pdf',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'text/plain',
    'image/jpeg',
    'image/png',
    'application/vnd.ms-powerpoint',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation'
  ];

  if (allowedTypes.includes(file.mimetype)) {
    cb(null, true);
  } else {
    console.log('Rejected file type:', file.mimetype);
    cb(new Error(`File type ${file.mimetype} not allowed. Allowed types: PDF, DOC, DOCX, TXT, PPT, PPTX, JPG, PNG`), false);
  }
};

const upload = multer({
  storage: storage,
  fileFilter: fileFilter,
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
    files: 10 // Maximum 10 files
  }
});

// Update the process-content endpoint
app.post('/process-content', (req, res) => {
  upload.array('files[]', 10)(req, res, async (err) => {
    try {
      console.log('Processing content request...');
      
      if (err instanceof multer.MulterError) {
        console.error('Multer error:', err);
        return res.status(400).json({ 
          success: false, 
          error: `File upload error: ${err.message}` 
        });
      } else if (err) {
        console.error('Upload error:', err);
        return res.status(500).json({ 
          success: false, 
          error: err.message 
        });
      }

      if (!req.files || req.files.length === 0) {
        console.log('No files received');
        return res.status(400).json({
          success: false,
          error: 'No files provided'
        });
      }

      console.log(`Processing ${req.files.length} files...`);
      const processedFiles = [];

      for (const file of req.files) {
        try {
          console.log(`Processing file: ${file.originalname}`);
          
          // Extract text based on file type
          let text = '';
          if (file.mimetype === 'application/pdf') {
            const dataBuffer = fs.readFileSync(file.path);
            const pdfData = await pdf(dataBuffer);
            text = pdfData.text;
          } else if (file.mimetype === 'text/plain') {
            text = fs.readFileSync(file.path, 'utf-8');
          } else if (file.mimetype.includes('word')) {
            const result = await mammoth.extractRawText({ path: file.path });
            text = result.value;
          } else if (file.mimetype.includes('powerpoint')) {
            // Handle PowerPoint files
            text = await extractTextFromPowerPoint(file.path);
          }

          console.log(`Extracted ${text.length} characters from ${file.originalname}`);

          // Generate simple summary
          const summary = text.length > 200 ? text.substring(0, 200) + '...' : text;

          processedFiles.push({
            filename: file.originalname,
            text: text,
            summary: summary,
            keyTerms: []
          });

          // Clean up uploaded file
          fs.unlinkSync(file.path);
          console.log(`Successfully processed ${file.originalname}`);

        } catch (error) {
          console.error(`Error processing file ${file.originalname}:`, error);
          // Continue with other files even if one fails
        }
      }

      if (processedFiles.length === 0) {
        return res.status(400).json({
          success: false,
          error: 'No files were successfully processed'
        });
      }

      console.log(`Successfully processed ${processedFiles.length} files`);
      res.json({
        success: true,
        processed_files: processedFiles
      });

    } catch (error) {
      console.error('Error in process-content:', error);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });
});

// Simplified PowerPoint text extraction
async function extractTextFromPowerPoint(filePath) {
  try {
    // For now, return a simple message since PowerPoint processing is complex
    return 'PowerPoint content processing is not available at the moment.';
  } catch (error) {
    console.error('Error extracting text from PowerPoint:', error);
    return '';
  }
}

// Handle Google OAuth callback
app.post('/auth/callback', async (req, res) => {
  try {
    const { code } = req.body;
    if (!code) {
      return res.status(400).json({ message: 'Authorization code is required' });
    }

    // Get tokens
    const { tokens } = await oauth2Client.getToken(code);
    oauth2Client.setCredentials(tokens);

    // Get user info
    const oauth2 = google.oauth2('v2');
    const userInfo = await oauth2.userinfo.get({ auth: oauth2Client });

    // Create or update user
    const user = await User.findOneAndUpdate(
      { email: userInfo.data.email },
      {
        email: userInfo.data.email,
        name: userInfo.data.name,
        picture: userInfo.data.picture,
        lastLogin: new Date()
      },
      { upsert: true, new: true }
    );

    // Generate JWT token
    const token = jwt.sign(
      { userId: user._id, email: user.email },
      'your-secret-key',
      { expiresIn: '24h' }
    );

    res.json({ user, token });
  } catch (error) {
    console.error('Auth callback error:', error);
    res.status(400).json({ 
      message: 'Authentication failed',
      error: error.message
    });
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

// Then update the route to use the correct multer middleware
app.post('/api/teacher/content/upload', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ message: 'No file uploaded' });
    }

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
    let text = '';
    if (req.file.mimetype === 'application/pdf') {
      const dataBuffer = fs.readFileSync(req.file.path);
      const data = await pdf(dataBuffer);
      text = data.text;
    } else if (req.file.mimetype.startsWith('image/')) {
      const config = {
        lang: "eng",
        oem: 1,
        psm: 3,
      };
      text = await tesseract.recognize(req.file.path, config);
    } else if (req.file.mimetype === 'text/plain') {
      text = fs.readFileSync(req.file.path, 'utf-8');
    } else if (req.file.mimetype.includes('word')) {
      const result = await mammoth.extractRawText({ path: req.file.path });
      text = result.value;
    }

    // Generate summary using our simple function
    const summary = generateSummary(text);

    // Extract key terms
    const tokens = tokenizer.tokenize(text.toLowerCase());
    const keyTerms = new Set();
    tokens.forEach(token => {
      if (token.length > 3) keyTerms.add(token);
    });

    // Save processed content
    const processedContent = new ProcessedContent({
      courseId,
      content: text,
      summary,
      keyTerms: Array.from(keyTerms).slice(0, 10),
      fileType: req.file.mimetype,
      originalFileName: req.file.originalname
    });
    await processedContent.save();

    // Clean up uploaded file
    fs.unlinkSync(req.file.path);

    res.json({ 
      message: 'File processed successfully',
      summary,
      keyTerms: Array.from(keyTerms).slice(0, 10)
    });

  } catch (error) {
    console.error('Upload error:', error);
    res.status(500).json({ message: 'Error processing file: ' + error.message });
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

// Update the question review endpoint
app.post('/api/questions/review', async (req, res) => {
  try {
    console.log('Received question review request:', req.body);
    const { questionId, status, feedback, questionData } = req.body;

    // Get questions collection
    const questions = db.collection("questions");

    if (questionData) {
      // New question being reviewed
      const questionDoc = {
        ...questionData,
        status: status || 'pending',
        feedback: feedback || '',
        createdAt: new Date(),
        reviewedAt: new Date(),
        _id: new mongoose.Types.ObjectId()
      };

      console.log('Saving new question:', questionDoc);
      const result = await questions.insertOne(questionDoc);
      console.log('Question saved with ID:', result.insertedId);

      return res.json({
        success: true,
        message: 'Question saved successfully',
        question: { ...questionDoc, _id: result.insertedId }
      });
    }

    // Updating existing question
    if (!questionId) {
      return res.status(400).json({
        success: false,
        message: 'Question ID is required for updates'
      });
    }

    const updateResult = await questions.findOneAndUpdate(
      { _id: new mongoose.Types.ObjectId(questionId) },
      {
        $set: {
          status: status,
          feedback: feedback,
          reviewedAt: new Date()
        }
      },
      { returnDocument: 'after' }
    );

    if (!updateResult.value) {
      return res.status(404).json({
        success: false,
        message: 'Question not found'
      });
    }

    return res.json({
      success: true,
      message: 'Question updated successfully',
      question: updateResult.value
    });

  } catch (error) {
    console.error('Error in question review:', error);
    return res.status(500).json({
      success: false,
      message: 'Failed to process question review',
      error: error.message
    });
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
    if (req.file.mimetring === 'application/msword' || 
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
    
    // Save questions to QuestionTemplate collection
    const savedQuestions = await QuestionTemplate.insertMany(
      questions.map(q => ({
        ...q,
        courseId,
        createdAt: new Date(),
        status: q.status || 'pending'
      }))
    );
    
    res.json({ questions: savedQuestions });
  } catch (error) {
    console.error('Error saving questions:', error);
    res.status(500).json({ message: 'Error saving questions' });
  }
});

// Add natural language processing helper function
const generateSummary = (text, maxLength = 150) => {
  // Simple extractive summarization
  const sentences = text.match(/[^.!?]+[.!?]+/g) || [];
  const topSentences = sentences.slice(0, 3); // Take first 3 sentences
  return topSentences.join(' ').substring(0, maxLength) + '...';
};

// Add this helper function for domain-specific term extraction
const extractDomainSpecificTerms = (text, courseId) => {
  // Dictionary of domain-specific terms
  const domainTerms = {
    'UE22CS352A': [ // Machine Learning course
      'algorithm', 'neural network', 'supervised learning', 'unsupervised learning',
      'classification', 'regression', 'clustering', 'deep learning', 'training',
      'validation', 'test set', 'overfitting', 'underfitting', 'bias', 'variance',
      'gradient descent', 'backpropagation', 'activation function', 'epoch', 'batch',
      'feature', 'label', 'model', 'prediction', 'accuracy', 'precision', 'recall'
    ],
    'UE22MA241B': [ // Linear Algebra course
      'matrix', 'vector', 'eigenvalue', 'eigenvector', 'determinant', 'linear',
      'transformation', 'basis', 'dimension', 'rank', 'nullspace', 'orthogonal',
      'projection', 'inverse', 'transpose', 'system', 'equation', 'span', 'subspace'
    ]
  };

  // Default to ML terms if courseId not provided
  const terms = domainTerms[courseId] || domainTerms['UE22CS352A'];
  const foundTerms = new Set();
  
  if (!text) return [];
  
  const textLower = text.toLowerCase();
  terms.forEach(term => {
    if (textLower.includes(term.toLowerCase())) {
      foundTerms.add(term);
    }
  });

  return Array.from(foundTerms);
};

// Update the content processing endpoint to use domain-specific extraction
app.post('/api/process-content', (req, res) => {
  upload.array('files', 10)(req, res, async (err) => {
    if (err instanceof multer.MulterError) {
      return res.status(400).json({ message: 'File upload error: ' + err.message });
    } else if (err) {
      return res.status(500).json({ message: 'Server error: ' + err.message });
    }

    try {
      if (!req.files || req.files.length === 0) {
        return res.status(400).json({ message: 'No files uploaded' });
      }

      const { courseId } = req.body;
      const processedResults = [];
      
      for (const file of req.files) {
        try {
    let content = '';
          
          // Process different file types
          if (file.mimetype.startsWith('image/')) {
            const config = {
              lang: "eng",
              oem: 1,
              psm: 3,
            };
            content = await tesseract.recognize(file.path, config);
          } else {
            switch (file.mimetype) {
      case 'application/pdf':
                const pdfData = await pdf(fs.readFileSync(file.path));
          content = pdfData.text;
        break;
              case 'text/plain':
                content = fs.readFileSync(file.path, 'utf-8');
                break;
      case 'application/msword':
      case 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                const result = await mammoth.extractRawText({ path: file.path });
          content = result.value;
        break;
            }
          }

          // Generate summary
          const summary = generateSummary(content);

          // Extract domain-specific terms
          const keyTerms = extractDomainSpecificTerms(content, courseId);

          // Save processed content
          const processedContent = new ProcessedContent({
            courseId,
            content,
            summary,
            keyTerms,
            fileType: file.mimetype,
            originalFileName: file.originalname
          });
          await processedContent.save();

          processedResults.push({
            filename: file.originalname,
            content, // Include content for question generation
            summary,
            keyTerms
          });

        } catch (fileError) {
          console.error(`Error processing file ${file.originalname}:`, fileError);
          processedResults.push({
            filename: file.originalname,
            error: `Failed to process file: ${fileError.message}`
          });
        } finally {
          // Clean up uploaded file
          fs.unlink(file.path, (err) => {
            if (err) console.error('Error deleting file:', err);
          });
        }
      }

      res.json({
        message: 'Content processed successfully',
        results: processedResults
      });

  } catch (error) {
    console.error('Content processing error:', error);
      res.status(500).json({ message: 'Error processing content: ' + error.message });
  }
  });
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

    await questiontemplates.save();

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

// Update the paper generation endpoint
app.post('/api/generate-paper', async (req, res) => {
  try {
    const { courseId, config } = req.body;
    
    // Get questions from questions collection
    const questions = db.collection("questions");
    const availableQuestions = await questions.find({
      courseId: courseId,
      status: 'approved'
    }).toArray();

    if (availableQuestions.length === 0) {
      return res.status(400).json({ 
        message: 'No approved questions found for this course' 
      });
    }

    // Get all existing papers to check for question usage
    const existingPapers = await QuestionSet.find({ 
      courseId: courseId,
      type: 'paper'
    }, 'questions');

    // Get all questions that have been used in papers
    const usedQuestionIds = new Set(
      existingPapers.flatMap(paper => paper.questions.map(q => q.toString()))
    );

    // Filter out previously used questions
    const unusedQuestions = availableQuestions.filter(q => !usedQuestionIds.has(q._id.toString()));

    if (unusedQuestions.length === 0) {
      return res.status(400).json({ 
        message: 'No unused approved questions available for this course' 
      });
    }

    // Calculate number of questions needed for each type and difficulty
    const mcqCount = Math.floor((config.questionTypes.mcq / 100) * unusedQuestions.length);
    const descriptiveCount = Math.floor((config.questionTypes.descriptive / 100) * unusedQuestions.length);

    // Filter questions by type
    const mcqs = unusedQuestions.filter(q => q.type === 'mcq');
    const descriptive = unusedQuestions.filter(q => q.type === 'descriptive');

    // Helper function to filter by difficulty
    const filterByDifficulty = (questions, difficultyLevel) => {
      return questions.filter(q => {
        const diff = q.difficulty;
        if (difficultyLevel === 'easy') return diff <= 0.4;
        if (difficultyLevel === 'moderate') return diff > 0.4 && diff <= 0.7;
        return diff > 0.7;
      });
    };

    // Select questions based on configuration
    const selectedQuestions = [];

    // Select MCQs
    if (mcqs.length > 0) {
      const easyMCQs = filterByDifficulty(mcqs, 'easy');
      const moderateMCQs = filterByDifficulty(mcqs, 'moderate');
      const challengingMCQs = filterByDifficulty(mcqs, 'challenging');

      const easyCount = Math.floor(mcqCount * (config.difficulty.easy / 100));
      const moderateCount = Math.floor(mcqCount * (config.difficulty.moderate / 100));
      const challengingCount = Math.floor(mcqCount * (config.difficulty.challenging / 100));

      selectedQuestions.push(
        ...easyMCQs.slice(0, easyCount),
        ...moderateMCQs.slice(0, moderateCount),
        ...challengingMCQs.slice(0, challengingCount)
      );
    }

    // Select Descriptive questions
    if (descriptive.length > 0) {
      const easyDesc = filterByDifficulty(descriptive, 'easy');
      const moderateDesc = filterByDifficulty(descriptive, 'moderate');
      const challengingDesc = filterByDifficulty(descriptive, 'challenging');

      const easyCount = Math.floor(descriptiveCount * (config.difficulty.easy / 100));
      const moderateCount = Math.floor(descriptiveCount * (config.difficulty.moderate / 100));
      const challengingCount = Math.floor(descriptiveCount * (config.difficulty.challenging / 100));

      selectedQuestions.push(
        ...easyDesc.slice(0, easyCount),
        ...moderateDesc.slice(0, moderateCount),
        ...challengingDesc.slice(0, challengingCount)
      );
    }

    // Shuffle the selected questions
    const shuffledQuestions = selectedQuestions.sort(() => Math.random() - 0.5);

    // Create new paper
    const paper = new QuestionSet({
      courseId,
      type: 'paper',
      title: config.title,
      questions: shuffledQuestions.map(q => q._id),
      totalMarks: config.totalMarks,
      duration: config.duration,
      status: 'draft',
      questionTypes: {
        mcq: shuffledQuestions.filter(q => q.type === 'mcq').length,
        descriptive: shuffledQuestions.filter(q => q.type === 'descriptive').length
      },
      difficulty: config.difficulty,
      createdAt: new Date()
    });

    await paper.save();

    // Update usedInPapers field for selected questions in the Question collection
    await Question.updateMany(
      { _id: { $in: shuffledQuestions.map(q => q._id) } },
      { $push: { usedInPapers: paper._id } }
    );

    res.json({
      message: 'Paper generated successfully',
      paper: {
        id: paper._id,
        title: paper.title,
        questionCount: shuffledQuestions.length,
        questions: shuffledQuestions,
        config: {
          totalMarks: config.totalMarks,
          duration: config.duration,
          questionTypes: paper.questionTypes,
          difficulty: config.difficulty
        }
      }
    });

  } catch (error) {
    console.error('Error generating paper:', error);
    res.status(500).json({ 
      message: 'Error generating paper', 
      error: error.message 
    });
  }
});

// Add endpoint for saving paper template
app.post('/api/paper-templates', async (req, res) => {
  try {
    const { courseId, template } = req.body;
    const token = req.headers.authorization?.split(' ')[1];
    
    if (!token) {
      return res.status(401).json({ message: 'No token provided' });
    }

    const decoded = jwt.verify(token, 'your-secret-key');
    const teacher = await User.findById(decoded.userId);

    if (!teacher || teacher.role !== 'teacher') {
      return res.status(403).json({ message: 'Not authorized' });
    }

    const paperTemplate = new PaperTemplate({
      ...template,
      courseId,
      createdBy: teacher._id
    });

    await paperTemplate.save();

    res.json({
      message: 'Paper template saved successfully',
      template: paperTemplate
    });

  } catch (error) {
    console.error('Error saving paper template:', error);
    res.status(500).json({ message: 'Error saving paper template' });
  }
});

// Add endpoint for getting paper templates
app.get('/api/paper-templates/:courseId', async (req, res) => {
  try {
    const { courseId } = req.params;
    const templates = await PaperTemplate.find({ courseId })
      .sort('-createdAt')
      .populate('createdBy', 'name');

    res.json({ templates });
  } catch (error) {
    console.error('Error fetching paper templates:', error);
    res.status(500).json({ message: 'Error fetching paper templates' });
  }
});

// Add endpoint for paper review (approval/rejection)
app.post('/api/papers/review', async (req, res) => {
  try {
    const { courseId, status, feedback, questions, timestamp } = req.body;

    // Validate required fields
    if (!courseId || !status || !questions) {
      return res.status(400).json({ 
        message: 'Missing required fields' 
      });
    }

    // Get the latest paper number for this course
    const latestPaper = await QuestionSet.findOne(
      { courseId, type: 'paper' },
      { paperNumber: 1 }
    ).sort({ paperNumber: -1 });

    const newPaperNumber = (latestPaper?.paperNumber || 0) + 1;

    // Create a new paper record in the database
    const paper = new QuestionSet({
      courseId,
      type: 'paper',
      paperNumber: newPaperNumber,
      title: `Machine Learning Paper ${newPaperNumber} - ${new Date().toLocaleDateString()}`,
      status: status,
      feedback: feedback,
      createdAt: timestamp,
      questionTypes: {
        mcq: questions.MCQs.length,
        descriptive: questions.Descriptive.length
      },
      difficulty: {
        easy: 30,
        moderate: 40,
        challenging: 30
      }
    });

    // Save the paper first
    await paper.save();
    console.log('Paper saved:', paper._id, 'Paper Number:', newPaperNumber);

    // If approved, save all questions as templates
    if (status === 'approved') {
      const savedQuestions = [];
      
      // Save MCQs to QuestionTemplate collection
      for (const mcq of questions.MCQs) {
        const template = new QuestionTemplate({
          courseId,
          questionText: mcq.text || mcq.question || mcq.questionText,
          options: Array.isArray(mcq.options) ? mcq.options.map(opt => ({
            text: typeof opt === 'object' ? opt.text : opt,
            isCorrect: mcq.answer === String.fromCharCode(65 + mcq.options.indexOf(opt))
          })) : [],
          type: 'mcq',
          difficulty: mcq.difficulty || 0.5,
          status: 'approved',
          explanation: mcq.explanation || '',
          metadata: {
            paperNumber: newPaperNumber,  // Add paper number to metadata
            topicDifficulty: mcq.difficulty || 0.5,
            conceptsCovered: [],
            timeToSolve: 2
          }
        });
        const savedTemplate = await template.save();
        savedQuestions.push(savedTemplate._id);
      }

      // Save Descriptive questions to QuestionTemplate collection
      for (const desc of questions.Descriptive) {
        const template = new QuestionTemplate({
          courseId,
          questionText: desc.text || desc.question || desc.questionText,
          type: 'descriptive',
          difficulty: desc.difficulty || 0.5,
          status: 'approved',
          suggestedPoints: desc.suggestedPoints || [],
          explanation: desc.explanation || '',
          metadata: {
            paperNumber: newPaperNumber,  // Add paper number to metadata
            topicDifficulty: desc.difficulty || 0.5,
            conceptsCovered: [],
            timeToSolve: 5
          }
        });
        const savedTemplate = await template.save();
        savedQuestions.push(savedTemplate._id);
      }

      // Update paper with saved question template IDs
      paper.questions = savedQuestions;
      await paper.save();
      console.log('Questions saved as templates and linked to paper');
    }

    res.json({
      message: `Paper ${newPaperNumber} ${status === 'approved' ? 'approved' : 'rejected'} successfully`,
      paper: {
        id: paper._id,
        paperNumber: newPaperNumber,
        title: paper.title,
        status: paper.status,
        questionTypes: paper.questionTypes,
        createdAt: paper.createdAt,
        savedQuestionCount: status === 'approved' ? paper.questions.length : 0
      }
    });

  } catch (error) {
    console.error('Error reviewing paper:', error);
    res.status(500).json({ 
      message: `Failed to ${req.body.status} paper`, 
      error: error.message 
    });
  }
});

// Update the save questions endpoint
app.post('/api/questions/save', async (req, res) => {
  try {
    console.log('Received question save request:', req.body);
    let questionsToSave = [];
    
    // Handle both single question and array of questions
    if (Array.isArray(req.body)) {
      questionsToSave = req.body;
    } else if (Array.isArray(req.body.questions)) {
      questionsToSave = req.body.questions;
    } else {
      // If single question object is received
      questionsToSave = [req.body];
    }

    if (questionsToSave.length === 0) {
      console.error('No valid questions data received');
      return res.status(400).json({
        success: false,
        message: 'No valid questions data'
      });
    }

    // Determine which collection to use based on courseId
    const firstQuestion = questionsToSave[0];
    const courseId = firstQuestion.courseId || '';
    const isDBMS = courseId.includes('351A') || courseId.toUpperCase().includes('DBMS');
    const isML = courseId.includes('ML') || courseId.toUpperCase().includes('MACHINE');
    
    // DBMS questions go to "dbmsquestions" collection, ML to "mlquestions"
    const QuestionModel = isDBMS ? DBMSQuestion : (isML ? MLQuestion : Question); // Default to generic Question
    const collectionName = isDBMS ? 'dbmsquestions' : (isML ? 'mlquestions' : 'questions');
    
    console.log(`Routing questions to ${collectionName} collection for courseId: ${courseId}`);

    // Prepare questions for saving
    const preparedQuestions = questionsToSave.map(question => {
      // Normalize question type
      const qType = question.type || (question.options ? 'mcq' : 'descriptive');
      const typeMap = {
        'TrueFalse': 'trueFalse',
        'FillBlank': 'fillInBlanks',
        'FillInBlanks': 'fillInBlanks',
        'MCQ': 'mcq',
        'Descriptive': 'descriptive'
      };
      const normalizedType = typeMap[qType] || qType.toLowerCase();
      
      // Convert difficulty string to number (1-10 scale)
      let normalizedDifficulty = 5; // Default moderate
      const diff = question.difficulty || 'moderate';
      if (typeof diff === 'string') {
        const difficultyMap = {
          'easy': 3,
          'moderate': 6,
          'hard': 9,
          'challenging': 9
        };
        normalizedDifficulty = difficultyMap[diff.toLowerCase()] || 5;
      } else if (typeof diff === 'number') {
        if (diff <= 1) {
          normalizedDifficulty = Math.round(diff * 10);
        } else if (diff > 10) {
          normalizedDifficulty = diff;
        } else {
          normalizedDifficulty = diff;
        }
      }
      
      // Extract correct answer from options
      const options = question.options ? question.options.map((opt, idx) => {
        const optText = typeof opt === 'string' ? opt : opt.text;
        const isCorrect = question.correct_answer === idx || 
                         question.answer === idx || 
                         question.answer === opt ||
                         (typeof opt === 'object' && opt.isCorrect) ||
                         (idx === 0 && !question.correct_answer && !question.answer);
        return {
          text: optText,
          isCorrect: isCorrect
        };
      }) : [];
      
      // Find correct answer index or value
      const correctAnswerIndex = options.findIndex(opt => opt.isCorrect);
      const correctAnswer = correctAnswerIndex >= 0 ? correctAnswerIndex : (question.correct_answer || question.answer || 0);
      
      // Extract topic from conceptsCovered or use default
      const conceptsCovered = question.metadata?.conceptsCovered || [question.concept || question.topic || ''];
      const topic = conceptsCovered[0] || question.concept || question.topic || (isDBMS ? 'DBMS' : 'General');
      
      const prepared = {
        courseId: question.courseId || courseId,
        questionText: question.questionText || question.question || question.text || question.statement,
        type: normalizedType,
        difficulty: normalizedDifficulty, // Now a number
        status: question.status || 'approved',
        options: options,
        explanation: question.explanation || '',
        answer: question.answer,
        correct_answer: question.correct_answer,
        correctAnswer: correctAnswer, // For Question model compatibility
        topic: topic, // Required field
        metadata: {
          generatedAt: question.metadata?.generatedAt || new Date(),
          conceptsCovered: conceptsCovered,
          timeToSolve: question.metadata?.timeToSolve || (normalizedType === 'mcq' ? 2 : 5)
        },
        createdAt: new Date(),
        usedInPapers: []
      };
      return prepared;
    });

    console.log('Saving questions to:', collectionName, preparedQuestions.length);

    // Save to the appropriate collection
    const result = await QuestionModel.insertMany(preparedQuestions);
    console.log(`Successfully saved ${result.length} questions to ${collectionName}`);

    return res.json({
      success: true,
      message: `Successfully saved ${result.length} questions to ${collectionName}`,
      savedQuestions: result,
      collection: collectionName
    });

  } catch (error) {
    console.error('Error saving questions:', error);
    return res.status(500).json({
      success: false,
      message: 'Failed to save questions',
      error: error.message
    });
  }
});

// Update the batch save endpoint to save to questions collection
app.post('/api/questions/save-batch', async (req, res) => {
  try {
    const { courseId, questions } = req.body;

    if (!courseId || !questions || !Array.isArray(questions)) {
      return res.status(400).json({
        success: false,
        message: 'Invalid request data'
      });
    }

    // Determine which collection to use based on courseId
    const isDBMS = courseId.includes('351A') || courseId.toUpperCase().includes('DBMS');
    const isML = courseId.includes('ML') || courseId.toUpperCase().includes('MACHINE');
    
    // DBMS questions go to "dbmsquestions" collection, ML to "mlquestions"
    const QuestionModel = isDBMS ? DBMSQuestion : (isML ? MLQuestion : Question); // Default to generic Question
    const collectionName = isDBMS ? 'dbmsquestions' : (isML ? 'mlquestions' : 'questions');
    
    console.log(`Batch saving questions to ${collectionName} collection for courseId: ${courseId}`);

    // Prepare questions for saving
    const preparedQuestions = questions.map(question => {
      // Determine correct answer index
      let correctAnswerIndex = -1;
      if (question.correct_answer !== undefined) {
        correctAnswerIndex = question.correct_answer;
      } else if (question.answer !== undefined) {
        correctAnswerIndex = question.answer;
      } else if (typeof question.correct_answer === 'string') {
        // Try to find by string match
        correctAnswerIndex = question.options?.findIndex(opt => {
          const optText = typeof opt === 'string' ? opt : opt.text;
          return optText === question.correct_answer;
        }) || -1;
      }

      // Normalize question type
      const qType = question.type || (question.options ? 'mcq' : 'descriptive');
      const typeMap = {
        'TrueFalse': 'trueFalse',
        'FillBlank': 'fillInBlanks',
        'FillInBlanks': 'fillInBlanks',
        'MCQ': 'mcq',
        'Descriptive': 'descriptive'
      };
      const normalizedType = typeMap[qType] || qType.toLowerCase();
      
      // Convert difficulty string to number (1-10 scale)
      let normalizedDifficulty = 5; // Default moderate
      const diff = question.difficulty || 'moderate';
      if (typeof diff === 'string') {
        const difficultyMap = {
          'easy': 3,
          'moderate': 6,
          'hard': 9,
          'challenging': 9
        };
        normalizedDifficulty = difficultyMap[diff.toLowerCase()] || 5;
      } else if (typeof diff === 'number') {
        if (diff <= 1) {
          normalizedDifficulty = Math.round(diff * 10);
        } else {
          normalizedDifficulty = diff;
        }
      }
      
      // Extract topic
      const conceptsCovered = question.concepts || [question.concept || question.topic || ''];
      const topic = conceptsCovered[0] || question.concept || question.topic || (isDBMS ? 'DBMS' : 'General');
      
      // Build options with correct answer
      const options = question.options ? question.options.map((opt, idx) => {
        const optText = typeof opt === 'object' ? opt.text : opt;
        const isCorrect = correctAnswerIndex === idx || 
                         (typeof opt === 'object' && opt.isCorrect) ||
                         (correctAnswerIndex === -1 && idx === 0); // Default first if no answer specified
        return {
          text: optText,
          isCorrect: isCorrect
        };
      }) : [];
      
      // Find correct answer
      const finalCorrectAnswerIndex = options.findIndex(opt => opt.isCorrect);
      const correctAnswer = finalCorrectAnswerIndex >= 0 ? finalCorrectAnswerIndex : (correctAnswerIndex >= 0 ? correctAnswerIndex : 0);
      
      return {
        courseId,
        questionText: question.text || question.question || question.questionText || question.statement,
        type: normalizedType,
        options: options,
        difficulty: normalizedDifficulty, // Now a number
        status: 'approved',
        explanation: question.explanation || '',
        answer: question.answer,
        correct_answer: question.correct_answer,
        correctAnswer: correctAnswer, // For Question model compatibility
        topic: topic, // Required field
        metadata: {
          generatedAt: new Date(),
          conceptsCovered: conceptsCovered,
          timeToSolve: normalizedType === 'mcq' ? 2 : 5
        },
        createdAt: new Date(),
        usedInPapers: []
      };
    });

    // Save to the appropriate collection
    const result = await QuestionModel.insertMany(preparedQuestions);
    console.log(`Successfully saved ${result.length} questions to ${collectionName}`);

    res.json({
      success: true,
      message: `Successfully saved ${result.length} questions to ${collectionName}`,
      savedCount: result.length,
      questions: result,
      collection: collectionName
    });

  } catch (error) {
    console.error('Error saving questions:', error);
    res.status(500).json({
      success: false,
      message: 'Error saving questions',
      error: error.message
    });
  }
});

// Add this error handling middleware
app.use((err, req, res, next) => {
  console.error('Global error handler:', err);
  res.status(err.status || 500).json({
    message: err.message || 'Internal Server Error',
    stack: process.env.NODE_ENV === 'development' ? err.stack : undefined
  });
});

// Add these routes in server/index.js after the existing routes

// Route for scheduling ISA
app.post('/api/schedule-isa', async (req, res) => {
  try {
    const { courseId, date, time } = req.body;
    const token = req.headers.authorization?.split(' ')[1];
    
    if (!token) {
      return res.status(401).json({ message: 'No token provided' });
    }

    const decoded = jwt.verify(token, 'your-secret-key');
    const teacher = await User.findById(decoded.userId);

    if (!teacher || teacher.role !== 'teacher') {
      return res.status(403).json({ message: 'Not authorized' });
    }

    // Create new ISA schedule with proper status
    const isaSchedule = new QuestionSet({
      courseId,
      type: 'isa',
      title: `ISA - ${new Date(date).toLocaleDateString()}`,
      scheduledDate: date,
      scheduledTime: time,
      status: 'scheduled', // This should now be valid
      createdBy: teacher._id,
      createdAt: new Date(),
      questionTypes: {
        mcq: 0,
        descriptive: 0
      },
      difficulty: {
        easy: 30,
        moderate: 40,
        challenging: 30
      }
    });

    const savedIsa = await isaSchedule.save();

    // Update course with new ISA
    const course = await Course.findOneAndUpdate(
      { id: courseId },
      { 
        $push: { 
          isas: {
            id: savedIsa._id,
            date,
            time
          }
        }
      },
      { new: true, upsert: true }
    );

    // Send success response
    res.status(200).json({
      message: 'ISA scheduled successfully',
      isa: savedIsa,
      course
    });

  } catch (error) {
    console.error('Error scheduling ISA:', error);
    res.status(500).json({ 
      message: 'Error scheduling ISA', 
      error: error.message,
      details: error.errors // Include validation errors if any
    });
  }
});

// Route for creating quiz
app.post('/api/create-quiz', async (req, res) => {
  try {
    const { courseId, title, date } = req.body;
    const token = req.headers.authorization?.split(' ')[1];
    
    if (!token) {
      return res.status(401).json({ message: 'No token provided' });
    }

    const decoded = jwt.verify(token, 'your-secret-key');
    const teacher = await User.findById(decoded.userId);

    if (!teacher || teacher.role !== 'teacher') {
      return res.status(403).json({ message: 'Not authorized' });
    }

    // Create new quiz
    const quiz = new QuestionSet({
      courseId,
      type: 'quiz',
      title,
      scheduledDate: date,
      status: 'scheduled',
      createdBy: teacher._id,
      createdAt: new Date(),
      questionTypes: {
        mcq: 0,
        descriptive: 0
      }
    });

    await quiz.save();

    // Update course with new quiz
    await mongoose.connection.db.collection('courses').updateOne(
      { id: courseId },
      { 
        $push: { 
          quizzes: {
            id: quiz._id,
            title,
            date
          }
        }
      }
    );

    res.json({
      message: 'Quiz created successfully',
      quiz
    });

  } catch (error) {
    console.error('Error creating quiz:', error);
    res.status(500).json({ message: 'Error creating quiz' });
  }
});

// Route for pushing paper
app.post('/api/push-paper', upload.single('paper'), async (req, res) => {
  try {
    const { courseId } = req.body;
    const token = req.headers.authorization?.split(' ')[1];
    
    if (!token) {
      return res.status(401).json({ message: 'No token provided' });
    }

    if (!req.file) {
      return res.status(400).json({ message: 'No file uploaded' });
    }

    const decoded = jwt.verify(token, 'your-secret-key');
    const teacher = await User.findById(decoded.userId);

    if (!teacher || teacher.role !== 'teacher') {
      return res.status(403).json({ message: 'Not authorized' });
    }

    // Create new paper record
    const paper = new QuestionSet({
      courseId,
      type: 'paper',
      title: req.file.originalname,
      status: 'published',
      createdBy: teacher._id,
      createdAt: new Date(),
      fileUrl: req.file.path // Save the file path
    });

    await paper.save();

    // Update course with new paper
    await mongoose.connection.db.collection('courses').updateOne(
      { id: courseId },
      { 
        $push: { 
          papers: {
            id: paper._id,
            name: req.file.originalname,
            uploadedAt: new Date()
          }
        }
      }
    );

    res.json({
      message: 'Paper pushed successfully',
      paper
    });

  } catch (error) {
    console.error('Error pushing paper:', error);
    res.status(500).json({ message: 'Error pushing paper' });
  }
});

// Mount routes with /api prefix
app.use('/api/auth', authRoutes);
app.use('/api/papers', paperRoutes);
app.use('/api/schedules', scheduleRoutes);
app.use('/api/diagnostic', diagnosticRoutes);

// Add a catch-all route for /api/* to handle 404s properly
app.use('/api/*', (req, res) => {
  res.status(404).json({ message: 'API endpoint not found' });
});

// Add endpoint to check database status
app.get('/api/db/status', async (req, res) => {
  try {
    const db = mongoose.connection.db;
    const collections = await db.listCollections().toArray();
    const questionCount = await questionsCollection.countDocuments();
    const templateCount = await questionTemplatesCollection.countDocuments();
    
    res.json({
      success: true,
      connected: mongoose.connection.readyState === 1,
      collections: collections.map(c => c.name),
      counts: {
        questions: questionCount,
        templates: templateCount
      }
    });
  } catch (error) {
    console.error('Database status check error:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Add a debug endpoint to check database connection
app.get('/api/debug/db', async (req, res) => {
  try {
    const db = mongoose.connection.db;
    const collections = await db.listCollections().toArray();
    const questionsCollection = db.collection('questions');
    const count = await questionsCollection.countDocuments();
    
    res.json({
      connected: mongoose.connection.readyState === 1,
      database: db.databaseName,
      collections: collections.map(c => c.name),
      questionCount: count,
      connectionString: MONGODB_URI.replace(/\/\/[^:]+:[^@]+@/, '//***:***@') // Hide credentials
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message,
      connectionState: mongoose.connection.readyState
    });
  }
});

// Route to get exam questions
app.get('/api/exams/:examId', async (req, res) => {
  try {
    const token = req.headers.authorization?.split(' ')[1];
    if (!token) {
      return res.status(401).json({ message: 'No token provided' });
    }

    const decoded = jwt.verify(token, 'your-secret-key');
    const student = await User.findById(decoded.userId);

    if (!student || student.role !== 'student') {
      return res.status(403).json({ message: 'Not authorized' });
    }

    const schedule = await Schedule.findById(req.params.examId).populate('paperId');
    if (!schedule) {
      return res.status(404).json({ message: 'Exam not found' });
    }

    // Check if exam is available
    const now = new Date();
    const examDate = new Date(schedule.date);
    const [hours, minutes] = schedule.time.split(':');
    examDate.setHours(parseInt(hours), parseInt(minutes));
    const examStartTime = new Date(examDate.getTime() - 5 * 60000); // 5 minutes before
    const examEndTime = new Date(examDate.getTime() + schedule.duration * 60000);

    if (now < examStartTime || now > examEndTime) {
      return res.status(403).json({ message: 'Exam is not available at this time' });
    }

    if (!schedule.paperId) {
      return res.status(404).json({ message: 'No question paper associated with this exam' });
    }

    // Get questions from the associated paper
    const paper = schedule.paperId;
    
    // Remove correct answers from questions before sending to student
    const sanitizedQuestions = paper.questions.map(q => ({
      _id: q._id,
      text: q.questionText,
      type: q.type,
      options: q.type === 'mcq' ? q.options.map(opt => opt.text) : undefined
    }));

    res.json({
      title: schedule.title,
      duration: schedule.duration,
      questions: sanitizedQuestions,
      totalMarks: paper.totalMarks
    });

  } catch (error) {
    console.error('Error fetching exam:', error);
    res.status(500).json({ message: 'Error fetching exam', error: error.message });
  }
});

// Route to submit exam answers
app.post('/api/exams/:examId/submit', async (req, res) => {
  try {
    const token = req.headers.authorization?.split(' ')[1];
    if (!token) {
      return res.status(401).json({ message: 'No token provided' });
    }

    const decoded = jwt.verify(token, 'your-secret-key');
    const student = await User.findById(decoded.userId);

    if (!student || student.role !== 'student') {
      return res.status(403).json({ message: 'Not authorized' });
    }

    const { answers } = req.body;
    const schedule = await Schedule.findById(req.params.examId).populate('paperId');
    
    if (!schedule) {
      return res.status(404).json({ message: 'Exam not found' });
    }

    if (!schedule.paperId) {
      return res.status(404).json({ message: 'No question paper associated with this exam' });
    }

    const paper = schedule.paperId;
    
    // Calculate score
    let score = 0;
    const questionResponses = [];

    for (const question of paper.questions) {
      const answer = answers[question._id];
      let isCorrect = false;

      if (question.type === 'mcq') {
        // For MCQs, check if selected option is correct
        isCorrect = question.options[answer]?.isCorrect || false;
        if (isCorrect) score++;
      }

      questionResponses.push({
        questionId: question._id,
        selectedOption: answer,
        isCorrect,
        timeSpent: 0 // This could be tracked in the frontend if needed
      });
    }

    // Save student's progress
    let studentProgress = await StudentProgress.findOne({
      studentId: student._id,
      courseId: schedule.courseId
    });

    if (!studentProgress) {
      studentProgress = new StudentProgress({
        studentId: student._id,
        courseId: schedule.courseId,
        isas: [],
        quizzes: [],
        overallProgress: {
          topicsCompleted: [],
          averageQuizScore: 0,
          averageIsaScore: 0,
          totalTimeSpent: 0,
          strengthTopics: [],
          weaknessTopics: [],
          lastUpdated: new Date()
        }
      });
    }

    // Add exam result to student's progress
    if (schedule.type === 'ISA') {
      studentProgress.isas.push({
        isaId: paper._id,
        score,
        completedAt: new Date()
      });
    } else {
      studentProgress.quizzes.push({
        quizId: paper._id,
        score,
        timeTaken: schedule.duration,
        completedAt: new Date(),
        answers: questionResponses
      });
    }

    // Update overall progress
    const examScores = schedule.type === 'ISA' 
      ? studentProgress.isas.map(isa => isa.score)
      : studentProgress.quizzes.map(quiz => quiz.score);
    
    const averageScore = examScores.reduce((a, b) => a + b, 0) / examScores.length;

    if (schedule.type === 'ISA') {
      studentProgress.overallProgress.averageIsaScore = averageScore;
    } else {
      studentProgress.overallProgress.averageQuizScore = averageScore;
    }

    studentProgress.overallProgress.lastUpdated = new Date();
    await studentProgress.save();

    res.json({
      message: 'Exam submitted successfully',
      score,
      totalQuestions: paper.questions.length,
      totalMarks: paper.totalMarks
    });

  } catch (error) {
    console.error('Error submitting exam:', error);
    res.status(500).json({ message: 'Error submitting exam', error: error.message });
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.status(200).json({
    status: 'healthy',
    service: 'backend',
    timestamp: new Date().toISOString()
  });
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
}); 