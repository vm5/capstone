const express = require('express');
const router = express.Router();
const Paper = require('../models/Paper');
const Question = require('../models/Question');
const DBMSQuestion = require('../models/DBMSQuestion');
const MLQuestion = require('../models/MLQuestion');
const auth = require('../middleware/auth');
const axios = require('axios');
const fs = require('fs').promises;
const path = require('path');
const PDFDocument = require('pdfkit');
const mongoose = require('mongoose'); // Added for QuestionTemplate model

// Helper function to convert numeric difficulty to label
function getDifficultyLabel(difficultyScore) {
  if (difficultyScore <= 3) return 'easy';
  if (difficultyScore <= 6) return 'moderate';
  return 'challenging';
}

// Get approved questions for a course
router.get('/approved-questions', async (req, res) => {
  try {
    const { courseId } = req.query;
    
    if (!courseId) {
      return res.status(400).json({ message: 'Course ID is required' });
    }

    // Determine which collection to use based on courseId
    const isDBMS = courseId.includes('351A') || courseId.toUpperCase().includes('DBMS');
    const isML = courseId.includes('ML') || courseId.toUpperCase().includes('MACHINE');
    
    // DBMS questions are in "dbmsquestions" collection, ML in "mlquestions"
    const QuestionModel = isDBMS ? DBMSQuestion : (isML ? MLQuestion : Question); // Default to generic Question
    const collectionName = isDBMS ? 'dbmsquestions' : (isML ? 'mlquestions' : 'questions');
    
    console.log(`Fetching approved questions from ${collectionName} for courseId: ${courseId}`);

    // Get questions from the appropriate collection - STRICT courseId match
    let questions = [];
    try {
      questions = await QuestionModel.find({
        courseId: courseId, // Exact match required
        status: 'approved'
      }).sort({ createdAt: -1 });
      console.log(`Found ${questions.length} approved questions in ${collectionName} for courseId: ${courseId}`);
    } catch (e) {
      console.error(`Error fetching from ${collectionName}:`, e);
    }

    // Also try the generic Question collection as fallback - STRICT courseId match
    let genericQuestions = [];
    try {
      genericQuestions = await Question.find({
        courseId: courseId, // Exact match required
        status: 'approved'
      }).sort({ createdAt: -1 });
      console.log(`Found ${genericQuestions.length} approved questions in generic questions collection for courseId: ${courseId}`);
    } catch (e) {
      console.log('No generic Question model found or error:', e.message);
    }

    // Get questions from QuestionTemplate if it exists - STRICT courseId match
    let templateQuestions = [];
    try {
      const QuestionTemplate = mongoose.model('QuestionTemplate');
      templateQuestions = await QuestionTemplate.find({
        courseId: courseId, // Exact match required
        status: 'approved'
      }).sort({ createdAt: -1 });
    } catch (e) {
      // QuestionTemplate model might not exist, ignore error
      console.log('No QuestionTemplate model found');
    }

    // Combine and normalize questions - ensure courseId matches exactly
    const allQuestions = [
      ...questions.map(q => q.toObject()),
      ...genericQuestions.map(q => q.toObject()),
      ...templateQuestions.map(q => q.toObject())
    ]
    .filter(q => {
      // STRICT courseId filtering - only include questions for this exact course
      const qCourseId = q.courseId;
      return qCourseId === courseId;
    })
    .map(q => {
      // Normalize question type - handle various formats
      let normalizedType = q.type || 'mcq';
      const typeMap = {
        'TrueFalse': 'trueFalse',
        'FillBlank': 'fillInBlanks',
        'FillInBlanks': 'fillInBlanks',
        'MCQ': 'mcq',
        'Descriptive': 'descriptive',
        'descriptive': 'descriptive',
        'mcq': 'mcq',
        'trueFalse': 'trueFalse',
        'fillInBlanks': 'fillInBlanks'
      };
      normalizedType = typeMap[normalizedType] || normalizedType.toLowerCase();
      if (!normalizedType && q.options && q.options.length > 0) {
        normalizedType = 'mcq';
      }
      
      // Normalize difficulty to 1-10 scale if it's 0-1
      let normalizedDifficulty = q.difficulty;
      if (typeof q.difficulty === 'number' && q.difficulty <= 1) {
        normalizedDifficulty = Math.round(q.difficulty * 10);
      } else if (typeof q.difficulty === 'string') {
        const difficultyMap = {
          'easy': 3,
          'moderate': 6,
          'hard': 9,
          'challenging': 9
        };
        normalizedDifficulty = difficultyMap[q.difficulty.toLowerCase()] || 5;
      } else if (!normalizedDifficulty) {
        normalizedDifficulty = 5;
      }
      
      return {
        ...q,
        type: normalizedType,
        difficulty: normalizedDifficulty,
        courseId: q.courseId || courseId // Ensure courseId is set
      };
    });

    // Remove duplicates based on questionText
    const uniqueQuestions = [];
    const seenTexts = new Set();
    for (const q of allQuestions) {
      const text = q.questionText || q.question || q.text || q.statement;
      if (text && !seenTexts.has(text)) {
        seenTexts.add(text);
        uniqueQuestions.push(q);
      }
    }

    console.log(`Found ${uniqueQuestions.length} unique approved questions for course ${courseId} (from ${collectionName})`);

    res.json(uniqueQuestions);
  } catch (error) {
    console.error('Error fetching approved questions:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

// Course-scoped papers MUST be declared before wildcard :id routes
router.get('/course/:courseId', auth, async (req, res) => {
  try {
    const { courseId } = req.params;
    const userId = req.user.googleId || req.user._id;
    
    // Strict filtering: exact courseId match and user match
    const papers = await Paper.find({
      courseId: courseId, // Exact match required
      createdBy: userId
    }).sort('-createdAt');
    
    console.log(`[Course Filter] Requested courseId: ${courseId}, User: ${userId}, Found papers: ${papers.length}`);
    papers.forEach(p => {
      console.log(`  - Paper: ${p.title}, courseId: ${p.courseId}, createdBy: ${p.createdBy}`);
    });
    
    // Double-check: filter out any papers that don't match (defensive)
    const filtered = papers.filter(p => p.courseId === courseId && (p.createdBy?.toString() === userId?.toString() || p.createdBy?.toString() === req.user._id?.toString()));
    
    res.json(filtered);
  } catch (error) {
    console.error('Error fetching course papers:', error);
    res.status(500).json({ message: error.message });
  }
});

// Get paper raw data as PDF
router.get('/:id/raw-pdf', async (req, res) => {
  try {
    const paper = await Paper.findById(req.params.id);
    if (!paper) {
      return res.status(404).json({ message: 'Paper not found' });
    }

    // Create a PDF document
    const doc = new PDFDocument({
      size: 'A4',
      margin: 50,
      info: {
        Title: `Raw Data - ${paper.title}`,
        Author: 'Dashboard App'
      }
    });
    
    // Set response headers
    res.setHeader('Content-Type', 'application/pdf');
    res.setHeader('Content-Disposition', `attachment; filename=raw-data-${paper.title.replace(/\s+/g, '_')}-${paper._id}.pdf`);
    
    // Pipe the PDF to the response
    doc.pipe(res);

    // Add title
    doc.fontSize(16)
      .text('Raw Paper Data', { align: 'center' })
      .moveDown();

    // Add paper info header
    doc.fontSize(12)
      .text(`Title: ${paper.title}`)
      .text(`Course: ${paper.courseId}`)
      .text(`Created: ${new Date(paper.createdAt).toLocaleString()}`)
      .moveDown();

    // Add raw data with syntax highlighting-like formatting
    const formatValue = (key, value, indent = 0) => {
      const indentStr = '    '.repeat(indent);
      if (typeof value === 'object' && value !== null) {
        if (Array.isArray(value)) {
          doc.fontSize(10).text(`${indentStr}${key}: [`, { continued: false });
          value.forEach((item, index) => {
            if (typeof item === 'object') {
              doc.text(`${indentStr}    {`, { continued: false });
              Object.entries(item).forEach(([k, v]) => formatValue(k, v, indent + 2));
              doc.text(`${indentStr}    }${index < value.length - 1 ? ',' : ''}`, { continued: false });
            } else {
              doc.text(`${indentStr}    ${JSON.stringify(item)}${index < value.length - 1 ? ',' : ''}`, { continued: false });
            }
          });
          doc.text(`${indentStr}]`, { continued: false });
        } else {
          doc.text(`${indentStr}${key}: {`, { continued: false });
          Object.entries(value).forEach(([k, v]) => formatValue(k, v, indent + 1));
          doc.text(`${indentStr}}`, { continued: false });
        }
      } else {
        doc.text(`${indentStr}${key}: ${JSON.stringify(value)}`, { continued: false });
      }
    };

    // Format and add each section
    doc.fontSize(11).text('Paper Data:', { underline: true }).moveDown();
    Object.entries(paper.toObject()).forEach(([key, value]) => {
      formatValue(key, value);
      doc.moveDown(0.5);
    });

    // Finalize the PDF
    doc.end();

  } catch (error) {
    console.error('Error generating raw data PDF:', error);
    res.status(500).json({ message: 'Error generating PDF' });
  }
});

// Get paper by ID (temporarily removed auth for testing)
router.get('/:id', async (req, res) => {
  try {
    const paper = await Paper.findById(req.params.id);
    if (!paper) {
      return res.status(404).json({ message: 'Paper not found' });
    }

    // If HTML format is requested, render HTML view
    if (req.query.format === 'html') {
      const marksPerQuestion = Math.round(paper.totalMarks / paper.questions.length);
      
      const html = `
        <!DOCTYPE html>
        <html>
        <head>
          <title>${paper.title || 'Examination Paper'}</title>
          <style>
            body {
              font-family: Arial, sans-serif;
              max-width: 800px;
              margin: 20px auto;
              padding: 20px;
              line-height: 1.6;
            }
            .header {
              text-align: center;
              margin-bottom: 30px;
              border-bottom: 2px solid #333;
              padding-bottom: 20px;
            }
            .paper-info {
              text-align: right;
              margin-bottom: 30px;
            }
            .question {
              margin-bottom: 25px;
              padding: 15px;
              background: #f9f9f9;
              border-radius: 5px;
            }
            .question-text {
              font-weight: bold;
              margin-bottom: 10px;
            }
            .options {
              margin-left: 20px;
            }
            .option {
              margin: 5px 0;
            }
            .marks {
              float: right;
              color: #666;
            }
            .btn {
              display: inline-block;
              padding: 10px 20px;
              text-decoration: none;
              border-radius: 5px;
              margin: 0 10px 20px;
              transition: background-color 0.3s;
              color: white;
            }
            .btn-primary {
              background-color: #007bff;
            }
            .btn-primary:hover {
              background-color: #0056b3;
            }
            .btn-secondary {
              background-color: #6c757d;
            }
            .btn-secondary:hover {
              background-color: #545b62;
            }
            .actions {
              text-align: center;
              margin-bottom: 30px;
            }
            #rawData {
              display: none;
              margin: 20px 0;
              padding: 15px;
              background: #f8f9fa;
              border: 1px solid #ddd;
              border-radius: 5px;
              white-space: pre-wrap;
              font-family: monospace;
              font-size: 14px;
            }
          </style>
          <script>
            function toggleRawData() {
              const rawData = document.getElementById('rawData');
              const btn = document.getElementById('toggleBtn');
              if (rawData.style.display === 'none') {
                rawData.style.display = 'block';
                btn.textContent = 'Hide Raw Data';
              } else {
                rawData.style.display = 'none';
                btn.textContent = 'View Raw Data';
              }
            }
          </script>
        </head>
        <body>
          <div class="header">
            <h1>${paper.title || 'Examination Paper'}</h1>
            <h3>Course: ${paper.courseId}</h3>
          </div>
          
          <div class="actions">
            <a href="/papers/${paper._id}/pdf" class="btn btn-primary" download>
              Download as PDF
            </a>
            <button onclick="toggleRawData()" id="toggleBtn" class="btn btn-secondary">
              View Raw Data
            </button>
          </div>

          <div id="rawData">
${JSON.stringify(paper, null, 2)}
          </div>
          
          <div class="paper-info">
            <p>Total Marks: ${paper.totalMarks}</p>
            <p>Duration: ${paper.duration} minutes</p>
          </div>

          ${paper.questions.map((question, index) => `
            <div class="question">
              <div class="question-text">
                Q${index + 1}. ${question.questionText}
                <span class="marks">[${question.marks || marksPerQuestion} marks]</span>
              </div>
              ${question.type === 'mcq' ? `
                <div class="options">
                  ${question.options.map((option, optIndex) => `
                    <div class="option">
                      ${String.fromCharCode(65 + optIndex)}) ${option.text}
                    </div>
                  `).join('')}
                </div>
              ` : ''}
            </div>
          `).join('')}
        </body>
        </html>
      `;
      
      res.send(html);
    } else {
      // Return enhanced JSON response with download links at top level
      res.json({
        downloadLinks: {
          formattedPdf: `http://localhost:5000/papers/${paper._id}/pdf`,
          rawDataPdf: `http://localhost:5000/papers/${paper._id}/raw-pdf`
        },
        paper: {
          _id: paper._id,
          title: paper.title,
          courseId: paper.courseId,
          totalMarks: paper.totalMarks,
          duration: paper.duration,
          type: paper.type,
          status: paper.status,
          createdAt: paper.createdAt,
          createdBy: paper.createdBy,
          metadata: paper.metadata,
          questionTypes: paper.questionTypes,
          difficulty: paper.difficulty,
          questions: paper.questions.map(q => ({
            _id: q._id,
            type: q.type,
            questionText: q.questionText,
            marks: q.marks,
            difficulty: q.difficulty,
            options: q.type === 'mcq' ? q.options.map(opt => ({
              text: opt.text,
              isCorrect: opt.isCorrect
            })) : undefined,
            correctAnswer: q.correctAnswer
          }))
        },
        rawData: JSON.parse(JSON.stringify(paper))
      });
    }
  } catch (error) {
    console.error('Error fetching paper:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

// Get paper as PDF
router.get('/:id/pdf', async (req, res) => {
  try {
    const paper = await Paper.findById(req.params.id);
    if (!paper) {
      return res.status(404).json({ message: 'Paper not found' });
    }

    // Create a PDF document
    const doc = new PDFDocument({
      size: 'A4',
      margin: 50
    });
    
    // Set response headers
    res.setHeader('Content-Type', 'application/pdf');
    res.setHeader('Content-Disposition', `attachment; filename=${paper.title.replace(/\s+/g, '_')}-${paper._id}.pdf`);
    
    // Pipe the PDF to the response
    doc.pipe(res);

    // Add header
    doc.fontSize(18).text(paper.title || 'Examination Paper', { align: 'center' });
    doc.fontSize(14).text(`Course: ${paper.courseId}`, { align: 'center' });
    doc.moveDown();

    // Add paper info
    doc.fontSize(12)
      .text(`Total Marks: ${paper.totalMarks}`, { align: 'right' })
      .text(`Duration: ${paper.duration} minutes`, { align: 'right' });
    
    // Add instructions if available
    if (paper.metadata?.instructions) {
      doc.moveDown()
        .fontSize(12)
        .text('Instructions:', { underline: true })
        .text(paper.metadata.instructions);
    }
    
    doc.moveDown().moveDown();

    // Add questions
    paper.questions.forEach((question, index) => {
      // Question number and text
      doc.fontSize(12)
        .text(`Q${index + 1}. ${question.questionText}`, {
          continued: true,
          width: doc.page.width - 100
        });
      
      // Add marks
      doc.text(` [${question.marks} marks]`, { align: 'right' });
      doc.moveDown();

      // For MCQ questions, add options
      if (question.type === 'mcq' && question.options) {
        question.options.forEach((option, optIndex) => {
          doc.fontSize(11)
            .text(`${String.fromCharCode(65 + optIndex)}) ${option.text}`, {
              indent: 30,
              width: doc.page.width - 130
            });
        });
        doc.moveDown();
      }

      // For descriptive questions, add space for answer
      if (question.type === 'descriptive') {
        doc.moveDown()
          .fontSize(10)
          .text('Answer:', { indent: 30 })
          .moveTo(doc.x + 30, doc.y)
          .lineTo(doc.page.width - 50, doc.y)
          .stroke();
        
        // Add multiple lines for answer
        for (let i = 0; i < 4; i++) {
          doc.moveDown()
            .moveTo(doc.x + 30, doc.y)
            .lineTo(doc.page.width - 50, doc.y)
            .stroke();
        }
      }

      // Add space between questions
      doc.moveDown().moveDown();
    });

    // Finalize the PDF
    doc.end();

  } catch (error) {
    console.error('Error generating PDF:', error);
    res.status(500).json({ message: 'Error generating PDF' });
  }
});

router.post('/generate-paper', auth, async (req, res) => {
  try {
    const { courseId, config } = req.body;
    console.log('Received paper generation request:', { courseId, config });
    
    if (!courseId || !config) {
      return res.status(400).json({ 
        success: false, 
        message: 'Missing required fields: courseId and config' 
      });
    }

    // Validate and normalize difficulty distribution
    const defaultDifficulty = { easy: 0.6, moderate: 0.3, challenging: 0.1 };
    if (!config.difficulty || 
        typeof config.difficulty !== 'object' ||
        Object.values(config.difficulty).every(val => val === 0)) {
      console.log('Using default difficulty distribution:', defaultDifficulty);
      config.difficulty = defaultDifficulty;
    } else {
      // Normalize difficulty values to sum to 1
      const total = Object.values(config.difficulty).reduce((sum, val) => sum + val, 0);
      if (total === 0) {
        config.difficulty = defaultDifficulty;
      } else if (total !== 1) {
        config.difficulty = {
          easy: config.difficulty.easy / total || 0,
          moderate: config.difficulty.moderate / total || 0,
          challenging: config.difficulty.challenging / total || 0
        };
      }
    }

    console.log('Using difficulty distribution:', config.difficulty);

    // Validate question types
    if (!config.questionTypes || typeof config.questionTypes !== 'object') {
      return res.status(400).json({ 
        success: false, 
        message: 'questionTypes must be provided as an object' 
      });
    }

    // Ensure at least one question type is selected
    const totalQuestions = Object.values(config.questionTypes).reduce((sum, count) => sum + count, 0);
    if (totalQuestions === 0) {
      return res.status(400).json({ 
        success: false, 
        message: 'At least one question type must have a count greater than 0' 
      });
    }

    // Validate individual question type counts
    const validTypes = ['mcq', 'descriptive', 'trueFalse', 'fillInBlanks'];
    const invalidTypes = Object.keys(config.questionTypes).filter(type => !validTypes.includes(type));
    if (invalidTypes.length > 0) {
      return res.status(400).json({
        success: false,
        message: `Invalid question types: ${invalidTypes.join(', ')}`
      });
    }
    
    // Determine which collection to use based on courseId
    const isDBMS = courseId.includes('351A') || courseId.toUpperCase().includes('DBMS');
    const isML = courseId.includes('ML') || courseId.toUpperCase().includes('MACHINE');
    
    // DBMS questions are in "dbmsquestions" collection, ML in "mlquestions"
    const QuestionModel = isDBMS ? DBMSQuestion : (isML ? MLQuestion : Question); // Default to generic Question
    const collectionName = isDBMS ? 'dbmsquestions' : (isML ? 'mlquestions' : 'questions');
    
    console.log(`Fetching questions from ${collectionName} for paper generation, courseId: ${courseId}`);

    // Get questions from the appropriate collection - STRICT courseId match
    const [questions, genericQuestions, templateQuestions] = await Promise.all([
      QuestionModel.find({
        courseId: courseId, // Exact match required
        status: 'approved'
      }).sort({ createdAt: -1 }).catch(() => []),
      Question.find({
        courseId: courseId, // Exact match required
        status: 'approved'
      }).sort({ createdAt: -1 }).catch(() => []),
      mongoose.model('QuestionTemplate').find({
        courseId: courseId, // Exact match required
        status: 'approved'
      }).sort({ createdAt: -1 }).catch(() => []) // Ignore errors if model doesn't exist
    ]);

    console.log(`Raw questions from ${collectionName} for courseId ${courseId}:`, questions.length);
    console.log('Raw generic questions for courseId:', genericQuestions.length);
    console.log('Raw template questions for courseId:', templateQuestions.length);
    
    // Log question types for debugging
    const allRawQuestions = [...questions, ...genericQuestions, ...templateQuestions];
    const typeCounts = {};
    allRawQuestions.forEach(q => {
      const qType = q.type || q.toObject?.().type || 'unknown';
      typeCounts[qType] = (typeCounts[qType] || 0) + 1;
    });
    console.log('Question types found:', typeCounts);

    // Combine and normalize questions - ensure courseId matches exactly
    const allQuestions = [
      ...questions.map(q => {
        const qObj = q.toObject();
        return { ...qObj, source: collectionName };
      }),
      ...genericQuestions.map(q => {
        const qObj = q.toObject();
        return { ...qObj, source: 'generic' };
      }),
      ...templateQuestions.map(q => {
        const qObj = q.toObject();
        return { ...qObj, source: 'templates' };
      })
    ]
    .filter(q => {
      // STRICT courseId filtering - only include questions for this exact course
      const qCourseId = q.courseId || q.course?.id || q.courseId;
      return qCourseId === courseId;
    })
    .map(q => {
      // Normalize question type - handle various formats
      let normalizedType = q.type || 'mcq';
      const typeMap = {
        'TrueFalse': 'trueFalse',
        'FillBlank': 'fillInBlanks',
        'FillInBlanks': 'fillInBlanks',
        'MCQ': 'mcq',
        'Descriptive': 'descriptive',
        'descriptive': 'descriptive',
        'mcq': 'mcq',
        'trueFalse': 'trueFalse',
        'fillInBlanks': 'fillInBlanks'
      };
      normalizedType = typeMap[normalizedType] || normalizedType.toLowerCase();
      
      // Normalize difficulty - handle both string and numeric values
      let normalizedDifficulty = q.difficulty;
      
      // If difficulty is a string, convert to number
      if (typeof q.difficulty === 'string') {
        const difficultyMap = {
          'easy': 3,
          'moderate': 6,
          'hard': 9,
          'challenging': 9
        };
        normalizedDifficulty = difficultyMap[q.difficulty.toLowerCase()] || 5;
      } else if (typeof q.difficulty === 'number') {
        // If it's a number between 0-1, convert to 1-10 scale
        if (q.difficulty <= 1) {
          normalizedDifficulty = Math.round(q.difficulty * 10);
        } else if (q.difficulty > 10) {
          // If it's already on 1-10 scale, keep it
          normalizedDifficulty = q.difficulty;
        }
      } else {
        // Default to moderate if undefined/null
        normalizedDifficulty = 5;
      }
      
      return {
        ...q,
        type: normalizedType,
        difficulty: normalizedDifficulty,
        courseId: q.courseId || courseId // Ensure courseId is set
      };
    });

    console.log(`Found ${allQuestions.length} questions for course ${courseId}`);

    if (!allQuestions.length) {
      return res.status(400).json({ 
        success: false, 
        message: 'No approved questions found for this course' 
      });
    }

    // Group questions by type and difficulty
    const groupedQuestions = {
      mcq: {
        easy: allQuestions.filter(q => q.type === 'mcq' && q.difficulty <= 4),
        moderate: allQuestions.filter(q => q.type === 'mcq' && q.difficulty > 4 && q.difficulty <= 7),
        challenging: allQuestions.filter(q => q.type === 'mcq' && q.difficulty > 7)
      },
      descriptive: {
        easy: allQuestions.filter(q => q.type === 'descriptive' && q.difficulty <= 4),
        moderate: allQuestions.filter(q => q.type === 'descriptive' && q.difficulty > 4 && q.difficulty <= 7),
        challenging: allQuestions.filter(q => q.type === 'descriptive' && q.difficulty > 7)
      },
      trueFalse: {
        easy: allQuestions.filter(q => q.type === 'trueFalse' && q.difficulty <= 4),
        moderate: allQuestions.filter(q => q.type === 'trueFalse' && q.difficulty > 4 && q.difficulty <= 7),
        challenging: allQuestions.filter(q => q.type === 'trueFalse' && q.difficulty > 7)
      },
      fillInBlanks: {
        easy: allQuestions.filter(q => q.type === 'fillInBlanks' && q.difficulty <= 4),
        moderate: allQuestions.filter(q => q.type === 'fillInBlanks' && q.difficulty > 4 && q.difficulty <= 7),
        challenging: allQuestions.filter(q => q.type === 'fillInBlanks' && q.difficulty > 7)
      }
    };

    // Log question distribution
    const distribution = {
      mcq: {
        easy: groupedQuestions.mcq.easy.length,
        moderate: groupedQuestions.mcq.moderate.length,
        challenging: groupedQuestions.mcq.challenging.length,
        total: groupedQuestions.mcq.easy.length + groupedQuestions.mcq.moderate.length + groupedQuestions.mcq.challenging.length
      },
      descriptive: {
        easy: groupedQuestions.descriptive.easy.length,
        moderate: groupedQuestions.descriptive.moderate.length,
        challenging: groupedQuestions.descriptive.challenging.length,
        total: groupedQuestions.descriptive.easy.length + groupedQuestions.descriptive.moderate.length + groupedQuestions.descriptive.challenging.length
      },
      trueFalse: {
        easy: groupedQuestions.trueFalse.easy.length,
        moderate: groupedQuestions.trueFalse.moderate.length,
        challenging: groupedQuestions.trueFalse.challenging.length,
        total: groupedQuestions.trueFalse.easy.length + groupedQuestions.trueFalse.moderate.length + groupedQuestions.trueFalse.challenging.length
      },
      fillInBlanks: {
        easy: groupedQuestions.fillInBlanks.easy.length,
        moderate: groupedQuestions.fillInBlanks.moderate.length,
        challenging: groupedQuestions.fillInBlanks.challenging.length,
        total: groupedQuestions.fillInBlanks.easy.length + groupedQuestions.fillInBlanks.moderate.length + groupedQuestions.fillInBlanks.challenging.length
      }
    };

    console.log('Question distribution:', distribution);
    console.log('Requested question types:', config.questionTypes);

    // Validate we have enough questions of each type
    for (const [type, count] of Object.entries(config.questionTypes)) {
      if (count > 0) {
        const availableCount = distribution[type]?.total || 0;
        console.log(`Checking ${type}: need ${count}, have ${availableCount}`);
        
        if (availableCount < count) {
          // Log detailed info for debugging
          console.log(`Available ${type} questions:`, {
            easy: distribution[type]?.easy || 0,
            moderate: distribution[type]?.moderate || 0,
            challenging: distribution[type]?.challenging || 0,
            total: availableCount
          });
          
          // Only return error if we actually checked and found insufficient questions
          // Don't show error if type wasn't requested or if there are 0 available
          if (availableCount === 0) {
            return res.status(400).json({
              success: false,
              message: `No ${type} questions available for course ${courseId}. Please generate and approve ${type} questions first.`
            });
          } else {
            return res.status(400).json({
              success: false,
              message: `Not enough ${type} questions for course ${courseId}. Need ${count} but only have ${availableCount} available.`
            });
          }
        }
      }
    }

    // Select questions based on difficulty distribution
    const selectQuestions = (type, count) => {
      if (count === 0) return [];
      
      const questions = [];
      const { easy = 0.6, moderate = 0.3, challenging = 0.1 } = config.difficulty || {};
      
      // Calculate number of questions for each difficulty
      const easyCount = Math.round(count * easy);
      const moderateCount = Math.round(count * moderate);
      const challengingCount = count - easyCount - moderateCount;
      
      console.log(`Selecting ${type} questions:`, { easyCount, moderateCount, challengingCount });

      // Helper to get random questions
      const getRandomQuestions = (pool, n) => {
        if (pool.length < n) {
          console.log(`Warning: Not enough ${type} questions in pool. Need ${n}, have ${pool.length}`);
        }
        const shuffled = [...pool].sort(() => 0.5 - Math.random());
        return shuffled.slice(0, n);
      };
      
      // Get questions for each difficulty level
      const selectedEasy = getRandomQuestions(groupedQuestions[type].easy, easyCount);
      const selectedModerate = getRandomQuestions(groupedQuestions[type].moderate, moderateCount);
      const selectedChallenging = getRandomQuestions(groupedQuestions[type].challenging, challengingCount);

      questions.push(...selectedEasy, ...selectedModerate, ...selectedChallenging);
      
      console.log(`Selected ${questions.length} ${type} questions:`, {
        easy: selectedEasy.length,
        moderate: selectedModerate.length,
        challenging: selectedChallenging.length
      });

      return questions;
    };

    // Select questions for each type
    const selectedQuestions = [];
    for (const [type, count] of Object.entries(config.questionTypes)) {
        const typeQuestions = selectQuestions(type, count);
      if (typeQuestions.length < count) {
      return res.status(400).json({ 
        success: false, 
          message: `Not enough ${type} questions available. Need ${count} but only found ${typeQuestions.length}.`
        });
      }
      selectedQuestions.push(...typeQuestions);
    }

    console.log('Final selected questions:', selectedQuestions.map(q => ({
      id: q._id,
      type: q.type,
      difficulty: q.difficulty,
      source: q.source
    })));

    // Assign marks by type and compute total
    const embeddedQuestions = selectedQuestions.map(q => ({
      _id: q._id,
      type: q.type,
      questionText: q.questionText,
      options: q.type === 'mcq' ? q.options : undefined,
      correctAnswer: q.correctAnswer,
      difficulty: q.difficulty,
      marks: q.type === 'descriptive' ? 4 : 1
    }));

    const computedTotalMarks = embeddedQuestions.reduce((sum, q) => sum + (q.marks || 0), 0);

    // Create the paper with embedded questions
    const paper = new Paper({
      title: config.title,
      courseId,
      totalMarks: computedTotalMarks,
      duration: config.duration,
      type: config.type || 'exam',
      status: 'active',
      questions: embeddedQuestions,
      questionTypes: {
        model: 'QuestionTemplate',
        mcq: config.questionTypes.mcq || 0,
        descriptive: config.questionTypes.descriptive || 0
      },
      createdBy: req.user?._id || 'system',
      metadata: {
        difficulty: config.difficulty,
        questionTypes: config.questionTypes
      }
    });

    // Save paper first
    const savedPaper = await paper.save();
    console.log('Paper saved with questions:', JSON.stringify(savedPaper.questions, null, 2));

    // Now populate the questions
    const populatedPaper = await Paper.findById(savedPaper._id);

    if (!populatedPaper.questions || populatedPaper.questions.length === 0) {
      console.error('Failed to save questions. Selected questions:', selectedQuestions.map(q => ({
        id: q._id,
        type: q.type,
        questionText: q.questionText,
        options: q.type === 'mcq' ? q.options : undefined,
        difficulty: q.difficulty
      })));
      await Paper.findByIdAndDelete(savedPaper._id); // Cleanup failed paper
      return res.status(500).json({
        success: false,
        message: 'Failed to save questions to paper'
      });
    }

    res.json({
      success: true,
      paper: populatedPaper,
      questionCount: selectedQuestions.length,
      message: `Paper "${config.title}" created successfully with ${selectedQuestions.length} questions`
    });
  } catch (error) {
    console.error('Error generating paper:', error);
    res.status(500).json({ 
      success: false, 
      message: error.message || 'Error generating paper',
      error: error.toString()
    });
  }
});

// Get all papers including files from ml/papers directory
// NOTE: This returns ALL papers for the user regardless of course
// Frontend should use /course/:courseId instead for course-scoped results
router.get('/', auth, async (req, res) => {
  try {
    // Get papers from database using googleId
    const dbPapers = await Paper.find({ createdBy: req.user.googleId || req.user._id })
      .populate('questions')
      .sort('-createdAt');
    
    console.log(`[WARNING] GET /api/papers called - returning ALL papers (${dbPapers.length}) for user ${req.user.googleId || req.user._id}. Frontend should use /course/:courseId for course filtering.`);

    // Get papers from ml/papers directory
    const papersDir = path.join(__dirname, '../../ml/papers');
    let filePapers = [];
    
    try {
      const files = await fs.readdir(papersDir);
      filePapers = files
        .filter(file => file.endsWith('.pdf'))
        .map(file => ({
          _id: file,
          title: file.replace('.pdf', ''),
          type: 'pdf',
          source: 'file',
          createdAt: new Date()
        }));
    } catch (error) {
      console.error('Error reading papers directory:', error);
    }

    // Combine papers from both sources
    const allPapers = [...dbPapers, ...filePapers];
    res.json(allPapers);
  } catch (error) {
    console.error('Error fetching papers:', error);
    res.status(500).json({ message: error.message });
  }
});

// Create a new paper
router.post('/', auth, async (req, res) => {
  try {
    // Get all approved questions for this course
    const questions = await Question.find({
      courseId: req.body.courseId,
      status: 'approved'
    }).sort({ createdAt: -1 });

    if (!questions.length) {
      return res.status(400).json({ 
        success: false, 
        message: 'No approved questions found for this course' 
      });
    }

    // Create the paper with all approved questions
    const paper = new Paper({
      ...req.body,
      questions: questions.map(q => q._id),
      createdBy: req.user.googleId || req.user._id
    });
    
    await paper.save();
    
    // Populate the questions before sending response
    await paper.populate('questions');
    
    res.status(201).json({ 
      success: true, 
      paper,
      questionCount: questions.length
    });
  } catch (error) {
    console.error('Error creating paper:', error);
    res.status(400).json({ 
      success: false, 
      message: error.message 
    });
  }
});

// Update a paper
router.patch('/:id', auth, async (req, res) => {
  try {
    const paper = await Paper.findOne({
      _id: req.params.id,
      createdBy: req.user._id
    });

    if (!paper) {
      return res.status(404).json({ message: 'Paper not found' });
    }

    // Don't allow updating createdBy field
    delete req.body.createdBy;
    
    Object.assign(paper, req.body);
    await paper.save();
    
    res.json(paper);
  } catch (error) {
    res.status(400).json({ message: error.message });
  }
});

// Delete a paper
router.delete('/:id', auth, async (req, res) => {
  try {
    const paper = await Paper.findOne({
      _id: req.params.id,
      createdBy: req.user._id
    });

    if (!paper) {
      return res.status(404).json({ message: 'Paper not found' });
    }

    await paper.remove();
    res.json({ message: 'Paper deleted' });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// Get papers by course route was moved earlier to avoid route conflicts

// Download a paper file
router.get('/download/:filename', auth, async (req, res) => {
  try {
    const filename = req.params.filename;
    const filePath = path.join(__dirname, '../../ml/papers', filename);
    
    try {
      await fs.access(filePath);
      res.download(filePath);
    } catch (error) {
      res.status(404).json({ message: 'Paper file not found' });
    }
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// Add route to check questions for a course
router.get('/check-questions/:courseId', auth, async (req, res) => {
  try {
    const { courseId } = req.params;
    
    // Get all questions for the course
    const questions = await Question.find({ courseId }).sort({ createdAt: -1 });
    
    // Group questions by status and type
    const questionStats = {
      total: questions.length,
      byStatus: {
        approved: questions.filter(q => q.status === 'approved').length,
        pending: questions.filter(q => q.status === 'pending').length,
        rejected: questions.filter(q => q.status === 'rejected').length
      },
      byType: {
        mcq: questions.filter(q => q.type === 'mcq').length,
        descriptive: questions.filter(q => q.type === 'descriptive').length,
        trueFalse: questions.filter(q => q.type === 'trueFalse').length,
        fillInBlanks: questions.filter(q => q.type === 'fillInBlanks').length
      },
      approvedByType: {
        mcq: questions.filter(q => q.type === 'mcq' && q.status === 'approved').length,
        descriptive: questions.filter(q => q.type === 'descriptive' && q.status === 'approved').length,
        trueFalse: questions.filter(q => q.type === 'trueFalse' && q.status === 'approved').length,
        fillInBlanks: questions.filter(q => q.type === 'fillInBlanks' && q.status === 'approved').length
      }
    };

    res.json({
      success: true,
      courseId,
      stats: questionStats,
      questions: questions.map(q => ({
        id: q._id,
        type: q.type,
        status: q.status,
        difficulty: q.difficulty,
        topic: q.topic,
        questionText: q.questionText
      }))
    });
  } catch (error) {
    console.error('Error checking questions:', error);
    res.status(500).json({ 
      success: false, 
      message: 'Error checking questions', 
      error: error.message 
    });
  }
});

// Add HTML view route
router.get('/:id/view', async (req, res) => {
  try {
    const paper = await Paper.findById(req.params.id);
    if (!paper) {
      return res.status(404).send('Paper not found');
    }

    // Generate HTML
    const html = `
      <!DOCTYPE html>
      <html>
      <head>
        <title>${paper.title} - ${paper.courseId}</title>
        <style>
          body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
          }
          .paper {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
          }
          .header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #eee;
          }
          .paper-info {
            text-align: right;
            color: #666;
            margin-bottom: 20px;
          }
          .question {
            margin-bottom: 30px;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 6px;
            border: 1px solid #eee;
          }
          .question-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            color: #2196f3;
          }
          .options {
            margin-left: 20px;
          }
          .option {
            margin: 10px 0;
            padding: 10px;
            background: white;
            border: 1px solid #ddd;
            border-radius: 4px;
          }
          .answer-space {
            margin-top: 15px;
            padding: 15px;
            background: white;
            border: 1px dashed #ccc;
            min-height: 100px;
          }
          .explanation {
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px dashed #ddd;
            color: #666;
            font-style: italic;
          }
        </style>
      </head>
      <body>
        <div class="paper">
          <div class="header">
            <h1>${paper.title}</h1>
            <h2>${paper.courseId}</h2>
          </div>
          
          <div class="paper-info">
            <p>Total Marks: ${paper.totalMarks}</p>
            <p>Duration: ${paper.duration} minutes</p>
          </div>

          ${paper.questions.map((q, index) => `
            <div class="question">
              <div class="question-header">
                <div>Question ${index + 1}</div>
                <div>${q.marks || 2} marks</div>
              </div>
              
              <div class="question-text">
                ${q.questionText}
              </div>
              
              ${q.type === 'mcq' ? `
                <div class="options">
                  ${q.options.map((opt, i) => `
                    <div class="option">
                      ${String.fromCharCode(65 + i)}. ${opt.text}
                    </div>
                  `).join('')}
                </div>
              ` : `
                <div class="answer-space">
                  Answer space
                </div>
              `}
              
              ${q.explanation ? `
                <div class="explanation">
                  <strong>Explanation:</strong> ${q.explanation}
                </div>
              ` : ''}
            </div>
          `).join('')}
        </div>
      </body>
      </html>
    `;

    res.send(html);
  } catch (error) {
    console.error('Error rendering paper:', error);
    res.status(500).send('Error rendering paper');
  }
});

module.exports = router; 