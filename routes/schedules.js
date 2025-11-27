const express = require('express');
const router = express.Router();
const Schedule = require('../models/Schedule');
const StudentProgress = require('../models/StudentProgress');
const Paper = require('../models/Paper');
const auth = require('../middleware/auth');
const mongoose = require('mongoose');

// Helper function to check if a string is a valid MongoDB ObjectId
const isValidObjectId = (id) => {
  try {
    return mongoose.Types.ObjectId.isValid(id);
  } catch (error) {
    return false;
  }
};

// Get all schedules
router.get('/', auth, async (req, res) => {
  try {
    const schedules = await Schedule.find()
      .populate('paperId')
      .sort({ startTime: 1 });
    res.json(schedules);
  } catch (error) {
    console.error('Error fetching schedules:', error);
    res.status(500).json({ message: error.message });
  }
});

// Get a specific schedule
router.get('/:id', auth, async (req, res) => {
  try {
    const schedule = await Schedule.findById(req.params.id).populate('paperId');
    if (!schedule) {
      return res.status(404).json({ message: 'Schedule not found' });
    }
    res.json(schedule);
  } catch (error) {
    console.error('Error fetching schedule:', error);
    res.status(500).json({ message: error.message });
  }
});

// Create a new schedule
router.post('/', auth, async (req, res) => {
  try {
    // Validate paper exists if it's an ISA exam
    if (req.body.type === 'ISA') {
      if (!req.body.paperId) {
        return res.status(400).json({ message: 'Paper ID is required for ISA exams' });
      }
      const paper = await Paper.findById(req.body.paperId);
      if (!paper) {
        return res.status(404).json({ message: 'Paper not found' });
      }
    }

    // Parse dates and ensure they're in UTC
    const startTime = new Date(req.body.startTime);
    const endTime = new Date(req.body.endTime);

    // Log date information for debugging
    console.log('Schedule creation dates:', {
      receivedStartTime: req.body.startTime,
      receivedEndTime: req.body.endTime,
      parsedStartTime: startTime.toISOString(),
      parsedEndTime: endTime.toISOString(),
      currentServerTime: new Date().toISOString()
    });

    // Create schedule object with base properties
    const scheduleData = {
      ...req.body,
      startTime,
      endTime,
      createdBy: req.user.googleId || req.user._id
    };

    // Add paperType only for ISA exams
    if (req.body.type === 'ISA') {
      scheduleData.paperType = 'mongodb';
    }

    const schedule = new Schedule(scheduleData);
    await schedule.save();
    
    // Populate paper details before sending response
    await schedule.populate('paperId');
    res.status(201).json(schedule);
  } catch (error) {
    console.error('Error creating schedule:', error);
    res.status(400).json({ message: error.message });
  }
});

// Submit exam answers
router.post('/:id/submit', auth, async (req, res) => {
  try {
    const { answers, timeTaken, questionTimes } = req.body;
    const schedule = await Schedule.findById(req.params.id);

    if (!schedule) {
      return res.status(404).json({ message: 'Schedule not found' });
    }

    // Get student's progress
    let studentProgress = await StudentProgress.findOne({
      studentId: req.user._id,
      courseId: schedule.courseId
    });

    if (!studentProgress) {
      studentProgress = new StudentProgress({
        studentId: req.user._id,
        courseId: schedule.courseId,
        learningZone: 'EASY' // Default to easy if no diagnostic test taken
      });
    }

    // Calculate scores for each difficulty level
    const scores = {
      easy: { scored: 0, total: 0 },
      medium: { scored: 0, total: 0 },
      hard: { scored: 0, total: 0 }
    };

    // Evaluate answers
    schedule.questions.forEach(question => {
      const studentAnswer = answers[question._id];
      const maxScore = question.marks;
      let score = 0;

      if (question.type === 'mcq') {
        score = studentAnswer === question.correctAnswer ? maxScore : 0;
      } else {
        // For descriptive questions, implement manual scoring or AI-based scoring
        score = 0; // Default to 0 for now
      }

      // Add to appropriate difficulty level scores
      if (question.difficultyLevel <= 3) {
        scores.easy.scored += score;
        scores.easy.total += maxScore;
      } else if (question.difficultyLevel <= 6) {
        scores.medium.scored += score;
        scores.medium.total += maxScore;
      } else {
        scores.hard.scored += score;
        scores.hard.total += maxScore;
      }
    });

    // Calculate weighted scores
    const weightedScores = studentProgress.calculateWeightedScore(scores);

    // Update student's learning zone based on performance
    studentProgress.updateLearningZone(weightedScores.weightedTotalScore);

    // Add to exam history
    studentProgress.examHistory.push({
      examId: schedule._id,
      scores: weightedScores.scores,
      totalScore: weightedScores.totalScore,
      weightedTotalScore: weightedScores.weightedTotalScore,
      completedAt: new Date()
    });

    await studentProgress.save();

    res.json({
      scores: weightedScores.scores,
      totalScore: weightedScores.totalScore,
      weightedTotalScore: weightedScores.weightedTotalScore,
      newLearningZone: studentProgress.learningZone,
      timeBonus: timeTaken < schedule.duration * 60
    });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// Update a schedule
router.patch('/:id', auth, async (req, res) => {
  try {
    const schedule = await Schedule.findById(req.params.id);
    if (!schedule) {
      return res.status(404).json({ message: 'Schedule not found' });
    }

    // Only allow the creator to update the schedule
    if (schedule.createdBy.toString() !== req.user._id.toString()) {
      return res.status(403).json({ message: 'Not authorized to update this schedule' });
    }

    // For ISA exams, ensure paperType is always 'mongodb'
    if (req.body.type === 'ISA' || schedule.type === 'ISA') {
      req.body.paperType = 'mongodb';
    }

    Object.assign(schedule, req.body);
    await schedule.save();
    res.json(schedule);
  } catch (error) {
    console.error('Error updating schedule:', error);
    res.status(400).json({ message: error.message });
  }
});

// Delete a schedule
router.delete('/:id', auth, async (req, res) => {
  try {
    const schedule = await Schedule.findById(req.params.id);
    if (!schedule) {
      return res.status(404).json({ message: 'Schedule not found' });
    }

    // Only allow the creator to delete the schedule
    if (schedule.createdBy.toString() !== req.user._id.toString()) {
      return res.status(403).json({ message: 'Not authorized to delete this schedule' });
    }

    await schedule.deleteOne();
    res.json({ message: 'Schedule deleted successfully' });
  } catch (error) {
    console.error('Error deleting schedule:', error);
    res.status(500).json({ message: error.message });
  }
});

module.exports = router; 