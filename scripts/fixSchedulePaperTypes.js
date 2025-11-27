require('dotenv').config({ path: '../.env' });
const mongoose = require('mongoose');
const Schedule = require('../models/Schedule');
const Paper = require('../models/Paper');
const Question = require('../models/Question');

// MongoDB connection string - match the one from server/index.js
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/pesuprep';

async function fixSchedulePaperTypes() {
  let connection;
  try {
    // Connect to MongoDB
    connection = await mongoose.connect(MONGODB_URI, {
      useNewUrlParser: true,
      useUnifiedTopology: true,
      serverSelectionTimeoutMS: 30000,
      socketTimeoutMS: 90000,
      retryWrites: true,
      retryReads: true,
      w: 'majority'
    });
    console.log('Connected to MongoDB');

    // Get direct access to the MongoDB collections
    const scheduleCollection = mongoose.connection.db.collection('schedules');
    const paperCollection = mongoose.connection.db.collection('papers');

    // Find all ISA schedules that either have no paperType or have paperType: 'pdf'
    const schedules = await scheduleCollection.find({
      type: 'ISA',
      $or: [
        { paperType: { $exists: false } },
        { paperType: 'pdf' }
      ]
    }).toArray();

    console.log(`Found ${schedules.length} ISA schedules to update`);

    // Update each schedule
    for (const schedule of schedules) {
      try {
        // If paperId is a PDF file path, create a new Paper document
        if (typeof schedule.paperId === 'string' && schedule.paperId.endsWith('.pdf')) {
          console.log(`Converting PDF paper ${schedule.paperId} to MongoDB document...`);
          
          // Create a new paper document directly in MongoDB to bypass pre-save hooks
          const paperDoc = {
            title: `ISA Paper - ${new Date(schedule.startTime).toLocaleDateString()}`,
            type: 'exam',
            questions: [],
            totalMarks: 40,
            duration: schedule.duration || 60,
            courseId: schedule.courseId,
            createdBy: schedule.createdBy,
            status: 'active',
            difficulty: {
              easy: 0,
              moderate: 0,
              challenging: 0
            },
            questionTypes: {
              mcq: 0,
              descriptive: 0
            },
            createdAt: new Date()
          };
          
          // Insert the paper document
          const result = await paperCollection.insertOne(paperDoc);
          console.log(`Created new Paper document with ID: ${result.insertedId}`);
          
          // Update the schedule directly in MongoDB
          await scheduleCollection.updateOne(
            { _id: schedule._id },
            { 
              $set: {
                paperId: result.insertedId,
                paperType: 'mongodb'
              }
            }
          );
        } else {
          // Just update the paperType
          await scheduleCollection.updateOne(
            { _id: schedule._id },
            { 
              $set: {
                paperType: 'mongodb'
              }
            }
          );
        }
        console.log(`Updated schedule: ${schedule._id}`);
      } catch (error) {
        console.error(`Error updating schedule ${schedule._id}:`, error);
        // Continue with next schedule even if this one fails
        continue;
      }
    }

    console.log('All schedules updated successfully');
  } catch (error) {
    console.error('Error fixing schedule paper types:', error);
    process.exit(1);
  } finally {
    if (connection) {
      await mongoose.connection.close();
      console.log('Disconnected from MongoDB');
    }
    process.exit(0);
  }
}

// Run the migration
fixSchedulePaperTypes().catch(console.error); 