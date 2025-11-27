const mongoose = require('mongoose');

const PlayerStatsSchema = new mongoose.Schema({
  email: { type: String, required: true, index: true, unique: true },
  stats: {
    level: { type: Number, default: 1 },
    xp: { type: Number, default: 0 },
    nextLevelXp: { type: Number, default: 1000 },
    completedTopics: { type: [Number], default: [] },
    achievements: { type: [String], default: [] },
    inventory: { type: [String], default: [] },
    streak: { type: Number, default: 0 },
    coins: { type: Number, default: 0 }
  },
  updatedAt: { type: Date, default: Date.now }
});

PlayerStatsSchema.index({ email: 1 });

module.exports = mongoose.model('PlayerStats', PlayerStatsSchema);


