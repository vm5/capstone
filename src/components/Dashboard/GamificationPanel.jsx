import React, { useEffect } from 'react';

// A focused, single-source-of-truth gamification panel.
// - Uses parent `playerStats` state for level / xp / streak
// - Persists via user-specific `playerStats_${email}` localStorage key
// - Maintains daily streak without duplicating unrelated data
export default function GamificationPanel({ playerStats, setPlayerStats, userEmail }) {
  // Update daily streak once per day based on lastActiveDate (user-specific)
  useEffect(() => {
    if (!userEmail) return; // Wait for user email
    
    try {
      const metaKey = userEmail ? `playerStatsMeta_${userEmail}` : 'playerStatsMeta';
      const meta = JSON.parse(localStorage.getItem(metaKey) || '{}');
      const today = new Date();
      const todayKey = today.toISOString().slice(0, 10); // YYYY-MM-DD

      if (meta.lastActiveDate === todayKey) {
        return; // already updated today
      }

      if (!meta.lastActiveDate) {
        // first day seen
        setPlayerStats(prev => ({ ...prev, streak: 1 }));
        localStorage.setItem(metaKey, JSON.stringify({ lastActiveDate: todayKey }));
        return;
      }

      const last = new Date(meta.lastActiveDate + 'T00:00:00');
      const diffDays = Math.floor((today - last) / (1000 * 60 * 60 * 24));

      if (diffDays === 1) {
        // consecutive day → increment streak
        setPlayerStats(prev => ({ ...prev, streak: (prev.streak || 0) + 1 }));
      } else if (diffDays > 1) {
        // missed a day → reset streak
        setPlayerStats(prev => ({ ...prev, streak: 0 }));
      }

      localStorage.setItem(metaKey, JSON.stringify({ lastActiveDate: todayKey }));
    } catch (e) {
      // Non-fatal; do not block UI
      // console.warn('Streak update failed', e);
    }
  }, [setPlayerStats, userEmail]);

  const progressPct = Math.max(0, Math.min(100,
    (playerStats.xp / playerStats.nextLevelXp) * 100
  ));

  return (
    <div className="xp-level-section">
      <h2 className="xp-level-title">
        Level <span>{playerStats.level}</span>
      </h2>

      <div className="level-info">
        <div className="level-badge">
          Current Level: {playerStats.level}
        </div>
        <div className="next-reward">
          Next Reward at Level {playerStats.level + 1}
        </div>
      </div>

      <div className="level-progress">
        <div
          className="level-progress-bar"
          style={{ width: `${progressPct}%` }}
        />
        <div className="level-progress-text">
          {playerStats.xp} / {playerStats.nextLevelXp} XP
        </div>
      </div>

      <div className="xp-milestones">
        <span>Level {playerStats.level}</span>
        <span>{playerStats.nextLevelXp - playerStats.xp} XP until next level</span>
        <span>Level {playerStats.level + 1}</span>
      </div>

      <div className="streak-counter" style={{ marginTop: '12px', display: 'inline-flex' }}>
        <span className="streak-flame">🔥</span>
        <span>{playerStats.streak || 0} day streak</span>
      </div>
    </div>
  );
}


