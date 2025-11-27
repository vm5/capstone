import React, { useEffect, useState } from 'react';
import { getLeaderboard, getTopStreaks } from '../../services/api';

export default function Leaderboard() {
  const [rows, setRows] = useState([]);
  const [streaks, setStreaks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const load = async () => {
      try {
        setLoading(true);
        setError(null);
        const [lb, st] = await Promise.all([
          getLeaderboard().catch(() => null),
          getTopStreaks().catch(() => null)
        ]);

        // Map streaks for quick lookup
        const streakMap = (st || []).reduce((m, s) => {
          const key = (s.email || s.name || '').toLowerCase();
          m[key] = s.streak || 0;
          return m;
        }, {});

        if (lb && lb.length) {
          const augmented = lb.map(r => ({
            ...r,
            streak: r.streak ?? streakMap[(r.email || r.name || '').toLowerCase()] ?? 0
          }));
          setRows(augmented);
        } else {
          // Fallback to local current user only (user-specific stats)
          const user = JSON.parse(localStorage.getItem('user') || '{}');
          const statsKey = user.email ? `playerStats_${user.email}` : 'playerStats';
          const stats = JSON.parse(localStorage.getItem(statsKey) || '{}');
          // Try meta streak
          const metaKey = user.email ? `playerStatsMeta_${user.email}` : 'playerStatsMeta';
          const meta = JSON.parse(localStorage.getItem(metaKey) || '{}');
          setRows([{ name: user.name || user.email || 'You', xp: stats.xp || 0, level: stats.level || 1, streak: stats.streak || meta.streak || 0 }]);
        }
        if (st && st.length) setStreaks(st);
      } catch (e) {
        setError('Failed to load leaderboard');
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  if (loading) return <div className="upcoming-box">Loading leaderboard...</div>;
  if (error) return <div className="upcoming-box">{error}</div>;

  return (
    <div className="upcoming-container">
      <div className="upcoming-box">
        <h3>Leaderboard (All Active Users)</h3>
        <div className="event-item" style={{ display: 'flex', fontWeight: 600, padding: '8px 0', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
          <div style={{ flex: 2 }}>User</div>
          <div style={{ width: 90, textAlign: 'left' }}>Status</div>
          <div style={{ flex: 1, textAlign: 'right' }}>XP</div>
          <div style={{ flex: 1, textAlign: 'right' }}>Level</div>
          <div style={{ flex: 1, textAlign: 'right' }}>Streak 🔥</div>
        </div>
        {(rows || []).map((r, i) => {
          const status = (r.status || '').toLowerCase();
          const dotColor = status === 'online' ? '#4caf50' : status === 'away' ? '#ffc107' : '#9e9e9e';
          return (
            <div key={i} className="event-item isa" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ flex: 2 }}>{r.name || r.email || 'Student'}</div>
              <div style={{ width: 90, display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: dotColor }} />
                <span style={{ textTransform: 'capitalize' }}>{status || 'offline'}</span>
              </div>
              <div style={{ flex: 1, textAlign: 'right' }}>{r.xp ?? 0}</div>
              <div style={{ flex: 1, textAlign: 'right' }}>L{r.level ?? 1}</div>
              <div style={{ flex: 1, textAlign: 'right' }}>{r.streak ?? 0}</div>
            </div>
          );
        })}
      </div>

      <div className="upcoming-box">
        <h3>Top Streaks</h3>
        {(streaks || []).map((s, i) => (
          <div key={i} className="event-item quiz" style={{ display: 'flex', justifyContent: 'space-between' }}>
            <div>{i + 1}. {s.name || s.email || 'Student'}</div>
            <div>🔥 {s.streak} days</div>
          </div>
        ))}
      </div>
    </div>
  );
}


