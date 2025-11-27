const jwt = require('jsonwebtoken');
const mongoose = require('mongoose');

const auth = async (req, res, next) => {
  try {
    const token = req.headers.authorization?.split(' ')[1];
    
    if (!token) {
      return res.status(401).json({ message: 'No token provided' });
    }

    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'your-secret-key');
    
    // Set user info from token
    req.user = {
      _id: decoded.userId,
      googleId: decoded.googleId,  // Make sure we're including googleId
      email: decoded.email,
      name: decoded.name,
      role: decoded.role,
      picture: decoded.picture
    };
    
    next();
  } catch (error) {
    console.error('Auth middleware error:', error);
    res.status(401).json({ message: 'Invalid token' });
  }
};

module.exports = auth; 