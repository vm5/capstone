const express = require('express');
const passport = require('passport');
const GoogleStrategy = require('passport-google-oauth20').Strategy;
const jwt = require('jsonwebtoken');
const router = express.Router();
const oauth2Config = require('../config/oauth2.config');

// Configure Passport's Google Strategy
passport.use(new GoogleStrategy({
    clientID: oauth2Config.clientId,
    clientSecret: oauth2Config.clientSecret,
    callbackURL: oauth2Config.redirectUri,
    passReqToCallback: true,
    scope: ['profile', 'email']
  },
  async (req, accessToken, refreshToken, profile, done) => {
    try {
      // Get role from state parameter
      const role = req.query.state || 'student';
      const email = profile.emails[0].value;

      // For students, allow all valid PES email domains
      if (role === 'student') {
        const pesEmailPatterns = [
          '@pes.edu',
          '@pesu.pes.edu',
          '@pesu.edu.in',
          '@pes.edu.in',
          '@pesuniversity.edu.in',
          '@pesuacademy.com'
        ];
        if (!pesEmailPatterns.some(pattern => email.toLowerCase().endsWith(pattern))) {
          const error = new Error('Invalid email domain for student');
          error.status = 403;
          error.details = 'Students must use their PES University email address. Valid domains: ' + pesEmailPatterns.join(', ');
          return done(error);
        }
      }
      // For teachers, allow both PES and personal emails
      // No additional validation needed as they can use any email

      const user = {
        googleId: profile.id,
        email: email,
        name: profile.displayName,
        role: role,
        picture: profile.photos[0].value
      };
      return done(null, user);
    } catch (error) {
      return done(error);
    }
  }
));

// Serialize user for the session
passport.serializeUser((user, done) => {
  done(null, user);
});

// Deserialize user from the session
passport.deserializeUser((user, done) => {
  done(null, user);
});

// Initialize Google OAuth login
router.get('/google', (req, res, next) => {
  const { role } = req.query;
  if (!role) {
    return res.redirect(`${oauth2Config.frontendCallbackUrl}?error=${encodeURIComponent('Role is required')}`);
  }

  passport.authenticate('google', {
    scope: ['profile', 'email'],
    state: role,
    prompt: 'select_account',
    accessType: 'offline'
  })(req, res, next);
});

// Google OAuth callback
router.get('/callback', 
  passport.authenticate('google', { 
    session: false,
    failureRedirect: oauth2Config.frontendCallbackUrl,
    failureMessage: true
  }),
  (req, res) => {
    try {
      const { user } = req;
      
      // Create JWT token
      const token = jwt.sign(
        { 
          userId: user.googleId,
          email: user.email, 
          name: user.name, 
          role: user.role,
          picture: user.picture 
        },
        process.env.JWT_SECRET || 'your-secret-key',
        { expiresIn: '24h' }
      );

      // Redirect to frontend with token
      const redirectUrl = new URL(oauth2Config.frontendCallbackUrl);
      redirectUrl.searchParams.append('token', token);
      redirectUrl.searchParams.append('user', encodeURIComponent(JSON.stringify(user)));
      
      res.redirect(redirectUrl.toString());
    } catch (error) {
      console.error('Auth callback error:', error);
      res.redirect(`${oauth2Config.frontendCallbackUrl}?error=${encodeURIComponent('Authentication failed')}`);
    }
  }
);

// Verify token endpoint
router.post('/verify', (req, res) => {
  try {
    const token = req.headers.authorization?.split(' ')[1];
    if (!token) {
      return res.status(401).json({ error: 'No token provided' });
    }

    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'your-secret-key');
    res.json({ user: decoded });
  } catch (error) {
    res.status(401).json({ error: 'Invalid token' });
  }
});

module.exports = router; 