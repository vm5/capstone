const API_URL = 'http://localhost:5000/api';

// Google OAuth 2.0 Configuration
const GOOGLE_CLIENT_ID = '245320520839-rl82ksu9ic4s9skadnei2tdmlbhlocvf.apps.googleusercontent.com';
const GOOGLE_AUTH_URL = 'https://accounts.google.com/o/oauth2/v2/auth';
const REDIRECT_URI = 'http://localhost:3000/auth/callback';

export const initiateGoogleAuth = (role) => {
  const params = new URLSearchParams({
    client_id: GOOGLE_CLIENT_ID,
    redirect_uri: REDIRECT_URI,
    response_type: 'code',
    scope: 'email profile',
    prompt: 'select_account',
    access_type: 'offline',
    state: role
  });

  window.location.href = `${GOOGLE_AUTH_URL}?${params.toString()}`;
};

export const registerUser = async (userData) => {
  try {
    const response = await fetch(`${API_URL}/oauth/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(userData),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || 'Registration failed');
    }

    return await response.json();
  } catch (error) {
    console.error('Registration error:', error);
    throw error;
  }
};

export const handleGoogleCallback = async (code, role) => {
  try {
    const response = await fetch(`${API_URL}/auth/google/callback`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        code,
        role: role || new URLSearchParams(window.location.search).get('state')
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || 'Authentication failed');
    }

    const data = await response.json();
    
    if (!data.user || !data.token) {
      throw new Error('Invalid response from server');
    }

    // Save user data and token
    localStorage.setItem('user', JSON.stringify(data.user));
    localStorage.setItem('token', data.token);

    return {
      user: data.user,
      token: data.token
    };
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
};
