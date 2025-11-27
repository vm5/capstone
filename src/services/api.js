import axios from 'axios';

// API Configuration
const API_BASE_URL = '/api';  // This will be proxied to http://localhost:5000
const ML_BASE_URL = 'http://localhost:5000';

// Google OAuth Configuration
const GOOGLE_CLIENT_ID = '245320520839-rl82ksu9ic4s9skadnei2tdmlbhlocvf.apps.googleusercontent.com';
const REDIRECT_URI = 'http://localhost:5000/auth/callback';

// Create axios instances with default config
export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // Increased timeout to 60 seconds
  headers: {
    'Content-Type': 'application/json'
  }
});

const mlApi = axios.create({
  baseURL: ML_BASE_URL,
  timeout: 60000, // Increased timeout to 60 seconds
  headers: {
    'Content-Type': 'application/json'
  }
});

// Add request interceptor to include auth token
const addAuthToken = (config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
};

const handleAuthError = (error) => {
  console.error('API Error:', error);
  
  // Handle timeout errors specifically
  if (error.code === 'ECONNABORTED' || error.response?.status === 504) {
    throw new Error('Request timed out. Please try again.');
  }
  
  // Handle unauthorized error
  if (error.response?.status === 401) {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    window.location.href = '/';
  }

  throw new Error(error.response?.data?.message || error.message);
};

// Add interceptors to both API instances
api.interceptors.request.use(addAuthToken, (error) => Promise.reject(error));
api.interceptors.response.use((response) => response, handleAuthError);

mlApi.interceptors.request.use(addAuthToken, (error) => Promise.reject(error));
mlApi.interceptors.response.use((response) => response, handleAuthError);

export const initiateGoogleAuth = (role) => {
  // Let the backend handle the OAuth flow
  window.location.href = `${API_BASE_URL}/auth/google?role=${role}`;
};

export const handleGoogleCallback = async () => {
  try {
    // Extract token and user data from URL parameters
    const urlParams = new URLSearchParams(window.location.search);
    const token = urlParams.get('token');
    const userStr = urlParams.get('user');
    const error = urlParams.get('error');
    
    if (error) {
      throw new Error(decodeURIComponent(error));
    }

    if (!token || !userStr) {
      throw new Error('Invalid authentication response');
    }

    // Parse user data
    const user = JSON.parse(decodeURIComponent(userStr));
    
    return {
      token,
      user
    };
  } catch (error) {
    console.error('Google callback error:', error);
    throw error;
  }
};

export const registerUser = async (userData) => {
  try {
    const response = await fetch(`${API_BASE_URL}/oauth/register`, {
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

// File upload helper
export const uploadFiles = async (files) => {
  try {
    console.log('Starting file upload...', files);
    const formData = new FormData();
    
    // Append each file to formData
    for (let file of files) {
      formData.append('files[]', file);
    }

    // Log formData contents for debugging
    for (let [key, value] of formData.entries()) {
      console.log('FormData entry:', key, value instanceof File ? value.name : value);
    }

    const response = await axios.post(`${API_BASE_URL}/process-content`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      timeout: 60000, // 60 second timeout for file uploads
      onUploadProgress: (progressEvent) => {
        const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        console.log('Upload progress:', percentCompleted);
      }
    });

    console.log('Upload response:', response.data);
    return response.data;
  } catch (error) {
    console.error('Upload error:', error);
    if (error.response) {
      // Server responded with error
      throw new Error(error.response.data.error || 'Error uploading files');
    } else if (error.request) {
      // No response received
      throw new Error('Server not responding. Please check your connection and try again.');
    } else {
      // Error in request setup
      throw new Error('Error preparing file upload');
    }
  }
};

// Question generation helper
export const generateQuestions = async (content) => {
  try {
    const response = await mlApi.post('/generate-questions', { text: content });
    return response.data;
  } catch (error) {
    console.error('Question generation error:', error);
    throw new Error('Error generating questions: ' + (error.response?.data?.message || error.message));
  }
};

// Dataset upload helper
export const uploadDataset = async (files) => {
  try {
    const formData = new FormData();
    for (let file of files) {
      formData.append('file', file);
    }

    const response = await mlApi.post('/upload-dataset', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    return response.data;
  } catch (error) {
    console.error('Dataset upload error:', error);
    throw new Error('Error uploading dataset: ' + (error.response?.data?.message || error.message));
  }
};

// Schedule management helpers
export const createSchedule = async (scheduleData) => {
  try {
    const response = await api.post('/schedules', scheduleData);
    return response.data;
  } catch (error) {
    console.error('Schedule creation error:', error);
    throw new Error('Error creating schedule: ' + (error.response?.data?.message || error.message));
  }
};

export const getSchedules = async () => {
  try {
    const response = await api.get('/schedules');
    return response.data;
  } catch (error) {
    console.error('Schedule fetch error:', error);
    throw new Error('Error fetching schedules: ' + (error.response?.data?.message || error.message));
  }
};

export const updateSchedule = async (scheduleId, scheduleData) => {
  try {
    const response = await api.patch(`/schedules/${scheduleId}`, scheduleData);
    return response.data;
  } catch (error) {
    console.error('Schedule update error:', error);
    throw new Error('Error updating schedule: ' + (error.response?.data?.message || error.message));
  }
};

export const deleteSchedule = async (scheduleId) => {
  try {
    const response = await api.delete(`/schedules/${scheduleId}`);
    return response.data;
  } catch (error) {
    console.error('Schedule deletion error:', error);
    throw new Error('Error deleting schedule: ' + (error.response?.data?.message || error.message));
  }
};

// Paper management helpers
export const createPaper = async (paperData) => {
  try {
    console.log('Creating paper with data:', paperData);
    
    // Validate required fields
    if (!paperData.courseId) {
      throw new Error('Course ID is required');
    }
    
    if (!paperData.config) {
      throw new Error('Paper configuration is required');
    }

    // Get approved questions from the papers route
    console.log('Fetching approved questions for course:', paperData.courseId);
    
    const questionsResponse = await api.get(`/papers/approved-questions`, {
      params: {
        courseId: paperData.courseId
      }
    });

    // Log response for debugging
    console.log('Questions response:', questionsResponse.data);

    const approvedQuestions = questionsResponse.data || [];
    
    console.log('Found approved questions:', approvedQuestions.length);
    
    if (!approvedQuestions || approvedQuestions.length === 0) {
      throw new Error('No approved questions found. Please generate and approve questions first.');
    }

    // Check if we have enough questions of each type
    const questionTypeCounts = {};
    approvedQuestions.forEach(q => {
      questionTypeCounts[q.type] = (questionTypeCounts[q.type] || 0) + 1;
    });

    console.log('Available question counts by type:', questionTypeCounts);
    console.log('Required question counts:', paperData.config.questionTypes);

    for (const [type, count] of Object.entries(paperData.config.questionTypes)) {
      if (count > 0 && (!questionTypeCounts[type] || questionTypeCounts[type] < count)) {
        throw new Error(`Not enough approved ${type} questions. Need ${count} but only have ${questionTypeCounts[type] || 0}.`);
      }
    }

    // Validate question types
    if (!paperData.config.questionTypes || 
        typeof paperData.config.questionTypes !== 'object' ||
        Object.values(paperData.config.questionTypes).every(count => count === 0)) {
      throw new Error('At least one question type must be selected');
    }

    // Ensure all required fields are present
    const requiredFields = ['title', 'totalMarks', 'duration'];
    const missingFields = requiredFields.filter(field => !paperData.config[field]);
    if (missingFields.length > 0) {
      throw new Error(`Missing required fields: ${missingFields.join(', ')}`);
    }

    console.log('Creating paper with approved questions...');
    
    // Send questions to the server
    const response = await api.post('/papers/generate-paper', {
      ...paperData,
      type: 'exam',
      status: 'active',
      createdBy: localStorage.getItem('userId') || 'system',
      questions: approvedQuestions
    });

    console.log('Paper created successfully:', response.data);
    return response.data;
  } catch (error) {
    console.error('Paper creation error:', error);
    if (error.response?.status === 404) {
      console.error('API endpoint not found:', error.config?.url);
      throw new Error('API endpoint not found. Please check server configuration.');
    }
    if (error.response?.status === 400) {
      throw new Error(error.response.data.message || 'No approved questions found for this course. Please generate and approve questions first.');
    }
    if (error.response?.status === 500) {
      console.error('Server error details:', error.response?.data);
      throw new Error('Server error: ' + (error.response?.data?.message || 'Internal server error'));
    }
    throw new Error('Error creating paper: ' + (error.response?.data?.message || error.message));
  }
};

export const getPapers = async () => {
  try {
    const response = await api.get('/api/papers');
    return response.data || [];
  } catch (error) {
    console.error('Paper fetch error:', error);
    throw new Error('Error fetching papers: ' + (error.response?.data?.message || error.message));
  }
};

export const downloadPaper = async (filename) => {
  try {
    const response = await api.get(`/api/papers/download/${filename}`, {
      responseType: 'blob'
    });
    return response.data;
  } catch (error) {
    console.error('Error downloading paper:', error);
    throw error;
  }
};

export const getPaperById = async (paperId) => {
  try {
    const response = await api.get(`/api/papers/${paperId}`);
    return response.data;
  } catch (error) {
    console.error('Paper fetch error:', error);
    throw new Error('Error fetching paper: ' + (error.response?.data?.message || error.message));
  }
};

export const getPapersByCourse = async (courseId) => {
  try {
    // baseURL already `/api`, so request should NOT prefix with /api again
    const response = await api.get(`/papers/course/${courseId}`);
    return response.data;
  } catch (error) {
    console.error('Paper fetch error:', error);
    throw new Error('Error fetching papers: ' + (error.response?.data?.message || error.message));
  }
};

export const updatePaper = async (paperId, paperData) => {
  try {
    const response = await api.patch(`/api/papers/${paperId}`, paperData);
    return response.data;
  } catch (error) {
    console.error('Paper update error:', error);
    throw new Error('Error updating paper: ' + (error.response?.data?.message || error.message));
  }
};

export const deletePaper = async (paperId) => {
  try {
    const response = await api.delete(`/api/papers/${paperId}`);
    return response.data;
  } catch (error) {
    console.error('Paper deletion error:', error);
    throw new Error('Error deleting paper: ' + (error.response?.data?.message || error.message));
  }
};

export default api; 

// Leaderboard helpers (graceful fallbacks if backend not ready)
export const getLeaderboard = async () => {
  try {
    const res = await api.get('/leaderboard');
    return res.data?.rows || [];
  } catch (e) {
    // fallback empty to allow UI to render current user only
    return [];
  }
};

export const getTopStreaks = async () => {
  try {
    const res = await api.get('/leaderboard/top-streaks');
    return res.data?.rows || [];
  } catch (e) {
    return [];
  }
};

// Persist player stats to backend (upsert)
export const savePlayerStats = async (email, stats) => {
  try {
    await api.post('/player-stats', { email, stats });
  } catch (e) {
    // silent fail; UI will keep local copy
  }
};

export const loadPlayerStats = async (email) => {
  try {
    const res = await api.get(`/player-stats/${encodeURIComponent(email)}`);
    return res.data?.stats || null;
  } catch (e) {
    return null;
  }
};