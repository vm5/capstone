import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Header from './components/Header/Header';
import Footer from './components/Footer/Footer';
import Dashboard from './components/Dashboard/Dashboard';
import TeacherDashboard from './components/TeacherDashboard/TeacherDashboard';
import Chatbot from './components/Chatbot/Chatbot';
import ExamPage from './components/Dashboard/ExamPage';
import './App.css';

// Protected route wrapper
const ProtectedRoute = ({ element: Component, allowedRole }) => {
  const userData = localStorage.getItem('user');
  if (!userData) {
    return <Navigate to="/" replace />;
  }

  const user = JSON.parse(userData);
  if (user.role !== allowedRole) {
    return <Navigate to="/" replace />;
  }

  return <Component />;
};

// Protected exam route for students
const ProtectedExamRoute = () => {
  const userData = localStorage.getItem('user');

  if (!userData) {
    return <Navigate to="/" replace />;
  }

  const user = JSON.parse(userData);
  if (user.role !== 'student') {
    return <Navigate to="/" replace />;
  }

  return <ExamPage />;
};

// Landing page component that includes both Header and Footer
const LandingPage = () => (
  <>
    <Header />
    <Footer />
  </>
);

function App() {
  return (
    <Router>
      <div className="app-container">
        <Routes>
          {/* Main entry point with Header and Footer */}
          <Route path="/" element={<LandingPage />} />
          
          {/* Auth callback route - Header will handle the auth processing */}
          <Route path="/auth/callback" element={<LandingPage />} />
          
          {/* Protected Dashboard routes - no Footer */}
          <Route 
            path="/dashboard" 
            element={<ProtectedRoute element={Dashboard} allowedRole="student" />} 
          />
          <Route 
            path="/teacher-dashboard" 
            element={<ProtectedRoute element={TeacherDashboard} allowedRole="teacher" />} 
          />
          
          {/* Protected Exam page route */}
          <Route path="/exam" element={<ProtectedExamRoute />} />
          
          {/* Catch-all redirect to home */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>

        {/* Chatbot widget that appears on all pages */}
        <Chatbot />
      </div>
    </Router>
  );
}

export default App;