require('dotenv').config();

// Use exact values instead of relying on env variables
const config = {
  clientId: '245320520839-rl82ksu9ic4s9skadnei2tdmlbhlocvf.apps.googleusercontent.com',
  clientSecret: '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',
  redirectUri: 'http://localhost:5000/auth/callback',
  frontendCallbackUrl: 'http://localhost:3000/auth/callback',
  scopes: [
    'https://www.googleapis.com/auth/userinfo.profile',
    'https://www.googleapis.com/auth/userinfo.email'
  ],
  auth_uri: "https://accounts.google.com/o/oauth2/v2/auth",
  token_uri: "https://oauth2.googleapis.com/token",
  auth_provider_x509_cert_url: "https://www.googleapis.com/oauth2/v1/certs",
  project_id: "pesuprep-dashboard"
};

module.exports = config; 