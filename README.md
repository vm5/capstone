 ML-Powered Question Generation & Personalization Platform
Tech Stack: MERN (MongoDB, Express, React, Node.js), Hugging Face Transformers, DistilGPT-2 (fine-tuned), Retrieval-Augmented Generation (RAG)

🧠 Overview
This project is an intelligent, full-stack web platform that enables dynamic test creation for educators and personalized learning paths for students. By leveraging a fine-tuned DistilGPT-2 model with Retrieval-Augmented Generation (RAG), the system can generate high-quality, contextually relevant questions tailored to various subjects, topics, and difficulty levels.

Designed using the MERN stack, this solution ensures seamless educator-student interactions through customizable question banks, adaptive assessments, and real-time performance tracking.

🚀 Key Features
✍️ AI-Generated Questions using fine-tuned DistilGPT-2 with RAG for context-based generation

🎯 Difficulty Level Control: Easy, Medium, Hard

🏷️ Topic Tagging & Filtering for precise test targeting

🧩 Adaptive Test Paths tailored to each student’s progress

📊 Performance Insights for students and educators

🤝 Educator-Student Interface with shared assessment access and instant feedback

🛠️ Tech Stack
Layer	Technology
Frontend	React, Tailwind CSS (optional)
Backend	Node.js, Express.js
Database	MongoDB (Mongoose ODM)
AI Models	Hugging Face Transformers (DistilGPT-2, RAG)
Deployment	Docker / Vercel / Heroku (optional)

🧱 System Architecture
csharp
Copy
Edit
[Educator Input / Student Profile]
            ↓
       [React Frontend]
            ↓
[Express API ←→ MongoDB ←→ AI Model Service]
            ↓
    [DistilGPT-2 + RAG Pipeline]
            ↓
   [Question Set Generation Engine]
            ↓
  [Personalized Assessments / Reports]
🧪 Installation & Setup
1. Clone the repository
bash
Copy
Edit
git clone https://github.com/your-username/ai-question-platform.git
cd ai-question-platform
2. Install dependencies
Backend

bash
Copy
Edit
cd server
npm install
Frontend

bash
Copy
Edit
cd client
npm install
3. Run the project
In separate terminals:

bash
Copy
Edit
# Backend
cd server
npm start

# Frontend
cd client
npm start
4. Start the AI model service
Set up and run your DistilGPT-2 + RAG model (local or Hugging Face Inference API):

bash
Copy
Edit
# Example using Python + Transformers
python ai_service/generate_questions.py
🧠 AI Model Details
🔹 DistilGPT-2 (Fine-Tuned)
Fine-tuned on educational question-answer pairs

Handles MCQs, Fill-in-the-blanks, and short answers

🔹 Retrieval-Augmented Generation (RAG)
Enhances factual correctness by retrieving support text from a document store (e.g., FAISS or ElasticSearch)

Improves contextual relevance of generated questions

💡 How It Works
Educator Inputs Topic + Difficulty

System Retrieves Context (via RAG)

DistilGPT-2 Generates Questions

Tagged & Stored in MongoDB

Students Take Tests → Platform Adapts Based on Performance

📁 Project Structure
bash
Copy
Edit
ai-question-platform/
├── client/                # React Frontend
├── server/                # Express API & MongoDB Logic
├── ai_service/            # ML model interface (DistilGPT-2 + RAG)
├── README.md
└── .env                   # API keys, DB URI, etc.
📊 Sample Use Cases
Educators can create assessments in seconds, filtered by topic and difficulty.

Students receive adaptive quizzes that evolve with their learning pace.

Administrators gain analytics on student progress and content performance.
