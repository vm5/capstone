from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from transformers import pipeline, set_seed, AutoTokenizer, AutoModelForSequenceClassification
import re
import random
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
import os
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from datasets import Dataset
from collections import deque

# Set random seed for reproducibility
set_seed(42)
np.random.seed(42)

# Load the fine-tuned model
question_generator = pipeline(
    "text-generation",
    model="C:/Users/Admin/Music/dashboard/dashboard-app/ml/fine_tuned_distilgpt2",
    tokenizer="distilgpt2",
    device=-1,
    truncation=True,
    max_length=512,
    num_return_sequences=1,
    temperature=0.8,
    top_p=0.95,
    top_k=50,
    repetition_penalty=1.2
)

# Initialize BERT classifier for ML topics
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Constants for ML models
MAX_WORDS = 10000
MAX_LEN = 200
EMBEDDING_DIM = 100

class AdaptiveMLClassifier:
    def __init__(self, max_history=100):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            'bert-base-uncased', 
            num_labels=2
        ).to(self.device)
        
        # Initialize LoRA configuration
        self.peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,  # rank of LoRA update matrices
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            target_modules=["query", "value"]
        )
        
        # Create PEFT model
        self.model = get_peft_model(self.base_model, self.peft_config)
        
        # Initialize context history
        self.context_history = deque(maxlen=max_history)
        self.labels_history = deque(maxlen=max_history)
        
        # TF-IDF vectorizer for keyword analysis
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # Initialize with sample data
        self._initialize_with_samples()

    def _initialize_with_samples(self):
        """Initialize the model with sample texts"""
        sample_texts = [
            ("neural networks deep learning backpropagation", "neural_networks"),
            ("machine learning classification regression", "general_ml"),
            ("CNN RNN LSTM deep learning", "neural_networks"),
            ("decision trees random forests clustering", "general_ml"),
        ]
        
        for text, label in sample_texts:
            self.context_history.append(text)
            self.labels_history.append(label)
        
        self.tfidf.fit([text for text, _ in sample_texts])
        self._update_model()

    def _update_model(self):
        """Update the model with current context history"""
        if len(self.context_history) < 2:
            return

        # Prepare training data
        train_texts = list(self.context_history)
        train_labels = [1 if label == "neural_networks" else 0 for label in self.labels_history]
        
        # Create dataset
        train_encodings = self.tokenizer(
            train_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask'],
            'labels': train_labels
        })

        # Training arguments
        training_args = {
            'per_device_train_batch_size': 4,
            'num_train_epochs': 3,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
        }

        # Fine-tune the model
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=training_args['learning_rate'])
        
        for epoch in range(training_args['num_train_epochs']):
            for batch in train_dataset.shuffle().iter(batch_size=training_args['per_device_train_batch_size']):
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    labels=batch['labels'].to(self.device)
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()

    def predict(self, text):
        """Predict using the adaptive model and keyword verification"""
        # Add to context history
        self.context_history.append(text)
        
        # Preprocess text
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Get model prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            model_pred = probs[0].cpu().numpy()

        # Keyword-based verification
        text_lower = text.lower()
        nn_count = sum(text_lower.count(term.lower()) for term in ML_CATEGORIES['neural_networks'])
        ml_count = sum(text_lower.count(term.lower()) for term in ML_CATEGORIES['general_ml'])
        
        # Combine model prediction with keyword counts
        if nn_count > ml_count * 1.5:
            prediction = 'neural_networks'
        elif ml_count > nn_count * 1.5:
            prediction = 'general_ml'
        else:
            prediction = 'neural_networks' if model_pred[0] > 0.5 else 'general_ml'
        
        # Update context with prediction
        self.labels_history.append(prediction)
        
        # Periodically update the model
        if len(self.context_history) % 10 == 0:  # Update every 10 predictions
            self._update_model()
        
        return prediction

    def get_key_terms(self, text):
        """Extract key terms using TF-IDF"""
        tfidf_matrix = self.tfidf.transform([text])
        feature_names = self.tfidf.get_feature_names_out()
        return dict(zip(feature_names, tfidf_matrix.toarray()[0]))

# Update ML_CATEGORIES with framework-specific terms
ML_CATEGORIES = {
    'neural_networks': [
        'neural network', 'deep learning', 'activation function', 'backpropagation',
        'neurons', 'layers', 'perceptron', 'CNN', 'RNN', 'LSTM', 'GRU', 'ANN', 'DNN',
        'convolutional', 'recurrent', 'transformer', 'attention mechanism',
        'pytorch', 'deep neural network', 'weight initialization', 'batch normalization'
    ],
    'general_ml': [
        'machine learning', 'supervised learning', 'unsupervised learning',
        'classification', 'regression', 'clustering', 'decision trees',
        'random forest', 'support vector machine', 'gradient descent',
        'scikit-learn', 'sklearn', 'xgboost', 'lightgbm', 'cross-validation',
        'feature engineering', 'ensemble methods', 'bagging', 'boosting'
    ]
}

def classify_text_with_bert(text):
    """Use BERT to classify the text into ML categories."""
    # Prepare text chunks if the text is too long
    max_length = 512
    chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
    
    # Process each chunk
    results = []
    for chunk in chunks:
        # Tokenize the text
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512, padding=True)
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            results.append(predictions[0].numpy())
    
    # Average the results if there were multiple chunks
    final_prediction = np.mean(results, axis=0)
    
    # Count keyword occurrences for each category
    category_scores = {
        'neural_networks': 0,
        'general_ml': 0
    }
    
    text_lower = text.lower()
    for category, keywords in ML_CATEGORIES.items():
        for keyword in keywords:
            if keyword in text_lower:
                category_scores[category] += text_lower.count(keyword)
    
    # Combine BERT predictions with keyword counts
    if category_scores['neural_networks'] > category_scores['general_ml']:
        return 'neural_networks'
    elif category_scores['neural_networks'] < category_scores['general_ml']:
        return 'general_ml'
    else:
        # If keyword counts are equal, use BERT prediction
        return 'neural_networks' if final_prediction[0] > final_prediction[1] else 'general_ml'

def extract_meaningful_topic(text):
    """Enhanced topic extraction using BERT and keyword analysis."""
    # First use BERT to classify the general topic
    main_category = classify_text_with_bert(text)
    
    # Extract specific technical terms based on the category
    technical_terms = ML_CATEGORIES[main_category]
    
    # Count occurrences of each term
    term_counts = {}
    text_lower = text.lower()
    for term in technical_terms:
        count = text_lower.count(term)
        if count > 0:
            term_counts[term] = count
    
    # If we found specific terms, return the most frequent one
    if term_counts:
        return max(term_counts.items(), key=lambda x: x[1])[0]
    
    # Fallback to the main category
    return 'neural networks' if main_category == 'neural_networks' else 'machine learning'

# Initialize the classifier
adaptive_classifier = AdaptiveMLClassifier()

def generate_questions(text):
    """Generate questions using adaptive classification"""
    if not text.strip():
        return {"error": "No input text provided."}

    try:
        # Use adaptive classification
        content_type = adaptive_classifier.predict(text)
        print(f"Detected content type: {content_type}")
        
        # Extract topic and relevant terms
        topic = extract_meaningful_topic(text)
        print(f"Extracted topic: {topic}")
        
        # Get key terms using the adaptive classifier
        key_terms = adaptive_classifier.get_key_terms(text)
        
        if content_type == 'neural_networks':
            neural_network_terms = extract_neural_network_concepts(text)
            mcqs = generate_neural_network_mcqs(text, neural_network_terms)
            desc_questions = generate_neural_network_descriptive(text, neural_network_terms)
        else:
            ml_terms = extract_machine_learning_concepts(text)
            mcqs = generate_machine_leaning_mcqs(text, ml_terms)
            desc_questions = generate_machine_learning_descriptive(text, ml_terms)
        
        return {
            "MCQs": mcqs[:4],
            "Descriptive": desc_questions[:4],
            "detected_topic": topic,
            "content_type": content_type,
            "key_terms": dict(sorted(key_terms.items(), key=lambda x: x[1], reverse=True)[:5])
        }

    except Exception as e:
        print(f"❌ Error in Question Generation: {e}")
        return {"error": f"An error occurred during question generation: {str(e)}"}

def extract_machine_learning_concepts(text):
    ml_terms = {
         "components": ["feature", "label", "dataset", "training data", "test data", "model", "parameter", "hyperparameter", "loss function", "evaluation metric"],
    "types": ["supervised learning", "unsupervised learning", "reinforcement learning", "semi-supervised learning", "self-supervised learning"],
    "concepts": ["gradient descent", "stochastic gradient descent", "learning rate", "epoch", "batch size", "overfitting", "underfitting", "regularization", "cross-validation"],
    "evaluation": ["accuracy", "precision", "recall", "F1-score", "ROC-AUC", "mean squared error", "R-squared"],
    "applications": ["classification", "regression", "clustering", "dimensionality reduction", "anomaly detection", "recommendation systems"]
}
    found_terms = {category: [] for category in ml_terms}
    
    # Find which terms appear in the text
    for category, terms in ml_terms.items():
        for term in terms:
            if term.lower() in text.lower():
                found_terms[category].append(term)
    
    # Also extract any sentences about neural networks
    sentences = text.split('.')
    ml_sentences = [s.strip() for s in sentences if any(term.lower() in s.lower() for term in ["neural network", "neuron", "ANN", "DNN"])]
    
    found_terms["sentences"] = ml_sentences
    return found_terms

def generate_machine_leaning_mcqs(text, nn_terms):
    mcqs = []
    
    # Pre-defined question templates for neural networks
    question_templates = [
    # Structure questions
    {
        "template": "What is a feature in machine learning?",
        "options": [
            "An individual measurable property or characteristic of the data",
            "A type of machine learning algorithm",
            "The final output of a model",
            "The number of training iterations"
        ],
        "answer": "A"
    },
    {
        "template": "Which of the following is NOT a type of machine learning?",
        "options": [
            "Supervised learning",
            "Unsupervised learning",
            "Quantum learning",
            "Reinforcement learning"
        ],
        "answer": "C"
    },
    {
        "template": "What is the primary goal of supervised learning?",
        "options": [
            "To find hidden patterns in unlabeled data",
            "To predict labels based on input data",
            "To reward an agent for taking the best actions",
            "To remove noisy data from a dataset"
        ],
        "answer": "B"
    },
    # Functioning questions
    {
        "template": "What does overfitting in machine learning mean?",
        "options": [
            "The model performs well on training data but poorly on new data",
            "The model performs equally well on training and test data",
            "The model does not learn anything from the training data",
            "The model has too few parameters to capture the data patterns"
        ],
        "answer": "A"
    },
    {
        "template": "Which metric is commonly used to evaluate a classification model?",
        "options": [
            "Mean Squared Error",
            "R-squared",
            "Accuracy",
            "Silhouette Score"
        ],
        "answer": "C"
    },
    {
        "template": "What is the purpose of cross-validation in machine learning?",
        "options": [
            "To improve the model's performance by reducing bias and variance",
            "To increase the size of the training dataset",
            "To eliminate the need for a test dataset",
            "To ensure the model only learns from noisy data"
        ],
        "answer": "A"
    },
    {
        "template": "Which of the following is an example of unsupervised learning?",
        "options": [
            "Spam email classification",
            "Customer segmentation",
            "Stock price prediction",
            "Medical diagnosis based on past cases"
        ],
        "answer": "B"
    },
    # Applications
    {
        "template": "Which of the following is NOT a common application of machine learning?",
        "options": [
            "Sorting a list of numbers",
            "Fraud detection",
            "Recommendation systems",
            "Image recognition"
        ],
        "answer": "A"
    },
    {
        "template": "What is the key characteristic of reinforcement learning?",
        "options": [
            "The model learns by receiving rewards and penalties",
            "The model is trained only on labeled data",
            "The model groups data without predefined categories",
            "The model only works with numerical data"
        ],
        "answer": "A"
    }
]
    # Shuffle and select questions
    import random
    random.shuffle(question_templates)
    
    # Get 4 questions with different answer patterns
    selected_templates = question_templates[:8]
    answer_patterns = ["A", "B", "C", "D"]
    
    for i, template in enumerate(selected_templates[:4]):
        # Modify the correct answer to create varied patterns (A, B, C, D)
        correct_idx = ord(template["answer"]) - ord("A")
        new_correct_idx = i % 4
        
        # Rearrange options to make different answers correct
        options = template["options"].copy()
        correct_option = options[correct_idx]
        
        # If we need to make a different answer correct
        if correct_idx != new_correct_idx:
            # Swap the current correct answer with the option at the new position
            options[correct_idx] = options[new_correct_idx]
            options[new_correct_idx] = correct_option
        
        mcqs.append({
            "question": template["template"],
            "options": options,
            "answer": answer_patterns[new_correct_idx]
        })
    
    return mcqs

def generate_machine_learning_descriptive(text, nn_terms):
    descriptive_templates = [
    "Explain the difference between supervised, unsupervised, and reinforcement learning with examples.",
    
    "Describe the steps involved in training a machine learning model from data preprocessing to evaluation.",
    
    "What are the key differences between overfitting and underfitting? How can they be prevented?",
    
    "Discuss the importance of feature selection and feature engineering in machine learning. How do they impact model performance?",
    
    "What are common challenges in machine learning model development, and how can they be addressed?",
    
    "Explain the concept of bias and variance in machine learning. How do they affect model generalization?",
    
    "Describe the different types of evaluation metrics used for classification and regression problems. Provide examples.",
    
    "How does cross-validation improve model performance and reliability? Discuss different cross-validation techniques.",
    
    "What are some real-world applications of machine learning across different industries? Provide specific use cases.",
    
    "Discuss the ethical considerations and challenges associated with machine learning, including bias and fairness."
]

    
    # Return a random selection of 4 descriptive questions
    import random
    random.shuffle(descriptive_templates)
    return descriptive_templates[:4]

def extract_neural_network_concepts(text):
    """Extract neural network specific concepts from text."""
    nn_terms = {
        "components": ["neuron", "node", "layer", "weight", "bias", "activation function", "input layer", "hidden layer", "output layer"],
        "types": ["ANN", "DNN", "CNN", "RNN", "LSTM", "GRU", "feed-forward", "recurrent neural network"],
        "concepts": ["backpropagation", "gradient descent", "learning rate", "epoch", "batch size", "overfitting", "underfitting", "regularization", "dropout"],
        "activation": ["sigmoid", "tanh", "ReLU", "Leaky ReLU", "softmax"],
        "applications": ["image recognition", "natural language processing", "speech recognition", "computer vision"]
    }
    
    found_terms = {category: [] for category in nn_terms}
    
    # Find which terms appear in the text
    for category, terms in nn_terms.items():
        for term in terms:
            if term.lower() in text.lower():
                found_terms[category].append(term)
    
    # Also extract any sentences about neural networks
    sentences = text.split('.')
    nn_sentences = [s.strip() for s in sentences if any(term.lower() in s.lower() for term in ["neural network", "neuron", "ANN", "DNN"])]
    
    found_terms["sentences"] = nn_sentences
    return found_terms

def generate_neural_network_mcqs(text, nn_terms):
    """Generate neural network specific MCQs without using 'according to the text'."""
    mcqs = []
    
    # Pre-defined question templates for neural networks
    question_templates = [
        # Structure questions
        {
            "template": "What is the most fundamental unit of a neural network?",
            "options": ["Neuron", "Layer", "Weight", "Network"],
            "answer": "A"
        },
        {
            "template": "Which of the following best describes an artificial neural network?",
            "options": [
                "A computational model inspired by the human brain's structure",
                "A statistical method for clustering data points",
                "A rule-based expert system for decision making",
                "A symbolic representation of logical relationships"
            ],
            "answer": "A"
        },
        {
            "template": "How do neurons in an artificial neural network primarily interact with each other?",
            "options": [
                "Through weighted connections",
                "Through direct binary signals",
                "Through chemical transmitters",
                "Through quantum entanglement"
            ],
            "answer": "A"
        },
        # Functioning questions
        {
            "template": "What is the primary role of activation functions in neural networks?",
            "options": [
                "To introduce non-linearity into the network's output",
                "To decrease the network's computation time",
                "To reduce the number of neurons required",
                "To eliminate the need for backpropagation"
            ],
            "answer": "A"
        },
        {
            "template": "In a deep neural network (DNN), what characterizes the hidden layers?",
            "options": [
                "They process intermediate features between input and output",
                "They are directly connected to external data sources",
                "They contain only inactive neurons until activated",
                "They store the final results of computation"
            ],
            "answer": "A"
        },
        {
            "template": "Which statement about artificial neural networks is FALSE?",
            "options": [
                "They can only solve classification problems",
                "They are inspired by biological neural systems",
                "They can learn from example data",
                "They consist of interconnected processing units"
            ],
            "answer": "A"
        },
        # Applications
        {
            "template": "Which of the following is NOT a common application of neural networks?",
            "options": [
                "Exact mathematical proof validation",
                "Image recognition",
                "Natural language processing",
                "Pattern detection"
            ],
            "answer": "A"
        },
        {
            "template": "What distinguishes deep neural networks from traditional neural networks?",
            "options": [
                "The presence of multiple hidden layers",
                "The use of only digital rather than analog computations",
                "The requirement for quantum computing hardware",
                "The absence of activation functions"
            ],
            "answer": "A"
        }
    ]
    
    # Shuffle and select questions
    import random
    random.shuffle(question_templates)
    
    # Get 4 questions with different answer patterns
    selected_templates = question_templates[:8]
    answer_patterns = ["A", "B", "C", "D"]
    
    for i, template in enumerate(selected_templates[:4]):
        # Modify the correct answer to create varied patterns (A, B, C, D)
        correct_idx = ord(template["answer"]) - ord("A")
        new_correct_idx = i % 4
        
        # Rearrange options to make different answers correct
        options = template["options"].copy()
        correct_option = options[correct_idx]
        
        # If we need to make a different answer correct
        if correct_idx != new_correct_idx:
            # Swap the current correct answer with the option at the new position
            options[correct_idx] = options[new_correct_idx]
            options[new_correct_idx] = correct_option
        
        mcqs.append({
            "question": template["template"],
            "options": options,
            "answer": answer_patterns[new_correct_idx]
        })
    
    return mcqs

def generate_neural_network_descriptive(text, nn_terms):
    """Generate descriptive questions for neural networks without 'according to the text'."""
    descriptive_templates = [
        "Explain the structure and functioning of artificial neurons. How do they process information?",
        
        "Compare and contrast artificial neural networks with biological neural networks. What similarities and differences exist in their design and operation?",
        
        "Describe the process of training a neural network. What role does backpropagation play in this process?",
        
        "What are activation functions in neural networks? Explain different types of activation functions and their purposes.",
        
        "Explain how deep neural networks differ from single-layer neural networks in terms of capabilities and applications.",
        
        "Discuss potential applications of neural networks in real-world scenarios. Provide specific examples.",
        
        "What challenges are commonly encountered when designing and training neural networks? How can these challenges be addressed?",
        
        "Explain how neural networks can be evaluated for performance. What metrics are typically used?"
    ]
    
    # Return a random selection of 4 descriptive questions
    import random
    random.shuffle(descriptive_templates)
    return descriptive_templates[:4]

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-type", "application/json")
        self.end_headers()

        if self.path == "/":
            self.wfile.write(json.dumps({"message": "✅ Server is running!"}).encode())
        elif self.path.startswith("/generate-questions"):
            sample_text = "What is artificial intelligence?"
            response = generate_questions(sample_text)
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path == "/generate-questions":
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data.decode("utf-8"))
                response = generate_questions(data["text"])
                self.wfile.write(json.dumps(response).encode())
            except json.JSONDecodeError:
                error_response = {"error": "Invalid JSON received"}
                self.wfile.write(json.dumps(error_response).encode())
            except Exception as e:
                error_response = {"error": f"An error occurred: {str(e)}"}
                self.wfile.write(json.dumps(error_response).encode())
        else:
            self.send_error(404, "Not Found")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

if __name__ == "__main__":
    server_address = ("0.0.0.0", 8000)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    print(f"✅ Server running on http://{server_address[0]}:{server_address[1]}")
    httpd.serve_forever()
