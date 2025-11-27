from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import pytesseract
from PIL import Image
import fitz
import json

class QuestionGenerator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.model = AutoModelForSequenceClassification.from_pretrained("distilgpt2")
        self.vectorizer = TfidfVectorizer(max_features=1000)
        
    # ... (rest of the QuestionGenerator class implementation from your app.py)
