# ML Python Service Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    libpoppler-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY ml/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies that might be needed
RUN pip install --no-cache-dir \
    peft \
    datasets \
    accelerate \
    pymongo

# Copy ML service source code
COPY ml/ ./

# Create necessary directories
RUN mkdir -p uploads summaries papers feedback_data

# Set environment variables for Flask
ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=8000
ENV FLASK_DEBUG=0

# Expose port 8000
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start the Flask app
CMD ["python", "app.py"]

