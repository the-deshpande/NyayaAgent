FROM python:3.11-slim

# Create a non-root user for Hugging Face security
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Switch to root temporarily to install system packages
USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libcairo2-dev \
    && rm -rf /var/lib/apt/lists/*

# Switch back to the non-root user
USER user

# Copy requirements first for better layer caching
COPY --chown=user requirements.txt .

# Install dependencies (Excluding torch heavy stuff if running pure API mode, but included for complete sentence-transformers)
RUN pip install --no-cache-dir -r requirements.txt

# Download common spacy model if needed by langdetect/spacy
RUN python -m spacy download en_core_web_sm || echo "Skipping spacy download"

# Copy the rest of the project files
COPY --chown=user . .

# Expose port (Hugging Face uses 7860)
EXPOSE 7860

# Command to run the application (FastAPI now instead of Streamlit)
CMD ["python", "server.py"]
