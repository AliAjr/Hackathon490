# Use lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for pandas, scikit-learn, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY requirements.txt .
COPY interface.py .
COPY Hotspot_crash_prediction_universal.py .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit default port
EXPOSE 8501

# Command: first train model, then launch Streamlit
CMD ["bash", "-c", "python Hotspot_crash_prediction_universal.py && streamlit run interface.py --server.port=8501 --server.address=0.0.0.0"]
