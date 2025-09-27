# ===== 1) Base image =====
FROM python:3.11-slim AS app

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /app

# Streamlit config
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501
EXPOSE 8501

# Default: run the UI
CMD ["streamlit", "run", "interface.py", "--server.address=0.0.0.0", "--server.port=8501"]
