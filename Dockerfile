# ============================================================
# Base image
# ============================================================

FROM python:3.11-slim

# ============================================================
# Environment configuration
# ============================================================

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ============================================================
# Working directory inside container
# ============================================================

WORKDIR /app

# ============================================================
# Copy dependency file first (Docker cache optimization)
# ============================================================

COPY requirements.txt .

# ============================================================
# Install dependencies
# ============================================================

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================
# Copy API source code and artifacts
# ============================================================

COPY mlops_pipeline/src/model_deploy.py .
COPY mlops_pipeline/src/artifacts/ ./artifacts/

# ============================================================
# Expose API port
# ============================================================

EXPOSE 8000

# ============================================================
# Start FastAPI using uvicorn
# ============================================================

CMD ["uvicorn", "model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]