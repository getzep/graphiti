# Build stage
FROM python:3.12-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy dependency-related files first (cache-friendly)
COPY ./pyproject.toml ./poetry.lock* ./README.md /app/
COPY ./server/pyproject.toml ./server/poetry.lock* /app/server/
# <-- musi być przed buildem!
COPY ./graphiti_core /app/graphiti_core  

# Configure Poetry
RUN poetry config virtualenvs.create false

# Build local package & install it
RUN poetry build -f wheel && pip install dist/*.whl

# Install server dependencies
WORKDIR /app/server
RUN poetry install --no-interaction --no-ansi --only main --no-root

# Download spaCy model and GoEmotions model
RUN python -m spacy download en_core_web_sm && \
    python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
               AutoTokenizer.from_pretrained('monologg/bert-base-cased-goemotions-original'); \
               AutoModelForSequenceClassification.from_pretrained('monologg/bert-base-cased-goemotions-original')"

# Runtime stage
FROM python:3.12-slim

# Copy only necessary files from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# App directory and source code
WORKDIR /app
COPY ./server /app

# Environment
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Run the app
CMD uvicorn graph_service.main:app --host 0.0.0.0 --port $PORT
