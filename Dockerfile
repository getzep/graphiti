# Build stage
FROM python:3.12-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy only the files needed for installation
COPY ./pyproject.toml ./poetry.lock* ./README.md /app/
COPY ./graphiti_core /app/graphiti_core
COPY ./server/pyproject.toml ./server/poetry.lock* /app/server/

RUN poetry config virtualenvs.create false 

# Install the local package
RUN poetry build && pip install dist/*.whl

# Install server dependencies
WORKDIR /app/server
RUN poetry install --no-interaction --no-ansi --only main --no-root

FROM python:3.12-slim

# Copy only the necessary files from the builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create the app directory and copy server files
WORKDIR /app
COPY ./server /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
# Command to run the application

CMD uvicorn graph_service.main:app --host 0.0.0.0 --port $PORT