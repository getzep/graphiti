# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# gcc is needed for compiling some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
# Poetry is used for dependency management
RUN pip install --no-cache-dir poetry

# Copy only requirements to cache them in docker layer
# This is done before copying the rest of the code to leverage Docker cache
COPY pyproject.toml poetry.lock* /app/

# Configure Poetry and install project dependencies
# virtualenvs.create false: Install packages directly in system Python
# --no-interaction: Do not ask any interactive question
# --no-ansi: Do not use ANSI codes in output
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Copy the rest of the application's code
COPY . /app

# Command to run the application
CMD ["fastapi", "run", "graph_service/main.py", "--port", "8000"]