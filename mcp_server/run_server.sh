#!/bin/bash

# Script to run the Graphiti MCP server

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.10 or higher."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "pip3 is not installed. Please install pip for Python 3."
    exit 1
fi

# Check if virtual environment exists, if not create one
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if environment variables are set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY environment variable is not set."
    echo "Some functionality may be limited."
fi

if [ -z "$NEO4J_PASSWORD" ]; then
    echo "Warning: NEO4J_PASSWORD environment variable is not set."
    echo "Using default password 'password'."
fi

# Run the server
echo "Starting Graphiti MCP server..."
python graphiti_mcp_server.py
