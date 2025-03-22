#!/bin/bash

# Script to run the Graphiti MCP server
#
# Usage:
#   ./run_server.sh [options]
#
# Options:
#   --llm-client=TYPE   Specify the LLM client type (openai, openai_generic, anthropic)
#   --model=NAME        Specify the model name to use
#
# Examples:
#   ./run_server.sh                                # Use default (Anthropic if available, otherwise OpenAI)
#   ./run_server.sh --llm-client=anthropic         # Use Anthropic explicitly
#   ./run_server.sh --llm-client=openai            # Use OpenAI
#   ./run_server.sh --llm-client=anthropic --model=claude-3-5-sonnet-20240620  # Use specific model

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
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Warning: ANTHROPIC_API_KEY environment variable is not set."
    
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "Warning: OPENAI_API_KEY environment variable is also not set."
        echo "An LLM API key is required for full functionality."
    else
        echo "Using OpenAI as fallback LLM provider."
    fi
else
    echo "Using Anthropic as the LLM provider."
fi

# Check Neo4j credentials
if [ -z "$NEO4J_URI" ]; then
    echo "Warning: NEO4J_URI environment variable is not set."
    echo "Using default URI 'bolt://localhost:7687'."
    export NEO4J_URI="bolt://localhost:7687"
fi

if [ -z "$NEO4J_USER" ]; then
    echo "Warning: NEO4J_USER environment variable is not set."
    echo "Using default user 'neo4j'."
    export NEO4J_USER="neo4j"
fi

if [ -z "$NEO4J_PASSWORD" ]; then
    echo "Warning: NEO4J_PASSWORD environment variable is not set."
    echo "Using default password 'password'."
    export NEO4J_PASSWORD="password"
fi

# Check if Neo4j is running
echo "Checking Neo4j connection..."
if ! nc -z -w 5 $(echo $NEO4J_URI | sed -E 's|bolt://([^:]+).*|\1|') $(echo $NEO4J_URI | sed -E 's|bolt://[^:]+:([0-9]+).*|\1|' || echo "7687"); then
    echo "Warning: Could not connect to Neo4j at $NEO4J_URI"
    echo "Make sure Neo4j is running and accessible."
    echo "The server will start, but operations requiring Neo4j will fail."
fi

# Parse command line arguments
LLM_CLIENT=""
MODEL=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --llm-client=*)
      LLM_CLIENT="${1#*=}"
      shift
      ;;
    --model=*)
      MODEL="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Run the server
echo "Starting Graphiti MCP server..."

# Build the command with optional arguments
COMMAND="python graphiti_mcp_server.py"

if [ ! -z "$LLM_CLIENT" ]; then
  COMMAND="$COMMAND --llm-client $LLM_CLIENT"
fi

if [ ! -z "$MODEL" ]; then
  COMMAND="$COMMAND --model $MODEL"
fi

# Execute the command
echo "Running: $COMMAND"
eval $COMMAND
