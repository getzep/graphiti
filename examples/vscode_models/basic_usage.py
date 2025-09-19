#!/usr/bin/env python3
"""
Basic usage example for Graphiti with VS Code Models integration.

This example demonstrates how to use Graphiti with VS Code's built-in AI models
without requiring external API keys.

Prerequisites:
- VS Code with language model extensions (GitHub Copilot, Azure OpenAI, etc.)
- graphiti-core[vscodemodels] installed
- Running Neo4j instance

Usage:
    python basic_usage.py
"""

import asyncio
import os
from datetime import datetime
from graphiti_core import Graphiti
from graphiti_core.llm_client.vscode_client import VSCodeClient
from graphiti_core.embedder.vscode_embedder import VSCodeEmbedder, VSCodeEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig

async def main():
    """Basic example of using Graphiti with VS Code models."""
    
    # Configure VS Code clients
    llm_client = VSCodeClient(
        config=LLMConfig(
            model="gpt-4o-mini",  # VS Code model name
            small_model="gpt-4o-mini"
        )
    )
    
    embedder = VSCodeEmbedder(
        config=VSCodeEmbedderConfig(
            embedding_model="embedding-001",  # VS Code embedding model
            embedding_dim=1024,  # 1024-dimensional vectors
            use_fallback=True
        )
    )
    
    # Initialize Graphiti
    graphiti = Graphiti(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
        llm_client=llm_client,
        embedder=embedder
    )
    
    # Add some example episodes
    episodes = [
        "John is a software engineer who works at TechCorp. He specializes in Python development.",
        "Sarah is the CTO at TechCorp. She has been leading the engineering team for 5 years.",
        "TechCorp is developing a new AI-powered application using machine learning.",
        "John and Sarah are collaborating on the AI project, with John handling the backend implementation."
    ]
    
    print("Adding episodes to the knowledge graph...")
    current_time = datetime.now()
    for i, episode in enumerate(episodes):
        await graphiti.add_episode(
            name=f"Episode {i+1}",
            episode_body=episode,
            source_description="Example data",
            reference_time=current_time
        )
        print(f"✓ Added episode {i+1}")
    
    # Search for information
    print("\nSearching for information about TechCorp...")
    search_results = await graphiti.search(
        query="Tell me about TechCorp and its employees",
        center_node_uuid=None,
        num_results=5
    )
    
    print("Search Results:")
    for i, result in enumerate(search_results):
        print(f"{i+1}. {result.fact[:100]}...")
    
    print("\nExample completed successfully!")
    print("VS Code models integration is working properly.")

if __name__ == "__main__":
    asyncio.run(main())