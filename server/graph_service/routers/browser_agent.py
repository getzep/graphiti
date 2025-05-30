"""
Copyright 2025, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from server.graph_service.config import get_graphiti

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/browser-agent",
    tags=["browser_agent"],
)


class WebPageContent(BaseModel):
    """Model for web page content extracted by the browser agent."""
    content: Dict[str, Any] = Field(..., description="The extracted content from the web page")
    categories: List[str] = Field(default_factory=list, description="Categories for the content")
    source: Dict[str, str] = Field(..., description="Source information about the web page")


class CategoryResponse(BaseModel):
    """Model for AI-generated categories."""
    categories: List[str] = Field(..., description="AI-generated categories for the content")


@router.post("/save", response_model=Dict[str, str])
async def ingest_web_page(
    data: WebPageContent,
    graphiti: Graphiti = Depends(get_graphiti),
) -> Dict[str, str]:
    """
    Ingest web page content into the knowledge graph.
    
    This endpoint accepts content extracted from a web page by the browser agent,
    processes it, and adds it to the knowledge graph as an episode with appropriate
    metadata and categorization.
    """
    try:
        # Prepare the episode content
        episode_content = {
            "raw_content": data.content,
            "categories": data.categories,
            "source_metadata": data.source,
        }
        
        # Convert to JSON string for storage
        episode_body = json.dumps(episode_content)
        
        # Generate a name for the episode based on the page title
        page_title = data.source.get("title", "Web Page")
        episode_name = f"Web: {page_title[:50]}"
        
        # Add the episode to the graph
        episode_result = await graphiti.add_episode(
            name=episode_name,
            episode_body=episode_body,
            source=EpisodeType.json,
            source_description="Web page content extracted by browser agent",
            reference_time=datetime.now(timezone.utc),
        )
        
        return {"status": "success", "message": "Web page content added to knowledge graph", "episode_id": str(episode_result.episode.uuid)}
    
    except Exception as e:
        logger.error(f"Error ingesting web page content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to ingest web page content: {str(e)}")


@router.post("/categorize", response_model=CategoryResponse)
async def categorize_content(
    data: Dict[str, Any],
    graphiti: Graphiti = Depends(get_graphiti),
) -> CategoryResponse:
    """
    Generate AI categories for the provided web page content.
    
    This endpoint uses the LLM to analyze the content and suggest appropriate
    categories or ontology classifications.
    """
    try:
        # Extract the content from the request
        content = data.get("content", {})
        
        # Prepare a prompt for the LLM to categorize the content
        prompt = f"""
        Analyze the following web page content and suggest 3-7 appropriate categories or ontology classifications.
        Focus on the main topics, entities, and themes present in the content.
        
        Content: {json.dumps(content, indent=2)}
        
        Return only a list of categories, one per line.
        """
        
        # Use the LLM to generate categories
        llm_response = await graphiti.llm_client.acompletion(prompt)
        
        # Parse the response to extract categories
        categories = [
            category.strip() 
            for category in llm_response.split("\n") 
            if category.strip() and not category.strip().startswith("-")
        ]
        
        # Ensure we have at least some categories
        if not categories:
            categories = ["Web Content", "Uncategorized"]
        
        return CategoryResponse(categories=categories)
    
    except Exception as e:
        logger.error(f"Error categorizing content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to categorize content: {str(e)}")


@router.get("/healthcheck")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint for the browser agent API.
    
    This endpoint is used by the browser extension to verify connectivity
    to the Graphiti server.
    """
    return {"status": "ok", "service": "browser_agent"}