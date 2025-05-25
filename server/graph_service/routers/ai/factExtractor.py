"""
Main fact extractor module that coordinates the extraction and storage of facts, emotions, and entities.
"""
import json
import logging
from typing import Dict, Any, Optional

from .config import OPENAI_API_KEY
from .function_specs import functionsSpec
from .extraction import extract_facts_emotions_entities
from .neo4j_operations import (
    get_existing_data,
    store_extracted_data,
    get_relationships_data,
    get_top_items
)

# Configure logger
logger = logging.getLogger(__name__)

# Alias dla kompatybilności wstecznej - stara nazwa funkcji
extractFactsAndStore = extractAllAndStore

async def extractAllAndStore(graphiti, message, group_id, chat_history, shirt_slug):
    """
    Extract facts, emotions, and entities from `message.content` using OpenAI function calls,
    then store them in Neo4j as connected nodes under Episodic(uuid).
    
    Args:
        graphiti: Graphiti database connection
        message: Message object containing content to analyze
        group_id: Group ID for the message
        chat_history: Chat history for context
        shirt_slug: Shirt slug to associate with the data
        
    Returns:
        Dictionary containing top facts, emotions, and entities
    """
    # Jeśli nie ma message lub message.content, nic nie rób
    if not message or not hasattr(message, 'content') or not message.content or not message.content.strip():
        return None
        
    try:
        # 1. Pobierz istniejące fakty, emocje i encje dla danego group_id
        async with graphiti.driver.session() as session:
            existing_data = await get_existing_data(session, group_id)
            
        # 2. Wykonaj ekstrakcję faktów, emocji i encji
        logger.info(f"[Graphiti] Extracting facts, emotions, and entities. History: {chat_history}, Shirt Slug: {shirt_slug}")
        extraction_results = await extract_facts_emotions_entities(
            message_content=message.content,
            existing_emotions=existing_data["emotions"],
            existing_entities=existing_data["entities"],
            chat_history=chat_history
        )
        
        facts = extraction_results["facts"]
        emotions = extraction_results["emotions"]
        entities = extraction_results["entities"]
        
        # Log token usage
        usage = extraction_results["usage"]
        logger.info(
            f"[Graphiti] Token usage - Input: {usage['input_tokens']}, "
            f"Output: {usage['output_tokens']}, Total: {usage['total_tokens']}"
        )
        
        # 3. Store data in Neo4j
        async with graphiti.driver.session() as session:
            # Store main data
            await store_extracted_data(
                session, 
                message.uuid, 
                group_id, 
                facts, 
                emotions, 
                entities, 
                shirt_slug
            )
            
            # Fetch relationship data (not used but could be useful for future analysis)
            relationships = await get_relationships_data(session, group_id)
            
        # 4. Get top facts, emotions, and entities
        async with graphiti.driver.session() as session:
            top_items = await get_top_items(session, group_id)
            
        return top_items
    except Exception as e:
        logger.error(f"[Graphiti] Error in extractAllAndStore: {str(e)}")
        return {
            "top_facts": [],
            "top_emotions": [],
            "top_entities": [],
            "error": str(e)
        }

# Alias dla kompatybilności wstecznej
extractFactsAndStore = extractAllAndStore
