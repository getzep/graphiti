#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import uvicorn
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title='Graphiti API', version='1.0.0')

class AddMemoryRequest(BaseModel):
    name: str
    episode_body: str
    group_id: str = 'ika-production'

class SearchRequest(BaseModel):
    query: str
    group_ids: List[str] = ['ika-production']

memories = []

@app.get('/')
async def root():
    return {
        'status': 'running',
        'version': '1.0.0',
        'memories_count': len(memories)
    }

@app.get('/health')
async def health():
    return {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat()
    }

@app.get('/status')
async def status():
    return {
        'api': 'running',
        'memories_stored': len(memories),
        'ollama': os.getenv('OLLAMA_HOST', 'not configured'),
        'falkordb': os.getenv('FALKORDB_HOST', 'not configured')
    }

@app.post('/add_memory')
async def add_memory(request: AddMemoryRequest):
    memory = {
        'id': len(memories) + 1,
        'name': request.name,
        'body': request.episode_body,
        'group_id': request.group_id,
        'created': datetime.utcnow().isoformat()
    }
    memories.append(memory)
    
    return {
        'success': True,
        'episode_id': memory['id'],
        'message': f"Memory '{request.name}' added successfully"
    }

@app.post('/search')
async def search(request: SearchRequest):
    results = []
    for memory in memories:
        if memory['group_id'] in request.group_ids:
            if request.query.lower() in memory['name'].lower() or request.query.lower() in memory['body'].lower():
                results.append(memory)
    
    return {
        'success': True,
        'query': request.query,
        'count': len(results),
        'results': results
    }

if __name__ == '__main__':
    logger.info('Starting Graphiti API Server')
    uvicorn.run(app, host='0.0.0.0', port=8000)
