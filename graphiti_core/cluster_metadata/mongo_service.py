"""
MongoDB service adapter for Graphiti cluster metadata.

This module provides a bridge to the MongoDB service, adapting the provided
MongoDB service class for use within the Graphiti cluster metadata system.
"""

import logging
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo.errors import ConnectionFailure, ConfigurationError
import os

logger = logging.getLogger(__name__)

# Global variables for singleton pattern
_mongo_client: Optional[AsyncIOMotorClient] = None
_mongo_db: Optional[AsyncIOMotorDatabase] = None


async def initialize_mongo_client() -> AsyncIOMotorClient:
    """
    Initialize MongoDB client for cluster metadata operations.
    
    This adapts the provided MongoDB service pattern to work within
    the Graphiti project structure.
    
    Returns:
        AsyncIOMotorClient: MongoDB client instance
    """
    global _mongo_client
    
    if _mongo_client is None:
        try:
            # Get MongoDB URI from environment
            mongo_uri = os.getenv('MONGO_URI') or os.getenv('MONGODB_URI')
            if not mongo_uri:
                logger.error("MONGO_URI is not configured in environment variables.")
                raise ConfigurationError("MONGO_URI must be set in environment variables or .env file.")
            
            logger.info("Connecting to MongoDB for cluster metadata...")
            
            # Create AsyncIOMotorClient with recommended settings
            _mongo_client = AsyncIOMotorClient(
                mongo_uri,
                serverSelectionTimeoutMS=60000,  # 60 seconds
                connectTimeoutMS=120000,          # 120 seconds  
                socketTimeoutMS=600000,           # 10 minutes
                maxPoolSize=100,
                minPoolSize=0
            )
            
            # Verify connection
            if _mongo_client is not None:
                await _mongo_client.admin.command('ping')
            logger.info("Successfully connected to MongoDB for cluster metadata.")
            
        except ConfigurationError:
            _mongo_client = None
            raise
        except ConnectionFailure as cf:
            logger.error(f"Failed to connect to MongoDB: {cf}")
            _mongo_client = None
            raise
        except Exception as e:
            logger.error(f"Unexpected error during MongoDB client creation: {e}", exc_info=True)
            _mongo_client = None
            raise ConnectionFailure(f"Unexpected error during MongoDB client creation: {e}")
    
    return _mongo_client


async def get_mongo_connection() -> AsyncIOMotorDatabase:
    """
    Get MongoDB database connection for cluster metadata.
    
    Returns:
        AsyncIOMotorDatabase: Database instance for cluster operations
    """
    global _mongo_db
    
    if _mongo_db is None:
        try:
            # Initialize client
            client = await initialize_mongo_client()
            
            # Get database name from environment
            db_name = os.getenv('DB_NAME') or os.getenv('MONGODB_DB_NAME') or 'defence_board'
            
            _mongo_db = client[db_name]
            logger.info(f"Successfully accessed database for cluster metadata: '{db_name}'")
            
        except Exception as e:
            logger.error(f"Error accessing MongoDB database: {e}", exc_info=True)
            _mongo_db = None
            if not isinstance(e, (ConnectionFailure, ConfigurationError)):
                raise ConfigurationError(f"Unexpected error during database access: {e}")
            raise
    
    return _mongo_db

async def get_collection(collection_name: str) -> AsyncIOMotorCollection:
        """Get cluster_metadata collection from MongoDB"""
        try:
            db: AsyncIOMotorDatabase = await get_mongo_connection()
            return db[collection_name]
        except ImportError:
            logger.error("MongoDB service not available. Ensure mongo_service is properly configured.")
            raise ConfigurationError("MongoDB service unavailable")


async def close_mongo_connection():
    """Close MongoDB connection"""
    global _mongo_client, _mongo_db
    
    if _mongo_client:
        try:
            _mongo_client.close()
            logger.info("MongoDB cluster metadata connection closed.")
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {e}", exc_info=True)
        finally:
            _mongo_client = None
            _mongo_db = None
