"""
Standalone script for field database setup.
Creates necessary constraints and indexes for FieldNode system.
Can be run independently or called from other modules.
"""

import asyncio
import logging
import os
from dotenv import load_dotenv

from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.models.nodes.field_db_queries import (
    CREATE_FIELD_CLUSTER_CONSTRAINT,
    CREATE_FIELD_INDEXES,
)

logger = logging.getLogger(__name__)

async def setup_field_database(driver):
    """Set up database constraints and indexes for FieldNode system"""
    
    logger.info("Creating field uniqueness constraint...")
    await driver.execute_query(CREATE_FIELD_CLUSTER_CONSTRAINT)
    
    logger.info("Creating field indexes for efficient queries...")
    for index_query in CREATE_FIELD_INDEXES:
        await driver.execute_query(index_query)
    
    logger.info("✅ Field database setup complete!")
    logger.info("✅ Ready for fast label-based queries!")

async def main():
    """Main function for standalone execution"""
    load_dotenv()
    
    # Get connection details from environment
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    
    print(f"Connecting to Neo4j at {uri}...")
    
    driver = Neo4jDriver(uri, user, password)
    
    try:
        await setup_field_database(driver)
        print("🎯 Field database setup completed successfully!")
    except Exception as e:
        logger.error(f"Failed to set up field database: {e}")
        raise
    finally:
        await driver.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main()) 