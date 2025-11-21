#!/usr/bin/env python3
"""
Test script to validate VS Code models integration without requiring full setup.

This script performs basic validation of the VS Code integration components
to ensure they can be imported and initialized correctly.
"""

import sys
import logging
import os

# Add the root directory to Python path for imports
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, root_dir)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all VS Code integration components can be imported."""
    try:
        from graphiti_core.llm_client.vscode_client import VSCodeClient
        from graphiti_core.embedder.vscode_embedder import VSCodeEmbedder, VSCodeEmbedderConfig
        from graphiti_core.llm_client.config import LLMConfig
        logger.info("✓ All imports successful")
        return True
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def test_client_initialization():
    """Test that VS Code clients can be initialized."""
    try:
        from graphiti_core.llm_client.vscode_client import VSCodeClient
        from graphiti_core.embedder.vscode_embedder import VSCodeEmbedder, VSCodeEmbedderConfig
        from graphiti_core.llm_client.config import LLMConfig
        
        # Test LLM client initialization
        llm_config = LLMConfig(model="test-model", small_model="test-small-model")
        llm_client = VSCodeClient(config=llm_config)
        logger.info("✓ VSCodeClient initialized successfully")
        
        # Test embedder initialization
        embedder_config = VSCodeEmbedderConfig(
            embedding_model="test-embedding",
            embedding_dim=1024,
            use_fallback=True
        )
        embedder = VSCodeEmbedder(config=embedder_config)
        logger.info("✓ VSCodeEmbedder initialized successfully")
        
        return True
    except Exception as e:
        logger.error(f"✗ Client initialization failed: {e}")
        return False

def test_configuration():
    """Test that configurations are set correctly."""
    try:
        from graphiti_core.embedder.vscode_embedder import VSCodeEmbedderConfig
        from graphiti_core.llm_client.config import LLMConfig
        
        # Test LLM config
        llm_config = LLMConfig(model="gpt-4o-mini", small_model="gpt-4o-mini")
        assert llm_config.model == "gpt-4o-mini"
        assert llm_config.small_model == "gpt-4o-mini"
        logger.info("✓ LLM configuration test passed")
        
        # Test embedder config
        embedder_config = VSCodeEmbedderConfig(
            embedding_model="embedding-001",
            embedding_dim=1024,
            use_fallback=True
        )
        assert embedder_config.embedding_model == "embedding-001"
        assert embedder_config.embedding_dim == 1024
        assert embedder_config.use_fallback == True
        logger.info("✓ Embedder configuration test passed")
        
        return True
    except Exception as e:
        logger.error(f"✗ Configuration test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    logger.info("Starting VS Code models integration validation...")
    
    tests = [
        ("Import Test", test_imports),
        ("Client Initialization Test", test_client_initialization),
        ("Configuration Test", test_configuration),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} ---")
        if test_func():
            passed += 1
        else:
            failed += 1
    
    logger.info(f"\n--- Test Results ---")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    
    if failed == 0:
        logger.info("🎉 All tests passed! VS Code models integration is ready.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())