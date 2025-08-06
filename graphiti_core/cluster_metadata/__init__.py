"""
Cluster Metadata Module for Hybrid MongoDB + Neo4j Architecture

This module provides cluster metadata management functionality for the hybrid
architecture where MongoDB handles fast cluster lookups and validation while
Neo4j manages field relationships and graph data.

Main Components:
- ClusterMetadataService: Primary service for cluster CRUD operations
- Models: Pydantic models for MongoDB documents and API requests
- Validators: Business logic validation for clusters and sub-clusters
- Exceptions: Custom error types for cluster operations

Usage:
    from graphiti_core.cluster_metadata import ClusterMetadataService, ClusterCreateRequest
    
    service = ClusterMetadataService()
    cluster = await service.create_cluster(request)
"""

import logging

# Main service class
from .cluster_service import ClusterMetadataService

# Pydantic models
from .models import (
    ClusterCreateRequest,
    ClusterUpdateRequest,
    ClusterStats,
    ClusterSearchRequest,
    ClusterCriteriaRequest,
    FieldCreateRequest,
    ClusterValidationResult
)

# Validators
from .validator import EnhancedClusterValidator

# Exceptions
from .exceptions import (
    ClusterError,
    ClusterNotFoundError,
    DuplicateClusterError,
    ClusterValidationError,
    InvalidOrganizationError,
    ClusterConnectionError,
    ClusterUpdateError
)

# Set up logging for the module
logger = logging.getLogger(__name__)

# Export public API
__all__ = [
    # Service
    'ClusterMetadataService',
    
    # Models
    'ClusterCreateRequest',
    'ClusterUpdateRequest',
    'ClusterStats',
    'ClusterSearchRequest',
    'ClusterCriteriaRequest',
    'FieldCreateRequest',
    'ClusterValidationResult',
    
    # Validators
    'EnhancedClusterValidator',
    
    # Exceptions
    'ClusterError',
    'ClusterNotFoundError',
    'DuplicateClusterError', 
    'ClusterValidationError',
    'InvalidOrganizationError',
    'ClusterConnectionError',
    'ClusterUpdateError'
]

# Module metadata
__version__ = '1.0.0'
__author__ = 'Graphiti Hybrid Architecture Team'
__description__ = 'MongoDB cluster metadata management for hybrid architecture'

# Initialize module
logger.info(f"Initialized cluster_metadata module v{__version__}")

# Configuration constants
DEFAULT_COLLECTION_NAME = "cluster_metadata"
MAX_CLUSTER_NAME_LENGTH = 100
MAX_DESCRIPTION_LENGTH = 1000
MAX_SUB_CLUSTERS_PER_CLUSTER = 50
