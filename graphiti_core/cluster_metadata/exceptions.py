"""
Custom exceptions for cluster metadata operations.

These exceptions provide specific error handling for MongoDB cluster metadata
operations, allowing proper error propagation and handling in the API layer.
This module is specifically designed for MongoDB operations and does not
interact with the Graphiti knowledge graph functionality.
"""

from typing import Optional, List
from pymongo.errors import PyMongoError


class ClusterError(PyMongoError):
    """Base exception for MongoDB cluster-related errors"""
    pass


class ClusterNotFoundError(ClusterError):
    """Raised when a cluster is not found in MongoDB"""
    
    def __init__(self, cluster_id: str):
        self.cluster_id = cluster_id
        self.message = f"Cluster '{cluster_id}' not found"
        super().__init__(self.message)


class DuplicateClusterError(ClusterError):
    """Raised when attempting to create a cluster that already exists"""
    
    def __init__(self, cluster_id: str):
        self.cluster_id = cluster_id
        self.message = f"Cluster '{cluster_id}' already exists"
        super().__init__(self.message)


class InvalidOrganizationError(ClusterError):
    """Raised when organization is not found in MongoDB integration configurations"""
    
    def __init__(self, organization: str, approved_orgs: Optional[List[str]] = None):
        self.organization = organization
        self.approved_orgs = approved_orgs or []
        
        if not organization or organization.strip() == "":
            self.message = "Organization name cannot be empty"
        else:
            self.message = f"Organization '{organization}' does not exist or has no Splunk integration configured"
            if self.approved_orgs:
                self.message += f". Available organizations: {', '.join(self.approved_orgs)}"
        
        super().__init__(self.message)


class ClusterValidationError(ClusterError):
    """Raised when MongoDB cluster validation fails with multiple validation errors"""
    
    def __init__(self, message: str, validation_errors: Optional[List[str]] = None):
        self.validation_errors = validation_errors or []
        self.message = message
        
        if self.validation_errors:
            self.message += f": {'; '.join(self.validation_errors)}"
        
        super().__init__(self.message)


class ClusterConnectionError(ClusterError):
    """Raised when MongoDB connection fails for cluster operations"""
    
    def __init__(self, message: str = "Failed to connect to MongoDB cluster metadata service"):
        self.message = message
        super().__init__(self.message)


class ClusterUpdateError(ClusterError):
    """Raised when an error occurs during MongoDB cluster update operations"""
    
    def __init__(self, cluster_id: str, message: str):
        self.cluster_id = cluster_id
        self.message = f"Error updating cluster '{cluster_id}' in MongoDB: {message}"
        super().__init__(self.message)


########################################################################
# Field-related exceptions for cluster_fields operations
########################################################################


class FieldError(PyMongoError):
    """Base exception for MongoDB field-related errors"""
    pass

class FieldUpdateError(FieldError):
    """Raised when an error occurs during field update operations"""
    
    def __init__(self, field_uuid: str, message: str):
        self.field_uuid = field_uuid
        self.message = f"Error updating field '{field_uuid}': {message}"
        super().__init__(self.message)



class FieldAlreadyExistsError(Exception):
    """Raised when attempting to add a field that already exists in the cluster"""
    def __init__(self, field_uuid: str, cluster_uuid: str):
        self.field_uuid = field_uuid
        self.cluster_uuid = cluster_uuid
        self.message = f"Field '{field_uuid}' already exists in cluster '{cluster_uuid}'"
        super().__init__(self.message)


class FieldNotFoundError(Exception):
    """Raised when a field is not found in the cluster fields collection"""
    def __init__(self, field_uuid: str, cluster_uuid: Optional[str] = None):
        self.field_uuid = field_uuid
        self.cluster_uuid = cluster_uuid
        if cluster_uuid:
            self.message = f"Field '{field_uuid}' not found in cluster '{cluster_uuid}'"
        else:
            self.message = f"Field '{field_uuid}' not found"
        super().__init__(self.message)


class FieldValidationError(Exception):
    """Raised when field validation fails based on Neo4j constraints"""
    def __init__(self, message: str, validation_errors: Optional[List[str]] = None):
        self.validation_errors = validation_errors or []
        self.message = message
        if self.validation_errors:
            self.message += f": {'; '.join(self.validation_errors)}"
        super().__init__(self.message)