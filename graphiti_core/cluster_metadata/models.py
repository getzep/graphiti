"""
Pydantic models for MongoDB cluster metadata documents.

These models define the structure for cluster metadata stored in MongoDB,
following the hybrid architecture where MongoDB handles fast cluster lookups
and Neo4j manages field relationships.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from graphiti_core.utils.datetime_utils import utc_now

class ClusterCreateRequest(BaseModel):
    """Request model for creating a new cluster"""
    macro_name: str = Field(description="Macro identifier (e.g., 'linux_audit')")
    organization: str = Field(description="Organization name (e.g., 'batelco')")
    cluster_uuid: str = Field(description="Unique cluster UUID From the graphiti UUID service")
    description: str = Field(description="Cluster description")
    cluster_id: str = Field(description="Cluster identifier (macro_organization, e.g., 'linux_audit_batelco' or 'complex_macro_name_batelco')")
    status: Optional[str] = Field(default="active", description="Cluster status: active, inactive, deprecated")
    total_fields: Optional[int] = Field(default=0, description="Total number of unique fields in cluster")
    created_by: Optional[str] = Field(default="system", description="User creating the cluster")
    

    @field_validator('macro_name')
    @classmethod
    def validate_macro_name(cls, v):
        """Validate macro name format (basic syntax validation only)"""
        if not v or not isinstance(v, str):
            raise ValueError("macro_name must be a non-empty string")
        
        # Remove underscores for alphanumeric check
        cleaned = v.replace('_', '')
        if not cleaned.isalnum():
            raise ValueError("macro_name must be alphanumeric with optional underscores")
        
        # Basic length check
        if len(v) > 50:
            raise ValueError("macro_name too long (max 50 characters)")
            
        return v.lower()
    
    @field_validator('organization')
    @classmethod
    def validate_organization(cls, v):
        """Validate organization format (basic syntax validation only)"""
        if not v or not isinstance(v, str):
            raise ValueError("organization must be a non-empty string")
            
        # Remove underscores for alphanumeric check
        cleaned = v.replace('_', '')
        if not cleaned.isalnum():
            raise ValueError("organization must be alphanumeric with optional underscores")
            
        # Basic length check
        if len(v) > 50:
            raise ValueError("organization too long (max 50 characters)")
            
        return v  # Preserve original case (don't convert to lowercase)
    
    @field_validator('status')
    @classmethod
    def validate_status(cls, v):
        """Validate status if provided"""
        if v is not None:
            allowed_statuses = ['active', 'inactive', 'deprecated']
            if v not in allowed_statuses:
                raise ValueError(f"status must be one of {allowed_statuses}")
        return v



class ClusterUpdateRequest(BaseModel):
    """Request model for updating cluster metadata"""
    total_fields: Optional[int] = Field(default=0, description="Total number of unique fields in cluster")
    cluster_uuid: str = Field(description="Unique cluster UUID From the graphiti UUID service")
    description: Optional[str] = Field(None, description="Updated description")
    status: Optional[str] = Field(None, description="Updated status")
    last_updated: Optional[datetime] = Field(default_factory=utc_now, description="Last updated timestamp")
    
    @field_validator('status')
    @classmethod
    def validate_status(cls, v):
        """Validate status if provided"""
        if v is not None:
            allowed_statuses = ['active', 'inactive', 'deprecated']
            if v not in allowed_statuses:
                raise ValueError(f"status must be one of {allowed_statuses}")
        return v
    
    
class ClusterStats(BaseModel):
    """Statistics for a cluster"""
    cluster_uuid: str = Field(description="Unique cluster UUID")
    cluster_id: str = Field(description="Generated cluster ID (macro_organization, flexible format)")
    total_fields: int = Field(description="Total number of unique fields in cluster")
    status: str = Field(description="Current cluster status")
    created_at: datetime = Field(description="Cluster creation timestamp")
    last_updated: datetime = Field(description="Last update timestamp")
    organization: str = Field(description="Organization name")
    created_by: str = Field(default="system", description="User who created the cluster")
    description: str  = Field(description="Cluster description")
    macro_name: str = Field(description="Macro identifier")


class ClusterSearchRequest(BaseModel):
    """Request model for searching clusters with flexible criteria"""
    document_id: Optional[str] = Field(None, description="MongoDB document ID for direct cluster lookup")
    organization: Optional[str] = Field(None, description="Filter by organization")
    macro_name: Optional[str] = Field(None, description="Filter by macro name")
    cluster_id: Optional[str] = Field(None, description="Filter by cluster identifier (flexible macro_organization format)")
    cluster_uuid: Optional[str] = Field(None, description="Filter by specific cluster UUID")
    
    @model_validator(mode='after')
    def validate_search_criteria(self):
        """Ensure only one valid search combination is provided"""
        # Check which fields are provided (non-None and non-empty)
        has_id = self.document_id is not None and str(self.document_id).strip()
        has_organization = self.organization is not None and str(self.organization).strip()
        has_macro_name = self.macro_name is not None and str(self.macro_name).strip()
        has_cluster_id = self.cluster_id is not None and str(self.cluster_id).strip()
        has_cluster_uuid = self.cluster_uuid is not None and str(self.cluster_uuid).strip()
        
        # Count how many search approaches are being used
        search_approaches = []
        
        if has_id:
            search_approaches.append("document_id")
        
        if has_organization and has_macro_name:
            search_approaches.append("organization + macro_name")
        elif has_organization or has_macro_name:
            # One without the other is invalid
            if has_organization:
                raise ValueError("organization must be provided together with macro_name")
            else:
                raise ValueError("macro_name must be provided together with organization")
        
        if has_cluster_id:
            search_approaches.append("cluster_id")
        
        if has_cluster_uuid:
            search_approaches.append("cluster_uuid")
        
        # Ensure exactly one search approach is used
        if len(search_approaches) == 0:
            raise ValueError(
                "At least one search criteria must be provided: "
                "either 'document_id', 'organization + macro_name', 'cluster_id', or 'cluster_uuid'"
            )
        elif len(search_approaches) > 1:
            raise ValueError(
                f"Only one search approach should be used, but got: {', '.join(search_approaches)}. "
                "Please use either 'document_id', 'organization + macro_name', 'cluster_id', or 'cluster_uuid'."
            )
        
        return self
    
    @field_validator('document_id')
    @classmethod
    def validate_mongodb_id(cls, v):
        """Validate MongoDB ObjectId format if provided"""
        if v is not None and str(v).strip():
            # Basic validation for MongoDB ObjectId (24 character hex string)
            cleaned = str(v).strip()
            if len(cleaned) == 24 and all(c in '0123456789abcdefABCDEF' for c in cleaned):
                return cleaned
            else:
                raise ValueError("document_id must be a valid 24-character MongoDB ObjectId")
        return v
    
    @field_validator('cluster_id')
    @classmethod
    def validate_cluster_id_format(cls, v):
        """Validate cluster identifier format if provided"""
        if v is not None and str(v).strip():
            cleaned = str(v).strip()
            
            # Basic validation - must be non-empty and reasonable length
            if not cleaned:
                raise ValueError("cluster_id cannot be empty")
            
            if len(cleaned) > 200:
                raise ValueError("cluster_id too long (max 200 characters)")
            
            # Must contain at least one underscore (to separate macro parts from organization)
            if '_' not in cleaned:
                raise ValueError("cluster_id must contain at least one underscore")
            
            return cleaned.lower()
        return v
    
    @field_validator('cluster_uuid')
    @classmethod
    def validate_cluster_uuid_format(cls, v):
        """Validate UUID format if provided"""
        if v is not None and str(v).strip():
            import re
            cleaned = str(v).strip()
            # Basic UUID format validation
            uuid_pattern = r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
            if not re.match(uuid_pattern, cleaned):
                raise ValueError("cluster_uuid must be a valid UUID format")
            return cleaned
        return v
    
    @field_validator('organization')
    @classmethod
    def validate_organization_format(cls, v):
        """Validate organization format if provided"""
        if v is not None and str(v).strip():
            cleaned = str(v).strip()
            # Remove underscores for alphanumeric check
            alphanumeric_part = cleaned.replace('_', '')
            if not alphanumeric_part.isalnum():
                raise ValueError("organization must be alphanumeric with optional underscores")
            
            # Basic length check
            if len(cleaned) > 50:
                raise ValueError("organization too long (max 50 characters)")
                
            return cleaned  # Preserve original case
        return v
    
    @field_validator('macro_name')
    @classmethod
    def validate_macro_name_format(cls, v):
        """Validate macro name format if provided"""
        if v is not None and str(v).strip():
            cleaned = str(v).strip()
            # Remove underscores for alphanumeric check
            alphanumeric_part = cleaned.replace('_', '')
            if not alphanumeric_part.isalnum():
                raise ValueError("macro_name must be alphanumeric with optional underscores")
            
            # Basic length check
            if len(cleaned) > 50:
                raise ValueError("macro_name too long (max 50 characters)")
                
            return cleaned.lower()
        return v

class FieldCreateRequest(BaseModel):
    """Request model for creating a field (used by hybrid field manager)"""
    field_name: str = Field(description="Security field name")
    field_uuid: str = Field(description="Unique field UUID from the graphiti UUID service")
    description: str = Field(description="Field description")
    examples: List[str] = Field(default_factory=lambda: [], description="Example values")
    data_type: str = Field(description="Field data type")
    cluster_uuid: str = Field(description="Target cluster UUID")
    created_by: Optional[str] = Field(default="system", description="User creating the field")
    created_at: datetime = Field(default_factory=utc_now, description="Field creation timestamp")
    last_updated: datetime = Field(default_factory=utc_now, description="Last updated timestamp")


class ClusterValidationResult(BaseModel):
    """Result of cluster validation"""
    is_valid: bool = Field(description="Overall validation result")
    cluster_exists: bool = Field(description="Whether cluster exists in system")
    cluster_uuid: Optional[str] = Field(None, description="Cluster UUID if exists")
    errors: List[str] = Field(default_factory=lambda: [], description="Validation error messages")
    warnings: List[str] = Field(default_factory=lambda: [], description="Validation warnings")


class ClusterCriteriaRequest(BaseModel):
    """Request model for searching clusters by multiple criteria"""
    organization: Optional[str] = Field(None, description="Filter by organization")
    macro_name: Optional[str] = Field(None, description="Filter by macro name")
    status: Optional[str] = Field(None, description="Filter by status (active, inactive, deprecated)")
    created_by: Optional[str] = Field(None, description="Filter by creator")
    
    @field_validator('organization')
    @classmethod
    def validate_organization_format(cls, v):
        """Validate organization format if provided"""
        if v is not None and str(v).strip():
            cleaned = str(v).strip()
            # Remove underscores for alphanumeric check
            alphanumeric_part = cleaned.replace('_', '')
            if not alphanumeric_part.isalnum():
                raise ValueError("organization must be alphanumeric with optional underscores")
            
            # Basic length check
            if len(cleaned) > 50:
                raise ValueError("organization too long (max 50 characters)")
                
            return cleaned  # Preserve original case
        return v
    
    @field_validator('macro_name')
    @classmethod
    def validate_macro_name_format(cls, v):
        """Validate macro name format if provided"""
        if v is not None and str(v).strip():
            cleaned = str(v).strip()
            # Remove underscores for alphanumeric check
            alphanumeric_part = cleaned.replace('_', '')
            if not alphanumeric_part.isalnum():
                raise ValueError("macro_name must be alphanumeric with optional underscores")
            
            # Basic length check
            if len(cleaned) > 50:
                raise ValueError("macro_name too long (max 50 characters)")
                
            return cleaned.lower()
        return v
    
    @field_validator('status')
    @classmethod
    def validate_status_format(cls, v):
        """Validate status if provided"""
        if v is not None and str(v).strip():
            allowed_statuses = ['active', 'inactive', 'deprecated']
            cleaned = str(v).strip().lower()
            if cleaned not in allowed_statuses:
                raise ValueError(f"status must be one of {allowed_statuses}")
            return cleaned
        return v
    
    @field_validator('created_by')
    @classmethod
    def validate_created_by_format(cls, v):
        """Validate created_by if provided"""
        if v is not None and str(v).strip():
            cleaned = str(v).strip()
            # Basic length check
            if len(cleaned) > 100:
                raise ValueError("created_by too long (max 100 characters)")
            return cleaned
        return v
    
    @model_validator(mode='after')
    def validate_at_least_one_criteria(self):
        """Ensure at least one search criteria is provided"""
        has_organization = self.organization is not None and str(self.organization).strip()
        has_macro_name = self.macro_name is not None and str(self.macro_name).strip()
        has_status = self.status is not None and str(self.status).strip()
        has_created_by = self.created_by is not None and str(self.created_by).strip()
        
        if not any([has_organization, has_macro_name, has_status, has_created_by]):
            raise ValueError("At least one search criteria must be provided")
        
        return self
