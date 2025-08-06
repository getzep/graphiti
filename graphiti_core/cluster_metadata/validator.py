"""
Updated ClusterValidator that uses the efficient cache service.

This demonstrates how to integrate the ValidationCacheService with the existing validator.
"""

import re
from typing import List, Optional, Set
from .models import ClusterCreateRequest, ClusterUpdateRequest
from .mongo_service import get_collection
from .exceptions import (
    InvalidOrganizationError,
)
from .cache_service import get_cache_service

class EnhancedClusterValidator:
    """
    Enhanced validator for cluster metadata operations with efficient caching.
    
    Provides validation for cluster patterns, organization names, macro names,
    with persistent, expiring cache support.
    """
    
    def __init__(self):
        """Initialize validator with cache service."""
        # Get the centralized cache service
        self.cache_service = get_cache_service()
    
    async def validate_organization(self, organization: str) -> str:
        """
        Validate organization name using cached results and dynamic integration configs.
        
        Args:
            organization: Organization name to validate
            
        Returns:
            Normalized organization name in proper case as stored in splunk_macros_kb
            
        Raises:
            InvalidOrganizationError: If organization is not approved
        """
        if not organization:
            raise InvalidOrganizationError("", [])
        
        org_normalized = organization.lower().strip()
        
        # Check cache first for performance
        cached_result = self.cache_service.get_organization_validity(org_normalized)
        if cached_result is not None:
            if cached_result:
                # Get the proper case name from splunk_macros_kb
                proper_case_name = await self.get_proper_case_organization_name(org_normalized)
                return proper_case_name if proper_case_name else org_normalized
            else:
                raise InvalidOrganizationError(organization, [])
        
        # Not in cache, validate against integration configs
        try:
            is_valid_integration = await self.is_organization_in_integrations(org_normalized)
            
            if is_valid_integration:
                # Get the proper case name from splunk_macros_kb
                proper_case_name = await self.get_proper_case_organization_name(org_normalized)
                
                # Cache the result for future use
                self.cache_service.cache_organization(org_normalized, True)
                
                return proper_case_name if proper_case_name else organization
            else:
                # Cache negative result
                self.cache_service.cache_organization(org_normalized, False)
                raise InvalidOrganizationError(organization, [])
                
        except Exception as e:
            # Log warning and cache negative result to avoid repeated failures
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to validate organization against integrations: {e}")
            
            # Cache negative result for short period to avoid repeated DB hits
            self.cache_service.cache_organization(org_normalized, False)
            raise InvalidOrganizationError(organization, [])

    async def validate_macro_name(self, macro_name: str, organization: str) -> bool:
        """
        Validate macro name with caching support by searching organization's macros in splunk_macros_kb collection.
        
        Args:
            macro_name: Macro name to validate
            organization: Organization context for validation
            
        Returns:
            True if macro is valid for the organization, False otherwise
        """
        if not macro_name or not organization:
            return False
        
        macro_normalized = macro_name.lower().strip()
        org_normalized = organization.lower().strip()
        
        # Check cache first
        cached_result = self.cache_service.get_macro_validity(macro_normalized, org_normalized)
        if cached_result is not None:
            return cached_result
        
        try:
            # Get the collection
            collection = await get_collection("splunk_macros_kb")
            
            # Search for all macros for this organization (case insensitive)
            query = {"customer": {"$regex": f"^{org_normalized}$", "$options": "i"}}
            cursor = collection.find(query)
            org_macros = await cursor.to_list(length=None)
            
            # Check if organization has any macros
            if not org_macros:
                # Cache negative result for organization
                self.cache_service.cache_macro(macro_normalized, org_normalized, False)
                return False
            
            # Extract all macro names for this organization
            available_macros = set()
            for macro_doc in org_macros:
                if "macro_name" in macro_doc and macro_doc["macro_name"]:
                    available_macros.add(macro_doc["macro_name"].lower().strip())
            
            # Check if the requested macro exists for this organization
            is_valid = macro_normalized in available_macros
            
            # Cache the result
            self.cache_service.cache_macro(macro_normalized, org_normalized, is_valid)
            
            return is_valid
                
        except Exception as e:
            # Log error and cache negative result
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to validate macro '{macro_name}' for organization '{organization}': {e}")
            
            # Cache negative result to avoid repeated DB hits
            self.cache_service.cache_macro(macro_normalized, org_normalized, False)
            
            return False
    
    async def get_proper_case_organization_name(self, org_normalized: str) -> str:
        """
        Get the proper case organization name as stored in splunk_macros_kb.
        
        Args:
            org_normalized: Lowercase normalized organization name
            
        Returns:
            Organization name in proper case as stored in splunk_macros_kb
        """
        try:
            # Get the collection
            collection = await get_collection("splunk_macros_kb")
            
            # Search for the organization in the macro knowledge base (case insensitive)
            query = {"customer": {"$regex": f"^{org_normalized}$", "$options": "i"}}
            cursor = collection.find(query).limit(1)
            org_docs = await cursor.to_list(length=1)
            
            if org_docs:
                # Return the organization name as stored in the database
                return org_docs[0].get("customer", org_normalized.title())
            else:
                # Fallback to title case if not found
                return org_normalized.title()
                
        except Exception as e:
            # Log error and return title case as fallback
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to get proper case for organization '{org_normalized}': {e}")
            return org_normalized.title()
    
    @staticmethod
    async def is_organization_in_integrations(org_name: str) -> bool:
        """
        Check if organization has Splunk integration configured.
        
        This method includes caching at the integration level for bulk operations.
        
        Args:
            org_name: Organization name to validate
            
        Returns:
            True if organization has valid Splunk integration
        """
        try:
            # Get cache service for integration-level caching
            cache_service = get_cache_service()
            
            # Check if we have cached integration organizations
            cached_orgs = cache_service.get_integration_organizations()
            if cached_orgs is not None:
                return org_name.lower() in cached_orgs
            
            # Not cached, fetch from database
            collection = await get_collection("integration-configs")
            
            # Query for documents with config.secret ending with "_SPLUNK" (case insensitive)
            query = {
                "config.secret": {"$regex": "_SPLUNK$", "$options": "i"}
            }
            
            # Find integration configs that match the pattern
            cursor = collection.find(query)
            integration_configs = await cursor.to_list(length=None)
            
            # Extract organization names from integration configs
            valid_organizations = set()
            
            for config in integration_configs:
                # Extract organization from secret name (assuming format: "ORGNAME_SPLUNK")
                secret = config.get("config", {}).get("secret", "")
                if secret and secret.upper().endswith("_SPLUNK"):
                    # Extract org name (everything before "_SPLUNK")
                    org_from_secret = secret[:-7].lower()  # Remove "_SPLUNK" and lowercase
                    valid_organizations.add(org_from_secret)
                else:
                    raise ValueError("Invalid integration config format")

            # Cache the results for future use
            cache_service.cache_integration_organizations(valid_organizations)
            
            # Check if the requested organization is in the valid list
            return org_name.lower() in valid_organizations
            
        except Exception as e:
            # Log error but don't fail validation - fallback to individual caching
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to check organization in integrations: {e}")
            return False
    
    def invalidate_organization_cache(self, organization: str) -> None:
        """
        Invalidate cached data for a specific organization.
        
        Use this when you know an organization's status has changed.
        """
        self.cache_service.invalidate_organization(organization)
    
    def clear_all_cache(self) -> None:
        """Clear all cached validation data."""
        self.cache_service.clear_all()
    
    def get_cache_statistics(self) -> dict:
        """Get cache performance statistics."""
        return self.cache_service.get_cache_stats()
