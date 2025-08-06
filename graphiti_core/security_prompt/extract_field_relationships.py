"""
Copyright 2024, Zep Software, Inc.

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
from typing import Any, Protocol, TypedDict

from pydantic import BaseModel, Field

from .models import Message, PromptFunction, PromptVersion


class SecurityFieldRelationship(BaseModel):
    """Security field relationship with cluster isolation constraints"""
    relationship_type: str = Field(..., description='BELONGS_TO, CONTAINS, or FIELD_RELATES_TO')
    source_field_uuid: str = Field(..., description='UUID of source security field')
    target_field_uuid: str = Field(..., description='UUID of target security field')
    cluster_partition_id: str = Field(..., description='Cluster isolation constraint')
    semantic_context: str = Field(..., description='Why this relationship exists')
    confidence: float = Field(default=1.0, description='Relationship confidence score (0.0 to 1.0)')
    valid_at: str | None = Field(
        None,
        description='When the relationship became valid. Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS.SSSSSSZ)',
    )
    invalid_at: str | None = Field(
        None,
        description='When the relationship stopped being valid. Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS.SSSSSSZ)',
    )


class ExtractedSecurityRelationships(BaseModel):
    """Collection of extracted security field relationships"""
    relationships: list[SecurityFieldRelationship]


class MissingRelationships(BaseModel):
    """Security field relationships that weren't extracted"""
    missing_relationships: list[str] = Field(..., description="relationships that weren't extracted")


class SecurityFieldRelationshipPrompt(Protocol):
    """Protocol for security field relationship extraction prompts"""
    extract_belongs_to_relationships: PromptVersion
    extract_contains_relationships: PromptVersion
    extract_field_correlations: PromptVersion
    validate_cluster_isolation: PromptVersion


class Versions(TypedDict):
    """Type definitions for security field relationship prompt functions"""
    extract_belongs_to_relationships: PromptFunction
    extract_contains_relationships: PromptFunction
    extract_field_correlations: PromptFunction
    validate_cluster_isolation: PromptFunction


def extract_belongs_to_relationships(context: dict[str, Any]) -> list[Message]:
    """Extract BELONGS_TO relationships between Field and SubCluster nodes"""
    return [
        Message(
            role='system',
            content='You are an expert relationship extractor that identifies BELONGS_TO relationships between Field and SubCluster nodes. '
            '1. Extracted relationships should maintain cluster isolation constraints.'
            '2. All Fields must belong to SubClusters within the same organizational Cluster.'
            '3. Validate hierarchical positioning and organizational boundaries.',
        ),
        Message(
            role='user',
            content=f"""
<EXISTING_FIELD_RELATIONSHIPS>
{json.dumps(context['field_relationships'], indent=2)}
</EXISTING_FIELD_RELATIONSHIPS>

<CURRENT_FIELD_DOCUMENTATION>
{context['field_documentation']}
</CURRENT_FIELD_DOCUMENTATION>

<SECURITY_FIELDS>
{context['security_fields']} 
</SECURITY_FIELDS>

<CLUSTER_HIERARCHY>
{json.dumps(context['cluster_hierarchy'], indent=2)}
</CLUSTER_HIERARCHY>

<CLUSTER_ORGANIZATION_RULES>
{json.dumps(context['cluster_rules'], indent=2)}
</CLUSTER_ORGANIZATION_RULES>

# TASK
Extract all BELONGS_TO relationships between Field and SubCluster nodes based on the CURRENT FIELD DOCUMENTATION.
Only extract relationships that:
- Connect Field nodes to SubCluster nodes within the same organizational Cluster
- Are clearly stated or unambiguously implied in the CURRENT FIELD DOCUMENTATION
- Maintain cluster isolation constraints (no cross-cluster field memberships)
- Respect hierarchical positioning rules

Relationship Extraction Guidelines:
1. **Field → SubCluster BELONGS_TO**: Extract membership relationships where Fields belong to functional SubClusters
2. **Cluster Isolation**: Ensure all relationships respect organizational boundaries (same cluster_partition_id)
3. **Hierarchical Validation**: Validate that SubClusters exist within the target Cluster before creating relationships
4. **Semantic Context**: Provide clear explanation for why each relationship exists
5. **Confidence Scoring**: Assign confidence scores based on documentation clarity

Constraint Validation:
- Field and SubCluster must belong to the same organizational Cluster
- SubCluster must be a valid functional grouping within the Cluster
- No duplicate BELONGS_TO relationships for the same Field-SubCluster pair
- Maintain referential integrity with existing cluster hierarchy

{context['custom_relationship_prompt']}
""",
        ),
    ]


def extract_contains_relationships(context: dict[str, Any]) -> list[Message]:
    """Extract CONTAINS relationships between Cluster and SubCluster nodes"""
    return [
        Message(
            role='system',
            content='You are an expert relationship extractor that identifies CONTAINS relationships between Cluster and SubCluster nodes. '
            '1. Extracted relationships should enforce SubCluster uniqueness within Clusters.'
            '2. Each SubCluster must belong to exactly one Cluster.'
            '3. Validate organizational hierarchy and namespace isolation.',
        ),
        Message(
            role='user',
            content=f"""
<EXISTING_FIELD_RELATIONSHIPS>
{json.dumps(context['field_relationships'], indent=2)}
</EXISTING_FIELD_RELATIONSHIPS>

<CURRENT_FIELD_DOCUMENTATION>
{context['field_documentation']}
</CURRENT_FIELD_DOCUMENTATION>

<SECURITY_FIELDS>
{context['security_fields']} 
</SECURITY_FIELDS>

<CLUSTER_HIERARCHY>
{json.dumps(context['cluster_hierarchy'], indent=2)}
</CLUSTER_HIERARCHY>

<CLUSTER_ORGANIZATION_RULES>
{json.dumps(context['cluster_rules'], indent=2)}
</CLUSTER_ORGANIZATION_RULES>

# TASK
Extract all CONTAINS relationships between Cluster and SubCluster nodes based on the CURRENT FIELD DOCUMENTATION.
Only extract relationships that:
- Connect Cluster nodes to their functional SubCluster subdivisions
- Are clearly stated or unambiguously implied in the CURRENT FIELD DOCUMENTATION
- Maintain SubCluster uniqueness within each Cluster
- Establish proper organizational hierarchy

Relationship Extraction Guidelines:
1. **Cluster → SubCluster CONTAINS**: Extract hierarchical relationships where Clusters contain functional SubClusters
2. **Uniqueness Validation**: Ensure no duplicate SubCluster names within the same Cluster
3. **Organizational Boundaries**: Each SubCluster belongs to exactly one Cluster (1:1 relationship)
4. **Namespace Isolation**: SubCluster names are isolated by their parent Cluster
5. **Functional Grouping**: SubClusters represent functional categories within organizational boundaries

Constraint Validation:
- SubCluster names must be unique within their parent Cluster
- Each SubCluster can only have one CONTAINS relationship (from one Cluster)
- No cross-cluster SubCluster relationships
- Maintain organizational hierarchy integrity

{context['custom_relationship_prompt']}
""",
        ),
    ]


def extract_field_correlations(context: dict[str, Any]) -> list[Message]:
    """Extract FIELD_RELATES_TO relationships between Field nodes within same Cluster"""
    return [
        Message(
            role='system',
            content='You are an expert relationship extractor that identifies semantic correlations between Field nodes. '
            '1. All FIELD_RELATES_TO relationships must respect cluster isolation constraints.'
            '2. Fields can only relate to other Fields within the same organizational Cluster.'
            '3. Focus on semantic connections like correlation, similarity, or derivation.',
        ),
        Message(
            role='user',
            content=f"""
<EXISTING_FIELD_RELATIONSHIPS>
{json.dumps(context['field_relationships'], indent=2)}
</EXISTING_FIELD_RELATIONSHIPS>

<CURRENT_FIELD_DOCUMENTATION>
{context['field_documentation']}
</CURRENT_FIELD_DOCUMENTATION>

<SECURITY_FIELDS>
{context['security_fields']} 
</SECURITY_FIELDS>

<CLUSTER_HIERARCHY>
{json.dumps(context['cluster_hierarchy'], indent=2)}
</CLUSTER_HIERARCHY>

<SEMANTIC_RELATIONSHIP_TYPES>
{json.dumps(context['semantic_relationship_types'], indent=2)}
</SEMANTIC_RELATIONSHIP_TYPES>

# TASK
Extract all FIELD_RELATES_TO relationships between Field nodes based on the CURRENT FIELD DOCUMENTATION.
Only extract relationships that:
- Connect Field nodes to other Field nodes within the same organizational Cluster
- Represent semantic connections (correlation, similarity, derivation, dependency)
- Are clearly stated or unambiguously implied in the CURRENT FIELD DOCUMENTATION
- Maintain cluster isolation constraints

Relationship Extraction Guidelines:
1. **Field → Field FIELD_RELATES_TO**: Extract semantic relationships between Fields
2. **Cluster Isolation**: Both Fields must belong to the same organizational Cluster (same cluster_partition_id)
3. **Semantic Types**: Use specific relationship names (CORRELATES_WITH, SIMILAR_TO, DERIVED_FROM, DEPENDS_ON)
4. **Bidirectional Analysis**: Consider if relationships should be bidirectional or unidirectional
5. **Confidence Scoring**: Assign confidence based on documentation strength and semantic clarity

Semantic Relationship Categories:
- **CORRELATES_WITH**: Fields that frequently appear together in security events
- **SIMILAR_TO**: Fields with similar data types, purposes, or values
- **DERIVED_FROM**: Fields calculated or extracted from other fields
- **DEPENDS_ON**: Fields that require other fields for proper interpretation
- **VALIDATES**: Fields that validate or verify values of other fields

Constraint Validation:
- Both source and target Fields must belong to the same Cluster
- No self-referential relationships (Field relating to itself)
- Validate semantic relationship type matches documentation context
- Maintain cluster isolation throughout relationship extraction

{context['custom_relationship_prompt']}
""",
        ),
    ]


def validate_cluster_isolation(context: dict[str, Any]) -> list[Message]:
    """Validate that all extracted relationships respect cluster isolation constraints"""
    return [
        Message(
            role='system',
            content='You are an AI assistant that validates cluster isolation constraints for security field relationships.',
        ),
        Message(
            role='user',
            content=f"""
<EXTRACTED_RELATIONSHIPS>
{json.dumps(context['extracted_relationships'], indent=2)}
</EXTRACTED_RELATIONSHIPS>

<CLUSTER_HIERARCHY>
{json.dumps(context['cluster_hierarchy'], indent=2)}
</CLUSTER_HIERARCHY>

<CLUSTER_ORGANIZATION_RULES>
{json.dumps(context['cluster_rules'], indent=2)}
</CLUSTER_ORGANIZATION_RULES>

Given the above extracted relationships and cluster hierarchy, validate that all relationships respect cluster isolation constraints.

Validation Criteria:
1. **BELONGS_TO Relationships**: Field and SubCluster must belong to the same Cluster
2. **CONTAINS Relationships**: SubCluster names must be unique within each Cluster
3. **FIELD_RELATES_TO Relationships**: Both Fields must belong to the same Cluster
4. **Organizational Isolation**: No cross-cluster relationships allowed
5. **Hierarchical Integrity**: All relationships must maintain proper hierarchy

Constraint Violations to Check:
- Cross-cluster Field relationships (different cluster_partition_id values)
- Duplicate SubCluster names within the same Cluster
- Field memberships spanning multiple organizational Clusters
- Invalid hierarchical relationships (Field directly connected to Cluster)
- Missing cluster_partition_id validation

Report any constraint violations and suggest corrections to maintain organizational isolation and hierarchical integrity.
""",
        ),
    ]


versions: Versions = {
    'extract_belongs_to_relationships': extract_belongs_to_relationships,
    'extract_contains_relationships': extract_contains_relationships,
    'extract_field_correlations': extract_field_correlations,
    'validate_cluster_isolation': validate_cluster_isolation,
}
