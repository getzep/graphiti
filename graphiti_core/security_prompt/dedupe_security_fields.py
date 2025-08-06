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


class SecurityFieldDuplicate(BaseModel):
    """Security field duplicate detection and resolution"""
    id: int = Field(..., description='Integer id of the security field')
    duplicate_idx: int = Field(
        ...,
        description='Index of the duplicate field. If no duplicate fields are found, default to -1.',
    )
    name: str = Field(
        ...,
        description='Name of the security field. Should be the most complete and descriptive name. Do not include any JSON formatting.',
    )
    duplicates: list[int] = Field(
        ...,
        description='Index of all duplicate fields.',
    )
    cluster_context: str = Field(
        ...,
        description='Organizational cluster context for deduplication scope',
    )
    semantic_similarity_score: float = Field(
        ...,
        description='Semantic similarity score (0.0 to 1.0) for duplicate detection',
    )


class SecurityFieldResolutions(BaseModel):
    """Collection of security field duplicate resolutions"""
    field_resolutions: list[SecurityFieldDuplicate] = Field(..., description='List of resolved security fields')


class SecurityFieldDeduplicationPrompt(Protocol):
    """Protocol for security field deduplication prompts"""
    dedupe_single_field: PromptVersion
    dedupe_field_list: PromptVersion
    dedupe_fields_batch: PromptVersion


class Versions(TypedDict):
    """Type definitions for security field deduplication prompt functions"""
    dedupe_single_field: PromptFunction
    dedupe_field_list: PromptFunction
    dedupe_fields_batch: PromptFunction


def dedupe_single_field(context: dict[str, Any]) -> list[Message]:
    """Determine if a new security field is a duplicate of existing fields"""
    return [
        Message(
            role='system',
            content='You are a helpful assistant that determines whether or not a NEW SECURITY FIELD is a duplicate of any EXISTING SECURITY FIELDS using semantic similarity and hierarchical context.',
        ),
        Message(
            role='user',
            content=f"""
        <EXISTING FIELD RELATIONSHIPS>
        {json.dumps(context['field_relationships'], indent=2)}
        </EXISTING FIELD RELATIONSHIPS>
        
        <CURRENT FIELD DOCUMENTATION>
        {context['field_documentation']}
        </CURRENT FIELD DOCUMENTATION>
        
        <NEW SECURITY FIELD>
        {json.dumps(context['extracted_security_field'], indent=2)}
        </NEW SECURITY FIELD>
        
        <FIELD TYPE DESCRIPTION>
        {json.dumps(context['field_type_description'], indent=2)}
        </FIELD TYPE DESCRIPTION>

        <EXISTING SECURITY FIELDS>
        {json.dumps(context['existing_security_fields'], indent=2)}
        </EXISTING SECURITY FIELDS>
        
        <CLUSTER ORGANIZATION RULES>
        {json.dumps(context['cluster_rules'], indent=2)}
        </CLUSTER ORGANIZATION RULES>

        Determine if the NEW SECURITY FIELD is a duplicate of any EXISTING SECURITY FIELDS.

        Deduplication Criteria:
        1. **Field Name Similarity**: Compare field names accounting for naming variations (src_ip vs source_ip)
        2. **Semantic Similarity**: Compare field descriptions and purposes for semantic equivalence
        3. **Data Type Matching**: Fields with same data types and similar value patterns
        4. **Cluster Context**: Consider organizational cluster context (same cluster may allow name variations)
        5. **Hierarchical Position**: Compare SubCluster and Cluster membership

        Evaluation Guidelines:
        - **Exact Duplicates**: Same field name, description, and cluster context
        - **Naming Variations**: Different names but same semantic meaning (user_id vs userid)  
        - **Cluster Isolation**: Fields in different organizational clusters are NOT duplicates
        - **Data Type Consistency**: Compare data types, examples, and validation rules
        - **Functional Equivalence**: Fields serving the same purpose in security monitoring

        Decision Logic:
        - If fields have semantic similarity score > 0.8 AND same cluster context → DUPLICATE
        - If fields have identical names AND same cluster → DUPLICATE  
        - If fields have different organizational clusters → NOT DUPLICATE
        - If fields have significantly different data types or purposes → NOT DUPLICATE

        Provide reasoning for duplicate detection decision including similarity analysis.
        """,
        ),
    ]


def dedupe_field_list(context: dict[str, Any]) -> list[Message]:
    """Deduplicate a list of security fields within organizational context"""
    return [
        Message(
            role='system',
            content='You are a helpful assistant that identifies and resolves duplicates within a list of security fields while maintaining organizational cluster isolation.',
        ),
        Message(
            role='user',
            content=f"""
        <FIELD DOCUMENTATION>
        {context['field_documentation']}
        </FIELD DOCUMENTATION>
        
        <SECURITY FIELD LIST>
        {json.dumps(context['security_field_list'], indent=2)}
        </SECURITY FIELD LIST>
        
        <CLUSTER HIERARCHY>
        {json.dumps(context['cluster_hierarchy'], indent=2)}
        </CLUSTER HIERARCHY>
        
        <CLUSTER ORGANIZATION RULES>
        {json.dumps(context['cluster_rules'], indent=2)}
        </CLUSTER ORGANIZATION RULES>

        Identify and resolve duplicate security fields within the SECURITY FIELD LIST.

        Deduplication Process:
        1. **Cluster-Based Grouping**: Group fields by organizational cluster (cluster_partition_id)
        2. **Within-Cluster Deduplication**: Identify duplicates only within same organizational boundaries
        3. **Semantic Analysis**: Compare field descriptions for semantic similarity
        4. **Name Normalization**: Standardize field naming conventions (src_ip, source_ip → src_ip)
        5. **Resolution Strategy**: Merge duplicate field properties while preserving all relationships

        Resolution Guidelines:
        - **Field Name**: Use most comprehensive and standard naming convention
        - **Description**: Merge descriptions to include all relevant information
        - **Data Type**: Use most specific data type definition
        - **Examples**: Combine example values from all duplicate instances
        - **Cluster Relationships**: Preserve all SubCluster memberships across duplicates
        - **Organizational Isolation**: Maintain separate instances across different clusters

        Duplicate Detection Criteria:
        - Same field name variations within same cluster
        - Semantic similarity score > 0.8 for descriptions
        - Identical data types and similar example values
        - Same functional purpose in security monitoring

        Output the resolved field list with duplicate consolidation and preserved cluster relationships.
        """,
        ),
    ]


def dedupe_fields_batch(context: dict[str, Any]) -> list[Message]:
    """Batch deduplication of security fields across multiple extractions"""
    return [
        Message(
            role='system',
            content='You are a helpful assistant that performs batch deduplication of security fields across multiple extraction batches while maintaining cluster isolation and relationship integrity.',
        ),
        Message(
            role='user',
            content=f"""
        <EXISTING FIELD RELATIONSHIPS>
        {json.dumps(context['field_relationships'], indent=2)}
        </EXISTING FIELD RELATIONSHIPS>
        
        <CURRENT EXTRACTION BATCH>
        {json.dumps(context['current_field_batch'], indent=2)}
        </CURRENT EXTRACTION BATCH>
        
        <PREVIOUS EXTRACTION BATCHES>
        {json.dumps(context['previous_field_batches'], indent=2)}
        </PREVIOUS EXTRACTION BATCHES>
        
        <CLUSTER HIERARCHY>
        {json.dumps(context['cluster_hierarchy'], indent=2)}
        </CLUSTER HIERARCHY>
        
        <CLUSTER ORGANIZATION RULES>
        {json.dumps(context['cluster_rules'], indent=2)}
        </CLUSTER ORGANIZATION RULES>

        Perform batch deduplication across CURRENT EXTRACTION BATCH and PREVIOUS EXTRACTION BATCHES.

        Batch Deduplication Strategy:
        1. **Cross-Batch Analysis**: Compare current batch against all previous batches
        2. **Cluster Isolation**: Maintain organizational boundaries during deduplication
        3. **Incremental Processing**: Preserve existing field relationships and UUIDs
        4. **Conflict Resolution**: Handle cases where same field appears with different properties
        5. **Relationship Preservation**: Maintain all existing BELONGS_TO, CONTAINS, FIELD_RELATES_TO relationships

        Processing Guidelines:
        - **UUID Preservation**: Keep existing UUIDs for previously processed fields
        - **Property Merging**: Enhance field definitions with new information from current batch
        - **Relationship Updates**: Add new cluster/subcluster relationships without breaking existing ones
        - **Validation**: Ensure all deduplicated fields maintain cluster isolation constraints
        - **Version Control**: Track field property changes across extraction batches

        Conflict Resolution Rules:
        - If field exists with different data_type → Use most specific/recent definition
        - If field has new examples → Merge with existing examples (no duplicates)
        - If field gains new cluster membership → Add relationships while preserving existing
        - If field descriptions differ → Merge to create comprehensive description
        - If field appears in new cluster → Create separate instance (maintain isolation)

        Output consolidated field definitions with preserved UUIDs and enhanced relationships.
        """,
        ),
    ]


versions: Versions = {
    'dedupe_single_field': dedupe_single_field,
    'dedupe_field_list': dedupe_field_list,
    'dedupe_fields_batch': dedupe_fields_batch,
}
