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


class ExtractedSecurityField(BaseModel):
    """Extracted security field with hierarchical context"""
    name: str = Field(..., description='Security field name (e.g., src_ip, user_id)')
    field_type: str = Field(..., description='Type: Field, Cluster, or SubCluster')
    field_name: str = Field(..., description='Actual field name for Field nodes')
    description: str = Field(..., description='Field description for AI context and embedding')
    data_type: str = Field(..., description='Data type (string, integer, ip_address, etc.)')
    examples: list[str] = Field(default_factory=list, description='Example values for validation')
    cluster_context: str = Field(..., description='Organizational cluster this field belongs to')
    subcluster_context: str | None = Field(default=None, description='Functional sub-cluster context')
    hierarchical_position: dict = Field(..., description='Position in cluster hierarchy')


class ExtractedSecurityFields(BaseModel):
    """Collection of extracted security fields"""
    extracted_fields: list[ExtractedSecurityField] = Field(..., description='List of extracted security fields')


class MissedSecurityFields(BaseModel):
    """Security fields that weren't extracted"""
    missed_fields: list[str] = Field(..., description="Names of security fields that weren't extracted")


class SecurityFieldClassificationTriple(BaseModel):
    """Security field classification with hierarchical context"""
    uuid: str = Field(description='UUID of the security field')
    name: str = Field(description='Name of the security field')
    field_type: str = Field(description='Field, Cluster, or SubCluster')
    cluster_partition_id: str = Field(description='Cluster isolation identifier')


class SecurityFieldClassification(BaseModel):
    """Collection of security field classifications"""
    field_classifications: list[SecurityFieldClassificationTriple] = Field(
        ..., description='List of security field classification triples.'
    )


class SecurityFieldPrompt(Protocol):
    """Protocol for security field extraction prompts"""
    extract_field_documentation: PromptVersion
    extract_field_specifications: PromptVersion
    extract_field_hierarchy: PromptVersion
    validate_field_extraction: PromptVersion
    classify_security_fields: PromptVersion
    extract_field_properties: PromptVersion


class Versions(TypedDict):
    """Type definitions for security field prompt functions"""
    extract_field_documentation: PromptFunction
    extract_field_specifications: PromptFunction
    extract_field_hierarchy: PromptFunction
    validate_field_extraction: PromptFunction
    classify_security_fields: PromptFunction
    extract_field_properties: PromptFunction


def extract_field_documentation_raw(context: dict[str, Any]) -> list[Message]:
    """Extract security fields from field documentation with hierarchical context"""
    sys_prompt = """You are an AI assistant specialized in extracting security field definitions from documentation. 
    Your primary task is to identify and classify Field, Cluster, and SubCluster nodes from security field specifications."""

    user_prompt = f"""
<EXISTING FIELD RELATIONSHIPS>
{json.dumps(context['field_relationships'], indent=2)}
</EXISTING FIELD RELATIONSHIPS>

<CURRENT FIELD DOCUMENTATION>
{context['field_documentation']}
</CURRENT FIELD DOCUMENTATION>

<CLUSTER HIERARCHY CONTEXT>
{json.dumps(context['cluster_hierarchy'], indent=2)}
</CLUSTER HIERARCHY CONTEXT>

<SECURITY FIELD TYPES>
{json.dumps(context['security_field_types'], indent=2)}
</SECURITY FIELD TYPES>

Instructions:

You are given security field documentation and existing cluster hierarchy. Your task is to extract **security field nodes** defined **explicitly or implicitly** in the CURRENT FIELD DOCUMENTATION.

1. **Field Node Extraction**: Identify individual security fields (e.g., src_ip, user_id, process_name) with their:
   - Field name and description
   - Data type and example values  
   - Cluster and SubCluster membership context

2. **Cluster Node Identification**:
   - Extract organizational/macro clusters (e.g., linux_audit_batelco, windows_security_sico)
   - Identify organization and macro_type properties
   - **Exclude** clusters mentioned only in EXISTING FIELD RELATIONSHIPS (they are for context only)

3. **SubCluster Node Detection**:
   - Extract functional sub-clusters (e.g., UserAuthentication, NetworkAccess, ProcessExecution)
   - Identify parent cluster relationships
   - Establish hierarchical positioning within organizational boundaries

4. **Hierarchical Validation**:
   - Ensure Fields belong to valid SubClusters within their assigned Cluster
   - Validate SubCluster uniqueness within parent Clusters
   - Maintain cluster isolation constraints

5. **Field Property Extraction**:
   - Extract data types, descriptions, and example values
   - Identify field occurrence and distinct value counts
   - Generate embeddings for semantic search capabilities

6. **Formatting**:
   - Be **explicit and unambiguous** in naming security fields (use full field names when available)
   - Maintain organizational context for proper cluster assignment

{context['custom_field_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_field_specifications(context: dict[str, Any]) -> list[Message]:
    """Extract security fields from structured field specifications"""
    sys_prompt = """You are an AI assistant that extracts security field specifications from structured documentation. 
    Your primary task is to parse field definitions and create Field, Cluster, and SubCluster nodes with proper hierarchical relationships."""

    user_prompt = f"""
<FIELD SPECIFICATION DOCUMENT>
{context['field_specifications']}
</FIELD SPECIFICATION DOCUMENT>

<CLUSTER ORGANIZATION RULES>
{json.dumps(context['cluster_rules'], indent=2)}
</CLUSTER ORGANIZATION RULES>

<SECURITY FIELD TYPES>
{context['security_field_types']}
</SECURITY FIELD TYPES>

{context['custom_field_prompt']}

Extract security field definitions from the FIELD SPECIFICATION DOCUMENT that explicitly define field properties and relationships.
For each field extracted, determine its hierarchical position based on the CLUSTER ORGANIZATION RULES.
Indicate the classified field type (Field/Cluster/SubCluster) by providing its security_field_type.

Guidelines:
1. Always extract individual Field nodes with their complete specifications (name, description, data_type, examples)
2. Identify organizational Cluster nodes and their macro_type/organization properties
3. Extract functional SubCluster nodes and their parent Cluster relationships
4. Ensure cluster isolation constraints are maintained
5. DO NOT extract relationships between fields (these will be handled separately)
6. Focus on field specifications that define actual security monitoring fields
7. Maintain proper hierarchical context for each extracted field
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_field_hierarchy(context: dict[str, Any]) -> list[Message]:
    """Extract security field hierarchical structure from text"""
    sys_prompt = """You are an AI assistant that extracts security field hierarchical structures from text documentation. 
    Your primary task is to identify organizational clusters, functional sub-clusters, and field relationships."""

    user_prompt = f"""
<FIELD HIERARCHY TEXT>
{context['field_documentation']}
</FIELD HIERARCHY TEXT>

<SECURITY FIELD TYPES>
{context['security_field_types']}
</SECURITY FIELD TYPES>

Given the above text, extract security field hierarchy information that explicitly or implicitly defines organizational structure.
For each field extracted, also determine its hierarchical position based on the provided organizational context.
Indicate the classified field type (Field/Cluster/SubCluster) by providing its security_field_type.

{context['custom_field_prompt']}

Guidelines:
1. Extract organizational Clusters with their macro_type and organization properties
2. Identify functional SubClusters and their parent Cluster relationships  
3. Extract individual Fields with their cluster/subcluster membership context
4. Avoid creating nodes for relationships or actions between fields
5. Avoid creating nodes for temporal information like dates or timestamps
6. Be as explicit as possible in field names, using full names and avoiding abbreviations
7. Maintain cluster isolation constraints throughout extraction
8. Focus on security-relevant field definitions and organizational structures
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def validate_field_extraction(context: dict[str, Any]) -> list[Message]:
    """Validate completeness and accuracy of security field extraction"""
    sys_prompt = """You are an AI assistant that validates security field extraction completeness and accuracy"""

    user_prompt = f"""
<EXISTING FIELD RELATIONSHIPS>
{json.dumps(context['field_relationships'], indent=2)}
</EXISTING FIELD RELATIONSHIPS>

<CURRENT FIELD DOCUMENTATION>
{context['field_documentation']}
</CURRENT FIELD DOCUMENTATION>

<EXTRACTED SECURITY FIELDS>
{context['extracted_security_fields']}
</EXTRACTED SECURITY FIELDS>

Given the above field documentation, existing relationships, and list of extracted security fields; determine if any fields, clusters, or subclusters haven't been extracted.

Validation Criteria:
1. All individual security fields defined in documentation are extracted
2. All organizational clusters referenced are identified  
3. All functional subclusters mentioned are captured
4. Hierarchical relationships are properly identified
5. Cluster isolation constraints are maintained
6. Field specifications include complete property definitions (data_type, examples, description)

Missing Field Analysis:
- Compare documentation against extracted fields for completeness
- Identify any organizational structures not captured
- Validate hierarchical positioning accuracy
- Ensure all security-relevant fields are included
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def classify_security_fields(context: dict[str, Any]) -> list[Message]:
    """Classify security field nodes based on hierarchical position and organizational context"""
    sys_prompt = """You are an AI assistant that classifies security field nodes based on their hierarchical position and organizational context"""

    user_prompt = f"""
<EXISTING FIELD RELATIONSHIPS>
{json.dumps(context['field_relationships'], indent=2)}
</EXISTING FIELD RELATIONSHIPS>

<CURRENT FIELD DOCUMENTATION>
{context['field_documentation']}
</CURRENT FIELD DOCUMENTATION>
    
<EXTRACTED SECURITY FIELDS>
{context['extracted_security_fields']}
</EXTRACTED SECURITY FIELDS>
    
<SECURITY FIELD TYPES>
{context['security_field_types']}
</SECURITY FIELD TYPES>

<CLUSTER ORGANIZATION RULES>
{json.dumps(context['cluster_rules'], indent=2)}
</CLUSTER ORGANIZATION RULES>
    
Given the above field documentation, extracted security fields, and organizational rules, classify each extracted field node.
    
Guidelines:
1. Each field must be classified as exactly one type: Field, Cluster, or SubCluster
2. Only use the provided SECURITY FIELD TYPES for classification
3. Validate hierarchical positioning: Fields→SubClusters→Clusters
4. Ensure cluster isolation constraints are maintained
5. If organizational context is ambiguous, classify as Field and specify cluster_partition_id
6. Maintain consistency with existing field relationships where applicable

Classification Logic:
- **Field**: Individual security fields with specific data types and examples
- **Cluster**: Organizational/macro clusters with organization and macro_type properties
- **SubCluster**: Functional sub-clusters within organizational boundaries
- **Validation**: Each classification must respect cluster isolation rules
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_field_properties(context: dict[str, Any]) -> list[Message]:
    """Extract detailed properties for security fields"""
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts security field properties from the provided documentation.',
        ),
        Message(
            role='user',
            content=f"""

        <FIELD DOCUMENTATION>
        {json.dumps(context['field_relationships'], indent=2)}
        {json.dumps(context['field_documentation'], indent=2)}
        </FIELD DOCUMENTATION>

        Given the above FIELD DOCUMENTATION and the following SECURITY FIELD, update any of its attributes based on the information provided
        in the documentation. Use the provided attribute descriptions to better understand how each attribute should be determined.

        Guidelines:
        1. Do not hallucinate field property values if they cannot be found in the current context.
        2. Only use the provided FIELD DOCUMENTATION and SECURITY FIELD to set attribute values.
        3. The description attribute represents a comprehensive description of the SECURITY FIELD, and should be updated with new information about the field from the FIELD DOCUMENTATION. 
            Descriptions must be no longer than 250 words.
        4. Extract data_type, examples, and cluster membership information where available
        5. Maintain hierarchical context (cluster/subcluster relationships)
        6. Focus on security-relevant properties and validation rules
        
        <SECURITY FIELD>
        {context['security_field']}
        </SECURITY FIELD>
        """,
        ),
    ]


versions: Versions = {
    'extract_field_documentation': extract_field_documentation_raw,
    'extract_field_specifications': extract_field_specifications,
    'extract_field_hierarchy': extract_field_hierarchy,
    'validate_field_extraction': validate_field_extraction,
    'classify_security_fields': classify_security_fields,
    'extract_field_properties': extract_field_properties,
}
