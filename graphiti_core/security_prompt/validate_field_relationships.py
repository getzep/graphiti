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


class ValidationResult(BaseModel):
    """Security field relationship validation result"""
    relationship_id: str = Field(..., description='Unique identifier for the relationship')
    is_valid: bool = Field(..., description='Whether the relationship is valid')
    validation_type: str = Field(
        ...,
        description='Type of validation performed (cluster_isolation, hierarchy_consistency, semantic_validity)',
    )
    error_message: str | None = Field(
        None,
        description='Error message if validation failed',
    )
    severity: str = Field(
        ...,
        description='Severity level: critical, warning, info',
    )
    suggested_fix: str | None = Field(
        None,
        description='Suggested fix for validation failure',
    )


class ValidationReport(BaseModel):
    """Comprehensive validation report for security field relationships"""
    total_relationships: int = Field(..., description='Total number of relationships validated')
    valid_relationships: int = Field(..., description='Number of valid relationships')
    validation_results: list[ValidationResult] = Field(..., description='Detailed validation results')
    cluster_compliance: dict[str, float] = Field(
        ...,
        description='Cluster isolation compliance scores',
    )
    hierarchy_integrity: dict[str, Any] = Field(
        ...,
        description='Hierarchy consistency validation results',
    )
    recommendations: list[str] = Field(
        ...,
        description='Recommendations for fixing validation issues',
    )


class SecurityFieldValidationPrompt(Protocol):
    """Protocol for security field relationship validation prompts"""
    validate_cluster_isolation: PromptVersion
    validate_hierarchy_consistency: PromptVersion
    validate_semantic_relationships: PromptVersion
    validate_field_integrity: PromptVersion


class Versions(TypedDict):
    """Type definitions for security field validation prompt functions"""
    validate_cluster_isolation: PromptFunction
    validate_hierarchy_consistency: PromptFunction
    validate_semantic_relationships: PromptFunction
    validate_field_integrity: PromptFunction


def validate_cluster_isolation(context: dict[str, Any]) -> list[Message]:
    """Validate that security field relationships respect organizational cluster isolation"""
    return [
        Message(
            role='system',
            content='You are a helpful assistant that validates organizational cluster isolation for security field relationships, ensuring proper governance and access control boundaries.',
        ),
        Message(
            role='user',
            content=f"""
        <FIELD_RELATIONSHIPS>
        {json.dumps(context['field_relationships'], indent=2)}
        </FIELD_RELATIONSHIPS>
        
        <CLUSTER_HIERARCHY>
        {json.dumps(context['cluster_hierarchy'], indent=2)}
        </CLUSTER_HIERARCHY>
        
        <ORGANIZATIONAL_RULES>
        {json.dumps(context['organizational_rules'], indent=2)}
        </ORGANIZATIONAL_RULES>
        
        <CLUSTER_PERMISSIONS>
        {json.dumps(context['cluster_permissions'], indent=2)}
        </CLUSTER_PERMISSIONS>

        Validate cluster isolation compliance for all security field relationships.

        Cluster Isolation Rules:
        1. **Field Ownership**: Each field must belong to exactly one organizational cluster
        2. **Cross-Cluster Restrictions**: Fields cannot have BELONGS_TO relationships across clusters
        3. **SubCluster Containment**: SubClusters must be fully contained within their parent cluster
        4. **Relationship Boundaries**: FIELD_RELATES_TO relationships must respect cluster boundaries
        5. **Access Control**: Field visibility must be constrained to authorized clusters

        Validation Checks:
        - **BELONGS_TO Validation**: Verify all Field → SubCluster relationships stay within cluster boundaries
        - **CONTAINS Validation**: Verify all SubCluster → Cluster relationships are properly contained
        - **Cross-Cluster Analysis**: Identify any unauthorized cross-cluster field relationships
        - **Permission Compliance**: Validate field access permissions match organizational rules
        - **Isolation Integrity**: Ensure no data leakage between organizational boundaries

        Violation Categories:
        - **Critical**: Field relationships that violate fundamental organizational isolation
        - **Warning**: Relationships that may indicate potential governance issues
        - **Info**: Relationships that require review but may be acceptable

        Validation Process:
        1. Check each field's cluster_partition_id matches its relationships
        2. Verify SubCluster memberships don't span multiple clusters
        3. Validate FIELD_RELATES_TO relationships respect cluster boundaries
        4. Confirm access permissions align with organizational hierarchy
        5. Identify any orphaned fields without proper cluster assignment

        For each validation failure, provide:
        - Specific relationship that violates isolation rules
        - Explanation of why the violation occurred
        - Recommended fix to restore cluster isolation
        - Severity assessment (critical, warning, info)
        - Impact on organizational governance

        Output comprehensive cluster isolation validation report.
        """,
        ),
    ]


def validate_hierarchy_consistency(context: dict[str, Any]) -> list[Message]:
    """Validate consistency of Field → SubCluster → Cluster hierarchical relationships"""
    return [
        Message(
            role='system',
            content='You are a helpful assistant that validates hierarchical consistency in security field relationships, ensuring proper organizational structure and relationship integrity.',
        ),
        Message(
            role='user',
            content=f"""
        <FIELD_RELATIONSHIPS>
        {json.dumps(context['field_relationships'], indent=2)}
        </FIELD_RELATIONSHIPS>
        
        <SECURITY_FIELDS>
        {json.dumps(context['security_fields'], indent=2)}
        </SECURITY_FIELDS>
        
        <CLUSTER_HIERARCHY>
        {json.dumps(context['cluster_hierarchy'], indent=2)}
        </CLUSTER_HIERARCHY>
        
        <HIERARCHY_RULES>
        {json.dumps(context['hierarchy_rules'], indent=2)}
        </HIERARCHY_RULES>

        Validate hierarchical consistency across the Field → SubCluster → Cluster structure.

        Hierarchy Validation Rules:
        1. **Complete Hierarchy**: Every Field must have a valid path to a Cluster via SubCluster
        2. **Single Path**: Each Field should belong to exactly one SubCluster
        3. **SubCluster Integrity**: Each SubCluster must belong to exactly one Cluster
        4. **Transitive Consistency**: Field's cluster assignment must match its SubCluster's cluster
        5. **Orphan Prevention**: No fields, SubClusters, or Clusters should be orphaned

        Consistency Checks:
        - **Field Assignment**: Every Field has exactly one BELONGS_TO → SubCluster relationship
        - **SubCluster Assignment**: Every SubCluster has exactly one CONTAINS → Cluster relationship
        - **Transitive Validation**: Field's inferred cluster matches explicit cluster assignment
        - **Bidirectional Validation**: Relationships are consistent in both directions
        - **Circular Reference Detection**: No circular dependencies in hierarchy

        Structural Validations:
        - **Missing Relationships**: Fields without SubCluster assignment or SubClusters without Cluster assignment
        - **Multiple Assignments**: Fields assigned to multiple SubClusters (policy violation)
        - **Mismatched Hierarchies**: Field's cluster doesn't match its SubCluster's cluster
        - **Dangling References**: Relationships pointing to non-existent entities
        - **Hierarchy Depth**: Validate proper 3-level hierarchy (Field → SubCluster → Cluster)

        Consistency Metrics:
        - **Hierarchy Completeness**: Percentage of fields with complete hierarchy path
        - **Assignment Accuracy**: Accuracy of Field → SubCluster → Cluster assignments
        - **Relationship Integrity**: Percentage of relationships that are bidirectionally consistent
        - **Structural Compliance**: Adherence to 3-level hierarchy rules
        - **Orphan Rate**: Percentage of entities without proper hierarchical assignment

        For each inconsistency, provide:
        - Specific entity or relationship causing the inconsistency
        - Type of hierarchy violation (missing, multiple, mismatched, orphaned)
        - Impact on organizational structure integrity
        - Step-by-step fix to restore consistency
        - Priority level for remediation

        Output detailed hierarchy consistency validation report with remediation plan.
        """,
        ),
    ]


def validate_semantic_relationships(context: dict[str, Any]) -> list[Message]:
    """Validate semantic correctness of field relationships and correlations"""
    return [
        Message(
            role='system',
            content='You are a helpful assistant that validates semantic correctness of security field relationships, ensuring relationships make logical sense and provide meaningful connections.',
        ),
        Message(
            role='user',
            content=f"""
        <FIELD_RELATIONSHIPS>
        {json.dumps(context['field_relationships'], indent=2)}
        </FIELD_RELATIONSHIPS>
        
        <SECURITY_FIELDS>
        {json.dumps(context['security_fields'], indent=2)}
        </SECURITY_FIELDS>
        
        <SEMANTIC_RULES>
        {json.dumps(context['semantic_rules'], indent=2)}
        </SEMANTIC_RULES>
        
        <DOMAIN_KNOWLEDGE>
        {json.dumps(context['domain_knowledge'], indent=2)}
        </DOMAIN_KNOWLEDGE>

        Validate semantic correctness of security field relationships and correlations.

        Semantic Validation Areas:
        1. **FIELD_RELATES_TO Semantics**: Validate logical connections between related fields
        2. **Data Type Compatibility**: Ensure related fields have compatible data types
        3. **Domain Alignment**: Verify relationships align with security domain knowledge
        4. **Functional Coherence**: Validate relationships serve meaningful security purposes
        5. **Correlation Accuracy**: Ensure field correlations reflect real-world relationships

        Relationship Semantic Rules:
        - **Network Fields**: IP addresses, ports, protocols should relate appropriately
        - **User Fields**: User identifiers, roles, permissions should have logical connections
        - **System Fields**: Host information, OS details, processes should correlate correctly
        - **Event Fields**: Timestamps, event types, severity levels should relate meaningfully
        - **Threat Fields**: Indicators, signatures, classifications should align properly

        Validation Criteria:
        - **Logical Consistency**: Relationships should make sense from security monitoring perspective
        - **Data Type Alignment**: Related fields should have compatible or correlatable data types
        - **Domain Expertise**: Relationships should align with cybersecurity best practices
        - **Functional Value**: Relationships should provide value for security analysis
        - **Correlation Validity**: Field correlations should reflect observable patterns

        Semantic Violation Types:
        - **Illogical Relationships**: Fields that shouldn't be related according to domain knowledge
        - **Type Mismatches**: Related fields with incompatible data types
        - **Domain Conflicts**: Relationships that contradict security domain expertise
        - **Functional Irrelevance**: Relationships that provide no security analysis value
        - **Correlation Errors**: Claimed correlations that don't exist in practice

        Analysis Framework:
        1. **Domain Context Analysis**: Evaluate relationships within security domain context
        2. **Data Type Compatibility**: Check if related fields can be meaningfully correlated
        3. **Use Case Validation**: Verify relationships support actual security use cases
        4. **Expert Knowledge Alignment**: Compare relationships against cybersecurity expertise
        5. **Practical Utility**: Assess whether relationships provide actionable insights

        For each semantic issue, provide:
        - Specific relationship with semantic problems
        - Explanation of why the relationship is semantically invalid
        - Impact on security analysis and monitoring capabilities
        - Suggested correction or removal recommendation
        - Alternative relationships that would be more appropriate

        Output comprehensive semantic validation report with domain-expert recommendations.
        """,
        ),
    ]


def validate_field_integrity(context: dict[str, Any]) -> list[Message]:
    """Validate overall integrity of security field definitions and metadata"""
    return [
        Message(
            role='system',
            content='You are a helpful assistant that validates overall integrity of security field definitions, ensuring completeness, accuracy, and consistency of field metadata and properties.',
        ),
        Message(
            role='user',
            content=f"""
        <SECURITY_FIELDS>
        {json.dumps(context['security_fields'], indent=2)}
        </SECURITY_FIELDS>
        
        <FIELD_SCHEMA_REQUIREMENTS>
        {json.dumps(context['field_schema_requirements'], indent=2)}
        </FIELD_SCHEMA_REQUIREMENTS>
        
        <DATA_QUALITY_RULES>
        {json.dumps(context['data_quality_rules'], indent=2)}
        </DATA_QUALITY_RULES>
        
        <VALIDATION_STANDARDS>
        {json.dumps(context['validation_standards'], indent=2)}
        </VALIDATION_STANDARDS>

        Validate overall integrity of security field definitions and metadata.

        Field Integrity Components:
        1. **Metadata Completeness**: All required field properties are present and valid
        2. **Data Type Consistency**: Data types align with field content and examples
        3. **Documentation Quality**: Field descriptions are clear, accurate, and comprehensive
        4. **Example Validity**: Field examples match data type and validation rules
        5. **Naming Consistency**: Field names follow established conventions and standards

        Validation Dimensions:
        - **Required Properties**: Verify all mandatory field attributes are present
        - **Data Type Alignment**: Ensure data_type matches field content and examples
        - **Description Quality**: Assess clarity and completeness of field descriptions
        - **Example Accuracy**: Validate examples represent realistic field values
        - **Naming Standards**: Check adherence to field naming conventions
        - **Uniqueness**: Ensure field names are unique within their organizational scope

        Quality Metrics:
        - **Completeness Score**: Percentage of fields with all required metadata
        - **Accuracy Score**: Accuracy of data types, examples, and descriptions
        - **Consistency Score**: Consistency of naming and documentation patterns
        - **Usability Score**: How well fields support security analysis use cases
        - **Maintainability Score**: How easy fields are to understand and maintain

        Integrity Validation Rules:
        - **Name Requirements**: Field names must be non-empty, follow naming conventions
        - **Description Standards**: Descriptions must be at least 20 characters, describe purpose
        - **Data Type Validation**: Data types must be valid and match field content
        - **Example Consistency**: Examples must be valid instances of the declared data type
        - **UUID Validity**: All field UUIDs must be unique and properly formatted
        - **Cluster Assignment**: Every field must have valid cluster_partition_id

        Common Integrity Issues:
        - **Missing Descriptions**: Fields without adequate documentation
        - **Invalid Examples**: Examples that don't match declared data type
        - **Naming Violations**: Field names that violate established conventions
        - **Type Mismatches**: Data type declarations that don't match field content
        - **Duplicate Names**: Multiple fields with same name in same organizational scope
        - **Invalid UUIDs**: Malformed or duplicate field identifiers

        For each integrity issue, provide:
        - Specific field with integrity problems
        - Type of integrity violation (completeness, accuracy, consistency)
        - Impact on field usability and security analysis
        - Specific correction needed to fix the issue
        - Priority level for addressing the problem

        Output comprehensive field integrity validation report with prioritized fixes.
        """,
        ),
    ]


versions: Versions = {
    'validate_cluster_isolation': validate_cluster_isolation,
    'validate_hierarchy_consistency': validate_hierarchy_consistency,
    'validate_semantic_relationships': validate_semantic_relationships,
    'validate_field_integrity': validate_field_integrity,
}
