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


class SecurityFieldSummary(BaseModel):
    """Summary of security field extraction and relationships"""
    cluster_id: str = Field(..., description='Organizational cluster identifier')
    total_fields: int = Field(..., description='Total number of security fields processed')
    field_categories: dict[str, int] = Field(
        ...,
        description='Count of fields by category (network, system, user, etc.)',
    )
    relationship_counts: dict[str, int] = Field(
        ...,
        description='Count of relationships by type (BELONGS_TO, CONTAINS, FIELD_RELATES_TO)',
    )
    cluster_hierarchy: dict[str, Any] = Field(
        ...,
        description='Cluster and SubCluster organizational structure',
    )
    quality_metrics: dict[str, float] = Field(
        ...,
        description='Quality metrics for extraction process',
    )


class SecurityFieldInsights(BaseModel):
    """Insights derived from security field analysis"""
    coverage_analysis: dict[str, Any] = Field(
        ...,
        description='Analysis of security field coverage by domain',
    )
    relationship_patterns: dict[str, Any] = Field(
        ...,
        description='Common patterns in field relationships',
    )
    cluster_distribution: dict[str, Any] = Field(
        ...,
        description='Distribution of fields across organizational clusters',
    )
    recommendations: list[str] = Field(
        ...,
        description='Recommendations for improving field organization',
    )


class SecurityFieldSummarizationPrompt(Protocol):
    """Protocol for security field summarization prompts"""
    summarize_field_extraction: PromptVersion
    summarize_field_relationships: PromptVersion
    summarize_cluster_organization: PromptVersion
    analyze_field_coverage: PromptVersion


class Versions(TypedDict):
    """Type definitions for security field summarization prompt functions"""
    summarize_field_extraction: PromptFunction
    summarize_field_relationships: PromptFunction
    summarize_cluster_organization: PromptFunction
    analyze_field_coverage: PromptFunction


def summarize_field_extraction(context: dict[str, Any]) -> list[Message]:
    """Summarize security field extraction results across clusters"""
    return [
        Message(
            role='system',
            content='You are a helpful assistant that creates comprehensive summaries of security field extraction results, providing insights into organizational coverage and data quality.',
        ),
        Message(
            role='user',
            content=f"""
        <EXTRACTED SECURITY FIELDS>
        {json.dumps(context['extracted_security_fields'], indent=2)}
        </EXTRACTED SECURITY FIELDS>
        
        <FIELD RELATIONSHIPS>
        {json.dumps(context['field_relationships'], indent=2)}
        </FIELD RELATIONSHIPS>
        
        <CLUSTER HIERARCHY>
        {json.dumps(context['cluster_hierarchy'], indent=2)}
        </CLUSTER HIERARCHY>
        
        <EXTRACTION METADATA>
        {json.dumps(context['extraction_metadata'], indent=2)}
        </EXTRACTION_METADATA>

        Create a comprehensive summary of the security field extraction process.

        Summary Components:
        1. **Extraction Statistics**: Total fields, successful extractions, failed extractions
        2. **Field Categorization**: Group fields by security domain (network, system, application, user)
        3. **Cluster Distribution**: Number of fields per organizational cluster
        4. **Data Quality Metrics**: Completeness, accuracy, consistency scores
        5. **Relationship Coverage**: Distribution of BELONGS_TO, CONTAINS, FIELD_RELATES_TO relationships

        Analysis Areas:
        - **Field Completeness**: Percentage of fields with complete metadata (description, data_type, examples)
        - **Naming Consistency**: Analysis of field naming conventions across clusters
        - **Documentation Quality**: Assessment of field descriptions and examples quality
        - **Cluster Isolation**: Validation that organizational boundaries are properly maintained
        - **Hierarchical Structure**: Analysis of Field → SubCluster → Cluster relationships

        Quality Metrics:
        - **Extraction Success Rate**: Percentage of successful field extractions
        - **Relationship Accuracy**: Accuracy of field relationship assignments
        - **Cluster Compliance**: Adherence to organizational isolation rules
        - **Semantic Consistency**: Consistency of similar fields across clusters
        - **Documentation Coverage**: Percentage of fields with adequate documentation

        Provide actionable insights for improving field extraction and organization.
        """,
        ),
    ]


def summarize_field_relationships(context: dict[str, Any]) -> list[Message]:
    """Summarize security field relationship patterns and hierarchies"""
    return [
        Message(
            role='system',
            content='You are a helpful assistant that analyzes and summarizes security field relationship patterns, providing insights into organizational structure and field dependencies.',
        ),
        Message(
            role='user',
            content=f"""
        <FIELD RELATIONSHIPS>
        {json.dumps(context['field_relationships'], indent=2)}
        </FIELD_RELATIONSHIPS>
        
        <SECURITY FIELDS>
        {json.dumps(context['security_fields'], indent=2)}
        </SECURITY_FIELDS>
        
        <CLUSTER HIERARCHY>
        {json.dumps(context['cluster_hierarchy'], indent=2)}
        </CLUSTER_HIERARCHY>
        
        <RELATIONSHIP_VALIDATION_RESULTS>
        {json.dumps(context['relationship_validation'], indent=2)}
        </RELATIONSHIP_VALIDATION_RESULTS>

        Analyze and summarize security field relationship patterns.

        Relationship Analysis:
        1. **BELONGS_TO Relationships**: Field → SubCluster assignments and patterns
        2. **CONTAINS Relationships**: SubCluster → Cluster hierarchical structure
        3. **FIELD_RELATES_TO Relationships**: Inter-field dependencies and correlations
        4. **Cross-Cluster Patterns**: Common field types across different organizations
        5. **Hierarchical Consistency**: Validation of organizational structure integrity

        Pattern Recognition:
        - **Common Field Groups**: Frequently co-occurring security fields
        - **Cluster Specialization**: Unique field patterns per organizational cluster
        - **Relationship Density**: Average number of relationships per field
        - **Hierarchical Depth**: Analysis of SubCluster organization complexity
        - **Cross-References**: Fields that appear in multiple organizational contexts

        Organizational Insights:
        - **Cluster Autonomy**: How well each cluster maintains independent field definitions
        - **Standardization Opportunities**: Fields that could benefit from cross-cluster standardization
        - **Relationship Gaps**: Missing relationships that should exist based on field semantics
        - **Redundancy Analysis**: Overlapping field definitions across clusters
        - **Governance Compliance**: Adherence to organizational isolation requirements

        Output relationship summary with patterns, insights, and recommendations.
        """,
        ),
    ]


def summarize_cluster_organization(context: dict[str, Any]) -> list[Message]:
    """Summarize organizational cluster structure and field distribution"""
    return [
        Message(
            role='system',
            content='You are a helpful assistant that analyzes organizational cluster structure for security field management, providing insights into governance and field distribution patterns.',
        ),
        Message(
            role='user',
            content=f"""
        <CLUSTER_HIERARCHY>
        {json.dumps(context['cluster_hierarchy'], indent=2)}
        </CLUSTER_HIERARCHY>
        
        <CLUSTER_FIELD_DISTRIBUTION>
        {json.dumps(context['cluster_field_distribution'], indent=2)}
        </CLUSTER_FIELD_DISTRIBUTION>
        
        <ORGANIZATIONAL_RULES>
        {json.dumps(context['organizational_rules'], indent=2)}
        </ORGANIZATIONAL_RULES>
        
        <COMPLIANCE_METRICS>
        {json.dumps(context['compliance_metrics'], indent=2)}
        </COMPLIANCE_METRICS>

        Analyze and summarize organizational cluster structure for security field management.

        Cluster Analysis Components:
        1. **Organizational Structure**: Cluster → SubCluster → Field hierarchy analysis
        2. **Field Distribution**: How security fields are distributed across clusters
        3. **Governance Compliance**: Adherence to organizational isolation rules
        4. **Resource Allocation**: Field management workload across clusters
        5. **Standardization Assessment**: Opportunities for cross-cluster field standards

        Governance Metrics:
        - **Isolation Compliance**: Percentage of fields properly isolated to their clusters
        - **Hierarchy Integrity**: Validation of SubCluster → Cluster relationships
        - **Access Control**: Proper field visibility boundaries across organizations
        - **Data Sovereignty**: Each cluster maintains control over its field definitions
        - **Policy Adherence**: Compliance with organizational field management policies

        Distribution Analysis:
        - **Field Density**: Average number of fields per cluster and SubCluster
        - **Category Balance**: Distribution of field types (network, system, user) per cluster
        - **Complexity Metrics**: Organizational complexity indicators
        - **Growth Patterns**: How field collections grow within organizational boundaries
        - **Resource Requirements**: Estimated field management effort per cluster

        Strategic Insights:
        - **Cluster Maturity**: Assessment of each cluster's field management maturity
        - **Standardization Opportunities**: Fields that could benefit from common definitions
        - **Governance Gaps**: Areas where organizational policies need strengthening
        - **Scalability Analysis**: How well the structure supports organizational growth
        - **Operational Efficiency**: Recommendations for improving field management processes

        Provide comprehensive organizational analysis with actionable governance recommendations.
        """,
        ),
    ]


def analyze_field_coverage(context: dict[str, Any]) -> list[Message]:
    """Analyze security field coverage across domains and organizational clusters"""
    return [
        Message(
            role='system',
            content='You are a helpful assistant that analyzes security field coverage across different security domains and organizational clusters, identifying gaps and optimization opportunities.',
        ),
        Message(
            role='user',
            content=f"""
        <SECURITY_FIELD_INVENTORY>
        {json.dumps(context['security_field_inventory'], indent=2)}
        </SECURITY_FIELD_INVENTORY>
        
        <SECURITY_DOMAIN_TAXONOMY>
        {json.dumps(context['security_domain_taxonomy'], indent=2)}
        </SECURITY_DOMAIN_TAXONOMY>
        
        <CLUSTER_COVERAGE_MATRIX>
        {json.dumps(context['cluster_coverage_matrix'], indent=2)}
        </CLUSTER_COVERAGE_MATRIX>
        
        <INDUSTRY_BENCHMARKS>
        {json.dumps(context['industry_benchmarks'], indent=2)}
        </INDUSTRY_BENCHMARKS>

        Analyze security field coverage across domains and organizational clusters.

        Coverage Analysis Dimensions:
        1. **Security Domain Coverage**: Network, System, Application, User, Threat Intelligence
        2. **Organizational Coverage**: Field availability across different clusters
        3. **Functional Coverage**: Coverage of security monitoring, detection, response functions
        4. **Data Source Coverage**: Coverage of different security data sources and feeds
        5. **Compliance Coverage**: Coverage of regulatory and compliance requirements

        Analysis Framework:
        - **Domain Completeness**: Percentage coverage of essential security fields per domain
        - **Cross-Cluster Consistency**: Consistency of field coverage across organizations
        - **Gap Identification**: Critical security fields missing from cluster inventories
        - **Redundancy Assessment**: Overlapping field coverage that could be optimized
        - **Benchmark Comparison**: How field coverage compares to industry standards

        Coverage Metrics:
        - **Essential Field Coverage**: Coverage of fundamental security monitoring fields
        - **Advanced Field Coverage**: Coverage of sophisticated security analysis fields
        - **Compliance Field Coverage**: Coverage of regulatory requirement fields
        - **Integration Field Coverage**: Coverage of fields needed for tool integration
        - **Custom Field Coverage**: Coverage of organization-specific security fields

        Gap Analysis:
        - **Critical Gaps**: Missing fields that pose security monitoring risks
        - **Opportunity Gaps**: Fields that could enhance security capabilities
        - **Compliance Gaps**: Missing fields required for regulatory compliance
        - **Integration Gaps**: Missing fields needed for security tool integration
        - **Standardization Gaps**: Inconsistent field definitions across clusters

        Provide comprehensive coverage analysis with prioritized recommendations for field expansion.
        """,
        ),
    ]


versions: Versions = {
    'summarize_field_extraction': summarize_field_extraction,
    'summarize_field_relationships': summarize_field_relationships,
    'summarize_cluster_organization': summarize_cluster_organization,
    'analyze_field_coverage': analyze_field_coverage,
}
