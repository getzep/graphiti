"""
Copyright 2025, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Dense vs Normal Episode Ingestion Example
-----------------------------------------
This example demonstrates how Graphiti handles different types of content:

1. Normal Content (prose, narrative, conversations):
   - Lower entity density (few entities per token)
   - Processed in a single LLM call
   - Examples: meeting transcripts, news articles, documentation

2. Dense Content (structured data with many entities):
   - High entity density (many entities per token)
   - Automatically chunked for reliable extraction
   - Examples: bulk data imports, cost reports, entity-dense JSON

The chunking behavior is controlled by environment variables:
- CHUNK_MIN_TOKENS: Minimum tokens before considering chunking (default: 1000)
- CHUNK_DENSITY_THRESHOLD: Entity density threshold (default: 0.15)
- CHUNK_TOKEN_SIZE: Target size per chunk (default: 3000)
- CHUNK_OVERLAP_TOKENS: Overlap between chunks (default: 200)
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from logging import INFO

from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

#################################################
# CONFIGURATION
#################################################

logging.basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

load_dotenv()

neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')

if not neo4j_uri or not neo4j_user or not neo4j_password:
    raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')


#################################################
# EXAMPLE DATA
#################################################

# Normal content: A meeting transcript (low entity density)
# This is prose/narrative content with few entities per token.
# It will NOT trigger chunking - processed in a single LLM call.
NORMAL_EPISODE_CONTENT = """
Meeting Notes - Q4 Planning Session

Alice opened the meeting by reviewing our progress on the mobile app redesign.
She mentioned that the user research phase went well and highlighted key findings
from the customer interviews conducted last month.

Bob then presented the engineering timeline. He explained that the backend API
refactoring is about 60% complete and should be finished by end of November.
The team has resolved most of the performance issues identified in the load tests.

Carol raised concerns about the holiday freeze period affecting our deployment
schedule. She suggested we move the beta launch to early December to give the
QA team enough time for regression testing before the code freeze.

David agreed with Carol's assessment and proposed allocating two additional
engineers from the platform team to help with the testing effort. He also
mentioned that the documentation needs to be updated before the release.

Action items:
- Alice will finalize the design specs by Friday
- Bob will coordinate with the platform team on resource allocation
- Carol will update the project timeline in Jira
- David will schedule a follow-up meeting for next Tuesday

The meeting concluded at 3:30 PM with agreement to reconvene next week.
"""

# Dense content: AWS cost data (high entity density)
# This is structured data with many entities per token.
# It WILL trigger chunking - processed in multiple LLM calls.
DENSE_EPISODE_CONTENT = {
    'report_type': 'AWS Cost Breakdown',
    'months': [
        {
            'period': '2025-01',
            'services': [
                {'name': 'Amazon S3', 'cost': 2487.97},
                {'name': 'Amazon RDS', 'cost': 1071.74},
                {'name': 'Amazon ECS', 'cost': 853.74},
                {'name': 'Amazon OpenSearch', 'cost': 389.74},
                {'name': 'AWS Secrets Manager', 'cost': 265.77},
                {'name': 'CloudWatch', 'cost': 232.34},
                {'name': 'Amazon VPC', 'cost': 238.39},
                {'name': 'EC2 Other', 'cost': 226.82},
                {'name': 'Amazon EC2 Compute', 'cost': 78.27},
                {'name': 'Amazon DocumentDB', 'cost': 65.40},
                {'name': 'Amazon ECR', 'cost': 29.00},
                {'name': 'Amazon ELB', 'cost': 37.53},
            ],
        },
        {
            'period': '2025-02',
            'services': [
                {'name': 'Amazon S3', 'cost': 2721.04},
                {'name': 'Amazon RDS', 'cost': 1035.77},
                {'name': 'Amazon ECS', 'cost': 779.49},
                {'name': 'Amazon OpenSearch', 'cost': 357.90},
                {'name': 'AWS Secrets Manager', 'cost': 268.57},
                {'name': 'CloudWatch', 'cost': 224.57},
                {'name': 'Amazon VPC', 'cost': 215.15},
                {'name': 'EC2 Other', 'cost': 213.86},
                {'name': 'Amazon EC2 Compute', 'cost': 70.70},
                {'name': 'Amazon DocumentDB', 'cost': 59.07},
                {'name': 'Amazon ECR', 'cost': 33.92},
                {'name': 'Amazon ELB', 'cost': 33.89},
            ],
        },
        {
            'period': '2025-03',
            'services': [
                {'name': 'Amazon S3', 'cost': 2952.31},
                {'name': 'Amazon RDS', 'cost': 1198.79},
                {'name': 'Amazon ECS', 'cost': 869.78},
                {'name': 'Amazon OpenSearch', 'cost': 389.75},
                {'name': 'AWS Secrets Manager', 'cost': 271.33},
                {'name': 'CloudWatch', 'cost': 233.00},
                {'name': 'Amazon VPC', 'cost': 238.31},
                {'name': 'EC2 Other', 'cost': 227.78},
                {'name': 'Amazon EC2 Compute', 'cost': 78.21},
                {'name': 'Amazon DocumentDB', 'cost': 65.40},
                {'name': 'Amazon ECR', 'cost': 33.75},
                {'name': 'Amazon ELB', 'cost': 37.54},
            ],
        },
        {
            'period': '2025-04',
            'services': [
                {'name': 'Amazon S3', 'cost': 3189.62},
                {'name': 'Amazon RDS', 'cost': 1102.30},
                {'name': 'Amazon ECS', 'cost': 848.19},
                {'name': 'Amazon OpenSearch', 'cost': 379.14},
                {'name': 'AWS Secrets Manager', 'cost': 270.89},
                {'name': 'CloudWatch', 'cost': 230.64},
                {'name': 'Amazon VPC', 'cost': 230.54},
                {'name': 'EC2 Other', 'cost': 220.18},
                {'name': 'Amazon EC2 Compute', 'cost': 75.70},
                {'name': 'Amazon DocumentDB', 'cost': 63.29},
                {'name': 'Amazon ECR', 'cost': 35.21},
                {'name': 'Amazon ELB', 'cost': 36.30},
            ],
        },
        {
            'period': '2025-05',
            'services': [
                {'name': 'Amazon S3', 'cost': 3423.07},
                {'name': 'Amazon RDS', 'cost': 1014.50},
                {'name': 'Amazon ECS', 'cost': 874.75},
                {'name': 'Amazon OpenSearch', 'cost': 389.71},
                {'name': 'AWS Secrets Manager', 'cost': 274.91},
                {'name': 'CloudWatch', 'cost': 233.28},
                {'name': 'Amazon VPC', 'cost': 238.53},
                {'name': 'EC2 Other', 'cost': 227.27},
                {'name': 'Amazon EC2 Compute', 'cost': 78.27},
                {'name': 'Amazon DocumentDB', 'cost': 65.40},
                {'name': 'Amazon ECR', 'cost': 37.42},
                {'name': 'Amazon ELB', 'cost': 37.52},
            ],
        },
        {
            'period': '2025-06',
            'services': [
                {'name': 'Amazon S3', 'cost': 3658.14},
                {'name': 'Amazon RDS', 'cost': 963.60},
                {'name': 'Amazon ECS', 'cost': 942.45},
                {'name': 'Amazon OpenSearch', 'cost': 379.06},
                {'name': 'AWS Secrets Manager', 'cost': 282.41},
                {'name': 'CloudWatch', 'cost': 230.69},
                {'name': 'Amazon VPC', 'cost': 230.64},
                {'name': 'EC2 Other', 'cost': 220.11},
                {'name': 'Amazon EC2 Compute', 'cost': 75.74},
                {'name': 'Amazon DocumentDB', 'cost': 63.30},
                {'name': 'Amazon ECR', 'cost': 39.49},
                {'name': 'Amazon ELB', 'cost': 36.30},
            ],
        },
    ],
}

# Message content: A conversation (moderate entity density)
# This demonstrates message-type episodes with speaker patterns.
MESSAGE_EPISODE_CONTENT = """User: I need to set up a new development environment for the analytics project.
Assistant: I can help with that. Which cloud provider are you using - AWS, GCP, or Azure?
User: We're using AWS. The project requires PostgreSQL, Redis, and Elasticsearch.
Assistant: Great choice. For PostgreSQL, I recommend Amazon RDS with the db.r5.large instance type. For Redis, you can use Amazon ElastiCache. For Elasticsearch, Amazon OpenSearch Service would be ideal.
User: What about the compute layer? We need to run Python data pipelines.
Assistant: For Python data pipelines, I suggest Amazon ECS with Fargate for serverless containers, or AWS Lambda for smaller, event-driven tasks. If you need more control, EC2 instances with Auto Scaling would work well.
User: Let's go with ECS Fargate. Can you also recommend a CI/CD setup?
Assistant: For CI/CD with ECS Fargate, I recommend AWS CodePipeline integrated with CodeBuild for building container images, and CodeDeploy for blue-green deployments. You can store your container images in Amazon ECR.
"""


async def main():
    graphiti = Graphiti(neo4j_uri, neo4j_user, neo4j_password)

    try:
        #################################################
        # EXAMPLE 1: Normal Content (No Chunking)
        #################################################
        # This prose content has low entity density.
        # Graphiti will process it in a single LLM call.
        #################################################

        print('=' * 60)
        print('EXAMPLE 1: Normal Content (Meeting Transcript)')
        print('=' * 60)
        print(f'Content length: {len(NORMAL_EPISODE_CONTENT)} characters')
        print(f'Estimated tokens: ~{len(NORMAL_EPISODE_CONTENT) // 4}')
        print('Expected behavior: Single LLM call (no chunking)')
        print()

        await graphiti.add_episode(
            name='Q4 Planning Meeting',
            episode_body=NORMAL_EPISODE_CONTENT,
            source=EpisodeType.text,
            source_description='Meeting transcript',
            reference_time=datetime.now(timezone.utc),
        )
        print('Successfully added normal episode\n')

        #################################################
        # EXAMPLE 2: Dense Content (Chunking Triggered)
        #################################################
        # This structured data has high entity density.
        # Graphiti will automatically chunk it for
        # reliable extraction across multiple LLM calls.
        #################################################

        print('=' * 60)
        print('EXAMPLE 2: Dense Content (AWS Cost Report)')
        print('=' * 60)
        dense_json = json.dumps(DENSE_EPISODE_CONTENT)
        print(f'Content length: {len(dense_json)} characters')
        print(f'Estimated tokens: ~{len(dense_json) // 4}')
        print('Expected behavior: Multiple LLM calls (chunking enabled)')
        print()

        await graphiti.add_episode(
            name='AWS Cost Report 2025 H1',
            episode_body=dense_json,
            source=EpisodeType.json,
            source_description='AWS cost breakdown by service',
            reference_time=datetime.now(timezone.utc),
        )
        print('Successfully added dense episode\n')

        #################################################
        # EXAMPLE 3: Message Content
        #################################################
        # Conversation content with speaker patterns.
        # Chunking preserves message boundaries.
        #################################################

        print('=' * 60)
        print('EXAMPLE 3: Message Content (Conversation)')
        print('=' * 60)
        print(f'Content length: {len(MESSAGE_EPISODE_CONTENT)} characters')
        print(f'Estimated tokens: ~{len(MESSAGE_EPISODE_CONTENT) // 4}')
        print('Expected behavior: Depends on density threshold')
        print()

        await graphiti.add_episode(
            name='Dev Environment Setup Chat',
            episode_body=MESSAGE_EPISODE_CONTENT,
            source=EpisodeType.message,
            source_description='Support conversation',
            reference_time=datetime.now(timezone.utc),
        )
        print('Successfully added message episode\n')

        #################################################
        # SEARCH RESULTS
        #################################################

        print('=' * 60)
        print('SEARCH: Verifying extracted entities')
        print('=' * 60)

        # Search for entities from normal content
        print("\nSearching for: 'Q4 planning meeting participants'")
        results = await graphiti.search('Q4 planning meeting participants')
        print(f'Found {len(results)} results')
        for r in results[:3]:
            print(f'  - {r.fact}')

        # Search for entities from dense content
        print("\nSearching for: 'AWS S3 costs'")
        results = await graphiti.search('AWS S3 costs')
        print(f'Found {len(results)} results')
        for r in results[:3]:
            print(f'  - {r.fact}')

        # Search for entities from message content
        print("\nSearching for: 'ECS Fargate recommendations'")
        results = await graphiti.search('ECS Fargate recommendations')
        print(f'Found {len(results)} results')
        for r in results[:3]:
            print(f'  - {r.fact}')

    finally:
        await graphiti.close()
        print('\nConnection closed')


if __name__ == '__main__':
    asyncio.run(main())
