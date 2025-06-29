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

import os
from datetime import datetime, timezone

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from graphiti_core.graphiti import Graphiti
from graphiti_core.helpers import validate_excluded_entity_types

pytestmark = pytest.mark.integration

pytest_plugins = ('pytest_asyncio',)

load_dotenv()

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')


# Test entity type definitions
class Person(BaseModel):
    """A human person mentioned in the conversation."""

    first_name: str | None = Field(None, description='First name of the person')
    last_name: str | None = Field(None, description='Last name of the person')
    occupation: str | None = Field(None, description='Job or profession of the person')


class Organization(BaseModel):
    """A company, institution, or organized group."""

    organization_type: str | None = Field(
        None, description='Type of organization (company, NGO, etc.)'
    )
    industry: str | None = Field(
        None, description='Industry or sector the organization operates in'
    )


class Location(BaseModel):
    """A geographic location, place, or address."""

    location_type: str | None = Field(
        None, description='Type of location (city, country, building, etc.)'
    )
    coordinates: str | None = Field(None, description='Geographic coordinates if available')


@pytest.mark.asyncio
async def test_exclude_default_entity_type():
    """Test excluding the default 'Entity' type while keeping custom types."""
    graphiti = Graphiti(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    try:
        await graphiti.build_indices_and_constraints()

        # Define entity types but exclude the default 'Entity' type
        entity_types = {
            'Person': Person,
            'Organization': Organization,
        }

        # Add an episode that would normally create both Entity and custom type entities
        episode_content = (
            'John Smith works at Acme Corporation in New York. The weather is nice today.'
        )

        result = await graphiti.add_episode(
            name='Business Meeting',
            episode_body=episode_content,
            source_description='Meeting notes',
            reference_time=datetime.now(timezone.utc),
            entity_types=entity_types,
            excluded_entity_types=['Entity'],  # Exclude default type
            group_id='test_exclude_default',
        )

        # Verify that nodes were created (custom types should still work)
        assert result is not None

        # Search for nodes to verify only custom types were created
        search_results = await graphiti.search_(
            query='John Smith Acme Corporation', group_ids=['test_exclude_default']
        )

        # Check that entities were created but with specific types, not default 'Entity'
        found_nodes = search_results.nodes
        for node in found_nodes:
            assert 'Entity' in node.labels  # All nodes should have Entity label
            # But they should also have specific type labels
            assert any(label in ['Person', 'Organization'] for label in node.labels), (
                f'Node {node.name} should have a specific type label, got: {node.labels}'
            )

        # Clean up
        await _cleanup_test_nodes(graphiti, 'test_exclude_default')

    finally:
        await graphiti.close()


@pytest.mark.asyncio
async def test_exclude_specific_custom_types():
    """Test excluding specific custom entity types while keeping others."""
    graphiti = Graphiti(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    try:
        await graphiti.build_indices_and_constraints()

        # Define multiple entity types
        entity_types = {
            'Person': Person,
            'Organization': Organization,
            'Location': Location,
        }

        # Add an episode with content that would create all types
        episode_content = (
            'Sarah Johnson from Google visited the San Francisco office to discuss the new project.'
        )

        result = await graphiti.add_episode(
            name='Office Visit',
            episode_body=episode_content,
            source_description='Visit report',
            reference_time=datetime.now(timezone.utc),
            entity_types=entity_types,
            excluded_entity_types=['Organization', 'Location'],  # Exclude these types
            group_id='test_exclude_custom',
        )

        assert result is not None

        # Search for nodes to verify only Person and Entity types were created
        search_results = await graphiti.search_(
            query='Sarah Johnson Google San Francisco', group_ids=['test_exclude_custom']
        )

        found_nodes = search_results.nodes

        # Should have Person and Entity type nodes, but no Organization or Location
        for node in found_nodes:
            assert 'Entity' in node.labels
            # Should not have excluded types
            assert 'Organization' not in node.labels, (
                f'Found excluded Organization in node: {node.name}'
            )
            assert 'Location' not in node.labels, f'Found excluded Location in node: {node.name}'

        # Should find at least one Person entity (Sarah Johnson)
        person_nodes = [n for n in found_nodes if 'Person' in n.labels]
        assert len(person_nodes) > 0, 'Should have found at least one Person entity'

        # Clean up
        await _cleanup_test_nodes(graphiti, 'test_exclude_custom')

    finally:
        await graphiti.close()


@pytest.mark.asyncio
async def test_exclude_all_types():
    """Test excluding all entity types (edge case)."""
    graphiti = Graphiti(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    try:
        await graphiti.build_indices_and_constraints()

        entity_types = {
            'Person': Person,
            'Organization': Organization,
        }

        # Exclude all types
        result = await graphiti.add_episode(
            name='No Entities',
            episode_body='This text mentions John and Microsoft but no entities should be created.',
            source_description='Test content',
            reference_time=datetime.now(timezone.utc),
            entity_types=entity_types,
            excluded_entity_types=['Entity', 'Person', 'Organization'],  # Exclude everything
            group_id='test_exclude_all',
        )

        assert result is not None

        # Search for nodes - should find very few or none from this episode
        search_results = await graphiti.search_(
            query='John Microsoft', group_ids=['test_exclude_all']
        )

        # There should be minimal to no entities created
        found_nodes = search_results.nodes
        assert len(found_nodes) == 0, (
            f'Expected no entities, but found: {[n.name for n in found_nodes]}'
        )

        # Clean up
        await _cleanup_test_nodes(graphiti, 'test_exclude_all')

    finally:
        await graphiti.close()


@pytest.mark.asyncio
async def test_exclude_no_types():
    """Test normal behavior when no types are excluded (baseline test)."""
    graphiti = Graphiti(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    try:
        await graphiti.build_indices_and_constraints()

        entity_types = {
            'Person': Person,
            'Organization': Organization,
        }

        # Don't exclude any types
        result = await graphiti.add_episode(
            name='Normal Behavior',
            episode_body='Alice Smith works at TechCorp.',
            source_description='Normal test',
            reference_time=datetime.now(timezone.utc),
            entity_types=entity_types,
            excluded_entity_types=None,  # No exclusions
            group_id='test_exclude_none',
        )

        assert result is not None

        # Search for nodes - should find entities of all types
        search_results = await graphiti.search_(
            query='Alice Smith TechCorp', group_ids=['test_exclude_none']
        )

        found_nodes = search_results.nodes
        assert len(found_nodes) > 0, 'Should have found some entities'

        # Should have both Person and Organization entities
        person_nodes = [n for n in found_nodes if 'Person' in n.labels]
        org_nodes = [n for n in found_nodes if 'Organization' in n.labels]

        assert len(person_nodes) > 0, 'Should have found Person entities'
        assert len(org_nodes) > 0, 'Should have found Organization entities'

        # Clean up
        await _cleanup_test_nodes(graphiti, 'test_exclude_none')

    finally:
        await graphiti.close()


def test_validation_valid_excluded_types():
    """Test validation function with valid excluded types."""
    entity_types = {
        'Person': Person,
        'Organization': Organization,
    }

    # Valid exclusions
    assert validate_excluded_entity_types(['Entity'], entity_types) is True
    assert validate_excluded_entity_types(['Person'], entity_types) is True
    assert validate_excluded_entity_types(['Entity', 'Person'], entity_types) is True
    assert validate_excluded_entity_types(None, entity_types) is True
    assert validate_excluded_entity_types([], entity_types) is True


def test_validation_invalid_excluded_types():
    """Test validation function with invalid excluded types."""
    entity_types = {
        'Person': Person,
        'Organization': Organization,
    }

    # Invalid exclusions should raise ValueError
    with pytest.raises(ValueError, match='Invalid excluded entity types'):
        validate_excluded_entity_types(['InvalidType'], entity_types)

    with pytest.raises(ValueError, match='Invalid excluded entity types'):
        validate_excluded_entity_types(['Person', 'NonExistentType'], entity_types)


@pytest.mark.asyncio
async def test_excluded_types_parameter_validation_in_add_episode():
    """Test that add_episode validates excluded_entity_types parameter."""
    graphiti = Graphiti(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    try:
        entity_types = {
            'Person': Person,
        }

        # Should raise ValueError for invalid excluded type
        with pytest.raises(ValueError, match='Invalid excluded entity types'):
            await graphiti.add_episode(
                name='Invalid Test',
                episode_body='Test content',
                source_description='Test',
                reference_time=datetime.now(timezone.utc),
                entity_types=entity_types,
                excluded_entity_types=['NonExistentType'],
                group_id='test_validation',
            )

    finally:
        await graphiti.close()


async def _cleanup_test_nodes(graphiti: Graphiti, group_id: str):
    """Helper function to clean up test nodes."""
    try:
        # Get all nodes for this group
        search_results = await graphiti.search_(query='*', group_ids=[group_id])

        # Delete all found nodes
        for node in search_results.nodes:
            await node.delete(graphiti.driver)

    except Exception as e:
        # Log but don't fail the test if cleanup fails
        print(f'Warning: Failed to clean up test nodes for group {group_id}: {e}')
