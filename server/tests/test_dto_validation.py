"""Unit tests for request DTO validation (no database or network required)."""

import pytest
from pydantic import ValidationError

from graph_service.dto import AddEntityNodeRequest, AddMessagesRequest


def test_add_entity_node_request_rejects_unsafe_group_id():
    # The write path must reject group_ids the read/delete path would refuse to match,
    # so a bad group_id surfaces as a 422 (ValidationError) rather than creating an
    # unreachable record.
    with pytest.raises(ValidationError):
        AddEntityNodeRequest(uuid='u', group_id='bad"group', name='n')


def test_add_entity_node_request_accepts_valid_group_id():
    request = AddEntityNodeRequest(uuid='u', group_id='valid-group_1', name='n')
    assert request.group_id == 'valid-group_1'


def test_add_messages_request_rejects_unsafe_group_id():
    with pytest.raises(ValidationError):
        AddMessagesRequest(group_id='bad"group', messages=[])


def test_add_messages_request_accepts_valid_group_id():
    request = AddMessagesRequest(group_id='valid-group_1', messages=[])
    assert request.group_id == 'valid-group_1'
