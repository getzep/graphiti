"""
Helpers for normalizing edge temporal bounds.
"""

import logging
import re
from datetime import datetime

from graphiti_core.edges import EntityEdge
from graphiti_core.utils.datetime_utils import ensure_utc

logger = logging.getLogger(__name__)

_FUTURE_ASSERTION_RE = re.compile(
    r'\b('
    r'will|shall|is going to|are going to|going to|'
    r'plans? to|planned to|planning to|'
    r'intends? to|intended to|'
    r'expects? to|expected to|'
    r'promises? to|promised to|'
    r'scheduled to|is scheduled to|are scheduled to|'
    r'set to|aims? to|aimed to|committed to'
    r")\b|[a-z]+'ll\b",
    re.IGNORECASE,
)

_DEADLINE_RE = re.compile(
    r'\b(by|before|no later than|deadline|due by|due on)\b',
    re.IGNORECASE,
)

_ONGOING_STATE_END_RE = re.compile(
    r'\b(until|through|thru|expires?|expired|ends?|ended|ending|'
    r'stops?|stopped|ceases?|ceased|no longer)\b',
    re.IGNORECASE,
)


def normalize_future_temporal_bounds(
    fact: str,
    reference_time: datetime | None,
    valid_at: datetime | None,
    invalid_at: datetime | None,
) -> tuple[datetime | None, datetime | None]:
    """Normalize temporal bounds for future-tense assertions.

    For planned/promised/scheduled future facts, ``valid_at`` is the statement's
    reference time, not the future occurrence or deadline date. The future date
    remains part of the fact text or typed attributes. ``invalid_at`` is only for
    facts that stop being true; a soft deadline like "will complete by Friday"
    is not an invalidation time.
    """
    reference_time_utc = ensure_utc(reference_time)
    valid_at_utc = ensure_utc(valid_at)
    invalid_at_utc = ensure_utc(invalid_at)

    if reference_time_utc is None or not _FUTURE_ASSERTION_RE.search(fact):
        return valid_at_utc, invalid_at_utc

    if valid_at_utc is None or valid_at_utc > reference_time_utc:
        valid_at_utc = reference_time_utc

    if (
        invalid_at_utc is not None
        and invalid_at_utc > reference_time_utc
        and _DEADLINE_RE.search(fact)
        and not _ONGOING_STATE_END_RE.search(fact)
    ):
        invalid_at_utc = None

    return valid_at_utc, invalid_at_utc


def apply_extracted_timestamps(
    edge: EntityEdge,
    valid_at: str | None,
    invalid_at: str | None,
    reference_time: datetime | None,
) -> None:
    """Parse ISO timestamp strings onto an edge and normalize future bounds."""
    if valid_at:
        try:
            edge.valid_at = ensure_utc(datetime.fromisoformat(valid_at.replace('Z', '+00:00')))
        except ValueError:
            logger.debug('Error parsing valid_at: %s', valid_at)
    if invalid_at:
        try:
            edge.invalid_at = ensure_utc(datetime.fromisoformat(invalid_at.replace('Z', '+00:00')))
        except ValueError:
            logger.debug('Error parsing invalid_at: %s', invalid_at)
    edge.valid_at, edge.invalid_at = normalize_future_temporal_bounds(
        edge.fact,
        reference_time,
        edge.valid_at,
        edge.invalid_at,
    )
