import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any


def parse_tool_payload(result: Any) -> dict[str, Any]:
    def _extract_structured_result(payload: dict[str, Any]) -> dict[str, Any] | None:
        structured_result = payload.get('result', {}).get('structuredContent', {}).get('result')
        return structured_result if isinstance(structured_result, dict) else None

    def _normalize(payload: dict[str, Any]) -> dict[str, Any]:
        structured_result = _extract_structured_result(payload)
        if structured_result is not None:
            return structured_result
        if 'result' in payload and isinstance(payload['result'], dict):
            meaningful_keys = {
                'message',
                'status',
                'nodes',
                'facts',
                'episodes',
                'error',
                'episode_uuid',
                'state',
            }
            if not (meaningful_keys & payload.keys()):
                return payload['result']
        return payload

    if isinstance(result, dict):
        return _normalize(result)
    if hasattr(result, 'content') and result.content:
        first_content = result.content[0]
        text = getattr(first_content, 'text', None)
        if isinstance(text, str):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return {}
            return _normalize(parsed) if isinstance(parsed, dict) else {}
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            return {}
        return _normalize(parsed) if isinstance(parsed, dict) else {}
    return {}


def extract_episode_uuid(result: Any) -> str | None:
    payload = parse_tool_payload(result)
    episode_uuid = payload.get('episode_uuid')
    return episode_uuid if isinstance(episode_uuid, str) and episode_uuid else None


async def wait_for_ingest_completion(
    call_tool: Callable[[str, dict[str, Any]], Awaitable[Any]],
    *,
    episode_uuids: list[str],
    group_id: str | None,
    max_wait: int,
    poll_interval: float,
) -> bool:
    pending = [episode_uuid for episode_uuid in episode_uuids if episode_uuid]
    if not pending:
        return False

    started_at = time.time()
    while pending and (time.time() - started_at) < max_wait:
        completed_this_round: list[str] = []

        for episode_uuid in pending:
            arguments = {'episode_uuid': episode_uuid}
            if group_id is not None:
                arguments['group_id'] = group_id

            result = await call_tool('get_ingest_status', arguments)
            payload = parse_tool_payload(result)
            state = payload.get('state')

            if state == 'completed':
                completed_this_round.append(episode_uuid)
            elif state == 'failed':
                return False

        if completed_this_round:
            pending = [episode_uuid for episode_uuid in pending if episode_uuid not in completed_this_round]
            if not pending:
                return True

        await asyncio.sleep(poll_interval)

    return not pending
