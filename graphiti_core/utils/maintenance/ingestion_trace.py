"""
ingestion trace helpers that explain why a stage made each decision.

The helper in this module is storage-neutral. It writes trace records only through
GraphitiClients.ingestion_trace when an application provides an implementation.
"""

import json
import logging
import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Annotated, Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, create_model

from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.ingestion_trace import IngestionTraceRecord
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.prompts.models import Message

logger = logging.getLogger(__name__)

TOutput = TypeVar('TOutput', bound=BaseModel)

_TRACE_OMIT_KEYS = frozenset({'name_embedding', 'fact_embedding'})
_TRACE_SCHEMA_SKIP_KEYS = frozenset({'node_episode_index_map'})
_SAFE_IDENT = re.compile(r'[^0-9A-Za-z_]')
_STAGE_EXPLANATION_INSTRUCTIONS = {
    'combined_extraction': (
        'For each extracted node or edge, identify the episode span that caused the extraction. '
        'Use a short quote or precise paraphrase, then explain why that span produced the node or '
        'edge. Do not use the output name or fact as the entire explanation.'
    ),
    'resolve_nodes': (
        'For each node decision, identify the extracted entity, the candidate entities it was '
        'compared with or that no candidates existed, and the new-or-merge decision. Cite the '
        'episode evidence and candidate content that support the decision. Do not use a node name '
        'as the entire explanation.'
    ),
    'resolve_edges': (
        'For each edge decision, identify the extracted edge, the candidate edges it was compared '
        'with or that no candidates existed, and the new, merge, invalidate, or expire decision. '
        'Cite the episode evidence and candidate content that support the decision. Do not use a '
        'fact as the entire explanation.'
    ),
}


class IdentifiedDecisionExplanation(BaseModel):
    model_config = ConfigDict(extra='forbid')

    uuid: str = Field(..., description='UUID of the original object this explanation refers to')
    explanation: str = Field(
        ...,
        description=(
            'Episode-grounded explanation of the decision, including relevant comparison '
            'evidence when the stage uses candidates'
        ),
    )


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False)


def _strip_omitted_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_omitted_fields(item)
            for key, item in value.items()
            if key not in _TRACE_OMIT_KEYS
        }
    if isinstance(value, list):
        return [_strip_omitted_fields(item) for item in value]
    return value


def _trace_model_name(output_model: type[BaseModel], stage: str) -> str:
    suffix = re.sub(r'[^0-9a-zA-Z_]', '_', stage).strip('_') or 'Stage'
    return f'{output_model.__name__}{suffix}IngestionTrace'


def _schema_name(path: str) -> str:
    name = _SAFE_IDENT.sub('_', path).strip('_') or 'Field'
    if name[0].isdigit():
        name = f'F_{name}'
    return name[:60]


def _field_ident(key: str, used: set[str]) -> str:
    ident = _SAFE_IDENT.sub('_', key).strip('_') or 'field'
    if ident[0].isdigit():
        ident = f'id_{ident}'
    candidate = ident
    suffix = 2
    while candidate in used:
        candidate = f'{ident}_{suffix}'
        suffix += 1
    used.add(candidate)
    return candidate


def _skip_trace_schema_key(key: str) -> bool:
    return key in _TRACE_OMIT_KEYS or key in _TRACE_SCHEMA_SKIP_KEYS


def _stage_explanation_instruction(stage: str) -> str:
    return _STAGE_EXPLANATION_INSTRUCTIONS.get(
        stage,
        (
            'Ground each explanation in the episode and stage input context. Identify the '
            'evidence and decision instead of restating an output value.'
        ),
    )


def _has_uuid_field(value: BaseModel) -> bool:
    return 'uuid' in type(value).model_fields


def _type_for(value: Any, path: str) -> Any:
    if isinstance(value, BaseModel):
        if _has_uuid_field(value):
            return IdentifiedDecisionExplanation
        return _model_for_container(value, path)
    if isinstance(value, dict):
        return _model_for_mapping(value, path)
    if isinstance(value, list):
        return _type_for_sequence(value, path)
    if isinstance(value, tuple):
        item_types = tuple(_type_for(item, f'{path}_{index}') for index, item in enumerate(value))
        if not item_types:
            return tuple[()]
        return tuple[item_types]  # type: ignore[misc,valid-type]
    return str


def _type_for_sequence(value: list[Any], path: str) -> Any:
    count = len(value)
    if count == 0:
        item_type: Any = IdentifiedDecisionExplanation
    elif isinstance(value[0], tuple):
        item_types = tuple(
            _type_for(item, f'{path}_item_{index}') for index, item in enumerate(value[0])
        )
        item_type = tuple[item_types] if item_types else tuple[()]  # type: ignore[misc,valid-type]
    else:
        item_type = _type_for(value[0], f'{path}_item')
    return Annotated[list[item_type], Field(min_length=count, max_length=count)]


def _model_for_mapping(value: dict[Any, Any], path: str) -> type[BaseModel]:
    used_names: set[str] = set()
    field_defs: dict[str, Any] = {}
    for key, child in value.items():
        alias = str(key)
        if _skip_trace_schema_key(alias):
            continue
        ident = _field_ident(alias, used_names)
        field_defs[ident] = (
            _type_for(child, f'{path}_{ident}'),
            Field(..., alias=alias, description=f'Explanation for `{alias}`'),
        )
    return create_model(
        _schema_name(f'{path}_Map'),
        __config__=ConfigDict(extra='forbid', populate_by_name=True),
        **field_defs,
    )


def _model_for_container(value: BaseModel, path: str) -> type[BaseModel]:
    field_defs: dict[str, Any] = {}
    for field_name in type(value).model_fields:
        if _skip_trace_schema_key(field_name):
            continue
        field_defs[field_name] = (
            _type_for(getattr(value, field_name), f'{path}_{field_name}'),
            Field(..., description=f'Explanation for `{field_name}`'),
        )
    return create_model(
        _schema_name(path),
        __config__=ConfigDict(extra='forbid'),
        **field_defs,
    )


def build_trace_model(
    output: BaseModel,
    stage: str,
    output_model: type[BaseModel] | None = None,
) -> type[BaseModel]:
    """Build a structured-output model whose keys match this stage output."""
    model_type = output_model or type(output)
    stage_instruction = _stage_explanation_instruction(stage)
    inner_model = _type_for(output, f'{model_type.__name__}_{stage}_Explanations')
    if not isinstance(inner_model, type) or not issubclass(inner_model, BaseModel):
        raise TypeError('decision explanation schema must be a Pydantic model')
    return create_model(
        _trace_model_name(model_type, stage),
        __config__=ConfigDict(extra='forbid'),
        decision_explanations=(
            inner_model,
            Field(
                ...,
                description=(
                    'Explanations for each decision in the original stage output. '
                    'Use the schema keys. For objects with a uuid, fill uuid and '
                    'explanation. Never name an explanation field summary. '
                    f'{stage_instruction}'
                ),
            ),
        ),
    )


def _dump_decision_explanations(traced: BaseModel) -> dict[str, Any]:
    dumped = traced.model_dump(mode='json', by_alias=True).get('decision_explanations')
    if not isinstance(dumped, dict):
        raise TypeError('decision_explanations must dump to a JSON object')
    return dumped


def merge_decision_explanations(original: Any, explanations: Any) -> Any:
    """Copy stage output and attach each available decision explanation inline."""
    if isinstance(original, dict):
        merged = deepcopy(original)
        if not isinstance(explanations, dict):
            return merged

        original_uuid = original.get('uuid')
        explanation_uuid = explanations.get('uuid')
        explanation = explanations.get('explanation')
        if (
            original_uuid is not None
            and explanation_uuid == original_uuid
            and isinstance(explanation, str)
        ):
            merged['explanation'] = explanation

        for key, value in original.items():
            if key not in explanations or key in {'uuid', 'explanation'}:
                continue
            merged[key] = merge_decision_explanations(value, explanations[key])
        return merged

    if isinstance(original, list):
        if not isinstance(explanations, list):
            return deepcopy(original)

        explanations_by_uuid = {
            item['uuid']: item
            for item in explanations
            if isinstance(item, dict) and isinstance(item.get('uuid'), str)
        }
        merged_items = []
        for index, item in enumerate(original):
            if isinstance(item, dict) and isinstance(item.get('uuid'), str):
                item_explanation = explanations_by_uuid.get(item['uuid'])
            elif index < len(explanations):
                item_explanation = explanations[index]
            else:
                item_explanation = None
            merged_items.append(merge_decision_explanations(item, item_explanation))
        return merged_items

    if isinstance(explanations, str):
        return {'value': deepcopy(original), 'explanation': explanations}

    return deepcopy(original)


def _model_for_size(clients: GraphitiClients, model_size: ModelSize) -> str | None:
    if model_size == ModelSize.small:
        return clients.llm_client.small_model or clients.llm_client.model
    return clients.llm_client.model


def _safe_model_for_size(clients: GraphitiClients, model_size: ModelSize) -> str | None:
    try:
        return _model_for_size(clients, model_size)
    except Exception:
        logger.warning('Failed to read ingestion trace model metadata', exc_info=True)
        return None


def _build_trace_messages(
    *,
    stage: str,
    prompt_name: str,
    input_context: dict[str, Any],
    original_output: dict[str, Any],
) -> list[Message]:
    stage_instruction = _stage_explanation_instruction(stage)
    return [
        Message(
            role='system',
            content=(
                'You are a careful analyst for a knowledge-graph ingestion pipeline. '
                'Explain why the previous LLM stage made each decision. Use the stage '
                'prompt name, the episode text, previous episodes, and any dedupe '
                'candidates in the input context. Do not correct, rewrite, add, or '
                'remove pipeline output. Follow the response schema exactly. '
                f'{stage_instruction}'
            ),
        ),
        Message(
            role='user',
            content=f"""
<STAGE>
{stage}
</STAGE>

<PROMPT_NAME>
{prompt_name}
</PROMPT_NAME>

<INPUT_CONTEXT>
{_safe_json(input_context)}
</INPUT_CONTEXT>

<ORIGINAL_OUTPUT>
{_safe_json(original_output)}
</ORIGINAL_OUTPUT>

Fill `decision_explanations` using the structured output schema.

Requirements:
- Use only the schema keys. The schema is derived from ORIGINAL_OUTPUT.
- {stage_instruction}
- For objects that have a `uuid`, copy that uuid and put the why-text in
  `explanation`. Do not emit domain fields such as `summary`, `name`, `labels`,
  or `fact`. A node's `summary` in ORIGINAL_OUTPUT is evidence to reason about,
  not a place to put your explanation.
- For maps of identifiers (for example `uuid_map`), use the original keys from
  the schema and put an explanation string on each key.
- Ground each explanation in INPUT_CONTEXT and the named stage prompt. Cite
  episode wording or a candidate when that evidence is what justified the
  decision.
- Do not copy original values, correct them, or propose a better extraction.
""",
        ),
    ]


async def _write_trace(clients: GraphitiClients, record: IngestionTraceRecord) -> None:
    if clients.ingestion_trace is None:
        return
    try:
        await clients.ingestion_trace.write_trace(record)
    except Exception:
        logger.warning(
            'Failed to write ingestion trace',
            exc_info=True,
            extra={
                'trace_id': record.trace_id,
                'workflow_run_id': record.workflow_run_id,
                'stage': record.stage,
            },
        )


async def trace_step_output(
    clients: GraphitiClients,
    *,
    output: TOutput,
    output_model: type[TOutput],
    stage: str,
    prompt_name: str,
    input_context: dict[str, Any],
    episode_uuids: list[str],
    account_uuid: str | None = None,
    project_uuid: str | None = None,
    graph_uuid: str | None = None,
    workflow_run_id: str | None = None,
    retry_count: int = 3,
    model_size: ModelSize = ModelSize.medium,
    group_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> TOutput:
    """Explain LLM stage decisions without changing the pipeline output.

    The original output is always returned. Trace writes are best-effort.
    `retry_count` is the number of retries after the initial attempt.
    """

    trace_model = build_trace_model(output, stage, output_model=output_model)
    original_output = _strip_omitted_fields(output.model_dump(mode='json'))
    sanitized_input_context = _strip_omitted_fields(input_context)
    attempts = max(1, retry_count + 1)
    last_error: Exception | None = None
    prompt = f'ingestion_trace.{prompt_name}'
    trace_id = str(uuid4())
    model = _safe_model_for_size(clients, model_size)

    for attempt in range(1, attempts + 1):
        try:
            messages = _build_trace_messages(
                stage=stage,
                prompt_name=prompt_name,
                input_context=sanitized_input_context,
                original_output=original_output,
            )
            llm_response = await clients.llm_client.generate_response(
                messages,
                response_model=trace_model,
                model_size=model_size,
                group_id=group_id,
                prompt_name=prompt,
            )
            traced = trace_model.model_validate(llm_response)
            decision_explanations = _dump_decision_explanations(traced)
            displayed_output = merge_decision_explanations(original_output, decision_explanations)

            await _write_trace(
                clients,
                IngestionTraceRecord(
                    trace_id=trace_id,
                    workflow_run_id=workflow_run_id,
                    stage=stage,
                    prompt_name=prompt_name,
                    account_uuid=account_uuid,
                    project_uuid=project_uuid,
                    graph_uuid=graph_uuid,
                    episode_uuids=episode_uuids,
                    model=model,
                    model_size=model_size.value,
                    validation_status='recorded',
                    attempt_count=attempt,
                    input_context=sanitized_input_context,
                    original_output=displayed_output,
                    decision_explanations=decision_explanations,
                    metadata=metadata or {},
                    created_at=datetime.now(timezone.utc),
                ),
            )
            return output
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                logger.warning(
                    'ingestion trace attempt failed; retrying',
                    exc_info=True,
                    extra={
                        'workflow_run_id': workflow_run_id,
                        'stage': stage,
                        'attempt': attempt,
                        'max_attempts': attempts,
                    },
                )

    logger.error(
        'ingestion trace failed after retries; using original output',
        extra={
            'workflow_run_id': workflow_run_id,
            'stage': stage,
            'attempts': attempts,
            'error': str(last_error) if last_error else None,
        },
    )
    await _write_trace(
        clients,
        IngestionTraceRecord(
            trace_id=trace_id,
            workflow_run_id=workflow_run_id,
            stage=stage,
            prompt_name=prompt_name,
            account_uuid=account_uuid,
            project_uuid=project_uuid,
            graph_uuid=graph_uuid,
            episode_uuids=episode_uuids,
            model=model,
            model_size=model_size.value,
            validation_status='failed',
            attempt_count=attempts,
            input_context=sanitized_input_context,
            original_output=original_output,
            decision_explanations={},
            error=str(last_error) if last_error else None,
            metadata=metadata or {},
            created_at=datetime.now(timezone.utc),
        ),
    )
    return output
