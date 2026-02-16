from __future__ import annotations

from typing import Any

from .common import (
    expect_bool,
    expect_dict,
    expect_int,
    expect_non_empty_str,
    expect_number,
    expect_str,
)

METRIC_KEYS = ('privacy_risk', 'simplicity', 'merge_conflict_risk', 'auditability')

_FILTERED_HISTORY_FIELDS = {
    'privacy_risk': {'base', 'block_penalty', 'ambiguous_penalty'},
    'simplicity': {'base', 'commit_divisor', 'commit_cap', 'ambiguous_penalty'},
    'merge_conflict_risk': {'base', 'commit_divisor', 'commit_cap', 'ambiguous_penalty'},
    'auditability': {'base', 'block_penalty', 'ambiguous_penalty'},
}

_CLEAN_FOUNDATION_FIELDS = {
    'privacy_risk': {'base'},
    'simplicity': {'base', 'commit_bonus_divisor', 'commit_bonus_cap'},
    'merge_conflict_risk': {'base'},
    'auditability': {'base'},
}


def validate_migration_sync_policy(
    payload: object,
    *,
    context: str = 'migration_sync_policy',
    strict: bool = False,
) -> dict[str, Any]:
    """Validate migration/sync policy schema.

    In strict mode, all first-class sections are required.
    In non-strict mode, sections are validated when present.
    """

    policy = expect_dict(payload, context=context)

    expect_int(policy.get('version'), context=f'{context}.version', min_value=1)

    for block_name in ('origin', 'upstream'):
        block = policy.get(block_name)
        if block is None:
            if strict:
                raise ValueError(f'{context}.{block_name} is required in strict mode')
            continue

        block_dict = expect_dict(block, context=f'{context}.{block_name}')
        expect_non_empty_str(block_dict.get('remote'), context=f'{context}.{block_name}.remote')
        expect_non_empty_str(block_dict.get('branch'), context=f'{context}.{block_name}.branch')
        if block_name == 'upstream' and 'url' in block_dict:
            expect_str(block_dict.get('url'), context=f'{context}.upstream.url')
        elif block_name == 'upstream' and strict:
            raise ValueError(f'{context}.upstream.url is required in strict mode')

    sync_policy = policy.get('sync_button_policy')
    if sync_policy is None:
        if strict:
            raise ValueError(f'{context}.sync_button_policy is required in strict mode')
    else:
        sync_policy_dict = expect_dict(sync_policy, context=f'{context}.sync_button_policy')
        expect_bool(
            sync_policy_dict.get('require_clean_worktree'),
            context=f'{context}.sync_button_policy.require_clean_worktree',
        )
        expect_int(
            sync_policy_dict.get('max_origin_only_commits'),
            context=f'{context}.sync_button_policy.max_origin_only_commits',
            min_value=0,
        )
        expect_bool(
            sync_policy_dict.get('require_upstream_only_commits'),
            context=f'{context}.sync_button_policy.require_upstream_only_commits',
        )

    scorecard = policy.get('scorecard')
    if scorecard is None:
        if strict:
            raise ValueError(f'{context}.scorecard is required in strict mode')
    else:
        scorecard_dict = expect_dict(scorecard, context=f'{context}.scorecard')
        expect_number(
            scorecard_dict.get('clean_foundation_threshold'),
            context=f'{context}.scorecard.clean_foundation_threshold',
            min_value=0,
        )
        weights = expect_dict(scorecard_dict.get('weights'), context=f'{context}.scorecard.weights')
        total_weight = 0.0
        for metric in METRIC_KEYS:
            total_weight += expect_number(
                weights.get(metric),
                context=f'{context}.scorecard.weights.{metric}',
                min_value=0,
            )
        if total_weight <= 0:
            raise ValueError(f'{context}.scorecard.weights must sum to > 0')

    schedule = policy.get('schedule')
    if schedule is None:
        if strict:
            raise ValueError(f'{context}.schedule is required in strict mode')
    else:
        schedule_dict = expect_dict(schedule, context=f'{context}.schedule')
        expect_non_empty_str(schedule_dict.get('timezone'), context=f'{context}.schedule.timezone')
        expect_non_empty_str(schedule_dict.get('weekly_day'), context=f'{context}.schedule.weekly_day')
        expect_non_empty_str(schedule_dict.get('cron_utc'), context=f'{context}.schedule.cron_utc')

    history_metrics = policy.get('history_metrics')
    if history_metrics is not None:
        history_metrics_dict = expect_dict(history_metrics, context=f'{context}.history_metrics')

        for candidate_name, allowed_fields in (
            ('filtered_history', _FILTERED_HISTORY_FIELDS),
            ('clean_foundation', _CLEAN_FOUNDATION_FIELDS),
        ):
            candidate_cfg = history_metrics_dict.get(candidate_name)
            if candidate_cfg is None:
                continue
            candidate_dict = expect_dict(
                candidate_cfg,
                context=f'{context}.history_metrics.{candidate_name}',
            )

            extra_metrics = set(candidate_dict) - set(allowed_fields)
            if extra_metrics:
                extras = ', '.join(sorted(extra_metrics))
                raise ValueError(
                    f'{context}.history_metrics.{candidate_name} has unsupported metrics: {extras}',
                )

            for metric_name, metric_cfg in candidate_dict.items():
                metric_dict = expect_dict(
                    metric_cfg,
                    context=f'{context}.history_metrics.{candidate_name}.{metric_name}',
                )
                allowed = allowed_fields[metric_name]
                extra_fields = set(metric_dict) - allowed
                if extra_fields:
                    extras = ', '.join(sorted(extra_fields))
                    raise ValueError(
                        f'{context}.history_metrics.{candidate_name}.{metric_name} '
                        f'has unsupported fields: {extras}',
                    )
                for key, value in metric_dict.items():
                    expect_number(
                        value,
                        context=f'{context}.history_metrics.{candidate_name}.{metric_name}.{key}',
                        min_value=0,
                    )

    return policy
