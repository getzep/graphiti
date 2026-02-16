from __future__ import annotations

from typing import Any

from .common import expect_dict, expect_int, expect_non_empty_str


def validate_delta_contract_policy(
    payload: object,
    *,
    context: str = 'delta_contract_policy',
) -> dict[str, Any]:
    """Validate delta contract policy metadata schema."""

    policy = expect_dict(payload, context=context)
    expect_int(policy.get('version'), context=f'{context}.version', min_value=1)

    targets = expect_dict(policy.get('targets'), context=f'{context}.targets')
    if not targets:
        raise ValueError(f'{context}.targets must not be empty')

    for target_name, target_cfg in targets.items():
        target = expect_dict(target_cfg, context=f'{context}.targets.{target_name}')
        expect_int(
            target.get('current_version'),
            context=f'{context}.targets.{target_name}.current_version',
            min_value=1,
        )
        expect_non_empty_str(
            target.get('migration_script'),
            context=f'{context}.targets.{target_name}.migration_script',
        )
        expect_non_empty_str(
            target.get('notes'),
            context=f'{context}.targets.{target_name}.notes',
        )

    return policy
