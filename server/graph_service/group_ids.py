from __future__ import annotations

import hashlib
from typing import Literal

Scope = Literal['user', 'workspace', 'session']


def resolve_group_id(scope: Scope, key: str, prefix: str = 'graphiti') -> str:
    if not key:
        raise ValueError('key must not be empty')

    digest = hashlib.sha256(key.encode('utf-8')).hexdigest()[:32]
    return f'{prefix}_{scope}_{digest}'
