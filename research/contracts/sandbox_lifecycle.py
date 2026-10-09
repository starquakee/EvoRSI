"""A cancelled status acknowledges a request, not completed process cleanup."""
from __future__ import annotations

import json
from typing import Any


def cleanup_confirmed(body: Any) -> bool:
    if not isinstance(body, dict) or body.get('status') not in ('completed', 'failed', 'cancelled'):
        return False
    if not isinstance(body.get('completed_at'), str) or not body['completed_at']:
        return False
    result = body.get('result')
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (ValueError, TypeError):
            return False
    if not isinstance(result, dict):
        return False
    return result.get('worker_cleanup_verified') is True or (
        result.get('execution_never_started') is True and body.get('started_at') is None
    )
