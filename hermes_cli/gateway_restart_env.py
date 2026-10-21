"""Gateway restart / respawn env helpers for delegate-child fence hygiene.

A gateway is never a ``delegate_task`` child. An inherited
``HERMES_DELEGATED_CHILD_CONTEXT`` (or stale ``HERMES_KANBAN_TASK``) fences the
gateway's embedded dispatcher so every board write fails with PermissionError.
Host entry points scrub these markers at their own startup / spawn boundaries
(#136152 gateway scrub contract). In-process cron masking is intentionally
not used — see the two-process regression review on that PR.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def scrub_delegate_child_env_markers(environ) -> bool:
    """Strip the delegate-child marker and any stale kanban task id.

    Returns True when something was stripped. An inherited marker fences the
    gateway's embedded dispatcher: every board write would fail with
    PermissionError.
    """
    from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER

    stripped = False
    for key in (DELEGATED_CHILD_ENV_MARKER, "HERMES_KANBAN_TASK"):
        if environ.get(key):
            stripped = True
            environ.pop(key, None)
    if stripped:
        logger.warning(
            "gateway started with delegate-child env marker(s) "
            "(HERMES_DELEGATED_CHILD_CONTEXT / HERMES_KANBAN_TASK); "
            "stripping them — a gateway is never a delegate child"
        )
    return stripped
