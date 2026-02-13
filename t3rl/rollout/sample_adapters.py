"""Shared sample adaptation helpers for rollout modules."""

from __future__ import annotations

from slime.utils.types import Sample

from t3rl.interaction.types import Status


def map_status(status: Status) -> Sample.Status:
    mapping = {
        Status.COMPLETED: Sample.Status.COMPLETED,
        Status.TRUNCATED: Sample.Status.TRUNCATED,
        Status.ABORTED: Sample.Status.ABORTED,
    }
    return mapping.get(status, Sample.Status.ABORTED)
