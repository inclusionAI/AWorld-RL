"""BFCL eval metric hooks."""

from __future__ import annotations

import logging

from t3rl.eval.common import log_eval_metric_dict

logger = logging.getLogger(__name__)


def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool:
    return log_eval_metric_dict(
        data,
        extra_metrics,
        metrics_key="bfcl_metrics",
        metric_keys=["progress", "accuracy", "format_reward"],
        fallback_metric_root="eval/bfcl",
        logger=logger,
        eval_function_hint="t3rl.rollout.bfcl.eval_rollout_with_metrics",
    )
