"""Common metric aggregation helpers shared by rollout/eval hooks."""

from __future__ import annotations

from typing import Any


def _flatten_samples(samples) -> list[Any]:
    flat_samples: list[Any] = []
    for item in samples:
        if isinstance(item, list):
            flat_samples.extend(item)
        else:
            flat_samples.append(item)
    return flat_samples


def log_rollout_metric_dict(
    samples,
    rollout_extra_metrics,
    *,
    metrics_key: str,
    metric_keys: list[str],
) -> bool:
    flat_samples = _flatten_samples(samples)

    grouped_metrics: dict[str, dict[str, float]] = {}
    grouped_counts: dict[str, int] = {}

    for sample in flat_samples:
        metadata = getattr(sample, "metadata", None)
        if not isinstance(metadata, dict):
            continue

        metrics = metadata.get(metrics_key)
        if not isinstance(metrics, dict):
            continue

        data_source = metadata.get("data_source", "unknown")
        if data_source not in grouped_metrics:
            grouped_metrics[data_source] = {key: 0.0 for key in metric_keys}
            grouped_counts[data_source] = 0

        for key in metric_keys:
            if key in metrics:
                grouped_metrics[data_source][key] += float(metrics[key])
        grouped_counts[data_source] += 1

    if rollout_extra_metrics is None:
        return False

    total_metrics = {key: 0.0 for key in metric_keys}
    total_count = 0

    for data_source, metrics_sum in grouped_metrics.items():
        count = grouped_counts[data_source]
        if count == 0:
            continue

        for key, value in metrics_sum.items():
            rollout_extra_metrics[f"rollout/{data_source}/{key}"] = value / count
            total_metrics[key] += value

        total_count += count

    if total_count > 0:
        for key, value in total_metrics.items():
            rollout_extra_metrics[f"rollout/average/{key}"] = value / total_count

    return False


def log_eval_metric_dict(
    data,
    extra_metrics,
    *,
    metrics_key: str,
    metric_keys: list[str],
    fallback_metric_root: str,
    logger,
    eval_function_hint: str,
) -> bool:
    if extra_metrics is None:
        logger.warning(
            "log_eval_rollout_data: extra_metrics is None, eval metrics are skipped. "
            "Set --eval-function-path %s to guarantee metric injection.",
            eval_function_hint,
        )
        return False

    metrics_sink: dict[str, float] = extra_metrics
    grouped_metrics: dict[str, dict[str, float]] = {}
    grouped_counts: dict[str, int] = {}

    for dataset_name, dataset_data in data.items():
        samples = dataset_data.get("samples", [])
        for sample in samples:
            metadata = getattr(sample, "metadata", None)
            if not isinstance(metadata, dict):
                continue

            metrics = metadata.get(metrics_key)
            if not isinstance(metrics, dict):
                continue

            data_source = metadata.get("data_source") or dataset_name or "unknown"
            if data_source not in grouped_metrics:
                grouped_metrics[data_source] = {key: 0.0 for key in metric_keys}
                grouped_counts[data_source] = 0

            for key in metric_keys:
                if key in metrics:
                    grouped_metrics[data_source][key] += float(metrics[key])
            grouped_counts[data_source] += 1

    dataset_names = list(data.keys())
    if len(dataset_names) == 1:
        metric_root = f"eval/{dataset_names[0]}"
    else:
        metric_root = fallback_metric_root
        logger.warning(
            "log_eval_rollout_data: multiple eval datasets detected (%s), fallback metric root to '%s'",
            dataset_names,
            metric_root,
        )

    total_metrics = {key: 0.0 for key in metric_keys}
    total_count = 0

    for data_source, metrics_sum in grouped_metrics.items():
        count = grouped_counts[data_source]
        if count == 0:
            continue

        for key, value in metrics_sum.items():
            metrics_sink[f"{metric_root}/{data_source}/{key}"] = value / count
            total_metrics[key] += value

        total_count += count

    if total_count > 0:
        for key, value in total_metrics.items():
            metrics_sink[f"{metric_root}/average/{key}"] = value / total_count
    else:
        logger.warning(
            "log_eval_rollout_data: no %s found in eval samples; skipping eval metric logging",
            metrics_key,
        )

    return False
