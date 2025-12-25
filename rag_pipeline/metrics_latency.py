"""Latency metrics aggregation."""

from typing import List, Dict, Any, Optional
import statistics


def compute_latency_stats(latencies: List[float]) -> Dict[str, float]:
    """Compute summary statistics for latency measurements."""
    if not latencies:
        return {
            "mean": 0.0, "median": 0.0, "std": 0.0,
            "min": 0.0, "max": 0.0, "p95": 0.0, "p99": 0.0, "count": 0,
        }

    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)

    return {
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "std": statistics.stdev(latencies) if n > 1 else 0.0,
        "min": min(latencies),
        "max": max(latencies),
        "p95": sorted_latencies[int(n * 0.95)] if n >= 20 else sorted_latencies[-1],
        "p99": sorted_latencies[int(n * 0.99)] if n >= 100 else sorted_latencies[-1],
        "count": n,
    }


def aggregate_latencies(
    results: List[Dict[str, Any]],
    latency_keys: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """Aggregate latency metrics from RAG query results."""
    if latency_keys is None:
        latency_keys = ["embed", "retrieve", "inference", "e2e"]

    aggregated = {}
    for key in latency_keys:
        values = []
        for result in results:
            lat = result.get("latencies", {})
            if key in lat and lat[key] is not None:
                values.append(lat[key])
        aggregated[key] = compute_latency_stats(values)

    return aggregated


def format_latency_report(
    aggregated: Dict[str, Dict[str, float]],
    config_name: str = "",
    as_milliseconds: bool = True
) -> str:
    """Format aggregated latencies as a report string."""
    multiplier = 1000 if as_milliseconds else 1
    unit = "ms" if as_milliseconds else "s"

    lines = []
    if config_name:
        lines.append(f"Latency Report: {config_name}")
    else:
        lines.append("Latency Report")

    lines.append("")
    lines.append(f"{'Metric':<15} {'Mean':>10} {'Median':>10} {'Std':>10} {'P95':>10} {'Count':>8}")
    lines.append("-" * 65)

    display_names = {
        "embed": "Embedding",
        "retrieve": "Retrieval",
        "inference": "Inference",
        "e2e": "End-to-End",
    }

    for key in ["embed", "retrieve", "inference", "e2e"]:
        if key not in aggregated:
            continue
        stats = aggregated[key]
        name = display_names.get(key, key)
        lines.append(
            f"{name:<15} "
            f"{stats['mean'] * multiplier:>9.2f}{unit} "
            f"{stats['median'] * multiplier:>9.2f}{unit} "
            f"{stats['std'] * multiplier:>9.2f}{unit} "
            f"{stats['p95'] * multiplier:>9.2f}{unit} "
            f"{stats['count']:>8}"
        )

    return "\n".join(lines)


def latency_stats_to_row(
    aggregated: Dict[str, Dict[str, float]],
    config_info: Dict[str, str],
    as_milliseconds: bool = True
) -> Dict[str, Any]:
    """Convert aggregated latencies to a flat dict for CSV."""
    multiplier = 1000 if as_milliseconds else 1
    row = dict(config_info)

    for lat_type in ["embed", "retrieve", "inference", "e2e"]:
        if lat_type not in aggregated:
            continue
        stats = aggregated[lat_type]
        prefix = f"{lat_type}_"
        row[f"{prefix}mean_ms"] = stats["mean"] * multiplier
        row[f"{prefix}median_ms"] = stats["median"] * multiplier
        row[f"{prefix}std_ms"] = stats["std"] * multiplier
        row[f"{prefix}p95_ms"] = stats["p95"] * multiplier
        row[f"{prefix}count"] = stats["count"]

    return row
