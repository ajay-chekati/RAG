"""Retrieval quality metrics: Hit@k, Recall@k, MRR@k, NDCG@k."""

from typing import List, Dict, Any, Set, Optional
import math
from collections import defaultdict
import statistics


def hit_at_k(gold_doc_ids: Set[str], retrieved_doc_ids: List[str], k: int) -> float:
    """1 if any gold document is in top-k, 0 otherwise."""
    if not gold_doc_ids:
        return 0.0
    top_k = set(retrieved_doc_ids[:k])
    return 1.0 if bool(gold_doc_ids & top_k) else 0.0


def recall_at_k(gold_doc_ids: Set[str], retrieved_doc_ids: List[str], k: int) -> float:
    """Fraction of gold documents found in top-k."""
    if not gold_doc_ids:
        return 0.0
    top_k = set(retrieved_doc_ids[:k])
    num_found = len(gold_doc_ids & top_k)
    return num_found / len(gold_doc_ids)


def precision_at_k(gold_doc_ids: Set[str], retrieved_doc_ids: List[str], k: int) -> float:
    """Fraction of top-k that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    num_relevant = sum(1 for doc_id in top_k if doc_id in gold_doc_ids)
    return num_relevant / k


def mrr_at_k(gold_doc_ids: Set[str], retrieved_doc_ids: List[str], k: int) -> float:
    """Reciprocal rank of first relevant document in top-k."""
    if not gold_doc_ids:
        return 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in gold_doc_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(gold_doc_ids: Set[str], retrieved_doc_ids: List[str], k: int) -> float:
    """Normalized discounted cumulative gain at k (binary relevance)."""
    if not gold_doc_ids:
        return 0.0

    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in gold_doc_ids:
            dcg += 1.0 / math.log2(rank + 1)

    num_relevant = min(len(gold_doc_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, num_relevant + 1))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_retrieval_metrics(
    gold_doc_ids: Set[str],
    retrieved_doc_ids: List[str],
    k_values: Optional[List[int]] = None
) -> Dict[str, float]:
    """Compute all retrieval metrics for a single query at various k values."""
    if k_values is None:
        k_values = [1, 3, 5, 10]

    metrics = {}
    for k in k_values:
        metrics[f"hit@{k}"] = hit_at_k(gold_doc_ids, retrieved_doc_ids, k)
        metrics[f"recall@{k}"] = recall_at_k(gold_doc_ids, retrieved_doc_ids, k)
        metrics[f"precision@{k}"] = precision_at_k(gold_doc_ids, retrieved_doc_ids, k)
        metrics[f"mrr@{k}"] = mrr_at_k(gold_doc_ids, retrieved_doc_ids, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(gold_doc_ids, retrieved_doc_ids, k)

    return metrics


def aggregate_retrieval_metrics(all_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Aggregate retrieval metrics across multiple queries."""
    if not all_metrics:
        return {}

    metric_values = defaultdict(list)
    for metrics in all_metrics:
        for name, value in metrics.items():
            metric_values[name].append(value)

    aggregated = {}
    for name, values in metric_values.items():
        n = len(values)
        aggregated[name] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if n > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "count": n,
        }

    return aggregated


def format_retrieval_report(aggregated: Dict[str, Dict[str, float]], config_name: str = "") -> str:
    """Format aggregated retrieval metrics as a report string."""
    lines = []
    if config_name:
        lines.append(f"Retrieval Metrics: {config_name}")
    else:
        lines.append("Retrieval Metrics")

    lines.append("")
    lines.append(f"{'Metric':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Count':>8}")
    lines.append("-" * 65)

    metric_order = []
    k_values = set()
    for name in aggregated.keys():
        parts = name.split("@")
        if len(parts) == 2:
            k_values.add(int(parts[1]))

    for k in sorted(k_values):
        for metric_type in ["hit", "recall", "precision", "mrr", "ndcg"]:
            metric_name = f"{metric_type}@{k}"
            if metric_name in aggregated:
                metric_order.append(metric_name)

    for name in metric_order:
        stats = aggregated[name]
        lines.append(
            f"{name:<15} "
            f"{stats['mean']:>10.4f} "
            f"{stats['std']:>10.4f} "
            f"{stats['min']:>10.4f} "
            f"{stats['max']:>10.4f} "
            f"{stats['count']:>8}"
        )

    return "\n".join(lines)


def retrieval_stats_to_row(
    aggregated: Dict[str, Dict[str, float]],
    config_info: Dict[str, str]
) -> Dict[str, Any]:
    """Convert aggregated retrieval metrics to a flat dict for CSV."""
    row = dict(config_info)
    for name, stats in aggregated.items():
        row[f"{name}_mean"] = stats["mean"]
        row[f"{name}_std"] = stats["std"]
    return row
