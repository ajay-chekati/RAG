"""Evaluation dataset loader for MultiHop-RAG."""

from typing import List, Dict, Any, Optional
from datasets import load_dataset


def load_multihop_queries(
    dataset_name: str = "yixuantt/MultiHopRAG",
    config_name: str = "MultiHopRAG",
    split: str = "train",
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Load the MultiHop-RAG query dataset and return structured evaluation items."""
    ds = load_dataset(dataset_name, config_name)
    data = ds[split]

    if limit:
        data = data.select(range(min(limit, len(data))))

    eval_items = []
    for idx, row in enumerate(data):
        evidence_list = row.get("evidence_list", [])
        gold_evidence_urls = {ev.get("url") for ev in evidence_list if ev.get("url")}

        eval_items.append({
            "id": f"q_{idx}",
            "question": row.get("query", ""),
            "gold_answer": row.get("answer", ""),
            "question_type": row.get("question_type", ""),
            "gold_evidence": evidence_list,
            "gold_evidence_urls": gold_evidence_urls,
            "num_hops": len(evidence_list),
        })

    return eval_items
