"""
Answer quality metrics: Exact Match, Token F1, LLM-as-judge.

Includes optional caching for LLM judge results to avoid redundant API calls.
"""

from typing import List, Dict, Any, Optional
import re
import string
import json
from collections import Counter, defaultdict
import statistics

# Import cache utilities (optional dependency)
try:
    from rag_pipeline.simple_cache import SimpleCache, make_judge_key
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False


def normalize_answer(text: str) -> str:
    """Normalize text: lowercase, remove punctuation/articles, strip whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = ' '.join(text.split())
    return text


def get_tokens(text: str) -> List[str]:
    """Tokenize normalized text into words."""
    return normalize_answer(text).split()


def exact_match(gold_answer: str, model_answer: str) -> float:
    """1 if normalized answers are identical, 0 otherwise."""
    return 1.0 if normalize_answer(gold_answer) == normalize_answer(model_answer) else 0.0


def compute_token_f1(gold_answer: str, model_answer: str) -> Dict[str, float]:
    """Compute token-level precision, recall, and F1."""
    gold_tokens = get_tokens(gold_answer)
    model_tokens = get_tokens(model_answer)

    if not gold_tokens and not model_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    if not gold_tokens or not model_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    gold_counts = Counter(gold_tokens)
    model_counts = Counter(model_tokens)
    common = sum((gold_counts & model_counts).values())

    precision = common / len(model_tokens)
    recall = common / len(gold_tokens)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def create_judge_prompt(
    question: str,
    gold_answer: str,
    model_answer: str,
    include_reasoning_check: bool = True
) -> str:
    """Create prompt for LLM-as-judge evaluation."""
    reasoning_instruction = ""
    if include_reasoning_check:
        reasoning_instruction = """
- reasoning_quality: Rate 0-1 how well the answer demonstrates multi-step reasoning
"""

    return f"""You are evaluating a question-answering system. Compare the model's answer to the ground truth.

Question:
{question}

Ground Truth Answer:
{gold_answer}

Model Answer:
{model_answer}

Evaluate the model answer and respond with ONLY a valid JSON object (no other text):
{{
    "correctness": <float 0-1, how correct is the answer>,
    "completeness": <float 0-1, does it fully answer the question>,{reasoning_instruction}
    "label": "<'correct', 'partially_correct', or 'incorrect'>",
    "reason": "<brief explanation>"
}}"""


def parse_judge_response(response: str) -> Dict[str, Any]:
    """Parse JSON response from judge LLM."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return {
        "correctness": 0.0,
        "completeness": 0.0,
        "label": "error",
        "reason": f"Failed to parse: {response[:200]}",
        "parse_error": True,
    }


def judge_answer_with_llm(
    question: str,
    gold_answer: str,
    model_answer: str,
    llm_client,
    include_reasoning_check: bool = True
) -> Dict[str, Any]:
    """Use LLM to judge answer quality (no caching)."""
    from langchain_core.messages import HumanMessage

    prompt = create_judge_prompt(
        question=question,
        gold_answer=gold_answer,
        model_answer=model_answer or "(No answer provided)",
        include_reasoning_check=include_reasoning_check
    )

    response = llm_client.invoke([HumanMessage(content=prompt)])
    response_text = response.content if hasattr(response, 'content') else str(response)

    return parse_judge_response(response_text)


def judge_answer_with_cache(
    question: str,
    gold_answer: str,
    model_answer: str,
    llm_client,
    judge_model: str,
    cache: Optional["SimpleCache"] = None,
    include_reasoning_check: bool = True
) -> tuple:
    """
    Use LLM to judge answer quality with optional caching.
    
    Args:
        question: The original question
        gold_answer: Ground truth answer
        model_answer: Model's generated answer
        llm_client: LangChain LLM client
        judge_model: Name of the judge model (for cache key)
        cache: Optional SimpleCache for judge results
        include_reasoning_check: Whether to ask about reasoning quality
        
    Returns:
        Tuple of (judge_result_dict, was_cached: bool)
    """
    # Check cache first
    if cache is not None and CACHE_AVAILABLE:
        key = make_judge_key(judge_model, question, gold_answer, model_answer or "")
        cached = cache.get(key)
        if cached is not None:
            return cached, True
    
    # Call LLM (cache miss or no cache)
    result = judge_answer_with_llm(
        question=question,
        gold_answer=gold_answer,
        model_answer=model_answer,
        llm_client=llm_client,
        include_reasoning_check=include_reasoning_check
    )
    
    # Store in cache if available
    if cache is not None and CACHE_AVAILABLE:
        cache.set(key, result)
    
    return result, False


def compute_answer_metrics(
    gold_answer: str,
    model_answer: str,
    question: Optional[str] = None,
    llm_client=None,
    use_llm_judge: bool = True,
    judge_model: Optional[str] = None,
    judge_cache: Optional["SimpleCache"] = None
) -> Dict[str, Any]:
    """
    Compute all answer quality metrics for a single query.
    
    Args:
        gold_answer: Ground truth answer
        model_answer: Model's generated answer
        question: The original question (needed for LLM judge)
        llm_client: LangChain LLM client for judge
        use_llm_judge: Whether to use LLM-as-judge
        judge_model: Name of judge model (for cache key)
        judge_cache: Optional SimpleCache for judge results
        
    Returns:
        Dict with all computed metrics
    """
    metrics = {}

    # Exact match
    metrics["exact_match"] = exact_match(gold_answer, model_answer or "")

    # Token F1
    f1_scores = compute_token_f1(gold_answer, model_answer or "")
    metrics["token_precision"] = f1_scores["precision"]
    metrics["token_recall"] = f1_scores["recall"]
    metrics["token_f1"] = f1_scores["f1"]

    # LLM Judge (with optional caching)
    if use_llm_judge and llm_client and question:
        if judge_cache is not None and judge_model:
            # Use cached version
            judge_result, was_cached = judge_answer_with_cache(
                question=question,
                gold_answer=gold_answer,
                model_answer=model_answer or "",
                llm_client=llm_client,
                judge_model=judge_model,
                cache=judge_cache
            )
            metrics["judge_cached"] = was_cached
        else:
            # No caching
            judge_result = judge_answer_with_llm(
                question=question,
                gold_answer=gold_answer,
                model_answer=model_answer or "",
                llm_client=llm_client
            )
            metrics["judge_cached"] = False
        
        metrics["judge_correctness"] = judge_result.get("correctness", 0.0)
        metrics["judge_completeness"] = judge_result.get("completeness", 0.0)
        metrics["judge_reasoning"] = judge_result.get("reasoning_quality", 0.0)
        metrics["judge_label"] = judge_result.get("label", "unknown")
        metrics["judge_reason"] = judge_result.get("reason", "")
        metrics["judge_parse_error"] = judge_result.get("parse_error", False)

    return metrics


def aggregate_answer_metrics(all_metrics: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Aggregate answer metrics across multiple queries."""
    if not all_metrics:
        return {}

    numeric_keys = [
        "exact_match", "token_precision", "token_recall", "token_f1",
        "judge_correctness", "judge_completeness", "judge_reasoning"
    ]

    metric_values = defaultdict(list)
    for metrics in all_metrics:
        for key in numeric_keys:
            if key in metrics and metrics[key] is not None:
                try:
                    metric_values[key].append(float(metrics[key]))
                except (ValueError, TypeError):
                    pass

    aggregated = {}
    for name, values in metric_values.items():
        if not values:
            continue
        n = len(values)
        aggregated[name] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if n > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "count": n,
        }

    label_counts = defaultdict(int)
    for metrics in all_metrics:
        label = metrics.get("judge_label")
        if label:
            label_counts[label] += 1

    if label_counts:
        total = sum(label_counts.values())
        aggregated["judge_label_distribution"] = {
            "correct": label_counts.get("correct", 0) / total,
            "partially_correct": label_counts.get("partially_correct", 0) / total,
            "incorrect": label_counts.get("incorrect", 0) / total,
            "error": label_counts.get("error", 0) / total,
            "count": total,
        }

    return aggregated


def format_answer_report(aggregated: Dict[str, Dict[str, float]], config_name: str = "") -> str:
    """Format aggregated answer metrics as a report string."""
    lines = []
    if config_name:
        lines.append(f"Answer Quality Metrics: {config_name}")
    else:
        lines.append("Answer Quality Metrics")

    lines.append("")
    lines.append(f"{'Metric':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Count':>8}")
    lines.append("-" * 70)

    metric_order = [
        "exact_match", "token_precision", "token_recall", "token_f1",
        "judge_correctness", "judge_completeness", "judge_reasoning"
    ]

    display_names = {
        "exact_match": "Exact Match",
        "token_precision": "Token Precision",
        "token_recall": "Token Recall",
        "token_f1": "Token F1",
        "judge_correctness": "Judge Correct.",
        "judge_completeness": "Judge Complete.",
        "judge_reasoning": "Judge Reasoning",
    }

    for name in metric_order:
        if name not in aggregated:
            continue
        stats = aggregated[name]
        display = display_names.get(name, name)
        lines.append(
            f"{display:<20} "
            f"{stats['mean']:>10.4f} "
            f"{stats['std']:>10.4f} "
            f"{stats['min']:>10.4f} "
            f"{stats['max']:>10.4f} "
            f"{stats['count']:>8}"
        )

    if "judge_label_distribution" in aggregated:
        dist = aggregated["judge_label_distribution"]
        lines.append("")
        lines.append("Judge Label Distribution:")
        for label in ["correct", "partially_correct", "incorrect", "error"]:
            if label in dist:
                lines.append(f"  {label}: {dist[label]*100:.1f}%")

    return "\n".join(lines)


def answer_stats_to_row(
    aggregated: Dict[str, Dict[str, float]],
    config_info: Dict[str, str]
) -> Dict[str, Any]:
    """Convert aggregated answer metrics to a flat dict for CSV."""
    row = dict(config_info)

    for name, stats in aggregated.items():
        if name == "judge_label_distribution":
            for label, pct in stats.items():
                if label != "count":
                    row[f"judge_{label}_pct"] = pct
        elif isinstance(stats, dict) and "mean" in stats:
            row[f"{name}_mean"] = stats["mean"]
            row[f"{name}_std"] = stats["std"]

    return row
