"""RAG evaluation experiment runner with caching."""

import json
import argparse
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from langchain_openai import ChatOpenAI

from rag_pipeline import config
from rag_pipeline.query_handler import run_rag_once
from rag_pipeline.eval_dataset import load_multihop_queries
from rag_pipeline.metrics_latency import aggregate_latencies, format_latency_report, latency_stats_to_row
from rag_pipeline.metrics_retrieval import compute_retrieval_metrics, aggregate_retrieval_metrics, format_retrieval_report, retrieval_stats_to_row
from rag_pipeline.metrics_answer import compute_answer_metrics, aggregate_answer_metrics, format_answer_report, answer_stats_to_row
from rag_pipeline.simple_cache import CacheManager


DEFAULT_CONFIGURATIONS = [
    ("chroma", "nomic-embed-text", None),
]

K_VALUES = [1, 3, 5, 10]

# Default cache directory
DEFAULT_CACHE_DIR = "eval_cache"


def evaluate_single_query(
    eval_item: Dict[str, Any],
    backend_name: str,
    embedding_model: str,
    persist_dir: Optional[str],
    llm_client: Optional[ChatOpenAI],
    k: int = 10,
    generate_answer: bool = True,
    use_llm_judge: bool = True,
    cache_manager: Optional[CacheManager] = None,
) -> Dict[str, Any]:
    """Evaluate a single query and compute all metrics."""
    question = eval_item["question"]
    gold_answer = eval_item["gold_answer"]
    gold_evidence_urls = eval_item["gold_evidence_urls"]

    # Get caches from manager (or None)
    embedding_cache = cache_manager.embeddings if cache_manager else None
    answer_cache = cache_manager.answers if cache_manager else None
    judge_cache = cache_manager.judge if cache_manager else None

    # Run RAG pipeline with caching
    rag_result = run_rag_once(
        question=question,
        backend_name=backend_name,
        embedding_model=embedding_model,
        persist_dir=persist_dir,
        k=k,
        generate_answer=generate_answer,
        embedding_cache=embedding_cache,
        answer_cache=answer_cache,
    )

    retrieved_doc_ids = [
        meta.get("doc_id") or meta.get("source", "")
        for meta in rag_result["retrieved_metadata"]
    ]

    # Compute retrieval metrics
    retrieval_metrics = compute_retrieval_metrics(
        gold_doc_ids=gold_evidence_urls,
        retrieved_doc_ids=retrieved_doc_ids,
        k_values=K_VALUES
    )

    # Compute answer metrics (with judge caching)
    answer_metrics = compute_answer_metrics(
        gold_answer=gold_answer,
        model_answer=rag_result["answer"],
        question=question,
        llm_client=llm_client if use_llm_judge else None,
        use_llm_judge=use_llm_judge,
        judge_model=config.LLM_MODEL_NAME if use_llm_judge else None,
        judge_cache=judge_cache if use_llm_judge else None,
    )

    return {
        "query_id": eval_item["id"],
        "question": question,
        "question_type": eval_item.get("question_type", ""),
        "num_hops": eval_item.get("num_hops", 0),
        "gold_answer": gold_answer,
        "model_answer": rag_result["answer"],
        "retrieved_doc_ids": retrieved_doc_ids,
        "gold_evidence_urls": list(gold_evidence_urls),
        "latencies": rag_result["latencies"],
        "retrieval_metrics": retrieval_metrics,
        "answer_metrics": answer_metrics,
        "cache_status": rag_result.get("cache_status", {}),
    }


def evaluate_configuration(
    eval_items: List[Dict[str, Any]],
    backend_name: str,
    embedding_model: str,
    persist_dir: Optional[str] = None,
    k: int = 10,
    generate_answer: bool = True,
    use_llm_judge: bool = True,
    progress_interval: int = 50,
    cache_manager: Optional[CacheManager] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Evaluate a configuration on all eval items."""
    config_name = f"{backend_name}_{embedding_model}"
    print(f"\nEvaluating: {config_name}")
    
    if cache_manager:
        cache_manager.reset_all_stats()  # Reset stats for this configuration

    llm_client = None
    if generate_answer or use_llm_judge:
        llm_client = ChatOpenAI(
            base_url=config.EMBEDDING_API_BASE,
            api_key=config.EMBEDDING_API_KEY,
            model=config.LLM_MODEL_NAME,
            temperature=0
        )

    per_query_results = []

    for i, eval_item in enumerate(tqdm(eval_items, desc=f"Evaluating {config_name}")):
        try:
            result = evaluate_single_query(
                eval_item=eval_item,
                backend_name=backend_name,
                embedding_model=embedding_model,
                persist_dir=persist_dir,
                llm_client=llm_client,
                k=k,
                generate_answer=generate_answer,
                use_llm_judge=use_llm_judge,
                cache_manager=cache_manager,
            )
            per_query_results.append(result)
        except Exception as e:
            print(f"\nError on query {eval_item['id']}: {e}")
            per_query_results.append({
                "query_id": eval_item["id"],
                "error": str(e),
            })

        if (i + 1) % progress_interval == 0:
            completed = [r for r in per_query_results if "error" not in r]
            if completed:
                interim_latencies = aggregate_latencies(completed)
                interim_retrieval = aggregate_retrieval_metrics([r["retrieval_metrics"] for r in completed])
                interim_answer = aggregate_answer_metrics([r["answer_metrics"] for r in completed])

                print(f"\n--- Interim ({i+1} queries) ---")
                print(f"Hit@5: {interim_retrieval.get('hit@5', {}).get('mean', 0):.3f}")
                print(f"Recall@5: {interim_retrieval.get('recall@5', {}).get('mean', 0):.3f}")
                print(f"Token F1: {interim_answer.get('token_f1', {}).get('mean', 0):.3f}")
                print(f"E2E: {interim_latencies.get('e2e', {}).get('mean', 0)*1000:.1f}ms")
                
                # Print cache stats
                if cache_manager:
                    stats = cache_manager.all_stats()
                    print(f"Cache hits - Embed: {stats['embeddings']['hits']}, "
                          f"Answer: {stats['answers']['hits']}, "
                          f"Judge: {stats['judge']['hits']}")

    completed_results = [r for r in per_query_results if "error" not in r]

    # Collect cache statistics
    cache_stats = {}
    if cache_manager:
        cache_stats = cache_manager.all_stats()

    aggregated = {
        "config": {
            "backend": backend_name,
            "embedding_model": embedding_model,
            "persist_dir": persist_dir,
            "k": k,
            "generate_answer": generate_answer,
            "use_llm_judge": use_llm_judge,
            "caching_enabled": cache_manager is not None,
        },
        "num_queries": len(eval_items),
        "num_completed": len(completed_results),
        "num_errors": len(per_query_results) - len(completed_results),
        "latency_stats": aggregate_latencies(completed_results),
        "retrieval_stats": aggregate_retrieval_metrics([r["retrieval_metrics"] for r in completed_results]),
        "answer_stats": aggregate_answer_metrics([r["answer_metrics"] for r in completed_results]),
        "cache_stats": cache_stats,
    }

    # Print reports
    print(f"\n{format_latency_report(aggregated['latency_stats'], config_name)}")
    print(f"\n{format_retrieval_report(aggregated['retrieval_stats'], config_name)}")
    print(f"\n{format_answer_report(aggregated['answer_stats'], config_name)}")
    
    # Print cache report
    if cache_manager:
        cache_manager.print_stats()

    return per_query_results, aggregated


def run_experiments(
    configurations: Optional[List[Tuple[str, str, Optional[str]]]] = None,
    eval_limit: Optional[int] = None,
    k: int = 10,
    generate_answer: bool = True,
    use_llm_judge: bool = True,
    output_dir: str = "eval_results",
    enable_cache: bool = True,
    cache_dir: Optional[str] = None,
    persist_cache: bool = True,
) -> Dict[str, Any]:
    """Run evaluation experiments across configurations."""
    if configurations is None:
        configurations = DEFAULT_CONFIGURATIONS

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("RAG Evaluation with Caching")
    print("=" * 60)
    print(f"Configurations: {len(configurations)}")
    print(f"Output: {output_path}")
    print(f"Caching: {'Enabled' if enable_cache else 'Disabled'}")
    
    # Initialize cache manager
    cache_manager = None
    if enable_cache:
        cache_dir = cache_dir or DEFAULT_CACHE_DIR
        cache_manager = CacheManager(cache_dir=cache_dir if persist_cache else None)
        
        if persist_cache:
            load_results = cache_manager.load_all(cache_dir)
            loaded_count = sum(1 for v in load_results.values() if v)
            if loaded_count > 0:
                print(f"Loaded {loaded_count} cache(s) from {cache_dir}")
                # Show loaded cache sizes
                for name, cache in [("embeddings", cache_manager.embeddings), 
                                    ("answers", cache_manager.answers),
                                    ("judge", cache_manager.judge)]:
                    if cache.size() > 0:
                        print(f"  {name}: {cache.size()} items")

    print("\nLoading evaluation dataset...")
    eval_items = load_multihop_queries(limit=eval_limit)
    print(f"Loaded {len(eval_items)} queries")

    all_results = {}
    all_summaries = []

    for backend_name, embedding_model, persist_dir in configurations:
        config_name = f"{backend_name}_{embedding_model.replace(':', '_').replace('/', '_')}"

        per_query, aggregated = evaluate_configuration(
            eval_items=eval_items,
            backend_name=backend_name,
            embedding_model=embedding_model,
            persist_dir=persist_dir,
            k=k,
            generate_answer=generate_answer,
            use_llm_judge=use_llm_judge,
            cache_manager=cache_manager,
        )

        all_results[config_name] = {
            "per_query": per_query,
            "aggregated": aggregated,
        }
        all_summaries.append(aggregated)

        per_query_path = output_path / f"{config_name}_{timestamp}_per_query.jsonl"
        with open(per_query_path, 'w') as f:
            for result in per_query:
                f.write(json.dumps(result, default=str) + "\n")
        print(f"Saved: {per_query_path}")

    # Save summary
    summary_path = output_path / f"summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\nSaved summary: {summary_path}")

    # Save comparison CSV
    comparison = generate_comparison_table(all_summaries)
    comparison_path = output_path / f"comparison_{timestamp}.csv"
    save_comparison_csv(comparison, comparison_path)
    print(f"Saved comparison: {comparison_path}")

    # Persist caches to disk
    if cache_manager and persist_cache:
        cache_dir = cache_dir or DEFAULT_CACHE_DIR
        cache_manager.save_all(cache_dir)
        print(f"\nSaved caches to: {cache_dir}")
        cache_manager.print_stats()

    return {
        "results": all_results,
        "summaries": all_summaries,
        "comparison": comparison,
        "output_dir": str(output_path),
        "timestamp": timestamp,
        "cache_stats": cache_manager.all_stats() if cache_manager else None,
    }


def generate_comparison_table(summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate comparison table from configuration summaries."""
    rows = []

    for summary in summaries:
        config_info = summary.get("config", {})

        row = {
            "backend": config_info.get("backend", ""),
            "embedding_model": config_info.get("embedding_model", ""),
            "num_queries": summary.get("num_completed", 0),
            "caching_enabled": config_info.get("caching_enabled", False),
        }

        # Add latency stats
        lat_row = latency_stats_to_row(summary.get("latency_stats", {}), {})
        row.update(lat_row)

        # Add retrieval stats
        retr_stats = summary.get("retrieval_stats", {})
        for metric in ["hit@5", "recall@5", "mrr@10", "ndcg@10"]:
            if metric in retr_stats:
                row[f"{metric}_mean"] = retr_stats[metric].get("mean", 0)

        # Add answer stats
        ans_stats = summary.get("answer_stats", {})
        for metric in ["exact_match", "token_f1", "judge_correctness"]:
            if metric in ans_stats:
                row[f"{metric}_mean"] = ans_stats[metric].get("mean", 0)

        # Add cache stats
        cache_stats = summary.get("cache_stats", {})
        if cache_stats:
            for cache_name in ["embeddings", "answers", "judge"]:
                if cache_name in cache_stats:
                    row[f"cache_{cache_name}_hit_rate"] = cache_stats[cache_name].get("hit_rate", 0)

        rows.append(row)

    return rows


def save_comparison_csv(rows: List[Dict[str, Any]], path: Path):
    """Save comparison table to CSV."""
    if not rows:
        return

    all_keys = []
    for row in rows:
        for key in row.keys():
            if key not in all_keys:
                all_keys.append(key)

    with open(path, 'w') as f:
        f.write(",".join(all_keys) + "\n")
        for row in rows:
            values = []
            for key in all_keys:
                val = row.get(key, "")
                if isinstance(val, float):
                    values.append(f"{val:.4f}")
                elif isinstance(val, bool):
                    values.append(str(val))
                else:
                    values.append(str(val))
            f.write(",".join(values) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation experiments with caching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with caching enabled (default)
  python -m rag_pipeline.run_experiments --limit 100
  
  # Run without caching (for baseline comparison)
  python -m rag_pipeline.run_experiments --limit 100 --no-cache
  
  # Run with persistent cache (saves/loads from disk)
  python -m rag_pipeline.run_experiments --limit 100 --cache-dir my_cache
"""
    )

    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries")
    parser.add_argument("--k", type=int, default=10, help="Number of documents to retrieve")
    parser.add_argument("--no-generate", action="store_true", help="Skip answer generation")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM judge")
    parser.add_argument("--output-dir", type=str, default="eval_results", help="Output directory")
    parser.add_argument("--backend", type=str, default="chroma", help="Vector store backend")
    parser.add_argument("--embedding-model", type=str, default=None, help="Embedding model")
    
    # Caching options
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR, help="Cache directory")
    parser.add_argument("--no-persist-cache", action="store_true", help="Don't persist cache to disk")

    args = parser.parse_args()

    embedding_model = args.embedding_model or config.EMBEDDING_MODEL_NAME
    configurations = [(args.backend, embedding_model, None)]

    results = run_experiments(
        configurations=configurations,
        eval_limit=args.limit,
        k=args.k,
        generate_answer=not args.no_generate,
        use_llm_judge=not args.no_judge,
        output_dir=args.output_dir,
        enable_cache=not args.no_cache,
        cache_dir=args.cache_dir,
        persist_cache=not args.no_persist_cache,
    )

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"Results: {results['output_dir']}")
    
    if results.get("cache_stats"):
        print("\nFinal Cache Statistics:")
        for name, stats in results["cache_stats"].items():
            hit_rate = stats.get("hit_rate", 0) * 100
            print(f"  {name}: {stats['hits']} hits, {stats['misses']} misses ({hit_rate:.1f}% hit rate)")


if __name__ == "__main__":
    main()
