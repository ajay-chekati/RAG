"""Simple caching for RAG evaluation: embeddings, answers, and judge results."""

import json
import hashlib
from typing import Any, Optional, Dict, Tuple, List
from pathlib import Path


def normalize_key(key: Tuple) -> str:
    """Convert tuple key to string for storage."""
    # Convert tuple elements to strings and join
    key_parts = []
    for part in key:
        if isinstance(part, (list, tuple)):
            # Sort lists/tuples for consistent ordering
            key_parts.append(str(sorted(part)))
        else:
            key_parts.append(str(part))
    return "|||".join(key_parts)


def hash_key(key: Tuple) -> str:
    """Create MD5 hash of key for compact storage."""
    normalized = normalize_key(key)
    return hashlib.md5(normalized.encode()).hexdigest()


class SimpleCache:
    """In-memory cache with JSON persistence and hit/miss tracking."""
    
    def __init__(self, name: str = "cache", use_hash_keys: bool = False):
        self.name = name
        self.use_hash_keys = use_hash_keys
        self._store: Dict[str, Any] = {}
        self._hits = 0
        self._misses = 0
    
    def _make_key(self, key: Tuple) -> str:
        """Convert tuple key to string key."""
        if self.use_hash_keys:
            return hash_key(key)
        return normalize_key(key)
    
    def has(self, key: Tuple) -> bool:
        """Check if key exists in cache."""
        str_key = self._make_key(key)
        return str_key in self._store
    
    def get(self, key: Tuple, default: Any = None) -> Any:
        """Get value from cache, updating hit/miss stats."""
        str_key = self._make_key(key)
        if str_key in self._store:
            self._hits += 1
            return self._store[str_key]
        else:
            self._misses += 1
            return default
    
    def set(self, key: Tuple, value: Any) -> None:
        """Store value in cache."""
        str_key = self._make_key(key)
        self._store[str_key] = value
    
    def clear(self) -> None:
        """Clear all cached values and reset statistics."""
        self._store.clear()
        self._hits = 0
        self._misses = 0
    
    def size(self) -> int:
        """Return number of items in cache."""
        return len(self._store)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        
        return {
            "name": self.name,
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate": hit_rate,
            "size": len(self._store),
        }
    
    def reset_stats(self) -> None:
        """Reset hit/miss counters without clearing the cache."""
        self._hits = 0
        self._misses = 0
    
    def save(self, path: str) -> None:
        """Save cache to JSON file."""
        data = {
            "name": self.name,
            "use_hash_keys": self.use_hash_keys,
            "store": self._store,
            "stats": {
                "hits": self._hits,
                "misses": self._misses,
            }
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path: str) -> bool:
        """Load cache from JSON file. Returns True on success."""
        if not Path(path).exists():
            return False
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            self._store = data.get("store", {})
            stats = data.get("stats", {})
            self._hits = stats.get("hits", 0)
            self._misses = stats.get("misses", 0)
            return True
        except (json.JSONDecodeError, KeyError):
            return False
    
    def __repr__(self) -> str:
        stats = self.stats()
        return f"SimpleCache(name='{self.name}', size={stats['size']}, hit_rate={stats['hit_rate']:.2%})"


class CacheManager:
    """Manages embedding, answer, and judge caches together."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        
        # Create individual caches
        self.embeddings = SimpleCache(name="embeddings", use_hash_keys=False)
        self.answers = SimpleCache(name="answers", use_hash_keys=True)  # Long keys
        self.judge = SimpleCache(name="judge", use_hash_keys=True)  # Long keys
        
        # Load from disk if cache_dir specified and files exist
        if cache_dir:
            self.load_all(cache_dir)
    
    def all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all caches."""
        return {
            "embeddings": self.embeddings.stats(),
            "answers": self.answers.stats(),
            "judge": self.judge.stats(),
        }
    
    def print_stats(self) -> None:
        """Print a formatted summary of all cache statistics."""
        print("\n" + "=" * 50)
        print("Cache Statistics")
        print("=" * 50)
        
        for name, stats in self.all_stats().items():
            hit_rate_pct = stats["hit_rate"] * 100
            print(f"{name:12}: {stats['hits']:5} hits / {stats['misses']:5} misses "
                  f"({hit_rate_pct:5.1f}% hit rate, {stats['size']} items)")
        
        print("=" * 50)
    
    def save_all(self, cache_dir: str) -> None:
        """Save all caches to a directory."""
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self.embeddings.save(f"{cache_dir}/embeddings_cache.json")
        self.answers.save(f"{cache_dir}/answers_cache.json")
        self.judge.save(f"{cache_dir}/judge_cache.json")
    
    def load_all(self, cache_dir: str) -> Dict[str, bool]:
        """Load all caches from directory."""
        results = {
            "embeddings": self.embeddings.load(f"{cache_dir}/embeddings_cache.json"),
            "answers": self.answers.load(f"{cache_dir}/answers_cache.json"),
            "judge": self.judge.load(f"{cache_dir}/judge_cache.json"),
        }
        return results
    
    def clear_all(self) -> None:
        """Clear all caches."""
        self.embeddings.clear()
        self.answers.clear()
        self.judge.clear()
    
    def reset_all_stats(self) -> None:
        """Reset statistics for all caches without clearing data."""
        self.embeddings.reset_stats()
        self.answers.reset_stats()
        self.judge.reset_stats()


def make_embedding_key(embedding_model: str, text: str) -> Tuple[str, str, str]:
    """Create cache key for embedding."""
    return (embedding_model, "query", text)


def make_answer_key(llm_model: str, question: str, doc_ids: List[str]) -> Tuple[str, str, Tuple]:
    """Create cache key for LLM answer."""
    return (llm_model, question, tuple(sorted(doc_ids)))


def make_judge_key(
    judge_model: str, 
    question: str, 
    gold_answer: str, 
    model_answer: str
) -> Tuple[str, str, str, str]:
    """Create cache key for judge result."""
    return (judge_model, question, gold_answer, model_answer or "")


if __name__ == "__main__":
    # Demo usage
    print("SimpleCache Demo")
    print("-" * 40)
    
    # Create a cache
    cache = SimpleCache(name="demo")
    
    # Simulate some operations
    key1 = ("model-a", "query", "What is AI?")
    key2 = ("model-a", "query", "What is ML?")
    key3 = ("model-b", "query", "What is AI?")
    
    # First access - misses
    print(f"Get key1 (miss): {cache.get(key1, 'NOT FOUND')}")
    print(f"Get key2 (miss): {cache.get(key2, 'NOT FOUND')}")
    
    # Store values
    cache.set(key1, [0.1, 0.2, 0.3])
    cache.set(key2, [0.4, 0.5, 0.6])
    
    # Second access - hits
    print(f"Get key1 (hit): {cache.get(key1)}")
    print(f"Get key2 (hit): {cache.get(key2)}")
    print(f"Get key3 (miss): {cache.get(key3, 'NOT FOUND')}")
    
    # Print stats
    print(f"\nCache: {cache}")
    print(f"Stats: {cache.stats()}")
    
    # Test persistence
    cache.save("/tmp/demo_cache.json")
    
    new_cache = SimpleCache(name="demo_loaded")
    new_cache.load("/tmp/demo_cache.json")
    print(f"\nLoaded cache: {new_cache}")
    print(f"Get key1 from loaded: {new_cache.get(key1)}")
    
    # Test CacheManager
    print("\n" + "=" * 40)
    print("CacheManager Demo")
    print("=" * 40)
    
    manager = CacheManager()
    
    # Use individual caches
    manager.embeddings.set(make_embedding_key("nomic", "test"), [0.1, 0.2])
    manager.answers.set(make_answer_key("llama", "question?", ["doc1", "doc2"]), "answer")
    manager.judge.set(make_judge_key("llama", "q", "gold", "model"), {"score": 0.9})
    
    # Simulate hits
    manager.embeddings.get(make_embedding_key("nomic", "test"))
    manager.answers.get(make_answer_key("llama", "question?", ["doc1", "doc2"]))
    
    manager.print_stats()

