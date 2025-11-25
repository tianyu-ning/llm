# tokenizer.py
from functools import lru_cache
from typing import Optional

# LRU cache keyed by tokenizer id + text content
@lru_cache(maxsize=8192)
def _token_len_cached(tokenizer_id: int, text: str) -> int:
    # fallback simple heuristic if tokenizer can't be used
    return max(1, len(text) // 4)

def count_tokens(tokenizer, text: Optional[str]) -> int:
    """Return token count for text using tokenizer; use cache keyed by tokenizer id."""
    if text is None:
        return 0
    tid = id(tokenizer) if tokenizer is not None else 0
    try:
        if tokenizer is None:
            return _token_len_cached(tid, text)
        # try using tokenizer.encode where available
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return _token_len_cached(tid, text)

def clear_token_cache():
    try:
        _token_len_cached.cache_clear()
    except Exception:
        pass