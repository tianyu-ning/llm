# llm/utils.py
import json
import os
from typing import Any

def atomic_write(path: str, data: Any) -> None:
    """Write JSON to path atomically."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def clean_text(text: str) -> str:
    if not text:
        return ""
    import re
    try:
        return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    except Exception:
        return re.sub(r'[^\x00-\x7F\u4e00-\u9fff]+', '', text)

def truncate_text(text: str, max_length: int) -> str:
    if text is None:
        return ""
    if len(text) > max_length:
        return text[:max_length] + "...(已截断)"
    return text