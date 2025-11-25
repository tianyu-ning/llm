# llm/config.py
import os
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from .utils import atomic_write

CONFIG = {
    "conversations_dir": "conversations",
    "models_dir": "models",
    "config_file": "app_config.json",
    "models_config_file": "models_config.json",
    "max_conversations": 100,
    "max_message_length": 8000,
    "default_timeout": 120,
    "max_input_tokens": 6000,
    "reserved_tokens": 500,
    "allow_trust_remote_code": False
}

# ensure dirs
Path(CONFIG["conversations_dir"]).mkdir(exist_ok=True)
Path(CONFIG["models_dir"]).mkdir(exist_ok=True)

def load_models_config(defaults: Dict[str,str]) -> Dict[str,str]:
    try:
        if os.path.exists(CONFIG["models_config_file"]):
            with open(CONFIG["models_config_file"], 'r', encoding='utf-8') as f:
                cfg = json.load(f)
                return cfg.get("models", defaults)
    except Exception:
        pass
    return defaults

def save_models_config(models: Dict[str,str]) -> None:
    try:
        atomic_write(CONFIG["models_config_file"], {"models": models})
    except Exception:
        # best-effort; caller should handle logging
        pass

def load_config() -> Dict[str,Any]:
    try:
        if os.path.exists(CONFIG["config_file"]):
            with open(CONFIG["config_file"], 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {"last_model": None, "last_conversation": None}

def save_config(cfg: Dict[str,Any]) -> None:
    try:
        atomic_write(CONFIG["config_file"], cfg)
    except Exception:
        pass