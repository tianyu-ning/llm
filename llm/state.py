# llm/state.py
from dataclasses import dataclass, field
from typing import Any, Dict
from datetime import datetime
import threading

@dataclass
class GlobalState:
    model: Any = None
    tokenizer: Any = None
    current_model: str | None = None
    current_conversation_id: str | None = None
    conversations: Dict[str, dict] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=lambda: {
        'total_requests': 0,
        'total_tokens': 0,
        'total_time': 0,
        'start_time': datetime.now(),
        'failed_requests': 0
    })
    generation_stop_event: Any = None
    is_generating: bool = False
    model_lock: threading.Lock = field(default_factory=threading.Lock)
    default_params: Dict[str, Any] = field(default_factory=lambda: {
        'max_new_tokens': 2048,
        'temperature': 0.7,
        'top_p': 0.9,
        'repetition_penalty': 1.1,
        'max_history': 10,
        'top_k': 50,
        'do_sample': True
    })
    current_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.current_params:
            self.current_params = self.default_params.copy()

# single app-wide state instance
state = GlobalState()