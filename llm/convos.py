
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from .config import CONFIG
from .state import state
from .utils import atomic_write

CONV_DIR = Path(CONFIG["conversations_dir"])
CONV_DIR.mkdir(exist_ok=True)

def load_conversations():
    state.conversations = {}
    for path in CONV_DIR.glob("*.json"):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                convo = json.load(f)
                state.conversations[convo["id"]] = convo
        except Exception:
            # ignore broken file
            continue
    # keep latest N
    sorted_convos = sorted(state.conversations.values(), key=lambda x: x.get("updated_at", ""), reverse=True)[:CONFIG["max_conversations"]]
    state.conversations = {c["id"]: c for c in sorted_convos}

def save_conversation(conversation_id: str):
    if conversation_id not in state.conversations:
        return
    convo = state.conversations[conversation_id]
    convo["updated_at"] = datetime.now().isoformat()
    path = CONV_DIR / f"{conversation_id}.json"
    atomic_write(str(path), convo)

def create_new_conversation(title="新对话") -> str:
    conversation_id = str(uuid.uuid4())
    state.conversations[conversation_id] = {
        "id": conversation_id,
        "title": title,
        "messages": [],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "model_used": state.current_model
    }
    state.current_conversation_id = conversation_id
    save_conversation(conversation_id)
    return conversation_id

def delete_conversation(conversation_id: str) -> bool:
    if conversation_id not in state.conversations:
        return False
    del state.conversations[conversation_id]
    path = CONV_DIR / f"{conversation_id}.json"
    if path.exists():
        path.unlink()
    # fallback current conversation
    if state.current_conversation_id == conversation_id:
        state.current_conversation_id = next(iter(state.conversations.keys()), None)
        if state.current_conversation_id is None:
            create_new_conversation()
    return True

def get_conversation_history(conversation_id: str) -> List[List[str]]:
    if conversation_id not in state.conversations:
        return []
    msgs = state.conversations[conversation_id]["messages"]
    pairs = []
    for i in range(0, len(msgs), 2):
        if i + 1 < len(msgs):
            pairs.append([msgs[i]["content"], msgs[i+1]["content"]])
    return pairs

def get_conversation_dropdown_options():
    options = []
    for conv_id, convo in sorted(state.conversations.items(), key=lambda x: x[1].get("updated_at",""), reverse=True):
        title = convo["title"]
        message_count = len(convo.get("messages", [])) // 2
        options.append((f"{title} ({message_count}条消息)", conv_id))
    return options