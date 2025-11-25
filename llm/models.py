
import os, time, gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
from .state import state
from .config import CONFIG, load_config, save_config
from .tokenizer import clear_token_cache
from .utils import clean_text

def load_model_to_cpu(model_path: str, model_display_name: str) -> Tuple[str,str]:
    trc = CONFIG.get("allow_trust_remote_code", False)
    state.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trc)
    clear_token_cache()
    state.model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map=None,
        low_cpu_mem_usage=True,
        trust_remote_code=trc
    )
    state.model = state.model.to('cpu')
    state.current_model = model_display_name
    cfg = load_config()
    cfg["last_model"] = model_display_name
    save_config(cfg)
    return f"✅ **{model_display_name}** 已加载到CPU", ""

def load_model_with_fallback(model_path: str, model_display_name: str) -> Tuple[str,str]:
    # Guarded: assumes caller holds state.model_lock if needed
    if not torch.cuda.is_available():
        return load_model_to_cpu(model_path, model_display_name)
    trc = CONFIG.get("allow_trust_remote_code", False)
    state.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trc)
    clear_token_cache()
    try:
        state.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=trc
        )
    except Exception:
        state.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map=None, low_cpu_mem_usage=True, trust_remote_code=trc)
        state.model = state.model.to('cuda')
    state.current_model = model_display_name
    cfg = load_config(); cfg["last_model"] = model_display_name; save_config(cfg)
    return f"✅ **{model_display_name}** 已加载: {state.model.device}", ""

def load_model(path: str, model_display_name: str) -> Tuple[str,str]:
    if state.model is not None:
        unload_model()
    if not os.path.exists(path):
        return f"❌ 模型路径不存在: {path}", ""
    # prevent concurrent loads
    with state.model_lock:
        return load_model_with_fallback(path, model_display_name)

def unload_model() -> Tuple[str,str]:
    with state.model_lock:
        if state.model is None:
            return "ℹ️ 没有加载的模型", ""
        # gracefully stop generation
        if getattr(state, "is_generating", False) and state.generation_stop_event:
            state.generation_stop_event.set()
        try:
            if hasattr(state.model, "device") and "cuda" in str(state.model.device):
                try:
                    state.model = state.model.cpu()
                except Exception:
                    pass
            del state.model
            del state.tokenizer
        except Exception:
            pass
        state.model = None
        state.tokenizer = None
        state.current_model = None
        clear_token_cache()
        gc.collect()
        torch.cuda.empty_cache()
        return "✅ 模型已卸载", ""