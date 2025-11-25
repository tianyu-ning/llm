# 完整 Gradio 界面（迁移自 app.py，但引用模块化函数）

import time
import threading
import gc
import logging
from typing import Generator, Tuple, List
from transformers import TextIteratorStreamer
from .state import state
from .convos import get_conversation_history, save_conversation
from .tokenizer import count_tokens
from .utils import clean_text, truncate_text
from .config import CONFIG

logger = logging.getLogger(__name__)

def _safe_get_refs():
    """在持锁条件下安全获取 model 和 tokenizer 引用以供生成使用。"""
    with state.model_lock:
        if state.model is None or state.tokenizer is None:
            return None, None
        return state.model, state.tokenizer

def chat_stream(message: str, conversation_id: str) -> Generator[Tuple[List[Tuple[str,str]], str, str], None, None]:
    """
    流式生成。返回（history, stats_html, system_html）与原来格式兼容。
    采用局部引用 model_ref / tokenizer_ref 来避免并发卸载时崩溃。
    """
    model_ref, tokenizer_ref = _safe_get_refs()
    if model_ref is None or tokenizer_ref is None:
        # 立即给出提示
        yield get_conversation_history(conversation_id) + [[message, "⚠️ 请先加载模型！"]], "", ""
        return

    # 标记生成开始
    state.is_generating = True
    state.generation_stop_event = threading.Event()

    try:
        conversation = state.conversations.get(conversation_id, {"messages":[]})
        max_history = state.current_params.get('max_history', 10)
        all_messages = conversation.get("messages", [])
        messages_to_keep = min(len(all_messages), max_history * 2)
        recent_messages = all_messages[-messages_to_keep:] if messages_to_keep>0 else []

        clean_message = truncate_text(clean_text(message))
        messages_for_model = recent_messages + [{"role":"user","content":clean_message}]
        # NOTE: 这里假设 tokenizer 包含 apply_chat_template；如果没有，UI 层将回退
        try:
            text = tokenizer_ref.apply_chat_template(
                messages_for_model,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=getattr(tokenizer_ref, "enable_thinking", False)
            )
        except Exception:
            text = "\n".join([f"{m['role']}: {m['content']}" for m in messages_for_model]) + "\nassistant:"

        # 简单输入截断
        input_ids = tokenizer_ref.encode(text)
        max_input_length = CONFIG["max_input_tokens"] - CONFIG["reserved_tokens"]
        if len(input_ids) > max_input_length:
            text = tokenizer_ref.decode(input_ids[:max_input_length])
            if not text.endswith("..."):
                text += "...(上下文已截断)"

        model_device = getattr(model_ref, "device", "cpu")
        model_inputs = tokenizer_ref([text], return_tensors="pt").to(model_device)

        streamer = TextIteratorStreamer(tokenizer_ref, skip_prompt=True, timeout=CONFIG["default_timeout"])
        generation_kwargs = dict(
            **model_inputs,
            max_new_tokens=state.current_params.get('max_new_tokens', 2048),
            temperature=state.current_params.get('temperature', 0.7),
            top_p=state.current_params.get('top_p', 0.9),
            top_k=state.current_params.get('top_k', 50),
            do_sample=state.current_params.get('do_sample', True),
            pad_token_id=tokenizer_ref.eos_token_id,
            repetition_penalty=state.current_params.get('repetition_penalty', 1.1),
            streamer=streamer
        )

        start_time = time.time()
        thread = threading.Thread(target=model_ref.generate, kwargs=generation_kwargs)
        thread.daemon = True
        thread.start()

        generated_text = ""
        thinking_content = ""
        is_thinking = True

        for new_text in streamer:
            if state.generation_stop_event and state.generation_stop_event.is_set():
                logger.info("生成被用户停止")
                break
            generated_text += new_text

            if is_thinking and "</think>" in generated_text:
                parts = generated_text.split("</think>", 1)
                thinking_content = parts[0] + "</think>"
                generated_text = parts[1] if len(parts) > 1 else ""
                is_thinking = False

            if is_thinking:
                partial_response = f"🤔 **思考中...**\n\n{thinking_content + generated_text}"
            else:
                if thinking_content:
                    partial_response = f"<details style='margin-bottom: 10px;'><summary>🧠 思考过程</summary>\n\n{thinking_content}\n\n</details>\n\n{generated_text}"
                else:
                    partial_response = generated_text

            partial_response = clean_text(partial_response)
            current_history = get_conversation_history(conversation_id) + [[clean_message, partial_response]]
            yield current_history, "", ""  # stats/system placeholders (UI 会调用 get_stats_html / get_system_info_html)

        generation_time = time.time() - start_time
        if state.generation_stop_event and state.generation_stop_event.is_set():
            generated_text += "\n\n---\n*生成已被用户停止*"

        output_ids = tokenizer_ref.encode(generated_text)
        tokens_generated = len(output_ids)

        # update stats & conversation
        state.stats['total_requests'] += 1
        state.stats['total_tokens'] += tokens_generated
        state.stats['total_time'] += generation_time

        conversation.setdefault("messages", []).append({"role":"user","content":clean_message})
        conversation.setdefault("messages", []).append({"role":"assistant","content":generated_text})
        conversation["model_used"] = state.current_model

        if len(conversation.get("messages", [])) == 2:
            conversation["title"] = clean_message[:20] + "..." if len(clean_message)>20 else clean_message

        save_conversation(conversation_id)

        final_history = get_conversation_history(conversation_id)
        yield final_history, "", ""

    except Exception as e:
        logger.error(f"生成时出错: {e}")
        state.stats['failed_requests'] += 1
        yield get_conversation_history(conversation_id) + [[message, f"❌ 生成响应时出错: {e}"]], "", ""
    finally:
        state.is_generating = False
        state.generation_stop_event = None

def stop_generation() -> str:
    if state.generation_stop_event and state.is_generating:
        state.generation_stop_event.set()
        return "🛑 正在停止生成..."
    return "ℹ️ 没有正在进行的生成"