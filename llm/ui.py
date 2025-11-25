"""
模块化的 Gradio UI（迁移自单文件 app.py）
依赖：llm.config, llm.state, llm.models, llm.convos, llm.tokenizer, llm.generation, llm.utils
所有 handler 返回基础数据或 gr.update，以便测试和 UI 层分离。
"""

import gradio as gr
from typing import List, Tuple, Optional
from . import convos, models, tokenizer as tkmod, generation, utils
from .state import state
from .config import CONFIG
from .utils import clean_text, truncate_text
from .convos import get_conversation_dropdown_options, get_conversation_history
from .models import load_model, unload_model
from .generation import chat_stream, stop_generation
from datetime import datetime
import json
import tempfile
import logging

logger = logging.getLogger(__name__)

# System and stats HTML renderers are ported from app.py for UI convenience.
def get_system_info_html(get_system_info_fn):
    """Wrap to generate HTML (get_system_info_fn is passed from main to keep separation)"""
    system_info = get_system_info_fn()
    # Small version of the original HTML generator for system info
    html = f"""
    <div style="background: linear-gradient(135deg,#4facfe 0%, #00f2fe 100%); color: white; padding: 12px; border-radius: 10px;">
      <h3 style="margin-top:0;">💻 系统监控</h3>
      <div>CPU: {system_info.get('cpu_usage', 0):.1f}%</div>
      <div>内存: {system_info.get('memory_usage', 0):.1f}%</div>
      <div>磁盘: {system_info.get('disk_usage', 0):.1f}%</div>
    """
    if system_info.get('cuda_available', False):
        html += f"<div>GPU: {system_info.get('torch_gpu_name','未知')} | 显存: {system_info.get('torch_gpu_memory_allocated', 0):.2f}G</div>"
    else:
        html += "<div style='color: #ff6b6b;'>⚠️ CUDA不可用，模型运行在CPU上</div>"
    html += "</div>"
    return html

def get_stats_html(get_stats_fn):
    st = get_stats_fn()
    run_time = datetime.now() - st['start_time']
    minutes = int(run_time.total_seconds() // 60)
    avg_time = st['total_time'] / max(1, st['total_requests'])
    html = f"""
    <div style="padding:12px;border-radius:10px;background:linear-gradient(135deg,#f093fb 0,#f5576c 100%);color:white;">
      <div>请求数: {st['total_requests']}</div>
      <div>Token数: {st['total_tokens']}</div>
      <div>平均响应: {avg_time:.2f}s</div>
      <div>运行: {minutes}m</div>
    </div>
    """
    return html


def create_interface(get_system_info_fn=None, get_stats_fn=None):
    """
    建立完整 Gradio 界面，类似原始 app.py 的 UI 及事件绑定。
    get_system_info_fn & get_stats_fn 是可注入回调（用于测试或模块化）
    """
    # default callbacks if not provided (keep minimal)
    if get_system_info_fn is None:
        from . import utils as _u
        get_system_info_fn = lambda: {}  # minimal; main.py will pass real functions
    if get_stats_fn is None:
        get_stats_fn = lambda: state.stats

    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        title="大模型对话系统",
        css="""
        .gradio-container { max-width: 95% !important; }
        .conversation-item { padding: 8px 12px; margin: 2px 0; border-radius: 6px; cursor: pointer; }
        .conversation-item:hover { background: rgba(0,0,0,0.05); }
        .conversation-active { background: rgba(59,130,246,0.1); border-left: 3px solid #3b82f6; }
        """
    ) as demo:

        gr.Markdown("# 🤖 大模型对话系统\n**模块化 UI（迁移）**")

        # 初始选择列表
        conv_options = get_conversation_dropdown_options()
        conversation_value = state.current_conversation_id or (conv_options[0][1] if conv_options else "")

        # left column control panel
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### 💬 对话管理")
                with gr.Row():
                    new_convo_btn = gr.Button("🆕 新建对话", variant="primary", size="sm")
                    refresh_convos_btn = gr.Button("🔄 刷新", variant="secondary", size="sm")
                conversation_dropdown = gr.Dropdown(choices=conv_options, value=conversation_value, label="选择对话", interactive=True, filterable=True)
                conversation_state = gr.State(state.current_conversation_id)

                with gr.Row():
                    delete_convo_btn = gr.Button("🗑️ 删除对话", variant="stop", size="sm")
                    export_convo_btn = gr.Button("📤 导出对话", variant="secondary", size="sm")

                gr.Markdown("### 🚀 模型控制")
                model_dropdown = gr.Dropdown(choices=list(models.MODEL_PATHS.keys()), value=state.current_model or (list(models.MODEL_PATHS.keys())[0] if models.MODEL_PATHS else ""), label="选择模型", filterable=True)
                with gr.Row():
                    load_btn = gr.Button("🔄 加载模型", variant="primary")
                    unload_btn = gr.Button("🗑️ 卸载模型", variant="secondary")
                load_status = gr.Markdown("👆 请选择并加载模型")
                model_info_html = gr.HTML()

                with gr.Accordion("🔧 模型管理", open=False):
                    new_model_name = gr.Textbox(label="模型名称", placeholder="显示名称")
                    new_model_path = gr.Textbox(label="模型路径", placeholder="本地路径")
                    with gr.Row():
                        add_model_btn = gr.Button("➕ 添加模型")
                        remove_model_btn = gr.Button("➖ 移除模型")

                gr.Markdown("### ⚙️ 生成参数")
                max_new_tokens = gr.Slider(512, 8192, value=state.current_params['max_new_tokens'], step=256, label="生成长度")
                temperature = gr.Slider(0.1, 2.0, value=state.current_params['temperature'], step=0.1, label="温度")
                top_p = gr.Slider(0.1, 1.0, value=state.current_params['top_p'], step=0.1, label="Top-P")
                top_k = gr.Slider(1, 100, value=state.current_params['top_k'], step=1, label="Top-K")
                repetition_penalty = gr.Slider(1.0, 2.0, value=state.current_params['repetition_penalty'], step=0.1, label="重复惩罚")
                max_history_slider = gr.Slider(1, 50, value=state.current_params['max_history'], step=1, label="对话记忆轮数")
                do_sample = gr.Checkbox(value=state.current_params['do_sample'], label="随机采样")
                with gr.Row():
                    update_btn = gr.Button("💾 保存参数", variant="primary")
                    reset_btn = gr.Button("🔄 重置默认", variant="secondary")
                param_status = gr.Markdown("✅ 参数已就绪")

                gr.Markdown("### 📊 系统与统计")
                with gr.Row():
                    refresh_sys_btn = gr.Button("🔄 刷新状态", variant="secondary", size="sm")
                    stop_btn = gr.Button("⏹️ 停止生成", variant="stop", size="sm")
                    clean_memory_btn = gr.Button("🧹 清理内存", variant="secondary", size="sm")
                system_html = gr.HTML(get_system_info_html(get_system_info_fn))
                stats_html = gr.HTML(get_stats_html(get_stats_fn))

            # right column chat area
            with gr.Column(scale=2, min_width=600):
                current_conv_md = gr.Markdown(lambda: convos.get_current_conversation_info() if hasattr(convos, 'get_current_conversation_info') else "当前对话")
                chatbot = gr.Chatbot(value=get_conversation_history(state.current_conversation_id), label="💬 智能对话", height=520, type="tuples", show_copy_button=True)
                with gr.Row():
                    msg_box = gr.Textbox(placeholder="请输入问题，按Enter发送", lines=2, max_lines=5, show_label=False)
                    send_btn = gr.Button("🚀 发送")
                # quick buttons
                with gr.Row():
                    quick_clear = gr.Button("🗑️ 清空当前")
                    quick_example1 = gr.Button("👋 打个招呼")
                    quick_example2 = gr.Button("📝 写段代码")
                    quick_example3 = gr.Button("🤔 解释概念")

                message_counter = gr.HTML(lambda: f"当前对话: {len(convos.get_conversation_history(state.current_conversation_id))//2} 条消息")

        # ---------- handlers ----------
        def handle_load_model(model_name: str):
            if not model_name:
                return "❌ 请选择模型", "", get_stats_html(get_stats_fn), get_system_info_html(get_system_info_fn)
            if model_name not in models.MODEL_PATHS:
                return "❌ 模型路径不可用", "", get_stats_html(get_stats_fn), get_system_info_html(get_system_info_fn)
            result, info_html = load_model(models.MODEL_PATHS[model_name], model_name)
            return result, info_html, get_stats_html(get_stats_fn), get_system_info_html(get_system_info_fn)

        def handle_unload_model():
            result, info_html = unload_model()
            return result, info_html, get_stats_html(get_stats_fn), get_system_info_html(get_system_info_fn)

        def handle_send(message: str, conversation_id: str):
            # Streaming generator - adapted to gradio's streaming expectations
            if not message.strip():
                yield get_conversation_history(conversation_id), get_stats_html(get_stats_fn), get_system_info_html(get_system_info_fn), "", convos.get_current_conversation_info()
                return
            for history, stats_html, system_html in chat_stream(message, conversation_id):
                # stats_html and system_html are placeholders - UI uses our renderers
                yield history, get_stats_html(get_stats_fn), get_system_info_html(get_system_info_fn), "", convos.get_current_conversation_info()

        def handle_new_conversation():
            new_id = convos.create_new_conversation("新对话")
            cfg = json.loads(json.dumps({"last_conversation": new_id}))  # light update; real save uses config module
            # update dropdown choices
            opts = convos.get_conversation_dropdown_options()
            return gr.update(choices=opts, value=new_id), convos.get_conversation_history(new_id), convos.get_current_conversation_info(), new_id

        def handle_refresh_conversations():
            convos.load_conversations()
            opts = convos.get_conversation_dropdown_options()
            cur = state.current_conversation_id or (opts[0][1] if opts else "")
            return gr.update(choices=opts, value=cur)

        def handle_conversation_change(conv_id: str):
            if conv_id and conv_id in state.conversations:
                state.current_conversation_id = conv_id
                # persist selection
                # config.save_config(...) -- main app should call this
                return convos.get_conversation_history(conv_id), convos.get_current_conversation_info(), conv_id
            return gr.update(), gr.update(), state.current_conversation_id

        def handle_delete_conversation(conv_id: str):
            if not conv_id:
                return gr.update(), gr.update(), gr.update(), gr.update(), "❌ 删除失败"
            success = convos.delete_conversation(conv_id)
            if success:
                opts = convos.get_conversation_dropdown_options()
                current_value = state.current_conversation_id or (opts[0][1] if opts else "")
                history = convos.get_conversation_history(state.current_conversation_id) if state.current_conversation_id else []
                return gr.update(choices=opts, value=current_value), history, convos.get_current_conversation_info(), state.current_conversation_id, "✅ 对话已删除"
            return gr.update(), gr.update(), gr.update(), gr.update(), "❌ 删除失败"

        def handle_export_conversation(conv_id: str):
            if not conv_id or conv_id not in state.conversations:
                return gr.update(value=None), "", "❌ 导出失败"
            convo = state.conversations[conv_id]
            export_data = {"title": convo["title"], "created_at": convo["created_at"], "messages": convo["messages"], "model_used": convo.get("model_used","未知")}
            export_str = json.dumps(export_data, ensure_ascii=False, indent=2)
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8")
            tf.write(export_str); tf.flush(); tf.close()
            filename = f"{convo['title']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            return gr.update(value=tf.name), filename, "✅ 导出成功"

        def handle_clear_current_conversation(conv_id: str):
            if not conv_id or conv_id not in state.conversations:
                return [], convos.get_current_conversation_info()
            state.conversations[conv_id]["messages"] = []
            convos.save_conversation(conv_id)
            return [], convos.get_current_conversation_info()

        def handle_quick_example(kind: str):
            examples = {
                "greeting": "你好！请介绍一下你自己。",
                "code": "请用Python写一个快速排序算法，并添加详细注释。",
                "explain": "请用通俗易懂的方式解释什么是机器学习。"
            }
            return examples.get(kind, "你好！")

        # bind events
        load_btn.click(fn=handle_load_model, inputs=[model_dropdown], outputs=[load_status, model_info_html, stats_html, system_html])
        unload_btn.click(fn=handle_unload_model, outputs=[load_status, model_info_html, stats_html, system_html])
        send_event = msg_box.submit(fn=handle_send, inputs=[msg_box, conversation_state], outputs=[chatbot, stats_html, system_html, msg_box, current_conv_md])
        send_btn.click(fn=handle_send, inputs=[msg_box, conversation_state], outputs=[chatbot, stats_html, system_html, msg_box, current_conv_md])
        new_convo_btn.click(fn=handle_new_conversation, outputs=[conversation_dropdown, chatbot, current_conv_md, conversation_state])
        refresh_convos_btn.click(fn=handle_refresh_conversations, outputs=[conversation_dropdown])
        conversation_dropdown.change(fn=handle_conversation_change, inputs=[conversation_dropdown], outputs=[chatbot, current_conv_md, conversation_state])
        delete_convo_btn.click(fn=lambda: handle_delete_conversation(state.current_conversation_id), outputs=[conversation_dropdown, chatbot, current_conv_md, conversation_state, load_status])
        export_convo_btn.click(fn=lambda: handle_export_conversation(state.current_conversation_id), outputs=[gr.File(), gr.Textbox(), load_status])
        quick_clear.click(fn=lambda: handle_clear_current_conversation(state.current_conversation_id), outputs=[chatbot, current_conv_md])
        quick_example1.click(fn=lambda: handle_quick_example("greeting"), outputs=[msg_box])
        quick_example2.click(fn=lambda: handle_quick_example("code"), outputs=[msg_box])
        quick_example3.click(fn=lambda: handle_quick_example("explain"), outputs=[msg_box])
        refresh_sys_btn.click(fn=lambda: [get_system_info_html(get_system_info_fn), get_stats_html(get_stats_fn)], outputs=[system_html, stats_html])
        stop_btn.click(fn=stop_generation, outputs=[load_status])
        clean_memory_btn.click(fn=models.unload_model, outputs=[load_status])  # reuse unload to ~free
        add_model_btn.click(fn=models.add_model_path, inputs=[new_model_name, new_model_path], outputs=[load_status, model_dropdown])
        remove_model_btn.click(fn=models.remove_model_path, inputs=[model_dropdown], outputs=[load_status, model_dropdown])
        update_btn.click(fn=lambda a,b,c,d,e,f,g: update_and_report(a,b,c,d,e,f,g), inputs=[max_new_tokens, temperature, top_p, repetition_penalty, max_history_slider, top_k, do_sample], outputs=[param_status])
        reset_btn.click(fn=lambda: reset_and_report(), outputs=[max_new_tokens, temperature, top_p, repetition_penalty, max_history_slider, top_k, do_sample, param_status])

    return demo

# helper for param update/reset (keeps state centrally)
def update_and_report(max_new_tokens, temperature, top_p, repetition_penalty, max_history, top_k, do_sample):
    state.current_params.update({
        'max_new_tokens': max_new_tokens,
        'temperature': temperature,
        'top_p': top_p,
        'repetition_penalty': repetition_penalty,
        'max_history': max_history,
        'top_k': top_k,
        'do_sample': do_sample
    })
    return f"✅ 参数已更新 | 长度: {max_new_tokens} | 温度: {temperature}"

def reset_and_report():
    state.current_params = state.default_params.copy()
    return (
        state.default_params['max_new_tokens'],
        state.default_params['temperature'],
        state.default_params['top_p'],
        state.default_params['repetition_penalty'],
        state.default_params['max_history'],
        state.default_params.get('top_k', 50),
        state.default_params.get('do_sample', True),
        "✅ 参数已重置为默认值"
    )