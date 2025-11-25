import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import time
import psutil
import GPUtil
import os
from datetime import datetime
import threading
import re
import gc
import json
import uuid
import shutil
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Any

# 设置PyTorch主线程数
torch.set_num_threads(4)
# 设置Inter-op线程数（通常1个就够了）
torch.set_num_interop_threads(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 配置常量
CONFIG = {
    "conversations_dir": "conversations",
    "models_dir": "models",
    "config_file": "app_config.json",
    "models_config_file": "models_config.json",
    "max_conversations": 100,
    "max_message_length": 8000,
    "default_timeout": 120,
    "max_input_tokens": 6000,  # 增加输入token限制
    "reserved_tokens": 500     # 为生成预留的token数
}

# 创建必要的目录
Path(CONFIG["conversations_dir"]).mkdir(exist_ok=True)
Path(CONFIG["models_dir"]).mkdir(exist_ok=True)

# 检查CUDA可用性
def check_cuda_availability():
    """检查CUDA可用性并返回详细信息"""
    cuda_info = {
        'available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
        'device_name': None,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
    }
    
    if cuda_info['available'] and cuda_info['device_count'] > 0:
        try:
            cuda_info['device_name'] = torch.cuda.get_device_name(0)
            # 检查GPU内存
            if torch.cuda.is_available():
                torch.cuda.init()
                cuda_info['memory_allocated'] = torch.cuda.memory_allocated(0) / (1024**3)
                cuda_info['memory_reserved'] = torch.cuda.memory_reserved(0) / (1024**3)
                cuda_info['total_memory'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except Exception as e:
            logger.error(f"获取CUDA设备信息失败: {e}")
    
    return cuda_info

# 打印CUDA信息
cuda_info = check_cuda_availability()
logger.info(f"CUDA可用性: {cuda_info['available']}")
logger.info(f"GPU数量: {cuda_info['device_count']}")
if cuda_info['available']:
    logger.info(f"GPU名称: {cuda_info['device_name']}")
    logger.info(f"CUDA版本: {cuda_info['cuda_version']}")
    logger.info(f"GPU总内存: {cuda_info['total_memory']:.2f} GB")

# 从配置文件加载模型列表
def load_models_config():
    """从配置文件加载模型列表"""
    default_models = {
        "Qwen3-1.7B": "/Data/llm_modl_data/qwen3-1.7B",
        "Qwen3-4B-Thinking-FP8": "/Data/llm_modl_data/qwen3-4B-Thinking-2507-FP8"
    }
    
    try:
        if os.path.exists(CONFIG["models_config_file"]):
            with open(CONFIG["models_config_file"], 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get("models", default_models)
    except Exception as e:
        logger.error(f"加载模型配置失败: {e}")
    
    return default_models

def save_models_config(models):
    """保存模型配置"""
    try:
        with open(CONFIG["models_config_file"], 'w', encoding='utf-8') as f:
            json.dump({"models": models}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"保存模型配置失败: {e}")

# 加载模型配置
MODEL_PATHS = load_models_config()

# 全局状态
class GlobalState:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_model = None
        self.current_conversation_id = None
        self.conversations = {}
        self.stats = {
            'total_requests': 0,
            'total_tokens': 0,
            'total_time': 0,
            'start_time': datetime.now(),
            'failed_requests': 0
        }
        self.generation_stop_event = None
        self.is_generating = False
        self.model_lock = threading.Lock()  # 模型访问锁
        self.model_max_length = 8192  # 假设模型最大长度
        
        # 默认参数
        self.default_params = {
            'max_new_tokens': 2048,
            'temperature': 0.7,
            'top_p': 0.9,
            'repetition_penalty': 1.1,
            'max_history': 10,
            'top_k': 50,
            'do_sample': True
        }
        self.current_params = self.default_params.copy()

state = GlobalState()

# 配置管理
def load_config():
    """加载应用配置"""
    try:
        if os.path.exists(CONFIG["config_file"]):
            with open(CONFIG["config_file"], 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"加载配置失败: {e}")
    return {"last_model": None, "last_conversation": None}

def save_config(config):
    """保存应用配置"""
    try:
        with open(CONFIG["config_file"], 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"保存配置失败: {e}")

# 对话管理
def load_conversations():
    """加载所有对话"""
    state.conversations = {}
    conversations_dir = Path(CONFIG["conversations_dir"])
    
    for file_path in conversations_dir.glob("*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)
                state.conversations[conversation_data["id"]] = conversation_data
        except Exception as e:
            logger.error(f"加载对话失败 {file_path}: {e}")
    
    # 按更新时间排序，只保留最新的N个对话
    sorted_convos = sorted(
        state.conversations.values(),
        key=lambda x: x.get("updated_at", ""),
        reverse=True
    )[:CONFIG["max_conversations"]]
    
    state.conversations = {conv["id"]: conv for conv in sorted_convos}
    logger.info(f"已加载 {len(state.conversations)} 个对话")

def save_conversation(conversation_id=None):
    """保存当前对话"""
    if conversation_id not in state.conversations:
        return
    
    conversation = state.conversations[conversation_id]
    conversation["updated_at"] = datetime.now().isoformat()
    
    file_path = Path(CONFIG["conversations_dir"]) / f"{conversation_id}.json"
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(conversation, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"保存对话失败: {e}")

def create_new_conversation(title="新对话"):
    """创建新对话"""
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
    logger.info(f"创建新对话: {title}")
    return conversation_id

def delete_conversation(conversation_id):
    """删除对话"""
    if conversation_id in state.conversations:
        title = state.conversations[conversation_id]["title"]
        del state.conversations[conversation_id]
        file_path = Path(CONFIG["conversations_dir"]) / f"{conversation_id}.json"
        if file_path.exists():
            file_path.unlink()
        
        # 如果删除的是当前对话，切换到最新对话或创建新对话
        if state.current_conversation_id == conversation_id:
            if state.conversations:
                state.current_conversation_id = list(state.conversations.keys())[0]
            else:
                create_new_conversation()
        
        logger.info(f"删除对话: {title}")
        return True
    return False

def get_conversation_history(conversation_id):
    """获取对话的Gradio格式历史"""
    if conversation_id not in state.conversations:
        return []
    
    messages = state.conversations[conversation_id]["messages"]
    history = []
    for i in range(0, len(messages), 2):
        if i + 1 < len(messages):
            history.append([
                messages[i]["content"],
                messages[i + 1]["content"]
            ])
    return history

# 文本处理
def clean_text(text):
    """清理文本中的非法UTF-8字符"""
    if not text:
        return ""
    try:
        # 移除控制字符但保留换行和制表符
        cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        return cleaned
    except:
        return re.sub(r'[^\x00-\x7F\u4e00-\u9fff]+', '', text)

def truncate_text(text, max_length=CONFIG["max_message_length"]):
    """截断文本到指定长度"""
    if len(text) > max_length:
        return text[:max_length] + "...(已截断)"
    return text

# 系统监控
def get_system_info():
    """获取系统信息"""
    info = {}
    
    try:
        # CPU信息
        info['cpu_usage'] = psutil.cpu_percent(interval=0.1)
        info['memory_usage'] = psutil.virtual_memory().percent
        info['memory_used_gb'] = psutil.virtual_memory().used / (1024**3)
        info['memory_total_gb'] = psutil.virtual_memory().total / (1024**3)
        
        # CUDA信息
        info['cuda_available'] = torch.cuda.is_available()
        info['cuda_device_count'] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        if info['cuda_available'] and info['cuda_device_count'] > 0:
            try:
                # PyTorch GPU信息
                info['torch_gpu_name'] = torch.cuda.get_device_name(0)
                info['torch_gpu_memory_allocated'] = torch.cuda.memory_allocated(0) / (1024**3)
                info['torch_gpu_memory_reserved'] = torch.cuda.memory_reserved(0) / (1024**3)
                info['torch_gpu_total_memory'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                # GPUtil信息（备用）
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    info['gpu_name'] = gpu.name
                    info['gpu_usage'] = gpu.load * 100
                    info['gpu_memory_used'] = gpu.memoryUsed
                    info['gpu_memory_total'] = gpu.memoryTotal
                    info['gpu_temperature'] = gpu.temperature
                else:
                    # 如果没有GPUtil信息，使用PyTorch的信息
                    info['gpu_name'] = info['torch_gpu_name']
                    info['gpu_usage'] = (info['torch_gpu_memory_allocated'] / info['torch_gpu_total_memory']) * 100
                    info['gpu_memory_used'] = info['torch_gpu_memory_allocated'] * 1024  # 转换为MB
                    info['gpu_memory_total'] = info['torch_gpu_total_memory'] * 1024     # 转换为MB
                    info['gpu_temperature'] = 0  # PyTorch不提供温度信息
            except Exception as e:
                logger.warning(f"获取GPU信息失败: {e}")
                info['cuda_available'] = False
        
        # 磁盘信息
        disk_usage = psutil.disk_usage('/')
        info['disk_usage'] = disk_usage.percent
        info['disk_free_gb'] = disk_usage.free / (1024**3)
        
    except Exception as e:
        logger.error(f"获取系统信息失败: {e}")
    
    return info

def get_system_info_html():
    """获取系统信息的HTML显示"""
    system_info = get_system_info()
    
    html = """
    <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 15px; border-radius: 10px; margin: 10px 0;">
        <h3 style="margin-top: 0; margin-bottom: 15px;">💻 系统监控</h3>
    """
    
    try:
        # CPU和内存
        html += f"""
            <div style="margin: 10px 0;">
                <div style="display: flex; justify-content: space-between; font-size: 0.9em;">
                    <span>🖥️ CPU</span>
                    <span>{system_info.get('cpu_usage', 0):.1f}%</span>
                </div>
                <div style="background: rgba(255,255,255,0.3); border-radius: 10px; height: 6px; margin-top: 5px;">
                    <div style="background: #ff6b6b; width: {system_info.get('cpu_usage', 0)}%; height: 100%; border-radius: 10px;"></div>
                </div>
            </div>
            
            <div style="margin: 10px 0;">
                <div style="display: flex; justify-content: space-between; font-size: 0.9em;">
                    <span>💾 内存</span>
                    <span>{system_info.get('memory_usage', 0):.1f}%</span>
                </div>
                <div style="background: rgba(255,255,255,0.3); border-radius: 10px; height: 6px; margin-top: 5px;">
                    <div style="background: #4ecdc4; width: {system_info.get('memory_usage', 0)}%; height: 100%; border-radius: 10px;"></div>
                </div>
            </div>
        """
        
        # GPU信息
        if system_info.get('cuda_available', False):
            gpu_memory_usage = (system_info.get('torch_gpu_memory_allocated', 0) / system_info.get('torch_gpu_total_memory', 1)) * 100
            
            html += f"""
                <div style="margin: 10px 0;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.9em;">
                        <span>🎮 GPU内存</span>
                        <span>{gpu_memory_usage:.1f}%</span>
                    </div>
                    <div style="background: rgba(255,255,255,0.3); border-radius: 10px; height: 6px; margin-top: 5px;">
                        <div style="background: #45b7d1; width: {gpu_memory_usage}%; height: 100%; border-radius: 10px;"></div>
                    </div>
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px;">
                    <div style="text-align: center;">
                        <div style="font-size: 0.9em; font-weight: bold;">{system_info.get('cuda_device_count', 0)}</div>
                        <div style="font-size: 0.7em;">GPU数量</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 0.9em; font-weight: bold;">{system_info.get('torch_gpu_memory_allocated', 0):.1f}G</div>
                        <div style="font-size: 0.7em;">已用显存</div>
                    </div>
                </div>
                <div style="margin-top: 5px; font-size: 0.7em; text-align: center;">
                    {system_info.get('torch_gpu_name', '未知GPU')}
                </div>
            """
        else:
            html += """
                <div style="margin: 10px 0; text-align: center; color: #ff6b6b;">
                    ⚠️ CUDA不可用，模型运行在CPU上
                </div>
            """
        
        # 磁盘信息
        html += f"""
            <div style="margin: 10px 0;">
                <div style="display: flex; justify-content: space-between; font-size: 0.9em;">
                    <span>💾 磁盘</span>
                    <span>{system_info.get('disk_usage', 0):.1f}%</span>
                </div>
                <div style="background: rgba(255,255,255,0.3); border-radius: 10px; height: 6px; margin-top: 5px;">
                    <div style="background: #f9c74f; width: {system_info.get('disk_usage', 0)}%; height: 100%; border-radius: 10px;"></div>
                </div>
            </div>
        """
    
    except Exception as e:
        logger.error(f"生成系统信息HTML失败: {e}")
        html += "<div style='color: #ff6b6b;'>系统信息获取失败</div>"
    
    html += "</div>"
    return html

def get_stats_html():
    """获取统计信息的HTML显示"""
    try:
        run_time = datetime.now() - state.stats['start_time']
        hours = run_time.total_seconds() // 3600
        minutes = (run_time.total_seconds() % 3600) // 60
        
        avg_time = state.stats['total_time'] / max(state.stats['total_requests'], 1)
        current_conv = state.conversations.get(state.current_conversation_id, {})
        
        success_rate = 100
        if state.stats['total_requests'] > 0:
            success_rate = ((state.stats['total_requests'] - state.stats['failed_requests']) / state.stats['total_requests']) * 100
        
        # 获取当前模型和设备信息
        device_info = "CPU"
        if state.model is not None:
            device_info = str(state.model.device)
            if 'cuda' in device_info:
                device_info = f"GPU:{device_info.split(':')[-1]}"
        
        html = f"""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 15px; border-radius: 10px; margin: 10px 0;">
            <h3 style="margin-top: 0; margin-bottom: 15px;">📊 使用统计</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div style="text-align: center;">
                    <div style="font-size: 1.3em; font-weight: bold;">{state.stats['total_requests']}</div>
                    <div style="font-size: 0.8em;">请求数</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.3em; font-weight: bold;">{state.stats['total_tokens']}</div>
                    <div style="font-size: 0.8em;">Token数</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.3em; font-weight: bold;">{avg_time:.1f}s</div>
                    <div style="font-size: 0.8em;">平均响应</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.3em; font-weight: bold;">{success_rate:.1f}%</div>
                    <div style="font-size: 0.8em;">成功率</div>
                </div>
            </div>
            <div style="text-align: center; margin-top: 10px; font-size: 0.8em;">
                运行: {int(hours)}h{int(minutes)}m | 对话: {len(state.conversations)}<br>
                设备: {device_info} | 模型: {state.current_model or '无'}
            </div>
        </div>
        """
        return html
    except Exception as e:
        logger.error(f"生成统计信息HTML失败: {e}")
        return "<div>统计信息加载失败</div>"

# 模型管理
def get_model_info(model_path):
    """获取模型基本信息"""
    try:
        model_name = os.path.basename(model_path)
        
        info = {
            "name": model_name,
            "path": model_path,
            "parameters": "未知",
            "type": "未知",
            "size_gb": "未知"
        }
        
        # 估算参数规模
        if "1.7B" in model_name or "1.7b" in model_name:
            info["parameters"] = "17亿"
            info["size_gb"] = "~3.5GB"
        elif "4B" in model_name or "4b" in model_name:
            info["parameters"] = "40亿"
            info["size_gb"] = "~8GB"
        elif "7B" in model_name or "7b" in model_name:
            info["parameters"] = "70亿"
            info["size_gb"] = "~14GB"
        elif "14B" in model_name or "14b" in model_name:
            info["parameters"] = "140亿"
            info["size_gb"] = "~28GB"
            
        # 判断模型类型
        if "Thinking" in model_name:
            info["type"] = "思考增强型"
        elif "Chat" in model_name:
            info["type"] = "对话优化型"
        elif "Instruct" in model_name:
            info["type"] = "指令调优型"
        else:
            info["type"] = "基础模型"
            
        return info
    except Exception as e:
        logger.error(f"获取模型信息失败: {e}")
        return {
            "name": os.path.basename(model_path),
            "path": model_path,
            "parameters": "未知",
            "type": "未知",
            "size_gb": "未知"
        }

def load_model_with_fallback(model_path, model_display_name):
    """加载模型，带有GPU回退机制"""
    cuda_info = check_cuda_availability()
    
    if not cuda_info['available']:
        logger.warning("CUDA不可用，将加载模型到CPU")
        return load_model_to_cpu(model_path, model_display_name)
    
    # 检查GPU内存是否足够
    available_memory_gb = cuda_info['total_memory'] - cuda_info['memory_allocated']
    
    # 尝试GPU加载
    try:
        logger.info(f"尝试加载模型到GPU: {model_display_name}")
        
        # 清除GPU缓存
        torch.cuda.empty_cache()
        gc.collect()
        
        # 加载分词器
        state.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # 方法1: 使用device_map="auto"（推荐用于多GPU）
        try:
            logger.info("尝试方法1: device_map='auto'")
            state.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        except Exception as e1:
            logger.warning(f"方法1失败: {e1}")
            
            # 方法2: 使用device_map="cuda"
            try:
                logger.info("尝试方法2: device_map='cuda'")
                state.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="cuda",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            except Exception as e2:
                logger.warning(f"方法2失败: {e2}")
                
                # 方法3: 手动指定设备
                try:
                    logger.info("尝试方法3: 手动指定设备")
                    state.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        device_map=None,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    )
                    state.model = state.model.to('cuda')
                except Exception as e3:
                    logger.error(f"所有GPU加载方法都失败: {e3}")
                    raise
        
        state.current_model = model_display_name
        
        # 验证模型是否在GPU上
        if hasattr(state.model, 'device'):
            device_str = str(state.model.device)
            logger.info(f"模型加载成功，设备: {device_str}")
            
            if 'cuda' not in device_str:
                logger.warning("模型未在GPU上运行，将回退到CPU")
                return load_model_to_cpu(model_path, model_display_name)
        else:
            # 检查第一个参数的设备
            for param in state.model.parameters():
                device_str = str(param.device)
                logger.info(f"模型参数设备: {device_str}")
                break
        
        # 保存配置
        config = load_config()
        config["last_model"] = model_display_name
        save_config(config)
        
        system_info = get_system_info()
        device_info = f"运行设备: {state.model.device}"
        
        if system_info.get('cuda_available', False):
            device_info += f" | GPU内存占用: {system_info.get('torch_gpu_memory_allocated', 0):.2f} GB"
        
        logger.info(f"模型GPU加载成功: {model_display_name}")
        return f"✅ **{model_display_name}** 加载成功！\n\n{device_info}", get_model_display_info(model_path)
        
    except Exception as e:
        logger.error(f"GPU加载失败: {e}")
        # 回退到CPU
        return load_model_to_cpu(model_path, model_display_name)

def load_model_to_cpu(model_path, model_display_name):
    """加载模型到CPU"""
    try:
        logger.info(f"加载模型到CPU: {model_display_name}")
        
        # 加载分词器
        state.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # 加载模型到CPU
        state.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # CPU上使用float32
            device_map=None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        state.model = state.model.to('cpu')
        state.current_model = model_display_name
        
        # 保存配置
        config = load_config()
        config["last_model"] = model_display_name
        save_config(config)
        
        logger.info(f"模型CPU加载成功: {model_display_name}")
        return f"✅ **{model_display_name}** 已加载到CPU", get_model_display_info(model_path)
    except Exception as e:
        logger.error(f"CPU加载失败: {e}")
        return f"❌ 模型加载失败: {str(e)}", ""

def load_model(model_path, model_display_name):
    """加载模型（主函数）"""
    with state.model_lock:
        try:
            logger.info(f"开始加载模型: {model_display_name}")
            
            # 如果已经有模型加载，先卸载
            if state.model is not None:
                unload_model()
            
            # 检查模型路径是否存在
            if not os.path.exists(model_path):
                return f"❌ 模型路径不存在: {model_path}", ""
            
            # 尝试加载模型，带有GPU回退机制
            return load_model_with_fallback(model_path, model_display_name)
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return f"❌ 模型加载失败: {str(e)}", ""

def unload_model():
    """卸载模型 - 增强版本"""
    with state.model_lock:
        if state.model is not None:
            model_name = state.current_model
            
            # 记录卸载前的内存使用
            if torch.cuda.is_available():
                allocated_before = torch.cuda.memory_allocated() / (1024**3)
                reserved_before = torch.cuda.memory_reserved() / (1024**3)
                logger.info(f"卸载前 - 已分配: {allocated_before:.2f}GB, 保留: {reserved_before:.2f}GB")
            
            try:
                # 首先将模型移回CPU（如果它在GPU上）
                if hasattr(state.model, 'device') and 'cuda' in str(state.model.device):
                    try:
                        state.model = state.model.cpu()
                        logger.info("模型已移动到CPU")
                        # 给GPU一些时间来释放内存
                        time.sleep(0.5)
                    except Exception as e:
                        logger.warning(f"移动模型到CPU失败: {e}")
                
                # 删除模型和分词器
                del state.model
                del state.tokenizer
                
            except Exception as e:
                logger.warning(f"删除模型引用时出错: {e}")
            finally:
                state.model = None
                state.tokenizer = None
                state.current_model = None
            
            # 强制垃圾回收
            for i in range(3):  # 多次垃圾回收
                gc.collect()
                time.sleep(0.1)
            
            # 清理GPU内存
            if torch.cuda.is_available():
                try:
                    # 清空缓存
                    torch.cuda.empty_cache()
                    
                    # 同步所有设备
                    for i in range(torch.cuda.device_count()):
                        torch.cuda.synchronize(i)
                    
                    # 再次清空缓存
                    torch.cuda.empty_cache()
                    
                    # 记录卸载后的内存使用
                    allocated_after = torch.cuda.memory_allocated() / (1024**3)
                    reserved_after = torch.cuda.memory_reserved() / (1024**3)
                    
                    memory_freed = allocated_before - allocated_after
                    logger.info(f"卸载后 - 已分配: {allocated_after:.2f}GB, 保留: {reserved_after:.2f}GB")
                    logger.info(f"释放内存: {memory_freed:.2f}GB")
                    
                    # 如果释放的内存很少，记录警告
                    if memory_freed < 0.1:
                        logger.warning("模型卸载后释放的内存很少，可能仍有引用未清除")
                        
                except Exception as e:
                    logger.warning(f"重置GPU内存失败: {e}")
            
            logger.info(f"模型已卸载: {model_name}")
            return "✅ 模型已卸载", ""
        return "ℹ️ 没有加载的模型", ""

def get_model_display_info(model_path):
    """获取模型显示信息"""
    info = get_model_info(model_path)
    
    # 获取当前设备信息
    device_info = "CPU"
    if state.model is not None:
        device_info = str(state.model.device)
    
    cuda_info = check_cuda_availability()
    gpu_status = "✅ 可用" if cuda_info['available'] else "❌ 不可用"
    
    return f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 10px; margin: 10px 0;">
        <h3 style="margin-top: 0; margin-bottom: 10px;">📝 模型信息</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
            <div style="text-align: center;">
                <div style="font-size: 1.2em; font-weight: bold;">{info['parameters']}</div>
                <div style="font-size: 0.8em;">参数规模</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.2em; font-weight: bold;">{info['type']}</div>
                <div style="font-size: 0.8em;">模型类型</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.2em; font-weight: bold;">{info['size_gb']}</div>
                <div style="font-size: 0.8em;">预估大小</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.2em; font-weight: bold;">{device_info}</div>
                <div style="font-size: 0.8em;">运行设备</div>
            </div>
        </div>
        <div style="margin-top: 10px; font-size: 0.8em; text-align: center;">
            GPU状态: {gpu_status} | 路径: {os.path.basename(model_path)}
        </div>
    </div>
    """

# 智能历史消息截断
def smart_truncate_messages(messages, max_tokens=CONFIG["max_input_tokens"]):
    """智能截断历史消息，保留最重要的上下文"""
    if not state.tokenizer:
        return messages
    
    # 计算当前消息的token数量
    def count_tokens(msg_list):
        total_tokens = 0
        for msg in msg_list:
            try:
                tokens = state.tokenizer.encode(msg["content"], add_special_tokens=False)
                total_tokens += len(tokens)
            except:
                # 如果编码失败，使用字符数估算
                total_tokens += len(msg["content"]) // 4
        return total_tokens
    
    current_tokens = count_tokens(messages)
    
    # 如果token数在限制内，直接返回
    if current_tokens <= max_tokens:
        return messages
    
    logger.info(f"输入token数({current_tokens})超过限制({max_tokens})，进行智能截断")
    
    # 智能截断策略：
    # 1. 始终保留系统消息（如果有）
    # 2. 优先保留最近的对话
    # 3. 如果仍然超长，逐步删除最旧的消息
    
    truncated_messages = messages.copy()
    
    # 首先尝试删除最旧的消息（除了系统消息）
    while count_tokens(truncated_messages) > max_tokens and len(truncated_messages) > 1:
        # 跳过系统消息（如果有）
        if truncated_messages[0].get("role") == "system":
            if len(truncated_messages) > 2:
                # 删除第一个非系统消息
                truncated_messages.pop(1)
            else:
                break
        else:
            # 删除第一条消息
            truncated_messages.pop(0)
    
    # 如果仍然超长，尝试截断单条消息的内容
    if count_tokens(truncated_messages) > max_tokens:
        for i, msg in enumerate(truncated_messages):
            if len(msg["content"]) > 500:  # 只截断长消息
                truncated_messages[i]["content"] = msg["content"][:500] + "...(已截断)"
                if count_tokens(truncated_messages) <= max_tokens:
                    break
    
    final_tokens = count_tokens(truncated_messages)
    logger.info(f"截断后token数: {final_tokens}, 保留了 {len(truncated_messages)} 条消息")
    
    return truncated_messages

# 对话生成 - 修复版本
def chat_stream(message, conversation_id):
    """流式聊天函数 - 修复输入截断问题"""
    if state.model is None or state.tokenizer is None:
        yield get_conversation_history(conversation_id) + [[message, "⚠️ 请先加载模型！"]], get_stats_html(), get_system_info_html()
        return
    
    # 设置生成状态
    state.is_generating = True
    state.generation_stop_event = threading.Event()
    
    try:
        # 获取对话历史
        conversation = state.conversations[conversation_id]
        
        # 使用max_history参数限制历史消息数量
        max_history = state.current_params['max_history']
        all_messages = conversation["messages"]
        
        # 计算要保留的消息数量（最近的n轮对话）
        messages_to_keep = min(len(all_messages), max_history * 2)
        recent_messages = all_messages[-messages_to_keep:] if messages_to_keep > 0 else []
        
        # 添加当前用户输入
        clean_message = truncate_text(clean_text(message))
        messages_for_model = recent_messages + [{"role": "user", "content": clean_message}]
        
        # 智能截断消息
        messages_for_model = smart_truncate_messages(messages_for_model)
        
        # 应用聊天模板
        try:
            text = state.tokenizer.apply_chat_template(
                messages_for_model,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=getattr(state.tokenizer, 'enable_thinking', False)
            )
        except Exception as e:
            logger.warning(f"应用聊天模板失败: {e}")
            # 回退到简单格式
            text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages_for_model]) + "\nassistant:"
        
        # 检查输入长度（使用更宽松的限制）
        input_ids = state.tokenizer.encode(text)
        max_input_length = CONFIG["max_input_tokens"] - CONFIG["reserved_tokens"]
        
        if len(input_ids) > max_input_length:
            logger.info(f"输入过长({len(input_ids)} tokens)，进行截断")
            # 保留更多的上下文，从后面截断可能会破坏格式，所以从前面删除一些历史
            # 但为了保持完整性，我们使用tokenizer的截断功能
            text = state.tokenizer.decode(input_ids[:max_input_length])
            # 添加截断标记
            if not text.endswith("..."):
                text += "...(上下文已截断)"
        
        # 将输入移动到模型所在的设备
        model_device = state.model.device if hasattr(state.model, 'device') else 'cpu'
        model_inputs = state.tokenizer([text], return_tensors="pt").to(model_device)

        # 创建流式生成器
        streamer = TextIteratorStreamer(state.tokenizer, skip_prompt=True, timeout=CONFIG["default_timeout"])
        
        # 生成参数
        generation_kwargs = dict(
            **model_inputs,
            max_new_tokens=state.current_params['max_new_tokens'],
            temperature=state.current_params['temperature'],
            top_p=state.current_params['top_p'],
            top_k=state.current_params.get('top_k', 50),
            do_sample=state.current_params.get('do_sample', True),
            pad_token_id=state.tokenizer.eos_token_id,
            repetition_penalty=state.current_params['repetition_penalty'],
            streamer=streamer
        )

        # 在单独线程中生成
        start_time = time.time()
        thread = threading.Thread(target=state.model.generate, kwargs=generation_kwargs)
        thread.daemon = True  # 设置为守护线程
        thread.start()
        
        # 流式输出
        generated_text = ""
        thinking_content = ""
        is_thinking = True
        
        for new_text in streamer:
            # 检查停止信号
            if state.generation_stop_event and state.generation_stop_event.is_set():
                logger.info("生成被用户停止")
                break
                
            generated_text += new_text
            
            # 尝试解析思考内容
            if is_thinking and "</think>" in generated_text:
                parts = generated_text.split("</think>", 1)
                thinking_content = parts[0] + "</think>"
                generated_text = parts[1] if len(parts) > 1 else ""
                is_thinking = False
            
            # 组合输出
            if is_thinking:
                partial_response = f"🤔 **思考中...**\n\n{thinking_content + generated_text}"
            else:
                if thinking_content:
                    partial_response = f"<details style='margin-bottom: 10px;'><summary>🧠 思考过程</summary>\n\n{thinking_content}\n\n</details>\n\n{generated_text}"
                else:
                    partial_response = generated_text
            
            # 清理文本
            partial_response = clean_text(partial_response)
            
            # 更新对话显示
            current_history = get_conversation_history(conversation_id) + [[clean_message, partial_response]]
            
            yield current_history, get_stats_html(), get_system_info_html()
        
        generation_time = time.time() - start_time
        
        # 如果被停止，添加停止标记
        if state.generation_stop_event and state.generation_stop_event.is_set():
            generated_text += "\n\n---\n*生成已被用户停止*"
        
        # 计算生成的token数量
        output_ids = state.tokenizer.encode(generated_text)
        tokens_generated = len(output_ids)
        
        # 更新统计信息
        state.stats['total_requests'] += 1
        state.stats['total_tokens'] += tokens_generated
        state.stats['total_time'] += generation_time
        
        # 保存到对话历史
        conversation["messages"].append({"role": "user", "content": clean_message})
        conversation["messages"].append({"role": "assistant", "content": generated_text})
        conversation["model_used"] = state.current_model
        
        # 更新对话标题（如果是第一条消息）
        if len(conversation["messages"]) == 2:
            conversation["title"] = clean_message[:20] + "..." if len(clean_message) > 20 else clean_message
        
        save_conversation(conversation_id)
        
        # 最终输出
        if thinking_content:
            final_response = f"<details style='margin-bottom: 10px;'><summary>🧠 思考过程</summary>\n\n{thinking_content}\n\n</details>\n\n{generated_text}"
        else:
            final_response = generated_text
        
        # 添加响应时间信息
        if not (state.generation_stop_event and state.generation_stop_event.is_set()):
            final_response += f"\n\n---\n*响应时间: {generation_time:.2f}s | 生成Token: {tokens_generated}*"
        
        # 最终更新
        final_history = get_conversation_history(conversation_id)
        
        yield final_history, get_stats_html(), get_system_info_html()
        
    except Exception as e:
        logger.error(f"生成响应时出错: {e}")
        state.stats['failed_requests'] += 1
        error_msg = f"❌ 生成响应时出错: {str(e)}"
        yield get_conversation_history(conversation_id) + [[message, error_msg]], get_stats_html(), get_system_info_html()
    finally:
        state.is_generating = False
        state.generation_stop_event = None

def stop_generation():
    """停止生成"""
    if state.generation_stop_event and state.is_generating:
        state.generation_stop_event.set()
        return "🛑 正在停止生成..."
    return "ℹ️ 没有正在进行的生成"

# 参数管理
def update_params(max_new_tokens, temperature, top_p, repetition_penalty, max_history, top_k, do_sample):
    """更新参数"""
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

def reset_params():
    """重置参数为默认值"""
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

# 模型配置管理
def add_model_path(model_name, model_path):
    """添加模型路径"""
    if model_name and model_path:
        MODEL_PATHS[model_name] = model_path
        save_models_config(MODEL_PATHS)
        return f"✅ 已添加模型: {model_name}", gr.Dropdown(choices=list(MODEL_PATHS.keys()))
    return "❌ 模型名称和路径不能为空", gr.Dropdown(choices=list(MODEL_PATHS.keys()))

def remove_model_path(model_name):
    """移除模型路径"""
    if model_name in MODEL_PATHS:
        del MODEL_PATHS[model_name]
        save_models_config(MODEL_PATHS)
        return f"✅ 已移除模型: {model_name}", gr.Dropdown(choices=list(MODEL_PATHS.keys()))
    return "❌ 模型不存在", gr.Dropdown(choices=list(MODEL_PATHS.keys()))

# 强制内存清理
def force_clean_memory():
    """强制清理内存"""
    try:
        # 强制垃圾回收
        gc.collect()
        
        # 清理GPU内存
        if torch.cuda.is_available():
            before_allocated = torch.cuda.memory_allocated() / (1024**3)
            before_reserved = torch.cuda.memory_reserved() / (1024**3)
            
            # 清空缓存
            torch.cuda.empty_cache()
            
            # 尝试重置设备
            try:
                for i in range(torch.cuda.device_count()):
                    torch.cuda.synchronize(i)
                    torch.cuda.empty_cache()
            except:
                pass
            
            after_allocated = torch.cuda.memory_allocated() / (1024**3)
            after_reserved = torch.cuda.memory_reserved() / (1024**3)
            
            logger.info(f"内存清理完成 - 已分配: {before_allocated:.2f}GB -> {after_allocated:.2f}GB, "
                       f"保留: {before_reserved:.2f}GB -> {after_reserved:.2f}GB")
            
            return f"🧹 内存清理完成！释放了 {before_allocated - after_allocated:.2f}GB 显存"
        else:
            return "🧹 CPU内存清理完成"
            
    except Exception as e:
        logger.error(f"内存清理失败: {e}")
        return f"❌ 内存清理失败: {str(e)}"

# 初始化应用
def initialize_app():
    """初始化应用"""
    try:
        # 加载对话
        load_conversations()
        
        # 加载配置
        config = load_config()
        
        # 创建默认对话（如果没有对话）
        if not state.conversations:
            create_new_conversation()
        else:
            # 恢复上次对话或使用第一个对话
            state.current_conversation_id = config.get("last_conversation") or list(state.conversations.keys())[0]
        
        # 自动加载上次使用的模型
        last_model = config.get("last_model")
        if last_model and last_model in MODEL_PATHS:
            logger.info(f"自动加载上次使用的模型: {last_model}")
            load_model(MODEL_PATHS[last_model], last_model)
        
        logger.info("应用初始化完成")
    except Exception as e:
        logger.error(f"应用初始化失败: {e}")

# 工具函数
def get_conversation_dropdown_options():
    """获取对话下拉选项"""
    options = []
    for conv_id, conversation in sorted(
        state.conversations.items(),
        key=lambda x: x[1].get("updated_at", ""),
        reverse=True
    ):
        title = conversation["title"]
        message_count = len(conversation["messages"]) // 2
        display_text = f"{title} ({message_count}条消息)"
        options.append((display_text, conv_id))
    return options

def get_current_conversation_info():
    """获取当前对话信息"""
    if state.current_conversation_id in state.conversations:
        conv = state.conversations[state.current_conversation_id]
        model_info = f" | 模型: {conv.get('model_used', '未记录')}" if conv.get('model_used') else ""
        return f"### 📝 当前对话: {conv['title']} ({len(conv['messages'])//2} 条消息{model_info})"
    return "### 📝 当前对话: 无"

# 创建Gradio界面
def create_interface():
    """创建Gradio界面"""
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate"
        ),
        title="大模型对话系统",
        css="""
        .gradio-container {
            max-width: 95% !important;
        }
        .conversation-item {
            padding: 8px 12px;
            margin: 2px 0;
            border-radius: 6px;
            cursor: pointer;
        }
        .conversation-item:hover {
            background: rgba(0,0,0,0.05);
        }
        .conversation-active {
            background: rgba(59, 130, 246, 0.1);
            border-left: 3px solid #3b82f6;
        }
        .warning-text {
            color: #e74c3c;
            font-size: 0.9em;
        }
        .success-text {
            color: #27ae60;
            font-size: 0.9em;
        }
        """
    ) as demo:
        gr.Markdown("""
        # 🤖 大模型对话系统
        **基于 Qwen 系列模型的智能对话平台**
        """)
        
        # 显示CUDA状态
        cuda_status = check_cuda_availability()
        if cuda_status['available']:
            gr.Markdown(f"### 🎮 GPU状态: ✅ 可用 - {cuda_status['device_name']} ({cuda_status['device_count']}个GPU)")
        else:
            gr.Markdown("### 🎮 GPU状态: ❌ 不可用 - 将使用CPU运行")
        
        # 初始化对话选项
        conversation_options = get_conversation_dropdown_options()
        current_conversation_info = get_current_conversation_info()
        
        with gr.Row(equal_height=False):
            # 左侧控制面板
            with gr.Column(scale=1, min_width=300):
                # 对话管理区域
                with gr.Group():
                    gr.Markdown("### 💬 对话管理")
                    with gr.Row():
                        new_convo_btn = gr.Button("🆕 新建对话", variant="primary", size="sm")
                        refresh_convos_btn = gr.Button("🔄 刷新", variant="secondary", size="sm")
                    
                    # 对话选择下拉菜单
                    conversation_dropdown = gr.Dropdown(
                        choices=[opt[0] for opt in conversation_options],
                        value=conversation_options[0][0] if conversation_options else "",
                        label="选择对话",
                        interactive=True,
                        filterable=True
                    )
                    conversation_state = gr.State(state.current_conversation_id)
                    
                    # 对话操作按钮
                    with gr.Row():
                        delete_convo_btn = gr.Button("🗑️ 删除对话", variant="stop", size="sm")
                        export_convo_btn = gr.Button("📤 导出对话", variant="secondary", size="sm")
                
                # 模型控制区域
                with gr.Group():
                    gr.Markdown("### 🚀 模型控制")
                    model_dropdown = gr.Dropdown(
                        choices=list(MODEL_PATHS.keys()),
                        value=state.current_model or (list(MODEL_PATHS.keys())[0] if MODEL_PATHS else ""),
                        label="选择模型",
                        filterable=True
                    )
                    
                    with gr.Row():
                        load_btn = gr.Button("🔄 加载模型", variant="primary", scale=1)
                        unload_btn = gr.Button("🗑️ 卸载模型", variant="secondary", scale=1)
                    
                    load_status = gr.Markdown("👆 请选择并加载模型")
                    model_info_html = gr.HTML()
                    
                    # 模型管理扩展
                    with gr.Accordion("🔧 模型管理", open=False):
                        with gr.Row():
                            new_model_name = gr.Textbox(label="模型名称", placeholder="输入模型显示名称")
                            new_model_path = gr.Textbox(label="模型路径", placeholder="输入模型本地路径")
                        
                        with gr.Row():
                            add_model_btn = gr.Button("➕ 添加模型", size="sm")
                            remove_model_btn = gr.Button("➖ 移除模型", size="sm")
                
                # 参数设置区域
                with gr.Group():
                    gr.Markdown("### ⚙️ 生成参数")
                    
                    with gr.Row():
                        max_new_tokens = gr.Slider(512, 8192, value=state.current_params['max_new_tokens'], step=256, label="生成长度")
                        temperature = gr.Slider(0.1, 2.0, value=state.current_params['temperature'], step=0.1, label="温度")
                    
                    with gr.Row():
                        top_p = gr.Slider(0.1, 1.0, value=state.current_params['top_p'], step=0.1, label="Top-P")
                        top_k = gr.Slider(1, 100, value=state.current_params.get('top_k', 50), step=1, label="Top-K")
                    
                    with gr.Row():
                        repetition_penalty = gr.Slider(1.0, 2.0, value=state.current_params['repetition_penalty'], step=0.1, label="重复惩罚")
                        max_history_slider = gr.Slider(1, 50, value=state.current_params['max_history'], step=1, label="对话记忆轮数")
                    
                    do_sample = gr.Checkbox(value=state.current_params.get('do_sample', True), label="随机采样")
                    
                    with gr.Row():
                        update_btn = gr.Button("💾 保存参数", variant="primary", scale=1)
                        reset_btn = gr.Button("🔄 重置默认", variant="secondary", scale=1)
                    
                    param_status = gr.Markdown("✅ 参数已就绪")
                
                # 系统信息区域
                with gr.Group():
                    gr.Markdown("### 📊 系统状态")
                    with gr.Row():
                        refresh_sys_btn = gr.Button("🔄 刷新状态", variant="secondary", size="sm")
                        stop_btn = gr.Button("⏹️ 停止生成", variant="stop", size="sm")
                        clean_memory_btn = gr.Button("🧹 清理内存", variant="secondary", size="sm")
                    
                    system_info_html = gr.HTML(get_system_info_html())
                    stats_html = gr.HTML(get_stats_html())
            
            # 右侧聊天区域
            with gr.Column(scale=2, min_width=500):
                current_conversation_display = gr.Markdown(current_conversation_info)
                
                chatbot = gr.Chatbot(
                    value=get_conversation_history(state.current_conversation_id),
                    label="💬 智能对话",
                    height=500,
                    type="tuples",
                    show_copy_button=True,
                    avatar_images=(
                        "https://cdn-icons-png.flaticon.com/512/149/149071.png",  # 用户头像
                        "https://cdn-icons-png.flaticon.com/512/4712/4712035.png"  # 助手头像
                    ),
                    placeholder="对话记录将显示在这里..."
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="",
                        placeholder="请输入您的问题或指令...（按Enter发送，Shift+Enter换行）",
                        lines=2,
                        max_lines=5,
                        scale=4,
                        show_label=False
                    )
                    submit_btn = gr.Button("🚀 发送", variant="primary", scale=1)
                
                # 快捷操作按钮
                with gr.Row():
                    gr.Markdown("**💡 快捷操作:**")
                    quick_clear = gr.Button("🗑️ 清空当前", size="sm")
                    quick_example1 = gr.Button("👋 打个招呼", size="sm")
                    quick_example2 = gr.Button("📝 写段代码", size="sm")
                    quick_example3 = gr.Button("🤔 解释概念", size="sm")
                
                # 消息计数器
                message_counter = gr.HTML(f"""
                <div style="text-align: right; font-size: 0.8em; color: #666; margin-top: 5px;">
                    当前对话: {len(state.conversations.get(state.current_conversation_id, {}).get('messages', []))//2} 条消息
                </div>
                """)
        
        # 事件处理函数
        def handle_load_model(model_name):
            if model_name in MODEL_PATHS:
                model_path = MODEL_PATHS[model_name]
                load_result, model_html = load_model(model_path, model_name)
                return load_result, model_html, get_stats_html(), get_system_info_html()
            return "❌ 请选择有效的模型", "", get_stats_html(), get_system_info_html()
        
        def handle_unload_model():
            unload_result, model_html = unload_model()
            return unload_result, model_html, get_stats_html(), get_system_info_html()
        
        def handle_submit(message, conversation_id):
            if not message.strip():
                yield get_conversation_history(conversation_id), get_stats_html(), get_system_info_html(), "", get_current_conversation_info()
                return
                
            for updated_history, stats_html_content, system_html_content in chat_stream(message, conversation_id):
                yield updated_history, stats_html_content, system_html_content, "", get_current_conversation_info()
        
        def handle_new_conversation():
            new_id = create_new_conversation()
            # 保存配置
            config = load_config()
            config["last_conversation"] = new_id
            save_config(config)
            
            # 更新下拉选项
            options = get_conversation_dropdown_options()
            new_conv_title = f"新对话 (0条消息)"
            
            return (
                gr.Dropdown(choices=[opt[0] for opt in options], value=new_conv_title),
                get_conversation_history(new_id),
                get_current_conversation_info(),
                new_id
            )
        
        def handle_refresh_conversations():
            load_conversations()
            options = get_conversation_dropdown_options()
            current_title = None
            for title, conv_id in options:
                if conv_id == state.current_conversation_id:
                    current_title = title
                    break
            
            return gr.Dropdown(
                choices=[opt[0] for opt in options], 
                value=current_title or (options[0][0] if options else "")
            )
        
        def handle_conversation_change(selected_title):
            """处理对话切换"""
            # 根据标题找到对应的对话ID
            for title, conv_id in get_conversation_dropdown_options():
                if title == selected_title:
                    state.current_conversation_id = conv_id
                    # 保存配置
                    config = load_config()
                    config["last_conversation"] = conv_id
                    save_config(config)
                    return (
                        get_conversation_history(conv_id),
                        get_current_conversation_info(),
                        conv_id
                    )
            return gr.update(), gr.update(), state.current_conversation_id
        
        def handle_delete_conversation():
            """删除当前对话"""
            if state.current_conversation_id:
                success = delete_conversation(state.current_conversation_id)
                if success:
                    options = get_conversation_dropdown_options()
                    current_title = options[0][0] if options else "新对话"
                    return (
                        gr.Dropdown(choices=[opt[0] for opt in options], value=current_title),
                        get_conversation_history(state.current_conversation_id),
                        get_current_conversation_info(),
                        state.current_conversation_id,
                        "✅ 对话已删除"
                    )
            return gr.update(), gr.update(), gr.update(), gr.update(), "❌ 删除对话失败"
        
        def handle_export_conversation():
            """导出当前对话"""
            if state.current_conversation_id in state.conversations:
                conversation = state.conversations[state.current_conversation_id]
                export_data = {
                    "title": conversation["title"],
                    "model_used": conversation.get("model_used", "未知"),
                    "created_at": conversation["created_at"],
                    "messages": conversation["messages"]
                }
                filename = f"{conversation['title']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                return gr.File(value=json.dumps(export_data, ensure_ascii=False, indent=2), label=filename, visible=True)
            return gr.File(visible=False)
        
        def handle_clear_current_conversation():
            if state.current_conversation_id in state.conversations:
                state.conversations[state.current_conversation_id]["messages"] = []
                save_conversation(state.current_conversation_id)
                return [], get_current_conversation_info()
            return [], get_current_conversation_info()
        
        def handle_quick_example(example_type):
            examples = {
                "greeting": "你好！请介绍一下你自己。",
                "code": "请用Python写一个快速排序算法，并添加详细注释。",
                "explain": "请用通俗易懂的方式解释什么是机器学习。"
            }
            return examples.get(example_type, "你好！")
        
        # 绑定事件
        load_btn.click(
            fn=handle_load_model,
            inputs=[model_dropdown],
            outputs=[load_status, model_info_html, stats_html, system_info_html]
        )
        
        unload_btn.click(
            fn=handle_unload_model,
            outputs=[load_status, model_info_html, stats_html, system_info_html]
        )
        
        # 提交消息
        submit_event = msg.submit(
            fn=handle_submit,
            inputs=[msg, conversation_state],
            outputs=[chatbot, stats_html, system_info_html, msg, current_conversation_display]
        )
        
        submit_btn.click(
            fn=handle_submit,
            inputs=[msg, conversation_state],
            outputs=[chatbot, stats_html, system_info_html, msg, current_conversation_display]
        )
        
        stop_btn.click(
            fn=stop_generation,
            outputs=[load_status]
        )
        
        # 内存清理
        clean_memory_btn.click(
            fn=force_clean_memory,
            outputs=[load_status]
        )
        
        # 对话管理
        new_convo_btn.click(
            fn=handle_new_conversation,
            outputs=[conversation_dropdown, chatbot, current_conversation_display, conversation_state]
        )
        
        refresh_convos_btn.click(
            fn=handle_refresh_conversations,
            outputs=[conversation_dropdown]
        )
        
        delete_convo_btn.click(
            fn=handle_delete_conversation,
            outputs=[conversation_dropdown, chatbot, current_conversation_display, conversation_state, load_status]
        )
        
        export_convo_btn.click(
            fn=handle_export_conversation,
            outputs=[gr.File(visible=True)]
        )
        
        # 对话切换事件
        conversation_dropdown.change(
            fn=handle_conversation_change,
            inputs=[conversation_dropdown],
            outputs=[chatbot, current_conversation_display, conversation_state]
        )
        
        # 模型管理
        add_model_btn.click(
            fn=add_model_path,
            inputs=[new_model_name, new_model_path],
            outputs=[load_status, model_dropdown]
        )
        
        remove_model_btn.click(
            fn=remove_model_path,
            inputs=[model_dropdown],
            outputs=[load_status, model_dropdown]
        )
        
        # 参数管理
        update_btn.click(
            fn=update_params,
            inputs=[max_new_tokens, temperature, top_p, repetition_penalty, max_history_slider, top_k, do_sample],
            outputs=[param_status]
        )
        
        reset_btn.click(
            fn=reset_params,
            outputs=[max_new_tokens, temperature, top_p, repetition_penalty, max_history_slider, top_k, do_sample, param_status]
        )
        
        # 系统状态
        refresh_sys_btn.click(
            fn=lambda: [get_system_info_html(), get_stats_html()],
            outputs=[system_info_html, stats_html]
        )
        
        # 快捷操作
        quick_clear.click(
            fn=handle_clear_current_conversation,
            outputs=[chatbot, current_conversation_display]
        )
        
        quick_example1.click(
            fn=lambda: handle_quick_example("greeting"),
            outputs=[msg]
        )
        
        quick_example2.click(
            fn=lambda: handle_quick_example("code"),
            outputs=[msg]
        )
        
        quick_example3.click(
            fn=lambda: handle_quick_example("explain"),
            outputs=[msg]
        )
        
        return demo

# 启动应用
if __name__ == "__main__":
    # 初始化应用
    initialize_app()
    
    # 创建界面
    demo = create_interface()
    
    # 启动服务
    try:
        demo.queue(
            max_size=20,
            api_open=False
        ).launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            inbrowser=True,
            show_error=True,
            quiet=False,
            debug=False
        )
    except Exception as e:
        logger.error(f"启动应用失败: {e}")
        print(f"启动失败: {e}")