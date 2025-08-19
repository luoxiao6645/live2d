# -*- coding: utf-8 -*-
"""
优化后的Live2D AI助手服务器
集成安全管理器、改进错误处理和性能优化
"""

import os
import re
import requests
import logging
import json
import tempfile
import asyncio
import threading
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, Any, List
from functools import wraps
from flask import Flask, jsonify, send_from_directory, request, Response, stream_with_context, g
from flask_cors import CORS
import edge_tts
from ollama_client import OllamaClient

# 导入安全模块
try:
    from security.security_manager import SecurityManager, SecurityConfig
    from security.secure_flask_app import create_secure_app
    SECURITY_AVAILABLE = True
    logging.info("安全模块加载成功")
except ImportError as e:
    SECURITY_AVAILABLE = False
    logging.warning(f"安全模块加载失败: {e}")
    logging.warning("将使用基础安全功能")

# 导入优化的动画控制器
try:
    from emotion_system.optimized_advanced_animation_controller import OptimizedAdvancedAnimationController
    from emotion_system.optimized_animation_sequencer import OptimizedAnimationSequencer
    OPTIMIZED_ANIMATION_AVAILABLE = True
    logging.info("优化动画控制器加载成功")
except ImportError as e:
    OPTIMIZED_ANIMATION_AVAILABLE = False
    logging.warning(f"优化动画控制器加载失败: {e}")
    # 回退到原始版本
    try:
        from emotion_system import AdvancedAnimationController
        EMOTION_SYSTEM_AVAILABLE = True
        logging.info("原始情感系统模块加载成功")
    except ImportError as e2:
        EMOTION_SYSTEM_AVAILABLE = False
        logging.warning(f"情感系统模块完全加载失败: {e2}")

# 导入RAG相关模块
try:
    from rag_manager import RAGManager
    from document_processor import DocumentProcessor
    RAG_AVAILABLE = True
    logging.info("RAG模块加载成功")
except ImportError as e:
    RAG_AVAILABLE = False
    logging.warning(f"RAG模块加载失败: {e}")
    logging.warning("请安装必要的依赖项: pip install langchain langchain-community chromadb sentence-transformers")

# 导入多模态系统模块
try:
    from multimodal_system import MultimodalManager
    MULTIMODAL_SYSTEM_AVAILABLE = True
    logging.info("多模态系统模块加载成功")
except ImportError as e:
    MULTIMODAL_SYSTEM_AVAILABLE = False
    logging.warning(f"多模态系统模块加载失败: {e}")
    logging.warning("多模态系统功能将不可用")

# 导入语音情感系统模块
try:
    from voice_emotion_system import VoiceEmotionManager
    VOICE_EMOTION_SYSTEM_AVAILABLE = True
    logging.info("语音情感系统模块加载成功")
except ImportError as e:
    VOICE_EMOTION_SYSTEM_AVAILABLE = False
    logging.warning(f"语音情感系统模块加载失败: {e}")
    logging.warning("语音情感系统功能将不可用")

# 高级RAG系统模块（可选功能）
ADVANCED_RAG_SYSTEM_AVAILABLE = False
try:
    from advanced_rag_system import AdvancedRAGManager
    ADVANCED_RAG_SYSTEM_AVAILABLE = True
    logging.info("高级RAG系统模块加载成功")
except ImportError:
    logging.info("高级RAG系统模块未安装，使用基础RAG功能")

# 导入语音识别模块
try:
    # 先尝试导入必要的依赖项
    import numpy
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import librosa

    # 如果依赖项存在，则导入语音识别模块
    from voice_recognition import register_voice_routes, get_recognizer
    VOICE_RECOGNITION_AVAILABLE = True
    logging.info("语音识别模块已加载，所有依赖项已安装")
except ImportError as e:
    VOICE_RECOGNITION_AVAILABLE = False
    logging.warning(f"语音识别模块加载失败: {e}")
    logging.warning("请安装必要的依赖项: pip install numpy torch transformers librosa")

# ======== 配置部分 ========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live2d_server.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 自定义JSON编码器，处理枚举类型
class CustomJSONEncoder(json.JSONEncoder):
    """
    自定义JSON编码器，支持枚举类型序列化
    """
    def default(self, obj):
        """
        处理特殊对象类型的序列化
        
        Args:
            obj: 要序列化的对象
            
        Returns:
            序列化后的值
        """
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def convert_enum_keys(obj):
    """
    递归转换字典中的枚举键为字符串
    
    Args:
        obj: 要转换的对象
        
    Returns:
        转换后的对象
    """
    if isinstance(obj, dict):
        new_dict = {}
        for key, value in obj.items():
            # 转换枚举键为字符串
            if isinstance(key, Enum):
                new_key = key.value
            else:
                new_key = key
            # 递归处理值
            new_dict[new_key] = convert_enum_keys(value)
        return new_dict
    elif isinstance(obj, list):
        return [convert_enum_keys(item) for item in obj]
    elif isinstance(obj, Enum):
        return obj.value
    else:
        return obj

def safe_json_dumps(obj):
    """
    安全的JSON序列化，处理枚举类型
    
    Args:
        obj: 要序列化的对象
        
    Returns:
        JSON字符串
    """
    try:
        # 先转换枚举键，再序列化
        converted_obj = convert_enum_keys(obj)
        return json.dumps(converted_obj, cls=CustomJSONEncoder, ensure_ascii=False)
    except Exception as e:
        logger.error(f"JSON序列化失败: {e}")
        return json.dumps({"error": "序列化失败", "details": str(e)})

# ======== 应用初始化 ========
app = Flask(__name__)
CORS(app)

# 配置Flask应用
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
app.config['JSON_AS_ASCII'] = False
app.json_encoder = CustomJSONEncoder

# 服务器配置
OLLAMA_API_URL = "http://127.0.0.1:11434"
MAX_HISTORY = 10  # 最大历史记录轮数
SESSION_TIMEOUT = 7200  # 会话超时时间（秒）
MAX_CONCURRENT_REQUESTS = 50  # 最大并发请求数
REQUEST_TIMEOUT = 300  # 请求超时时间（秒）

# 安全配置
if SECURITY_AVAILABLE:
    security_config = SecurityConfig(
        rate_limit_requests=100,  # 每分钟100个请求
        rate_limit_window=60,
        burst_limit=10,
        max_file_size=50 * 1024 * 1024,  # 50MB
        allowed_extensions=['txt', 'pdf', 'doc', 'docx', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'wav', 'mp3'],
        max_text_length=10000,
        max_query_length=1000,
        enable_security_headers=True
    )
    
    # 创建安全增强的应用
    secure_app = create_secure_app(app, security_config)
    security_manager = secure_app.get_security_manager()
else:
    secure_app = None
    security_manager = None

# 初始化Ollama客户端
try:
    ollama_client = OllamaClient(base_url=OLLAMA_API_URL)
    OLLAMA_CLIENT_AVAILABLE = True
    logging.info("Ollama客户端初始化成功")
except Exception as e:
    OLLAMA_CLIENT_AVAILABLE = False
    ollama_client = None
    logging.error(f"Ollama客户端初始化失败: {e}")
    logging.warning("将使用直接API调用方式")

# ======== 优化的会话管理类 ========
class OptimizedSessionManager:
    """
    优化的会话管理器
    
    功能改进：
    1. 线程安全的会话管理
    2. 内存使用优化
    3. 自动清理过期会话
    4. 会话统计和监控
    5. 异常处理和恢复
    """
    
    def __init__(self, max_sessions: int = 1000, cleanup_interval: int = 3600):
        """
        初始化会话管理器
        
        Args:
            max_sessions: 最大会话数
            cleanup_interval: 清理间隔（秒）
        """
        self.sessions = {}
        self.lock = threading.RLock()
        self.max_sessions = max_sessions
        self.cleanup_interval = cleanup_interval
        
        # 统计信息
        self.stats = {
            'total_sessions': 0,
            'active_sessions': 0,
            'expired_sessions': 0,
            'total_messages': 0
        }
        
        # 启动清理线程
        self._start_cleanup_thread()
        
        logger.info(f"会话管理器初始化完成，最大会话数: {max_sessions}")
    
    def _start_cleanup_thread(self):
        """
        启动后台清理线程
        """
        def cleanup_task():
            while True:
                try:
                    self.cleanup_expired()
                    time.sleep(self.cleanup_interval)
                except Exception as e:
                    logger.error(f"会话清理任务发生错误: {e}")
                    time.sleep(60)  # 发生错误时等待1分钟再重试
        
        cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
        cleanup_thread.start()
        logger.info("会话清理线程已启动")
    
    def create_session(self, model_name: str, user_id: Optional[str] = None) -> str:
        """
        创建新会话
        
        Args:
            model_name: 模型名称
            user_id: 用户ID（可选）
            
        Returns:
            str: 会话ID
        """
        try:
            with self.lock:
                # 检查会话数量限制
                if len(self.sessions) >= self.max_sessions:
                    # 清理最旧的会话
                    self._cleanup_oldest_sessions(int(self.max_sessions * 0.1))
                
                # 生成会话ID
                timestamp = datetime.now().timestamp()
                session_id = f"{timestamp}-{model_name}-{user_id or 'anonymous'}"
                
                # 创建会话数据
                self.sessions[session_id] = {
                    "history": [],
                    "model": model_name,
                    "user_id": user_id,
                    "created_at": datetime.now(),
                    "last_active": datetime.now(),
                    "message_count": 0,
                    "total_tokens": 0
                }
                
                # 更新统计
                self.stats['total_sessions'] += 1
                self.stats['active_sessions'] = len(self.sessions)
                
                logger.info(f"创建新会话: {session_id}")
                return session_id
                
        except Exception as e:
            logger.error(f"创建会话时发生错误: {e}")
            raise
    
    def add_message(self, session_id: str, role: str, content: str) -> bool:
        """
        添加消息到历史记录
        
        Args:
            session_id: 会话ID
            role: 角色（user/assistant）
            content: 消息内容
            
        Returns:
            bool: 是否成功添加
        """
        try:
            with self.lock:
                if session_id not in self.sessions:
                    logger.warning(f"会话不存在: {session_id}")
                    return False
                
                # 添加消息
                message = {
                    "role": role,
                    "content": content,
                    "timestamp": datetime.now(),
                    "tokens": self._estimate_tokens(content)
                }
                
                self.sessions[session_id]["history"].append(message)
                self.sessions[session_id]["last_active"] = datetime.now()
                self.sessions[session_id]["message_count"] += 1
                self.sessions[session_id]["total_tokens"] += message["tokens"]
                
                # 修剪历史记录
                self._trim_history(session_id)
                
                # 更新统计
                self.stats['total_messages'] += 1
                
                return True
                
        except Exception as e:
            logger.error(f"添加消息时发生错误: {e}")
            return False
    
    def get_context(self, session_id: str, max_tokens: int = 2048) -> str:
        """
        构建上下文字符串
        
        Args:
            session_id: 会话ID
            max_tokens: 最大token数
            
        Returns:
            str: 上下文字符串
        """
        try:
            with self.lock:
                if session_id not in self.sessions:
                    logger.warning(f"会话不存在: {session_id}")
                    return ""
                
                context = []
                token_count = 0
                
                # 从最新消息开始倒序处理
                for msg in reversed(self.sessions[session_id]["history"]):
                    msg_tokens = msg.get("tokens", self._estimate_tokens(msg["content"]))
                    
                    if token_count + msg_tokens > max_tokens:
                        break
                    
                    text = f"{msg['role']}: {msg['content']}"
                    context.insert(0, text)  # 保持时间顺序
                    token_count += msg_tokens
                
                return "\n".join(context)
                
        except Exception as e:
            logger.error(f"获取上下文时发生错误: {e}")
            return ""
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取会话信息
        
        Args:
            session_id: 会话ID
            
        Returns:
            Optional[Dict[str, Any]]: 会话信息
        """
        try:
            with self.lock:
                if session_id not in self.sessions:
                    return None
                
                session = self.sessions[session_id].copy()
                # 移除历史记录以减少数据量
                session.pop('history', None)
                return session
                
        except Exception as e:
            logger.error(f"获取会话信息时发生错误: {e}")
            return None
    
    def delete_session(self, session_id: str) -> bool:
        """
        删除会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            bool: 是否成功删除
        """
        try:
            with self.lock:
                if session_id in self.sessions:
                    del self.sessions[session_id]
                    self.stats['active_sessions'] = len(self.sessions)
                    logger.info(f"删除会话: {session_id}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"删除会话时发生错误: {e}")
            return False
    
    def _trim_history(self, session_id: str):
        """
        修剪历史记录长度
        
        Args:
            session_id: 会话ID
        """
        try:
            history = self.sessions[session_id]["history"]
            
            # 保留最近N轮对话（每轮包含用户和助手消息）
            max_messages = MAX_HISTORY * 2
            
            while len(history) > max_messages:
                removed_msg = history.pop(0)
                # 更新token计数
                self.sessions[session_id]["total_tokens"] -= removed_msg.get("tokens", 0)
                
        except Exception as e:
            logger.error(f"修剪历史记录时发生错误: {e}")
    
    def _estimate_tokens(self, text: str) -> int:
        """
        估算文本的token数量
        
        Args:
            text: 文本内容
            
        Returns:
            int: 估算的token数
        """
        try:
            if not text:
                return 0
            
            # 中文字符按1.5个token计算，英文单词按0.8个token计算
            chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
            english_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
            other_chars = len(text) - chinese_chars - sum(len(word) for word in re.findall(r'\b[a-zA-Z]+\b', text))
            
            return int(chinese_chars * 1.5 + english_words * 0.8 + other_chars * 0.5)
            
        except Exception as e:
            logger.error(f"估算token时发生错误: {e}")
            return len(text) // 4  # 简单估算
    
    def cleanup_expired(self):
        """
        清理过期会话
        """
        try:
            with self.lock:
                now = datetime.now()
                expired_sessions = []
                
                for session_id, session_data in list(self.sessions.items()):
                    last_active = session_data.get("last_active", now)
                    if (now - last_active).total_seconds() > SESSION_TIMEOUT:
                        expired_sessions.append(session_id)
                
                # 删除过期会话
                for session_id in expired_sessions:
                    del self.sessions[session_id]
                
                # 更新统计
                self.stats['expired_sessions'] += len(expired_sessions)
                self.stats['active_sessions'] = len(self.sessions)
                
                if expired_sessions:
                    logger.info(f"清理过期会话: {len(expired_sessions)} 个")
                    
        except Exception as e:
            logger.error(f"清理过期会话时发生错误: {e}")
    
    def _cleanup_oldest_sessions(self, count: int):
        """
        清理最旧的会话
        
        Args:
            count: 要清理的会话数量
        """
        try:
            # 按最后活跃时间排序
            sorted_sessions = sorted(
                self.sessions.items(),
                key=lambda x: x[1].get("last_active", datetime.min)
            )
            
            # 删除最旧的会话
            for i in range(min(count, len(sorted_sessions))):
                session_id = sorted_sessions[i][0]
                del self.sessions[session_id]
                logger.info(f"清理最旧会话: {session_id}")
                
        except Exception as e:
            logger.error(f"清理最旧会话时发生错误: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        try:
            with self.lock:
                stats = self.stats.copy()
                stats['active_sessions'] = len(self.sessions)
                return stats
        except Exception as e:
            logger.error(f"获取统计信息时发生错误: {e}")
            return {}

# ======== 错误处理装饰器 ========
def handle_errors(func):
    """
    统一错误处理装饰器
    
    Args:
        func: 要装饰的函数
        
    Returns:
        装饰后的函数
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            logger.warning(f"参数错误 in {func.__name__}: {e}")
            return jsonify({
                "error": "参数错误",
                "code": "INVALID_PARAMETER",
                "details": str(e)
            }), 400
        except requests.exceptions.RequestException as e:
            logger.error(f"网络请求错误 in {func.__name__}: {e}")
            return jsonify({
                "error": "网络请求失败",
                "code": "NETWORK_ERROR",
                "details": "请检查网络连接和服务状态"
            }), 503
        except Exception as e:
            logger.error(f"未预期错误 in {func.__name__}: {e}", exc_info=True)
            return jsonify({
                "error": "服务器内部错误",
                "code": "INTERNAL_ERROR",
                "details": "请稍后重试或联系管理员"
            }), 500
    return wrapper

def validate_request_data(required_fields: List[str] = None, optional_fields: List[str] = None):
    """
    请求数据验证装饰器
    
    Args:
        required_fields: 必需字段列表
        optional_fields: 可选字段列表
        
    Returns:
        装饰器函数
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # 获取请求数据
                if request.is_json:
                    data = request.get_json()
                else:
                    data = request.form.to_dict()
                
                if not data:
                    return jsonify({
                        "error": "请求数据不能为空",
                        "code": "EMPTY_REQUEST"
                    }), 400
                
                # 检查必需字段
                if required_fields:
                    missing_fields = [field for field in required_fields if field not in data]
                    if missing_fields:
                        return jsonify({
                            "error": f"缺少必需字段: {', '.join(missing_fields)}",
                            "code": "MISSING_FIELDS",
                            "missing_fields": missing_fields
                        }), 400
                
                # 验证和清理数据
                if SECURITY_AVAILABLE and security_manager:
                    valid, error_msg, cleaned_data = security_manager.validate_input(data, 'json')
                    if not valid:
                        return jsonify({
                            "error": error_msg,
                            "code": "INVALID_INPUT"
                        }), 400
                    g.validated_data = cleaned_data
                else:
                    g.validated_data = data
                
                return func(*args, **kwargs)
                
            except Exception as e:
                logger.error(f"请求验证失败: {e}")
                return jsonify({
                    "error": "请求验证失败",
                    "code": "VALIDATION_ERROR",
                    "details": str(e)
                }), 400
        return wrapper
    return decorator

# ======== 初始化模块 ========
session_manager = OptimizedSessionManager()

# 初始化RAG模块
if RAG_AVAILABLE:
    try:
        rag_manager = RAGManager()
        document_processor = DocumentProcessor()
        logger.info("RAG管理器和文档处理器初始化成功")
    except Exception as e:
        logger.error(f"RAG模块初始化失败: {e}")
        RAG_AVAILABLE = False
        rag_manager = None
        document_processor = None
else:
    rag_manager = None
    document_processor = None

# 初始化情感系统
if OPTIMIZED_ANIMATION_AVAILABLE:
    try:
        emotion_controller = OptimizedAdvancedAnimationController(use_ml_emotion_analysis=False)
        logger.info("优化情感系统初始化成功")
    except Exception as e:
        logger.error(f"优化情感系统初始化失败: {e}")
        OPTIMIZED_ANIMATION_AVAILABLE = False
        emotion_controller = None
elif EMOTION_SYSTEM_AVAILABLE:
    try:
        emotion_controller = AdvancedAnimationController(use_ml_emotion_analysis=False)
        logger.info("原始情感系统初始化成功")
    except Exception as e:
        logger.error(f"情感系统初始化失败: {e}")
        EMOTION_SYSTEM_AVAILABLE = False
        emotion_controller = None
else:
    emotion_controller = None

# 初始化多模态系统
if MULTIMODAL_SYSTEM_AVAILABLE:
    try:
        multimodal_manager = MultimodalManager(
            rag_manager=rag_manager if RAG_AVAILABLE else None
        )
        logger.info("多模态系统初始化成功")
    except Exception as e:
        logger.error(f"多模态系统初始化失败: {e}")
        MULTIMODAL_SYSTEM_AVAILABLE = False
        multimodal_manager = None
else:
    multimodal_manager = None

# 初始化语音情感系统
if VOICE_EMOTION_SYSTEM_AVAILABLE:
    try:
        voice_emotion_manager = VoiceEmotionManager(
            emotion_controller=emotion_controller if (OPTIMIZED_ANIMATION_AVAILABLE or EMOTION_SYSTEM_AVAILABLE) else None
        )
        logger.info("语音情感系统初始化成功")
    except Exception as e:
        logger.error(f"语音情感系统初始化失败: {e}")
        VOICE_EMOTION_SYSTEM_AVAILABLE = False
        voice_emotion_manager = None
else:
    voice_emotion_manager = None

# 初始化高级RAG系统
if ADVANCED_RAG_SYSTEM_AVAILABLE:
    try:
        advanced_rag_manager = AdvancedRAGManager(
            rag_manager=rag_manager if RAG_AVAILABLE else None,
            multimodal_manager=multimodal_manager if MULTIMODAL_SYSTEM_AVAILABLE else None
        )
        logger.info("高级RAG系统初始化成功")
    except Exception as e:
        logger.error(f"高级RAG系统初始化失败: {e}")
        ADVANCED_RAG_SYSTEM_AVAILABLE = False
        advanced_rag_manager = None
else:
    advanced_rag_manager = None

# ======== 工具函数 ========
async def text_to_speech(text: str, voice: str = "zh-CN-XiaoxiaoNeural") -> Optional[str]:
    """
    文本转语音功能
    
    Args:
        text: 要转换的文本
        voice: 语音类型
        
    Returns:
        Optional[str]: 临时音频文件路径
    """
    temp_file = None
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_file.close()
        
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(temp_file.name)
        
        logger.info(f"语音合成成功: {len(text)} 字符")
        return temp_file.name
        
    except Exception as e:
        logger.error(f"Edge-TTS 语音合成失败: {e}")
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except:
                pass
        return None

def create_security_decorator():
    """
    创建安全装饰器
    
    Returns:
        装饰器函数
    """
    if SECURITY_AVAILABLE and security_manager:
        return security_manager.security_middleware()
    else:
        # 基础安全检查
        def basic_security(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # 基本的请求大小检查
                if request.content_length and request.content_length > 50 * 1024 * 1024:
                    return jsonify({"error": "请求体过大"}), 413
                return func(*args, **kwargs)
            return wrapper
        return basic_security

# 创建安全装饰器
security_check = create_security_decorator()

# ======== API路由 ========

@app.route('/generate', methods=['POST'])
@security_check
@handle_errors
@validate_request_data(required_fields=['prompt'], optional_fields=['session_id', 'model_name', 'role_prompt', 'temperature', 'top_p'])
def generate():
    """
    生成AI回复的主要端点
    
    Returns:
        流式响应或错误信息
    """
    try:
        data = g.validated_data
        session_id = data.get('session_id')
        user_input = data.get('prompt')
        role_prompt = data.get('role_prompt', '一个AI助手，会认真回答您的问题。')
        model_name = data.get('model_name', 'qwen2:0.5b')
        temperature = data.get('temperature', 0.7)
        top_p = data.get('top_p', 1.0)
        
        # 验证参数
        if not user_input or not user_input.strip():
            return jsonify({
                "error": "输入内容不能为空",
                "code": "EMPTY_INPUT"
            }), 400
        
        # 处理新会话
        if not session_id or not session_manager.get_session_info(session_id):
            session_id = session_manager.create_session(model_name)
        
        # 构建上下文
        context = session_manager.get_context(session_id)
        
        # 使用角色设定构建提示
        full_prompt = f"{role_prompt}\n\n对话历史：\n{context}\n\n用户：{user_input}\n助手："
        
        # 记录用户输入
        session_manager.add_message(session_id, "user", user_input)
        
        logger.info(f"处理生成请求 - 会话: {session_id}, 模型: {model_name}")
        
        # 流式响应生成
        def generate_stream():
            full_response = ""
            emotion_analysis_result = None
            
            try:
                # 情感分析（如果可用）
                if (OPTIMIZED_ANIMATION_AVAILABLE or EMOTION_SYSTEM_AVAILABLE) and emotion_controller:
                    try:
                        emotion_analysis_result = emotion_controller.process_text_input(
                            user_input, trigger_animation=True
                        )
                        
                        # 发送情感分析结果
                        yield safe_json_dumps({
                            "session_id": session_id,
                            "type": "emotion_analysis",
                            "emotion_data": emotion_analysis_result,
                            "done": False
                        }) + "\n"
                        
                    except Exception as e:
                        logger.warning(f"情感分析失败: {e}")
                
                # AI生成响应
                use_ollama_client = False
                response = None
                
                # 尝试使用OllamaClient
                if OLLAMA_CLIENT_AVAILABLE and ollama_client:
                    try:
                        response = ollama_client.generate_for_web(
                            model=model_name,
                            prompt=full_prompt,
                            temperature=temperature,
                            top_p=top_p
                        )
                        use_ollama_client = True
                        logger.debug("使用OllamaClient生成响应")
                    except Exception as e:
                        logger.error(f"OllamaClient调用失败，回退到直接API: {e}")
                        use_ollama_client = False
                
                # 回退到直接API调用
                if not use_ollama_client:
                    try:
                        response = requests.post(
                            f"{OLLAMA_API_URL}/api/generate",
                            json={
                                "model": model_name,
                                "prompt": full_prompt,
                                "stream": True,
                                "options": {
                                    "temperature": temperature,
                                    "top_p": top_p
                                }
                            },
                            stream=True,
                            timeout=REQUEST_TIMEOUT
                        )
                        response.raise_for_status()
                        logger.debug("使用直接API调用生成响应")
                    except requests.exceptions.RequestException as e:
                        logger.error(f"API调用失败: {e}")
                        yield safe_json_dumps({
                            "session_id": session_id,
                            "error": "AI服务暂时不可用",
                            "code": "AI_SERVICE_ERROR",
                            "done": True
                        }) + "\n"
                        return
                
                # 处理响应流
                chunk_count = 0
                if use_ollama_client:
                    # 处理OllamaClient的生成器响应
                    for chunk in response:
                        chunk_count += 1
                        if chunk.get('response'):
                            full_response += chunk['response']
                            yield safe_json_dumps({
                                "session_id": session_id,
                                "chunk": chunk['response'],
                                "done": False
                            }) + "\n"
                        
                        if chunk.get('done'):
                            break
                else:
                    # 处理requests的流式响应
                    for line in response.iter_lines():
                        if line:
                            try:
                                chunk = json.loads(line.decode('utf-8'))
                                chunk_count += 1
                                
                                if chunk.get('response'):
                                    full_response += chunk['response']
                                    yield safe_json_dumps({
                                        "session_id": session_id,
                                        "chunk": chunk['response'],
                                        "done": False
                                    }) + "\n"
                                
                                if chunk.get('done'):
                                    break
                                    
                            except json.JSONDecodeError as e:
                                logger.warning(f"JSON解析失败: {e}")
                                continue
                
                # 记录AI响应
                if full_response:
                    session_manager.add_message(session_id, "assistant", full_response)
                
                # 发送完成信号
                yield safe_json_dumps({
                    "session_id": session_id,
                    "type": "completion",
                    "full_response": full_response,
                    "chunk_count": chunk_count,
                    "done": True
                }) + "\n"
                
                logger.info(f"响应生成完成 - 会话: {session_id}, 块数: {chunk_count}")
                
            except Exception as e:
                logger.error(f"生成响应时发生错误: {e}", exc_info=True)
                yield safe_json_dumps({
                    "session_id": session_id,
                    "error": "响应生成失败",
                    "code": "GENERATION_ERROR",
                    "details": str(e),
                    "done": True
                }) + "\n"
            
            finally:
                # 清理资源
                if response and hasattr(response, 'close'):
                    try:
                        response.close()
                    except:
                        pass
        
        return Response(
            stream_with_context(generate_stream()),
            mimetype='application/json',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive'
            }
        )
        
    except Exception as e:
        logger.error(f"生成端点发生错误: {e}", exc_info=True)
        return jsonify({
            "error": "请求处理失败",
            "code": "REQUEST_ERROR",
            "details": str(e)
        }), 500

# 系统状态端点
@app.route('/api/status', methods=['GET'])
@security_check
@handle_errors
def get_system_status():
    """
    获取系统状态信息
    
    Returns:
        系统状态JSON
    """
    try:
        # 检查Ollama连接
        ollama_status = "disconnected"
        try:
            response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                ollama_status = "connected"
        except:
            pass
        
        # 获取会话统计
        session_stats = session_manager.get_stats()
        
        # 获取安全统计
        security_stats = {}
        if SECURITY_AVAILABLE and security_manager:
            security_stats = security_manager.get_stats()
        
        status = {
            "server_status": "running",
            "timestamp": datetime.now().isoformat(),
            "ollama_status": ollama_status,
            "modules": {
                "rag_available": RAG_AVAILABLE,
                "emotion_system_available": OPTIMIZED_ANIMATION_AVAILABLE or EMOTION_SYSTEM_AVAILABLE,
                "multimodal_available": MULTIMODAL_SYSTEM_AVAILABLE,
                "voice_emotion_available": VOICE_EMOTION_SYSTEM_AVAILABLE,
                "advanced_rag_available": ADVANCED_RAG_SYSTEM_AVAILABLE,
                "voice_recognition_available": VOICE_RECOGNITION_AVAILABLE,
                "security_available": SECURITY_AVAILABLE,
                "optimized_animation_available": OPTIMIZED_ANIMATION_AVAILABLE
            },
            "session_stats": session_stats,
            "security_stats": security_stats
        }
        
        return jsonify(status), 200
        
    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        return jsonify({
            "error": "获取系统状态失败",
            "code": "STATUS_ERROR",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    logger.info("启动Live2D AI助手服务器...")
    logger.info(f"安全功能: {'启用' if SECURITY_AVAILABLE else '禁用'}")
    logger.info(f"优化动画: {'启用' if OPTIMIZED_ANIMATION_AVAILABLE else '禁用'}")
    
    # 启动服务器
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )