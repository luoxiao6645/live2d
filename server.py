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
from flask import Flask, jsonify, send_from_directory, request, Response, stream_with_context
from flask_cors import CORS
import edge_tts
from ollama_client import OllamaClient

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

# 导入情感系统模块
try:
    from emotion_system import AdvancedAnimationController
    EMOTION_SYSTEM_AVAILABLE = True
    logging.info("情感系统模块加载成功")
except ImportError as e:
    EMOTION_SYSTEM_AVAILABLE = False
    logging.warning(f"情感系统模块加载失败: {e}")
    logging.warning("情感系统功能将不可用")

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
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 自定义JSON编码器，处理枚举类型
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)

def convert_enum_keys(obj):
    """递归转换字典中的枚举键为字符串"""
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
    """安全的JSON序列化，处理枚举类型"""
    # 先转换枚举键，再序列化
    converted_obj = convert_enum_keys(obj)
    return json.dumps(converted_obj, cls=CustomJSONEncoder, ensure_ascii=False)

app = Flask(__name__)
CORS(app)

OLLAMA_API_URL = "http://127.0.0.1:11434"
MAX_HISTORY = 10  # 最大历史记录轮数
SESSION_TIMEOUT = 7200  # 会话超时时间（秒）

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

# ======== 上下文管理类 ========
class SessionManager:
    def __init__(self):
        self.sessions = {}
        self.lock = threading.Lock()
        self.cleanup_interval = 3600  # 清理间隔（秒）

    def create_session(self, model_name):
        """创建新会话"""
        session_id = f"{datetime.now().timestamp()}-{model_name}"
        with self.lock:
            self.sessions[session_id] = {
                "history": [],
                "model": model_name,
                "last_active": datetime.now()
            }
        return session_id

    def add_message(self, session_id, role, content):
        """添加消息到历史记录"""
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id]["history"].append({
                    "role": role,
                    "content": content,
                    "timestamp": datetime.now()
                })
                self._trim_history(session_id)
                self.sessions[session_id]["last_active"] = datetime.now()

    def get_context(self, session_id, max_tokens=2048):
        """构建上下文字符串"""
        with self.lock:
            if session_id not in self.sessions:
                return ""

            context = []
            token_count = 0
            for msg in reversed(self.sessions[session_id]["history"]):
                text = f"{msg['role']}: {msg['content']}"
                tokens = self._estimate_tokens(text)

                if token_count + tokens > max_tokens:
                    break

                context.insert(0, text)  # 保持时间顺序
                token_count += tokens

            return "\n".join(context)

    def _trim_history(self, session_id):
        """维护历史记录长度"""
        history = self.sessions[session_id]["history"]
        while len(history) > MAX_HISTORY * 2:  # 保留最近N轮对话
            history.pop(0)
            history.pop(0)

    def _estimate_tokens(self, text):
        """Token估算（中文1.5/字，英文0.8/词）"""
        chinese = len(re.findall(r'[\u4e00-\u9fa5]', text))
        english = len(re.findall(r'\b[a-zA-Z]+\b', text))
        return int(chinese * 1.5 + english * 0.8)

    def cleanup_expired(self):
        """清理过期会话"""
        with self.lock:
            now = datetime.now()
            expired = [sid for sid, data in self.sessions.items()
                      if (now - data["last_active"]).seconds > SESSION_TIMEOUT]
            for sid in expired:
                del self.sessions[sid]
            logger.info(f"清理过期会话：{len(expired)} 个")

# ======== 初始化模块 ========
session_manager = SessionManager()

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
if EMOTION_SYSTEM_AVAILABLE:
    try:
        emotion_controller = AdvancedAnimationController(use_ml_emotion_analysis=False)
        logger.info("情感系统初始化成功")
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
            emotion_controller=emotion_controller if EMOTION_SYSTEM_AVAILABLE else None
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

# 启动后台清理线程
def cleanup_task():
    while True:
        session_manager.cleanup_expired()
        time.sleep(session_manager.cleanup_interval)

threading.Thread(target=cleanup_task, daemon=True).start()

# ======== 原有功能修改 ========
async def text_to_speech(text, voice="zh-CN-XiaoxiaoNeural"):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_file.close()
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(temp_file.name)
        return temp_file.name
    except Exception as e:
        logger.error(f"Edge-TTS 语音合成失败: {e}")
        return None

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        user_input = data.get('prompt')
        role_prompt = data.get('role_prompt', '一个AI助手，会认真回答您的问题。')  # 获取角色设定
        model_name = data.get('model_name', 'qwen2:0.5b')

        # 处理新会话
        if not session_id or session_id not in session_manager.sessions:
            session_id = session_manager.create_session(model_name)

        # 构建上下文
        context = session_manager.get_context(session_id)

        # 使用角色设定构建提示
        full_prompt = f"{role_prompt}\n\n对话历史：\n{context}\n\n用户：{user_input}\n助手："

        # 记录完整提示用于调试
        logger.info(f"完整提示: {full_prompt[:100]}...（已截断）")

        # 调用Ollama
        def generate_stream():
            full_response = ""
            emotion_analysis_result = None

            # 如果情感系统可用，先分析用户输入的情感
            if EMOTION_SYSTEM_AVAILABLE and emotion_controller:
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

            # 使用OllamaClient或直接API调用
            use_ollama_client = False
            if OLLAMA_CLIENT_AVAILABLE and ollama_client:
                try:
                    response = ollama_client.generate_for_web(
                        model=model_name,
                        prompt=full_prompt,
                        temperature=data.get('temperature', 0.7),
                        top_p=data.get('top_p', 1.0)
                    )
                    use_ollama_client = True
                except Exception as e:
                    logger.error(f"OllamaClient调用失败，回退到直接API: {e}")
                    use_ollama_client = False

            if not use_ollama_client:
                response = requests.post(
                    f"{OLLAMA_API_URL}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": full_prompt,
                        "stream": True,
                        "options": {
                            "temperature": data.get('temperature', 0.7),
                            "top_p": data.get('top_p', 1.0)
                        }
                    },
                    stream=True
                )

            try:
                logger.info(f"开始流式响应处理，使用OllamaClient: {use_ollama_client}")
                if use_ollama_client:
                    # 处理OllamaClient的生成器响应
                    chunk_count = 0
                    for chunk in response:
                        chunk_count += 1
                        logger.debug(f"OllamaClient chunk {chunk_count}: {chunk}")
                        if chunk.get('response'):
                            full_response += chunk['response']
                            yield safe_json_dumps({
                                "session_id": session_id,
                                "chunk": chunk['response'],
                                "done": False
                            }) + "\n"

                        if chunk.get('done'):
                            logger.info(f"OllamaClient流式响应完成，总共处理 {chunk_count} 个chunk")
                            break
                else:
                    # 处理requests的流式响应
                    line_count = 0
                    for line in response.iter_lines(decode_unicode=True):
                        if line:
                            line_count += 1
                            logger.debug(f"Requests line {line_count}: {line}")
                            chunk = json.loads(line)
                            if 'response' in chunk:
                                full_response += chunk['response']
                                yield safe_json_dumps({
                                    "session_id": session_id,
                                    "chunk": chunk['response'],
                                    "done": False
                                }) + "\n"

                            if chunk.get('done'):
                                logger.info(f"Requests流式响应完成，总共处理 {line_count} 行")
                                break

                # 分析AI回复的情感（用于调整表情）
                ai_emotion_result = None
                if EMOTION_SYSTEM_AVAILABLE and emotion_controller and full_response:
                    try:
                        ai_emotion_result = emotion_controller.process_text_input(
                            full_response, trigger_animation=True
                        )
                    except Exception as e:
                        logger.warning(f"AI回复情感分析失败: {e}")

                # 保存完整对话
                session_manager.add_message(session_id, "user", user_input)
                session_manager.add_message(session_id, "assistant", full_response)

                # 生成语音
                audio_file = asyncio.run(text_to_speech(full_response.strip()))

                # 准备完成数据
                completion_data = {
                    "session_id": session_id,
                    "chunk": "",
                    "done": True,
                    "audio_url": f"/audio/{os.path.basename(audio_file)}" if audio_file else None
                }

                # 添加情感分析结果
                if emotion_analysis_result:
                    completion_data["user_emotion"] = emotion_analysis_result
                if ai_emotion_result:
                    completion_data["ai_emotion"] = ai_emotion_result
                    completion_data["live2d_parameters"] = emotion_controller.get_current_live2d_parameters()

                yield safe_json_dumps(completion_data) + "\n"
            except Exception as e:
                logger.error(f"流式响应处理失败: {e}")
            finally:
                # 只有requests响应对象才有close方法
                if not use_ollama_client and hasattr(response, 'close'):
                    response.close()

        return Response(stream_with_context(generate_stream()), content_type='application/json')

    except Exception as e:
        logger.error(f"处理请求失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/rag_generate', methods=['POST'])
def rag_generate():
    """RAG增强的对话生成"""
    if not RAG_AVAILABLE:
        return jsonify({"error": "RAG功能不可用"}), 503

    try:
        data = request.get_json()
        session_id = data.get('session_id')
        user_input = data.get('prompt')
        role_prompt = data.get('role_prompt', '你是一个AI助手，会基于提供的文档内容认真回答用户的问题。')
        model_name = data.get('model_name', 'qwen2:0.5b')
        temperature = data.get('temperature', 0.7)
        use_rag = data.get('use_rag', True)

        # 处理新会话
        if not session_id or session_id not in session_manager.sessions:
            session_id = session_manager.create_session(model_name)

        def generate_rag_stream():
            full_response = ""
            retrieved_docs = []

            try:
                if use_rag:
                    # 使用RAG模式
                    rag_chain = rag_manager.create_rag_chain(model_name, temperature)

                    # 先检索相关文档
                    retrieved_docs = rag_manager.search_documents(user_input, k=3)

                    # 发送检索到的文档信息
                    if retrieved_docs:
                        docs_info = []
                        for doc in retrieved_docs:
                            docs_info.append({
                                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                                "source": doc.metadata.get("source_file", "未知"),
                                "doc_id": doc.metadata.get("doc_id", "")
                            })

                        yield json.dumps({
                            "session_id": session_id,
                            "type": "retrieved_docs",
                            "docs": docs_info,
                            "done": False
                        }) + "\n"

                    # 使用RAG链生成回答
                    try:
                        # 由于RAG链可能不支持流式输出，我们先获取完整回答
                        full_response = rag_chain.invoke(user_input)

                        # 模拟流式输出
                        words = full_response.split()
                        for i, word in enumerate(words):
                            chunk = word + " "
                            yield json.dumps({
                                "session_id": session_id,
                                "chunk": chunk,
                                "done": False
                            }) + "\n"
                            time.sleep(0.05)  # 小延迟模拟流式效果

                    except Exception as e:
                        logger.error(f"RAG生成失败，回退到普通模式: {e}")
                        # 回退到普通Ollama模式
                        use_rag = False

                if not use_rag:
                    # 普通模式（原有逻辑）
                    context = session_manager.get_context(session_id)
                    full_prompt = f"{role_prompt}\n\n对话历史：\n{context}\n\n用户：{user_input}\n助手："

                    # 使用OllamaClient或直接API调用
                    if OLLAMA_CLIENT_AVAILABLE and ollama_client:
                        try:
                            response = ollama_client.generate_for_web(
                                model=model_name,
                                prompt=full_prompt,
                                temperature=temperature,
                                top_p=data.get('top_p', 1.0)
                            )
                        except Exception as e:
                            logger.error(f"RAG模式OllamaClient调用失败，回退到直接API: {e}")
                            response = requests.post(
                                f"{OLLAMA_API_URL}/api/generate",
                                json={
                                    "model": model_name,
                                    "prompt": full_prompt,
                                    "stream": True,
                                    "options": {
                                        "temperature": temperature,
                                        "top_p": data.get('top_p', 1.0)
                                    }
                                },
                                stream=True
                            )
                    else:
                        response = requests.post(
                            f"{OLLAMA_API_URL}/api/generate",
                            json={
                                "model": model_name,
                                "prompt": full_prompt,
                                "stream": True,
                                "options": {
                                    "temperature": temperature,
                                    "top_p": data.get('top_p', 1.0)
                                }
                            },
                            stream=True
                        )

                    for line in response.iter_lines(decode_unicode=True):
                        if line:
                            chunk = json.loads(line)
                            if 'response' in chunk:
                                full_response += chunk['response']
                                yield json.dumps({
                                    "session_id": session_id,
                                    "chunk": chunk['response'],
                                    "done": False
                                }) + "\n"

                            if chunk.get('done'):
                                break

                # 保存对话历史
                session_manager.add_message(session_id, "user", user_input)
                session_manager.add_message(session_id, "assistant", full_response)

                # 生成语音
                audio_file = asyncio.run(text_to_speech(full_response.strip()))

                # 发送完成信号
                yield json.dumps({
                    "session_id": session_id,
                    "chunk": "",
                    "done": True,
                    "audio_url": f"/audio/{os.path.basename(audio_file)}" if audio_file else None,
                    "used_rag": use_rag,
                    "retrieved_docs_count": len(retrieved_docs) if retrieved_docs else 0
                }) + "\n"

            except Exception as e:
                logger.error(f"RAG生成过程中出错: {e}")
                yield json.dumps({
                    "session_id": session_id,
                    "error": str(e),
                    "done": True
                }) + "\n"

        return Response(stream_with_context(generate_rag_stream()), content_type='application/json')

    except Exception as e:
        logger.error(f"RAG请求处理失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ======== 新增会话管理接口 ========
@app.route('/api/sessions', methods=['POST'])
def create_new_session():
    data = request.get_json()
    model = data.get('model', 'qwen2:0.5b')
    session_id = session_manager.create_session(model)
    return jsonify({"session_id": session_id})

@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    with session_manager.lock:
        if session_id in session_manager.sessions:
            del session_manager.sessions[session_id]
            return jsonify({"status": "deleted"})
        return jsonify({"error": "Session not found"}), 404

# ======== RAG相关接口 ========
@app.route('/api/rag/status', methods=['GET'])
def get_rag_status():
    """获取RAG功能状态"""
    return jsonify({
        "available": RAG_AVAILABLE,
        "knowledge_base_info": rag_manager.get_knowledge_base_info() if RAG_AVAILABLE else None
    })

@app.route('/api/rag/upload', methods=['POST'])
def upload_document():
    """上传文档到知识库"""
    if not RAG_AVAILABLE:
        return jsonify({"error": "RAG功能不可用"}), 503

    try:
        if 'file' not in request.files:
            return jsonify({"error": "没有上传文件"}), 400

        file = request.files['file']

        # 保存上传的文件
        success, result, file_info = document_processor.save_uploaded_file(file)
        if not success:
            return jsonify({"error": result}), 400

        file_path = result

        # 处理文档并添加到知识库
        metadata = {
            "original_filename": file_info.get("original_filename"),
            "file_size": file_info.get("file_size"),
            "file_type": file_info.get("file_type")
        }

        doc_id = rag_manager.process_and_store_document(file_path, metadata)

        return jsonify({
            "success": True,
            "doc_id": doc_id,
            "file_info": file_info,
            "message": "文档上传并处理成功"
        })

    except Exception as e:
        logger.error(f"上传文档失败: {e}")
        return jsonify({"error": f"上传文档失败: {str(e)}"}), 500

@app.route('/api/rag/knowledge_base', methods=['GET'])
def get_knowledge_base():
    """获取知识库信息"""
    if not RAG_AVAILABLE:
        return jsonify({"error": "RAG功能不可用"}), 503

    try:
        kb_info = rag_manager.get_knowledge_base_info()
        uploaded_files = document_processor.list_uploaded_files()

        return jsonify({
            "knowledge_base": kb_info,
            "uploaded_files": uploaded_files
        })

    except Exception as e:
        logger.error(f"获取知识库信息失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/rag/search', methods=['POST'])
def search_documents():
    """搜索相关文档"""
    if not RAG_AVAILABLE:
        return jsonify({"error": "RAG功能不可用"}), 503

    try:
        data = request.get_json()
        query = data.get('query', '')
        k = data.get('k', 5)

        if not query:
            return jsonify({"error": "查询内容不能为空"}), 400

        results = rag_manager.search_documents(query, k)

        # 格式化结果
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": doc.metadata.get("source_file", "未知"),
                "doc_id": doc.metadata.get("doc_id", "")
            })

        return jsonify({
            "query": query,
            "results": formatted_results,
            "count": len(formatted_results)
        })

    except Exception as e:
        logger.error(f"搜索文档失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/rag/delete/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    """删除知识库中的文档"""
    if not RAG_AVAILABLE:
        return jsonify({"error": "RAG功能不可用"}), 503

    try:
        success = rag_manager.delete_document(doc_id)
        if success:
            return jsonify({"success": True, "message": "文档删除成功"})
        else:
            return jsonify({"error": "文档删除失败或文档不存在"}), 404

    except Exception as e:
        logger.error(f"删除文档失败: {e}")
        return jsonify({"error": str(e)}), 500

# ======== 情感系统相关接口 ========
@app.route('/api/emotion/status', methods=['GET'])
def get_emotion_status():
    """获取情感系统状态"""
    if not EMOTION_SYSTEM_AVAILABLE:
        return jsonify({"available": False, "error": "情感系统不可用"}), 503

    try:
        status = emotion_controller.get_system_status()
        return jsonify({
            "available": True,
            "status": status
        })
    except Exception as e:
        logger.error(f"获取情感系统状态失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/emotion/analyze', methods=['POST'])
def analyze_emotion():
    """分析文本情感"""
    if not EMOTION_SYSTEM_AVAILABLE:
        return jsonify({"error": "情感系统不可用"}), 503

    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({"error": "文本内容不能为空"}), 400

        result = emotion_controller.process_text_input(text, trigger_animation=False)
        return jsonify(result)

    except Exception as e:
        logger.error(f"情感分析失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/emotion/trigger', methods=['POST'])
def trigger_emotion():
    """触发特定情感状态"""
    if not EMOTION_SYSTEM_AVAILABLE:
        return jsonify({"error": "情感系统不可用"}), 503

    try:
        data = request.get_json()
        emotion_type = data.get('emotion_type', 'neutral')
        intensity = data.get('intensity', 0.8)

        success = emotion_controller.force_emotion(emotion_type, intensity)

        if success:
            current_state = emotion_controller.emotion_state_manager.get_current_state()
            return jsonify({
                "success": True,
                "current_state": current_state,
                "live2d_parameters": emotion_controller.get_current_live2d_parameters()
            })
        else:
            return jsonify({"error": "设置情感状态失败"}), 500

    except Exception as e:
        logger.error(f"触发情感失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/emotion/gesture', methods=['POST'])
def trigger_gesture():
    """触发手势动画"""
    if not EMOTION_SYSTEM_AVAILABLE:
        return jsonify({"error": "情感系统不可用"}), 503

    try:
        data = request.get_json()
        gesture_name = data.get('gesture_name', '')

        if not gesture_name:
            return jsonify({"error": "手势名称不能为空"}), 400

        # 提取额外参数
        kwargs = {k: v for k, v in data.items() if k != 'gesture_name'}

        success = emotion_controller.trigger_gesture(gesture_name, **kwargs)

        if success:
            return jsonify({
                "success": True,
                "gesture_name": gesture_name,
                "active_animations": emotion_controller.animation_sequencer.get_active_animations()
            })
        else:
            return jsonify({"error": f"触发手势失败: {gesture_name}"}), 500

    except Exception as e:
        logger.error(f"触发手势失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/emotion/parameters', methods=['GET'])
def get_live2d_parameters():
    """获取当前Live2D参数"""
    if not EMOTION_SYSTEM_AVAILABLE:
        return jsonify({"error": "情感系统不可用"}), 503

    try:
        parameters = emotion_controller.get_current_live2d_parameters()
        return jsonify({
            "success": True,
            "parameters": parameters,
            "timestamp": time.time()
        })

    except Exception as e:
        logger.error(f"获取Live2D参数失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/emotion/history', methods=['GET'])
def get_emotion_history():
    """获取情感历史记录"""
    if not EMOTION_SYSTEM_AVAILABLE:
        return jsonify({"error": "情感系统不可用"}), 503

    try:
        limit = request.args.get('limit', 10, type=int)
        history = emotion_controller.get_emotion_history(limit)

        return jsonify({
            "success": True,
            "history": history,
            "count": len(history)
        })

    except Exception as e:
        logger.error(f"获取情感历史失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/emotion/reset', methods=['POST'])
def reset_emotion():
    """重置情感状态到中性"""
    if not EMOTION_SYSTEM_AVAILABLE:
        return jsonify({"error": "情感系统不可用"}), 503

    try:
        emotion_controller.reset_to_neutral()

        return jsonify({
            "success": True,
            "message": "情感状态已重置到中性",
            "current_state": emotion_controller.emotion_state_manager.get_current_state()
        })

    except Exception as e:
        logger.error(f"重置情感状态失败: {e}")
        return jsonify({"error": str(e)}), 500

# ======== 多模态系统相关接口 ========
@app.route('/api/multimodal/status', methods=['GET'])
def get_multimodal_status():
    """获取多模态系统状态"""
    if not MULTIMODAL_SYSTEM_AVAILABLE:
        return jsonify({"available": False, "error": "多模态系统不可用"}), 503

    try:
        status = multimodal_manager.get_system_status()
        return jsonify({
            "available": True,
            "status": status
        })
    except Exception as e:
        logger.error(f"获取多模态系统状态失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/multimodal/upload', methods=['POST'])
def upload_image():
    """上传图像"""
    if not MULTIMODAL_SYSTEM_AVAILABLE:
        return jsonify({"error": "多模态系统不可用"}), 503

    try:
        # 检查文件
        if 'file' not in request.files:
            return jsonify({"error": "没有上传文件"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "没有选择文件"}), 400

        # 获取额外参数
        description = request.form.get('description', '')
        add_to_kb = request.form.get('add_to_knowledge_base', 'true').lower() == 'true'

        # 处理图像
        result = multimodal_manager.upload_and_process_image(
            file,
            description if description else None,
            add_to_kb
        )

        if result["success"]:
            return jsonify(result)
        else:
            return jsonify(result), 400

    except Exception as e:
        logger.error(f"图像上传失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/multimodal/analyze', methods=['POST'])
def analyze_image():
    """分析图像内容"""
    if not MULTIMODAL_SYSTEM_AVAILABLE:
        return jsonify({"error": "多模态系统不可用"}), 503

    try:
        data = request.get_json()
        image_id = data.get('image_id')
        question = data.get('question', '')

        if not image_id:
            return jsonify({"error": "缺少image_id参数"}), 400

        if question:
            # 问答分析
            result = multimodal_manager.analyze_image_with_question(image_id, question)
        else:
            # 获取基本信息
            result = multimodal_manager.get_image_info(image_id)

        return jsonify(result)

    except Exception as e:
        logger.error(f"图像分析失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/multimodal/search', methods=['POST'])
def search_multimodal():
    """多模态搜索"""
    if not MULTIMODAL_SYSTEM_AVAILABLE:
        return jsonify({"error": "多模态系统不可用"}), 503

    try:
        data = request.get_json()
        query = data.get('query', '')
        search_type = data.get('search_type', 'all')  # all, images, text

        if not query:
            return jsonify({"error": "搜索查询不能为空"}), 400

        result = multimodal_manager.search_multimodal_content(query, search_type)
        return jsonify(result)

    except Exception as e:
        logger.error(f"多模态搜索失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/multimodal/compare', methods=['POST'])
def compare_images():
    """比较图像"""
    if not MULTIMODAL_SYSTEM_AVAILABLE:
        return jsonify({"error": "多模态系统不可用"}), 503

    try:
        data = request.get_json()
        image_id1 = data.get('image_id1')
        image_id2 = data.get('image_id2')
        comparison_aspect = data.get('comparison_aspect', '整体相似性')

        if not image_id1 or not image_id2:
            return jsonify({"error": "需要提供两个图像ID"}), 400

        result = multimodal_manager.compare_images(image_id1, image_id2, comparison_aspect)
        return jsonify(result)

    except Exception as e:
        logger.error(f"图像比较失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/multimodal/images', methods=['GET'])
def list_images():
    """列出所有图像"""
    if not MULTIMODAL_SYSTEM_AVAILABLE:
        return jsonify({"error": "多模态系统不可用"}), 503

    try:
        result = multimodal_manager.list_processed_images()
        return jsonify(result)

    except Exception as e:
        logger.error(f"列出图像失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/multimodal/images/<image_id>', methods=['DELETE'])
def delete_image(image_id):
    """删除图像"""
    if not MULTIMODAL_SYSTEM_AVAILABLE:
        return jsonify({"error": "多模态系统不可用"}), 503

    try:
        result = multimodal_manager.delete_image(image_id)

        if result["success"]:
            return jsonify(result)
        else:
            return jsonify(result), 404

    except Exception as e:
        logger.error(f"删除图像失败: {e}")
        return jsonify({"error": str(e)}), 500

# ======== 语音情感系统相关接口 ========
@app.route('/api/voice/status', methods=['GET'])
def get_voice_emotion_status():
    """获取语音情感系统状态"""
    if not VOICE_EMOTION_SYSTEM_AVAILABLE:
        return jsonify({"available": False, "error": "语音情感系统不可用"}), 503

    try:
        status = voice_emotion_manager.get_system_status()
        return jsonify({
            "available": True,
            "status": status
        })
    except Exception as e:
        logger.error(f"获取语音情感系统状态失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/voice/synthesize', methods=['POST'])
def synthesize_emotion_voice():
    """情感语音合成"""
    if not VOICE_EMOTION_SYSTEM_AVAILABLE:
        return jsonify({"error": "语音情感系统不可用"}), 503

    try:
        data = request.get_json()
        text = data.get('text', '')
        emotion = data.get('emotion', 'neutral')
        intensity = float(data.get('intensity', 1.0))
        voice_name = data.get('voice_name')
        language = data.get('language')
        style = data.get('style')
        realtime = data.get('realtime', False)

        if not text:
            return jsonify({"error": "文本内容不能为空"}), 400

        # 转换情感类型
        from voice_emotion_system.voice_emotion_config import EmotionType, VoiceLanguage
        try:
            emotion_type = EmotionType(emotion)
        except ValueError:
            emotion_type = EmotionType.NEUTRAL

        # 转换语言
        voice_language = None
        if language:
            try:
                voice_language = VoiceLanguage(language)
            except ValueError:
                pass

        if realtime:
            # 实时合成
            task_id = voice_emotion_manager.synthesize_realtime(
                text, emotion_type, intensity, auto_play=data.get('auto_play', False)
            )

            if task_id:
                return jsonify({
                    "success": True,
                    "task_id": task_id,
                    "message": "实时合成任务已提交"
                })
            else:
                return jsonify({"error": "实时合成任务提交失败"}), 500
        else:
            # 同步合成
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                result = loop.run_until_complete(
                    voice_emotion_manager.synthesize_with_emotion(
                        text, emotion_type, intensity, voice_name, voice_language, style
                    )
                )
            finally:
                loop.close()

            return jsonify(result)

    except Exception as e:
        logger.error(f"情感语音合成失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/voice/task/<task_id>', methods=['GET'])
def get_voice_task_status(task_id):
    """获取语音合成任务状态"""
    if not VOICE_EMOTION_SYSTEM_AVAILABLE:
        return jsonify({"error": "语音情感系统不可用"}), 503

    try:
        status = voice_emotion_manager.get_task_status(task_id)
        return jsonify(status)
    except Exception as e:
        logger.error(f"获取任务状态失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/voice/task/<task_id>', methods=['DELETE'])
def cancel_voice_task(task_id):
    """取消语音合成任务"""
    if not VOICE_EMOTION_SYSTEM_AVAILABLE:
        return jsonify({"error": "语音情感系统不可用"}), 503

    try:
        success = voice_emotion_manager.cancel_task(task_id)
        if success:
            return jsonify({"success": True, "message": "任务已取消"})
        else:
            return jsonify({"success": False, "message": "任务取消失败"}), 404
    except Exception as e:
        logger.error(f"取消任务失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/voice/style', methods=['POST'])
def set_voice_style():
    """设置语音风格"""
    if not VOICE_EMOTION_SYSTEM_AVAILABLE:
        return jsonify({"error": "语音情感系统不可用"}), 503

    try:
        data = request.get_json()
        style_name = data.get('style_name', 'default')

        result = voice_emotion_manager.set_voice_style(style_name)
        return jsonify(result)

    except Exception as e:
        logger.error(f"设置语音风格失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/voice/voices', methods=['GET'])
def get_available_voices():
    """获取可用语音列表"""
    if not VOICE_EMOTION_SYSTEM_AVAILABLE:
        return jsonify({"error": "语音情感系统不可用"}), 503

    try:
        language = request.args.get('language')

        if language:
            from voice_emotion_system.voice_emotion_config import VoiceLanguage
            try:
                voice_language = VoiceLanguage(language)
                voices = voice_emotion_manager.get_available_voices(voice_language)
            except ValueError:
                voices = voice_emotion_manager.get_available_voices()
        else:
            voices = voice_emotion_manager.get_available_voices()

        return jsonify(voices)

    except Exception as e:
        logger.error(f"获取可用语音失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/voice/styles', methods=['GET'])
def get_available_styles():
    """获取可用语音风格"""
    if not VOICE_EMOTION_SYSTEM_AVAILABLE:
        return jsonify({"error": "语音情感系统不可用"}), 503

    try:
        styles = voice_emotion_manager.get_available_styles()
        return jsonify(styles)

    except Exception as e:
        logger.error(f"获取可用风格失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/voice/emotions', methods=['GET'])
def get_supported_emotions():
    """获取支持的情感类型"""
    if not VOICE_EMOTION_SYSTEM_AVAILABLE:
        return jsonify({"error": "语音情感系统不可用"}), 503

    try:
        emotions = voice_emotion_manager.get_supported_emotions()
        return jsonify({"emotions": emotions})

    except Exception as e:
        logger.error(f"获取支持的情感失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/voice/languages', methods=['GET'])
def get_supported_languages():
    """获取支持的语言"""
    if not VOICE_EMOTION_SYSTEM_AVAILABLE:
        return jsonify({"error": "语音情感系统不可用"}), 503

    try:
        languages = voice_emotion_manager.get_supported_languages()
        return jsonify({"languages": languages})

    except Exception as e:
        logger.error(f"获取支持的语言失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/voice/settings', methods=['POST'])
def update_voice_settings():
    """更新语音设置"""
    if not VOICE_EMOTION_SYSTEM_AVAILABLE:
        return jsonify({"error": "语音情感系统不可用"}), 503

    try:
        data = request.get_json()
        result = voice_emotion_manager.update_settings(data)
        return jsonify(result)

    except Exception as e:
        logger.error(f"更新语音设置失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/voice/test', methods=['POST'])
def test_voice_synthesis():
    """测试语音合成"""
    if not VOICE_EMOTION_SYSTEM_AVAILABLE:
        return jsonify({"error": "语音情感系统不可用"}), 503

    try:
        data = request.get_json()
        test_text = data.get('text', '你好，这是语音测试。')

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                voice_emotion_manager.test_voice_synthesis(test_text)
            )
        finally:
            loop.close()

        return jsonify(result)

    except Exception as e:
        logger.error(f"语音合成测试失败: {e}")
        return jsonify({"error": str(e)}), 500

# ======== 高级RAG系统相关接口 ========
@app.route('/api/advanced-rag/status', methods=['GET'])
def get_advanced_rag_status():
    """获取高级RAG系统状态"""
    if not ADVANCED_RAG_SYSTEM_AVAILABLE:
        return jsonify({"available": False, "error": "高级RAG系统不可用"}), 503

    try:
        status = advanced_rag_manager.get_system_status()
        return jsonify({
            "available": True,
            "status": status
        })
    except Exception as e:
        logger.error(f"获取高级RAG系统状态失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/advanced-rag/process-document', methods=['POST'])
def process_document_advanced():
    """高级文档处理"""
    if not ADVANCED_RAG_SYSTEM_AVAILABLE:
        return jsonify({"error": "高级RAG系统不可用"}), 503

    try:
        data = request.get_json()
        text = data.get('text', '')
        doc_id = data.get('doc_id')
        image_path = data.get('image_path')
        metadata = data.get('metadata', {})

        if not text:
            return jsonify({"error": "文本内容不能为空"}), 400

        result = advanced_rag_manager.process_document(
            text=text,
            doc_id=doc_id,
            image_path=image_path,
            metadata=metadata
        )

        return jsonify(result)

    except Exception as e:
        logger.error(f"高级文档处理失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/advanced-rag/query', methods=['POST'])
def advanced_query():
    """高级查询"""
    if not ADVANCED_RAG_SYSTEM_AVAILABLE:
        return jsonify({"error": "高级RAG系统不可用"}), 503

    try:
        data = request.get_json()
        query = data.get('query', '')
        query_type = data.get('query_type', 'auto')
        max_results = int(data.get('max_results', 10))
        include_reasoning = data.get('include_reasoning', True)

        if not query:
            return jsonify({"error": "查询内容不能为空"}), 400

        result = advanced_rag_manager.advanced_query(
            query=query,
            query_type=query_type,
            max_results=max_results,
            include_reasoning=include_reasoning
        )

        return jsonify(result)

    except Exception as e:
        logger.error(f"高级查询失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/advanced-rag/reasoning', methods=['POST'])
def multimodal_reasoning():
    """多模态推理"""
    if not ADVANCED_RAG_SYSTEM_AVAILABLE:
        return jsonify({"error": "高级RAG系统不可用"}), 503

    try:
        data = request.get_json()
        query = data.get('query', '')
        image_path = data.get('image_path')
        reasoning_type = data.get('reasoning_type', 'path_based')

        if not query:
            return jsonify({"error": "查询内容不能为空"}), 400

        result = advanced_rag_manager.multimodal_reasoning(
            query=query,
            image_path=image_path,
            reasoning_type=reasoning_type
        )

        return jsonify(result)

    except Exception as e:
        logger.error(f"多模态推理失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/advanced-rag/graph-query', methods=['POST'])
def graph_query():
    """图查询"""
    if not ADVANCED_RAG_SYSTEM_AVAILABLE:
        return jsonify({"error": "高级RAG系统不可用"}), 503

    try:
        data = request.get_json()
        query = data.get('query', '')

        if not query:
            return jsonify({"error": "查询内容不能为空"}), 400

        result = advanced_rag_manager.query_processor.process_query(query)

        return jsonify(result.to_dict())

    except Exception as e:
        logger.error(f"图查询失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/advanced-rag/vectorize', methods=['POST'])
def advanced_vectorize():
    """高级向量化"""
    if not ADVANCED_RAG_SYSTEM_AVAILABLE:
        return jsonify({"error": "高级RAG系统不可用"}), 503

    try:
        data = request.get_json()
        text = data.get('text', '')
        image_path = data.get('image_path')
        fusion_strategy = data.get('fusion_strategy', 'concatenate')

        if not text:
            return jsonify({"error": "文本内容不能为空"}), 400

        if image_path:
            # 图文混合向量化
            embedding = advanced_rag_manager.vectorizer.vectorize_multimodal(
                text, image_path, fusion_strategy
            )
            vector_type = "multimodal"
        else:
            # 纯文本向量化
            embedding = advanced_rag_manager.vectorizer.vectorize_text(text)
            vector_type = "text"

        result = {
            "success": True,
            "vector_type": vector_type,
            "dimension": len(embedding),
            "embedding": embedding.tolist(),
            "fusion_strategy": fusion_strategy if image_path else None
        }

        return jsonify(result)

    except Exception as e:
        logger.error(f"高级向量化失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/advanced-rag/build-graph', methods=['POST'])
def build_knowledge_graph():
    """构建知识图谱"""
    if not ADVANCED_RAG_SYSTEM_AVAILABLE:
        return jsonify({"error": "高级RAG系统不可用"}), 503

    try:
        data = request.get_json()
        text = data.get('text', '')
        doc_id = data.get('doc_id')

        if not text:
            return jsonify({"error": "文本内容不能为空"}), 400

        result = advanced_rag_manager.kg_builder.build_graph_from_document(text, doc_id)

        return jsonify(result)

    except Exception as e:
        logger.error(f"知识图谱构建失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/advanced-rag/graph-stats', methods=['GET'])
def get_graph_stats():
    """获取图谱统计信息"""
    if not ADVANCED_RAG_SYSTEM_AVAILABLE:
        return jsonify({"error": "高级RAG系统不可用"}), 503

    try:
        stats = advanced_rag_manager.kg_builder.get_graph_stats()
        return jsonify(stats)

    except Exception as e:
        logger.error(f"获取图谱统计失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/advanced-rag/clear-caches', methods=['POST'])
def clear_advanced_rag_caches():
    """清理高级RAG缓存"""
    if not ADVANCED_RAG_SYSTEM_AVAILABLE:
        return jsonify({"error": "高级RAG系统不可用"}), 503

    try:
        advanced_rag_manager.clear_caches()
        return jsonify({"success": True, "message": "缓存清理完成"})

    except Exception as e:
        logger.error(f"缓存清理失败: {e}")
        return jsonify({"error": str(e)}), 500

# ======== 其他原有路由保持不变 ========
@app.route('/audio/<filename>')
def serve_audio(filename):
    temp_dir = tempfile.gettempdir()
    return send_from_directory(temp_dir, filename)

@app.route('/api/ollama/status')
def ollama_status():
    """检查Ollama服务状态"""
    try:
        if OLLAMA_CLIENT_AVAILABLE and ollama_client:
            models = ollama_client.get_available_models()
            return jsonify({
                "status": "connected",
                "client_available": True,
                "available_models": models,
                "model_count": len(models)
            })
        else:
            # 尝试直接连接
            response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=5)
            models = [model["name"] for model in response.json().get("models", [])]
            return jsonify({
                "status": "connected",
                "client_available": False,
                "available_models": models,
                "model_count": len(models)
            })
    except Exception as e:
        return jsonify({
            "status": "disconnected",
            "error": str(e),
            "client_available": OLLAMA_CLIENT_AVAILABLE
        }), 503

@app.route('/get_model_list')
def get_model_list():
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    models = []
    try:
        if os.path.isdir(model_dir):
            for folder in os.listdir(model_dir):
                folder_path = os.path.join(model_dir, folder)
                if os.path.isdir(folder_path):
                    # 检查 Cubism 4 模型 (.model3.json)
                    model3_files = [f for f in os.listdir(folder_path) if f.endswith('.model3.json')]
                    if model3_files:
                        for file in model3_files:
                            models.append({
                                'name': f"{folder} (Cubism 4)",
                                'path': os.path.join('models', folder, file).replace('\\', '/'),
                                'type': 'cubism4'
                            })

                    # 检查 Cubism 2 模型 (.model.json)
                    model2_files = [f for f in os.listdir(folder_path) if f.endswith('.model.json')]
                    if model2_files:
                        for file in model2_files:
                            models.append({
                                'name': f"{folder} (Cubism 2)",
                                'path': os.path.join('models', folder, file).replace('\\', '/'),
                                'type': 'cubism2'
                            })

        logger.info(f"找到 {len(models)} 个模型")
        return jsonify(models)
    except Exception as e:
        logger.error(f"获取模型列表失败: {str(e)}")
        return jsonify({"error": "获取模型列表失败"}), 500

@app.route('/')
def serve_html():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'live2d_llm.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), path)

@app.route('/api/check_model_compatibility', methods=['POST'])
def check_model_compatibility():
    data = request.get_json()
    model_path = data.get('model_path')

    # 检查文件存在性
    full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
    if not os.path.exists(full_path):
        return jsonify({"error": "模型文件不存在", "path": full_path}), 404

    # 检查文件扩展名
    file_ext = os.path.splitext(full_path)[1]
    model_type = ""

    if file_ext == ".moc":
        model_type = "Cubism 2"
    elif file_ext == ".moc3":
        model_type = "Cubism 3/4"
    else:
        return jsonify({"error": "不支持的模型文件格式", "extension": file_ext}), 400

    # 返回文件信息
    file_size = os.path.getsize(full_path)
    return jsonify({
        "model_type": model_type,
        "file_size": file_size,
        "file_path": full_path,
        "status": "文件存在且格式正确"
    })

@app.route('/api/model_info/<path:model_path>')
def get_model_info(model_path):
    try:
        # 构建完整路径
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
        if not os.path.exists(full_path):
            return jsonify({"error": "文件不存在"}), 404

        # 读取模型配置文件
        if full_path.endswith('.model.json') or full_path.endswith('.model3.json'):
            with open(full_path, 'r', encoding='utf-8') as f:
                model_data = json.load(f)

            # 检查关联文件是否存在
            model_dir = os.path.dirname(full_path)
            missing_files = []

            # 检查 moc 文件
            moc_file = None
            if 'model' in model_data:
                moc_file = os.path.join(model_dir, model_data['model'])
                if not os.path.exists(moc_file):
                    missing_files.append(model_data['model'])

            # 返回模型信息
            return jsonify({
                "model_path": full_path,
                "model_type": "Cubism 2" if full_path.endswith('.model.json') else "Cubism 3/4",
                "missing_files": missing_files,
                "config_data": model_data
            })
        else:
            return jsonify({"error": "不支持的文件格式"}), 400

    except Exception as e:
        logger.error(f"获取模型信息失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/check_all_models', methods=['GET'])
def check_all_models():
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    model_status = []

    try:
        if os.path.isdir(model_dir):
            for folder in os.listdir(model_dir):
                folder_path = os.path.join(model_dir, folder)
                if os.path.isdir(folder_path):
                    # 检查配置文件
                    for file in os.listdir(folder_path):
                        if file.endswith('.model3.json') or file.endswith('.model.json'):
                            config_path = os.path.join(folder_path, file)
                            try:
                                with open(config_path, 'r', encoding='utf-8') as f:
                                    config_data = json.load(f)

                                # 检查 MOC 文件
                                moc_filename = config_data.get('model', '')
                                moc_path = os.path.join(folder_path, moc_filename)
                                moc_exists = os.path.exists(moc_path)

                                model_status.append({
                                    'folder': folder,
                                    'config_file': file,
                                    'moc_file': moc_filename,
                                    'moc_exists': moc_exists,
                                    'moc_size': os.path.getsize(moc_path) if moc_exists else 0,
                                    'config_type': 'Cubism 4' if file.endswith('.model3.json') else 'Cubism 2',
                                    'status': '正常' if moc_exists else '模型文件缺失'
                                })
                            except Exception as e:
                                model_status.append({
                                    'folder': folder,
                                    'config_file': file,
                                    'error': str(e),
                                    'status': '检查失败'
                                })

        return jsonify(model_status)
    except Exception as e:
        logger.error(f"检查模型文件失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/view_model_file/<path:model_path>')
def view_model_file(model_path):
    try:
        # 构建完整路径
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
        if not os.path.exists(full_path):
            return jsonify({"error": "文件不存在"}), 404

        # 读取文件前100个字节（二进制）
        file_stats = os.stat(full_path)
        file_info = {
            "path": full_path,
            "size": file_stats.st_size,
            "modified": datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        }

        # 读取文件头
        with open(full_path, 'rb') as f:
            header = f.read(100)
            # 转换为十六进制显示
            file_info["header_hex"] = header.hex(' ')

            # 判断文件类型
            if full_path.endswith('.moc'):
                file_info["expected_header"] = "moc file for Cubism 2"
            elif full_path.endswith('.moc3'):
                file_info["expected_header"] = "moc3 file for Cubism 3/4"

        return jsonify(file_info)
    except Exception as e:
        logger.error(f"查看模型文件失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/log_model_error', methods=['POST'])
def log_model_error():
    try:
        data = request.get_json()
        model_path = data.get('model_path', '未知')
        error_message = data.get('error', '未知错误')
        stack_trace = data.get('stack', '')

        logger.error(f"模型加载失败 - 路径: {model_path}")
        logger.error(f"错误信息: {error_message}")
        logger.error(f"堆栈跟踪: {stack_trace}")

        return jsonify({"status": "错误已记录"})
    except Exception as e:
        logger.error(f"记录模型错误失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/convert_model', methods=['POST'])
def convert_model():
    try:
        data = request.get_json()
        source_path = data.get('source')
        target_type = data.get('target_type', 'cubism4')

        # 构建完整路径
        full_source_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), source_path)
        if not os.path.exists(full_source_path):
            return jsonify({"error": "源模型文件不存在"}), 404

        # 从路径获取文件夹和文件名
        model_dir = os.path.dirname(full_source_path)
        model_name = os.path.basename(full_source_path)

        # 获取源模型类型
        source_type = "cubism2" if model_name.endswith('.model.json') else "cubism4"

        # 如果源类型和目标类型相同，不需要转换
        if source_type == target_type:
            return jsonify({
                "status": "无需转换",
                "message": "源模型已经是目标格式"
            })

        # 注意：自动转换需要额外的库或工具，这里仅提供一个示例
        # 实际转换逻辑需要根据具体工具和环境实现
        # 这里只返回一个指导信息

        return jsonify({
            "status": "需要手动转换",
            "source_type": source_type,
            "target_type": target_type,
            "message": "Live2D 模型在不同版本间需要使用 Live2D Cubism Editor 进行转换。"
                       "建议下载并安装 Live2D Cubism Editor，然后导入源模型并导出为目标格式。",
            "model_info": {
                "source_path": full_source_path,
                "model_dir": model_dir,
                "model_name": model_name
            }
        })

    except Exception as e:
        logger.error(f"模型转换检查失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/model_directory_structure', methods=['GET'])
def get_model_directory_structure():
    try:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        structure = {}

        if not os.path.isdir(model_dir):
            return jsonify({"error": "模型目录不存在"}), 404

        for folder in os.listdir(model_dir):
            folder_path = os.path.join(model_dir, folder)
            if os.path.isdir(folder_path):
                structure[folder] = {}
                for root, dirs, files in os.walk(folder_path):
                    rel_dir = os.path.relpath(root, folder_path)
                    if rel_dir == '.':
                        rel_dir = ''
                    if rel_dir not in structure[folder]:
                        structure[folder][rel_dir] = []
                    for f in files:
                        file_size = os.path.getsize(os.path.join(root, f))
                        file_type = os.path.splitext(f)[1]
                        structure[folder][rel_dir].append({
                            'name': f,
                            'size': file_size,
                            'type': file_type
                        })

        return jsonify(structure)
    except Exception as e:
        logger.error(f"获取模型目录结构失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/verify_model_files/<path:model_config_path>', methods=['GET'])
def verify_model_files(model_config_path):
    try:
        # 构建配置文件完整路径
        app_dir = os.path.dirname(os.path.abspath(__file__))
        full_config_path = os.path.join(app_dir, model_config_path)

        if not os.path.exists(full_config_path):
            return jsonify({"error": "模型配置文件不存在", "path": full_config_path}), 404

        model_dir = os.path.dirname(full_config_path)
        model_type = "cubism2" if full_config_path.endswith('.model.json') else "cubism4"

        # 读取配置文件
        with open(full_config_path, 'r', encoding='utf-8') as f:
            model_config = json.load(f)

        # 验证必要文件存在
        missing_files = []
        file_sizes = {}
        lfs_pointers = []

        # 检查MOC文件
        if 'model' in model_config:
            moc_file = os.path.join(model_dir, model_config['model'])
            if not os.path.exists(moc_file):
                missing_files.append(model_config['model'])
            else:
                file_size = os.path.getsize(moc_file)
                file_sizes[model_config['model']] = file_size

                # 检查是否是LFS指针
                if file_size < 1000:  # 小于1KB可能是LFS指针
                    try:
                        with open(moc_file, 'r', encoding='utf-8', errors='ignore') as f:
                            first_line = f.readline().strip()
                            if first_line.startswith('version https://git-lfs.github.com'):
                                lfs_pointers.append(model_config['model'])
                    except:
                        pass  # 二进制文件无法读取为文本

        # 检查纹理文件
        if 'textures' in model_config:
            for texture in model_config['textures']:
                texture_file = os.path.join(model_dir, texture)
                if not os.path.exists(texture_file):
                    missing_files.append(texture)
                else:
                    file_sizes[texture] = os.path.getsize(texture_file)

        # 检查物理文件
        if 'physics' in model_config:
            physics_file = os.path.join(model_dir, model_config['physics'])
            if not os.path.exists(physics_file):
                missing_files.append(model_config['physics'])
            else:
                file_sizes[model_config['physics']] = os.path.getsize(physics_file)

        # 检查动作文件
        if 'motions' in model_config:
            for motion_group in model_config['motions']:
                for motion in model_config['motions'][motion_group]:
                    if 'file' in motion:
                        motion_file = os.path.join(model_dir, motion['file'])
                        if not os.path.exists(motion_file):
                            missing_files.append(motion['file'])
                        else:
                            file_sizes[motion['file']] = os.path.getsize(motion_file)

        # 检查表情文件
        if 'expressions' in model_config:
            for expression in model_config['expressions']:
                if 'file' in expression:
                    expression_file = os.path.join(model_dir, expression['file'])
                    if not os.path.exists(expression_file):
                        missing_files.append(expression['file'])
                    else:
                        file_sizes[expression['file']] = os.path.getsize(expression_file)

        # 返回验证结果
        return jsonify({
            "model_path": full_config_path,
            "model_type": model_type,
            "missing_files": missing_files,
            "file_sizes": file_sizes,
            "lfs_pointers": lfs_pointers,
            "has_lfs_pointers": len(lfs_pointers) > 0,
            "config_data": model_config,
            "is_valid": len(missing_files) == 0 and len(lfs_pointers) == 0
        })

    except Exception as e:
        logger.error(f"验证模型文件失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/check_lfs_pointers', methods=['GET'])
def check_lfs_pointers():
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    lfs_pointers = []

    try:
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                if file.endswith('.moc3') or file.endswith('.moc'):
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)

                    # 检查文件是否太小（可能是LFS指针）
                    if file_size < 1000:  # 通常LFS指针文件<1KB
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                first_line = f.readline().strip()
                                if first_line.startswith('version https://git-lfs.github.com'):
                                    # 尝试获取预期文件大小
                                    expected_size = "未知"
                                    for line in f:
                                        if 'size' in line:
                                            size_match = re.search(r'size (\d+)', line)
                                            if size_match:
                                                expected_size = int(size_match.group(1))
                                                break

                                    rel_path = os.path.relpath(file_path, os.path.dirname(os.path.abspath(__file__)))
                                    lfs_pointers.append({
                                        'path': rel_path,
                                        'size': file_size,
                                        'expected_size': expected_size,
                                        'content': first_line
                                    })
                        except:
                            pass  # 二进制文件可能无法读取为文本

        # 获取Git LFS状态
        git_lfs_installed = False
        try:
            import subprocess
            result = subprocess.run(['git', 'lfs', 'version'], capture_output=True, text=True)
            git_lfs_installed = result.returncode == 0
        except:
            pass

        return jsonify({
            'lfs_pointers_found': len(lfs_pointers) > 0,
            'git_lfs_installed': git_lfs_installed,
            'files': lfs_pointers,
            'repo_path': os.path.dirname(os.path.abspath(__file__))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 注册语音识别路由
logger.info("=== 注册语音识别路由 ===")
if VOICE_RECOGNITION_AVAILABLE:
    register_voice_routes(app)
    logger.info("已成功注册语音识别路由，完整功能可用")
else:
    logger.warning("语音识别模块不可用，注册备用路由")
    # 如果语音识别模块加载失败，注册一个简单的路由来处理请求
    @app.route('/api/recognize_voice', methods=['GET', 'POST'])
    def recognize_voice_fallback():
        logger.info(f"收到语音识别请求，方法: {request.method}")

        if request.method == 'GET':
            logger.info("处理GET请求")
            return jsonify({
                "success": True,  # 返回成功状态，避免前端报错
                "status": "limited",
                "message": "语音识别服务处于有限模式，请安装必要的依赖项：numpy, torch, transformers, librosa"
            })
        else:
            logger.info("处理POST请求")

            # 处理文件上传，即使我们不能实际处理语音识别
            try:
                if 'audio' in request.files:
                    audio_file = request.files['audio']
                    language = request.form.get('language', 'zh')
                    logger.info(f"收到音频文件，语言: {language}")

                    # 返回一个提示消息，而不是错误
                    if language == 'zh':
                        message = "语音识别服务需要安装额外的依赖项。请安装numpy、torch、transformers和librosa。"
                    else:
                        message = "Voice recognition requires additional dependencies. Please install numpy, torch, transformers, and librosa."

                    return jsonify({
                        "success": True,  # 返回成功状态，避免前端报错
                        "text": message
                    })
                else:
                    logger.warning("没有上传音频文件")
                    return jsonify({
                        "success": False,
                        "error": "没有上传音频文件"
                    }), 400
            except Exception as e:
                logger.error(f"处理语音识别请求失败: {str(e)}")
                return jsonify({
                    "success": False,
                    "error": f"处理请求失败: {str(e)}"
                }), 500

    # 添加一个路由来检查语音识别服务状态
    @app.route('/api/check_voice_recognition', methods=['GET'])
    def check_voice_recognition_fallback():
        logger.info("检查语音识别服务状态")
        return jsonify({
            "success": True,  # 返回成功状态，避免前端报错
            "status": "limited",
            "message": "语音识别服务处于有限模式，请安装必要的依赖项：numpy, torch, transformers, librosa",
            "dependencies": {
                "required": ["numpy", "torch", "transformers", "librosa"],
                "installed": False
            }
        })

if __name__ == '__main__':
    app.run(port=5000, debug=True)