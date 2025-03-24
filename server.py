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
from flask import Flask, jsonify, send_from_directory, request, Response, stream_with_context
from flask_cors import CORS
import edge_tts

# ======== 配置部分 ========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

OLLAMA_API_URL = "http://127.0.0.1:11434"
MAX_HISTORY = 10  # 最大历史记录轮数
SESSION_TIMEOUT = 7200  # 会话超时时间（秒）

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
        model_name = data.get('model_name', 'qwen2:0.5b')
        
        # 处理新会话
        if not session_id or session_id not in session_manager.sessions:
            session_id = session_manager.create_session(model_name)
        
        # 构建上下文
        context = session_manager.get_context(session_id)
        full_prompt = f"对话历史：\n{context}\n\n用户：{user_input}\n助手："
        
        # 调用Ollama
        def generate_stream():
            full_response = ""
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
                            # 保存完整对话
                            session_manager.add_message(session_id, "user", user_input)
                            session_manager.add_message(session_id, "assistant", full_response)
                            # 生成语音
                            audio_file = asyncio.run(text_to_speech(full_response.strip()))
                            yield json.dumps({
                                "session_id": session_id,
                                "chunk": "",
                                "done": True,
                                "audio_url": f"/audio/{os.path.basename(audio_file)}" if audio_file else None
                            }) + "\n"
            finally:
                response.close()
        
        return Response(stream_with_context(generate_stream()), content_type='application/json')
    
    except Exception as e:
        logger.error(f"处理请求失败: {str(e)}")
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

# ======== 其他原有路由保持不变 ========
@app.route('/audio/<filename>')
def serve_audio(filename):
    temp_dir = tempfile.gettempdir()
    return send_from_directory(temp_dir, filename)

@app.route('/get_model_list')
def get_model_list():
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
    models = []
    try:
        if os.path.isdir(model_dir):
            for folder in os.listdir(model_dir):
                folder_path = os.path.join(model_dir, folder)
                if os.path.isdir(folder_path):
                    for file in os.listdir(folder_path):
                        if file.endswith('.model3.json'):
                            models.append({
                                'name': folder,
                                'path': os.path.join('model', folder, file).replace('\\', '/')
                            })
                            break
        return jsonify(models)
    except Exception as e:
        logger.error(f"获取模型列表失败: {str(e)}")
        return jsonify({"error": "Failed to get model list"}), 500

@app.route('/')
def serve_html():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'live2d_llm.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), path)

if __name__ == '__main__':
    app.run(port=5000, debug=True)