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

if __name__ == '__main__':
    app.run(port=5000, debug=True)