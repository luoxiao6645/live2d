import os
import tempfile
import logging
import asyncio
import numpy as np
from flask import Flask, request, jsonify
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperRecognizer:
    """使用Whisper模型进行语音识别的类"""

    def __init__(self, model_size="small", device=None):
        """
        初始化Whisper语音识别器

        参数:
            model_size: 模型大小，可选 "tiny", "base", "small", "medium", "large"
            device: 计算设备，None表示自动选择
        """
        self.model_size = model_size
        self.model_name = f"openai/whisper-{model_size}"

        # 自动选择设备
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")

        # 加载模型和处理器
        logger.info(f"正在加载Whisper模型 ({model_size})...")
        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        logger.info("Whisper模型加载完成")

        # 缓存目录
        self.temp_dir = tempfile.gettempdir()

    def recognize_from_file(self, audio_file, language="zh"):
        """
        从音频文件识别文本

        参数:
            audio_file: 音频文件路径
            language: 语言代码，默认中文

        返回:
            识别的文本
        """
        try:
            # 加载音频
            import librosa
            audio_array, sampling_rate = librosa.load(audio_file, sr=16000)

            # 处理音频
            input_features = self.processor(
                audio_array,
                sampling_rate=sampling_rate,
                return_tensors="pt"
            ).input_features.to(self.device)

            # 强制指定语言和任务
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language=language, task="transcribe"
            )

            # 生成文本
            predicted_ids = self.model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_length=448
            )

            # 解码文本
            transcription = self.processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]

            return {"success": True, "text": transcription}

        except Exception as e:
            logger.error(f"语音识别失败: {str(e)}")
            return {"success": False, "error": str(e)}

    def recognize_from_bytes(self, audio_bytes, language="zh"):
        """
        从音频字节数据识别文本

        参数:
            audio_bytes: 音频数据的字节
            language: 语言代码，默认中文

        返回:
            识别的文本
        """
        try:
            # 保存临时文件
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
            temp_file.write(audio_bytes)
            temp_file.close()

            # 从文件识别
            result = self.recognize_from_file(temp_file.name, language)

            # 删除临时文件
            os.unlink(temp_file.name)

            return result

        except Exception as e:
            logger.error(f"从字节数据识别失败: {str(e)}")
            return {"success": False, "error": str(e)}

# 全局变量，延迟初始化
whisper_recognizer = None

def get_recognizer():
    """获取或初始化语音识别器"""
    global whisper_recognizer
    if whisper_recognizer is None:
        whisper_recognizer = WhisperRecognizer(model_size="small")
    return whisper_recognizer

# Flask路由函数，可以集成到主服务器
def register_voice_routes(app):
    """注册语音识别相关的路由"""

    @app.route('/api/recognize_voice', methods=['GET', 'POST'])
    def recognize_voice():
        """处理语音识别请求"""
        try:
            # 记录请求方法和路径
            logger.info(f"收到语音识别请求，方法: {request.method}, 路径: {request.path}")

            # 如果是GET请求，返回状态信息
            if request.method == 'GET':
                logger.info("处理GET请求，返回状态信息")
                return jsonify({
                    "success": True,
                    "status": "ready",
                    "message": "语音识别服务已准备就绪，请使用POST方法上传音频文件"
                })

            # 处理POST请求
            logger.info("处理POST请求，检查音频文件")

            # 检查是否有文件上传
            if 'audio' not in request.files:
                logger.warning("没有上传音频文件")
                return jsonify({"success": False, "error": "没有上传音频文件"}), 400

            audio_file = request.files['audio']
            language = request.form.get('language', 'zh')
            logger.info(f"收到音频文件，语言: {language}")

            # 读取音频数据
            audio_bytes = audio_file.read()

            # 检查音频数据大小
            if len(audio_bytes) == 0:
                logger.warning("上传的音频文件为空")
                return jsonify({"success": False, "error": "上传的音频文件为空"}), 400

            # 获取识别器并识别
            logger.info("开始语音识别处理")
            recognizer = get_recognizer()
            result = recognizer.recognize_from_bytes(audio_bytes, language)

            logger.info(f"语音识别完成，结果: {result}")
            return jsonify(result)

        except Exception as e:
            logger.error(f"处理语音识别请求失败: {str(e)}")
            import traceback
            logger.error(f"异常详情: {traceback.format_exc()}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/check_voice_recognition', methods=['GET'])
    def check_voice_recognition():
        """检查语音识别服务状态"""
        try:
            # 检查CUDA是否可用
            cuda_available = torch.cuda.is_available()
            cuda_info = {
                "available": cuda_available,
                "device_count": torch.cuda.device_count() if cuda_available else 0,
                "device_name": torch.cuda.get_device_name(0) if cuda_available else "N/A"
            }

            # 检查模型是否已加载
            model_loaded = whisper_recognizer is not None

            return jsonify({
                "success": True,
                "status": "ready" if model_loaded else "not_initialized",
                "cuda": cuda_info,
                "model_info": {
                    "loaded": model_loaded,
                    "model_size": whisper_recognizer.model_size if model_loaded else None
                }
            })

        except Exception as e:
            logger.error(f"检查语音识别状态失败: {str(e)}")
            return jsonify({"success": False, "error": str(e)}), 500

# 测试代码
if __name__ == "__main__":
    # 创建测试应用
    app = Flask(__name__)
    register_voice_routes(app)

    # 运行测试服务器
    app.run(port=5001, debug=True)
