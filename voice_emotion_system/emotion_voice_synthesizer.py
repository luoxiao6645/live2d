"""
情感语音合成器

负责根据情感状态生成相应的语音，支持SSML标记和情感参数控制
"""

import logging
import asyncio
import hashlib
import os
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import edge_tts

from .voice_emotion_config import VoiceEmotionConfig, EmotionType, VoiceLanguage, VoiceGender

# 配置日志
logger = logging.getLogger(__name__)

class EmotionVoiceSynthesizer:
    """情感语音合成器类"""
    
    def __init__(self):
        """初始化情感语音合成器"""
        self.config = VoiceEmotionConfig()
        
        # 创建输出目录
        self.config.create_output_directory()
        
        # 语音缓存
        self.voice_cache = {}
        
        # 当前设置
        self.current_voice = "xiaoxiao"
        self.current_language = VoiceLanguage.CHINESE
        self.current_emotion = EmotionType.NEUTRAL
        self.emotion_intensity = 1.0
        
        logger.info("情感语音合成器初始化完成")
    
    async def synthesize_with_emotion(self, text: str, emotion: EmotionType = EmotionType.NEUTRAL,
                                    intensity: float = 1.0, voice_name: str = None,
                                    language: VoiceLanguage = None) -> Dict[str, Any]:
        """
        根据情感合成语音
        
        Args:
            text: 要合成的文本
            emotion: 情感类型
            intensity: 情感强度 (0.0-1.0)
            voice_name: 语音名称（可选）
            language: 语言（可选，自动检测）
            
        Returns:
            合成结果字典
        """
        try:
            # 检测语言
            if language is None:
                language = self.config.detect_language(text)
            
            # 选择语音
            if voice_name is None:
                voice_name = self.current_voice
            
            # 获取语音配置
            voice_config = self._get_voice_config(voice_name, language)
            if not voice_config:
                return {"success": False, "error": f"语音 {voice_name} 不可用"}
            
            # 生成SSML
            ssml = self._generate_ssml(text, emotion, intensity, voice_config, language)
            
            # 生成缓存键
            cache_key = self._generate_cache_key(ssml, voice_config["name"])
            
            # 检查缓存
            if self.config.AUDIO_CONFIG["cache_enabled"] and cache_key in self.voice_cache:
                cached_result = self.voice_cache[cache_key]
                if os.path.exists(cached_result["file_path"]):
                    logger.info(f"使用缓存语音: {cache_key}")
                    return cached_result
            
            # 合成语音
            audio_file_path = await self._synthesize_audio(ssml, voice_config["name"], cache_key)
            
            if audio_file_path:
                result = {
                    "success": True,
                    "file_path": audio_file_path,
                    "emotion": emotion.value,
                    "intensity": intensity,
                    "voice_name": voice_name,
                    "language": language.value,
                    "ssml": ssml,
                    "cache_key": cache_key,
                    "duration": self._get_audio_duration(audio_file_path)
                }
                
                # 添加到缓存
                if self.config.AUDIO_CONFIG["cache_enabled"]:
                    self.voice_cache[cache_key] = result
                    self._cleanup_cache()
                
                logger.info(f"语音合成成功: {emotion.value} - {voice_name}")
                return result
            else:
                return {"success": False, "error": "语音合成失败"}
                
        except Exception as e:
            logger.error(f"情感语音合成失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_voice_config(self, voice_name: str, language: VoiceLanguage) -> Optional[Dict[str, Any]]:
        """获取语音配置"""
        voices = self.config.get_voice_by_language(language)
        return voices.get(voice_name)
    
    def _generate_ssml(self, text: str, emotion: EmotionType, intensity: float,
                      voice_config: Dict[str, Any], language: VoiceLanguage) -> str:
        """生成SSML标记"""
        try:
            # 获取情感参数
            emotion_params = self.config.get_emotion_params(emotion, intensity)
            
            # 获取语音风格
            style = self.config.get_voice_style(emotion, voice_config)
            
            # 计算风格强度
            style_degree = min(2.0, max(0.1, intensity * 1.5))
            
            # 预处理文本
            processed_text = self._preprocess_text(text, emotion)
            
            # 选择SSML模板
            template_name = "with_style"
            if emotion_params.get("emphasis", "none") != "none":
                template_name = "with_emphasis"
            
            # 如果文本包含特殊标记，使用带停顿的模板
            if self._needs_breaks(processed_text):
                template_name = "with_breaks"
                processed_text = self._add_breaks(processed_text)
            
            # 生成SSML
            ssml = self.config.SSML_TEMPLATES[template_name].format(
                language=language.value,
                voice_name=voice_config["name"],
                style=style,
                style_degree=style_degree,
                rate=emotion_params["rate"],
                pitch=emotion_params["pitch"],
                volume=emotion_params["volume"],
                emphasis=emotion_params.get("emphasis", "moderate"),
                text=processed_text,
                text_with_breaks=processed_text
            ).strip()
            
            return ssml
            
        except Exception as e:
            logger.error(f"SSML生成失败: {e}")
            # 返回基础SSML
            return self.config.SSML_TEMPLATES["basic"].format(
                language=language.value,
                voice_name=voice_config["name"],
                rate="0%",
                pitch="0%",
                volume="0%",
                text=text
            ).strip()
    
    def _preprocess_text(self, text: str, emotion: EmotionType) -> str:
        """预处理文本"""
        # 清理文本
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 根据情感添加特殊处理
        if emotion == EmotionType.SURPRISED:
            # 惊讶情感，在感叹号前添加停顿
            text = re.sub(r'!', '<break time="200ms"/>!', text)
        elif emotion == EmotionType.CONFUSED:
            # 困惑情感，在问号前添加停顿
            text = re.sub(r'\?', '<break time="300ms"/>?', text)
        elif emotion == EmotionType.THINKING:
            # 思考情感，在句号和逗号后添加停顿
            text = re.sub(r'[。，]', lambda m: m.group() + '<break time="500ms"/>', text)
        
        return text
    
    def _needs_breaks(self, text: str) -> bool:
        """检查文本是否需要停顿"""
        return '<break' in text
    
    def _add_breaks(self, text: str) -> str:
        """添加停顿标记"""
        # 如果已经包含停顿标记，直接返回
        if '<break' in text:
            return text
        
        # 在标点符号后添加适当的停顿
        text = re.sub(r'[。！]', lambda m: m.group() + '<break time="500ms"/>', text)
        text = re.sub(r'[，、]', lambda m: m.group() + '<break time="200ms"/>', text)
        text = re.sub(r'[？]', lambda m: m.group() + '<break time="300ms"/>', text)
        
        return text
    
    async def _synthesize_audio(self, ssml: str, voice_name: str, cache_key: str) -> Optional[str]:
        """合成音频文件"""
        try:
            # 生成文件路径
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"voice_{cache_key}_{timestamp}.wav"
            file_path = os.path.join(self.config.AUDIO_CONFIG["output_directory"], filename)
            
            # 使用Edge-TTS合成
            communicate = edge_tts.Communicate(ssml, voice_name)
            
            # 保存音频文件
            await communicate.save(file_path)
            
            # 验证文件是否生成成功
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                logger.debug(f"音频文件生成成功: {file_path}")
                return file_path
            else:
                logger.error("音频文件生成失败或文件为空")
                return None
                
        except Exception as e:
            logger.error(f"音频合成失败: {e}")
            return None
    
    def _generate_cache_key(self, ssml: str, voice_name: str) -> str:
        """生成缓存键"""
        content = f"{ssml}_{voice_name}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _get_audio_duration(self, file_path: str) -> float:
        """获取音频时长"""
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(file_path)
            return len(audio) / 1000.0  # 转换为秒
        except Exception as e:
            logger.warning(f"获取音频时长失败: {e}")
            return 0.0
    
    def _cleanup_cache(self):
        """清理缓存"""
        try:
            max_cache_size = self.config.AUDIO_CONFIG["max_cache_size"]
            
            if len(self.voice_cache) > max_cache_size:
                # 按时间排序，删除最旧的缓存
                sorted_cache = sorted(
                    self.voice_cache.items(),
                    key=lambda x: os.path.getmtime(x[1]["file_path"]) if os.path.exists(x[1]["file_path"]) else 0
                )
                
                # 删除超出限制的缓存
                for i in range(len(sorted_cache) - max_cache_size):
                    cache_key, cache_data = sorted_cache[i]
                    
                    # 删除文件
                    if os.path.exists(cache_data["file_path"]):
                        os.remove(cache_data["file_path"])
                    
                    # 从缓存中移除
                    del self.voice_cache[cache_key]
                
                logger.info(f"缓存清理完成，保留 {max_cache_size} 个文件")
                
        except Exception as e:
            logger.error(f"缓存清理失败: {e}")
    
    def set_default_voice(self, voice_name: str, language: VoiceLanguage = None):
        """设置默认语音"""
        if language is None:
            language = self.current_language
        
        voice_config = self._get_voice_config(voice_name, language)
        if voice_config:
            self.current_voice = voice_name
            self.current_language = language
            logger.info(f"默认语音设置为: {voice_name} ({language.value})")
        else:
            logger.warning(f"语音 {voice_name} 不可用")
    
    def set_default_emotion(self, emotion: EmotionType, intensity: float = 1.0):
        """设置默认情感"""
        self.current_emotion = emotion
        self.emotion_intensity = max(0.0, min(1.0, intensity))
        logger.info(f"默认情感设置为: {emotion.value} (强度: {self.emotion_intensity})")
    
    def get_available_voices(self, language: VoiceLanguage = None) -> Dict[str, Any]:
        """获取可用语音列表"""
        if language is None:
            language = self.current_language
        
        voices = self.config.get_voice_by_language(language)
        return {
            "language": language.value,
            "voices": voices,
            "current_voice": self.current_voice
        }
    
    def get_supported_emotions(self) -> List[Dict[str, Any]]:
        """获取支持的情感列表"""
        emotions = []
        for emotion in EmotionType:
            params = self.config.get_emotion_params(emotion)
            emotions.append({
                "name": emotion.value,
                "display_name": self._get_emotion_display_name(emotion),
                "parameters": params
            })
        return emotions
    
    def _get_emotion_display_name(self, emotion: EmotionType) -> str:
        """获取情感显示名称"""
        display_names = {
            EmotionType.NEUTRAL: "中性",
            EmotionType.HAPPY: "开心",
            EmotionType.SAD: "悲伤",
            EmotionType.ANGRY: "生气",
            EmotionType.SURPRISED: "惊讶",
            EmotionType.CONFUSED: "困惑",
            EmotionType.SHY: "害羞",
            EmotionType.EXCITED: "兴奋",
            EmotionType.WORRIED: "担心",
            EmotionType.LOVING: "喜爱",
            EmotionType.THINKING: "思考",
            EmotionType.SLEEPY: "困倦"
        }
        return display_names.get(emotion, emotion.value)
    
    async def test_voice_synthesis(self, text: str = "你好，这是语音测试。") -> Dict[str, Any]:
        """测试语音合成"""
        try:
            result = await self.synthesize_with_emotion(
                text, 
                self.current_emotion, 
                self.emotion_intensity,
                self.current_voice,
                self.current_language
            )
            
            if result["success"]:
                logger.info("语音合成测试成功")
            else:
                logger.error(f"语音合成测试失败: {result['error']}")
            
            return result
            
        except Exception as e:
            logger.error(f"语音合成测试异常: {e}")
            return {"success": False, "error": str(e)}
    
    def get_synthesis_stats(self) -> Dict[str, Any]:
        """获取合成统计信息"""
        return {
            "cache_size": len(self.voice_cache),
            "current_voice": self.current_voice,
            "current_language": self.current_language.value,
            "current_emotion": self.current_emotion.value,
            "emotion_intensity": self.emotion_intensity,
            "output_directory": self.config.AUDIO_CONFIG["output_directory"]
        }
