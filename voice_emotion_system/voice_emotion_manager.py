"""
语音情感管理器

统一管理语音情感系统的各个组件，提供高级API接口
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Union

from .voice_emotion_config import VoiceEmotionConfig, EmotionType, VoiceLanguage, VoiceGender
from .emotion_voice_synthesizer import EmotionVoiceSynthesizer
from .voice_style_controller import VoiceStyleController
from .multilingual_voice_manager import MultilingualVoiceManager
from .realtime_voice_processor import RealtimeVoiceProcessor

# 配置日志
logger = logging.getLogger(__name__)

class VoiceEmotionManager:
    """语音情感管理器类"""
    
    def __init__(self, emotion_controller=None):
        """
        初始化语音情感管理器
        
        Args:
            emotion_controller: 情感控制器实例（可选）
        """
        self.config = VoiceEmotionConfig()
        
        # 初始化各个组件
        self.synthesizer = EmotionVoiceSynthesizer()
        self.style_controller = VoiceStyleController()
        self.multilingual_manager = MultilingualVoiceManager()
        self.realtime_processor = RealtimeVoiceProcessor(self.synthesizer)
        
        # 情感控制器集成
        self.emotion_controller = emotion_controller
        
        # 系统状态
        self.is_active = True
        self.current_settings = self._load_default_settings()
        
        # 启动实时处理器
        self.realtime_processor.start()
        
        logger.info("语音情感管理器初始化完成")
    
    def _load_default_settings(self) -> Dict[str, Any]:
        """加载默认设置"""
        return {
            "voice_style": "default",
            "language": VoiceLanguage.CHINESE,
            "voice_name": "xiaoxiao",
            "emotion_sensitivity": 1.0,
            "auto_language_detection": True,
            "realtime_synthesis": True,
            "auto_play": False
        }
    
    async def synthesize_with_emotion(self, text: str, emotion: EmotionType = None,
                                    intensity: float = None, voice_name: str = None,
                                    language: VoiceLanguage = None,
                                    style: str = None) -> Dict[str, Any]:
        """
        情感语音合成（高级接口）
        
        Args:
            text: 要合成的文本
            emotion: 情感类型（可选，自动检测）
            intensity: 情感强度（可选，自动计算）
            voice_name: 语音名称（可选，使用当前设置）
            language: 语言（可选，自动检测）
            style: 语音风格（可选，使用当前设置）
            
        Returns:
            合成结果
        """
        try:
            # 自动检测情感（如果未提供）
            if emotion is None and self.emotion_controller:
                emotion_result = self.emotion_controller.analyze_text_emotion(text)
                if emotion_result["success"]:
                    emotion = EmotionType(emotion_result["emotion_analysis"]["primary_emotion"])
                    if intensity is None:
                        intensity = emotion_result["emotion_analysis"]["intensity"]
            
            # 使用默认值
            if emotion is None:
                emotion = EmotionType.NEUTRAL
            if intensity is None:
                intensity = 1.0
            
            # 设置语音风格
            if style:
                self.style_controller.set_style(style)
            
            # 自动检测语言
            if language is None and self.current_settings["auto_language_detection"]:
                detection_result = self.multilingual_manager.detect_language(text)
                language = detection_result["language"]
            elif language is None:
                language = self.current_settings["language"]
            
            # 选择语音
            if voice_name is None:
                voice_name = self.style_controller.get_voice_for_language(language)
            
            # 调整情感强度
            adjusted_intensity = self.style_controller.adjust_emotion_intensity(emotion, intensity)
            
            # 跨语言情感调整
            if self.current_settings["language"] != language:
                adjusted_intensity = self.multilingual_manager.adjust_emotion_for_language(
                    emotion, adjusted_intensity, self.current_settings["language"], language
                )
            
            # 执行合成
            if detection_result.get("mixed_language", False):
                # 多语言文本处理
                results = await self.multilingual_manager.synthesize_multilingual_text(
                    text, emotion, adjusted_intensity, self.synthesizer
                )
                
                if results:
                    # 合并多个音频文件（简化处理，返回第一个结果）
                    main_result = results[0]
                    main_result["multilingual_results"] = results
                    return main_result
                else:
                    return {"success": False, "error": "多语言合成失败"}
            else:
                # 单语言合成
                result = await self.synthesizer.synthesize_with_emotion(
                    text, emotion, adjusted_intensity, voice_name, language
                )
                
                # 添加额外信息
                if result["success"]:
                    result["style_info"] = {
                        "current_style": self.style_controller.current_style,
                        "adjusted_intensity": adjusted_intensity,
                        "detected_language": language.value
                    }
                
                return result
            
        except Exception as e:
            logger.error(f"情感语音合成失败: {e}")
            return {"success": False, "error": str(e)}
    
    def synthesize_realtime(self, text: str, emotion: EmotionType = None,
                          intensity: float = None, priority: int = 1,
                          auto_play: bool = None) -> str:
        """
        实时语音合成
        
        Args:
            text: 要合成的文本
            emotion: 情感类型
            intensity: 情感强度
            priority: 任务优先级
            auto_play: 是否自动播放
            
        Returns:
            任务ID
        """
        try:
            # 使用默认值
            if emotion is None:
                emotion = EmotionType.NEUTRAL
            if intensity is None:
                intensity = 1.0
            if auto_play is None:
                auto_play = self.current_settings["auto_play"]
            
            # 调整情感强度
            adjusted_intensity = self.style_controller.adjust_emotion_intensity(emotion, intensity)
            
            # 提交任务
            task_id = self.realtime_processor.submit_synthesis_task(
                text, emotion, adjusted_intensity, priority, auto_play
            )
            
            logger.info(f"实时合成任务已提交: {task_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"实时语音合成失败: {e}")
            return ""
    
    def set_voice_style(self, style_name: str) -> Dict[str, Any]:
        """设置语音风格"""
        result = self.style_controller.set_style(style_name)
        if result["success"]:
            self.current_settings["voice_style"] = style_name
        return result
    
    def set_default_voice(self, voice_name: str, language: VoiceLanguage = None) -> Dict[str, Any]:
        """设置默认语音"""
        try:
            if language is None:
                language = self.current_settings["language"]
            
            self.synthesizer.set_default_voice(voice_name, language)
            self.current_settings["voice_name"] = voice_name
            self.current_settings["language"] = language
            
            return {"success": True, "message": f"默认语音设置为: {voice_name}"}
            
        except Exception as e:
            logger.error(f"设置默认语音失败: {e}")
            return {"success": False, "error": str(e)}
    
    def set_emotion_sensitivity(self, sensitivity: float) -> Dict[str, Any]:
        """设置情感敏感度"""
        try:
            sensitivity = max(0.0, min(2.0, sensitivity))
            self.style_controller.set_user_preference("emotion_sensitivity", sensitivity)
            self.current_settings["emotion_sensitivity"] = sensitivity
            
            return {"success": True, "message": f"情感敏感度设置为: {sensitivity}"}
            
        except Exception as e:
            logger.error(f"设置情感敏感度失败: {e}")
            return {"success": False, "error": str(e)}
    
    def get_available_voices(self, language: VoiceLanguage = None) -> Dict[str, Any]:
        """获取可用语音"""
        if language is None:
            language = self.current_settings["language"]
        
        return self.synthesizer.get_available_voices(language)
    
    def get_available_styles(self) -> Dict[str, Any]:
        """获取可用风格"""
        return self.style_controller.get_all_styles()
    
    def get_supported_emotions(self) -> List[Dict[str, Any]]:
        """获取支持的情感"""
        return self.synthesizer.get_supported_emotions()
    
    def get_supported_languages(self) -> List[Dict[str, Any]]:
        """获取支持的语言"""
        return self.multilingual_manager.get_supported_languages()
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        try:
            return {
                "is_active": self.is_active,
                "current_settings": self.current_settings,
                "synthesizer_stats": self.synthesizer.get_synthesis_stats(),
                "style_controller": {
                    "current_style": self.style_controller.current_style,
                    "custom_styles_count": len(self.style_controller.custom_styles)
                },
                "multilingual_stats": self.multilingual_manager.get_language_statistics(),
                "realtime_processor": self.realtime_processor.get_queue_status(),
                "performance_stats": self.realtime_processor.get_performance_stats()
            }
            
        except Exception as e:
            logger.error(f"获取系统状态失败: {e}")
            return {"error": str(e)}
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """获取任务状态"""
        return self.realtime_processor.get_task_status(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        return self.realtime_processor.cancel_task(task_id)
    
    def play_audio(self, file_path: str):
        """播放音频文件"""
        self.realtime_processor.add_to_playback_queue(file_path)
    
    def create_custom_style(self, style_name: str, style_config: Dict[str, Any]) -> Dict[str, Any]:
        """创建自定义风格"""
        return self.style_controller.create_custom_style(style_name, style_config)
    
    def delete_custom_style(self, style_name: str) -> Dict[str, Any]:
        """删除自定义风格"""
        return self.style_controller.delete_custom_style(style_name)
    
    def get_voice_recommendations(self, emotion: EmotionType,
                                language: VoiceLanguage = None,
                                gender_preference: VoiceGender = None) -> List[str]:
        """获取语音推荐"""
        if language is None:
            language = self.current_settings["language"]
        
        return self.multilingual_manager.get_voice_recommendations(
            language, emotion, gender_preference
        )
    
    def get_style_recommendations(self, emotion_history: List[EmotionType]) -> List[str]:
        """获取风格推荐"""
        return self.style_controller.get_style_recommendations(emotion_history)
    
    async def test_voice_synthesis(self, text: str = "你好，这是语音测试。") -> Dict[str, Any]:
        """测试语音合成"""
        try:
            result = await self.synthesize_with_emotion(text, EmotionType.HAPPY, 0.8)
            
            if result["success"]:
                logger.info("语音合成测试成功")
                
                # 自动播放测试音频
                if self.current_settings["auto_play"]:
                    self.play_audio(result["file_path"])
            else:
                logger.error(f"语音合成测试失败: {result['error']}")
            
            return result
            
        except Exception as e:
            logger.error(f"语音合成测试异常: {e}")
            return {"success": False, "error": str(e)}
    
    def update_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """更新设置"""
        try:
            updated_keys = []
            
            for key, value in settings.items():
                if key in self.current_settings:
                    self.current_settings[key] = value
                    updated_keys.append(key)
                    
                    # 应用特定设置
                    if key == "voice_style":
                        self.set_voice_style(value)
                    elif key == "emotion_sensitivity":
                        self.set_emotion_sensitivity(value)
                    elif key == "voice_name" or key == "language":
                        if "voice_name" in settings and "language" in settings:
                            self.set_default_voice(
                                settings["voice_name"],
                                VoiceLanguage(settings["language"])
                            )
            
            return {
                "success": True,
                "updated_keys": updated_keys,
                "message": f"已更新 {len(updated_keys)} 个设置"
            }
            
        except Exception as e:
            logger.error(f"更新设置失败: {e}")
            return {"success": False, "error": str(e)}
    
    def reset_to_default(self) -> Dict[str, Any]:
        """重置为默认设置"""
        try:
            self.current_settings = self._load_default_settings()
            self.style_controller.reset_to_default()
            
            # 重新设置默认语音
            self.synthesizer.set_default_voice(
                self.current_settings["voice_name"],
                self.current_settings["language"]
            )
            
            return {"success": True, "message": "已重置为默认设置"}
            
        except Exception as e:
            logger.error(f"重置设置失败: {e}")
            return {"success": False, "error": str(e)}
    
    def set_callbacks(self, on_synthesis_complete=None, on_playback_start=None,
                     on_playback_complete=None):
        """设置回调函数"""
        self.realtime_processor.set_callbacks(
            on_synthesis_complete, on_playback_start, on_playback_complete
        )
    
    def cleanup(self):
        """清理资源"""
        try:
            # 清理已完成的任务
            self.realtime_processor.clear_completed_tasks()
            
            # 清理语言检测缓存
            self.multilingual_manager.clear_language_cache()
            
            logger.info("语音情感系统资源清理完成")
            
        except Exception as e:
            logger.error(f"资源清理失败: {e}")
    
    def shutdown(self):
        """关闭语音情感管理器"""
        try:
            self.is_active = False
            self.realtime_processor.stop()
            self.cleanup()
            
            logger.info("语音情感管理器已关闭")
            
        except Exception as e:
            logger.error(f"关闭语音情感管理器失败: {e}")
    
    def __del__(self):
        """析构函数"""
        self.shutdown()
