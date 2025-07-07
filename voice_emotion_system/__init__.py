"""
Live2D AI助手 语音情感合成系统模块

这个模块提供了完整的语音情感合成功能，包括情感语音生成、语音风格控制、
多语言支持、实时语音处理等功能，让AI助手能够用富有情感的语音与用户交流。

主要组件：
- VoiceEmotionManager: 语音情感管理器
- EmotionVoiceSynthesizer: 情感语音合成器
- VoiceStyleController: 语音风格控制器
- MultilingualVoiceManager: 多语言语音管理器
- RealtimeVoiceProcessor: 实时语音处理器
"""

from .voice_emotion_manager import VoiceEmotionManager
from .emotion_voice_synthesizer import EmotionVoiceSynthesizer
from .voice_style_controller import VoiceStyleController
from .multilingual_voice_manager import MultilingualVoiceManager
from .realtime_voice_processor import RealtimeVoiceProcessor
from .voice_emotion_config import VoiceEmotionConfig

__version__ = "1.0.0"
__author__ = "Live2D AI Assistant Team"

__all__ = [
    "VoiceEmotionManager",
    "EmotionVoiceSynthesizer",
    "VoiceStyleController",
    "MultilingualVoiceManager",
    "RealtimeVoiceProcessor",
    "VoiceEmotionConfig"
]
