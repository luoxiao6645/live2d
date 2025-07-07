"""
语音情感系统配置文件

定义了语音合成、情感映射、语音风格等核心配置信息
"""

from typing import Dict, List, Tuple, Any
from enum import Enum
import os

class EmotionType(Enum):
    """情感类型枚举"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    CONFUSED = "confused"
    SHY = "shy"
    EXCITED = "excited"
    WORRIED = "worried"
    LOVING = "loving"
    THINKING = "thinking"
    SLEEPY = "sleepy"

class VoiceGender(Enum):
    """语音性别"""
    FEMALE = "female"
    MALE = "male"
    NEUTRAL = "neutral"

class VoiceLanguage(Enum):
    """支持的语言"""
    CHINESE = "zh-CN"
    ENGLISH = "en-US"
    JAPANESE = "ja-JP"
    KOREAN = "ko-KR"

class VoiceEmotionConfig:
    """语音情感系统配置类"""
    
    # 情感到语音参数的映射
    EMOTION_VOICE_MAPPING = {
        EmotionType.NEUTRAL: {
            "rate": "0%",           # 语速调整
            "pitch": "0%",          # 音调调整
            "volume": "0%",         # 音量调整
            "emphasis": "none",     # 强调程度
            "style": "neutral"      # 语音风格
        },
        EmotionType.HAPPY: {
            "rate": "+10%",
            "pitch": "+15%",
            "volume": "+5%",
            "emphasis": "moderate",
            "style": "cheerful"
        },
        EmotionType.SAD: {
            "rate": "-20%",
            "pitch": "-10%",
            "volume": "-10%",
            "emphasis": "reduced",
            "style": "sad"
        },
        EmotionType.ANGRY: {
            "rate": "+15%",
            "pitch": "+10%",
            "volume": "+15%",
            "emphasis": "strong",
            "style": "angry"
        },
        EmotionType.SURPRISED: {
            "rate": "+5%",
            "pitch": "+20%",
            "volume": "+10%",
            "emphasis": "strong",
            "style": "excited"
        },
        EmotionType.CONFUSED: {
            "rate": "-10%",
            "pitch": "±5%",         # 音调变化
            "volume": "-5%",
            "emphasis": "moderate",
            "style": "confused"
        },
        EmotionType.SHY: {
            "rate": "-15%",
            "pitch": "-5%",
            "volume": "-15%",
            "emphasis": "reduced",
            "style": "gentle"
        },
        EmotionType.EXCITED: {
            "rate": "+20%",
            "pitch": "+20%",
            "volume": "+10%",
            "emphasis": "strong",
            "style": "excited"
        },
        EmotionType.WORRIED: {
            "rate": "-5%",
            "pitch": "-5%",
            "volume": "-5%",
            "emphasis": "moderate",
            "style": "concerned"
        },
        EmotionType.LOVING: {
            "rate": "-5%",
            "pitch": "+5%",
            "volume": "0%",
            "emphasis": "moderate",
            "style": "gentle"
        },
        EmotionType.THINKING: {
            "rate": "-15%",
            "pitch": "-2%",
            "volume": "-5%",
            "emphasis": "reduced",
            "style": "thoughtful"
        },
        EmotionType.SLEEPY: {
            "rate": "-25%",
            "pitch": "-15%",
            "volume": "-20%",
            "emphasis": "none",
            "style": "sleepy"
        }
    }
    
    # 中文语音角色配置
    CHINESE_VOICES = {
        "xiaoxiao": {
            "name": "zh-CN-XiaoxiaoNeural",
            "gender": VoiceGender.FEMALE,
            "description": "晓晓 - 温柔女声",
            "styles": ["assistant", "chat", "cheerful", "sad", "angry", "fearful", "disgruntled", "serious", "affectionate", "gentle", "lyrical"]
        },
        "yunxi": {
            "name": "zh-CN-YunxiNeural",
            "gender": VoiceGender.MALE,
            "description": "云希 - 成熟男声",
            "styles": ["assistant", "chat", "cheerful", "sad", "angry", "fearful", "disgruntled", "serious", "depressed", "embarrassed"]
        },
        "xiaoyi": {
            "name": "zh-CN-XiaoyiNeural",
            "gender": VoiceGender.FEMALE,
            "description": "晓伊 - 甜美女声",
            "styles": ["assistant", "chat", "cheerful", "sad", "angry", "fearful", "disgruntled", "serious", "affectionate", "gentle"]
        },
        "yunjian": {
            "name": "zh-CN-YunjianNeural",
            "gender": VoiceGender.MALE,
            "description": "云健 - 活力男声",
            "styles": ["assistant", "chat", "cheerful", "sad", "angry", "fearful", "disgruntled", "serious", "sports_commentary"]
        },
        "xiaomeng": {
            "name": "zh-CN-XiaomengNeural",
            "gender": VoiceGender.FEMALE,
            "description": "晓梦 - 可爱女声",
            "styles": ["assistant", "chat", "cheerful", "sad", "angry", "fearful", "disgruntled", "serious", "calm"]
        }
    }
    
    # 英文语音角色配置
    ENGLISH_VOICES = {
        "aria": {
            "name": "en-US-AriaNeural",
            "gender": VoiceGender.FEMALE,
            "description": "Aria - Professional Female",
            "styles": ["assistant", "chat", "cheerful", "sad", "angry", "fearful", "disgruntled", "serious", "excited", "friendly", "hopeful", "shouting", "terrified", "unfriendly", "whispering"]
        },
        "davis": {
            "name": "en-US-DavisNeural",
            "gender": VoiceGender.MALE,
            "description": "Davis - Confident Male",
            "styles": ["assistant", "chat", "cheerful", "sad", "angry", "fearful", "disgruntled", "serious", "excited", "friendly", "hopeful", "shouting", "terrified", "unfriendly", "whispering"]
        },
        "jenny": {
            "name": "en-US-JennyNeural",
            "gender": VoiceGender.FEMALE,
            "description": "Jenny - Warm Female",
            "styles": ["assistant", "chat", "cheerful", "sad", "angry", "fearful", "disgruntled", "serious", "excited", "friendly", "hopeful", "shouting", "terrified", "unfriendly", "whispering"]
        },
        "guy": {
            "name": "en-US-GuyNeural",
            "gender": VoiceGender.MALE,
            "description": "Guy - Natural Male",
            "styles": ["assistant", "chat", "cheerful", "sad", "angry", "fearful", "disgruntled", "serious", "excited", "friendly", "hopeful", "shouting", "terrified", "unfriendly", "whispering"]
        }
    }
    
    # 情感到语音风格的映射
    EMOTION_STYLE_MAPPING = {
        EmotionType.NEUTRAL: "assistant",
        EmotionType.HAPPY: "cheerful",
        EmotionType.SAD: "sad",
        EmotionType.ANGRY: "angry",
        EmotionType.SURPRISED: "excited",
        EmotionType.CONFUSED: "disgruntled",
        EmotionType.SHY: "gentle",
        EmotionType.EXCITED: "excited",
        EmotionType.WORRIED: "fearful",
        EmotionType.LOVING: "affectionate",
        EmotionType.THINKING: "serious",
        EmotionType.SLEEPY: "calm"
    }
    
    # SSML模板配置
    SSML_TEMPLATES = {
        "basic": """
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{language}">
    <voice name="{voice_name}">
        <prosody rate="{rate}" pitch="{pitch}" volume="{volume}">
            {text}
        </prosody>
    </voice>
</speak>
""",
        
        "with_style": """
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{language}">
    <voice name="{voice_name}">
        <mstts:express-as style="{style}" styledegree="{style_degree}">
            <prosody rate="{rate}" pitch="{pitch}" volume="{volume}">
                {text}
            </prosody>
        </mstts:express-as>
    </voice>
</speak>
""",
        
        "with_emphasis": """
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{language}">
    <voice name="{voice_name}">
        <mstts:express-as style="{style}" styledegree="{style_degree}">
            <prosody rate="{rate}" pitch="{pitch}" volume="{volume}">
                <emphasis level="{emphasis}">{text}</emphasis>
            </prosody>
        </mstts:express-as>
    </voice>
</speak>
""",
        
        "with_breaks": """
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{language}">
    <voice name="{voice_name}">
        <mstts:express-as style="{style}" styledegree="{style_degree}">
            <prosody rate="{rate}" pitch="{pitch}" volume="{volume}">
                {text_with_breaks}
            </prosody>
        </mstts:express-as>
    </voice>
</speak>
"""
    }
    
    # 音频处理配置
    AUDIO_CONFIG = {
        "sample_rate": 24000,
        "bit_depth": 16,
        "channels": 1,
        "format": "wav",
        "quality": "high",
        "cache_enabled": True,
        "cache_ttl": 3600,  # 1小时
        "max_cache_size": 100,  # 最大缓存文件数
        "output_directory": "./audio_output",
        "temp_directory": "./temp_audio"
    }
    
    # 实时处理配置
    REALTIME_CONFIG = {
        "chunk_size": 1024,
        "buffer_size": 4096,
        "max_queue_size": 10,
        "processing_timeout": 30,
        "streaming_enabled": True,
        "low_latency_mode": True
    }
    
    # 语言检测配置
    LANGUAGE_DETECTION = {
        "enabled": True,
        "confidence_threshold": 0.8,
        "default_language": VoiceLanguage.CHINESE,
        "mixed_language_handling": "auto_switch",  # auto_switch, primary_only, separate
        "chinese_keywords": [
            "你好", "谢谢", "再见", "请问", "什么", "怎么", "为什么", "哪里", "什么时候", "多少"
        ],
        "english_keywords": [
            "hello", "thank", "goodbye", "please", "what", "how", "why", "where", "when", "much"
        ]
    }
    
    # 性能配置
    PERFORMANCE_CONFIG = {
        "max_concurrent_synthesis": 3,
        "synthesis_timeout": 30,
        "retry_attempts": 2,
        "error_recovery_enabled": True,
        "fallback_voice": "xiaoxiao",
        "fallback_language": VoiceLanguage.CHINESE
    }
    
    @classmethod
    def get_voice_by_language(cls, language: VoiceLanguage) -> Dict[str, Any]:
        """根据语言获取语音配置"""
        if language == VoiceLanguage.CHINESE:
            return cls.CHINESE_VOICES
        elif language == VoiceLanguage.ENGLISH:
            return cls.ENGLISH_VOICES
        else:
            return cls.CHINESE_VOICES  # 默认返回中文
    
    @classmethod
    def get_emotion_params(cls, emotion: EmotionType, intensity: float = 1.0) -> Dict[str, Any]:
        """
        获取情感对应的语音参数
        
        Args:
            emotion: 情感类型
            intensity: 情感强度 (0.0-1.0)
            
        Returns:
            语音参数字典
        """
        base_params = cls.EMOTION_VOICE_MAPPING.get(emotion, cls.EMOTION_VOICE_MAPPING[EmotionType.NEUTRAL])
        
        # 根据强度调整参数
        adjusted_params = base_params.copy()
        
        # 调整数值参数
        for param in ["rate", "pitch", "volume"]:
            if param in adjusted_params:
                value = adjusted_params[param]
                if value.endswith('%') and value != "0%":
                    # 提取数值并应用强度
                    if value.startswith('+'):
                        num_value = float(value[1:-1])
                        adjusted_params[param] = f"+{num_value * intensity:.0f}%"
                    elif value.startswith('-'):
                        num_value = float(value[1:-1])
                        adjusted_params[param] = f"-{num_value * intensity:.0f}%"
        
        return adjusted_params
    
    @classmethod
    def get_voice_style(cls, emotion: EmotionType, voice_config: Dict[str, Any]) -> str:
        """
        获取情感对应的语音风格
        
        Args:
            emotion: 情感类型
            voice_config: 语音配置
            
        Returns:
            语音风格名称
        """
        preferred_style = cls.EMOTION_STYLE_MAPPING.get(emotion, "assistant")
        available_styles = voice_config.get("styles", ["assistant"])
        
        # 如果首选风格可用，使用首选风格
        if preferred_style in available_styles:
            return preferred_style
        
        # 否则使用默认风格
        return available_styles[0] if available_styles else "assistant"
    
    @classmethod
    def create_output_directory(cls):
        """创建输出目录"""
        import os
        os.makedirs(cls.AUDIO_CONFIG["output_directory"], exist_ok=True)
        os.makedirs(cls.AUDIO_CONFIG["temp_directory"], exist_ok=True)
    
    @classmethod
    def detect_language(cls, text: str) -> VoiceLanguage:
        """
        简单的语言检测
        
        Args:
            text: 要检测的文本
            
        Returns:
            检测到的语言
        """
        if not cls.LANGUAGE_DETECTION["enabled"]:
            return cls.LANGUAGE_DETECTION["default_language"]
        
        # 统计中英文字符
        chinese_count = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        english_count = sum(1 for char in text if char.isalpha() and ord(char) < 128)
        
        # 基于字符比例判断
        total_chars = len(text)
        if total_chars == 0:
            return cls.LANGUAGE_DETECTION["default_language"]
        
        chinese_ratio = chinese_count / total_chars
        english_ratio = english_count / total_chars
        
        if chinese_ratio > 0.3:
            return VoiceLanguage.CHINESE
        elif english_ratio > 0.5:
            return VoiceLanguage.ENGLISH
        else:
            return cls.LANGUAGE_DETECTION["default_language"]
