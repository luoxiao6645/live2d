"""
多语言语音管理器

负责多语言语音支持、语言检测、跨语言情感映射等功能
"""

import logging
import re
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter

from .voice_emotion_config import VoiceEmotionConfig, EmotionType, VoiceLanguage, VoiceGender

# 配置日志
logger = logging.getLogger(__name__)

class MultilingualVoiceManager:
    """多语言语音管理器类"""
    
    def __init__(self):
        """初始化多语言语音管理器"""
        self.config = VoiceEmotionConfig()
        
        # 语言检测缓存
        self.language_cache = {}
        
        # 跨语言情感映射
        self.cross_language_emotion_mapping = self._init_cross_language_mapping()
        
        # 语言特定的语音特征
        self.language_features = self._init_language_features()
        
        logger.info("多语言语音管理器初始化完成")
    
    def _init_cross_language_mapping(self) -> Dict[str, Any]:
        """初始化跨语言情感映射"""
        return {
            # 中文到英文的情感强度调整
            "zh_to_en": {
                EmotionType.HAPPY: {"intensity_adjustment": 0.9},      # 英文表达更含蓄
                EmotionType.SAD: {"intensity_adjustment": 1.1},        # 英文悲伤表达更强烈
                EmotionType.ANGRY: {"intensity_adjustment": 0.8},      # 英文愤怒表达更克制
                EmotionType.EXCITED: {"intensity_adjustment": 1.2},    # 英文兴奋表达更夸张
                EmotionType.SHY: {"intensity_adjustment": 0.7},        # 英文害羞表达更微妙
            },
            
            # 英文到中文的情感强度调整
            "en_to_zh": {
                EmotionType.HAPPY: {"intensity_adjustment": 1.1},      # 中文表达更直接
                EmotionType.SAD: {"intensity_adjustment": 0.9},        # 中文悲伤表达更含蓄
                EmotionType.ANGRY: {"intensity_adjustment": 1.2},      # 中文愤怒表达更强烈
                EmotionType.EXCITED: {"intensity_adjustment": 0.8},    # 中文兴奋表达更内敛
                EmotionType.SHY: {"intensity_adjustment": 1.3},        # 中文害羞表达更明显
            }
        }
    
    def _init_language_features(self) -> Dict[VoiceLanguage, Dict[str, Any]]:
        """初始化语言特定特征"""
        return {
            VoiceLanguage.CHINESE: {
                "typical_rate_range": (-20, 20),      # 中文语速范围
                "typical_pitch_range": (-15, 25),     # 中文音调范围
                "emotion_expression": "direct",        # 情感表达方式：直接
                "pause_patterns": {
                    "sentence_end": 500,               # 句末停顿(ms)
                    "comma": 200,                      # 逗号停顿
                    "question": 300                    # 疑问停顿
                },
                "emphasis_style": "tone_based"         # 强调方式：基于声调
            },
            
            VoiceLanguage.ENGLISH: {
                "typical_rate_range": (-15, 25),      # 英文语速范围
                "typical_pitch_range": (-10, 30),     # 英文音调范围
                "emotion_expression": "subtle",        # 情感表达方式：微妙
                "pause_patterns": {
                    "sentence_end": 400,               # 句末停顿(ms)
                    "comma": 150,                      # 逗号停顿
                    "question": 250                    # 疑问停顿
                },
                "emphasis_style": "stress_based"       # 强调方式：基于重音
            }
        }
    
    def detect_language(self, text: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        检测文本语言
        
        Args:
            text: 要检测的文本
            use_cache: 是否使用缓存
            
        Returns:
            检测结果
        """
        try:
            # 生成缓存键
            cache_key = hash(text) if use_cache else None
            
            # 检查缓存
            if use_cache and cache_key in self.language_cache:
                return self.language_cache[cache_key]
            
            # 执行语言检测
            result = self._perform_language_detection(text)
            
            # 缓存结果
            if use_cache and cache_key:
                self.language_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"语言检测失败: {e}")
            return {
                "language": VoiceLanguage.CHINESE,
                "confidence": 0.5,
                "mixed_language": False,
                "segments": []
            }
    
    def _perform_language_detection(self, text: str) -> Dict[str, Any]:
        """执行语言检测"""
        # 清理文本
        cleaned_text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
        
        if not cleaned_text:
            return {
                "language": VoiceLanguage.CHINESE,
                "confidence": 0.5,
                "mixed_language": False,
                "segments": []
            }
        
        # 统计字符类型
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', cleaned_text))
        english_chars = len(re.findall(r'[a-zA-Z]', cleaned_text))
        total_chars = len(cleaned_text.replace(' ', ''))
        
        if total_chars == 0:
            return {
                "language": VoiceLanguage.CHINESE,
                "confidence": 0.5,
                "mixed_language": False,
                "segments": []
            }
        
        # 计算比例
        chinese_ratio = chinese_chars / total_chars
        english_ratio = english_chars / total_chars
        
        # 判断主要语言
        if chinese_ratio > 0.6:
            primary_language = VoiceLanguage.CHINESE
            confidence = min(0.95, chinese_ratio + 0.2)
        elif english_ratio > 0.6:
            primary_language = VoiceLanguage.ENGLISH
            confidence = min(0.95, english_ratio + 0.2)
        elif chinese_ratio > english_ratio:
            primary_language = VoiceLanguage.CHINESE
            confidence = 0.6
        else:
            primary_language = VoiceLanguage.ENGLISH
            confidence = 0.6
        
        # 检测混合语言
        mixed_language = chinese_ratio > 0.2 and english_ratio > 0.2
        
        # 分割混合语言文本
        segments = []
        if mixed_language:
            segments = self._segment_mixed_language_text(text)
        
        return {
            "language": primary_language,
            "confidence": confidence,
            "mixed_language": mixed_language,
            "chinese_ratio": chinese_ratio,
            "english_ratio": english_ratio,
            "segments": segments
        }
    
    def _segment_mixed_language_text(self, text: str) -> List[Dict[str, Any]]:
        """分割混合语言文本"""
        segments = []
        current_segment = ""
        current_language = None
        
        # 逐字符分析
        for char in text:
            if '\u4e00' <= char <= '\u9fff':  # 中文字符
                if current_language == VoiceLanguage.ENGLISH and current_segment.strip():
                    # 保存英文段落
                    segments.append({
                        "text": current_segment.strip(),
                        "language": VoiceLanguage.ENGLISH
                    })
                    current_segment = ""
                current_language = VoiceLanguage.CHINESE
                current_segment += char
            elif char.isalpha():  # 英文字符
                if current_language == VoiceLanguage.CHINESE and current_segment.strip():
                    # 保存中文段落
                    segments.append({
                        "text": current_segment.strip(),
                        "language": VoiceLanguage.CHINESE
                    })
                    current_segment = ""
                current_language = VoiceLanguage.ENGLISH
                current_segment += char
            else:  # 其他字符（标点、空格等）
                current_segment += char
        
        # 保存最后一个段落
        if current_segment.strip() and current_language:
            segments.append({
                "text": current_segment.strip(),
                "language": current_language
            })
        
        return segments
    
    def adjust_emotion_for_language(self, emotion: EmotionType, intensity: float,
                                  source_language: VoiceLanguage,
                                  target_language: VoiceLanguage) -> float:
        """
        根据语言调整情感强度
        
        Args:
            emotion: 情感类型
            intensity: 原始强度
            source_language: 源语言
            target_language: 目标语言
            
        Returns:
            调整后的强度
        """
        try:
            # 如果是同一语言，不需要调整
            if source_language == target_language:
                return intensity
            
            # 获取映射键
            if source_language == VoiceLanguage.CHINESE and target_language == VoiceLanguage.ENGLISH:
                mapping_key = "zh_to_en"
            elif source_language == VoiceLanguage.ENGLISH and target_language == VoiceLanguage.CHINESE:
                mapping_key = "en_to_zh"
            else:
                # 其他语言组合，暂时不调整
                return intensity
            
            # 获取调整参数
            mapping = self.cross_language_emotion_mapping.get(mapping_key, {})
            emotion_mapping = mapping.get(emotion, {})
            adjustment = emotion_mapping.get("intensity_adjustment", 1.0)
            
            # 应用调整
            adjusted_intensity = intensity * adjustment
            
            # 确保在有效范围内
            return max(0.0, min(1.0, adjusted_intensity))
            
        except Exception as e:
            logger.warning(f"情感强度调整失败: {e}")
            return intensity
    
    def get_language_specific_prosody(self, language: VoiceLanguage,
                                    emotion: EmotionType) -> Dict[str, Any]:
        """
        获取语言特定的韵律参数
        
        Args:
            language: 语言
            emotion: 情感
            
        Returns:
            韵律参数
        """
        try:
            features = self.language_features.get(language, {})
            
            # 基础韵律范围
            rate_range = features.get("typical_rate_range", (-15, 20))
            pitch_range = features.get("typical_pitch_range", (-10, 25))
            
            # 根据情感调整
            prosody_adjustments = {
                EmotionType.HAPPY: {
                    "rate_bias": 5,
                    "pitch_bias": 10
                },
                EmotionType.SAD: {
                    "rate_bias": -10,
                    "pitch_bias": -8
                },
                EmotionType.EXCITED: {
                    "rate_bias": 15,
                    "pitch_bias": 15
                },
                EmotionType.CALM: {
                    "rate_bias": -5,
                    "pitch_bias": -3
                }
            }
            
            adjustment = prosody_adjustments.get(emotion, {"rate_bias": 0, "pitch_bias": 0})
            
            return {
                "rate_range": (
                    rate_range[0] + adjustment["rate_bias"],
                    rate_range[1] + adjustment["rate_bias"]
                ),
                "pitch_range": (
                    pitch_range[0] + adjustment["pitch_bias"],
                    pitch_range[1] + adjustment["pitch_bias"]
                ),
                "pause_patterns": features.get("pause_patterns", {}),
                "emphasis_style": features.get("emphasis_style", "tone_based")
            }
            
        except Exception as e:
            logger.error(f"获取语言特定韵律失败: {e}")
            return {
                "rate_range": (-15, 20),
                "pitch_range": (-10, 25),
                "pause_patterns": {},
                "emphasis_style": "tone_based"
            }
    
    async def synthesize_multilingual_text(self, text: str, emotion: EmotionType,
                                         intensity: float, synthesizer) -> List[Dict[str, Any]]:
        """
        合成多语言文本
        
        Args:
            text: 文本内容
            emotion: 情感类型
            intensity: 情感强度
            synthesizer: 语音合成器实例
            
        Returns:
            合成结果列表
        """
        try:
            # 检测语言
            detection_result = self.detect_language(text)
            
            results = []
            
            if detection_result["mixed_language"]:
                # 处理混合语言文本
                for segment in detection_result["segments"]:
                    segment_text = segment["text"]
                    segment_language = segment["language"]
                    
                    # 调整情感强度
                    adjusted_intensity = self.adjust_emotion_for_language(
                        emotion, intensity,
                        detection_result["language"],
                        segment_language
                    )
                    
                    # 合成语音
                    result = await synthesizer.synthesize_with_emotion(
                        segment_text, emotion, adjusted_intensity,
                        language=segment_language
                    )
                    
                    if result["success"]:
                        result["segment_info"] = {
                            "original_text": segment_text,
                            "detected_language": segment_language.value,
                            "adjusted_intensity": adjusted_intensity
                        }
                        results.append(result)
                    else:
                        logger.warning(f"段落合成失败: {segment_text}")
            else:
                # 处理单一语言文本
                primary_language = detection_result["language"]
                
                result = await synthesizer.synthesize_with_emotion(
                    text, emotion, intensity,
                    language=primary_language
                )
                
                if result["success"]:
                    result["detection_info"] = detection_result
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"多语言文本合成失败: {e}")
            return []
    
    def get_voice_recommendations(self, language: VoiceLanguage,
                                emotion: EmotionType,
                                gender_preference: VoiceGender = None) -> List[str]:
        """
        获取语音推荐
        
        Args:
            language: 语言
            emotion: 情感
            gender_preference: 性别偏好
            
        Returns:
            推荐的语音列表
        """
        try:
            voices = self.config.get_voice_by_language(language)
            recommendations = []
            
            # 根据情感和性别筛选语音
            for voice_name, voice_config in voices.items():
                # 性别筛选
                if gender_preference and voice_config["gender"] != gender_preference:
                    continue
                
                # 检查是否支持所需的情感风格
                required_style = self.config.get_voice_style(emotion, voice_config)
                if required_style in voice_config.get("styles", []):
                    recommendations.append(voice_name)
            
            # 如果没有找到合适的语音，返回所有可用语音
            if not recommendations:
                recommendations = list(voices.keys())
            
            return recommendations[:3]  # 返回前3个推荐
            
        except Exception as e:
            logger.error(f"获取语音推荐失败: {e}")
            return ["xiaoxiao"]  # 返回默认语音
    
    def get_supported_languages(self) -> List[Dict[str, Any]]:
        """获取支持的语言列表"""
        return [
            {
                "code": VoiceLanguage.CHINESE.value,
                "name": "中文",
                "native_name": "中文",
                "voices_count": len(self.config.CHINESE_VOICES)
            },
            {
                "code": VoiceLanguage.ENGLISH.value,
                "name": "English",
                "native_name": "English",
                "voices_count": len(self.config.ENGLISH_VOICES)
            }
        ]
    
    def get_language_statistics(self) -> Dict[str, Any]:
        """获取语言使用统计"""
        return {
            "cache_size": len(self.language_cache),
            "supported_languages": len(self.language_features),
            "cross_language_mappings": len(self.cross_language_emotion_mapping)
        }
    
    def clear_language_cache(self):
        """清理语言检测缓存"""
        self.language_cache.clear()
        logger.info("语言检测缓存已清理")
