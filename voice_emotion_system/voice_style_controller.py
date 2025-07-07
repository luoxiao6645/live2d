"""
语音风格控制器

负责管理语音风格、角色切换、个性化设置等功能
"""

import logging
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from .voice_emotion_config import VoiceEmotionConfig, EmotionType, VoiceLanguage, VoiceGender

# 配置日志
logger = logging.getLogger(__name__)

class VoiceStyleController:
    """语音风格控制器类"""
    
    def __init__(self):
        """初始化语音风格控制器"""
        self.config = VoiceEmotionConfig()
        
        # 风格预设
        self.style_presets = self._load_style_presets()
        
        # 当前风格设置
        self.current_style = "default"
        self.custom_styles = {}
        
        # 用户偏好
        self.user_preferences = self._load_user_preferences()
        
        logger.info("语音风格控制器初始化完成")
    
    def _load_style_presets(self) -> Dict[str, Any]:
        """加载风格预设"""
        return {
            "default": {
                "name": "默认风格",
                "description": "平衡的语音风格，适合日常对话",
                "voice_preferences": {
                    VoiceLanguage.CHINESE: "xiaoxiao",
                    VoiceLanguage.ENGLISH: "aria"
                },
                "emotion_adjustments": {
                    EmotionType.HAPPY: {"intensity_multiplier": 1.0},
                    EmotionType.SAD: {"intensity_multiplier": 0.8},
                    EmotionType.EXCITED: {"intensity_multiplier": 0.9}
                },
                "prosody_settings": {
                    "base_rate": "0%",
                    "base_pitch": "0%",
                    "base_volume": "0%"
                }
            },
            
            "gentle": {
                "name": "温柔风格",
                "description": "温和柔软的语音风格，适合安慰和关怀",
                "voice_preferences": {
                    VoiceLanguage.CHINESE: "xiaoyi",
                    VoiceLanguage.ENGLISH: "jenny"
                },
                "emotion_adjustments": {
                    EmotionType.HAPPY: {"intensity_multiplier": 0.8},
                    EmotionType.SAD: {"intensity_multiplier": 1.2},
                    EmotionType.ANGRY: {"intensity_multiplier": 0.3},
                    EmotionType.LOVING: {"intensity_multiplier": 1.3}
                },
                "prosody_settings": {
                    "base_rate": "-10%",
                    "base_pitch": "+5%",
                    "base_volume": "-5%"
                }
            },
            
            "energetic": {
                "name": "活力风格",
                "description": "充满活力的语音风格，适合激励和鼓舞",
                "voice_preferences": {
                    VoiceLanguage.CHINESE: "yunjian",
                    VoiceLanguage.ENGLISH: "davis"
                },
                "emotion_adjustments": {
                    EmotionType.HAPPY: {"intensity_multiplier": 1.3},
                    EmotionType.EXCITED: {"intensity_multiplier": 1.4},
                    EmotionType.SAD: {"intensity_multiplier": 0.6},
                    EmotionType.SLEEPY: {"intensity_multiplier": 0.4}
                },
                "prosody_settings": {
                    "base_rate": "+15%",
                    "base_pitch": "+10%",
                    "base_volume": "+10%"
                }
            },
            
            "calm": {
                "name": "平静风格",
                "description": "沉稳平静的语音风格，适合专业和正式场合",
                "voice_preferences": {
                    VoiceLanguage.CHINESE: "yunxi",
                    VoiceLanguage.ENGLISH: "guy"
                },
                "emotion_adjustments": {
                    EmotionType.EXCITED: {"intensity_multiplier": 0.5},
                    EmotionType.ANGRY: {"intensity_multiplier": 0.4},
                    EmotionType.THINKING: {"intensity_multiplier": 1.2},
                    EmotionType.WORRIED: {"intensity_multiplier": 0.7}
                },
                "prosody_settings": {
                    "base_rate": "-5%",
                    "base_pitch": "-5%",
                    "base_volume": "0%"
                }
            },
            
            "cute": {
                "name": "可爱风格",
                "description": "甜美可爱的语音风格，适合轻松愉快的交流",
                "voice_preferences": {
                    VoiceLanguage.CHINESE: "xiaomeng",
                    VoiceLanguage.ENGLISH: "aria"
                },
                "emotion_adjustments": {
                    EmotionType.HAPPY: {"intensity_multiplier": 1.2},
                    EmotionType.SHY: {"intensity_multiplier": 1.4},
                    EmotionType.SURPRISED: {"intensity_multiplier": 1.3},
                    EmotionType.ANGRY: {"intensity_multiplier": 0.2}
                },
                "prosody_settings": {
                    "base_rate": "+5%",
                    "base_pitch": "+15%",
                    "base_volume": "+5%"
                }
            }
        }
    
    def _load_user_preferences(self) -> Dict[str, Any]:
        """加载用户偏好设置"""
        preferences_file = "user_voice_preferences.json"
        
        try:
            if os.path.exists(preferences_file):
                with open(preferences_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"加载用户偏好失败: {e}")
        
        # 返回默认偏好
        return {
            "preferred_style": "default",
            "preferred_language": VoiceLanguage.CHINESE.value,
            "preferred_voices": {
                VoiceLanguage.CHINESE.value: "xiaoxiao",
                VoiceLanguage.ENGLISH.value: "aria"
            },
            "emotion_sensitivity": 1.0,
            "custom_adjustments": {}
        }
    
    def _save_user_preferences(self):
        """保存用户偏好设置"""
        preferences_file = "user_voice_preferences.json"
        
        try:
            with open(preferences_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_preferences, f, ensure_ascii=False, indent=2)
            logger.info("用户偏好保存成功")
        except Exception as e:
            logger.error(f"保存用户偏好失败: {e}")
    
    def set_style(self, style_name: str) -> Dict[str, Any]:
        """
        设置语音风格
        
        Args:
            style_name: 风格名称
            
        Returns:
            设置结果
        """
        try:
            if style_name in self.style_presets:
                self.current_style = style_name
                self.user_preferences["preferred_style"] = style_name
                self._save_user_preferences()
                
                logger.info(f"语音风格设置为: {style_name}")
                return {
                    "success": True,
                    "style": style_name,
                    "description": self.style_presets[style_name]["description"]
                }
            elif style_name in self.custom_styles:
                self.current_style = style_name
                logger.info(f"自定义语音风格设置为: {style_name}")
                return {
                    "success": True,
                    "style": style_name,
                    "description": self.custom_styles[style_name].get("description", "自定义风格")
                }
            else:
                return {
                    "success": False,
                    "error": f"风格 '{style_name}' 不存在"
                }
                
        except Exception as e:
            logger.error(f"设置语音风格失败: {e}")
            return {"success": False, "error": str(e)}
    
    def get_current_style(self) -> Dict[str, Any]:
        """获取当前风格设置"""
        if self.current_style in self.style_presets:
            style_config = self.style_presets[self.current_style]
        elif self.current_style in self.custom_styles:
            style_config = self.custom_styles[self.current_style]
        else:
            style_config = self.style_presets["default"]
        
        return {
            "name": self.current_style,
            "config": style_config
        }
    
    def get_voice_for_language(self, language: VoiceLanguage) -> str:
        """根据当前风格和语言获取推荐语音"""
        current_style_config = self.get_current_style()["config"]
        
        # 优先使用风格中的语音偏好
        voice_preferences = current_style_config.get("voice_preferences", {})
        if language in voice_preferences:
            return voice_preferences[language]
        
        # 使用用户偏好
        preferred_voices = self.user_preferences.get("preferred_voices", {})
        if language.value in preferred_voices:
            return preferred_voices[language.value]
        
        # 使用默认语音
        default_voices = {
            VoiceLanguage.CHINESE: "xiaoxiao",
            VoiceLanguage.ENGLISH: "aria"
        }
        return default_voices.get(language, "xiaoxiao")
    
    def adjust_emotion_intensity(self, emotion: EmotionType, base_intensity: float) -> float:
        """根据当前风格调整情感强度"""
        current_style_config = self.get_current_style()["config"]
        
        # 获取情感调整设置
        emotion_adjustments = current_style_config.get("emotion_adjustments", {})
        
        if emotion in emotion_adjustments:
            multiplier = emotion_adjustments[emotion].get("intensity_multiplier", 1.0)
            adjusted_intensity = base_intensity * multiplier
        else:
            adjusted_intensity = base_intensity
        
        # 应用用户的情感敏感度设置
        sensitivity = self.user_preferences.get("emotion_sensitivity", 1.0)
        final_intensity = adjusted_intensity * sensitivity
        
        # 确保在有效范围内
        return max(0.0, min(1.0, final_intensity))
    
    def get_prosody_adjustments(self) -> Dict[str, str]:
        """获取韵律调整参数"""
        current_style_config = self.get_current_style()["config"]
        return current_style_config.get("prosody_settings", {
            "base_rate": "0%",
            "base_pitch": "0%",
            "base_volume": "0%"
        })
    
    def create_custom_style(self, style_name: str, style_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建自定义风格
        
        Args:
            style_name: 风格名称
            style_config: 风格配置
            
        Returns:
            创建结果
        """
        try:
            # 验证配置格式
            required_fields = ["name", "description"]
            for field in required_fields:
                if field not in style_config:
                    return {"success": False, "error": f"缺少必需字段: {field}"}
            
            # 添加时间戳
            style_config["created_at"] = datetime.now().isoformat()
            style_config["type"] = "custom"
            
            # 保存自定义风格
            self.custom_styles[style_name] = style_config
            
            # 保存到文件
            self._save_custom_styles()
            
            logger.info(f"自定义风格创建成功: {style_name}")
            return {
                "success": True,
                "style_name": style_name,
                "message": "自定义风格创建成功"
            }
            
        except Exception as e:
            logger.error(f"创建自定义风格失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _save_custom_styles(self):
        """保存自定义风格"""
        custom_styles_file = "custom_voice_styles.json"
        
        try:
            with open(custom_styles_file, 'w', encoding='utf-8') as f:
                json.dump(self.custom_styles, f, ensure_ascii=False, indent=2)
            logger.info("自定义风格保存成功")
        except Exception as e:
            logger.error(f"保存自定义风格失败: {e}")
    
    def _load_custom_styles(self):
        """加载自定义风格"""
        custom_styles_file = "custom_voice_styles.json"
        
        try:
            if os.path.exists(custom_styles_file):
                with open(custom_styles_file, 'r', encoding='utf-8') as f:
                    self.custom_styles = json.load(f)
                logger.info(f"加载了 {len(self.custom_styles)} 个自定义风格")
        except Exception as e:
            logger.warning(f"加载自定义风格失败: {e}")
            self.custom_styles = {}
    
    def get_all_styles(self) -> Dict[str, Any]:
        """获取所有可用风格"""
        all_styles = {}
        
        # 添加预设风格
        for name, config in self.style_presets.items():
            all_styles[name] = {
                **config,
                "type": "preset"
            }
        
        # 添加自定义风格
        for name, config in self.custom_styles.items():
            all_styles[name] = {
                **config,
                "type": "custom"
            }
        
        return {
            "current_style": self.current_style,
            "styles": all_styles
        }
    
    def delete_custom_style(self, style_name: str) -> Dict[str, Any]:
        """删除自定义风格"""
        try:
            if style_name in self.custom_styles:
                del self.custom_styles[style_name]
                self._save_custom_styles()
                
                # 如果删除的是当前风格，切换到默认风格
                if self.current_style == style_name:
                    self.set_style("default")
                
                logger.info(f"自定义风格删除成功: {style_name}")
                return {"success": True, "message": "自定义风格删除成功"}
            else:
                return {"success": False, "error": "风格不存在或不是自定义风格"}
                
        except Exception as e:
            logger.error(f"删除自定义风格失败: {e}")
            return {"success": False, "error": str(e)}
    
    def set_user_preference(self, key: str, value: Any) -> Dict[str, Any]:
        """设置用户偏好"""
        try:
            self.user_preferences[key] = value
            self._save_user_preferences()
            
            logger.info(f"用户偏好设置成功: {key} = {value}")
            return {"success": True, "message": "偏好设置成功"}
            
        except Exception as e:
            logger.error(f"设置用户偏好失败: {e}")
            return {"success": False, "error": str(e)}
    
    def get_user_preferences(self) -> Dict[str, Any]:
        """获取用户偏好"""
        return self.user_preferences.copy()
    
    def reset_to_default(self) -> Dict[str, Any]:
        """重置为默认设置"""
        try:
            self.current_style = "default"
            self.user_preferences = self._load_user_preferences()
            
            logger.info("语音风格重置为默认设置")
            return {"success": True, "message": "已重置为默认设置"}
            
        except Exception as e:
            logger.error(f"重置设置失败: {e}")
            return {"success": False, "error": str(e)}
    
    def get_style_recommendations(self, emotion_history: List[EmotionType]) -> List[str]:
        """根据情感历史推荐风格"""
        if not emotion_history:
            return ["default"]
        
        # 统计情感类型
        emotion_counts = {}
        for emotion in emotion_history:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # 根据主要情感推荐风格
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        
        recommendations = []
        
        if dominant_emotion in [EmotionType.HAPPY, EmotionType.EXCITED]:
            recommendations.extend(["energetic", "cute"])
        elif dominant_emotion in [EmotionType.SAD, EmotionType.WORRIED]:
            recommendations.extend(["gentle", "calm"])
        elif dominant_emotion in [EmotionType.LOVING, EmotionType.SHY]:
            recommendations.extend(["gentle", "cute"])
        elif dominant_emotion in [EmotionType.THINKING, EmotionType.CONFUSED]:
            recommendations.extend(["calm", "default"])
        else:
            recommendations.append("default")
        
        # 去重并限制数量
        unique_recommendations = []
        for rec in recommendations:
            if rec not in unique_recommendations:
                unique_recommendations.append(rec)
        
        return unique_recommendations[:3]
