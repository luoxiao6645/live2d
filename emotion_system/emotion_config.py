"""
情感系统配置文件

定义了情感类型、Live2D参数映射、动画配置等核心配置信息
"""

from typing import Dict, List, Tuple, Any
from enum import Enum

class EmotionType(Enum):
    """情感类型枚举"""
    NEUTRAL = "neutral"      # 中性
    HAPPY = "happy"          # 开心
    SAD = "sad"              # 悲伤
    ANGRY = "angry"          # 生气
    SURPRISED = "surprised"  # 惊讶
    CONFUSED = "confused"    # 困惑
    SHY = "shy"              # 害羞
    EXCITED = "excited"      # 兴奋
    WORRIED = "worried"      # 担心
    LOVING = "loving"        # 喜爱
    THINKING = "thinking"    # 思考
    SLEEPY = "sleepy"        # 困倦

class EmotionConfig:
    """情感系统配置类"""
    
    # 情感强度级别
    EMOTION_INTENSITY_LEVELS = {
        "very_low": 0.2,
        "low": 0.4,
        "medium": 0.6,
        "high": 0.8,
        "very_high": 1.0
    }
    
    # 情感关键词映射
    EMOTION_KEYWORDS = {
        EmotionType.HAPPY: [
            "开心", "高兴", "快乐", "愉快", "兴奋", "满意", "喜悦", "欢乐",
            "happy", "joy", "glad", "pleased", "excited", "cheerful", "delighted"
        ],
        EmotionType.SAD: [
            "难过", "悲伤", "伤心", "沮丧", "失望", "痛苦", "忧郁", "哀伤",
            "sad", "sorrow", "grief", "disappointed", "depressed", "upset", "melancholy"
        ],
        EmotionType.ANGRY: [
            "生气", "愤怒", "恼火", "烦躁", "气愤", "暴怒", "不满", "愤慨",
            "angry", "mad", "furious", "irritated", "annoyed", "rage", "frustrated"
        ],
        EmotionType.SURPRISED: [
            "惊讶", "震惊", "吃惊", "意外", "惊奇", "诧异", "惊愕",
            "surprised", "shocked", "amazed", "astonished", "stunned", "bewildered"
        ],
        EmotionType.CONFUSED: [
            "困惑", "疑惑", "迷茫", "不解", "纳闷", "费解", "茫然",
            "confused", "puzzled", "perplexed", "bewildered", "baffled", "mystified"
        ],
        EmotionType.SHY: [
            "害羞", "羞涩", "腼腆", "不好意思", "羞怯", "内向",
            "shy", "bashful", "timid", "embarrassed", "modest", "coy"
        ],
        EmotionType.EXCITED: [
            "兴奋", "激动", "热情", "狂热", "亢奋", "振奋",
            "excited", "thrilled", "enthusiastic", "energetic", "passionate", "eager"
        ],
        EmotionType.WORRIED: [
            "担心", "忧虑", "焦虑", "不安", "紧张", "恐惧", "害怕",
            "worried", "anxious", "concerned", "nervous", "afraid", "fearful", "uneasy"
        ],
        EmotionType.LOVING: [
            "喜爱", "爱", "喜欢", "钟爱", "热爱", "宠爱", "疼爱",
            "love", "like", "adore", "cherish", "affection", "fond", "care"
        ],
        EmotionType.THINKING: [
            "思考", "想", "考虑", "琢磨", "沉思", "深思", "反思",
            "think", "consider", "ponder", "contemplate", "reflect", "meditate"
        ],
        EmotionType.SLEEPY: [
            "困", "累", "疲倦", "想睡", "打瞌睡", "昏昏欲睡",
            "sleepy", "tired", "drowsy", "weary", "exhausted", "fatigue"
        ]
    }
    
    # Live2D参数映射
    LIVE2D_PARAM_MAPPING = {
        EmotionType.NEUTRAL: {
            "ParamEyeLOpen": 1.0,
            "ParamEyeROpen": 1.0,
            "ParamEyeLSmile": 0.0,
            "ParamEyeRSmile": 0.0,
            "ParamMouthOpenY": 0.0,
            "ParamMouthForm": 0.0,
            "ParamBrowLY": 0.0,
            "ParamBrowRY": 0.0,
            "ParamCheek": 0.0
        },
        EmotionType.HAPPY: {
            "ParamEyeLOpen": 0.6,
            "ParamEyeROpen": 0.6,
            "ParamEyeLSmile": 1.0,
            "ParamEyeRSmile": 1.0,
            "ParamMouthOpenY": 0.3,
            "ParamMouthForm": 1.0,
            "ParamBrowLY": 0.3,
            "ParamBrowRY": 0.3,
            "ParamCheek": 0.8
        },
        EmotionType.SAD: {
            "ParamEyeLOpen": 0.3,
            "ParamEyeROpen": 0.3,
            "ParamEyeLSmile": 0.0,
            "ParamEyeRSmile": 0.0,
            "ParamMouthOpenY": 0.0,
            "ParamMouthForm": -0.8,
            "ParamBrowLY": -0.8,
            "ParamBrowRY": -0.8,
            "ParamCheek": 0.0
        },
        EmotionType.ANGRY: {
            "ParamEyeLOpen": 0.8,
            "ParamEyeROpen": 0.8,
            "ParamEyeLSmile": 0.0,
            "ParamEyeRSmile": 0.0,
            "ParamMouthOpenY": 0.2,
            "ParamMouthForm": -0.6,
            "ParamBrowLY": -1.0,
            "ParamBrowRY": -1.0,
            "ParamCheek": 0.0
        },
        EmotionType.SURPRISED: {
            "ParamEyeLOpen": 1.0,
            "ParamEyeROpen": 1.0,
            "ParamEyeLSmile": 0.0,
            "ParamEyeRSmile": 0.0,
            "ParamMouthOpenY": 0.8,
            "ParamMouthForm": 0.0,
            "ParamBrowLY": 1.0,
            "ParamBrowRY": 1.0,
            "ParamCheek": 0.0
        },
        EmotionType.CONFUSED: {
            "ParamEyeLOpen": 0.7,
            "ParamEyeROpen": 0.7,
            "ParamEyeLSmile": 0.0,
            "ParamEyeRSmile": 0.0,
            "ParamMouthOpenY": 0.1,
            "ParamMouthForm": -0.3,
            "ParamBrowLY": -0.3,
            "ParamBrowRY": 0.3,
            "ParamCheek": 0.0
        },
        EmotionType.SHY: {
            "ParamEyeLOpen": 0.4,
            "ParamEyeROpen": 0.4,
            "ParamEyeLSmile": 0.6,
            "ParamEyeRSmile": 0.6,
            "ParamMouthOpenY": 0.0,
            "ParamMouthForm": 0.3,
            "ParamBrowLY": 0.2,
            "ParamBrowRY": 0.2,
            "ParamCheek": 1.0
        },
        EmotionType.EXCITED: {
            "ParamEyeLOpen": 1.0,
            "ParamEyeROpen": 1.0,
            "ParamEyeLSmile": 0.8,
            "ParamEyeRSmile": 0.8,
            "ParamMouthOpenY": 0.6,
            "ParamMouthForm": 1.0,
            "ParamBrowLY": 0.8,
            "ParamBrowRY": 0.8,
            "ParamCheek": 0.6
        },
        EmotionType.WORRIED: {
            "ParamEyeLOpen": 0.5,
            "ParamEyeROpen": 0.5,
            "ParamEyeLSmile": 0.0,
            "ParamEyeRSmile": 0.0,
            "ParamMouthOpenY": 0.0,
            "ParamMouthForm": -0.4,
            "ParamBrowLY": -0.6,
            "ParamBrowRY": -0.6,
            "ParamCheek": 0.0
        },
        EmotionType.LOVING: {
            "ParamEyeLOpen": 0.3,
            "ParamEyeROpen": 0.3,
            "ParamEyeLSmile": 1.0,
            "ParamEyeRSmile": 1.0,
            "ParamMouthOpenY": 0.0,
            "ParamMouthForm": 0.8,
            "ParamBrowLY": 0.4,
            "ParamBrowRY": 0.4,
            "ParamCheek": 1.0
        },
        EmotionType.THINKING: {
            "ParamEyeLOpen": 0.6,
            "ParamEyeROpen": 0.6,
            "ParamEyeLSmile": 0.0,
            "ParamEyeRSmile": 0.0,
            "ParamMouthOpenY": 0.0,
            "ParamMouthForm": -0.2,
            "ParamBrowLY": -0.2,
            "ParamBrowRY": -0.2,
            "ParamCheek": 0.0
        },
        EmotionType.SLEEPY: {
            "ParamEyeLOpen": 0.1,
            "ParamEyeROpen": 0.1,
            "ParamEyeLSmile": 0.0,
            "ParamEyeRSmile": 0.0,
            "ParamMouthOpenY": 0.2,
            "ParamMouthForm": 0.0,
            "ParamBrowLY": -0.3,
            "ParamBrowRY": -0.3,
            "ParamCheek": 0.0
        }
    }
    
    # 动画持续时间配置（毫秒）
    ANIMATION_DURATIONS = {
        "emotion_transition": 2000,    # 情感转换
        "blink": 300,                  # 眨眼
        "head_nod": 1000,             # 点头
        "head_shake": 1200,           # 摇头
        "wave": 2000,                 # 挥手
        "dance": 5000,                # 跳舞
        "thinking": 3000,             # 思考动作
        "surprise_jump": 800,         # 惊讶跳跃
        "shy_hide": 1500,             # 害羞躲藏
        "excited_bounce": 2500        # 兴奋弹跳
    }
    
    # 情感转换规则
    EMOTION_TRANSITION_RULES = {
        # 从任何情感都可以转换到中性
        "to_neutral": {
            "allowed_from": list(EmotionType),
            "transition_time": 1.5
        },
        # 特定情感转换规则
        "happy_to_excited": {
            "allowed_from": [EmotionType.HAPPY],
            "transition_time": 1.0
        },
        "sad_to_worried": {
            "allowed_from": [EmotionType.SAD],
            "transition_time": 2.0
        },
        "confused_to_thinking": {
            "allowed_from": [EmotionType.CONFUSED],
            "transition_time": 1.5
        }
    }
    
    # 情感衰减配置
    EMOTION_DECAY = {
        "decay_rate": 0.1,           # 每秒衰减率
        "min_intensity": 0.1,        # 最小强度阈值
        "neutral_threshold": 0.2     # 回到中性状态的阈值
    }
