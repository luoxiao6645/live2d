"""
Live2D AI助手 情感系统模块

这个模块提供了完整的情感分析、状态管理和动画控制功能，
让Live2D角色能够根据对话内容展现出丰富的情感表达。

主要组件：
- EmotionAnalyzer: 情感分析器
- EmotionStateManager: 情感状态管理器  
- AnimationSequencer: 动画序列编排器
- AdvancedAnimationController: 高级动画控制器
"""

from .emotion_analyzer import EmotionAnalyzer
from .emotion_state_manager import EmotionStateManager
from .animation_sequencer import AnimationSequencer
from .advanced_animation_controller import AdvancedAnimationController
from .emotion_config import EmotionConfig

__version__ = "1.0.0"
__author__ = "Live2D AI Assistant Team"

__all__ = [
    "EmotionAnalyzer",
    "EmotionStateManager", 
    "AnimationSequencer",
    "AdvancedAnimationController",
    "EmotionConfig"
]
