"""
情感状态管理器

负责管理Live2D角色的情感状态，包括：
1. 情感状态的转换和维护
2. 情感强度的衰减处理
3. 情感历史记录
4. 状态机逻辑
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import threading

from .emotion_config import EmotionType, EmotionConfig

# 配置日志
logger = logging.getLogger(__name__)

@dataclass
class EmotionState:
    """情感状态数据类"""
    emotion_type: EmotionType
    intensity: float
    timestamp: float
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EmotionTransition:
    """情感转换数据类"""
    from_emotion: EmotionType
    to_emotion: EmotionType
    start_time: float
    duration: float
    progress: float = 0.0
    easing_function: str = "ease_in_out"

class EmotionStateManager:
    """情感状态管理器类"""
    
    def __init__(self, max_history: int = 100):
        """
        初始化情感状态管理器
        
        Args:
            max_history: 最大历史记录数量
        """
        self.config = EmotionConfig()
        self.max_history = max_history
        
        # 当前情感状态
        self.current_state = EmotionState(
            emotion_type=EmotionType.NEUTRAL,
            intensity=0.1,
            timestamp=time.time()
        )
        
        # 情感历史记录
        self.emotion_history: deque = deque(maxlen=max_history)
        self.emotion_history.append(self.current_state)
        
        # 当前转换状态
        self.current_transition: Optional[EmotionTransition] = None
        
        # 状态锁
        self._state_lock = threading.Lock()
        
        # 衰减定时器
        self._decay_timer = None
        self._start_decay_timer()
        
        logger.info("情感状态管理器初始化完成")
    
    def update_emotion(self, emotion_analysis: Dict[str, Any]) -> bool:
        """
        根据情感分析结果更新情感状态
        
        Args:
            emotion_analysis: 情感分析结果
            
        Returns:
            是否成功更新状态
        """
        try:
            new_emotion_type = EmotionType(emotion_analysis["primary_emotion"])
            new_intensity = emotion_analysis["intensity"]
            confidence = emotion_analysis.get("confidence", 1.0)
            
            # 根据置信度调整强度
            adjusted_intensity = new_intensity * confidence
            
            with self._state_lock:
                # 检查是否需要状态转换
                if self._should_transition(new_emotion_type, adjusted_intensity):
                    return self._start_transition(new_emotion_type, adjusted_intensity, emotion_analysis)
                else:
                    # 只更新强度
                    return self._update_intensity(adjusted_intensity)
                    
        except Exception as e:
            logger.error(f"更新情感状态失败: {e}")
            return False
    
    def _should_transition(self, new_emotion: EmotionType, new_intensity: float) -> bool:
        """判断是否应该进行情感转换"""
        current_emotion = self.current_state.emotion_type
        current_intensity = self.current_state.intensity
        
        # 如果情感类型不同，且新强度足够高，则转换
        if new_emotion != current_emotion and new_intensity > 0.3:
            return True
        
        # 如果情感类型相同，但强度变化很大，也可能需要转换
        if new_emotion == current_emotion and abs(new_intensity - current_intensity) > 0.4:
            return True
        
        return False
    
    def _start_transition(self, target_emotion: EmotionType, target_intensity: float, 
                         metadata: Dict[str, Any]) -> bool:
        """开始情感转换"""
        try:
            # 计算转换持续时间
            transition_duration = self._calculate_transition_duration(
                self.current_state.emotion_type, target_emotion
            )
            
            # 创建转换对象
            self.current_transition = EmotionTransition(
                from_emotion=self.current_state.emotion_type,
                to_emotion=target_emotion,
                start_time=time.time(),
                duration=transition_duration
            )
            
            # 创建新的目标状态
            target_state = EmotionState(
                emotion_type=target_emotion,
                intensity=target_intensity,
                timestamp=time.time(),
                metadata=metadata
            )
            
            logger.info(f"开始情感转换: {self.current_state.emotion_type.value} -> {target_emotion.value}")
            return True
            
        except Exception as e:
            logger.error(f"开始情感转换失败: {e}")
            return False
    
    def _update_intensity(self, new_intensity: float) -> bool:
        """更新当前情感强度"""
        try:
            old_intensity = self.current_state.intensity
            self.current_state.intensity = max(0.1, min(1.0, new_intensity))
            self.current_state.timestamp = time.time()
            
            logger.debug(f"更新情感强度: {old_intensity:.2f} -> {self.current_state.intensity:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"更新情感强度失败: {e}")
            return False
    
    def _calculate_transition_duration(self, from_emotion: EmotionType, to_emotion: EmotionType) -> float:
        """计算情感转换持续时间"""
        # 基础转换时间
        base_duration = self.config.ANIMATION_DURATIONS["emotion_transition"] / 1000.0
        
        # 根据情感类型调整时间
        if from_emotion == EmotionType.NEUTRAL or to_emotion == EmotionType.NEUTRAL:
            return base_duration * 0.8  # 从/到中性状态转换较快
        
        # 相似情感转换较快
        similar_emotions = {
            (EmotionType.HAPPY, EmotionType.EXCITED),
            (EmotionType.SAD, EmotionType.WORRIED),
            (EmotionType.CONFUSED, EmotionType.THINKING)
        }
        
        emotion_pair = (from_emotion, to_emotion)
        if emotion_pair in similar_emotions or emotion_pair[::-1] in similar_emotions:
            return base_duration * 0.6
        
        return base_duration
    
    def update_transition(self) -> Optional[Dict[str, Any]]:
        """
        更新转换进度
        
        Returns:
            当前转换状态信息，如果没有转换则返回None
        """
        if not self.current_transition:
            return None
        
        with self._state_lock:
            current_time = time.time()
            elapsed = current_time - self.current_transition.start_time
            
            # 计算转换进度
            progress = min(elapsed / self.current_transition.duration, 1.0)
            self.current_transition.progress = progress
            
            # 如果转换完成
            if progress >= 1.0:
                self._complete_transition()
                return None
            
            # 返回当前转换状态
            return {
                "from_emotion": self.current_transition.from_emotion.value,
                "to_emotion": self.current_transition.to_emotion.value,
                "progress": progress,
                "easing_function": self.current_transition.easing_function
            }
    
    def _complete_transition(self):
        """完成情感转换"""
        if not self.current_transition:
            return
        
        # 更新当前状态为目标状态
        self.current_state.emotion_type = self.current_transition.to_emotion
        self.current_state.timestamp = time.time()
        
        # 添加到历史记录
        self.emotion_history.append(self.current_state)
        
        logger.info(f"情感转换完成: {self.current_transition.to_emotion.value}")
        
        # 清除转换状态
        self.current_transition = None
    
    def get_current_state(self) -> Dict[str, Any]:
        """获取当前情感状态"""
        with self._state_lock:
            return {
                "emotion_type": self.current_state.emotion_type.value,
                "emotion_label": self._get_emotion_label(self.current_state.emotion_type),
                "intensity": self.current_state.intensity,
                "intensity_label": self._get_intensity_label(self.current_state.intensity),
                "timestamp": self.current_state.timestamp,
                "duration": time.time() - self.current_state.timestamp,
                "is_transitioning": self.current_transition is not None,
                "transition_info": self.update_transition()
            }
    
    def get_live2d_parameters(self) -> Dict[str, float]:
        """
        获取当前情感状态对应的Live2D参数
        
        Returns:
            Live2D参数字典
        """
        with self._state_lock:
            base_params = self.config.LIVE2D_PARAM_MAPPING.get(
                self.current_state.emotion_type, 
                self.config.LIVE2D_PARAM_MAPPING[EmotionType.NEUTRAL]
            ).copy()
            
            # 根据强度调整参数
            intensity = self.current_state.intensity
            for param_name, param_value in base_params.items():
                # 强度影响参数的表现程度
                if param_name in ["ParamEyeLSmile", "ParamEyeRSmile", "ParamCheek"]:
                    base_params[param_name] = param_value * intensity
                elif param_name in ["ParamMouthForm", "ParamBrowLY", "ParamBrowRY"]:
                    base_params[param_name] = param_value * intensity
            
            # 如果正在转换，进行插值
            if self.current_transition:
                base_params = self._interpolate_parameters(base_params)
            
            return base_params
    
    def _interpolate_parameters(self, target_params: Dict[str, float]) -> Dict[str, float]:
        """在转换过程中插值参数"""
        if not self.current_transition:
            return target_params
        
        # 获取起始参数
        from_params = self.config.LIVE2D_PARAM_MAPPING.get(
            self.current_transition.from_emotion,
            self.config.LIVE2D_PARAM_MAPPING[EmotionType.NEUTRAL]
        )
        
        # 计算插值
        progress = self.current_transition.progress
        eased_progress = self._apply_easing(progress, self.current_transition.easing_function)
        
        interpolated_params = {}
        for param_name in target_params:
            from_value = from_params.get(param_name, 0.0)
            to_value = target_params[param_name]
            interpolated_params[param_name] = from_value + (to_value - from_value) * eased_progress
        
        return interpolated_params
    
    def _apply_easing(self, progress: float, easing_function: str) -> float:
        """应用缓动函数"""
        if easing_function == "linear":
            return progress
        elif easing_function == "ease_in":
            return progress * progress
        elif easing_function == "ease_out":
            return 1 - (1 - progress) * (1 - progress)
        elif easing_function == "ease_in_out":
            if progress < 0.5:
                return 2 * progress * progress
            else:
                return 1 - 2 * (1 - progress) * (1 - progress)
        else:
            return progress
    
    def _start_decay_timer(self):
        """启动情感衰减定时器"""
        def decay_loop():
            while True:
                time.sleep(1.0)  # 每秒检查一次
                self._apply_emotion_decay()
        
        self._decay_timer = threading.Thread(target=decay_loop, daemon=True)
        self._decay_timer.start()
    
    def _apply_emotion_decay(self):
        """应用情感衰减"""
        with self._state_lock:
            if self.current_transition:
                return  # 转换期间不衰减
            
            current_time = time.time()
            time_since_update = current_time - self.current_state.timestamp
            
            # 如果时间太短，不进行衰减
            if time_since_update < 5.0:
                return
            
            # 计算衰减
            decay_rate = self.config.EMOTION_DECAY["decay_rate"]
            new_intensity = self.current_state.intensity * (1 - decay_rate * time_since_update / 60.0)
            
            # 检查是否需要回到中性状态
            neutral_threshold = self.config.EMOTION_DECAY["neutral_threshold"]
            if new_intensity < neutral_threshold and self.current_state.emotion_type != EmotionType.NEUTRAL:
                self._start_transition(EmotionType.NEUTRAL, 0.1, {"reason": "emotion_decay"})
            else:
                self.current_state.intensity = max(
                    self.config.EMOTION_DECAY["min_intensity"], 
                    new_intensity
                )
                self.current_state.timestamp = current_time
    
    def _get_emotion_label(self, emotion_type: EmotionType) -> str:
        """获取情感类型的中文标签"""
        labels = {
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
        return labels.get(emotion_type, "未知")
    
    def _get_intensity_label(self, intensity: float) -> str:
        """获取强度标签"""
        if intensity < 0.3:
            return "轻微"
        elif intensity < 0.6:
            return "中等"
        elif intensity < 0.8:
            return "强烈"
        else:
            return "非常强烈"
    
    def get_emotion_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取情感历史记录"""
        with self._state_lock:
            history = list(self.emotion_history)[-limit:]
            return [
                {
                    "emotion_type": state.emotion_type.value,
                    "emotion_label": self._get_emotion_label(state.emotion_type),
                    "intensity": state.intensity,
                    "timestamp": state.timestamp,
                    "duration": state.duration
                }
                for state in history
            ]
    
    def force_emotion(self, emotion_type: EmotionType, intensity: float = 0.8) -> bool:
        """
        强制设置情感状态（用于测试或特殊场景）
        
        Args:
            emotion_type: 目标情感类型
            intensity: 情感强度
            
        Returns:
            是否成功设置
        """
        try:
            with self._state_lock:
                self.current_state = EmotionState(
                    emotion_type=emotion_type,
                    intensity=max(0.1, min(1.0, intensity)),
                    timestamp=time.time(),
                    metadata={"forced": True}
                )
                self.emotion_history.append(self.current_state)
                self.current_transition = None
                
            logger.info(f"强制设置情感状态: {emotion_type.value}, 强度: {intensity}")
            return True
            
        except Exception as e:
            logger.error(f"强制设置情感状态失败: {e}")
            return False
