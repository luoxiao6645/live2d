"""
动画序列编排器

负责编排和管理复杂的Live2D动画序列，包括：
1. 动画序列的创建和管理
2. 动画时间轴控制
3. 多个动画的协调和同步
4. 动画事件的触发和处理
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

from .emotion_config import EmotionType, EmotionConfig

# 配置日志
logger = logging.getLogger(__name__)

class AnimationType(Enum):
    """动画类型枚举"""
    EMOTION_TRANSITION = "emotion_transition"
    GESTURE = "gesture"
    IDLE = "idle"
    REACTION = "reaction"
    SPECIAL = "special"

class AnimationState(Enum):
    """动画状态枚举"""
    PENDING = "pending"
    PLAYING = "playing"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

@dataclass
class AnimationKeyframe:
    """动画关键帧"""
    time: float  # 相对时间（秒）
    parameters: Dict[str, float]  # Live2D参数值
    easing: str = "linear"  # 缓动函数
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AnimationSequence:
    """动画序列"""
    id: str
    name: str
    animation_type: AnimationType
    keyframes: List[AnimationKeyframe]
    duration: float
    loop: bool = False
    priority: int = 1  # 优先级，数字越大优先级越高
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 运行时状态
    state: AnimationState = AnimationState.PENDING
    start_time: float = 0.0
    current_time: float = 0.0
    loop_count: int = 0

class AnimationSequencer:
    """动画序列编排器类"""
    
    def __init__(self):
        """初始化动画序列编排器"""
        self.config = EmotionConfig()
        
        # 动画序列存储
        self.sequences: Dict[str, AnimationSequence] = {}
        self.active_sequences: Dict[str, AnimationSequence] = {}
        
        # 当前Live2D参数状态
        self.current_parameters: Dict[str, float] = {}
        
        # 动画更新线程
        self._update_thread = None
        self._running = False
        self._lock = threading.Lock()
        
        # 事件回调
        self.event_callbacks: Dict[str, List[Callable]] = {
            "animation_start": [],
            "animation_complete": [],
            "animation_cancel": [],
            "keyframe_reached": []
        }
        
        # 初始化预定义动画
        self._init_predefined_animations()
        
        # 启动更新线程
        self._start_update_thread()
        
        logger.info("动画序列编排器初始化完成")
    
    def _init_predefined_animations(self):
        """初始化预定义动画序列"""
        # 眨眼动画
        blink_sequence = self._create_blink_animation()
        self.register_sequence(blink_sequence)
        
        # 点头动画
        nod_sequence = self._create_nod_animation()
        self.register_sequence(nod_sequence)
        
        # 摇头动画
        shake_sequence = self._create_shake_animation()
        self.register_sequence(shake_sequence)
        
        # 挥手动画
        wave_sequence = self._create_wave_animation()
        self.register_sequence(wave_sequence)
        
        # 思考动画
        thinking_sequence = self._create_thinking_animation()
        self.register_sequence(thinking_sequence)
        
        logger.info("预定义动画序列初始化完成")
    
    def _create_blink_animation(self) -> AnimationSequence:
        """创建眨眼动画"""
        keyframes = [
            AnimationKeyframe(0.0, {"ParamEyeLOpen": 1.0, "ParamEyeROpen": 1.0}),
            AnimationKeyframe(0.1, {"ParamEyeLOpen": 0.0, "ParamEyeROpen": 0.0}, "ease_in"),
            AnimationKeyframe(0.2, {"ParamEyeLOpen": 0.0, "ParamEyeROpen": 0.0}),
            AnimationKeyframe(0.3, {"ParamEyeLOpen": 1.0, "ParamEyeROpen": 1.0}, "ease_out")
        ]
        
        return AnimationSequence(
            id="blink",
            name="眨眼",
            animation_type=AnimationType.IDLE,
            keyframes=keyframes,
            duration=0.3,
            priority=1
        )
    
    def _create_nod_animation(self) -> AnimationSequence:
        """创建点头动画"""
        keyframes = [
            AnimationKeyframe(0.0, {"ParamAngleX": 0.0}),
            AnimationKeyframe(0.3, {"ParamAngleX": 10.0}, "ease_in_out"),
            AnimationKeyframe(0.6, {"ParamAngleX": -5.0}, "ease_in_out"),
            AnimationKeyframe(1.0, {"ParamAngleX": 0.0}, "ease_in_out")
        ]
        
        return AnimationSequence(
            id="nod",
            name="点头",
            animation_type=AnimationType.GESTURE,
            keyframes=keyframes,
            duration=1.0,
            priority=3
        )
    
    def _create_shake_animation(self) -> AnimationSequence:
        """创建摇头动画"""
        keyframes = [
            AnimationKeyframe(0.0, {"ParamAngleY": 0.0}),
            AnimationKeyframe(0.2, {"ParamAngleY": -15.0}, "ease_in_out"),
            AnimationKeyframe(0.5, {"ParamAngleY": 15.0}, "ease_in_out"),
            AnimationKeyframe(0.8, {"ParamAngleY": -10.0}, "ease_in_out"),
            AnimationKeyframe(1.2, {"ParamAngleY": 0.0}, "ease_in_out")
        ]
        
        return AnimationSequence(
            id="shake",
            name="摇头",
            animation_type=AnimationType.GESTURE,
            keyframes=keyframes,
            duration=1.2,
            priority=3
        )
    
    def _create_wave_animation(self) -> AnimationSequence:
        """创建挥手动画"""
        keyframes = [
            AnimationKeyframe(0.0, {"ParamArmR": 0.0, "ParamAngleZ": 0.0}),
            AnimationKeyframe(0.3, {"ParamArmR": 1.0, "ParamAngleZ": 5.0}, "ease_out"),
            AnimationKeyframe(0.6, {"ParamArmR": 0.8, "ParamAngleZ": -5.0}, "ease_in_out"),
            AnimationKeyframe(0.9, {"ParamArmR": 1.0, "ParamAngleZ": 5.0}, "ease_in_out"),
            AnimationKeyframe(1.2, {"ParamArmR": 0.8, "ParamAngleZ": -5.0}, "ease_in_out"),
            AnimationKeyframe(1.5, {"ParamArmR": 1.0, "ParamAngleZ": 5.0}, "ease_in_out"),
            AnimationKeyframe(2.0, {"ParamArmR": 0.0, "ParamAngleZ": 0.0}, "ease_in")
        ]
        
        return AnimationSequence(
            id="wave",
            name="挥手",
            animation_type=AnimationType.GESTURE,
            keyframes=keyframes,
            duration=2.0,
            priority=4
        )
    
    def _create_thinking_animation(self) -> AnimationSequence:
        """创建思考动画"""
        keyframes = [
            AnimationKeyframe(0.0, {"ParamAngleX": 0.0, "ParamAngleY": 0.0, "ParamEyeBallX": 0.0}),
            AnimationKeyframe(0.5, {"ParamAngleX": -5.0, "ParamAngleY": 10.0, "ParamEyeBallX": 0.3}, "ease_in_out"),
            AnimationKeyframe(1.5, {"ParamAngleX": -8.0, "ParamAngleY": 15.0, "ParamEyeBallX": 0.5}, "ease_in_out"),
            AnimationKeyframe(2.5, {"ParamAngleX": -5.0, "ParamAngleY": 5.0, "ParamEyeBallX": -0.2}, "ease_in_out"),
            AnimationKeyframe(3.0, {"ParamAngleX": 0.0, "ParamAngleY": 0.0, "ParamEyeBallX": 0.0}, "ease_in_out")
        ]
        
        return AnimationSequence(
            id="thinking",
            name="思考",
            animation_type=AnimationType.REACTION,
            keyframes=keyframes,
            duration=3.0,
            priority=2
        )
    
    def register_sequence(self, sequence: AnimationSequence):
        """注册动画序列"""
        with self._lock:
            self.sequences[sequence.id] = sequence
            logger.debug(f"注册动画序列: {sequence.name} ({sequence.id})")
    
    def play_animation(self, sequence_id: str, **kwargs) -> bool:
        """
        播放动画序列
        
        Args:
            sequence_id: 动画序列ID
            **kwargs: 额外参数（如loop, priority等）
            
        Returns:
            是否成功开始播放
        """
        with self._lock:
            if sequence_id not in self.sequences:
                logger.warning(f"动画序列不存在: {sequence_id}")
                return False
            
            # 复制序列以避免修改原始定义
            sequence = self._copy_sequence(self.sequences[sequence_id])
            
            # 应用额外参数
            if "loop" in kwargs:
                sequence.loop = kwargs["loop"]
            if "priority" in kwargs:
                sequence.priority = kwargs["priority"]
            
            # 检查是否需要中断低优先级动画
            self._handle_animation_priority(sequence)
            
            # 设置动画状态
            sequence.state = AnimationState.PLAYING
            sequence.start_time = time.time()
            sequence.current_time = 0.0
            sequence.loop_count = 0
            
            # 添加到活动动画列表
            self.active_sequences[sequence.id] = sequence
            
            # 触发开始事件
            self._trigger_event("animation_start", sequence)
            
            logger.info(f"开始播放动画: {sequence.name}")
            return True
    
    def stop_animation(self, sequence_id: str) -> bool:
        """停止动画序列"""
        with self._lock:
            if sequence_id not in self.active_sequences:
                return False
            
            sequence = self.active_sequences[sequence_id]
            sequence.state = AnimationState.CANCELLED
            
            # 触发取消事件
            self._trigger_event("animation_cancel", sequence)
            
            # 从活动列表中移除
            del self.active_sequences[sequence_id]
            
            logger.info(f"停止动画: {sequence.name}")
            return True
    
    def pause_animation(self, sequence_id: str) -> bool:
        """暂停动画序列"""
        with self._lock:
            if sequence_id not in self.active_sequences:
                return False
            
            sequence = self.active_sequences[sequence_id]
            if sequence.state == AnimationState.PLAYING:
                sequence.state = AnimationState.PAUSED
                logger.info(f"暂停动画: {sequence.name}")
                return True
            
            return False
    
    def resume_animation(self, sequence_id: str) -> bool:
        """恢复动画序列"""
        with self._lock:
            if sequence_id not in self.active_sequences:
                return False
            
            sequence = self.active_sequences[sequence_id]
            if sequence.state == AnimationState.PAUSED:
                sequence.state = AnimationState.PLAYING
                # 重新计算开始时间
                sequence.start_time = time.time() - sequence.current_time
                logger.info(f"恢复动画: {sequence.name}")
                return True
            
            return False
    
    def get_current_parameters(self) -> Dict[str, float]:
        """获取当前Live2D参数"""
        with self._lock:
            return self.current_parameters.copy()
    
    def _start_update_thread(self):
        """启动动画更新线程"""
        self._running = True
        
        def update_loop():
            while self._running:
                self._update_animations()
                time.sleep(1/60.0)  # 60 FPS
        
        self._update_thread = threading.Thread(target=update_loop, daemon=True)
        self._update_thread.start()
    
    def _update_animations(self):
        """更新所有活动动画"""
        current_time = time.time()
        completed_animations = []
        
        with self._lock:
            for sequence_id, sequence in self.active_sequences.items():
                if sequence.state != AnimationState.PLAYING:
                    continue
                
                # 计算动画进度
                elapsed = current_time - sequence.start_time
                sequence.current_time = elapsed
                
                # 检查是否完成
                if elapsed >= sequence.duration:
                    if sequence.loop:
                        # 循环播放
                        sequence.start_time = current_time
                        sequence.current_time = 0.0
                        sequence.loop_count += 1
                    else:
                        # 标记为完成
                        sequence.state = AnimationState.COMPLETED
                        completed_animations.append(sequence_id)
                        continue
                
                # 更新参数
                self._update_sequence_parameters(sequence)
            
            # 移除完成的动画
            for sequence_id in completed_animations:
                sequence = self.active_sequences[sequence_id]
                self._trigger_event("animation_complete", sequence)
                del self.active_sequences[sequence_id]
    
    def _update_sequence_parameters(self, sequence: AnimationSequence):
        """更新单个动画序列的参数"""
        progress = sequence.current_time / sequence.duration
        
        # 找到当前时间对应的关键帧
        current_keyframe = None
        next_keyframe = None
        
        for i, keyframe in enumerate(sequence.keyframes):
            if keyframe.time <= progress:
                current_keyframe = keyframe
                if i + 1 < len(sequence.keyframes):
                    next_keyframe = sequence.keyframes[i + 1]
            else:
                if current_keyframe is None:
                    current_keyframe = keyframe
                break
        
        if current_keyframe is None:
            return
        
        # 计算插值参数
        if next_keyframe is None:
            # 使用当前关键帧的参数
            interpolated_params = current_keyframe.parameters.copy()
        else:
            # 在两个关键帧之间插值
            t = (progress - current_keyframe.time) / (next_keyframe.time - current_keyframe.time)
            t = max(0.0, min(1.0, t))
            
            # 应用缓动函数
            eased_t = self._apply_easing(t, current_keyframe.easing)
            
            interpolated_params = {}
            for param_name in current_keyframe.parameters:
                start_value = current_keyframe.parameters[param_name]
                end_value = next_keyframe.parameters.get(param_name, start_value)
                interpolated_params[param_name] = start_value + (end_value - start_value) * eased_t
        
        # 根据优先级合并参数
        for param_name, value in interpolated_params.items():
            if param_name not in self.current_parameters:
                self.current_parameters[param_name] = value
            else:
                # 高优先级动画覆盖低优先级
                self.current_parameters[param_name] = value
    
    def _apply_easing(self, t: float, easing: str) -> float:
        """应用缓动函数"""
        if easing == "linear":
            return t
        elif easing == "ease_in":
            return t * t
        elif easing == "ease_out":
            return 1 - (1 - t) * (1 - t)
        elif easing == "ease_in_out":
            if t < 0.5:
                return 2 * t * t
            else:
                return 1 - 2 * (1 - t) * (1 - t)
        elif easing == "bounce":
            if t < 0.5:
                return 2 * t * t
            else:
                return 1 - 2 * (1 - t) * (1 - t) * abs(np.sin(t * 10))
        else:
            return t
    
    def _handle_animation_priority(self, new_sequence: AnimationSequence):
        """处理动画优先级"""
        to_remove = []
        
        for sequence_id, active_sequence in self.active_sequences.items():
            if active_sequence.priority < new_sequence.priority:
                # 中断低优先级动画
                active_sequence.state = AnimationState.CANCELLED
                to_remove.append(sequence_id)
        
        for sequence_id in to_remove:
            sequence = self.active_sequences[sequence_id]
            self._trigger_event("animation_cancel", sequence)
            del self.active_sequences[sequence_id]
    
    def _copy_sequence(self, sequence: AnimationSequence) -> AnimationSequence:
        """复制动画序列"""
        return AnimationSequence(
            id=sequence.id,
            name=sequence.name,
            animation_type=sequence.animation_type,
            keyframes=sequence.keyframes.copy(),
            duration=sequence.duration,
            loop=sequence.loop,
            priority=sequence.priority,
            metadata=sequence.metadata.copy()
        )
    
    def _trigger_event(self, event_type: str, sequence: AnimationSequence):
        """触发事件回调"""
        if event_type in self.event_callbacks:
            for callback in self.event_callbacks[event_type]:
                try:
                    callback(sequence)
                except Exception as e:
                    logger.error(f"事件回调执行失败: {e}")
    
    def add_event_listener(self, event_type: str, callback: Callable):
        """添加事件监听器"""
        if event_type in self.event_callbacks:
            self.event_callbacks[event_type].append(callback)
    
    def get_active_animations(self) -> List[Dict[str, Any]]:
        """获取当前活动的动画列表"""
        with self._lock:
            return [
                {
                    "id": seq.id,
                    "name": seq.name,
                    "type": seq.animation_type.value,
                    "state": seq.state.value,
                    "progress": seq.current_time / seq.duration if seq.duration > 0 else 0,
                    "priority": seq.priority,
                    "loop": seq.loop,
                    "loop_count": seq.loop_count
                }
                for seq in self.active_sequences.values()
            ]
    
    def stop(self):
        """停止动画序列编排器"""
        self._running = False
        if self._update_thread:
            self._update_thread.join(timeout=1.0)
        logger.info("动画序列编排器已停止")
