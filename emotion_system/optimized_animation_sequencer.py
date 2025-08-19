# -*- coding: utf-8 -*-
"""
优化的动画序列管理器
改进线程安全性、资源管理和性能
"""

import threading
import time
import logging
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue, Empty
import weakref

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnimationState(Enum):
    """动画状态枚举"""
    IDLE = "idle"
    PLAYING = "playing"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"

class AnimationPriority(Enum):
    """动画优先级枚举"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class AnimationKeyframe:
    """动画关键帧数据类"""
    time: float
    parameters: Dict[str, float]
    easing: str = "linear"

@dataclass
class AnimationSequence:
    """动画序列数据类"""
    name: str
    keyframes: List[AnimationKeyframe]
    duration: float
    loop: bool = False
    priority: AnimationPriority = AnimationPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ActiveAnimation:
    """活跃动画数据类"""
    sequence: Optional[AnimationSequence]
    start_time: float
    current_time: float = 0.0
    state: AnimationState = AnimationState.PLAYING
    loop_count: int = 0
    extra_params: Dict[str, float] = field(default_factory=dict)

class OptimizedAnimationSequencer:
    """
    优化的动画序列管理器
    
    主要改进：
    1. 线程安全的动画管理
    2. 资源池化和重用
    3. 智能优先级调度
    4. 性能监控和统计
    5. 内存泄漏防护
    """
    
    def __init__(self, fps: int = 60, max_concurrent_animations: int = 10):
        """
        初始化优化的动画序列管理器
        
        Args:
            fps: 动画帧率
            max_concurrent_animations: 最大并发动画数量
        """
        self.fps = fps
        self.max_concurrent_animations = max_concurrent_animations
        self.frame_time = 1.0 / fps
        
        # 线程安全锁
        self._lock = threading.RLock()
        self._state_lock = threading.Lock()
        
        # 动画存储
        self._sequences: Dict[str, AnimationSequence] = {}
        self._active_animations: Dict[str, ActiveAnimation] = {}
        self._animation_queue = Queue(maxsize=100)
        
        # 当前Live2D参数
        self._current_parameters: Dict[str, float] = {}
        
        # 事件回调（使用弱引用防止内存泄漏）
        self._event_callbacks: Dict[str, List[weakref.ref]] = {
            'animation_start': [],
            'animation_end': [],
            'animation_loop': [],
            'parameter_update': []
        }
        
        # 控制标志
        self._running = False
        self._paused = False
        
        # 线程管理
        self._update_thread: Optional[threading.Thread] = None
        self._processor_thread: Optional[threading.Thread] = None
        
        # 性能统计
        self._stats = {
            'total_animations_played': 0,
            'active_animation_count': 0,
            'average_frame_time': 0.0,
            'dropped_frames': 0,
            'memory_usage': 0
        }
        
        # 资源池
        self._keyframe_pool: List[AnimationKeyframe] = []
        self._animation_pool: List[ActiveAnimation] = []
        
        # 初始化预定义动画
        self._initialize_predefined_animations()
        
        logger.info(f"优化动画序列管理器初始化完成 - FPS: {fps}, 最大并发: {max_concurrent_animations}")
    
    def _initialize_predefined_animations(self):
        """
        初始化预定义的动画序列
        """
        try:
            # 眨眼动画
            blink_keyframes = [
                AnimationKeyframe(0.0, {'ParamEyeLOpen': 1.0, 'ParamEyeROpen': 1.0}),
                AnimationKeyframe(0.1, {'ParamEyeLOpen': 0.0, 'ParamEyeROpen': 0.0}),
                AnimationKeyframe(0.2, {'ParamEyeLOpen': 1.0, 'ParamEyeROpen': 1.0})
            ]
            self.register_sequence(AnimationSequence(
                name="blink",
                keyframes=blink_keyframes,
                duration=0.2,
                priority=AnimationPriority.LOW
            ))
            
            # 点头动画
            nod_keyframes = [
                AnimationKeyframe(0.0, {'ParamAngleX': 0.0}),
                AnimationKeyframe(0.3, {'ParamAngleX': -10.0}),
                AnimationKeyframe(0.6, {'ParamAngleX': 0.0})
            ]
            self.register_sequence(AnimationSequence(
                name="nod",
                keyframes=nod_keyframes,
                duration=0.6,
                priority=AnimationPriority.NORMAL
            ))
            
            # 摇头动画
            shake_keyframes = [
                AnimationKeyframe(0.0, {'ParamAngleZ': 0.0}),
                AnimationKeyframe(0.2, {'ParamAngleZ': -15.0}),
                AnimationKeyframe(0.4, {'ParamAngleZ': 15.0}),
                AnimationKeyframe(0.6, {'ParamAngleZ': 0.0})
            ]
            self.register_sequence(AnimationSequence(
                name="shake",
                keyframes=shake_keyframes,
                duration=0.6,
                priority=AnimationPriority.NORMAL
            ))
            
            # 思考动画
            thinking_keyframes = [
                AnimationKeyframe(0.0, {'ParamAngleX': 0.0, 'ParamEyeBallX': 0.0}),
                AnimationKeyframe(1.0, {'ParamAngleX': 5.0, 'ParamEyeBallX': 0.3}),
                AnimationKeyframe(2.0, {'ParamAngleX': -3.0, 'ParamEyeBallX': -0.2}),
                AnimationKeyframe(3.0, {'ParamAngleX': 0.0, 'ParamEyeBallX': 0.0})
            ]
            self.register_sequence(AnimationSequence(
                name="thinking",
                keyframes=thinking_keyframes,
                duration=3.0,
                loop=True,
                priority=AnimationPriority.LOW
            ))
            
            logger.info("预定义动画序列初始化完成")
            
        except Exception as e:
            logger.error(f"初始化预定义动画时发生错误: {e}")
    
    def register_sequence(self, sequence: AnimationSequence) -> bool:
        """
        注册动画序列
        
        Args:
            sequence: 动画序列对象
            
        Returns:
            bool: 注册是否成功
        """
        try:
            with self._lock:
                if sequence.name in self._sequences:
                    logger.warning(f"动画序列 '{sequence.name}' 已存在，将被覆盖")
                
                # 验证动画序列
                if not self._validate_sequence(sequence):
                    logger.error(f"动画序列 '{sequence.name}' 验证失败")
                    return False
                
                self._sequences[sequence.name] = sequence
                logger.info(f"动画序列 '{sequence.name}' 注册成功")
                return True
                
        except Exception as e:
            logger.error(f"注册动画序列时发生错误: {e}")
            return False
    
    def _validate_sequence(self, sequence: AnimationSequence) -> bool:
        """
        验证动画序列的有效性
        
        Args:
            sequence: 动画序列对象
            
        Returns:
            bool: 序列是否有效
        """
        try:
            if not sequence.name or not sequence.keyframes:
                return False
            
            if sequence.duration <= 0:
                return False
            
            # 检查关键帧时间顺序
            prev_time = -1
            for keyframe in sequence.keyframes:
                if keyframe.time < prev_time:
                    return False
                prev_time = keyframe.time
            
            return True
            
        except Exception:
            return False
    
    def play_animation(self, name: str, extra_params: Optional[Dict[str, float]] = None, 
                      force: bool = False) -> bool:
        """
        播放动画
        
        Args:
            name: 动画名称
            extra_params: 额外参数
            force: 是否强制播放（忽略优先级）
            
        Returns:
            bool: 播放是否成功
        """
        try:
            with self._lock:
                if name not in self._sequences:
                    logger.warning(f"动画序列 '{name}' 不存在")
                    return False
                
                sequence = self._sequences[name]
                
                # 检查并发限制
                if len(self._active_animations) >= self.max_concurrent_animations:
                    if not force:
                        logger.warning(f"达到最大并发动画数量限制: {self.max_concurrent_animations}")
                        return False
                    else:
                        # 强制播放时，停止最低优先级的动画
                        self._stop_lowest_priority_animation()
                
                # 检查优先级冲突
                if not force and not self._check_priority_conflict(sequence):
                    logger.info(f"动画 '{name}' 因优先级冲突被跳过")
                    return False
                
                # 创建活跃动画
                active_animation = self._create_active_animation(sequence, extra_params)
                self._active_animations[name] = active_animation
                
                # 触发开始事件
                self._trigger_event('animation_start', name, sequence)
                
                # 更新统计
                self._stats['total_animations_played'] += 1
                self._stats['active_animation_count'] = len(self._active_animations)
                
                logger.info(f"动画 '{name}' 开始播放")
                return True
                
        except Exception as e:
            logger.error(f"播放动画时发生错误: {e}")
            return False
    
    def _check_priority_conflict(self, sequence: AnimationSequence) -> bool:
        """
        检查优先级冲突
        
        Args:
            sequence: 要播放的动画序列
            
        Returns:
            bool: 是否可以播放
        """
        try:
            for active_name, active_anim in self._active_animations.items():
                # 添加空值检查，防止访问已回收的动画对象
                if active_anim.sequence is None:
                    continue
                    
                if active_anim.sequence.priority.value >= sequence.priority.value:
                    # 检查参数冲突
                    if self._has_parameter_conflict(sequence, active_anim.sequence):
                        return False
            return True
            
        except Exception as e:
            logger.error(f"检查优先级冲突时发生错误: {e}")
            return False
    
    def _has_parameter_conflict(self, seq1: AnimationSequence, seq2: AnimationSequence) -> bool:
        """
        检查两个动画序列是否有参数冲突
        
        Args:
            seq1: 动画序列1
            seq2: 动画序列2
            
        Returns:
            bool: 是否有冲突
        """
        try:
            params1 = set()
            params2 = set()
            
            for keyframe in seq1.keyframes:
                params1.update(keyframe.parameters.keys())
            
            for keyframe in seq2.keyframes:
                params2.update(keyframe.parameters.keys())
            
            return bool(params1.intersection(params2))
            
        except Exception:
            return True  # 发生错误时保守处理
    
    def _create_active_animation(self, sequence: AnimationSequence, 
                               extra_params: Optional[Dict[str, float]]) -> ActiveAnimation:
        """
        创建活跃动画对象
        
        Args:
            sequence: 动画序列
            extra_params: 额外参数
            
        Returns:
            ActiveAnimation: 活跃动画对象
        """
        # 尝试从对象池获取
        if self._animation_pool:
            active_anim = self._animation_pool.pop()
            active_anim.sequence = sequence
            active_anim.start_time = time.time()
            active_anim.current_time = 0.0
            active_anim.state = AnimationState.PLAYING
            active_anim.loop_count = 0
            active_anim.extra_params = extra_params or {}
        else:
            active_anim = ActiveAnimation(
                sequence=sequence,
                start_time=time.time(),
                extra_params=extra_params or {}
            )
        
        return active_anim
    
    def _stop_lowest_priority_animation(self):
        """
        停止最低优先级的动画
        """
        try:
            if not self._active_animations:
                return
            
            # 过滤掉sequence为None的动画
            valid_animations = {
                name: anim for name, anim in self._active_animations.items() 
                if anim.sequence is not None
            }
            
            if not valid_animations:
                return
            
            lowest_priority = min(
                valid_animations.values(),
                key=lambda anim: anim.sequence.priority.value if anim.sequence is not None else float('inf')
            )
            
            # 找到对应的动画名称
            for name, anim in valid_animations.items():
                if anim is lowest_priority:
                    self.stop_animation(name)
                    break
                    
        except Exception as e:
            logger.error(f"停止最低优先级动画时发生错误: {e}")
    
    def stop_animation(self, name: str) -> bool:
        """
        停止指定动画
        
        Args:
            name: 动画名称
            
        Returns:
            bool: 停止是否成功
        """
        try:
            with self._lock:
                if name not in self._active_animations:
                    return False
                
                active_anim = self._active_animations[name]
                active_anim.state = AnimationState.STOPPED
                
                # 回收到对象池
                self._recycle_animation(active_anim)
                del self._active_animations[name]
                
                # 触发结束事件（仅在sequence不为空时）
                if active_anim.sequence is not None:
                    self._trigger_event('animation_end', name, active_anim.sequence)
                
                # 更新统计
                self._stats['active_animation_count'] = len(self._active_animations)
                
                logger.info(f"动画 '{name}' 已停止")
                return True
                
        except Exception as e:
            logger.error(f"停止动画时发生错误: {e}")
            return False
    
    def _recycle_animation(self, active_anim: ActiveAnimation):
        """
        回收动画对象到对象池
        
        优化的内存管理策略：
        1. 直接将sequence引用设置为None，避免创建不必要的空对象
        2. 清理所有相关引用和状态
        3. 线程安全的对象池管理
        4. 添加调试日志跟踪回收情况
        
        Args:
            active_anim: 活跃动画对象
        """
        try:
            # 记录回收前的状态（用于调试）
            sequence_name = getattr(active_anim.sequence, 'name', 'Unknown') if active_anim.sequence else 'None'
            logger.debug(f"开始回收动画对象: {sequence_name}")
            
            # 高效的引用清理策略
            # 直接设置为None而不是创建新对象，减少内存分配
            active_anim.sequence = None
            
            # 清理额外参数字典
            if hasattr(active_anim, 'extra_params') and active_anim.extra_params:
                active_anim.extra_params.clear()
            
            # 重置动画状态到初始值
            active_anim.current_time = 0.0
            active_anim.start_time = 0.0
            active_anim.state = AnimationState.IDLE
            active_anim.loop_count = 0
            
            # 线程安全的对象池管理
            # 限制池大小防止内存泄漏，同时提供对象重用
            with self._lock:
                if len(self._animation_pool) < 20:
                    self._animation_pool.append(active_anim)
                    logger.debug(f"动画对象已回收到池中，当前池大小: {len(self._animation_pool)}")
                else:
                    # 池已满，直接丢弃对象让GC处理
                    logger.debug("对象池已满，动画对象将被垃圾回收")
                
        except Exception as e:
            logger.error(f"回收动画对象时发生错误: {e}")
            # 确保即使出错也不会影响系统稳定性
            try:
                active_anim.sequence = None
                if hasattr(active_anim, 'extra_params'):
                    active_anim.extra_params.clear()
            except:
                pass  # 静默处理清理失败的情况
    
    def start(self):
        """
        启动动画序列管理器
        """
        try:
            with self._state_lock:
                if self._running:
                    logger.warning("动画序列管理器已在运行")
                    return
                
                self._running = True
                self._paused = False
                
                # 启动更新线程
                self._update_thread = threading.Thread(
                    target=self._update_loop,
                    name="AnimationUpdater",
                    daemon=True
                )
                self._update_thread.start()
                
                # 启动处理器线程
                self._processor_thread = threading.Thread(
                    target=self._processor_loop,
                    name="AnimationProcessor",
                    daemon=True
                )
                self._processor_thread.start()
                
                logger.info("优化动画序列管理器已启动")
                
        except Exception as e:
            logger.error(f"启动动画序列管理器时发生错误: {e}")
            self._running = False
    
    def stop(self):
        """
        停止动画序列管理器
        """
        try:
            with self._state_lock:
                if not self._running:
                    return
                
                self._running = False
                
                # 等待线程结束
                if self._update_thread and self._update_thread.is_alive():
                    self._update_thread.join(timeout=1.0)
                
                if self._processor_thread and self._processor_thread.is_alive():
                    self._processor_thread.join(timeout=1.0)
                
                # 清理活跃动画
                with self._lock:
                    self._active_animations.clear()
                
                logger.info("优化动画序列管理器已停止")
                
        except Exception as e:
            logger.error(f"停止动画序列管理器时发生错误: {e}")
    
    def _update_loop(self):
        """
        主更新循环
        """
        last_time = time.time()
        
        while self._running:
            try:
                current_time = time.time()
                delta_time = current_time - last_time
                
                if not self._paused:
                    frame_start = time.time()
                    self._update_animations(delta_time)
                    frame_end = time.time()
                    
                    # 更新性能统计
                    frame_time = frame_end - frame_start
                    self._stats['average_frame_time'] = (
                        self._stats['average_frame_time'] * 0.9 + frame_time * 0.1
                    )
                    
                    # 检查掉帧
                    if frame_time > self.frame_time * 1.5:
                        self._stats['dropped_frames'] += 1
                
                last_time = current_time
                
                # 控制帧率
                sleep_time = max(0, self.frame_time - (time.time() - current_time))
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                logger.error(f"动画更新循环发生错误: {e}")
                time.sleep(0.1)  # 防止错误循环
    
    def _processor_loop(self):
        """
        动画处理器循环
        """
        while self._running:
            try:
                # 处理队列中的动画请求
                try:
                    request = self._animation_queue.get(timeout=0.1)
                    self._process_animation_request(request)
                except Empty:
                    continue
                    
            except Exception as e:
                logger.error(f"动画处理器循环发生错误: {e}")
                time.sleep(0.1)
    
    def _process_animation_request(self, request):
        """
        处理动画请求
        
        Args:
            request: 动画请求
        """
        # 这里可以添加复杂的动画调度逻辑
        pass
    
    def _update_animations(self, delta_time: float):
        """
        更新所有活跃动画
        
        Args:
            delta_time: 时间增量
        """
        try:
            with self._lock:
                completed_animations = []
                
                for name, active_anim in self._active_animations.items():
                    # 添加空值检查，防止访问已回收的动画对象
                    if active_anim.sequence is None:
                        completed_animations.append(name)
                        continue
                        
                    if active_anim.state != AnimationState.PLAYING:
                        continue
                    
                    # 更新动画时间
                    active_anim.current_time += delta_time
                    
                    # 检查动画是否完成
                    if active_anim.current_time >= active_anim.sequence.duration:
                        if active_anim.sequence.loop:
                            # 循环动画
                            active_anim.current_time = 0.0
                            active_anim.loop_count += 1
                            self._trigger_event('animation_loop', name, active_anim.sequence)
                        else:
                            # 标记为完成
                            completed_animations.append(name)
                            continue
                    
                    # 更新动画参数
                    self._update_animation_parameters(active_anim)
                
                # 清理完成的动画
                for name in completed_animations:
                    self._complete_animation(name)
                    
        except Exception as e:
            logger.error(f"更新动画时发生错误: {e}")
    
    def _update_animation_parameters(self, active_anim: ActiveAnimation):
        """
        更新单个动画的参数
        
        Args:
            active_anim: 活跃动画对象
        """
        try:
            # 添加空值检查，防止访问已回收的动画对象
            if active_anim.sequence is None:
                logger.warning("尝试更新已回收动画的参数")
                return
                
            sequence = active_anim.sequence
            progress = active_anim.current_time / sequence.duration
            
            # 插值计算当前参数值
            current_params = self._interpolate_parameters(sequence.keyframes, progress)
            
            # 应用额外参数
            for param_name, value in active_anim.extra_params.items():
                current_params[param_name] = value
            
            # 更新全局参数
            for param_name, value in current_params.items():
                self._current_parameters[param_name] = value
            
            # 触发参数更新事件
            self._trigger_event('parameter_update', current_params)
            
        except Exception as e:
            logger.error(f"更新动画参数时发生错误: {e}")
    
    def _interpolate_parameters(self, keyframes: List[AnimationKeyframe], 
                              progress: float) -> Dict[str, float]:
        """
        插值计算参数值
        
        Args:
            keyframes: 关键帧列表
            progress: 动画进度 (0.0-1.0)
            
        Returns:
            Dict[str, float]: 插值后的参数值
        """
        try:
            if not keyframes:
                return {}
            
            # 找到当前进度对应的关键帧区间
            current_time = progress * keyframes[-1].time if keyframes else 0
            
            # 找到前后两个关键帧
            prev_keyframe = keyframes[0]
            next_keyframe = keyframes[-1]
            
            for i, keyframe in enumerate(keyframes):
                if keyframe.time >= current_time:
                    next_keyframe = keyframe
                    if i > 0:
                        prev_keyframe = keyframes[i - 1]
                    break
                prev_keyframe = keyframe
            
            # 计算插值
            if prev_keyframe.time == next_keyframe.time:
                return prev_keyframe.parameters.copy()
            
            t = (current_time - prev_keyframe.time) / (next_keyframe.time - prev_keyframe.time)
            t = max(0.0, min(1.0, t))  # 限制在 [0, 1] 范围内
            
            # 应用缓动函数
            t = self._apply_easing(t, next_keyframe.easing)
            
            # 线性插值
            result = {}
            all_params = set(prev_keyframe.parameters.keys()) | set(next_keyframe.parameters.keys())
            
            for param_name in all_params:
                prev_value = prev_keyframe.parameters.get(param_name, 0.0)
                next_value = next_keyframe.parameters.get(param_name, 0.0)
                result[param_name] = prev_value + (next_value - prev_value) * t
            
            return result
            
        except Exception as e:
            logger.error(f"插值计算时发生错误: {e}")
            return {}
    
    def _apply_easing(self, t: float, easing: str) -> float:
        """
        应用缓动函数
        
        Args:
            t: 时间参数 (0.0-1.0)
            easing: 缓动类型
            
        Returns:
            float: 缓动后的值
        """
        try:
            if easing == "ease_in":
                return t * t
            elif easing == "ease_out":
                return 1 - (1 - t) * (1 - t)
            elif easing == "ease_in_out":
                if t < 0.5:
                    return 2 * t * t
                else:
                    return 1 - 2 * (1 - t) * (1 - t)
            else:  # linear
                return t
                
        except Exception:
            return t  # 发生错误时返回原值
    
    def _complete_animation(self, name: str):
        """
        完成动画
        
        Args:
            name: 动画名称
        """
        try:
            if name in self._active_animations:
                active_anim = self._active_animations[name]
                active_anim.state = AnimationState.COMPLETED
                
                # 触发结束事件（仅在sequence不为空时）
                if active_anim.sequence is not None:
                    self._trigger_event('animation_end', name, active_anim.sequence)
                
                # 回收对象
                self._recycle_animation(active_anim)
                del self._active_animations[name]
                
                # 更新统计
                self._stats['active_animation_count'] = len(self._active_animations)
                
                logger.info(f"动画 '{name}' 已完成")
                
        except Exception as e:
            logger.error(f"完成动画时发生错误: {e}")
    
    def _trigger_event(self, event_type: str, *args):
        """
        触发事件回调
        
        Args:
            event_type: 事件类型
            *args: 事件参数
        """
        try:
            if event_type not in self._event_callbacks:
                return
            
            # 清理失效的弱引用
            valid_callbacks = []
            for callback_ref in self._event_callbacks[event_type]:
                callback = callback_ref()
                if callback is not None:
                    valid_callbacks.append(callback_ref)
                    try:
                        callback(*args)
                    except Exception as e:
                        logger.error(f"事件回调执行错误: {e}")
            
            self._event_callbacks[event_type] = valid_callbacks
            
        except Exception as e:
            logger.error(f"触发事件时发生错误: {e}")
    
    def add_event_listener(self, event_type: str, callback: Callable):
        """
        添加事件监听器
        
        Args:
            event_type: 事件类型
            callback: 回调函数
        """
        try:
            if event_type not in self._event_callbacks:
                logger.warning(f"未知的事件类型: {event_type}")
                return
            
            # 使用弱引用防止内存泄漏
            callback_ref = weakref.ref(callback)
            self._event_callbacks[event_type].append(callback_ref)
            
            logger.info(f"事件监听器已添加: {event_type}")
            
        except Exception as e:
            logger.error(f"添加事件监听器时发生错误: {e}")
    
    def get_current_parameters(self) -> Dict[str, float]:
        """
        获取当前Live2D参数
        
        Returns:
            Dict[str, float]: 当前参数字典
        """
        try:
            with self._lock:
                return self._current_parameters.copy()
        except Exception as e:
            logger.error(f"获取当前参数时发生错误: {e}")
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取性能统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        try:
            with self._lock:
                stats = self._stats.copy()
                stats['registered_sequences'] = len(self._sequences)
                stats['active_animations'] = list(self._active_animations.keys())
                return stats
        except Exception as e:
            logger.error(f"获取统计信息时发生错误: {e}")
            return {}
    
    def pause(self):
        """
        暂停动画更新
        """
        with self._state_lock:
            self._paused = True
            logger.info("动画序列管理器已暂停")
    
    def resume(self):
        """
        恢复动画更新
        """
        with self._state_lock:
            self._paused = False
            logger.info("动画序列管理器已恢复")
    
    def is_running(self) -> bool:
        """
        检查管理器是否在运行
        
        Returns:
            bool: 是否在运行
        """
        return self._running and not self._paused
    
    def cleanup(self):
        """
        清理资源
        """
        try:
            self.stop()
            
            with self._lock:
                self._sequences.clear()
                self._active_animations.clear()
                self._current_parameters.clear()
                self._event_callbacks.clear()
                self._keyframe_pool.clear()
                self._animation_pool.clear()
            
            logger.info("动画序列管理器资源已清理")
            
        except Exception as e:
            logger.error(f"清理资源时发生错误: {e}")
    
    def __del__(self):
        """
        析构函数
        """
        try:
            self.cleanup()
        except Exception:
            pass  # 忽略析构时的错误