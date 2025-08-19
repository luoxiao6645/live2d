"""优化后的高级动画控制器

主要优化：
1. 解决频繁眨眼问题 - 添加动画冲突检测
2. 改进线程安全性和资源管理
3. 增强错误处理和日志记录
4. 优化性能和内存使用
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import queue
import weakref

from .emotion_state_manager import EmotionStateManager
from .emotion_analyzer import EmotionAnalyzer
from .animation_sequencer import AnimationSequencer
from .emotion_config import EmotionType

# 配置日志
logger = logging.getLogger(__name__)

class AnimationPriority(Enum):
    """动画优先级枚举"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class AnimationRequest:
    """动画请求数据类"""
    animation_name: str
    priority: AnimationPriority
    timestamp: float
    parameters: Dict[str, Any]
    source: str  # 动画来源（auto_blink, idle, manual等）

class OptimizedAdvancedAnimationController:
    """优化后的高级动画控制器类"""
    
    def __init__(self):
        """初始化优化后的高级动画控制器"""
        try:
            # 核心组件初始化
            self.emotion_state_manager = EmotionStateManager()
            self.emotion_analyzer = EmotionAnalyzer()
            self.animation_sequencer = AnimationSequencer()
            
            # 控制器状态
            self.is_active = True
            self._last_interaction_time = time.time()
            
            # 优化后的自动动画控制
            self.auto_blink = True
            self.auto_idle_animations = True
            self._blink_interval_range = (3.0, 8.0)  # 增加眨眼间隔
            self._last_blink_time = 0.0
            self._min_blink_interval = 2.0  # 最小眨眼间隔
            
            # 线程安全控制
            self._lock = threading.RLock()  # 使用可重入锁
            self._animation_queue = queue.PriorityQueue()
            self._active_animations = set()  # 跟踪活动动画
            
            # 线程管理
            self._threads = weakref.WeakSet()  # 使用弱引用管理线程
            self._shutdown_event = threading.Event()
            
            # 性能监控
            self._animation_stats = {
                'total_animations': 0,
                'blink_count': 0,
                'idle_count': 0,
                'manual_count': 0,
                'conflicts_resolved': 0
            }
            
            # 事件回调系统
            self.event_callbacks = {
                "emotion_changed": [],
                "animation_started": [],
                "animation_completed": [],
                "animation_conflict": []
            }
            
            # 启动优化后的自动动画系统
            self._start_optimized_auto_animations()
            
            logger.info("优化后的高级动画控制器初始化完成")
            
        except Exception as e:
            logger.error(f"初始化高级动画控制器失败: {e}")
            raise
    
    def analyze_emotion(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        分析文本情感并触发相应动画
        
        Args:
            text: 要分析的文本
            context: 上下文信息
            
        Returns:
            情感分析结果
        """
        try:
            with self._lock:
                # 更新最后交互时间
                self._last_interaction_time = time.time()
                
                # 分析情感
                emotion_result = self.emotion_analyzer.analyze_emotion(text, context)
                
                if emotion_result and 'emotion' in emotion_result:
                    emotion_type = emotion_result['emotion']
                    intensity = emotion_result.get('intensity', 0.5)
                    
                    # 更新情感状态
                    success = self.emotion_state_manager.update_emotion(
                        EmotionType(emotion_type), intensity
                    )
                    
                    if success:
                        # 智能触发情感动画（避免与眨眼冲突）
                        self._trigger_emotion_animations_smart(emotion_type, intensity)
                        
                        # 触发情感变化事件
                        self._trigger_event("emotion_changed", {
                            "emotion": emotion_type,
                            "intensity": intensity,
                            "text": text
                        })
                
                return emotion_result
                
        except Exception as e:
            logger.error(f"情感分析失败: {e}")
            return {"error": str(e)}
    
    def _trigger_emotion_animations_smart(self, emotion_type: str, intensity: float):
        """
        智能触发情感动画，避免与其他动画冲突
        
        Args:
            emotion_type: 情感类型
            intensity: 情感强度
        """
        try:
            # 根据情感类型选择合适的动画
            animation_mapping = {
                "happy": "smile",
                "sad": "sad_expression",
                "angry": "frown",
                "surprised": "surprise",
                "thinking": "thinking",
                "neutral": None
            }
            
            animation_name = animation_mapping.get(emotion_type)
            if animation_name:
                # 创建动画请求
                request = AnimationRequest(
                    animation_name=animation_name,
                    priority=AnimationPriority.NORMAL,
                    timestamp=time.time(),
                    parameters={"intensity": intensity},
                    source="emotion"
                )
                
                # 添加到动画队列
                self._queue_animation_request(request)
                
        except Exception as e:
            logger.error(f"触发情感动画失败: {e}")
    
    def _queue_animation_request(self, request: AnimationRequest):
        """
        将动画请求添加到队列中
        
        Args:
            request: 动画请求
        """
        try:
            # 检查动画冲突
            if self._check_animation_conflict(request):
                self._animation_stats['conflicts_resolved'] += 1
                logger.debug(f"动画冲突已解决: {request.animation_name}")
                return
            
            # 添加到优先级队列（优先级越高，数值越小）
            priority_value = 5 - request.priority.value
            self._animation_queue.put((priority_value, request.timestamp, request))
            
            logger.debug(f"动画请求已排队: {request.animation_name}")
            
        except Exception as e:
            logger.error(f"排队动画请求失败: {e}")
    
    def _check_animation_conflict(self, request: AnimationRequest) -> bool:
        """
        检查动画冲突
        
        Args:
            request: 动画请求
            
        Returns:
            是否存在冲突
        """
        try:
            current_time = time.time()
            
            # 特殊处理眨眼动画冲突
            if request.animation_name == "blink":
                # 检查最近是否已经眨眼
                if current_time - self._last_blink_time < self._min_blink_interval:
                    logger.debug(f"眨眼动画被跳过，距离上次眨眼仅 {current_time - self._last_blink_time:.1f}秒")
                    return True
                
                # 检查是否有高优先级动画正在播放
                for active_anim in self._active_animations:
                    if active_anim != "blink" and request.priority.value <= AnimationPriority.NORMAL.value:
                        logger.debug(f"眨眼动画被跳过，{active_anim}正在播放")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"检查动画冲突失败: {e}")
            return False
    
    def _start_optimized_auto_animations(self):
        """
        启动优化后的自动动画系统
        """
        try:
            # 启动动画处理线程
            animation_thread = threading.Thread(
                target=self._animation_processor_loop,
                name="AnimationProcessor",
                daemon=True
            )
            animation_thread.start()
            self._threads.add(animation_thread)
            
            # 启动优化后的自动眨眼
            if self.auto_blink:
                blink_thread = threading.Thread(
                    target=self._optimized_auto_blink_loop,
                    name="OptimizedAutoBlink",
                    daemon=True
                )
                blink_thread.start()
                self._threads.add(blink_thread)
            
            # 启动优化后的空闲动画
            if self.auto_idle_animations:
                idle_thread = threading.Thread(
                    target=self._optimized_idle_loop,
                    name="OptimizedIdle",
                    daemon=True
                )
                idle_thread.start()
                self._threads.add(idle_thread)
            
            logger.info("优化后的自动动画系统已启动")
            
        except Exception as e:
            logger.error(f"启动自动动画系统失败: {e}")
    
    def _animation_processor_loop(self):
        """
        动画处理循环
        """
        while self.is_active and not self._shutdown_event.is_set():
            try:
                # 从队列中获取动画请求（超时1秒）
                try:
                    _, _, request = self._animation_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # 执行动画
                self._execute_animation_request(request)
                
            except Exception as e:
                logger.error(f"动画处理循环错误: {e}")
                time.sleep(0.1)
    
    def _execute_animation_request(self, request: AnimationRequest):
        """
        执行动画请求
        
        Args:
            request: 动画请求
        """
        try:
            with self._lock:
                # 再次检查冲突（双重检查）
                if self._check_animation_conflict(request):
                    return
                
                # 播放动画
                success = self.animation_sequencer.play_animation(
                    request.animation_name,
                    priority=request.priority.value,
                    **request.parameters
                )
                
                if success:
                    # 更新统计信息
                    self._animation_stats['total_animations'] += 1
                    if request.source == 'auto_blink':
                        self._animation_stats['blink_count'] += 1
                        self._last_blink_time = time.time()
                    elif request.source == 'idle':
                        self._animation_stats['idle_count'] += 1
                    elif request.source == 'manual':
                        self._animation_stats['manual_count'] += 1
                    
                    # 添加到活动动画集合
                    self._active_animations.add(request.animation_name)
                    
                    # 触发动画开始事件
                    self._trigger_event("animation_started", request)
                    
                    # 设置动画完成回调
                    threading.Timer(
                        self.animation_sequencer.sequences[request.animation_name].duration,
                        self._on_animation_completed,
                        args=[request.animation_name]
                    ).start()
                    
                    logger.debug(f"动画执行成功: {request.animation_name}")
                
        except Exception as e:
            logger.error(f"执行动画请求失败: {e}")
    
    def _on_animation_completed(self, animation_name: str):
        """
        动画完成回调
        
        Args:
            animation_name: 动画名称
        """
        try:
            with self._lock:
                self._active_animations.discard(animation_name)
                self._trigger_event("animation_completed", {"animation": animation_name})
                
        except Exception as e:
            logger.error(f"动画完成回调失败: {e}")
    
    def _optimized_auto_blink_loop(self):
        """
        优化后的自动眨眼循环
        """
        import random
        
        while self.is_active and not self._shutdown_event.is_set():
            try:
                # 动态计算眨眼间隔
                min_interval, max_interval = self._blink_interval_range
                interval = random.uniform(min_interval, max_interval)
                
                # 等待间隔时间
                if self._shutdown_event.wait(interval):
                    break
                
                # 创建眨眼请求
                if self.auto_blink and self.is_active:
                    request = AnimationRequest(
                        animation_name="blink",
                        priority=AnimationPriority.LOW,
                        timestamp=time.time(),
                        parameters={},
                        source="auto_blink"
                    )
                    
                    self._queue_animation_request(request)
                
            except Exception as e:
                logger.error(f"自动眨眼循环错误: {e}")
                time.sleep(1.0)
    
    def _optimized_idle_loop(self):
        """
        优化后的空闲动画循环
        """
        import random
        
        while self.is_active and not self._shutdown_event.is_set():
            try:
                # 每15秒检查一次
                if self._shutdown_event.wait(15.0):
                    break
                
                # 检查是否长时间无交互
                time_since_interaction = time.time() - self._last_interaction_time
                if time_since_interaction > 45.0:  # 45秒无交互
                    # 选择非眨眼的空闲动画
                    idle_animations = ["thinking", "nod"]
                    animation = random.choice(idle_animations)
                    
                    request = AnimationRequest(
                        animation_name=animation,
                        priority=AnimationPriority.LOW,
                        timestamp=time.time(),
                        parameters={},
                        source="idle"
                    )
                    
                    self._queue_animation_request(request)
                
            except Exception as e:
                logger.error(f"空闲动画循环错误: {e}")
                time.sleep(1.0)
    
    def trigger_gesture(self, gesture_name: str, **kwargs) -> bool:
        """
        手动触发手势动画
        
        Args:
            gesture_name: 手势名称
            **kwargs: 额外参数
            
        Returns:
            是否成功触发
        """
        try:
            request = AnimationRequest(
                animation_name=gesture_name,
                priority=AnimationPriority.HIGH,
                timestamp=time.time(),
                parameters=kwargs,
                source="manual"
            )
            
            self._queue_animation_request(request)
            
            logger.info(f"手动触发手势: {gesture_name}")
            return True
            
        except Exception as e:
            logger.error(f"触发手势失败: {e}")
            return False
    
    def get_current_live2d_parameters(self) -> Dict[str, float]:
        """
        获取当前Live2D参数
        
        Returns:
            当前参数字典
        """
        try:
            with self._lock:
                # 获取情感状态参数
                emotion_params = self.emotion_state_manager.get_live2d_parameters()
                
                # 获取动画参数
                animation_params = self.animation_sequencer.get_current_parameters()
                
                # 合并参数（动画参数优先级更高）
                final_params = emotion_params.copy()
                final_params.update(animation_params)
                
                return final_params
                
        except Exception as e:
            logger.error(f"获取Live2D参数失败: {e}")
            return {}
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        获取系统状态信息
        
        Returns:
            系统状态字典
        """
        try:
            with self._lock:
                current_state = self.emotion_state_manager.get_current_state()
                active_animations = list(self._active_animations)
                
                return {
                    "is_active": self.is_active,
                    "current_emotion": current_state,
                    "active_animations": active_animations,
                    "auto_blink": self.auto_blink,
                    "auto_idle_animations": self.auto_idle_animations,
                    "last_interaction_time": self._last_interaction_time,
                    "time_since_last_interaction": time.time() - self._last_interaction_time,
                    "animation_stats": self._animation_stats.copy(),
                    "queue_size": self._animation_queue.qsize(),
                    "thread_count": len(self._threads)
                }
                
        except Exception as e:
            logger.error(f"获取系统状态失败: {e}")
            return {"error": str(e)}
    
    def set_auto_animations(self, auto_blink: bool = None, auto_idle: bool = None):
        """
        设置自动动画开关
        
        Args:
            auto_blink: 是否启用自动眨眼
            auto_idle: 是否启用自动空闲动画
        """
        try:
            with self._lock:
                if auto_blink is not None:
                    self.auto_blink = auto_blink
                    logger.info(f"自动眨眼: {'开启' if auto_blink else '关闭'}")
                
                if auto_idle is not None:
                    self.auto_idle_animations = auto_idle
                    logger.info(f"自动空闲动画: {'开启' if auto_idle else '关闭'}")
                    
        except Exception as e:
            logger.error(f"设置自动动画失败: {e}")
    
    def add_event_listener(self, event_type: str, callback: Callable):
        """
        添加事件监听器
        
        Args:
            event_type: 事件类型
            callback: 回调函数
        """
        try:
            if event_type in self.event_callbacks:
                self.event_callbacks[event_type].append(callback)
                logger.debug(f"添加事件监听器: {event_type}")
            else:
                logger.warning(f"未知事件类型: {event_type}")
                
        except Exception as e:
            logger.error(f"添加事件监听器失败: {e}")
    
    def _trigger_event(self, event_type: str, data: Any):
        """
        触发事件
        
        Args:
            event_type: 事件类型
            data: 事件数据
        """
        try:
            if event_type in self.event_callbacks:
                for callback in self.event_callbacks[event_type]:
                    try:
                        callback(data)
                    except Exception as e:
                        logger.error(f"事件回调执行失败: {e}")
                        
        except Exception as e:
            logger.error(f"触发事件失败: {e}")
    
    def shutdown(self):
        """
        优雅关闭控制器
        """
        try:
            logger.info("开始关闭高级动画控制器...")
            
            # 设置关闭标志
            self.is_active = False
            self._shutdown_event.set()
            
            # 清空动画队列
            while not self._animation_queue.empty():
                try:
                    self._animation_queue.get_nowait()
                except queue.Empty:
                    break
            
            # 停止动画序列编排器
            if hasattr(self.animation_sequencer, 'stop'):
                self.animation_sequencer.stop()
            
            # 等待线程结束（最多等待5秒）
            for thread in list(self._threads):
                if thread.is_alive():
                    thread.join(timeout=5.0)
            
            logger.info("高级动画控制器已关闭")
            
        except Exception as e:
            logger.error(f"关闭控制器失败: {e}")
    
    def __del__(self):
        """析构函数"""
        try:
            self.shutdown()
        except:
            pass

# 为了向后兼容，保留原类名
AdvancedAnimationController = OptimizedAdvancedAnimationController
