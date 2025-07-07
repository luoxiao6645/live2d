"""
高级动画控制器

整合情感分析、状态管理和动画编排，提供统一的Live2D动画控制接口
"""

import logging
import time
from typing import Dict, List, Optional, Any, Callable
import threading

from .emotion_analyzer import EmotionAnalyzer
from .emotion_state_manager import EmotionStateManager
from .animation_sequencer import AnimationSequencer
from .emotion_config import EmotionType, EmotionConfig

# 配置日志
logger = logging.getLogger(__name__)

class AdvancedAnimationController:
    """高级动画控制器类"""
    
    def __init__(self, use_ml_emotion_analysis: bool = False):
        """
        初始化高级动画控制器
        
        Args:
            use_ml_emotion_analysis: 是否使用机器学习进行情感分析
        """
        self.config = EmotionConfig()
        
        # 初始化各个组件
        self.emotion_analyzer = EmotionAnalyzer(use_ml_model=use_ml_emotion_analysis)
        self.emotion_state_manager = EmotionStateManager()
        self.animation_sequencer = AnimationSequencer()
        
        # 控制器状态
        self.is_active = True
        self.auto_idle_animations = True
        self.auto_blink = True
        
        # 自动动画定时器
        self._idle_timer = None
        self._blink_timer = None
        self._last_interaction_time = time.time()
        
        # 事件回调
        self.event_callbacks: Dict[str, List[Callable]] = {
            "emotion_changed": [],
            "animation_triggered": [],
            "parameters_updated": []
        }
        
        # 启动自动动画
        self._start_auto_animations()
        
        logger.info("高级动画控制器初始化完成")
    
    def process_text_input(self, text: str, trigger_animation: bool = True) -> Dict[str, Any]:
        """
        处理文本输入，分析情感并更新动画
        
        Args:
            text: 输入文本
            trigger_animation: 是否触发相应动画
            
        Returns:
            处理结果
        """
        try:
            # 更新最后交互时间
            self._last_interaction_time = time.time()
            
            # 情感分析
            emotion_analysis = self.emotion_analyzer.analyze_emotion(text)
            
            # 更新情感状态
            state_updated = self.emotion_state_manager.update_emotion(emotion_analysis)
            
            # 获取当前状态
            current_state = self.emotion_state_manager.get_current_state()
            
            # 触发相应动画
            triggered_animations = []
            if trigger_animation and state_updated:
                triggered_animations = self._trigger_emotion_animations(
                    current_state["emotion_type"], 
                    current_state["intensity"]
                )
            
            # 触发事件
            self._trigger_event("emotion_changed", {
                "emotion_analysis": emotion_analysis,
                "current_state": current_state,
                "triggered_animations": triggered_animations
            })
            
            result = {
                "success": True,
                "emotion_analysis": emotion_analysis,
                "current_state": current_state,
                "triggered_animations": triggered_animations,
                "live2d_parameters": self.get_current_live2d_parameters()
            }
            
            logger.debug(f"文本处理完成: {emotion_analysis['primary_emotion']}")
            return result
            
        except Exception as e:
            logger.error(f"处理文本输入失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _trigger_emotion_animations(self, emotion_type: str, intensity: float) -> List[str]:
        """根据情感类型和强度触发相应动画"""
        triggered_animations = []
        
        try:
            emotion_enum = EmotionType(emotion_type)
            
            # 根据情感类型选择动画
            if emotion_enum == EmotionType.HAPPY and intensity > 0.6:
                # 开心时可能挥手或跳跃
                if intensity > 0.8:
                    self.animation_sequencer.play_animation("wave")
                    triggered_animations.append("wave")
            
            elif emotion_enum == EmotionType.SURPRISED and intensity > 0.5:
                # 惊讶时可能有跳跃动作
                self.animation_sequencer.play_animation("surprise_jump")
                triggered_animations.append("surprise_jump")
            
            elif emotion_enum == EmotionType.CONFUSED and intensity > 0.4:
                # 困惑时进入思考状态
                self.animation_sequencer.play_animation("thinking")
                triggered_animations.append("thinking")
            
            elif emotion_enum == EmotionType.SHY and intensity > 0.5:
                # 害羞时可能有躲藏动作
                self.animation_sequencer.play_animation("shy_hide")
                triggered_animations.append("shy_hide")
            
            elif emotion_enum == EmotionType.EXCITED and intensity > 0.7:
                # 兴奋时可能有弹跳动作
                self.animation_sequencer.play_animation("excited_bounce")
                triggered_animations.append("excited_bounce")
            
            # 根据情感强度调整眨眼频率
            self._adjust_blink_frequency(emotion_enum, intensity)
            
        except Exception as e:
            logger.error(f"触发情感动画失败: {e}")
        
        return triggered_animations
    
    def _adjust_blink_frequency(self, emotion_type: EmotionType, intensity: float):
        """根据情感调整眨眼频率"""
        base_interval = 3.0  # 基础眨眼间隔（秒）
        
        if emotion_type == EmotionType.EXCITED:
            # 兴奋时眨眼更频繁
            interval = base_interval * (1 - intensity * 0.5)
        elif emotion_type == EmotionType.SLEEPY:
            # 困倦时眨眼更慢
            interval = base_interval * (1 + intensity * 2)
        elif emotion_type == EmotionType.SURPRISED:
            # 惊讶时暂时不眨眼
            interval = base_interval * 3
        else:
            interval = base_interval
        
        # 这里可以调整眨眼定时器的间隔
        # 实际实现中需要重新设置定时器
    
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
            success = self.animation_sequencer.play_animation(gesture_name, **kwargs)
            
            if success:
                self._trigger_event("animation_triggered", {
                    "animation_name": gesture_name,
                    "trigger_type": "manual",
                    "parameters": kwargs
                })
                
                logger.info(f"手动触发手势: {gesture_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"触发手势失败: {e}")
            return False
    
    def force_emotion(self, emotion_type: str, intensity: float = 0.8) -> bool:
        """
        强制设置情感状态
        
        Args:
            emotion_type: 情感类型
            intensity: 情感强度
            
        Returns:
            是否成功设置
        """
        try:
            emotion_enum = EmotionType(emotion_type)
            success = self.emotion_state_manager.force_emotion(emotion_enum, intensity)
            
            if success:
                # 触发相应动画
                self._trigger_emotion_animations(emotion_type, intensity)
                
                logger.info(f"强制设置情感: {emotion_type}, 强度: {intensity}")
            
            return success
            
        except Exception as e:
            logger.error(f"强制设置情感失败: {e}")
            return False
    
    def get_current_live2d_parameters(self) -> Dict[str, float]:
        """获取当前Live2D参数"""
        try:
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
        """获取系统状态信息"""
        try:
            current_state = self.emotion_state_manager.get_current_state()
            active_animations = self.animation_sequencer.get_active_animations()
            
            return {
                "is_active": self.is_active,
                "current_emotion": current_state,
                "active_animations": active_animations,
                "auto_idle_animations": self.auto_idle_animations,
                "auto_blink": self.auto_blink,
                "last_interaction_time": self._last_interaction_time,
                "time_since_last_interaction": time.time() - self._last_interaction_time
            }
            
        except Exception as e:
            logger.error(f"获取系统状态失败: {e}")
            return {"error": str(e)}
    
    def _start_auto_animations(self):
        """启动自动动画"""
        if self.auto_blink:
            self._start_auto_blink()
        
        if self.auto_idle_animations:
            self._start_idle_animations()
    
    def _start_auto_blink(self):
        """启动自动眨眼"""
        def blink_loop():
            while self.is_active and self.auto_blink:
                # 随机眨眼间隔（2-5秒）
                import random
                interval = random.uniform(2.0, 5.0)
                time.sleep(interval)
                
                # 触发眨眼动画
                if self.is_active:
                    self.animation_sequencer.play_animation("blink", priority=1)
        
        self._blink_timer = threading.Thread(target=blink_loop, daemon=True)
        self._blink_timer.start()
    
    def _start_idle_animations(self):
        """启动空闲动画"""
        def idle_loop():
            while self.is_active and self.auto_idle_animations:
                time.sleep(10.0)  # 每10秒检查一次
                
                # 如果长时间没有交互，触发空闲动画
                time_since_interaction = time.time() - self._last_interaction_time
                if time_since_interaction > 30.0:  # 30秒无交互
                    # 随机选择空闲动画
                    import random
                    idle_animations = ["thinking", "blink"]
                    animation = random.choice(idle_animations)
                    
                    if self.is_active:
                        self.animation_sequencer.play_animation(animation, priority=1)
        
        self._idle_timer = threading.Thread(target=idle_loop, daemon=True)
        self._idle_timer.start()
    
    def set_auto_animations(self, auto_blink: bool = None, auto_idle: bool = None):
        """设置自动动画开关"""
        if auto_blink is not None:
            self.auto_blink = auto_blink
            logger.info(f"自动眨眼: {'开启' if auto_blink else '关闭'}")
        
        if auto_idle is not None:
            self.auto_idle_animations = auto_idle
            logger.info(f"自动空闲动画: {'开启' if auto_idle else '关闭'}")
    
    def add_event_listener(self, event_type: str, callback: Callable):
        """添加事件监听器"""
        if event_type in self.event_callbacks:
            self.event_callbacks[event_type].append(callback)
            logger.debug(f"添加事件监听器: {event_type}")
    
    def _trigger_event(self, event_type: str, data: Any):
        """触发事件"""
        if event_type in self.event_callbacks:
            for callback in self.event_callbacks[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"事件回调执行失败: {e}")
    
    def get_emotion_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取情感历史"""
        return self.emotion_state_manager.get_emotion_history(limit)
    
    def analyze_text_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量分析文本情感"""
        return self.emotion_analyzer.analyze_emotion_sequence(texts)
    
    def export_animation_data(self) -> Dict[str, Any]:
        """导出动画数据（用于调试或保存）"""
        try:
            return {
                "current_emotion_state": self.emotion_state_manager.get_current_state(),
                "active_animations": self.animation_sequencer.get_active_animations(),
                "current_parameters": self.get_current_live2d_parameters(),
                "system_status": self.get_system_status(),
                "emotion_history": self.get_emotion_history(20)
            }
        except Exception as e:
            logger.error(f"导出动画数据失败: {e}")
            return {"error": str(e)}
    
    def reset_to_neutral(self):
        """重置到中性状态"""
        try:
            # 停止所有动画
            for animation_id in list(self.animation_sequencer.active_sequences.keys()):
                self.animation_sequencer.stop_animation(animation_id)
            
            # 强制设置为中性情感
            self.force_emotion("neutral", 0.1)
            
            logger.info("重置到中性状态")
            
        except Exception as e:
            logger.error(f"重置状态失败: {e}")
    
    def shutdown(self):
        """关闭控制器"""
        try:
            self.is_active = False
            
            # 停止动画序列编排器
            self.animation_sequencer.stop()
            
            logger.info("高级动画控制器已关闭")
            
        except Exception as e:
            logger.error(f"关闭控制器失败: {e}")
    
    def __del__(self):
        """析构函数"""
        self.shutdown()
