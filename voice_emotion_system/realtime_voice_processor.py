"""
实时语音处理器

负责流式语音生成、音频缓存、播放队列管理等实时处理功能
"""

import logging
import asyncio
import threading
import queue
import time
import os
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

try:
    from pydub import AudioSegment
    from pydub.playback import play
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False
    logger.warning("pydub不可用，音频处理功能将受限")

from .voice_emotion_config import VoiceEmotionConfig, EmotionType, VoiceLanguage

# 配置日志
logger = logging.getLogger(__name__)

class AudioTask:
    """音频任务类"""
    
    def __init__(self, task_id: str, text: str, emotion: EmotionType,
                 intensity: float, priority: int = 1):
        self.task_id = task_id
        self.text = text
        self.emotion = emotion
        self.intensity = intensity
        self.priority = priority
        self.created_at = time.time()
        self.status = "pending"  # pending, processing, completed, failed
        self.result = None
        self.error = None

class RealtimeVoiceProcessor:
    """实时语音处理器类"""
    
    def __init__(self, synthesizer=None):
        """
        初始化实时语音处理器
        
        Args:
            synthesizer: 语音合成器实例
        """
        self.config = VoiceEmotionConfig()
        self.synthesizer = synthesizer
        
        # 任务队列
        self.task_queue = queue.PriorityQueue()
        self.processing_tasks = {}
        self.completed_tasks = {}
        
        # 音频缓存
        self.audio_cache = {}
        self.cache_stats = {"hits": 0, "misses": 0}
        
        # 播放队列
        self.playback_queue = queue.Queue()
        self.current_playback = None
        
        # 线程池
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.PERFORMANCE_CONFIG["max_concurrent_synthesis"]
        )
        
        # 控制标志
        self.is_running = False
        self.processing_thread = None
        self.playback_thread = None
        
        # 回调函数
        self.on_synthesis_complete = None
        self.on_playback_start = None
        self.on_playback_complete = None
        
        logger.info("实时语音处理器初始化完成")
    
    def start(self):
        """启动实时处理器"""
        if self.is_running:
            logger.warning("实时处理器已在运行")
            return
        
        self.is_running = True
        
        # 启动处理线程
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        # 启动播放线程
        if AUDIO_PROCESSING_AVAILABLE:
            self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
            self.playback_thread.start()
        
        logger.info("实时语音处理器已启动")
    
    def stop(self):
        """停止实时处理器"""
        self.is_running = False
        
        # 等待线程结束
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)
        
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=5)
        
        # 关闭线程池
        self.executor.shutdown(wait=True)
        
        logger.info("实时语音处理器已停止")
    
    def submit_synthesis_task(self, text: str, emotion: EmotionType = EmotionType.NEUTRAL,
                            intensity: float = 1.0, priority: int = 1,
                            auto_play: bool = False) -> str:
        """
        提交语音合成任务
        
        Args:
            text: 要合成的文本
            emotion: 情感类型
            intensity: 情感强度
            priority: 任务优先级 (1-5, 数字越小优先级越高)
            auto_play: 是否自动播放
            
        Returns:
            任务ID
        """
        try:
            # 生成任务ID
            task_id = f"task_{int(time.time() * 1000)}_{hash(text) % 10000}"
            
            # 创建任务
            task = AudioTask(task_id, text, emotion, intensity, priority)
            
            # 检查缓存
            cache_key = self._generate_cache_key(text, emotion, intensity)
            if cache_key in self.audio_cache:
                # 缓存命中
                task.status = "completed"
                task.result = self.audio_cache[cache_key]
                self.completed_tasks[task_id] = task
                self.cache_stats["hits"] += 1
                
                # 如果需要自动播放，添加到播放队列
                if auto_play and AUDIO_PROCESSING_AVAILABLE:
                    self.add_to_playback_queue(task.result["file_path"])
                
                logger.info(f"任务 {task_id} 使用缓存完成")
                return task_id
            
            self.cache_stats["misses"] += 1
            
            # 添加到队列
            self.task_queue.put((priority, time.time(), task))
            
            # 设置自动播放标志
            if auto_play:
                task.auto_play = True
            
            logger.info(f"任务 {task_id} 已提交到队列")
            return task_id
            
        except Exception as e:
            logger.error(f"提交合成任务失败: {e}")
            return ""
    
    def _processing_loop(self):
        """处理循环"""
        while self.is_running:
            try:
                # 获取任务（超时1秒）
                try:
                    priority, timestamp, task = self.task_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # 检查任务是否过期
                if time.time() - task.created_at > self.config.REALTIME_CONFIG["processing_timeout"]:
                    task.status = "failed"
                    task.error = "任务超时"
                    self.completed_tasks[task.task_id] = task
                    continue
                
                # 开始处理
                task.status = "processing"
                self.processing_tasks[task.task_id] = task
                
                # 提交到线程池
                future = self.executor.submit(self._process_task, task)
                
                # 等待完成（非阻塞）
                try:
                    result = future.result(timeout=0.1)
                    self._handle_task_completion(task, result)
                except:
                    # 任务还在处理中，继续下一个
                    pass
                
            except Exception as e:
                logger.error(f"处理循环异常: {e}")
    
    def _process_task(self, task: AudioTask) -> Dict[str, Any]:
        """处理单个任务"""
        try:
            if not self.synthesizer:
                return {"success": False, "error": "语音合成器不可用"}
            
            # 执行语音合成
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(
                    self.synthesizer.synthesize_with_emotion(
                        task.text, task.emotion, task.intensity
                    )
                )
            finally:
                loop.close()
            
            return result
            
        except Exception as e:
            logger.error(f"任务处理失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _handle_task_completion(self, task: AudioTask, result: Dict[str, Any]):
        """处理任务完成"""
        try:
            # 更新任务状态
            if result["success"]:
                task.status = "completed"
                task.result = result
                
                # 添加到缓存
                cache_key = self._generate_cache_key(task.text, task.emotion, task.intensity)
                self.audio_cache[cache_key] = result
                self._cleanup_cache()
                
                # 自动播放
                if hasattr(task, 'auto_play') and task.auto_play and AUDIO_PROCESSING_AVAILABLE:
                    self.add_to_playback_queue(result["file_path"])
                
                # 调用回调
                if self.on_synthesis_complete:
                    self.on_synthesis_complete(task.task_id, result)
                
                logger.info(f"任务 {task.task_id} 完成")
            else:
                task.status = "failed"
                task.error = result.get("error", "未知错误")
                logger.error(f"任务 {task.task_id} 失败: {task.error}")
            
            # 移动到完成队列
            self.completed_tasks[task.task_id] = task
            if task.task_id in self.processing_tasks:
                del self.processing_tasks[task.task_id]
            
        except Exception as e:
            logger.error(f"处理任务完成失败: {e}")
    
    def _playback_loop(self):
        """播放循环"""
        while self.is_running:
            try:
                # 获取播放任务
                try:
                    audio_file = self.playback_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # 播放音频
                self._play_audio_file(audio_file)
                
            except Exception as e:
                logger.error(f"播放循环异常: {e}")
    
    def _play_audio_file(self, file_path: str):
        """播放音频文件"""
        try:
            if not os.path.exists(file_path):
                logger.error(f"音频文件不存在: {file_path}")
                return
            
            self.current_playback = file_path
            
            # 调用播放开始回调
            if self.on_playback_start:
                self.on_playback_start(file_path)
            
            # 播放音频
            audio = AudioSegment.from_file(file_path)
            play(audio)
            
            # 调用播放完成回调
            if self.on_playback_complete:
                self.on_playback_complete(file_path)
            
            self.current_playback = None
            logger.info(f"音频播放完成: {file_path}")
            
        except Exception as e:
            logger.error(f"播放音频失败: {e}")
            self.current_playback = None
    
    def add_to_playback_queue(self, file_path: str):
        """添加到播放队列"""
        if AUDIO_PROCESSING_AVAILABLE:
            self.playback_queue.put(file_path)
            logger.info(f"音频已添加到播放队列: {file_path}")
        else:
            logger.warning("音频播放功能不可用")
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """获取任务状态"""
        # 检查处理中的任务
        if task_id in self.processing_tasks:
            task = self.processing_tasks[task_id]
            return {
                "task_id": task_id,
                "status": task.status,
                "created_at": task.created_at,
                "processing_time": time.time() - task.created_at
            }
        
        # 检查已完成的任务
        if task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
            result = {
                "task_id": task_id,
                "status": task.status,
                "created_at": task.created_at,
                "processing_time": time.time() - task.created_at
            }
            
            if task.status == "completed":
                result["result"] = task.result
            elif task.status == "failed":
                result["error"] = task.error
            
            return result
        
        return {"task_id": task_id, "status": "not_found"}
    
    def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态"""
        return {
            "pending_tasks": self.task_queue.qsize(),
            "processing_tasks": len(self.processing_tasks),
            "completed_tasks": len(self.completed_tasks),
            "playback_queue": self.playback_queue.qsize() if AUDIO_PROCESSING_AVAILABLE else 0,
            "current_playback": self.current_playback,
            "cache_stats": self.cache_stats,
            "is_running": self.is_running
        }
    
    def _generate_cache_key(self, text: str, emotion: EmotionType, intensity: float) -> str:
        """生成缓存键"""
        import hashlib
        content = f"{text}_{emotion.value}_{intensity:.2f}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _cleanup_cache(self):
        """清理缓存"""
        try:
            max_cache_size = self.config.AUDIO_CONFIG.get("max_cache_size", 100)
            
            if len(self.audio_cache) > max_cache_size:
                # 删除最旧的缓存项
                sorted_cache = sorted(
                    self.audio_cache.items(),
                    key=lambda x: os.path.getmtime(x[1]["file_path"]) if os.path.exists(x[1]["file_path"]) else 0
                )
                
                # 删除超出限制的缓存
                for i in range(len(sorted_cache) - max_cache_size):
                    cache_key, cache_data = sorted_cache[i]
                    
                    # 删除文件
                    if os.path.exists(cache_data["file_path"]):
                        os.remove(cache_data["file_path"])
                    
                    # 从缓存中移除
                    del self.audio_cache[cache_key]
                
                logger.info(f"缓存清理完成，保留 {max_cache_size} 个项目")
                
        except Exception as e:
            logger.error(f"缓存清理失败: {e}")
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        try:
            # 检查是否在处理队列中
            if task_id in self.processing_tasks:
                task = self.processing_tasks[task_id]
                task.status = "cancelled"
                self.completed_tasks[task_id] = task
                del self.processing_tasks[task_id]
                logger.info(f"任务 {task_id} 已取消")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"取消任务失败: {e}")
            return False
    
    def clear_completed_tasks(self):
        """清理已完成的任务"""
        self.completed_tasks.clear()
        logger.info("已完成任务已清理")
    
    def set_callbacks(self, on_synthesis_complete: Callable = None,
                     on_playback_start: Callable = None,
                     on_playback_complete: Callable = None):
        """设置回调函数"""
        self.on_synthesis_complete = on_synthesis_complete
        self.on_playback_start = on_playback_start
        self.on_playback_complete = on_playback_complete
        logger.info("回调函数已设置")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        cache_hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            "total_requests": total_requests,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.audio_cache),
            "active_threads": threading.active_count(),
            "executor_stats": {
                "max_workers": self.executor._max_workers,
                "pending_tasks": len(self.executor._pending_work_items) if hasattr(self.executor, '_pending_work_items') else 0
            }
        }
    
    def __del__(self):
        """析构函数"""
        self.stop()
