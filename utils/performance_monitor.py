# -*- coding: utf-8 -*-
"""
性能监控系统模块
提供系统资源监控、性能分析和报告功能
"""

import os
import time
import psutil
import threading
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
import json
from datetime import datetime, timedelta

# 可选导入GPUtil
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPUtil = None
    GPU_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """
    性能指标数据类
    
    存储单次性能测量的所有相关指标
    """
    timestamp: float                    # 测量时间戳
    cpu_percent: float                  # CPU使用率百分比
    memory_percent: float               # 内存使用率百分比
    memory_used_mb: float              # 已使用内存(MB)
    memory_available_mb: float         # 可用内存(MB)
    disk_usage_percent: float          # 磁盘使用率百分比
    disk_read_mb: float                # 磁盘读取速度(MB/s)
    disk_write_mb: float               # 磁盘写入速度(MB/s)
    network_sent_mb: float             # 网络发送速度(MB/s)
    network_recv_mb: float             # 网络接收速度(MB/s)
    process_count: int                 # 进程数量
    thread_count: int                  # 线程数量
    gpu_usage: Optional[float] = None  # GPU使用率(如果可用)
    gpu_memory: Optional[float] = None # GPU内存使用率(如果可用)

@dataclass
class AnimationPerformance:
    """
    动画性能指标数据类
    
    专门用于跟踪Live2D动画相关的性能指标
    """
    animation_name: str                # 动画名称
    start_time: float                  # 动画开始时间
    end_time: Optional[float] = None   # 动画结束时间
    duration: Optional[float] = None   # 动画持续时间
    frame_count: int = 0               # 渲染帧数
    dropped_frames: int = 0            # 丢帧数量
    avg_fps: Optional[float] = None    # 平均帧率
    cpu_usage: Optional[float] = None  # 动画期间CPU使用率
    memory_usage: Optional[float] = None # 动画期间内存使用率

class PerformanceMonitor:
    """
    性能监控系统
    
    功能包括:
    1. 实时系统资源监控
    2. 动画性能分析
    3. 性能数据存储和历史记录
    4. 性能报告生成
    5. 异常检测和告警
    6. 资源使用优化建议
    """
    
    def __init__(self, 
                 monitor_interval: float = 1.0,
                 max_history_size: int = 3600,
                 enable_gpu_monitoring: bool = True):
        """
        初始化性能监控器
        
        Args:
            monitor_interval: 监控间隔时间(秒)
            max_history_size: 最大历史记录数量
            enable_gpu_monitoring: 是否启用GPU监控
            
        异常处理:
            - 确保监控系统在初始化失败时能够优雅降级
            - 记录详细的初始化错误信息
        """
        try:
            self.monitor_interval = monitor_interval
            self.max_history_size = max_history_size
            self.enable_gpu_monitoring = enable_gpu_monitoring
            
            # 线程安全锁
            self._lock = threading.RLock()
            
            # 监控状态
            self._monitoring = False
            self._monitor_thread: Optional[threading.Thread] = None
            
            # 性能数据存储
            self._metrics_history: deque = deque(maxlen=max_history_size)
            self._animation_history: List[AnimationPerformance] = []
            
            # 当前活跃的动画监控
            self._active_animations: Dict[str, AnimationPerformance] = {}
            
            # 统计信息
            self._stats = {
                'total_measurements': 0,
                'monitoring_start_time': None,
                'peak_cpu_usage': 0.0,
                'peak_memory_usage': 0.0,
                'total_animations': 0,
                'avg_animation_duration': 0.0
            }
            
            # 告警阈值
            self._alert_thresholds = {
                'cpu_percent': 80.0,      # CPU使用率告警阈值
                'memory_percent': 85.0,   # 内存使用率告警阈值
                'disk_usage': 90.0,       # 磁盘使用率告警阈值
                'low_fps': 30.0           # 低帧率告警阈值
            }
            
            # GPU监控初始化
            self._gpu_available = False
            if enable_gpu_monitoring:
                self._init_gpu_monitoring()
            
            # 初始化基准测量
            self._baseline_metrics = self._get_current_metrics()
            
            logger.info("性能监控系统初始化成功")
            
        except Exception as e:
            logger.error(f"性能监控系统初始化失败: {type(e).__name__}: {e}")
            # 设置默认值确保系统可用
            self._monitoring = False
            self._gpu_available = False
            self._metrics_history = deque(maxlen=100)
            self._animation_history = []
            self._active_animations = {}
    
    def _init_gpu_monitoring(self):
        """
        初始化GPU监控功能
        
        尝试导入GPU监控库并检测GPU可用性
        
        异常处理:
            - 如果GPU监控库不可用，优雅降级到CPU监控
            - 记录GPU监控状态用于后续判断
        """
        try:
            if GPU_AVAILABLE and GPUtil is not None:
                gpus = GPUtil.getGPUs()
                if gpus:
                    self._gpu_available = True
                    logger.info(f"检测到 {len(gpus)} 个GPU设备，GPU监控已启用")
                else:
                    logger.info("未检测到GPU设备，仅使用CPU监控")
            else:
                logger.info("GPUtil库未安装，GPU监控不可用")
        except Exception as e:
            logger.warning(f"GPU监控初始化失败: {e}")
    
    def start_monitoring(self) -> bool:
        """
        启动性能监控
        
        Returns:
            bool: 启动是否成功
            
        异常处理:
            - 确保监控线程启动失败时不影响主程序
            - 提供详细的错误诊断信息
        """
        try:
            with self._lock:
                if self._monitoring:
                    logger.warning("性能监控已在运行中")
                    return True
                
                self._monitoring = True
                self._stats['monitoring_start_time'] = time.time()
                
                # 创建并启动监控线程
                self._monitor_thread = threading.Thread(
                    target=self._monitoring_loop,
                    name="PerformanceMonitor",
                    daemon=True
                )
                self._monitor_thread.start()
                
                logger.info(f"性能监控已启动，监控间隔: {self.monitor_interval}秒")
                return True
                
        except Exception as e:
            logger.error(f"启动性能监控失败: {type(e).__name__}: {e}")
            self._monitoring = False
            return False
    
    def stop_monitoring(self) -> bool:
        """
        停止性能监控
        
        Returns:
            bool: 停止是否成功
            
        异常处理:
            - 确保监控线程能够安全停止
            - 处理线程停止超时的情况
        """
        try:
            with self._lock:
                if not self._monitoring:
                    logger.info("性能监控未在运行")
                    return True
                
                self._monitoring = False
                
                # 等待监控线程结束
                if self._monitor_thread and self._monitor_thread.is_alive():
                    self._monitor_thread.join(timeout=5.0)
                    
                    if self._monitor_thread.is_alive():
                        logger.warning("监控线程停止超时")
                        return False
                
                logger.info("性能监控已停止")
                return True
                
        except Exception as e:
            logger.error(f"停止性能监控失败: {type(e).__name__}: {e}")
            return False
    
    def _monitoring_loop(self):
        """
        监控循环主函数
        
        在独立线程中运行，定期收集性能指标
        
        异常处理:
            - 确保单次测量失败不会终止整个监控循环
            - 记录详细的错误信息用于调试
        """
        logger.info("性能监控循环已启动")
        
        while self._monitoring:
            try:
                # 获取当前性能指标
                metrics = self._get_current_metrics()
                
                if metrics:
                    with self._lock:
                        # 存储指标
                        self._metrics_history.append(metrics)
                        self._stats['total_measurements'] += 1
                        
                        # 更新峰值统计
                        self._stats['peak_cpu_usage'] = max(
                            self._stats['peak_cpu_usage'], 
                            metrics.cpu_percent
                        )
                        self._stats['peak_memory_usage'] = max(
                            self._stats['peak_memory_usage'], 
                            metrics.memory_percent
                        )
                        
                        # 检查告警条件
                        self._check_alerts(metrics)
                
                # 等待下一次监控
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"监控循环中发生异常: {type(e).__name__}: {e}")
                # 短暂等待后继续监控
                time.sleep(1.0)
        
        logger.info("性能监控循环已结束")
    
    def _get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """
        获取当前系统性能指标
        
        Returns:
            Optional[PerformanceMetrics]: 当前性能指标，获取失败时返回None
            
        异常处理:
            - 处理系统API调用可能出现的各种异常
            - 确保部分指标获取失败时仍能返回可用数据
        """
        try:
            current_time = time.time()
            
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # 内存信息
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            memory_available_mb = memory.available / (1024 * 1024)
            
            # 磁盘信息
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent
            
            # 磁盘IO
            disk_io = psutil.disk_io_counters()
            disk_read_mb = disk_io.read_bytes / (1024 * 1024) if disk_io else 0
            disk_write_mb = disk_io.write_bytes / (1024 * 1024) if disk_io else 0
            
            # 网络IO
            network_io = psutil.net_io_counters()
            network_sent_mb = network_io.bytes_sent / (1024 * 1024) if network_io else 0
            network_recv_mb = network_io.bytes_recv / (1024 * 1024) if network_io else 0
            
            # 进程和线程信息
            process_count = len(psutil.pids())
            
            # 当前进程的线程数
            current_process = psutil.Process()
            thread_count = current_process.num_threads()
            
            # GPU信息(如果可用)
            gpu_usage = None
            gpu_memory = None
            if self._gpu_available and GPU_AVAILABLE and GPUtil is not None:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # 使用第一个GPU
                        gpu_usage = gpu.load * 100
                        gpu_memory = gpu.memoryUtil * 100
                except Exception as gpu_error:
                    logger.debug(f"GPU监控获取失败: {gpu_error}")
            
            return PerformanceMetrics(
                timestamp=current_time,
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                memory_available_mb=memory_available_mb,
                disk_usage_percent=disk_usage_percent,
                disk_read_mb=disk_read_mb,
                disk_write_mb=disk_write_mb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                process_count=process_count,
                thread_count=thread_count,
                gpu_usage=gpu_usage,
                gpu_memory=gpu_memory
            )
            
        except Exception as e:
            logger.error(f"获取性能指标失败: {type(e).__name__}: {e}")
            return None
    
    def _check_alerts(self, metrics: PerformanceMetrics):
        """
        检查性能告警条件
        
        Args:
            metrics: 当前性能指标
            
        功能说明:
            - 检查各项性能指标是否超过预设阈值
            - 记录告警信息到日志
            - 为后续告警通知功能预留接口
        """
        try:
            # CPU使用率告警
            if metrics.cpu_percent > self._alert_thresholds['cpu_percent']:
                logger.warning(f"CPU使用率告警: {metrics.cpu_percent:.1f}% (阈值: {self._alert_thresholds['cpu_percent']}%)")
            
            # 内存使用率告警
            if metrics.memory_percent > self._alert_thresholds['memory_percent']:
                logger.warning(f"内存使用率告警: {metrics.memory_percent:.1f}% (阈值: {self._alert_thresholds['memory_percent']}%)")
            
            # 磁盘使用率告警
            if metrics.disk_usage_percent > self._alert_thresholds['disk_usage']:
                logger.warning(f"磁盘使用率告警: {metrics.disk_usage_percent:.1f}% (阈值: {self._alert_thresholds['disk_usage']}%)")
            
        except Exception as e:
            logger.error(f"检查告警条件时发生异常: {e}")
    
    def start_animation_monitoring(self, animation_name: str) -> str:
        """
        开始监控特定动画的性能
        
        Args:
            animation_name: 动画名称
            
        Returns:
            str: 动画监控ID
            
        异常处理:
            - 确保动画监控启动失败时不影响主监控系统
            - 生成唯一的监控ID避免冲突
        """
        try:
            with self._lock:
                # 生成唯一的监控ID
                monitor_id = f"{animation_name}_{int(time.time() * 1000)}"
                
                # 创建动画性能记录
                animation_perf = AnimationPerformance(
                    animation_name=animation_name,
                    start_time=time.time()
                )
                
                self._active_animations[monitor_id] = animation_perf
                
                logger.info(f"开始监控动画性能: {animation_name} (ID: {monitor_id})")
                return monitor_id
                
        except Exception as e:
            logger.error(f"启动动画监控失败: {type(e).__name__}: {e}")
            return ""
    
    def stop_animation_monitoring(self, monitor_id: str) -> Optional[AnimationPerformance]:
        """
        停止动画性能监控
        
        Args:
            monitor_id: 动画监控ID
            
        Returns:
            Optional[AnimationPerformance]: 动画性能数据
            
        异常处理:
            - 处理监控ID不存在的情况
            - 确保性能数据计算的准确性
        """
        try:
            with self._lock:
                if monitor_id not in self._active_animations:
                    logger.warning(f"动画监控ID不存在: {monitor_id}")
                    return None
                
                animation_perf = self._active_animations.pop(monitor_id)
                animation_perf.end_time = time.time()
                animation_perf.duration = animation_perf.end_time - animation_perf.start_time
                
                # 计算平均帧率
                if animation_perf.duration > 0 and animation_perf.frame_count > 0:
                    animation_perf.avg_fps = animation_perf.frame_count / animation_perf.duration
                
                # 存储到历史记录
                self._animation_history.append(animation_perf)
                self._stats['total_animations'] += 1
                
                # 更新平均动画持续时间
                total_duration = sum(anim.duration or 0 for anim in self._animation_history)
                self._stats['avg_animation_duration'] = total_duration / len(self._animation_history)
                
                logger.info(f"动画监控完成: {animation_perf.animation_name}, 持续时间: {animation_perf.duration:.2f}秒")
                return animation_perf
                
        except Exception as e:
            logger.error(f"停止动画监控失败: {type(e).__name__}: {e}")
            return None
    
    def update_animation_frame(self, monitor_id: str, dropped: bool = False):
        """
        更新动画帧计数
        
        Args:
            monitor_id: 动画监控ID
            dropped: 是否为丢帧
            
        异常处理:
            - 处理监控ID不存在的情况
            - 确保帧计数更新的线程安全性
        """
        try:
            with self._lock:
                if monitor_id in self._active_animations:
                    animation_perf = self._active_animations[monitor_id]
                    animation_perf.frame_count += 1
                    
                    if dropped:
                        animation_perf.dropped_frames += 1
                        
                        # 检查低帧率告警
                        if animation_perf.frame_count > 0:
                            current_duration = time.time() - animation_perf.start_time
                            if current_duration > 0:
                                current_fps = animation_perf.frame_count / current_duration
                                if current_fps < self._alert_thresholds['low_fps']:
                                    logger.warning(f"动画 {animation_perf.animation_name} 帧率过低: {current_fps:.1f} FPS")
                
        except Exception as e:
            logger.error(f"更新动画帧计数失败: {e}")
    
    def get_current_stats(self) -> Dict[str, Any]:
        """
        获取当前性能统计信息
        
        Returns:
            Dict[str, Any]: 性能统计数据
            
        异常处理:
            - 确保统计数据获取失败时返回默认值
            - 处理数据计算过程中的异常
        """
        try:
            with self._lock:
                stats = self._stats.copy()
                
                # 添加实时统计
                if self._metrics_history:
                    latest_metrics = self._metrics_history[-1]
                    stats.update({
                        'current_cpu_percent': latest_metrics.cpu_percent,
                        'current_memory_percent': latest_metrics.memory_percent,
                        'current_memory_used_mb': latest_metrics.memory_used_mb,
                        'current_disk_usage_percent': latest_metrics.disk_usage_percent,
                        'monitoring_duration': time.time() - (stats['monitoring_start_time'] or time.time()),
                        'active_animations': len(self._active_animations),
                        'history_size': len(self._metrics_history)
                    })
                    
                    if latest_metrics.gpu_usage is not None:
                        stats['current_gpu_usage'] = latest_metrics.gpu_usage
                        stats['current_gpu_memory'] = latest_metrics.gpu_memory
                
                return stats
                
        except Exception as e:
            logger.error(f"获取性能统计失败: {e}")
            return {'error': str(e)}
    
    def get_performance_report(self, 
                             time_range_minutes: int = 60) -> Dict[str, Any]:
        """
        生成性能报告
        
        Args:
            time_range_minutes: 报告时间范围(分钟)
            
        Returns:
            Dict[str, Any]: 详细的性能报告
            
        异常处理:
            - 处理数据不足的情况
            - 确保报告生成失败时返回基本信息
        """
        try:
            with self._lock:
                current_time = time.time()
                start_time = current_time - (time_range_minutes * 60)
                
                # 筛选时间范围内的数据
                filtered_metrics = [
                    m for m in self._metrics_history 
                    if m.timestamp >= start_time
                ]
                
                if not filtered_metrics:
                    return {
                        'error': '指定时间范围内没有性能数据',
                        'time_range_minutes': time_range_minutes,
                        'current_time': datetime.fromtimestamp(current_time).isoformat()
                    }
                
                # 计算统计指标
                cpu_values = [m.cpu_percent for m in filtered_metrics]
                memory_values = [m.memory_percent for m in filtered_metrics]
                
                report = {
                    'report_time': datetime.fromtimestamp(current_time).isoformat(),
                    'time_range_minutes': time_range_minutes,
                    'data_points': len(filtered_metrics),
                    
                    # CPU统计
                    'cpu_stats': {
                        'avg': sum(cpu_values) / len(cpu_values),
                        'min': min(cpu_values),
                        'max': max(cpu_values),
                        'current': cpu_values[-1] if cpu_values else 0
                    },
                    
                    # 内存统计
                    'memory_stats': {
                        'avg': sum(memory_values) / len(memory_values),
                        'min': min(memory_values),
                        'max': max(memory_values),
                        'current': memory_values[-1] if memory_values else 0
                    },
                    
                    # 动画统计
                    'animation_stats': {
                        'total_animations': len(self._animation_history),
                        'active_animations': len(self._active_animations),
                        'avg_duration': self._stats.get('avg_animation_duration', 0)
                    },
                    
                    # 系统建议
                    'recommendations': self._generate_recommendations(filtered_metrics)
                }
                
                return report
                
        except Exception as e:
            logger.error(f"生成性能报告失败: {e}")
            return {'error': f'报告生成失败: {str(e)}'}
    
    def _generate_recommendations(self, metrics: List[PerformanceMetrics]) -> List[str]:
        """
        基于性能数据生成优化建议
        
        Args:
            metrics: 性能指标列表
            
        Returns:
            List[str]: 优化建议列表
            
        功能说明:
            - 分析性能数据趋势
            - 识别性能瓶颈
            - 提供具体的优化建议
        """
        recommendations = []
        
        try:
            if not metrics:
                return recommendations
            
            # CPU使用率分析
            avg_cpu = sum(m.cpu_percent for m in metrics) / len(metrics)
            if avg_cpu > 70:
                recommendations.append("CPU使用率较高，建议优化计算密集型操作或考虑升级硬件")
            
            # 内存使用率分析
            avg_memory = sum(m.memory_percent for m in metrics) / len(metrics)
            if avg_memory > 80:
                recommendations.append("内存使用率较高，建议优化内存使用或增加内存容量")
            
            # 磁盘使用率分析
            if metrics:
                latest_disk = metrics[-1].disk_usage_percent
                if latest_disk > 85:
                    recommendations.append("磁盘空间不足，建议清理临时文件或扩展存储空间")
            
            # 动画性能分析
            recent_animations = [
                anim for anim in self._animation_history 
                if anim.end_time and anim.end_time >= (time.time() - 3600)
            ]
            
            if recent_animations:
                avg_fps = sum(anim.avg_fps or 0 for anim in recent_animations) / len(recent_animations)
                if avg_fps < 30:
                    recommendations.append("动画帧率较低，建议优化渲染性能或降低动画复杂度")
                
                total_dropped = sum(anim.dropped_frames for anim in recent_animations)
                if total_dropped > 0:
                    recommendations.append(f"检测到 {total_dropped} 次丢帧，建议检查系统负载")
            
            if not recommendations:
                recommendations.append("系统性能良好，无需特别优化")
            
        except Exception as e:
            logger.error(f"生成优化建议失败: {e}")
            recommendations.append("无法生成优化建议，请检查系统状态")
        
        return recommendations
    
    def export_data(self, 
                   file_path: str, 
                   format_type: str = 'json') -> bool:
        """
        导出性能数据
        
        Args:
            file_path: 导出文件路径
            format_type: 导出格式 ('json', 'csv')
            
        Returns:
            bool: 导出是否成功
            
        异常处理:
            - 处理文件写入权限问题
            - 确保数据序列化过程的安全性
        """
        try:
            with self._lock:
                export_data = {
                    'export_time': datetime.now().isoformat(),
                    'stats': self._stats,
                    'metrics_count': len(self._metrics_history),
                    'animations_count': len(self._animation_history),
                    'recent_metrics': [
                        {
                            'timestamp': m.timestamp,
                            'cpu_percent': m.cpu_percent,
                            'memory_percent': m.memory_percent,
                            'memory_used_mb': m.memory_used_mb,
                            'disk_usage_percent': m.disk_usage_percent,
                            'gpu_usage': m.gpu_usage,
                            'gpu_memory': m.gpu_memory
                        }
                        for m in list(self._metrics_history)[-100:]  # 最近100条记录
                    ],
                    'recent_animations': [
                        {
                            'animation_name': a.animation_name,
                            'duration': a.duration,
                            'frame_count': a.frame_count,
                            'dropped_frames': a.dropped_frames,
                            'avg_fps': a.avg_fps
                        }
                        for a in self._animation_history[-50:]  # 最近50个动画
                    ]
                }
                
                if format_type.lower() == 'json':
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(export_data, f, indent=2, ensure_ascii=False)
                else:
                    logger.error(f"不支持的导出格式: {format_type}")
                    return False
                
                logger.info(f"性能数据已导出到: {file_path}")
                return True
                
        except Exception as e:
            logger.error(f"导出性能数据失败: {type(e).__name__}: {e}")
            return False
    
    def cleanup_old_data(self, days: int = 7):
        """
        清理旧的性能数据
        
        Args:
            days: 保留天数
            
        异常处理:
            - 确保清理过程不影响当前监控
            - 记录清理结果用于审计
        """
        try:
            with self._lock:
                cutoff_time = time.time() - (days * 24 * 60 * 60)
                
                # 清理旧的动画记录
                original_count = len(self._animation_history)
                self._animation_history = [
                    anim for anim in self._animation_history
                    if anim.end_time and anim.end_time >= cutoff_time
                ]
                
                cleaned_count = original_count - len(self._animation_history)
                
                logger.info(f"清理了 {cleaned_count} 条旧动画记录，保留 {len(self._animation_history)} 条")
                
        except Exception as e:
            logger.error(f"清理旧数据失败: {e}")
    
    def __del__(self):
        """
        析构函数，确保监控线程正确停止
        """
        try:
            if hasattr(self, '_monitoring') and self._monitoring:
                self.stop_monitoring()
        except Exception:
            pass  # 析构函数中不应抛出异常

# 全局性能监控器实例
performance_monitor = PerformanceMonitor()

# 便捷装饰器
def monitor_performance(func):
    """
    性能监控装饰器
    
    用于监控函数执行的性能指标
    
    Args:
        func: 要监控的函数
        
    Returns:
        装饰后的函数
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_metrics = performance_monitor._get_current_metrics()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            end_metrics = performance_monitor._get_current_metrics()
            
            duration = end_time - start_time
            
            if start_metrics and end_metrics:
                cpu_diff = end_metrics.cpu_percent - start_metrics.cpu_percent
                memory_diff = end_metrics.memory_used_mb - start_metrics.memory_used_mb
                
                logger.info(
                    f"函数 {func.__name__} 执行完成 - "
                    f"耗时: {duration:.3f}s, "
                    f"CPU变化: {cpu_diff:+.1f}%, "
                    f"内存变化: {memory_diff:+.1f}MB"
                )
    
    return wrapper