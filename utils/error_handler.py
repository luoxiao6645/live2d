# -*- coding: utf-8 -*-
"""
全面的错误处理和恢复系统
提供统一的异常处理、日志记录、健康检查和自动恢复功能
"""

import os
import sys
import traceback
import logging
import threading
import time
import json
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, List, Tuple
from functools import wraps
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

class ErrorSeverity(Enum):
    """
    错误严重程度枚举
    """
    LOW = "low"          # 轻微错误，不影响核心功能
    MEDIUM = "medium"    # 中等错误，影响部分功能
    HIGH = "high"        # 严重错误，影响核心功能
    CRITICAL = "critical" # 致命错误，系统无法正常运行

class ErrorCategory(Enum):
    """
    错误类别枚举
    """
    NETWORK = "network"           # 网络相关错误
    DATABASE = "database"         # 数据库相关错误
    FILE_IO = "file_io"          # 文件IO错误
    VALIDATION = "validation"     # 数据验证错误
    AUTHENTICATION = "auth"       # 认证授权错误
    PERMISSION = "permission"     # 权限错误
    RESOURCE = "resource"         # 资源不足错误
    EXTERNAL_API = "external_api" # 外部API错误
    INTERNAL = "internal"         # 内部逻辑错误
    UNKNOWN = "unknown"           # 未知错误

@dataclass
class ErrorInfo:
    """
    错误信息数据类
    """
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    details: str
    function_name: str
    file_name: str
    line_number: int
    stack_trace: str
    context: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_details: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        Returns:
            Dict[str, Any]: 字典格式的错误信息
        """
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['severity'] = self.severity.value
        data['category'] = self.category.value
        return data

class SystemHealthMonitor:
    """
    系统健康监控器
    
    功能：
    1. 监控系统资源使用情况
    2. 检测异常状态
    3. 提供健康评分
    4. 触发自动恢复机制
    """
    
    def __init__(self, check_interval: int = 60):
        """
        初始化健康监控器
        
        Args:
            check_interval: 检查间隔（秒）
        """
        self.check_interval = check_interval
        self.is_monitoring = False
        self.health_history = deque(maxlen=100)  # 保留最近100次检查结果
        self.thresholds = {
            'cpu_percent': 80.0,      # CPU使用率阈值
            'memory_percent': 85.0,   # 内存使用率阈值
            'disk_percent': 90.0,     # 磁盘使用率阈值
            'error_rate': 0.1,        # 错误率阈值（10%）
            'response_time': 5.0      # 响应时间阈值（秒）
        }
        self.alert_callbacks = []  # 告警回调函数列表
        self.lock = threading.RLock()
        
        # 启动监控线程
        self._start_monitoring()
    
    def _start_monitoring(self):
        """
        启动监控线程
        """
        def monitor_task():
            self.is_monitoring = True
            while self.is_monitoring:
                try:
                    health_data = self._collect_health_data()
                    self._analyze_health(health_data)
                    
                    with self.lock:
                        self.health_history.append(health_data)
                    
                    time.sleep(self.check_interval)
                    
                except Exception as e:
                    logging.error(f"健康监控任务发生错误: {e}")
                    time.sleep(30)  # 发生错误时等待30秒再重试
        
        monitor_thread = threading.Thread(target=monitor_task, daemon=True)
        monitor_thread.start()
        logging.info("系统健康监控已启动")
    
    def _collect_health_data(self) -> Dict[str, Any]:
        """
        收集系统健康数据
        
        Returns:
            Dict[str, Any]: 健康数据
        """
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存使用情况
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # 磁盘使用情况
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # 网络IO
            net_io = psutil.net_io_counters()
            
            # 进程信息
            process = psutil.Process()
            process_memory = process.memory_info()
            
            return {
                'timestamp': datetime.now(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_available': memory.available,
                'memory_used': memory.used,
                'disk_percent': disk_percent,
                'disk_free': disk.free,
                'network_bytes_sent': net_io.bytes_sent,
                'network_bytes_recv': net_io.bytes_recv,
                'process_memory_rss': process_memory.rss,
                'process_memory_vms': process_memory.vms,
                'process_cpu_percent': process.cpu_percent(),
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            }
            
        except Exception as e:
            logging.error(f"收集健康数据失败: {e}")
            return {
                'timestamp': datetime.now(),
                'error': str(e)
            }
    
    def _analyze_health(self, health_data: Dict[str, Any]):
        """
        分析健康数据并触发告警
        
        Args:
            health_data: 健康数据
        """
        try:
            alerts = []
            
            # 检查CPU使用率
            if health_data.get('cpu_percent', 0) > self.thresholds['cpu_percent']:
                alerts.append({
                    'type': 'high_cpu',
                    'severity': ErrorSeverity.HIGH,
                    'message': f"CPU使用率过高: {health_data['cpu_percent']:.1f}%",
                    'value': health_data['cpu_percent'],
                    'threshold': self.thresholds['cpu_percent']
                })
            
            # 检查内存使用率
            if health_data.get('memory_percent', 0) > self.thresholds['memory_percent']:
                alerts.append({
                    'type': 'high_memory',
                    'severity': ErrorSeverity.HIGH,
                    'message': f"内存使用率过高: {health_data['memory_percent']:.1f}%",
                    'value': health_data['memory_percent'],
                    'threshold': self.thresholds['memory_percent']
                })
            
            # 检查磁盘使用率
            if health_data.get('disk_percent', 0) > self.thresholds['disk_percent']:
                alerts.append({
                    'type': 'high_disk',
                    'severity': ErrorSeverity.CRITICAL,
                    'message': f"磁盘使用率过高: {health_data['disk_percent']:.1f}%",
                    'value': health_data['disk_percent'],
                    'threshold': self.thresholds['disk_percent']
                })
            
            # 触发告警回调
            for alert in alerts:
                self._trigger_alert(alert)
                
        except Exception as e:
            logging.error(f"分析健康数据失败: {e}")
    
    def _trigger_alert(self, alert: Dict[str, Any]):
        """
        触发告警
        
        Args:
            alert: 告警信息
        """
        try:
            logging.warning(f"系统告警: {alert['message']}")
            
            # 调用注册的告警回调函数
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logging.error(f"告警回调函数执行失败: {e}")
                    
        except Exception as e:
            logging.error(f"触发告警失败: {e}")
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        添加告警回调函数
        
        Args:
            callback: 回调函数
        """
        self.alert_callbacks.append(callback)
    
    def get_health_score(self) -> float:
        """
        计算系统健康评分（0-100）
        
        Returns:
            float: 健康评分
        """
        try:
            with self.lock:
                if not self.health_history:
                    return 100.0
                
                latest_data = self.health_history[-1]
                
                # 计算各项指标的得分
                cpu_score = max(0, 100 - latest_data.get('cpu_percent', 0))
                memory_score = max(0, 100 - latest_data.get('memory_percent', 0))
                disk_score = max(0, 100 - latest_data.get('disk_percent', 0))
                
                # 加权平均
                total_score = (cpu_score * 0.3 + memory_score * 0.4 + disk_score * 0.3)
                
                return min(100.0, max(0.0, total_score))
                
        except Exception as e:
            logging.error(f"计算健康评分失败: {e}")
            return 50.0  # 默认中等健康状态
    
    def get_health_summary(self) -> Dict[str, Any]:
        """
        获取健康状态摘要
        
        Returns:
            Dict[str, Any]: 健康状态摘要
        """
        try:
            with self.lock:
                if not self.health_history:
                    return {'status': 'no_data'}
                
                latest_data = self.health_history[-1]
                health_score = self.get_health_score()
                
                # 确定健康状态
                if health_score >= 90:
                    status = 'excellent'
                elif health_score >= 75:
                    status = 'good'
                elif health_score >= 50:
                    status = 'fair'
                elif health_score >= 25:
                    status = 'poor'
                else:
                    status = 'critical'
                
                return {
                    'status': status,
                    'score': health_score,
                    'timestamp': latest_data.get('timestamp', datetime.now()).isoformat(),
                    'cpu_percent': latest_data.get('cpu_percent', 0),
                    'memory_percent': latest_data.get('memory_percent', 0),
                    'disk_percent': latest_data.get('disk_percent', 0),
                    'checks_count': len(self.health_history)
                }
                
        except Exception as e:
            logging.error(f"获取健康状态摘要失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def stop_monitoring(self):
        """
        停止监控
        """
        self.is_monitoring = False
        logging.info("系统健康监控已停止")

class ErrorRecoveryManager:
    """
    错误恢复管理器
    
    功能：
    1. 自动错误恢复
    2. 恢复策略管理
    3. 恢复历史记录
    4. 恢复效果评估
    """
    
    def __init__(self):
        """
        初始化错误恢复管理器
        """
        self.recovery_strategies = {}  # 恢复策略映射
        self.recovery_history = deque(maxlen=1000)  # 恢复历史记录
        self.lock = threading.RLock()
        
        # 注册默认恢复策略
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """
        注册默认恢复策略
        """
        # 网络错误恢复策略
        self.register_strategy(
            ErrorCategory.NETWORK,
            self._network_recovery_strategy,
            "网络连接重试策略"
        )
        
        # 文件IO错误恢复策略
        self.register_strategy(
            ErrorCategory.FILE_IO,
            self._file_io_recovery_strategy,
            "文件IO重试策略"
        )
        
        # 资源不足错误恢复策略
        self.register_strategy(
            ErrorCategory.RESOURCE,
            self._resource_recovery_strategy,
            "资源清理策略"
        )
        
        # 外部API错误恢复策略
        self.register_strategy(
            ErrorCategory.EXTERNAL_API,
            self._external_api_recovery_strategy,
            "外部API重试策略"
        )
    
    def register_strategy(
        self,
        category: ErrorCategory,
        strategy_func: Callable[[ErrorInfo], Tuple[bool, str]],
        description: str
    ):
        """
        注册恢复策略
        
        Args:
            category: 错误类别
            strategy_func: 恢复策略函数
            description: 策略描述
        """
        with self.lock:
            self.recovery_strategies[category] = {
                'function': strategy_func,
                'description': description,
                'usage_count': 0,
                'success_count': 0
            }
        
        logging.info(f"注册恢复策略: {category.value} - {description}")
    
    def attempt_recovery(self, error_info: ErrorInfo) -> Tuple[bool, str]:
        """
        尝试错误恢复
        
        Args:
            error_info: 错误信息
            
        Returns:
            Tuple[bool, str]: (是否成功, 恢复详情)
        """
        try:
            with self.lock:
                strategy = self.recovery_strategies.get(error_info.category)
                
                if not strategy:
                    return False, f"未找到 {error_info.category.value} 类别的恢复策略"
                
                # 更新使用计数
                strategy['usage_count'] += 1
                
                # 执行恢复策略
                success, details = strategy['function'](error_info)
                
                if success:
                    strategy['success_count'] += 1
                
                # 记录恢复历史
                recovery_record = {
                    'timestamp': datetime.now(),
                    'error_id': error_info.error_id,
                    'category': error_info.category.value,
                    'strategy': strategy['description'],
                    'success': success,
                    'details': details
                }
                
                self.recovery_history.append(recovery_record)
                
                # 更新错误信息
                error_info.recovery_attempted = True
                error_info.recovery_successful = success
                error_info.recovery_details = details
                
                logging.info(
                    f"恢复尝试 - 错误ID: {error_info.error_id}, "
                    f"策略: {strategy['description']}, "
                    f"结果: {'成功' if success else '失败'}, "
                    f"详情: {details}"
                )
                
                return success, details
                
        except Exception as e:
            error_msg = f"执行恢复策略时发生错误: {e}"
            logging.error(error_msg)
            return False, error_msg
    
    def _network_recovery_strategy(self, error_info: ErrorInfo) -> Tuple[bool, str]:
        """
        网络错误恢复策略
        
        Args:
            error_info: 错误信息
            
        Returns:
            Tuple[bool, str]: (是否成功, 恢复详情)
        """
        try:
            import requests
            
            # 简单的网络连通性测试
            test_urls = [
                'http://127.0.0.1:11434/api/tags',  # Ollama本地服务
                'https://www.baidu.com',             # 外网连通性
            ]
            
            for url in test_urls:
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        return True, f"网络连接已恢复，测试URL: {url}"
                except:
                    continue
            
            return False, "网络连接仍然不可用"
            
        except Exception as e:
            return False, f"网络恢复策略执行失败: {e}"
    
    def _file_io_recovery_strategy(self, error_info: ErrorInfo) -> Tuple[bool, str]:
        """
        文件IO错误恢复策略
        
        Args:
            error_info: 错误信息
            
        Returns:
            Tuple[bool, str]: (是否成功, 恢复详情)
        """
        try:
            # 检查磁盘空间
            disk_usage = psutil.disk_usage('/')
            free_space_gb = disk_usage.free / (1024**3)
            
            if free_space_gb < 1.0:  # 少于1GB空间
                return False, f"磁盘空间不足: {free_space_gb:.2f}GB"
            
            # 检查文件权限（如果错误信息中包含文件路径）
            file_path = error_info.context.get('file_path')
            if file_path and os.path.exists(file_path):
                if os.access(file_path, os.R_OK | os.W_OK):
                    return True, f"文件访问权限正常: {file_path}"
                else:
                    return False, f"文件访问权限不足: {file_path}"
            
            return True, "文件IO环境检查通过"
            
        except Exception as e:
            return False, f"文件IO恢复策略执行失败: {e}"
    
    def _resource_recovery_strategy(self, error_info: ErrorInfo) -> Tuple[bool, str]:
        """
        资源不足错误恢复策略
        
        Args:
            error_info: 错误信息
            
        Returns:
            Tuple[bool, str]: (是否成功, 恢复详情)
        """
        try:
            import gc
            
            # 强制垃圾回收
            collected = gc.collect()
            
            # 检查内存使用情况
            memory = psutil.virtual_memory()
            
            if memory.percent < 85:  # 内存使用率低于85%
                return True, f"内存清理成功，回收对象: {collected}个，当前内存使用率: {memory.percent:.1f}%"
            else:
                return False, f"内存使用率仍然过高: {memory.percent:.1f}%"
                
        except Exception as e:
            return False, f"资源恢复策略执行失败: {e}"
    
    def _external_api_recovery_strategy(self, error_info: ErrorInfo) -> Tuple[bool, str]:
        """
        外部API错误恢复策略
        
        Args:
            error_info: 错误信息
            
        Returns:
            Tuple[bool, str]: (是否成功, 恢复详情)
        """
        try:
            # 等待一段时间后重试
            time.sleep(2)
            
            # 检查API服务状态
            api_url = error_info.context.get('api_url', 'http://127.0.0.1:11434')
            
            try:
                import requests
                response = requests.get(f"{api_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    return True, f"API服务已恢复: {api_url}"
                else:
                    return False, f"API服务返回错误状态码: {response.status_code}"
            except requests.exceptions.RequestException as e:
                return False, f"API服务仍然不可用: {e}"
                
        except Exception as e:
            return False, f"外部API恢复策略执行失败: {e}"
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """
        获取恢复统计信息
        
        Returns:
            Dict[str, Any]: 恢复统计信息
        """
        try:
            with self.lock:
                stats = {
                    'total_strategies': len(self.recovery_strategies),
                    'total_attempts': len(self.recovery_history),
                    'strategies': {},
                    'recent_attempts': []
                }
                
                # 策略统计
                for category, strategy in self.recovery_strategies.items():
                    success_rate = 0
                    if strategy['usage_count'] > 0:
                        success_rate = strategy['success_count'] / strategy['usage_count']
                    
                    stats['strategies'][category.value] = {
                        'description': strategy['description'],
                        'usage_count': strategy['usage_count'],
                        'success_count': strategy['success_count'],
                        'success_rate': success_rate
                    }
                
                # 最近的恢复尝试
                recent_count = min(10, len(self.recovery_history))
                for i in range(recent_count):
                    record = self.recovery_history[-(i+1)]
                    stats['recent_attempts'].append({
                        'timestamp': record['timestamp'].isoformat(),
                        'category': record['category'],
                        'strategy': record['strategy'],
                        'success': record['success'],
                        'details': record['details']
                    })
                
                return stats
                
        except Exception as e:
            logging.error(f"获取恢复统计信息失败: {e}")
            return {'error': str(e)}

class AdvancedErrorHandler:
    """
    高级错误处理器
    
    功能：
    1. 统一错误捕获和处理
    2. 错误分类和严重程度评估
    3. 自动恢复尝试
    4. 错误统计和分析
    5. 告警和通知
    """
    
    def __init__(self, enable_recovery: bool = True, enable_monitoring: bool = True):
        """
        初始化高级错误处理器
        
        Args:
            enable_recovery: 是否启用自动恢复
            enable_monitoring: 是否启用系统监控
        """
        self.error_history = deque(maxlen=10000)  # 保留最近10000个错误
        self.error_stats = defaultdict(int)       # 错误统计
        self.lock = threading.RLock()
        
        # 初始化子模块
        self.recovery_manager = ErrorRecoveryManager() if enable_recovery else None
        self.health_monitor = SystemHealthMonitor() if enable_monitoring else None
        
        # 错误处理配置
        self.config = {
            'auto_recovery_enabled': enable_recovery,
            'max_recovery_attempts': 3,
            'recovery_cooldown': 300,  # 恢复冷却时间（秒）
            'alert_threshold': 10,     # 告警阈值（每分钟错误数）
            'log_all_errors': True,
            'include_stack_trace': True
        }
        
        # 错误分类规则
        self.classification_rules = {
            'ConnectionError': (ErrorCategory.NETWORK, ErrorSeverity.HIGH),
            'TimeoutError': (ErrorCategory.NETWORK, ErrorSeverity.MEDIUM),
            'FileNotFoundError': (ErrorCategory.FILE_IO, ErrorSeverity.MEDIUM),
            'PermissionError': (ErrorCategory.PERMISSION, ErrorSeverity.HIGH),
            'MemoryError': (ErrorCategory.RESOURCE, ErrorSeverity.CRITICAL),
            'ValueError': (ErrorCategory.VALIDATION, ErrorSeverity.LOW),
            'KeyError': (ErrorCategory.VALIDATION, ErrorSeverity.LOW),
            'TypeError': (ErrorCategory.INTERNAL, ErrorSeverity.MEDIUM),
            'AttributeError': (ErrorCategory.INTERNAL, ErrorSeverity.MEDIUM),
            'ImportError': (ErrorCategory.INTERNAL, ErrorSeverity.HIGH),
            'ModuleNotFoundError': (ErrorCategory.INTERNAL, ErrorSeverity.HIGH)
        }
        
        logging.info("高级错误处理器初始化完成")
    
    def handle_error(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
        function_name: Optional[str] = None,
        auto_recover: bool = True
    ) -> ErrorInfo:
        """
        处理错误
        
        Args:
            exception: 异常对象
            context: 错误上下文信息
            function_name: 函数名称
            auto_recover: 是否尝试自动恢复
            
        Returns:
            ErrorInfo: 错误信息对象
        """
        try:
            # 生成错误ID
            error_id = f"ERR_{int(datetime.now().timestamp() * 1000)}_{id(exception)}"
            
            # 获取调用栈信息
            tb = traceback.extract_tb(exception.__traceback__)
            if tb:
                frame = tb[-1]
                file_name = os.path.basename(frame.filename)
                line_number = frame.lineno
                if not function_name:
                    function_name = frame.name
            else:
                file_name = "unknown"
                line_number = 0
                function_name = function_name or "unknown"
            
            # 分类错误
            category, severity = self._classify_error(exception)
            
            # 创建错误信息对象
            error_info = ErrorInfo(
                error_id=error_id,
                timestamp=datetime.now(),
                severity=severity,
                category=category,
                message=str(exception),
                details=f"{type(exception).__name__}: {str(exception)}",
                function_name=function_name,
                file_name=file_name,
                line_number=line_number,
                stack_trace=traceback.format_exc(),
                context=context or {}
            )
            
            # 记录错误
            self._record_error(error_info)
            
            # 尝试自动恢复
            if auto_recover and self.recovery_manager and self.config['auto_recovery_enabled']:
                if self._should_attempt_recovery(error_info):
                    success, details = self.recovery_manager.attempt_recovery(error_info)
                    if success:
                        logging.info(f"错误自动恢复成功: {error_id}")
                    else:
                        logging.warning(f"错误自动恢复失败: {error_id} - {details}")
            
            # 检查是否需要告警
            self._check_alert_conditions(error_info)
            
            return error_info
            
        except Exception as e:
            # 错误处理器本身发生错误时的兜底处理
            logging.critical(f"错误处理器发生严重错误: {e}")
            return ErrorInfo(
                error_id="ERR_HANDLER_FAILURE",
                timestamp=datetime.now(),
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.INTERNAL,
                message="错误处理器故障",
                details=str(e),
                function_name="handle_error",
                file_name="error_handler.py",
                line_number=0,
                stack_trace=traceback.format_exc(),
                context={'original_exception': str(exception)}
            )
    
    def _classify_error(self, exception: Exception) -> Tuple[ErrorCategory, ErrorSeverity]:
        """
        分类错误
        
        Args:
            exception: 异常对象
            
        Returns:
            Tuple[ErrorCategory, ErrorSeverity]: 错误类别和严重程度
        """
        exception_name = type(exception).__name__
        
        # 检查预定义规则
        if exception_name in self.classification_rules:
            return self.classification_rules[exception_name]
        
        # 基于异常消息的启发式分类
        message = str(exception).lower()
        
        if any(keyword in message for keyword in ['connection', 'network', 'timeout', 'unreachable']):
            return ErrorCategory.NETWORK, ErrorSeverity.HIGH
        elif any(keyword in message for keyword in ['permission', 'access', 'denied', 'forbidden']):
            return ErrorCategory.PERMISSION, ErrorSeverity.HIGH
        elif any(keyword in message for keyword in ['file', 'directory', 'path', 'not found']):
            return ErrorCategory.FILE_IO, ErrorSeverity.MEDIUM
        elif any(keyword in message for keyword in ['memory', 'resource', 'limit']):
            return ErrorCategory.RESOURCE, ErrorSeverity.HIGH
        elif any(keyword in message for keyword in ['api', 'service', 'endpoint']):
            return ErrorCategory.EXTERNAL_API, ErrorSeverity.MEDIUM
        elif any(keyword in message for keyword in ['validation', 'invalid', 'format']):
            return ErrorCategory.VALIDATION, ErrorSeverity.LOW
        else:
            return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM
    
    def _record_error(self, error_info: ErrorInfo):
        """
        记录错误信息
        
        Args:
            error_info: 错误信息
        """
        try:
            with self.lock:
                # 添加到历史记录
                self.error_history.append(error_info)
                
                # 更新统计信息
                self.error_stats['total_errors'] += 1
                self.error_stats[f'category_{error_info.category.value}'] += 1
                self.error_stats[f'severity_{error_info.severity.value}'] += 1
                
                # 记录日志
                if self.config['log_all_errors']:
                    log_level = self._get_log_level(error_info.severity)
                    log_message = (
                        f"错误处理 - ID: {error_info.error_id}, "
                        f"类别: {error_info.category.value}, "
                        f"严重程度: {error_info.severity.value}, "
                        f"函数: {error_info.function_name}, "
                        f"消息: {error_info.message}"
                    )
                    
                    if self.config['include_stack_trace'] and error_info.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                        log_message += f"\n堆栈跟踪:\n{error_info.stack_trace}"
                    
                    logging.log(log_level, log_message)
                    
        except Exception as e:
            logging.error(f"记录错误信息失败: {e}")
    
    def _get_log_level(self, severity: ErrorSeverity) -> int:
        """
        根据错误严重程度获取日志级别
        
        Args:
            severity: 错误严重程度
            
        Returns:
            int: 日志级别
        """
        level_map = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }
        return level_map.get(severity, logging.ERROR)
    
    def _should_attempt_recovery(self, error_info: ErrorInfo) -> bool:
        """
        判断是否应该尝试恢复
        
        Args:
            error_info: 错误信息
            
        Returns:
            bool: 是否应该尝试恢复
        """
        try:
            # 检查恢复冷却时间
            now = datetime.now()
            recent_attempts = [
                err for err in self.error_history
                if (err.category == error_info.category and
                    err.recovery_attempted and
                    (now - err.timestamp).total_seconds() < self.config['recovery_cooldown'])
            ]
            
            if len(recent_attempts) >= self.config['max_recovery_attempts']:
                logging.info(f"跳过恢复尝试，已达到最大尝试次数: {error_info.category.value}")
                return False
            
            # 只对特定类别的错误尝试恢复
            recoverable_categories = [
                ErrorCategory.NETWORK,
                ErrorCategory.FILE_IO,
                ErrorCategory.RESOURCE,
                ErrorCategory.EXTERNAL_API
            ]
            
            return error_info.category in recoverable_categories
            
        except Exception as e:
            logging.error(f"判断恢复条件失败: {e}")
            return False
    
    def _check_alert_conditions(self, error_info: ErrorInfo):
        """
        检查告警条件
        
        Args:
            error_info: 错误信息
        """
        try:
            # 检查错误频率
            now = datetime.now()
            recent_errors = [
                err for err in self.error_history
                if (now - err.timestamp).total_seconds() < 60  # 最近1分钟
            ]
            
            if len(recent_errors) >= self.config['alert_threshold']:
                logging.critical(
                    f"错误频率告警: 最近1分钟内发生 {len(recent_errors)} 个错误，"
                    f"超过阈值 {self.config['alert_threshold']}"
                )
            
            # 检查严重错误
            if error_info.severity == ErrorSeverity.CRITICAL:
                logging.critical(
                    f"严重错误告警: {error_info.error_id} - {error_info.message}"
                )
                
        except Exception as e:
            logging.error(f"检查告警条件失败: {e}")
    
    def get_error_stats(self) -> Dict[str, Any]:
        """
        获取错误统计信息
        
        Returns:
            Dict[str, Any]: 错误统计信息
        """
        try:
            with self.lock:
                stats = dict(self.error_stats)
                
                # 添加时间范围统计
                now = datetime.now()
                time_ranges = {
                    'last_hour': 3600,
                    'last_day': 86400,
                    'last_week': 604800
                }
                
                for range_name, seconds in time_ranges.items():
                    range_errors = [
                        err for err in self.error_history
                        if (now - err.timestamp).total_seconds() < seconds
                    ]
                    stats[f'errors_{range_name}'] = len(range_errors)
                
                # 添加恢复统计
                if self.recovery_manager:
                    recovery_stats = self.recovery_manager.get_recovery_stats()
                    stats['recovery'] = recovery_stats
                
                # 添加健康状态
                if self.health_monitor:
                    health_summary = self.health_monitor.get_health_summary()
                    stats['health'] = health_summary
                
                return stats
                
        except Exception as e:
            logging.error(f"获取错误统计信息失败: {e}")
            return {'error': str(e)}
    
    def get_recent_errors(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        获取最近的错误记录
        
        Args:
            count: 返回的错误数量
            
        Returns:
            List[Dict[str, Any]]: 最近的错误记录列表
        """
        try:
            with self.lock:
                recent_count = min(count, len(self.error_history))
                recent_errors = []
                
                for i in range(recent_count):
                    error_info = self.error_history[-(i+1)]
                    recent_errors.append(error_info.to_dict())
                
                return recent_errors
                
        except Exception as e:
            logging.error(f"获取最近错误记录失败: {e}")
            return []
    
    def create_error_decorator(self, auto_recover: bool = True):
        """
        创建错误处理装饰器
        
        Args:
            auto_recover: 是否自动恢复
            
        Returns:
            装饰器函数
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # 处理错误
                    error_info = self.handle_error(
                        exception=e,
                        context={
                            'function': func.__name__,
                            'args': str(args)[:200],  # 限制长度
                            'kwargs': str(kwargs)[:200]
                        },
                        function_name=func.__name__,
                        auto_recover=auto_recover
                    )
                    
                    # 根据错误严重程度决定是否重新抛出异常
                    if error_info.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                        raise
                    else:
                        # 对于轻微错误，返回None或默认值
                        return None
            
            return wrapper
        return decorator

# ======== 全局错误处理器实例 ========
# 创建全局错误处理器实例
global_error_handler = AdvancedErrorHandler(
    enable_recovery=True,
    enable_monitoring=True
)

# 创建错误处理装饰器
handle_errors = global_error_handler.create_error_decorator(auto_recover=True)
handle_errors_no_recovery = global_error_handler.create_error_decorator(auto_recover=False)

# ======== 便捷函数 ========
def log_error(exception: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorInfo:
    """
    记录错误的便捷函数
    
    Args:
        exception: 异常对象
        context: 错误上下文
        
    Returns:
        ErrorInfo: 错误信息
    """
    return global_error_handler.handle_error(exception, context)

def get_system_health() -> Dict[str, Any]:
    """
    获取系统健康状态的便捷函数
    
    Returns:
        Dict[str, Any]: 系统健康状态
    """
    if global_error_handler.health_monitor:
        return global_error_handler.health_monitor.get_health_summary()
    else:
        return {'status': 'monitoring_disabled'}

def get_error_summary() -> Dict[str, Any]:
    """
    获取错误摘要的便捷函数
    
    Returns:
        Dict[str, Any]: 错误摘要
    """
    return global_error_handler.get_error_stats()

if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 测试错误处理
    try:
        raise ValueError("测试错误")
    except Exception as e:
        error_info = log_error(e, {'test': True})
        print(f"错误ID: {error_info.error_id}")
    
    # 获取系统状态
    health = get_system_health()
    print(f"系统健康状态: {health}")
    
    # 获取错误统计
    stats = get_error_summary()
    print(f"错误统计: {stats}")