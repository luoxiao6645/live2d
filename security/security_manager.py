# -*- coding: utf-8 -*-
"""
安全管理器模块
提供API安全防护、输入验证、请求限制等功能
"""

import os
import re
import time
import hashlib
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from functools import wraps
from collections import defaultdict, deque
from flask import request, jsonify, g
import bleach
from werkzeug.utils import secure_filename

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SecurityConfig:
    """
    安全配置类
    """
    # 请求限制配置
    rate_limit_requests: int = 100  # 每分钟最大请求数
    rate_limit_window: int = 60     # 时间窗口（秒）
    burst_limit: int = 10           # 突发请求限制
    
    # 文件上传配置
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_extensions: List[str] = field(default_factory=lambda: [
        'txt', 'pdf', 'doc', 'docx', 'png', 'jpg', 'jpeg', 'gif', 'bmp'
    ])
    upload_path: str = "uploads"
    
    # 输入验证配置
    max_text_length: int = 10000
    max_query_length: int = 1000
    max_session_id_length: int = 64
    
    # IP白名单和黑名单
    ip_whitelist: List[str] = field(default_factory=list)
    ip_blacklist: List[str] = field(default_factory=list)
    
    # API密钥配置
    require_api_key: bool = False
    api_keys: List[str] = field(default_factory=list)
    
    # CORS配置
    allowed_origins: List[str] = field(default_factory=lambda: ['*'])
    allowed_methods: List[str] = field(default_factory=lambda: ['GET', 'POST', 'PUT', 'DELETE'])
    
    # 安全头配置
    enable_security_headers: bool = True
    
@dataclass
class RateLimitInfo:
    """
    请求限制信息
    """
    requests: deque = field(default_factory=deque)
    blocked_until: float = 0.0
    total_requests: int = 0
    blocked_requests: int = 0

class SecurityManager:
    """
    安全管理器
    
    功能包括：
    1. 请求频率限制
    2. 输入验证和清理
    3. 文件上传安全检查
    4. IP访问控制
    5. API密钥验证
    6. 安全头设置
    7. 恶意请求检测
    """
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        """
        初始化安全管理器
        
        Args:
            config: 安全配置对象
        """
        self.config = config or SecurityConfig()
        
        # 线程安全锁
        self._lock = threading.RLock()
        
        # 请求限制跟踪
        self._rate_limits: Dict[str, RateLimitInfo] = defaultdict(RateLimitInfo)
        
        # 恶意请求检测
        self._suspicious_ips: Dict[str, int] = defaultdict(int)
        self._blocked_ips: Dict[str, float] = {}  # IP -> 解封时间
        
        # 统计信息
        self._stats = {
            'total_requests': 0,
            'blocked_requests': 0,
            'malicious_requests': 0,
            'file_uploads': 0,
            'blocked_files': 0
        }
        
        # 恶意模式检测
        self._malicious_patterns = [
            r'<script[^>]*>.*?</script>',  # XSS
            r'javascript:',                # JavaScript协议
            r'on\w+\s*=',                 # 事件处理器
            r'\b(union|select|insert|update|delete|drop|create|alter)\b',  # SQL注入
            r'\.\.[\\/]',               # 路径遍历
            r'\$\{.*?\}',                 # 模板注入
            r'<%.*?%>',                   # 服务器端模板
        ]
        
        # 编译正则表达式
        self._compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self._malicious_patterns]
        
        # 确保上传目录存在
        self._ensure_upload_directory()
        
        logger.info("安全管理器初始化完成")
    
    def _ensure_upload_directory(self):
        """
        确保上传目录存在且安全
        """
        try:
            upload_dir = self.config.upload_path
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir, mode=0o755)
                logger.info(f"创建上传目录: {upload_dir}")
            
            # 创建.htaccess文件防止直接访问
            htaccess_path = os.path.join(upload_dir, '.htaccess')
            if not os.path.exists(htaccess_path):
                with open(htaccess_path, 'w') as f:
                    f.write("deny from all\n")
                    
        except Exception as e:
            logger.error(f"创建上传目录时发生错误: {e}")
    
    def rate_limit_decorator(self, requests_per_minute: Optional[int] = None):
        """
        请求频率限制装饰器
        
        Args:
            requests_per_minute: 每分钟允许的请求数
            
        Returns:
            装饰器函数
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self._check_rate_limit(requests_per_minute):
                    return jsonify({
                        "error": "请求过于频繁，请稍后再试",
                        "code": "RATE_LIMIT_EXCEEDED"
                    }), 429
                
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def _check_rate_limit(self, custom_limit: Optional[int] = None) -> bool:
        """
        检查请求频率限制
        
        Args:
            custom_limit: 自定义限制
            
        Returns:
            bool: 是否允许请求
        """
        try:
            client_ip = self._get_client_ip()
            current_time = time.time()
            
            with self._lock:
                # 更新统计
                self._stats['total_requests'] += 1
                
                # 检查IP是否被阻止
                if client_ip in self._blocked_ips:
                    if current_time < self._blocked_ips[client_ip]:
                        self._stats['blocked_requests'] += 1
                        return False
                    else:
                        # 解除阻止
                        del self._blocked_ips[client_ip]
                
                # 获取或创建限制信息
                rate_info = self._rate_limits[client_ip]
                
                # 清理过期请求
                window_start = current_time - self.config.rate_limit_window
                while rate_info.requests and rate_info.requests[0] < window_start:
                    rate_info.requests.popleft()
                
                # 检查请求数量
                limit = custom_limit or self.config.rate_limit_requests
                if len(rate_info.requests) >= limit:
                    self._stats['blocked_requests'] += 1
                    rate_info.blocked_requests += 1
                    
                    # 标记为可疑IP
                    self._suspicious_ips[client_ip] += 1
                    
                    # 如果频繁违规，临时阻止IP
                    if self._suspicious_ips[client_ip] > 5:
                        self._blocked_ips[client_ip] = current_time + 300  # 阻止5分钟
                        logger.warning(f"IP {client_ip} 因频繁违规被临时阻止")
                    
                    return False
                
                # 记录请求
                rate_info.requests.append(current_time)
                rate_info.total_requests += 1
                
                return True
                
        except Exception as e:
            logger.error(f"检查请求频率限制时发生错误: {e}")
            return True  # 发生错误时允许请求
    
    def _get_client_ip(self) -> str:
        """
        获取客户端IP地址
        
        Returns:
            str: 客户端IP地址
        """
        try:
            # 检查代理头
            if request.headers.get('X-Forwarded-For'):
                return request.headers.get('X-Forwarded-For').split(',')[0].strip()
            elif request.headers.get('X-Real-IP'):
                return request.headers.get('X-Real-IP')
            else:
                return request.remote_addr or '127.0.0.1'
        except Exception:
            return '127.0.0.1'
    
    def validate_input(self, data: Any, input_type: str = 'text') -> Tuple[bool, str, Any]:
        """
        验证和清理输入数据
        
        Args:
            data: 输入数据
            input_type: 输入类型 ('text', 'query', 'session_id', 'json')
            
        Returns:
            Tuple[bool, str, Any]: (是否有效, 错误信息, 清理后的数据)
        """
        try:
            if data is None:
                return False, "输入数据不能为空", None
            
            if input_type == 'text':
                return self._validate_text(data)
            elif input_type == 'query':
                return self._validate_query(data)
            elif input_type == 'session_id':
                return self._validate_session_id(data)
            elif input_type == 'json':
                return self._validate_json(data)
            else:
                return False, f"未知的输入类型: {input_type}", None
                
        except Exception as e:
            logger.error(f"验证输入时发生错误: {e}")
            return False, f"输入验证失败: {str(e)}", None
    
    def _validate_text(self, text: str) -> Tuple[bool, str, str]:
        """
        验证文本输入
        
        Args:
            text: 文本内容
            
        Returns:
            Tuple[bool, str, str]: (是否有效, 错误信息, 清理后的文本)
        """
        if not isinstance(text, str):
            return False, "文本必须是字符串类型", ""
        
        if len(text) > self.config.max_text_length:
            return False, f"文本长度超过限制 ({self.config.max_text_length} 字符)", ""
        
        # 检查恶意模式
        if self._detect_malicious_content(text):
            self._stats['malicious_requests'] += 1
            logger.warning(f"检测到恶意内容: {text[:100]}...")
            return False, "检测到潜在的恶意内容", ""
        
        # 清理HTML标签和危险字符
        cleaned_text = bleach.clean(text, tags=[], attributes={}, strip=True)
        
        return True, "", cleaned_text
    
    def _validate_query(self, query: str) -> Tuple[bool, str, str]:
        """
        验证查询输入
        
        Args:
            query: 查询内容
            
        Returns:
            Tuple[bool, str, str]: (是否有效, 错误信息, 清理后的查询)
        """
        if not isinstance(query, str):
            return False, "查询必须是字符串类型", ""
        
        if len(query) > self.config.max_query_length:
            return False, f"查询长度超过限制 ({self.config.max_query_length} 字符)", ""
        
        # 检查恶意模式
        if self._detect_malicious_content(query):
            self._stats['malicious_requests'] += 1
            return False, "检测到潜在的恶意查询", ""
        
        # 基本清理
        cleaned_query = query.strip()
        
        return True, "", cleaned_query
    
    def _validate_session_id(self, session_id: str) -> Tuple[bool, str, str]:
        """
        验证会话ID
        
        Args:
            session_id: 会话ID
            
        Returns:
            Tuple[bool, str, str]: (是否有效, 错误信息, 清理后的会话ID)
        """
        if not isinstance(session_id, str):
            return False, "会话ID必须是字符串类型", ""
        
        if len(session_id) > self.config.max_session_id_length:
            return False, f"会话ID长度超过限制 ({self.config.max_session_id_length} 字符)", ""
        
        # 会话ID应该只包含字母数字和连字符
        if not re.match(r'^[a-zA-Z0-9\-_]+$', session_id):
            return False, "会话ID包含无效字符", ""
        
        return True, "", session_id
    
    def _validate_json(self, data: dict) -> Tuple[bool, str, dict]:
        """
        验证JSON数据
        
        Args:
            data: JSON数据
            
        Returns:
            Tuple[bool, str, dict]: (是否有效, 错误信息, 清理后的数据)
        """
        if not isinstance(data, dict):
            return False, "数据必须是字典类型", {}
        
        # 递归验证和清理JSON中的字符串值
        cleaned_data = self._clean_json_recursively(data)
        
        return True, "", cleaned_data
    
    def _clean_json_recursively(self, obj: Any) -> Any:
        """
        递归清理JSON对象中的字符串
        
        Args:
            obj: 要清理的对象
            
        Returns:
            Any: 清理后的对象
        """
        try:
            if isinstance(obj, dict):
                return {k: self._clean_json_recursively(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [self._clean_json_recursively(item) for item in obj]
            elif isinstance(obj, str):
                # 检查恶意内容
                if self._detect_malicious_content(obj):
                    return "[已过滤的内容]"
                return bleach.clean(obj, tags=[], attributes={}, strip=True)
            else:
                return obj
        except Exception as e:
            logger.error(f"清理JSON时发生错误: {e}")
            return obj
    
    def _detect_malicious_content(self, content: str) -> bool:
        """
        检测恶意内容
        
        Args:
            content: 要检测的内容
            
        Returns:
            bool: 是否包含恶意内容
        """
        try:
            for pattern in self._compiled_patterns:
                if pattern.search(content):
                    return True
            return False
        except Exception as e:
            logger.error(f"检测恶意内容时发生错误: {e}")
            return False
    
    def validate_file_upload(self, file) -> Tuple[bool, str, Optional[str]]:
        """
        验证文件上传
        
        Args:
            file: 上传的文件对象
            
        Returns:
            Tuple[bool, str, Optional[str]]: (是否有效, 错误信息, 安全文件名)
        """
        try:
            self._stats['file_uploads'] += 1
            
            # 检查文件是否存在
            if not file or not file.filename:
                return False, "没有选择文件", None
            
            # 检查文件大小
            file.seek(0, 2)  # 移动到文件末尾
            file_size = file.tell()
            file.seek(0)  # 重置到文件开头
            
            if file_size > self.config.max_file_size:
                self._stats['blocked_files'] += 1
                return False, f"文件大小超过限制 ({self.config.max_file_size} 字节)", None
            
            # 检查文件扩展名
            filename = file.filename.lower()
            file_ext = filename.rsplit('.', 1)[1] if '.' in filename else ''
            
            if file_ext not in self.config.allowed_extensions:
                self._stats['blocked_files'] += 1
                return False, f"不允许的文件类型: {file_ext}", None
            
            # 生成安全的文件名
            safe_filename = secure_filename(file.filename)
            if not safe_filename:
                safe_filename = f"upload_{int(time.time())}.{file_ext}"
            
            # 检查文件内容（基本的魔数检查）
            if not self._validate_file_content(file, file_ext):
                self._stats['blocked_files'] += 1
                return False, "文件内容验证失败", None
            
            return True, "", safe_filename
            
        except Exception as e:
            logger.error(f"验证文件上传时发生错误: {e}")
            self._stats['blocked_files'] += 1
            return False, f"文件验证失败: {str(e)}", None
    
    def _validate_file_content(self, file, expected_ext: str) -> bool:
        """
        验证文件内容（魔数检查）
        
        Args:
            file: 文件对象
            expected_ext: 期望的文件扩展名
            
        Returns:
            bool: 文件内容是否有效
        """
        try:
            # 读取文件头部字节
            file.seek(0)
            header = file.read(16)
            file.seek(0)
            
            # 定义文件类型魔数
            magic_numbers = {
                'pdf': [b'%PDF'],
                'png': [b'\x89PNG\r\n\x1a\n'],
                'jpg': [b'\xff\xd8\xff'],
                'jpeg': [b'\xff\xd8\xff'],
                'gif': [b'GIF87a', b'GIF89a'],
                'bmp': [b'BM'],
                'txt': [],  # 文本文件没有固定魔数
                'doc': [b'\xd0\xcf\x11\xe0'],
                'docx': [b'PK\x03\x04']  # ZIP格式
            }
            
            if expected_ext not in magic_numbers:
                return True  # 未知类型，允许通过
            
            expected_magics = magic_numbers[expected_ext]
            if not expected_magics:  # 如txt等没有魔数的文件
                return True
            
            # 检查是否匹配任何期望的魔数
            for magic in expected_magics:
                if header.startswith(magic):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"验证文件内容时发生错误: {e}")
            return False
    
    def check_ip_access(self) -> Tuple[bool, str]:
        """
        检查IP访问权限
        
        Returns:
            Tuple[bool, str]: (是否允许访问, 错误信息)
        """
        try:
            client_ip = self._get_client_ip()
            
            # 检查黑名单
            if self.config.ip_blacklist and client_ip in self.config.ip_blacklist:
                logger.warning(f"IP {client_ip} 在黑名单中")
                return False, "访问被拒绝"
            
            # 检查白名单（如果配置了白名单）
            if self.config.ip_whitelist and client_ip not in self.config.ip_whitelist:
                logger.warning(f"IP {client_ip} 不在白名单中")
                return False, "访问被拒绝"
            
            return True, ""
            
        except Exception as e:
            logger.error(f"检查IP访问权限时发生错误: {e}")
            return True, ""  # 发生错误时允许访问
    
    def validate_api_key(self) -> Tuple[bool, str]:
        """
        验证API密钥
        
        Returns:
            Tuple[bool, str]: (是否有效, 错误信息)
        """
        try:
            if not self.config.require_api_key:
                return True, ""
            
            # 从请求头获取API密钥
            api_key = request.headers.get('X-API-Key') or request.headers.get('Authorization')
            
            if not api_key:
                return False, "缺少API密钥"
            
            # 处理Bearer token格式
            if api_key.startswith('Bearer '):
                api_key = api_key[7:]
            
            if api_key not in self.config.api_keys:
                logger.warning(f"无效的API密钥: {api_key[:10]}...")
                return False, "无效的API密钥"
            
            return True, ""
            
        except Exception as e:
            logger.error(f"验证API密钥时发生错误: {e}")
            return False, "API密钥验证失败"
    
    def add_security_headers(self, response):
        """
        添加安全响应头
        
        Args:
            response: Flask响应对象
            
        Returns:
            响应对象
        """
        try:
            if not self.config.enable_security_headers:
                return response
            
            # 基本安全头
            response.headers['X-Content-Type-Options'] = 'nosniff'
            response.headers['X-Frame-Options'] = 'DENY'
            response.headers['X-XSS-Protection'] = '1; mode=block'
            response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
            
            # CSP头
            csp = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:;"
            response.headers['Content-Security-Policy'] = csp
            
            # HSTS头（仅HTTPS）
            if request.is_secure:
                response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
            
            return response
            
        except Exception as e:
            logger.error(f"添加安全头时发生错误: {e}")
            return response
    
    def security_middleware(self):
        """
        安全中间件装饰器
        
        Returns:
            装饰器函数
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # IP访问检查
                ip_allowed, ip_error = self.check_ip_access()
                if not ip_allowed:
                    return jsonify({"error": ip_error}), 403
                
                # API密钥验证
                api_valid, api_error = self.validate_api_key()
                if not api_valid:
                    return jsonify({"error": api_error}), 401
                
                # 请求频率限制
                if not self._check_rate_limit():
                    return jsonify({
                        "error": "请求过于频繁，请稍后再试",
                        "code": "RATE_LIMIT_EXCEEDED"
                    }), 429
                
                # 执行原函数
                response = func(*args, **kwargs)
                
                # 添加安全头
                if hasattr(response, 'headers'):
                    response = self.add_security_headers(response)
                
                return response
            return wrapper
        return decorator
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取安全统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        try:
            with self._lock:
                stats = self._stats.copy()
                stats['active_rate_limits'] = len(self._rate_limits)
                stats['blocked_ips'] = len(self._blocked_ips)
                stats['suspicious_ips'] = len(self._suspicious_ips)
                return stats
        except Exception as e:
            logger.error(f"获取统计信息时发生错误: {e}")
            return {}
    
    def cleanup_expired_data(self):
        """
        清理过期数据
        """
        try:
            current_time = time.time()
            
            with self._lock:
                # 清理过期的IP阻止
                expired_ips = [ip for ip, unblock_time in self._blocked_ips.items() 
                             if current_time >= unblock_time]
                for ip in expired_ips:
                    del self._blocked_ips[ip]
                
                # 清理旧的请求记录
                window_start = current_time - self.config.rate_limit_window
                for ip, rate_info in list(self._rate_limits.items()):
                    while rate_info.requests and rate_info.requests[0] < window_start:
                        rate_info.requests.popleft()
                    
                    # 如果没有最近的请求，删除记录
                    if not rate_info.requests:
                        del self._rate_limits[ip]
                
                logger.info(f"清理了 {len(expired_ips)} 个过期IP阻止记录")
                
        except Exception as e:
            logger.error(f"清理过期数据时发生错误: {e}")

# 全局安全管理器实例
security_manager = SecurityManager()

# 便捷装饰器
def require_security(func):
    """
    安全检查装饰器
    
    Args:
        func: 要保护的函数
        
    Returns:
        装饰后的函数
    """
    return security_manager.security_middleware()(func)

def rate_limit(requests_per_minute: int = 60):
    """
    请求频率限制装饰器
    
    Args:
        requests_per_minute: 每分钟允许的请求数
        
    Returns:
        装饰器函数
    """
    return security_manager.rate_limit_decorator(requests_per_minute)