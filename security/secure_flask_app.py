# -*- coding: utf-8 -*-
"""
安全增强的Flask应用包装器
提供安全中间件集成和API端点保护
"""

import os
import logging
from typing import Optional, Dict, Any, Callable
from functools import wraps
from flask import Flask, request, jsonify, g
from werkzeug.exceptions import RequestEntityTooLarge

from .security_manager import SecurityManager, SecurityConfig

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecureFlaskApp:
    """
    安全增强的Flask应用包装器
    
    功能包括：
    1. 自动安全中间件集成
    2. 统一错误处理
    3. 请求验证
    4. 响应安全头
    5. 安全日志记录
    """
    
    def __init__(self, app: Flask, security_config: Optional[SecurityConfig] = None):
        """
        初始化安全Flask应用
        
        Args:
            app: Flask应用实例
            security_config: 安全配置
        """
        self.app = app
        self.security_manager = SecurityManager(security_config)
        
        # 注册中间件和错误处理器
        self._register_middleware()
        self._register_error_handlers()
        
        logger.info("安全Flask应用包装器初始化完成")
    
    def _register_middleware(self):
        """
        注册安全中间件
        """
        @self.app.before_request
        def security_check():
            """
            请求前安全检查
            """
            try:
                # 跳过静态文件和健康检查
                if request.endpoint in ['static', 'health']:
                    return None
                
                # IP访问控制
                ip_allowed, ip_error = self.security_manager.check_ip_access()
                if not ip_allowed:
                    logger.warning(f"IP访问被拒绝: {self.security_manager._get_client_ip()}")
                    return jsonify({"error": ip_error, "code": "IP_BLOCKED"}), 403
                
                # API密钥验证
                api_valid, api_error = self.security_manager.validate_api_key()
                if not api_valid:
                    logger.warning(f"API密钥验证失败: {request.headers.get('X-API-Key', 'None')[:10]}...")
                    return jsonify({"error": api_error, "code": "INVALID_API_KEY"}), 401
                
                # 请求大小检查
                if request.content_length and request.content_length > 50 * 1024 * 1024:  # 50MB
                    return jsonify({"error": "请求体过大", "code": "REQUEST_TOO_LARGE"}), 413
                
                # 存储客户端信息到g对象
                g.client_ip = self.security_manager._get_client_ip()
                g.request_start_time = request.environ.get('REQUEST_START_TIME')
                
                return None
                
            except Exception as e:
                logger.error(f"安全检查时发生错误: {e}")
                return jsonify({"error": "安全检查失败", "code": "SECURITY_ERROR"}), 500
        
        @self.app.after_request
        def add_security_headers(response):
            """
            添加安全响应头
            """
            try:
                return self.security_manager.add_security_headers(response)
            except Exception as e:
                logger.error(f"添加安全头时发生错误: {e}")
                return response
    
    def _register_error_handlers(self):
        """
        注册错误处理器
        """
        @self.app.errorhandler(400)
        def bad_request(error):
            """
            处理400错误
            """
            logger.warning(f"400错误: {error} - IP: {getattr(g, 'client_ip', 'unknown')}")
            return jsonify({
                "error": "请求格式错误",
                "code": "BAD_REQUEST",
                "details": str(error)
            }), 400
        
        @self.app.errorhandler(401)
        def unauthorized(error):
            """
            处理401错误
            """
            logger.warning(f"401错误: {error} - IP: {getattr(g, 'client_ip', 'unknown')}")
            return jsonify({
                "error": "未授权访问",
                "code": "UNAUTHORIZED",
                "details": "请提供有效的API密钥"
            }), 401
        
        @self.app.errorhandler(403)
        def forbidden(error):
            """
            处理403错误
            """
            logger.warning(f"403错误: {error} - IP: {getattr(g, 'client_ip', 'unknown')}")
            return jsonify({
                "error": "访问被禁止",
                "code": "FORBIDDEN",
                "details": str(error)
            }), 403
        
        @self.app.errorhandler(404)
        def not_found(error):
            """
            处理404错误
            """
            logger.info(f"404错误: {request.url} - IP: {getattr(g, 'client_ip', 'unknown')}")
            return jsonify({
                "error": "资源未找到",
                "code": "NOT_FOUND",
                "details": "请求的资源不存在"
            }), 404
        
        @self.app.errorhandler(413)
        def request_too_large(error):
            """
            处理413错误
            """
            logger.warning(f"413错误: 请求过大 - IP: {getattr(g, 'client_ip', 'unknown')}")
            return jsonify({
                "error": "请求体过大",
                "code": "REQUEST_TOO_LARGE",
                "details": "请求大小超过限制"
            }), 413
        
        @self.app.errorhandler(429)
        def rate_limit_exceeded(error):
            """
            处理429错误
            """
            logger.warning(f"429错误: 请求过于频繁 - IP: {getattr(g, 'client_ip', 'unknown')}")
            return jsonify({
                "error": "请求过于频繁",
                "code": "RATE_LIMIT_EXCEEDED",
                "details": "请稍后再试",
                "retry_after": 60
            }), 429
        
        @self.app.errorhandler(500)
        def internal_error(error):
            """
            处理500错误
            """
            logger.error(f"500错误: {error} - IP: {getattr(g, 'client_ip', 'unknown')}")
            return jsonify({
                "error": "服务器内部错误",
                "code": "INTERNAL_ERROR",
                "details": "服务暂时不可用，请稍后重试"
            }), 500
        
        @self.app.errorhandler(RequestEntityTooLarge)
        def handle_file_too_large(error):
            """
            处理文件过大错误
            """
            logger.warning(f"文件过大错误 - IP: {getattr(g, 'client_ip', 'unknown')}")
            return jsonify({
                "error": "上传文件过大",
                "code": "FILE_TOO_LARGE",
                "details": "文件大小超过限制"
            }), 413
    
    def secure_route(self, rule: str, **options):
        """
        创建安全路由装饰器
        
        Args:
            rule: 路由规则
            **options: 路由选项
            
        Returns:
            装饰器函数
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    # 请求频率限制检查
                    if not self.security_manager._check_rate_limit():
                        return jsonify({
                            "error": "请求过于频繁，请稍后再试",
                            "code": "RATE_LIMIT_EXCEEDED"
                        }), 429
                    
                    # 执行原函数
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    logger.error(f"路由 {rule} 执行时发生错误: {e}")
                    return jsonify({
                        "error": "请求处理失败",
                        "code": "PROCESSING_ERROR",
                        "details": str(e)
                    }), 500
            
            # 注册路由
            return self.app.route(rule, **options)(wrapper)
        
        return decorator
    
    def validate_json_input(self, required_fields: Optional[list] = None, 
                          optional_fields: Optional[list] = None):
        """
        JSON输入验证装饰器
        
        Args:
            required_fields: 必需字段列表
            optional_fields: 可选字段列表
            
        Returns:
            装饰器函数
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    # 检查Content-Type
                    if not request.is_json:
                        return jsonify({
                            "error": "请求必须是JSON格式",
                            "code": "INVALID_CONTENT_TYPE"
                        }), 400
                    
                    # 获取JSON数据
                    try:
                        data = request.get_json()
                    except Exception as e:
                        return jsonify({
                            "error": "JSON格式错误",
                            "code": "INVALID_JSON",
                            "details": str(e)
                        }), 400
                    
                    if data is None:
                        return jsonify({
                            "error": "请求体不能为空",
                            "code": "EMPTY_REQUEST"
                        }), 400
                    
                    # 验证必需字段
                    if required_fields:
                        missing_fields = [field for field in required_fields if field not in data]
                        if missing_fields:
                            return jsonify({
                                "error": f"缺少必需字段: {', '.join(missing_fields)}",
                                "code": "MISSING_FIELDS",
                                "missing_fields": missing_fields
                            }), 400
                    
                    # 验证和清理数据
                    valid, error_msg, cleaned_data = self.security_manager.validate_input(data, 'json')
                    if not valid:
                        return jsonify({
                            "error": error_msg,
                            "code": "INVALID_INPUT"
                        }), 400
                    
                    # 将清理后的数据添加到g对象
                    g.validated_data = cleaned_data
                    
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    logger.error(f"JSON验证时发生错误: {e}")
                    return jsonify({
                        "error": "输入验证失败",
                        "code": "VALIDATION_ERROR",
                        "details": str(e)
                    }), 500
            
            return wrapper
        return decorator
    
    def validate_file_upload(self, allowed_extensions: Optional[list] = None,
                           max_size: Optional[int] = None):
        """
        文件上传验证装饰器
        
        Args:
            allowed_extensions: 允许的文件扩展名列表
            max_size: 最大文件大小（字节）
            
        Returns:
            装饰器函数
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    # 检查是否有文件上传
                    if 'file' not in request.files:
                        return jsonify({
                            "error": "没有上传文件",
                            "code": "NO_FILE"
                        }), 400
                    
                    file = request.files['file']
                    
                    # 验证文件
                    valid, error_msg, safe_filename = self.security_manager.validate_file_upload(file)
                    if not valid:
                        return jsonify({
                            "error": error_msg,
                            "code": "INVALID_FILE"
                        }), 400
                    
                    # 将验证后的文件信息添加到g对象
                    g.uploaded_file = file
                    g.safe_filename = safe_filename
                    
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    logger.error(f"文件验证时发生错误: {e}")
                    return jsonify({
                        "error": "文件验证失败",
                        "code": "FILE_VALIDATION_ERROR",
                        "details": str(e)
                    }), 500
            
            return wrapper
        return decorator
    
    def add_health_check(self, endpoint: str = '/health'):
        """
        添加健康检查端点
        
        Args:
            endpoint: 健康检查端点路径
        """
        @self.app.route(endpoint, methods=['GET'])
        def health_check():
            """
            健康检查端点
            
            Returns:
                健康状态信息
            """
            try:
                stats = self.security_manager.get_stats()
                return jsonify({
                    "status": "healthy",
                    "timestamp": request.environ.get('REQUEST_START_TIME'),
                    "security_stats": stats
                }), 200
            except Exception as e:
                logger.error(f"健康检查失败: {e}")
                return jsonify({
                    "status": "unhealthy",
                    "error": str(e)
                }), 500
    
    def add_security_info_endpoint(self, endpoint: str = '/api/security/info'):
        """
        添加安全信息端点
        
        Args:
            endpoint: 安全信息端点路径
        """
        @self.app.route(endpoint, methods=['GET'])
        def security_info():
            """
            获取安全统计信息
            
            Returns:
                安全统计信息
            """
            try:
                stats = self.security_manager.get_stats()
                return jsonify({
                    "security_stats": stats,
                    "config": {
                        "rate_limit_enabled": True,
                        "file_upload_enabled": True,
                        "ip_filtering_enabled": bool(self.security_manager.config.ip_whitelist or 
                                                   self.security_manager.config.ip_blacklist),
                        "api_key_required": self.security_manager.config.require_api_key
                    }
                }), 200
            except Exception as e:
                logger.error(f"获取安全信息失败: {e}")
                return jsonify({
                    "error": "获取安全信息失败",
                    "details": str(e)
                }), 500
    
    def cleanup_security_data(self):
        """
        清理安全数据
        """
        try:
            self.security_manager.cleanup_expired_data()
            logger.info("安全数据清理完成")
        except Exception as e:
            logger.error(f"清理安全数据时发生错误: {e}")
    
    def get_security_manager(self) -> SecurityManager:
        """
        获取安全管理器实例
        
        Returns:
            SecurityManager: 安全管理器实例
        """
        return self.security_manager

def create_secure_app(app: Flask, security_config: Optional[SecurityConfig] = None) -> SecureFlaskApp:
    """
    创建安全增强的Flask应用
    
    Args:
        app: Flask应用实例
        security_config: 安全配置
        
    Returns:
        SecureFlaskApp: 安全增强的Flask应用
    """
    secure_app = SecureFlaskApp(app, security_config)
    
    # 添加默认端点
    secure_app.add_health_check()
    secure_app.add_security_info_endpoint()
    
    return secure_app