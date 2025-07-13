#!/usr/bin/env python3
"""
Ollama客户端模块

提供与Ollama API的交互功能
"""

import requests
import json
import logging
from typing import Dict, List, Optional, Any
import time

logger = logging.getLogger(__name__)

class OllamaClient:
    """Ollama API客户端"""
    
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 30):
        """
        初始化Ollama客户端
        
        Args:
            base_url: Ollama服务器地址
            timeout: 请求超时时间
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
    def is_available(self) -> bool:
        """
        检查Ollama服务是否可用
        
        Returns:
            bool: 服务是否可用
        """
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Ollama服务不可用: {e}")
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        获取可用模型列表
        
        Returns:
            List[Dict]: 模型列表
        """
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get('models', [])
        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
            return []
    
    def generate(self, model: str, prompt: str, system: str = None, 
                 stream: bool = False, **kwargs) -> Dict[str, Any]:
        """
        生成文本
        
        Args:
            model: 模型名称
            prompt: 用户提示
            system: 系统提示
            stream: 是否流式输出
            **kwargs: 其他参数
            
        Returns:
            Dict: 生成结果
        """
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": stream
            }
            
            if system:
                payload["system"] = system
                
            # 添加其他参数
            payload.update(kwargs)
            
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            if stream:
                return self._handle_stream_response(response)
            else:
                return response.json()
                
        except Exception as e:
            logger.error(f"生成文本失败: {e}")
            return {
                "response": "抱歉，我现在无法回答您的问题。",
                "error": str(e)
            }
    
    def chat(self, model: str, messages: List[Dict[str, str]], 
             stream: bool = False, **kwargs) -> Dict[str, Any]:
        """
        聊天对话
        
        Args:
            model: 模型名称
            messages: 消息列表
            stream: 是否流式输出
            **kwargs: 其他参数
            
        Returns:
            Dict: 聊天结果
        """
        try:
            payload = {
                "model": model,
                "messages": messages,
                "stream": stream
            }
            
            # 添加其他参数
            payload.update(kwargs)
            
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            if stream:
                return self._handle_stream_response(response)
            else:
                return response.json()
                
        except Exception as e:
            logger.error(f"聊天失败: {e}")
            return {
                "message": {
                    "role": "assistant",
                    "content": "抱歉，我现在无法回答您的问题。"
                },
                "error": str(e)
            }
    
    def _handle_stream_response(self, response):
        """处理流式响应"""
        try:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    if 'response' in data:
                        full_response += data['response']
                    elif 'message' in data and 'content' in data['message']:
                        full_response += data['message']['content']
                        
            return {
                "response": full_response,
                "done": True
            }
        except Exception as e:
            logger.error(f"处理流式响应失败: {e}")
            return {
                "response": "抱歉，处理响应时出现错误。",
                "error": str(e)
            }
    
    def pull_model(self, model: str) -> bool:
        """
        拉取模型
        
        Args:
            model: 模型名称
            
        Returns:
            bool: 是否成功
        """
        try:
            payload = {"name": model}
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json=payload,
                timeout=300  # 拉取模型可能需要更长时间
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"拉取模型失败: {e}")
            return False
    
    def delete_model(self, model: str) -> bool:
        """
        删除模型
        
        Args:
            model: 模型名称
            
        Returns:
            bool: 是否成功
        """
        try:
            payload = {"name": model}
            response = self.session.delete(
                f"{self.base_url}/api/delete",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"删除模型失败: {e}")
            return False
    
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """
        获取模型信息

        Args:
            model: 模型名称

        Returns:
            Dict: 模型信息
        """
        try:
            payload = {"name": model}
            response = self.session.post(
                f"{self.base_url}/api/show",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"获取模型信息失败: {e}")
            return {}

    def generate_for_web(self, model: str, prompt: str, **kwargs):
        """
        为Web应用生成流式响应

        Args:
            model: 模型名称
            prompt: 提示文本
            **kwargs: 其他参数

        Returns:
            Generator: 流式响应生成器
        """
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": True
            }
            payload.update(kwargs)

            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        data = json.loads(line)
                        yield data
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.error(f"Web生成失败: {e}")
            yield {
                "response": "抱歉，我现在无法回答您的问题。",
                "done": True,
                "error": str(e)
            }

# 创建默认客户端实例
default_client = OllamaClient()

def get_ollama_client() -> OllamaClient:
    """获取默认Ollama客户端实例"""
    return default_client
