"""
Live2D AI助手 多模态系统模块

这个模块提供了完整的多模态AI功能，包括图像理解、视觉语言模型集成、
图文混合检索等功能，让AI助手能够理解和处理图像内容。

主要组件：
- MultimodalManager: 多模态管理器
- ImageAnalyzer: 图像分析器
- VisionLanguageModel: 视觉语言模型
- MultimodalRAG: 多模态RAG处理器
- ImageProcessor: 图像预处理器
"""

from .multimodal_manager import MultimodalManager
from .image_analyzer import ImageAnalyzer
from .vision_language_model import VisionLanguageModel
from .multimodal_rag import MultimodalRAG
from .image_processor import ImageProcessor
from .multimodal_config import MultimodalConfig

__version__ = "1.0.0"
__author__ = "Live2D AI Assistant Team"

__all__ = [
    "MultimodalManager",
    "ImageAnalyzer",
    "VisionLanguageModel", 
    "MultimodalRAG",
    "ImageProcessor",
    "MultimodalConfig"
]
