"""
Live2D AI助手 高级RAG系统模块

这个模块提供了高级的检索增强生成功能，包括图文混合向量化、知识图谱构建、
智能内容融合、多模态推理等功能，让AI助手能够进行更深层次的知识理解和推理。

主要组件：
- AdvancedRAGManager: 高级RAG管理器
- MultimodalVectorizer: 图文混合向量化器
- KnowledgeGraphBuilder: 知识图谱构建器
- ContentFusionEngine: 内容融合引擎
- MultimodalReasoningEngine: 多模态推理引擎
- GraphQueryProcessor: 图查询处理器
"""

from .advanced_rag_manager import AdvancedRAGManager
from .multimodal_vectorizer import MultimodalVectorizer
from .knowledge_graph_builder import KnowledgeGraphBuilder
from .content_fusion_engine import ContentFusionEngine
from .multimodal_reasoning_engine import MultimodalReasoningEngine
from .graph_query_processor import GraphQueryProcessor
from .advanced_rag_config import AdvancedRAGConfig

__version__ = "1.0.0"
__author__ = "Live2D AI Assistant Team"

__all__ = [
    "AdvancedRAGManager",
    "MultimodalVectorizer",
    "KnowledgeGraphBuilder",
    "ContentFusionEngine",
    "MultimodalReasoningEngine",
    "GraphQueryProcessor",
    "AdvancedRAGConfig"
]
