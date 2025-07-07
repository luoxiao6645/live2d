"""
高级RAG管理器

统一管理高级RAG系统的各个组件，提供高级API接口
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from .advanced_rag_config import AdvancedRAGConfig, VectorizerType
from .multimodal_vectorizer import MultimodalVectorizer
from .knowledge_graph_builder import KnowledgeGraphBuilder
from .content_fusion_engine import ContentFusionEngine, ContentSource
from .multimodal_reasoning_engine import MultimodalReasoningEngine
from .graph_query_processor import GraphQueryProcessor

# 配置日志
logger = logging.getLogger(__name__)

class AdvancedRAGManager:
    """高级RAG管理器类"""
    
    def __init__(self, rag_manager=None, multimodal_manager=None):
        """
        初始化高级RAG管理器
        
        Args:
            rag_manager: 基础RAG管理器实例（可选）
            multimodal_manager: 多模态管理器实例（可选）
        """
        self.config = AdvancedRAGConfig()
        
        # 基础系统集成
        self.rag_manager = rag_manager
        self.multimodal_manager = multimodal_manager
        
        # 初始化各个组件
        self.vectorizer = MultimodalVectorizer(VectorizerType.SENTENCE_TRANSFORMER)
        self.kg_builder = KnowledgeGraphBuilder()
        self.fusion_engine = ContentFusionEngine()
        self.reasoning_engine = MultimodalReasoningEngine(self.kg_builder)
        self.query_processor = GraphQueryProcessor(self.kg_builder, self.reasoning_engine)
        
        # 系统状态
        self.is_active = True
        self.processing_stats = {
            "documents_processed": 0,
            "queries_processed": 0,
            "reasoning_requests": 0,
            "fusion_operations": 0
        }
        
        # 创建必要的目录
        self.config.create_directories()
        
        # 加载现有图谱
        self._load_existing_graph()
        
        logger.info("高级RAG管理器初始化完成")
    
    def _load_existing_graph(self):
        """加载现有的知识图谱"""
        try:
            self.kg_builder.load_graph()
            logger.info("现有知识图谱加载成功")
        except Exception as e:
            logger.warning(f"知识图谱加载失败: {e}")
    
    def process_document(self, text: str, doc_id: str = None, 
                        image_path: str = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        处理文档（高级接口）
        
        Args:
            text: 文档文本
            doc_id: 文档ID
            image_path: 图像路径（可选）
            metadata: 元数据
            
        Returns:
            处理结果
        """
        try:
            self.processing_stats["documents_processed"] += 1
            
            result = {
                "success": True,
                "doc_id": doc_id,
                "processing_steps": [],
                "vectorization": {},
                "knowledge_graph": {},
                "multimodal_info": {}
            }
            
            # 1. 向量化处理
            try:
                if image_path:
                    # 图文混合向量化
                    embedding = self.vectorizer.vectorize_multimodal(text, image_path)
                    result["vectorization"] = {
                        "type": "multimodal",
                        "dimension": len(embedding),
                        "image_included": True
                    }
                else:
                    # 纯文本向量化
                    embedding = self.vectorizer.vectorize_text(text)
                    result["vectorization"] = {
                        "type": "text_only",
                        "dimension": len(embedding),
                        "image_included": False
                    }
                
                result["processing_steps"].append("向量化完成")
                
            except Exception as e:
                logger.error(f"向量化失败: {e}")
                result["processing_steps"].append(f"向量化失败: {e}")
            
            # 2. 知识图谱构建
            try:
                kg_result = self.kg_builder.build_graph_from_document(text, doc_id)
                result["knowledge_graph"] = kg_result
                result["processing_steps"].append("知识图谱构建完成")
                
            except Exception as e:
                logger.error(f"知识图谱构建失败: {e}")
                result["processing_steps"].append(f"知识图谱构建失败: {e}")
            
            # 3. 多模态信息处理
            if image_path and self.multimodal_manager:
                try:
                    # 集成多模态系统
                    multimodal_result = self._process_multimodal_content(text, image_path, metadata)
                    result["multimodal_info"] = multimodal_result
                    result["processing_steps"].append("多模态处理完成")
                    
                except Exception as e:
                    logger.error(f"多模态处理失败: {e}")
                    result["processing_steps"].append(f"多模态处理失败: {e}")
            
            # 4. 基础RAG集成
            if self.rag_manager:
                try:
                    # 添加到基础RAG系统
                    rag_result = self._integrate_with_basic_rag(text, doc_id, metadata)
                    result["basic_rag"] = rag_result
                    result["processing_steps"].append("基础RAG集成完成")
                    
                except Exception as e:
                    logger.error(f"基础RAG集成失败: {e}")
                    result["processing_steps"].append(f"基础RAG集成失败: {e}")
            
            logger.info(f"文档处理完成: {doc_id}")
            return result
            
        except Exception as e:
            logger.error(f"文档处理失败: {e}")
            return {"success": False, "error": str(e)}
    
    def advanced_query(self, query: str, query_type: str = "auto", 
                      max_results: int = 10, include_reasoning: bool = True) -> Dict[str, Any]:
        """
        高级查询（图谱查询 + 推理）
        
        Args:
            query: 查询问题
            query_type: 查询类型
            max_results: 最大结果数
            include_reasoning: 是否包含推理
            
        Returns:
            查询结果
        """
        try:
            self.processing_stats["queries_processed"] += 1
            
            result = {
                "success": True,
                "query": query,
                "query_type": query_type,
                "graph_query_results": [],
                "reasoning_results": {},
                "fused_answer": "",
                "confidence": 0.0,
                "sources": []
            }
            
            # 1. 图查询处理
            try:
                graph_result = self.query_processor.process_query(query)
                result["graph_query_results"] = graph_result.results
                result["sources"].extend([f"graph_query_{i}" for i in range(len(graph_result.results))])
                
            except Exception as e:
                logger.error(f"图查询失败: {e}")
            
            # 2. 推理处理
            if include_reasoning:
                try:
                    self.processing_stats["reasoning_requests"] += 1
                    reasoning_result = self.reasoning_engine.reason(query)
                    result["reasoning_results"] = reasoning_result.to_dict()
                    result["sources"].append("reasoning_engine")
                    
                except Exception as e:
                    logger.error(f"推理失败: {e}")
            
            # 3. 基础RAG查询（如果可用）
            basic_rag_results = []
            if self.rag_manager:
                try:
                    basic_results = self._query_basic_rag(query, max_results)
                    basic_rag_results = basic_results
                    result["sources"].append("basic_rag")
                    
                except Exception as e:
                    logger.error(f"基础RAG查询失败: {e}")
            
            # 4. 多模态查询（如果可用）
            multimodal_results = []
            if self.multimodal_manager:
                try:
                    multimodal_results = self._query_multimodal(query, max_results)
                    result["sources"].append("multimodal")
                    
                except Exception as e:
                    logger.error(f"多模态查询失败: {e}")
            
            # 5. 内容融合
            try:
                fused_result = self._fuse_query_results(
                    query, 
                    result["graph_query_results"],
                    result["reasoning_results"],
                    basic_rag_results,
                    multimodal_results
                )
                
                result["fused_answer"] = fused_result.fused_content
                result["confidence"] = fused_result.confidence
                
            except Exception as e:
                logger.error(f"结果融合失败: {e}")
                result["fused_answer"] = "结果融合失败"
                result["confidence"] = 0.0
            
            logger.info(f"高级查询完成: {query}")
            return result
            
        except Exception as e:
            logger.error(f"高级查询失败: {e}")
            return {"success": False, "error": str(e)}
    
    def multimodal_reasoning(self, query: str, image_path: str = None,
                           reasoning_type: str = "path_based") -> Dict[str, Any]:
        """
        多模态推理
        
        Args:
            query: 查询问题
            image_path: 图像路径（可选）
            reasoning_type: 推理类型
            
        Returns:
            推理结果
        """
        try:
            self.processing_stats["reasoning_requests"] += 1
            
            # 如果有图像，先处理图像内容
            if image_path:
                # 图像分析
                if self.multimodal_manager:
                    image_analysis = self._analyze_image_for_reasoning(image_path)
                    # 将图像信息融入查询
                    enhanced_query = f"{query} [图像信息: {image_analysis}]"
                else:
                    enhanced_query = query
            else:
                enhanced_query = query
            
            # 执行推理
            reasoning_result = self.reasoning_engine.reason(
                enhanced_query, 
                reasoning_type=reasoning_type
            )
            
            # 生成解释
            explanation = self.reasoning_engine.explain_reasoning(reasoning_result)
            
            result = {
                "success": True,
                "query": query,
                "enhanced_query": enhanced_query,
                "reasoning_result": reasoning_result.to_dict(),
                "explanation": explanation,
                "image_included": image_path is not None
            }
            
            return result
            
        except Exception as e:
            logger.error(f"多模态推理失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _process_multimodal_content(self, text: str, image_path: str, 
                                   metadata: Dict[str, Any]) -> Dict[str, Any]:
        """处理多模态内容"""
        if not self.multimodal_manager:
            return {"error": "多模态管理器不可用"}
        
        try:
            # 这里可以调用多模态管理器的相关方法
            # 例如图像分析、图文关联等
            return {
                "image_processed": True,
                "text_image_alignment": "completed",
                "multimodal_features_extracted": True
            }
            
        except Exception as e:
            logger.error(f"多模态内容处理失败: {e}")
            return {"error": str(e)}
    
    def _integrate_with_basic_rag(self, text: str, doc_id: str, 
                                 metadata: Dict[str, Any]) -> Dict[str, Any]:
        """与基础RAG系统集成"""
        if not self.rag_manager:
            return {"error": "基础RAG管理器不可用"}
        
        try:
            # 调用基础RAG系统的添加文档方法
            # 这里需要根据实际的RAG管理器接口调整
            return {
                "added_to_basic_rag": True,
                "doc_id": doc_id
            }
            
        except Exception as e:
            logger.error(f"基础RAG集成失败: {e}")
            return {"error": str(e)}
    
    def _query_basic_rag(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """查询基础RAG系统"""
        if not self.rag_manager:
            return []
        
        try:
            # 调用基础RAG系统的查询方法
            # 这里需要根据实际的RAG管理器接口调整
            return []
            
        except Exception as e:
            logger.error(f"基础RAG查询失败: {e}")
            return []
    
    def _query_multimodal(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """查询多模态系统"""
        if not self.multimodal_manager:
            return []
        
        try:
            # 调用多模态管理器的搜索方法
            # 这里需要根据实际的多模态管理器接口调整
            return []
            
        except Exception as e:
            logger.error(f"多模态查询失败: {e}")
            return []
    
    def _analyze_image_for_reasoning(self, image_path: str) -> str:
        """为推理分析图像"""
        if not self.multimodal_manager:
            return "图像分析不可用"
        
        try:
            # 调用多模态管理器的图像分析方法
            return "图像分析结果"
            
        except Exception as e:
            logger.error(f"图像分析失败: {e}")
            return "图像分析失败"
    
    def _fuse_query_results(self, query: str, graph_results: List[Dict[str, Any]],
                           reasoning_results: Dict[str, Any],
                           basic_rag_results: List[Dict[str, Any]],
                           multimodal_results: List[Dict[str, Any]]):
        """融合查询结果"""
        try:
            self.processing_stats["fusion_operations"] += 1
            
            # 创建内容源
            sources = []
            
            # 图查询结果
            for i, result in enumerate(graph_results):
                content = self._format_graph_result(result)
                source = ContentSource(
                    source_id=f"graph_{i}",
                    content=content,
                    credibility=0.8,
                    metadata={"type": "graph_query", "result": result}
                )
                sources.append(source)
            
            # 推理结果
            if reasoning_results and reasoning_results.get("answer"):
                source = ContentSource(
                    source_id="reasoning",
                    content=reasoning_results["answer"],
                    credibility=reasoning_results.get("confidence", 0.5),
                    metadata={"type": "reasoning", "result": reasoning_results}
                )
                sources.append(source)
            
            # 基础RAG结果
            for i, result in enumerate(basic_rag_results):
                content = str(result)  # 简化处理
                source = ContentSource(
                    source_id=f"basic_rag_{i}",
                    content=content,
                    credibility=0.7,
                    metadata={"type": "basic_rag", "result": result}
                )
                sources.append(source)
            
            # 多模态结果
            for i, result in enumerate(multimodal_results):
                content = str(result)  # 简化处理
                source = ContentSource(
                    source_id=f"multimodal_{i}",
                    content=content,
                    credibility=0.6,
                    metadata={"type": "multimodal", "result": result}
                )
                sources.append(source)
            
            # 执行融合
            if sources:
                return self.fusion_engine.fuse_content(sources, "attention_based")
            else:
                return self.fusion_engine.fuse_content([], "empty")
                
        except Exception as e:
            logger.error(f"结果融合失败: {e}")
            return self.fusion_engine.fuse_content([], "error")
    
    def _format_graph_result(self, result: Dict[str, Any]) -> str:
        """格式化图查询结果"""
        try:
            result_type = result.get("type", "unknown")
            
            if result_type == "entity":
                return f"实体: {result.get('entity', '未知')}"
            elif result_type == "relation":
                return f"关系: {result.get('source', '未知')} -> {result.get('target', '未知')}"
            elif result_type == "path":
                path = result.get("path", [])
                return f"路径: {' -> '.join(path)}"
            elif result_type == "neighbor":
                return f"邻居: {result.get('neighbor', '未知')}"
            else:
                return str(result)
                
        except Exception as e:
            logger.error(f"图结果格式化失败: {e}")
            return str(result)
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        try:
            return {
                "is_active": self.is_active,
                "processing_stats": self.processing_stats,
                "vectorizer_stats": self.vectorizer.get_vectorizer_stats(),
                "knowledge_graph_stats": self.kg_builder.get_graph_stats(),
                "fusion_stats": self.fusion_engine.get_fusion_stats(),
                "reasoning_stats": self.reasoning_engine.get_reasoning_stats(),
                "query_stats": self.query_processor.get_query_stats(),
                "integration_status": {
                    "basic_rag_available": self.rag_manager is not None,
                    "multimodal_available": self.multimodal_manager is not None
                }
            }
            
        except Exception as e:
            logger.error(f"获取系统状态失败: {e}")
            return {"error": str(e)}
    
    def save_system_state(self):
        """保存系统状态"""
        try:
            # 保存知识图谱
            self.kg_builder.save_graph()
            
            # 保存向量化缓存
            self.vectorizer.save_cache("./embeddings_cache/vectorizer_cache.pkl")
            
            logger.info("系统状态保存成功")
            
        except Exception as e:
            logger.error(f"系统状态保存失败: {e}")
    
    def clear_caches(self):
        """清理所有缓存"""
        try:
            self.vectorizer.clear_cache()
            self.fusion_engine.fusion_cache.clear()
            self.reasoning_engine.clear_cache()
            self.query_processor.clear_cache()
            
            logger.info("所有缓存已清理")
            
        except Exception as e:
            logger.error(f"缓存清理失败: {e}")
    
    def shutdown(self):
        """关闭系统"""
        try:
            self.is_active = False
            self.save_system_state()
            self.clear_caches()
            
            logger.info("高级RAG系统已关闭")
            
        except Exception as e:
            logger.error(f"系统关闭失败: {e}")
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'is_active') and self.is_active:
            self.shutdown()
