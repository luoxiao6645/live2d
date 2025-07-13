"""
多模态RAG处理器

扩展现有的RAG系统，支持图文混合检索和生成
"""

import logging
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import hashlib

from .multimodal_config import MultimodalConfig
from .image_processor import ImageProcessor
from .image_analyzer import ImageAnalyzer
from .vision_language_model import VisionLanguageModel

# 配置日志
logger = logging.getLogger(__name__)

# 尝试导入RAG相关模块
try:
    from rag_manager import RAGManager
    from langchain_core.documents import Document
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logger.warning("RAG模块不可用，多模态RAG功能将受限")

class MultimodalRAG:
    """多模态RAG处理器类"""
    
    def __init__(self, rag_manager: Optional[Any] = None):
        """
        初始化多模态RAG处理器
        
        Args:
            rag_manager: 现有的RAG管理器实例
        """
        self.config = MultimodalConfig()
        self.image_processor = ImageProcessor()
        self.image_analyzer = ImageAnalyzer()
        self.vision_model = VisionLanguageModel()
        
        # RAG管理器
        self.rag_manager = rag_manager
        self.rag_available = RAG_AVAILABLE and rag_manager is not None
        
        # 多模态向量存储
        self.multimodal_documents = {}  # 存储图文文档
        self.image_embeddings = {}      # 存储图像嵌入
        
        logger.info(f"多模态RAG处理器初始化完成，RAG可用: {self.rag_available}")
    
    def add_image_document(self, image_path: str, description: str = None, 
                          metadata: Dict[str, Any] = None) -> str:
        """
        添加图像文档到知识库
        
        Args:
            image_path: 图像路径
            description: 图像描述（可选，会自动生成）
            metadata: 额外元数据
            
        Returns:
            文档ID
        """
        try:
            # 处理图像
            process_result = self.image_processor.process_image(
                open(image_path, 'rb').read(),
                os.path.basename(image_path)
            )
            
            if not process_result["success"]:
                raise ValueError(f"图像处理失败: {process_result['error']}")
            
            # 生成图像描述（如果未提供）
            if not description:
                desc_result = self.vision_model.generate_image_description(image_path)
                if desc_result["success"]:
                    description = desc_result["description"]
                else:
                    description = "图像内容描述生成失败"
            
            # 分析图像内容
            analysis_result = self.image_analyzer.analyze_image(image_path)
            
            # 生成文档ID
            doc_id = self._generate_multimodal_doc_id(image_path)
            
            # 构建文档元数据
            doc_metadata = {
                "doc_id": doc_id,
                "type": "image",
                "image_path": image_path,
                "processed_path": process_result.get("processed_path"),
                "thumbnail_path": process_result.get("thumbnail_path"),
                "description": description,
                "upload_time": datetime.now().isoformat(),
                "file_info": process_result.get("file_info", {}),
                "image_features": process_result.get("features", {}),
                **(metadata or {})
            }
            
            # 添加分析结果
            if analysis_result.get("success"):
                doc_metadata.update({
                    "analysis": analysis_result,
                    "objects": analysis_result.get("objects", []),
                    "emotions": analysis_result.get("emotions", []),
                    "scene": analysis_result.get("scene", ""),
                    "colors": analysis_result.get("colors", [])
                })
            
            # 如果RAG可用，添加到向量数据库
            if self.rag_available:
                # 创建文档对象
                document = Document(
                    page_content=description,
                    metadata=doc_metadata
                )
                
                # 添加到RAG系统
                self.rag_manager.vector_store.add_documents([document])
            
            # 存储到本地
            self.multimodal_documents[doc_id] = doc_metadata
            
            logger.info(f"图像文档添加成功: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"添加图像文档失败: {e}")
            raise
    
    def add_image_text_document(self, image_path: str, text_content: str, 
                               metadata: Dict[str, Any] = None) -> str:
        """
        添加图文混合文档到知识库
        
        Args:
            image_path: 图像路径
            text_content: 文本内容
            metadata: 额外元数据
            
        Returns:
            文档ID
        """
        try:
            # 处理图像
            process_result = self.image_processor.process_image(
                open(image_path, 'rb').read(),
                os.path.basename(image_path)
            )
            
            if not process_result["success"]:
                raise ValueError(f"图像处理失败: {process_result['error']}")
            
            # 生成图像描述
            desc_result = self.vision_model.generate_image_description(image_path)
            image_description = desc_result.get("description", "") if desc_result["success"] else ""
            
            # 合并图像描述和文本内容
            combined_content = f"{text_content}\n\n[图像描述]: {image_description}"
            
            # 生成文档ID
            doc_id = self._generate_multimodal_doc_id(f"{image_path}_{text_content[:50]}")
            
            # 构建文档元数据
            doc_metadata = {
                "doc_id": doc_id,
                "type": "image_text",
                "image_path": image_path,
                "processed_path": process_result.get("processed_path"),
                "thumbnail_path": process_result.get("thumbnail_path"),
                "text_content": text_content,
                "image_description": image_description,
                "combined_content": combined_content,
                "upload_time": datetime.now().isoformat(),
                "file_info": process_result.get("file_info", {}),
                **(metadata or {})
            }
            
            # 如果RAG可用，添加到向量数据库
            if self.rag_available:
                document = Document(
                    page_content=combined_content,
                    metadata=doc_metadata
                )
                self.rag_manager.vector_store.add_documents([document])
            
            # 存储到本地
            self.multimodal_documents[doc_id] = doc_metadata
            
            logger.info(f"图文文档添加成功: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"添加图文文档失败: {e}")
            raise
    
    def search_multimodal(self, query: str, include_images: bool = True, 
                         include_text: bool = True, k: int = 5) -> List[Dict[str, Any]]:
        """
        多模态搜索
        
        Args:
            query: 搜索查询
            include_images: 是否包含图像结果
            include_text: 是否包含文本结果
            k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        try:
            results = []
            
            # 如果RAG可用，使用向量搜索
            if self.rag_available:
                # 搜索向量数据库
                vector_results = self.rag_manager.search_documents(query, k=k*2)
                
                for doc in vector_results:
                    doc_type = doc.metadata.get("type", "text")
                    
                    # 根据类型过滤结果
                    if doc_type == "image" and not include_images:
                        continue
                    if doc_type in ["text", "image_text"] and not include_text:
                        continue
                    
                    result = {
                        "doc_id": doc.metadata.get("doc_id", ""),
                        "type": doc_type,
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "relevance_score": 0.8  # 占位符评分
                    }
                    
                    # 添加图像特定信息
                    if doc_type in ["image", "image_text"]:
                        result.update({
                            "image_path": doc.metadata.get("image_path"),
                            "thumbnail_path": doc.metadata.get("thumbnail_path"),
                            "image_description": doc.metadata.get("image_description", "")
                        })
                    
                    results.append(result)
            
            else:
                # 简单的关键词搜索
                results = self._simple_keyword_search(query, include_images, include_text, k)
            
            # 按相关性排序
            results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            
            return results[:k]
            
        except Exception as e:
            logger.error(f"多模态搜索失败: {e}")
            return []
    
    def _simple_keyword_search(self, query: str, include_images: bool, 
                              include_text: bool, k: int) -> List[Dict[str, Any]]:
        """简单的关键词搜索"""
        results = []
        query_lower = query.lower()
        
        for doc_id, doc_metadata in self.multimodal_documents.items():
            doc_type = doc_metadata.get("type", "text")
            
            # 类型过滤
            if doc_type == "image" and not include_images:
                continue
            if doc_type in ["text", "image_text"] and not include_text:
                continue
            
            # 计算相关性
            relevance_score = 0.0
            
            # 检查描述
            description = doc_metadata.get("description", "")
            if query_lower in description.lower():
                relevance_score += 0.8
            
            # 检查文本内容
            text_content = doc_metadata.get("text_content", "")
            if query_lower in text_content.lower():
                relevance_score += 0.9
            
            # 检查分析结果
            analysis = doc_metadata.get("analysis", {})
            if isinstance(analysis, dict):
                objects = analysis.get("objects", [])
                emotions = analysis.get("emotions", [])
                scene = analysis.get("scene", "")
                
                for obj in objects:
                    if query_lower in obj.lower():
                        relevance_score += 0.6
                
                for emotion in emotions:
                    if query_lower in emotion.lower():
                        relevance_score += 0.5
                
                if query_lower in scene.lower():
                    relevance_score += 0.7
            
            if relevance_score > 0:
                result = {
                    "doc_id": doc_id,
                    "type": doc_type,
                    "content": description or text_content,
                    "metadata": doc_metadata,
                    "relevance_score": relevance_score
                }
                
                if doc_type in ["image", "image_text"]:
                    result.update({
                        "image_path": doc_metadata.get("image_path"),
                        "thumbnail_path": doc_metadata.get("thumbnail_path"),
                        "image_description": doc_metadata.get("image_description", "")
                    })
                
                results.append(result)
        
        return results
    
    def generate_multimodal_response(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """
        基于搜索结果生成多模态回答
        
        Args:
            query: 用户查询
            search_results: 搜索结果
            
        Returns:
            生成的回答
        """
        try:
            if not search_results:
                return "抱歉，我没有找到相关的信息。"
            
            # 构建上下文
            context_parts = []
            image_count = 0
            text_count = 0
            
            for result in search_results:
                doc_type = result.get("type", "text")
                content = result.get("content", "")
                
                if doc_type == "image":
                    image_count += 1
                    context_parts.append(f"[图像{image_count}]: {content}")
                elif doc_type == "image_text":
                    image_count += 1
                    text_count += 1
                    context_parts.append(f"[图文{image_count}]: {content}")
                else:
                    text_count += 1
                    context_parts.append(f"[文本{text_count}]: {content}")
            
            context = "\n\n".join(context_parts)
            
            # 生成回答
            if image_count > 0:
                response = f"根据找到的{image_count}张图像和{text_count}段文本，我可以告诉您：\n\n"
            else:
                response = f"根据找到的{text_count}段相关文本，我可以告诉您：\n\n"
            
            # 简单的回答生成（可以集成到LLM中）
            response += self._generate_simple_answer(query, context)
            
            return response
            
        except Exception as e:
            logger.error(f"多模态回答生成失败: {e}")
            return "抱歉，生成回答时出现了错误。"
    
    def _generate_simple_answer(self, query: str, context: str) -> str:
        """生成简单回答"""
        # 这里可以集成到现有的LLM生成流程中
        # 目前提供简化版本
        
        if "什么" in query or "是什么" in query:
            return f"基于相关内容，{context[:200]}..."
        elif "如何" in query or "怎么" in query:
            return f"根据相关信息，{context[:200]}..."
        elif "为什么" in query:
            return f"从相关资料来看，{context[:200]}..."
        else:
            return f"关于您的问题，相关信息显示：{context[:200]}..."
    
    def _generate_multimodal_doc_id(self, content: str) -> str:
        """生成多模态文档ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"mm_{timestamp}_{content_hash}"
    
    def get_multimodal_stats(self) -> Dict[str, Any]:
        """获取多模态知识库统计信息"""
        try:
            total_docs = len(self.multimodal_documents)
            image_docs = sum(1 for doc in self.multimodal_documents.values() 
                           if doc.get("type") == "image")
            text_docs = sum(1 for doc in self.multimodal_documents.values() 
                          if doc.get("type") == "text")
            image_text_docs = sum(1 for doc in self.multimodal_documents.values() 
                                if doc.get("type") == "image_text")
            
            return {
                "total_documents": total_docs,
                "image_documents": image_docs,
                "text_documents": text_docs,
                "image_text_documents": image_text_docs,
                "rag_available": self.rag_available
            }
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {"error": str(e)}
    
    def delete_multimodal_document(self, doc_id: str) -> bool:
        """删除多模态文档"""
        try:
            # 从本地存储删除
            if doc_id in self.multimodal_documents:
                doc_metadata = self.multimodal_documents[doc_id]
                
                # 删除相关文件
                image_path = doc_metadata.get("image_path")
                processed_path = doc_metadata.get("processed_path")
                thumbnail_path = doc_metadata.get("thumbnail_path")
                
                for path in [processed_path, thumbnail_path]:
                    if path and os.path.exists(path):
                        try:
                            os.remove(path)
                        except Exception as e:
                            logger.warning(f"删除文件失败 {path}: {e}")
                
                del self.multimodal_documents[doc_id]
            
            # 从RAG系统删除（如果可用）
            if self.rag_available:
                # 这里需要实现从向量数据库删除的逻辑
                # 目前RAG管理器可能不支持按doc_id删除
                pass
            
            logger.info(f"多模态文档删除成功: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"删除多模态文档失败: {e}")
            return False
