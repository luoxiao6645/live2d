"""
多模态管理器

统一管理多模态系统的各个组件，提供高级API接口
"""

import logging
import os
from typing import Dict, List, Optional, Any, Union
from werkzeug.datastructures import FileStorage

from .multimodal_config import MultimodalConfig, VisionModelType
from .image_processor import ImageProcessor
from .image_analyzer import ImageAnalyzer
from .vision_language_model import VisionLanguageModel
from .multimodal_rag import MultimodalRAG

# 配置日志
logger = logging.getLogger(__name__)

class MultimodalManager:
    """多模态管理器类"""
    
    def __init__(self, rag_manager: Optional[Any] = None, 
                 vision_model_type: VisionModelType = VisionModelType.SIMPLE_TAGS):
        """
        初始化多模态管理器
        
        Args:
            rag_manager: RAG管理器实例
            vision_model_type: 视觉模型类型
        """
        self.config = MultimodalConfig()
        
        # 初始化各个组件
        self.image_processor = ImageProcessor()
        self.image_analyzer = ImageAnalyzer(vision_model_type)
        self.vision_model = VisionLanguageModel(vision_model_type)
        self.multimodal_rag = MultimodalRAG(rag_manager)
        
        # 状态管理
        self.is_active = True
        self.processed_images = {}  # 缓存处理过的图像信息
        
        logger.info("多模态管理器初始化完成")
    
    def upload_and_process_image(self, file: FileStorage, 
                                description: str = None,
                                add_to_knowledge_base: bool = True) -> Dict[str, Any]:
        """
        上传并处理图像
        
        Args:
            file: 上传的文件对象
            description: 图像描述（可选）
            add_to_knowledge_base: 是否添加到知识库
            
        Returns:
            处理结果
        """
        try:
            if not file or not file.filename:
                return {"success": False, "error": "没有选择文件"}
            
            # 读取文件数据
            file_data = file.read()
            filename = file.filename
            
            # 验证图像
            is_valid, error_msg = self.image_processor.validate_image(file_data, filename)
            if not is_valid:
                return {"success": False, "error": error_msg}
            
            # 处理图像
            process_result = self.image_processor.process_image(file_data, filename)
            if not process_result["success"]:
                return process_result
            
            # 保存处理后的图像路径
            processed_path = process_result["processed_path"]
            
            # 分析图像内容
            analysis_result = self.image_analyzer.analyze_image(processed_path)
            
            # 生成图像描述
            if not description:
                desc_result = self.vision_model.generate_image_description(processed_path)
                if desc_result["success"]:
                    description = desc_result["description"]
                else:
                    description = "自动生成描述失败"
            
            # 构建结果
            result = {
                "success": True,
                "image_id": process_result["file_info"]["image_id"],
                "processed_path": processed_path,
                "thumbnail_path": process_result["thumbnail_path"],
                "description": description,
                "analysis": analysis_result,
                "file_info": process_result["file_info"],
                "features": process_result["features"]
            }
            
            # 添加到知识库
            if add_to_knowledge_base:
                try:
                    doc_id = self.multimodal_rag.add_image_document(
                        processed_path, 
                        description,
                        {
                            "image_id": result["image_id"],
                            "original_filename": filename,
                            "analysis": analysis_result
                        }
                    )
                    result["doc_id"] = doc_id
                    result["added_to_knowledge_base"] = True
                except Exception as e:
                    logger.warning(f"添加到知识库失败: {e}")
                    result["added_to_knowledge_base"] = False
                    result["knowledge_base_error"] = str(e)
            
            # 缓存处理结果
            self.processed_images[result["image_id"]] = result
            
            logger.info(f"图像处理完成: {filename}")
            return result
            
        except Exception as e:
            logger.error(f"图像上传处理失败: {e}")
            return {"success": False, "error": str(e)}
    
    def analyze_image_with_question(self, image_id: str, question: str) -> Dict[str, Any]:
        """
        对图像进行问答分析
        
        Args:
            image_id: 图像ID
            question: 问题
            
        Returns:
            分析结果
        """
        try:
            # 获取图像信息
            if image_id not in self.processed_images:
                return {"success": False, "error": "图像不存在"}
            
            image_info = self.processed_images[image_id]
            image_path = image_info["processed_path"]
            
            # 使用视觉语言模型回答问题
            answer_result = self.vision_model.answer_image_question(image_path, question)
            
            if answer_result["success"]:
                result = {
                    "success": True,
                    "image_id": image_id,
                    "question": question,
                    "answer": answer_result.get("answer", answer_result.get("description", "")),
                    "model_info": answer_result.get("model", "unknown"),
                    "confidence": answer_result.get("confidence", 0.8)
                }
            else:
                result = {
                    "success": False,
                    "error": answer_result.get("error", "问答失败")
                }
            
            return result
            
        except Exception as e:
            logger.error(f"图像问答失败: {e}")
            return {"success": False, "error": str(e)}
    
    def search_multimodal_content(self, query: str, search_type: str = "all") -> Dict[str, Any]:
        """
        搜索多模态内容
        
        Args:
            query: 搜索查询
            search_type: 搜索类型 (all, images, text)
            
        Returns:
            搜索结果
        """
        try:
            include_images = search_type in ["all", "images"]
            include_text = search_type in ["all", "text"]
            
            # 执行搜索
            search_results = self.multimodal_rag.search_multimodal(
                query, 
                include_images=include_images,
                include_text=include_text,
                k=10
            )
            
            # 生成回答
            response = self.multimodal_rag.generate_multimodal_response(query, search_results)
            
            return {
                "success": True,
                "query": query,
                "search_type": search_type,
                "results": search_results,
                "response": response,
                "result_count": len(search_results)
            }
            
        except Exception as e:
            logger.error(f"多模态搜索失败: {e}")
            return {"success": False, "error": str(e)}
    
    def compare_images(self, image_id1: str, image_id2: str, 
                      comparison_aspect: str = "整体相似性") -> Dict[str, Any]:
        """
        比较两张图像
        
        Args:
            image_id1: 第一张图像ID
            image_id2: 第二张图像ID
            comparison_aspect: 比较方面
            
        Returns:
            比较结果
        """
        try:
            # 检查图像是否存在
            if image_id1 not in self.processed_images:
                return {"success": False, "error": f"图像不存在: {image_id1}"}
            if image_id2 not in self.processed_images:
                return {"success": False, "error": f"图像不存在: {image_id2}"}
            
            # 获取图像路径
            image_path1 = self.processed_images[image_id1]["processed_path"]
            image_path2 = self.processed_images[image_id2]["processed_path"]
            
            # 使用图像分析器比较
            analysis_comparison = self.image_analyzer.compare_images(image_path1, image_path2)
            
            # 使用视觉语言模型比较
            vlm_comparison = self.vision_model.compare_images_with_text(
                image_path1, image_path2, comparison_aspect
            )
            
            return {
                "success": True,
                "image_id1": image_id1,
                "image_id2": image_id2,
                "comparison_aspect": comparison_aspect,
                "analysis_comparison": analysis_comparison,
                "vlm_comparison": vlm_comparison,
                "overall_similarity": (
                    analysis_comparison.get("similarity", 0) + 
                    vlm_comparison.get("similarity_score", 0)
                ) / 2
            }
            
        except Exception as e:
            logger.error(f"图像比较失败: {e}")
            return {"success": False, "error": str(e)}
    
    def get_image_info(self, image_id: str) -> Dict[str, Any]:
        """
        获取图像信息
        
        Args:
            image_id: 图像ID
            
        Returns:
            图像信息
        """
        try:
            if image_id not in self.processed_images:
                return {"success": False, "error": "图像不存在"}
            
            image_info = self.processed_images[image_id].copy()
            image_info["success"] = True
            
            return image_info
            
        except Exception as e:
            logger.error(f"获取图像信息失败: {e}")
            return {"success": False, "error": str(e)}
    
    def list_processed_images(self) -> Dict[str, Any]:
        """
        列出所有处理过的图像
        
        Returns:
            图像列表
        """
        try:
            images = []
            for image_id, image_info in self.processed_images.items():
                images.append({
                    "image_id": image_id,
                    "filename": image_info.get("file_info", {}).get("original_filename", ""),
                    "description": image_info.get("description", ""),
                    "thumbnail_path": image_info.get("thumbnail_path", ""),
                    "upload_time": image_info.get("file_info", {}).get("timestamp", ""),
                    "analysis_summary": self._get_analysis_summary(image_info.get("analysis", {}))
                })
            
            # 按时间排序
            images.sort(key=lambda x: x.get("upload_time", ""), reverse=True)
            
            return {
                "success": True,
                "images": images,
                "total_count": len(images)
            }
            
        except Exception as e:
            logger.error(f"列出图像失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_analysis_summary(self, analysis: Dict[str, Any]) -> str:
        """获取分析摘要"""
        if not analysis or not analysis.get("success"):
            return "分析失败"
        
        parts = []
        
        if "objects" in analysis and analysis["objects"]:
            parts.append(f"物体: {', '.join(analysis['objects'][:3])}")
        
        if "emotions" in analysis and analysis["emotions"]:
            parts.append(f"情感: {', '.join(analysis['emotions'][:2])}")
        
        if "scene" in analysis and analysis["scene"]:
            parts.append(f"场景: {analysis['scene']}")
        
        return "; ".join(parts) if parts else "基础分析完成"
    
    def delete_image(self, image_id: str) -> Dict[str, Any]:
        """
        删除图像
        
        Args:
            image_id: 图像ID
            
        Returns:
            删除结果
        """
        try:
            if image_id not in self.processed_images:
                return {"success": False, "error": "图像不存在"}
            
            image_info = self.processed_images[image_id]
            
            # 删除文件
            files_to_delete = [
                image_info.get("processed_path"),
                image_info.get("thumbnail_path")
            ]
            
            deleted_files = []
            for file_path in files_to_delete:
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        deleted_files.append(file_path)
                    except Exception as e:
                        logger.warning(f"删除文件失败 {file_path}: {e}")
            
            # 从知识库删除
            doc_id = image_info.get("doc_id")
            if doc_id:
                self.multimodal_rag.delete_multimodal_document(doc_id)
            
            # 从缓存删除
            del self.processed_images[image_id]
            
            return {
                "success": True,
                "image_id": image_id,
                "deleted_files": deleted_files,
                "message": "图像删除成功"
            }
            
        except Exception as e:
            logger.error(f"删除图像失败: {e}")
            return {"success": False, "error": str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        获取系统状态
        
        Returns:
            系统状态信息
        """
        try:
            # 获取各组件状态
            multimodal_stats = self.multimodal_rag.get_multimodal_stats()
            
            return {
                "success": True,
                "is_active": self.is_active,
                "processed_images_count": len(self.processed_images),
                "multimodal_stats": multimodal_stats,
                "vision_model_type": self.vision_model.model_type.value,
                "components": {
                    "image_processor": True,
                    "image_analyzer": True,
                    "vision_language_model": True,
                    "multimodal_rag": self.multimodal_rag.rag_available
                }
            }
            
        except Exception as e:
            logger.error(f"获取系统状态失败: {e}")
            return {"success": False, "error": str(e)}
    
    def cleanup_old_data(self, days: int = 7) -> Dict[str, Any]:
        """
        清理旧数据
        
        Args:
            days: 保留天数
            
        Returns:
            清理结果
        """
        try:
            # 清理处理过的图像文件
            deleted_files = self.image_processor.cleanup_old_files(days)
            
            # 清理缓存中的过期数据
            import time
            current_time = time.time()
            cutoff_time = current_time - (days * 24 * 60 * 60)
            
            expired_images = []
            for image_id, image_info in list(self.processed_images.items()):
                timestamp_str = image_info.get("file_info", {}).get("timestamp", "")
                if timestamp_str:
                    try:
                        # 解析时间戳
                        from datetime import datetime
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S").timestamp()
                        if timestamp < cutoff_time:
                            expired_images.append(image_id)
                    except:
                        pass
            
            # 删除过期图像记录
            for image_id in expired_images:
                del self.processed_images[image_id]
            
            return {
                "success": True,
                "deleted_files": deleted_files,
                "expired_images": len(expired_images),
                "message": f"清理完成，删除了{deleted_files}个文件和{len(expired_images)}条记录"
            }
            
        except Exception as e:
            logger.error(f"数据清理失败: {e}")
            return {"success": False, "error": str(e)}
    
    def shutdown(self):
        """关闭多模态管理器"""
        try:
            self.is_active = False
            logger.info("多模态管理器已关闭")
        except Exception as e:
            logger.error(f"关闭多模态管理器失败: {e}")
    
    def __del__(self):
        """析构函数"""
        self.shutdown()
