"""
图文混合向量化器

负责文本、图像、图文混合内容的高级向量化处理
"""

import logging
import numpy as np
import hashlib
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import pickle
import os

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers不可用，将使用简化向量化")

try:
    import torch
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("torch不可用，图像向量化功能将受限")

from .advanced_rag_config import AdvancedRAGConfig, VectorizerType

# 配置日志
logger = logging.getLogger(__name__)

class MultimodalVectorizer:
    """图文混合向量化器类"""
    
    def __init__(self, vectorizer_type: VectorizerType = VectorizerType.SENTENCE_TRANSFORMER):
        """
        初始化图文混合向量化器
        
        Args:
            vectorizer_type: 向量化器类型
        """
        self.config = AdvancedRAGConfig()
        self.vectorizer_type = vectorizer_type
        
        # 向量化模型
        self.text_model = None
        self.image_model = None
        self.multimodal_model = None
        
        # 缓存
        self.embedding_cache = {}
        self.cache_stats = {"hits": 0, "misses": 0}
        
        # 初始化模型
        self._init_models()
        
        logger.info(f"图文混合向量化器初始化完成: {vectorizer_type.value}")
    
    def _init_models(self):
        """初始化向量化模型"""
        try:
            if self.vectorizer_type == VectorizerType.SENTENCE_TRANSFORMER:
                self._init_sentence_transformer()
            elif self.vectorizer_type == VectorizerType.CLIP:
                self._init_clip_model()
            else:
                self._init_simple_vectorizer()
                
        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            self._init_simple_vectorizer()
    
    def _init_sentence_transformer(self):
        """初始化Sentence Transformer模型"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ValueError("sentence-transformers库不可用")
        
        model_name = self.config.VECTORIZER_CONFIG["text_model"]
        
        try:
            self.text_model = SentenceTransformer(model_name)
            logger.info(f"Sentence Transformer模型加载成功: {model_name}")
        except Exception as e:
            logger.error(f"Sentence Transformer模型加载失败: {e}")
            raise
    
    def _init_clip_model(self):
        """初始化CLIP模型"""
        if not TORCH_AVAILABLE:
            raise ValueError("torch库不可用")
        
        try:
            # 这里可以添加CLIP模型的实际加载代码
            # import clip
            # self.multimodal_model, self.preprocess = clip.load("ViT-B/32")
            
            # 占位符
            self.multimodal_model = None
            logger.info("CLIP模型初始化完成（占位符）")
            
        except Exception as e:
            logger.error(f"CLIP模型初始化失败: {e}")
            raise
    
    def _init_simple_vectorizer(self):
        """初始化简单向量化器"""
        logger.info("使用简单向量化器")
        # 简单的TF-IDF向量化器作为后备
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.text_model = TfidfVectorizer(
            max_features=self.config.VECTORIZER_CONFIG["embedding_dimension"],
            ngram_range=(1, 2),
            stop_words=None
        )
    
    def vectorize_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        文本向量化
        
        Args:
            text: 输入文本
            use_cache: 是否使用缓存
            
        Returns:
            文本向量
        """
        try:
            # 生成缓存键
            cache_key = self._generate_cache_key("text", text) if use_cache else None
            
            # 检查缓存
            if use_cache and cache_key in self.embedding_cache:
                self.cache_stats["hits"] += 1
                return self.embedding_cache[cache_key]
            
            self.cache_stats["misses"] += 1
            
            # 执行向量化
            if isinstance(self.text_model, SentenceTransformer):
                # 使用Sentence Transformer
                embedding = self.text_model.encode(text, convert_to_numpy=True)
            else:
                # 使用简单向量化器
                embedding = self._simple_text_vectorize(text)
            
            # 缓存结果
            if use_cache and cache_key:
                self.embedding_cache[cache_key] = embedding
                self._cleanup_cache()
            
            return embedding
            
        except Exception as e:
            logger.error(f"文本向量化失败: {e}")
            # 返回零向量作为后备
            return np.zeros(self.config.VECTORIZER_CONFIG["embedding_dimension"])
    
    def vectorize_image(self, image_path: str, use_cache: bool = True) -> np.ndarray:
        """
        图像向量化
        
        Args:
            image_path: 图像路径
            use_cache: 是否使用缓存
            
        Returns:
            图像向量
        """
        try:
            # 生成缓存键
            cache_key = self._generate_cache_key("image", image_path) if use_cache else None
            
            # 检查缓存
            if use_cache and cache_key in self.embedding_cache:
                self.cache_stats["hits"] += 1
                return self.embedding_cache[cache_key]
            
            self.cache_stats["misses"] += 1
            
            # 执行向量化
            if self.multimodal_model:
                # 使用CLIP等多模态模型
                embedding = self._clip_image_vectorize(image_path)
            else:
                # 使用简单图像特征
                embedding = self._simple_image_vectorize(image_path)
            
            # 缓存结果
            if use_cache and cache_key:
                self.embedding_cache[cache_key] = embedding
                self._cleanup_cache()
            
            return embedding
            
        except Exception as e:
            logger.error(f"图像向量化失败: {e}")
            # 返回零向量作为后备
            return np.zeros(self.config.VECTORIZER_CONFIG["embedding_dimension"])
    
    def vectorize_multimodal(self, text: str, image_path: str = None, 
                           fusion_strategy: str = "concatenate") -> np.ndarray:
        """
        图文混合向量化
        
        Args:
            text: 文本内容
            image_path: 图像路径（可选）
            fusion_strategy: 融合策略 (concatenate, average, weighted, attention)
            
        Returns:
            融合后的向量
        """
        try:
            # 文本向量化
            text_embedding = self.vectorize_text(text)
            
            if image_path and os.path.exists(image_path):
                # 图像向量化
                image_embedding = self.vectorize_image(image_path)
                
                # 融合向量
                fused_embedding = self._fuse_embeddings(
                    text_embedding, image_embedding, fusion_strategy
                )
            else:
                # 只有文本
                fused_embedding = text_embedding
            
            return fused_embedding
            
        except Exception as e:
            logger.error(f"图文混合向量化失败: {e}")
            return np.zeros(self.config.VECTORIZER_CONFIG["embedding_dimension"])
    
    def _simple_text_vectorize(self, text: str) -> np.ndarray:
        """简单文本向量化"""
        try:
            # 使用TF-IDF向量化
            if hasattr(self.text_model, 'transform'):
                # 已训练的模型
                vector = self.text_model.transform([text]).toarray()[0]
            else:
                # 需要训练的模型
                vector = self.text_model.fit_transform([text]).toarray()[0]
            
            # 确保维度正确
            target_dim = self.config.VECTORIZER_CONFIG["embedding_dimension"]
            if len(vector) > target_dim:
                vector = vector[:target_dim]
            elif len(vector) < target_dim:
                vector = np.pad(vector, (0, target_dim - len(vector)))
            
            return vector.astype(np.float32)
            
        except Exception as e:
            logger.error(f"简单文本向量化失败: {e}")
            return np.random.random(self.config.VECTORIZER_CONFIG["embedding_dimension"]).astype(np.float32)
    
    def _simple_image_vectorize(self, image_path: str) -> np.ndarray:
        """简单图像向量化"""
        try:
            # 使用基础图像特征
            from PIL import Image
            
            with Image.open(image_path) as img:
                # 转换为RGB
                img = img.convert('RGB')
                
                # 调整大小
                img = img.resize((224, 224))
                
                # 提取颜色直方图特征
                hist_r = np.histogram(np.array(img)[:,:,0], bins=32, range=(0, 256))[0]
                hist_g = np.histogram(np.array(img)[:,:,1], bins=32, range=(0, 256))[0]
                hist_b = np.histogram(np.array(img)[:,:,2], bins=32, range=(0, 256))[0]
                
                # 合并特征
                features = np.concatenate([hist_r, hist_g, hist_b])
                
                # 归一化
                features = features / (features.sum() + 1e-8)
                
                # 调整维度
                target_dim = self.config.VECTORIZER_CONFIG["embedding_dimension"]
                if len(features) > target_dim:
                    features = features[:target_dim]
                elif len(features) < target_dim:
                    features = np.pad(features, (0, target_dim - len(features)))
                
                return features.astype(np.float32)
                
        except Exception as e:
            logger.error(f"简单图像向量化失败: {e}")
            return np.random.random(self.config.VECTORIZER_CONFIG["embedding_dimension"]).astype(np.float32)
    
    def _clip_image_vectorize(self, image_path: str) -> np.ndarray:
        """使用CLIP进行图像向量化"""
        try:
            # 这里是CLIP图像向量化的占位符
            # 实际实现需要加载CLIP模型
            
            # 模拟CLIP向量化
            return np.random.random(self.config.VECTORIZER_CONFIG["embedding_dimension"]).astype(np.float32)
            
        except Exception as e:
            logger.error(f"CLIP图像向量化失败: {e}")
            return self._simple_image_vectorize(image_path)
    
    def _fuse_embeddings(self, text_embedding: np.ndarray, image_embedding: np.ndarray,
                        strategy: str) -> np.ndarray:
        """融合文本和图像向量"""
        try:
            if strategy == "concatenate":
                # 拼接策略
                fused = np.concatenate([text_embedding, image_embedding])
                # 调整到目标维度
                target_dim = self.config.VECTORIZER_CONFIG["embedding_dimension"]
                if len(fused) > target_dim:
                    fused = fused[:target_dim]
                elif len(fused) < target_dim:
                    fused = np.pad(fused, (0, target_dim - len(fused)))
                return fused
                
            elif strategy == "average":
                # 平均策略
                return (text_embedding + image_embedding) / 2
                
            elif strategy == "weighted":
                # 加权策略（文本权重更高）
                text_weight = 0.7
                image_weight = 0.3
                return text_weight * text_embedding + image_weight * image_embedding
                
            elif strategy == "attention":
                # 注意力机制（简化版本）
                attention_weights = self._compute_attention_weights(text_embedding, image_embedding)
                return attention_weights[0] * text_embedding + attention_weights[1] * image_embedding
                
            else:
                # 默认使用平均策略
                return (text_embedding + image_embedding) / 2
                
        except Exception as e:
            logger.error(f"向量融合失败: {e}")
            return text_embedding  # 返回文本向量作为后备
    
    def _compute_attention_weights(self, text_embedding: np.ndarray, 
                                 image_embedding: np.ndarray) -> Tuple[float, float]:
        """计算注意力权重"""
        try:
            # 简化的注意力计算
            text_norm = np.linalg.norm(text_embedding)
            image_norm = np.linalg.norm(image_embedding)
            
            total_norm = text_norm + image_norm
            if total_norm == 0:
                return 0.5, 0.5
            
            text_weight = text_norm / total_norm
            image_weight = image_norm / total_norm
            
            return text_weight, image_weight
            
        except Exception as e:
            logger.error(f"注意力权重计算失败: {e}")
            return 0.5, 0.5
    
    def _generate_cache_key(self, modality: str, content: str) -> str:
        """生成缓存键"""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"{modality}_{content_hash[:16]}"
    
    def _cleanup_cache(self):
        """清理缓存"""
        try:
            max_cache_size = self.config.VECTORIZER_CONFIG["cache_size"]
            
            if len(self.embedding_cache) > max_cache_size:
                # 删除最旧的缓存项（简化实现）
                keys_to_remove = list(self.embedding_cache.keys())[:-max_cache_size]
                for key in keys_to_remove:
                    del self.embedding_cache[key]
                
                logger.info(f"缓存清理完成，保留 {max_cache_size} 个项目")
                
        except Exception as e:
            logger.error(f"缓存清理失败: {e}")
    
    def batch_vectorize_texts(self, texts: List[str]) -> List[np.ndarray]:
        """批量文本向量化"""
        try:
            if isinstance(self.text_model, SentenceTransformer):
                # 使用批量处理
                embeddings = self.text_model.encode(texts, convert_to_numpy=True, batch_size=self.config.VECTORIZER_CONFIG["batch_size"])
                return [emb for emb in embeddings]
            else:
                # 逐个处理
                return [self.vectorize_text(text) for text in texts]
                
        except Exception as e:
            logger.error(f"批量文本向量化失败: {e}")
            return [self.vectorize_text(text) for text in texts]
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """计算向量相似度"""
        try:
            # 使用余弦相似度
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"相似度计算失败: {e}")
            return 0.0
    
    def get_vectorizer_stats(self) -> Dict[str, Any]:
        """获取向量化器统计信息"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        cache_hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            "vectorizer_type": self.vectorizer_type.value,
            "cache_size": len(self.embedding_cache),
            "cache_hit_rate": cache_hit_rate,
            "total_requests": total_requests,
            "embedding_dimension": self.config.VECTORIZER_CONFIG["embedding_dimension"]
        }
    
    def save_cache(self, file_path: str):
        """保存缓存到文件"""
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logger.info(f"缓存保存成功: {file_path}")
        except Exception as e:
            logger.error(f"缓存保存失败: {e}")
    
    def load_cache(self, file_path: str):
        """从文件加载缓存"""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"缓存加载成功: {file_path}")
        except Exception as e:
            logger.error(f"缓存加载失败: {e}")
    
    def clear_cache(self):
        """清空缓存"""
        self.embedding_cache.clear()
        self.cache_stats = {"hits": 0, "misses": 0}
        logger.info("缓存已清空")
