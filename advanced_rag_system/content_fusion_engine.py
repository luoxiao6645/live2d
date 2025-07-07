"""
内容融合引擎

负责多源信息的智能融合、冲突处理和质量评估
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib

from .advanced_rag_config import AdvancedRAGConfig

# 配置日志
logger = logging.getLogger(__name__)

class ContentSource:
    """内容源类"""
    
    def __init__(self, source_id: str, content: str, metadata: Dict[str, Any] = None,
                 credibility: float = 1.0, timestamp: datetime = None):
        self.source_id = source_id
        self.content = content
        self.metadata = metadata or {}
        self.credibility = credibility
        self.timestamp = timestamp or datetime.now()
        self.quality_score = 0.0
        self.processed = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "source_id": self.source_id,
            "content": self.content,
            "metadata": self.metadata,
            "credibility": self.credibility,
            "timestamp": self.timestamp.isoformat(),
            "quality_score": self.quality_score,
            "processed": self.processed
        }

class FusionResult:
    """融合结果类"""
    
    def __init__(self, fused_content: str, confidence: float, sources: List[str],
                 fusion_strategy: str, quality_metrics: Dict[str, float] = None):
        self.fused_content = fused_content
        self.confidence = confidence
        self.sources = sources
        self.fusion_strategy = fusion_strategy
        self.quality_metrics = quality_metrics or {}
        self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "fused_content": self.fused_content,
            "confidence": self.confidence,
            "sources": self.sources,
            "fusion_strategy": self.fusion_strategy,
            "quality_metrics": self.quality_metrics,
            "created_at": self.created_at.isoformat()
        }

class ContentFusionEngine:
    """内容融合引擎类"""
    
    def __init__(self):
        """初始化内容融合引擎"""
        self.config = AdvancedRAGConfig()
        
        # 融合缓存
        self.fusion_cache = {}
        self.cache_stats = {"hits": 0, "misses": 0}
        
        # 质量评估器
        self.quality_assessor = QualityAssessor()
        
        # 冲突解决器
        self.conflict_resolver = ConflictResolver()
        
        logger.info("内容融合引擎初始化完成")
    
    def fuse_content(self, sources: List[ContentSource], 
                    strategy: str = None) -> FusionResult:
        """
        融合多源内容
        
        Args:
            sources: 内容源列表
            strategy: 融合策略
            
        Returns:
            融合结果
        """
        try:
            if not sources:
                return FusionResult("", 0.0, [], "empty")
            
            if len(sources) == 1:
                # 单一源，直接返回
                source = sources[0]
                return FusionResult(
                    source.content, 
                    source.credibility, 
                    [source.source_id],
                    "single_source"
                )
            
            # 选择融合策略
            if not strategy:
                strategy = self.config.CONTENT_FUSION_CONFIG["default_strategy"]
            
            # 检查缓存
            cache_key = self._generate_cache_key(sources, strategy)
            if cache_key in self.fusion_cache:
                self.cache_stats["hits"] += 1
                return self.fusion_cache[cache_key]
            
            self.cache_stats["misses"] += 1
            
            # 质量评估
            for source in sources:
                if not source.processed:
                    source.quality_score = self.quality_assessor.assess_quality(source)
                    source.processed = True
            
            # 冲突检测和解决
            if self.config.CONTENT_FUSION_CONFIG["conflict_resolution"]["enabled"]:
                sources = self.conflict_resolver.resolve_conflicts(sources)
            
            # 执行融合
            if strategy == "weighted_average":
                result = self._weighted_average_fusion(sources)
            elif strategy == "attention_based":
                result = self._attention_based_fusion(sources)
            elif strategy == "graph_based":
                result = self._graph_based_fusion(sources)
            elif strategy == "confidence_based":
                result = self._confidence_based_fusion(sources)
            else:
                result = self._simple_fusion(sources)
            
            # 缓存结果
            if self.config.CONTENT_FUSION_CONFIG["fusion_cache"]["enabled"]:
                self.fusion_cache[cache_key] = result
                self._cleanup_cache()
            
            return result
            
        except Exception as e:
            logger.error(f"内容融合失败: {e}")
            return FusionResult("", 0.0, [], "error")
    
    def _weighted_average_fusion(self, sources: List[ContentSource]) -> FusionResult:
        """加权平均融合"""
        try:
            # 计算权重
            total_weight = sum(source.quality_score * source.credibility for source in sources)
            
            if total_weight == 0:
                return self._simple_fusion(sources)
            
            # 按权重融合内容
            fused_parts = []
            source_ids = []
            total_confidence = 0
            
            for source in sources:
                weight = (source.quality_score * source.credibility) / total_weight
                
                if weight > 0.1:  # 只包含权重较高的源
                    fused_parts.append(f"[权重{weight:.2f}] {source.content}")
                    source_ids.append(source.source_id)
                    total_confidence += weight * source.credibility
            
            fused_content = "\n".join(fused_parts)
            confidence = total_confidence / len(source_ids) if source_ids else 0
            
            return FusionResult(
                fused_content, 
                confidence, 
                source_ids,
                "weighted_average",
                {"total_weight": total_weight, "sources_used": len(source_ids)}
            )
            
        except Exception as e:
            logger.error(f"加权平均融合失败: {e}")
            return self._simple_fusion(sources)
    
    def _attention_based_fusion(self, sources: List[ContentSource]) -> FusionResult:
        """基于注意力的融合"""
        try:
            # 计算注意力权重
            attention_weights = self._compute_attention_weights(sources)
            
            # 融合内容
            fused_parts = []
            source_ids = []
            total_confidence = 0
            
            for i, (source, weight) in enumerate(zip(sources, attention_weights)):
                if weight > 0.05:  # 注意力阈值
                    # 根据注意力权重调整内容
                    if weight > 0.5:
                        fused_parts.append(f"**{source.content}**")  # 高注意力
                    elif weight > 0.2:
                        fused_parts.append(source.content)  # 中等注意力
                    else:
                        fused_parts.append(f"({source.content})")  # 低注意力
                    
                    source_ids.append(source.source_id)
                    total_confidence += weight * source.credibility
            
            fused_content = " ".join(fused_parts)
            confidence = total_confidence
            
            return FusionResult(
                fused_content,
                confidence,
                source_ids,
                "attention_based",
                {"attention_weights": attention_weights.tolist()}
            )
            
        except Exception as e:
            logger.error(f"注意力融合失败: {e}")
            return self._simple_fusion(sources)
    
    def _graph_based_fusion(self, sources: List[ContentSource]) -> FusionResult:
        """基于图的融合"""
        try:
            # 构建源之间的相似性图
            similarity_matrix = self._compute_similarity_matrix(sources)
            
            # 使用PageRank算法计算重要性
            importance_scores = self._compute_pagerank_scores(similarity_matrix)
            
            # 按重要性排序融合
            sorted_indices = np.argsort(importance_scores)[::-1]
            
            fused_parts = []
            source_ids = []
            total_confidence = 0
            
            for i in sorted_indices:
                source = sources[i]
                importance = importance_scores[i]
                
                if importance > 0.1:  # 重要性阈值
                    fused_parts.append(source.content)
                    source_ids.append(source.source_id)
                    total_confidence += importance * source.credibility
            
            fused_content = "\n---\n".join(fused_parts)
            confidence = total_confidence / len(source_ids) if source_ids else 0
            
            return FusionResult(
                fused_content,
                confidence,
                source_ids,
                "graph_based",
                {"importance_scores": importance_scores.tolist()}
            )
            
        except Exception as e:
            logger.error(f"图融合失败: {e}")
            return self._simple_fusion(sources)
    
    def _confidence_based_fusion(self, sources: List[ContentSource]) -> FusionResult:
        """基于置信度的融合"""
        try:
            # 按置信度排序
            sorted_sources = sorted(sources, 
                                  key=lambda x: x.credibility * x.quality_score, 
                                  reverse=True)
            
            # 选择高置信度的源
            selected_sources = []
            cumulative_confidence = 0
            
            for source in sorted_sources:
                confidence = source.credibility * source.quality_score
                
                if confidence > 0.5 or len(selected_sources) == 0:  # 至少保留一个源
                    selected_sources.append(source)
                    cumulative_confidence += confidence
                
                if cumulative_confidence > 2.0:  # 置信度饱和
                    break
            
            # 融合选中的源
            fused_parts = []
            source_ids = []
            
            for source in selected_sources:
                fused_parts.append(source.content)
                source_ids.append(source.source_id)
            
            fused_content = "\n\n".join(fused_parts)
            confidence = cumulative_confidence / len(selected_sources)
            
            return FusionResult(
                fused_content,
                confidence,
                source_ids,
                "confidence_based",
                {"cumulative_confidence": cumulative_confidence}
            )
            
        except Exception as e:
            logger.error(f"置信度融合失败: {e}")
            return self._simple_fusion(sources)
    
    def _simple_fusion(self, sources: List[ContentSource]) -> FusionResult:
        """简单融合（连接所有内容）"""
        try:
            contents = [source.content for source in sources]
            source_ids = [source.source_id for source in sources]
            
            fused_content = "\n".join(contents)
            confidence = sum(source.credibility for source in sources) / len(sources)
            
            return FusionResult(
                fused_content,
                confidence,
                source_ids,
                "simple",
                {"source_count": len(sources)}
            )
            
        except Exception as e:
            logger.error(f"简单融合失败: {e}")
            return FusionResult("", 0.0, [], "error")
    
    def _compute_attention_weights(self, sources: List[ContentSource]) -> np.ndarray:
        """计算注意力权重"""
        try:
            # 基于质量分数和可信度计算注意力
            scores = np.array([
                source.quality_score * source.credibility 
                for source in sources
            ])
            
            # Softmax归一化
            exp_scores = np.exp(scores - np.max(scores))
            weights = exp_scores / np.sum(exp_scores)
            
            return weights
            
        except Exception as e:
            logger.error(f"注意力权重计算失败: {e}")
            return np.ones(len(sources)) / len(sources)
    
    def _compute_similarity_matrix(self, sources: List[ContentSource]) -> np.ndarray:
        """计算相似性矩阵"""
        try:
            n = len(sources)
            similarity_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        # 简单的文本相似度计算
                        similarity = self._compute_text_similarity(
                            sources[i].content, 
                            sources[j].content
                        )
                        similarity_matrix[i][j] = similarity
                    else:
                        similarity_matrix[i][j] = 1.0
            
            return similarity_matrix
            
        except Exception as e:
            logger.error(f"相似性矩阵计算失败: {e}")
            n = len(sources)
            return np.eye(n)
    
    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        try:
            # 简单的Jaccard相似度
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"文本相似度计算失败: {e}")
            return 0.0
    
    def _compute_pagerank_scores(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """计算PageRank分数"""
        try:
            n = similarity_matrix.shape[0]
            
            # 转换为转移矩阵
            row_sums = similarity_matrix.sum(axis=1)
            transition_matrix = similarity_matrix / (row_sums[:, np.newaxis] + 1e-8)
            
            # PageRank迭代
            damping = 0.85
            scores = np.ones(n) / n
            
            for _ in range(50):  # 最多50次迭代
                new_scores = (1 - damping) / n + damping * transition_matrix.T.dot(scores)
                
                if np.allclose(scores, new_scores, atol=1e-6):
                    break
                
                scores = new_scores
            
            return scores
            
        except Exception as e:
            logger.error(f"PageRank计算失败: {e}")
            n = similarity_matrix.shape[0]
            return np.ones(n) / n
    
    def _generate_cache_key(self, sources: List[ContentSource], strategy: str) -> str:
        """生成缓存键"""
        content_hash = hashlib.md5(
            (strategy + "".join(source.content for source in sources)).encode()
        ).hexdigest()
        return f"fusion_{content_hash[:16]}"
    
    def _cleanup_cache(self):
        """清理缓存"""
        try:
            max_size = self.config.CONTENT_FUSION_CONFIG["fusion_cache"]["max_size"]
            ttl = self.config.CONTENT_FUSION_CONFIG["fusion_cache"]["ttl"]
            
            if len(self.fusion_cache) > max_size:
                # 删除最旧的缓存项
                keys_to_remove = list(self.fusion_cache.keys())[:-max_size]
                for key in keys_to_remove:
                    del self.fusion_cache[key]
            
            # 删除过期项
            current_time = datetime.now()
            expired_keys = []
            
            for key, result in self.fusion_cache.items():
                if (current_time - result.created_at).total_seconds() > ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.fusion_cache[key]
                
        except Exception as e:
            logger.error(f"缓存清理失败: {e}")
    
    def get_fusion_stats(self) -> Dict[str, Any]:
        """获取融合统计信息"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        cache_hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            "cache_size": len(self.fusion_cache),
            "cache_hit_rate": cache_hit_rate,
            "total_requests": total_requests,
            "available_strategies": self.config.CONTENT_FUSION_CONFIG["fusion_strategies"]
        }

class QualityAssessor:
    """质量评估器"""
    
    def __init__(self):
        self.config = AdvancedRAGConfig()
    
    def assess_quality(self, source: ContentSource) -> float:
        """评估内容质量"""
        try:
            factors = self.config.CONTENT_FUSION_CONFIG["quality_assessment"]["factors"]
            weights = self.config.CONTENT_FUSION_CONFIG["quality_assessment"]["weights"]
            
            scores = []
            
            for factor in factors:
                if factor == "source_credibility":
                    scores.append(source.credibility)
                elif factor == "content_freshness":
                    scores.append(self._assess_freshness(source))
                elif factor == "information_completeness":
                    scores.append(self._assess_completeness(source))
                elif factor == "consistency_score":
                    scores.append(self._assess_consistency(source))
                else:
                    scores.append(0.5)  # 默认分数
            
            # 加权平均
            quality_score = sum(score * weight for score, weight in zip(scores, weights))
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.error(f"质量评估失败: {e}")
            return 0.5
    
    def _assess_freshness(self, source: ContentSource) -> float:
        """评估内容新鲜度"""
        try:
            age_hours = (datetime.now() - source.timestamp).total_seconds() / 3600
            
            # 24小时内为1.0，逐渐衰减
            if age_hours <= 24:
                return 1.0
            elif age_hours <= 168:  # 一周
                return 0.8
            elif age_hours <= 720:  # 一个月
                return 0.6
            else:
                return 0.4
                
        except Exception as e:
            logger.error(f"新鲜度评估失败: {e}")
            return 0.5
    
    def _assess_completeness(self, source: ContentSource) -> float:
        """评估信息完整性"""
        try:
            content_length = len(source.content)
            
            # 基于内容长度评估完整性
            if content_length < 50:
                return 0.3
            elif content_length < 200:
                return 0.6
            elif content_length < 500:
                return 0.8
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"完整性评估失败: {e}")
            return 0.5
    
    def _assess_consistency(self, source: ContentSource) -> float:
        """评估一致性"""
        try:
            # 简单的一致性检查
            content = source.content.lower()
            
            # 检查是否有矛盾词汇
            contradictions = ["但是", "然而", "不过", "相反", "however", "but", "nevertheless"]
            contradiction_count = sum(1 for word in contradictions if word in content)
            
            # 矛盾越少，一致性越高
            consistency = max(0.0, 1.0 - contradiction_count * 0.1)
            return consistency
            
        except Exception as e:
            logger.error(f"一致性评估失败: {e}")
            return 0.5

class ConflictResolver:
    """冲突解决器"""
    
    def __init__(self):
        self.config = AdvancedRAGConfig()
    
    def resolve_conflicts(self, sources: List[ContentSource]) -> List[ContentSource]:
        """解决内容冲突"""
        try:
            if len(sources) <= 1:
                return sources
            
            strategy = self.config.CONTENT_FUSION_CONFIG["conflict_resolution"]["strategy"]
            threshold = self.config.CONTENT_FUSION_CONFIG["conflict_resolution"]["threshold"]
            
            # 检测冲突
            conflicts = self._detect_conflicts(sources, threshold)
            
            if not conflicts:
                return sources
            
            # 解决冲突
            if strategy == "confidence_based":
                return self._resolve_by_confidence(sources, conflicts)
            elif strategy == "majority_vote":
                return self._resolve_by_majority(sources, conflicts)
            elif strategy == "latest":
                return self._resolve_by_timestamp(sources, conflicts)
            else:
                return sources
                
        except Exception as e:
            logger.error(f"冲突解决失败: {e}")
            return sources
    
    def _detect_conflicts(self, sources: List[ContentSource], threshold: float) -> List[Tuple[int, int]]:
        """检测冲突"""
        conflicts = []
        
        try:
            for i in range(len(sources)):
                for j in range(i + 1, len(sources)):
                    similarity = self._compute_content_similarity(
                        sources[i].content, 
                        sources[j].content
                    )
                    
                    # 相似度低于阈值认为是冲突
                    if similarity < threshold:
                        conflicts.append((i, j))
            
        except Exception as e:
            logger.error(f"冲突检测失败: {e}")
        
        return conflicts
    
    def _compute_content_similarity(self, content1: str, content2: str) -> float:
        """计算内容相似度"""
        try:
            words1 = set(content1.lower().split())
            words2 = set(content2.lower().split())
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"内容相似度计算失败: {e}")
            return 0.0
    
    def _resolve_by_confidence(self, sources: List[ContentSource], 
                              conflicts: List[Tuple[int, int]]) -> List[ContentSource]:
        """基于置信度解决冲突"""
        try:
            # 为每个冲突选择置信度更高的源
            to_remove = set()
            
            for i, j in conflicts:
                confidence_i = sources[i].credibility * sources[i].quality_score
                confidence_j = sources[j].credibility * sources[j].quality_score
                
                if confidence_i < confidence_j:
                    to_remove.add(i)
                else:
                    to_remove.add(j)
            
            # 返回未被移除的源
            return [source for idx, source in enumerate(sources) if idx not in to_remove]
            
        except Exception as e:
            logger.error(f"基于置信度的冲突解决失败: {e}")
            return sources
    
    def _resolve_by_majority(self, sources: List[ContentSource], 
                            conflicts: List[Tuple[int, int]]) -> List[ContentSource]:
        """基于多数投票解决冲突"""
        # 简化实现：返回原始源
        return sources
    
    def _resolve_by_timestamp(self, sources: List[ContentSource], 
                             conflicts: List[Tuple[int, int]]) -> List[ContentSource]:
        """基于时间戳解决冲突"""
        try:
            # 为每个冲突选择更新的源
            to_remove = set()
            
            for i, j in conflicts:
                if sources[i].timestamp < sources[j].timestamp:
                    to_remove.add(i)
                else:
                    to_remove.add(j)
            
            return [source for idx, source in enumerate(sources) if idx not in to_remove]
            
        except Exception as e:
            logger.error(f"基于时间戳的冲突解决失败: {e}")
            return sources
