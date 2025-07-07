"""
多模态推理引擎

负责基于知识图谱的推理、跨模态逻辑推理和路径查找
"""

import logging
import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime
from collections import defaultdict, deque

from .advanced_rag_config import AdvancedRAGConfig, EntityType, RelationType
from .knowledge_graph_builder import KnowledgeGraphBuilder

# 配置日志
logger = logging.getLogger(__name__)

class ReasoningPath:
    """推理路径类"""
    
    def __init__(self, nodes: List[str], edges: List[Dict[str, Any]], 
                 confidence: float, reasoning_type: str):
        self.nodes = nodes
        self.edges = edges
        self.confidence = confidence
        self.reasoning_type = reasoning_type
        self.length = len(nodes) - 1
        self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "confidence": self.confidence,
            "reasoning_type": self.reasoning_type,
            "length": self.length,
            "created_at": self.created_at.isoformat()
        }

class ReasoningResult:
    """推理结果类"""
    
    def __init__(self, query: str, answer: str, confidence: float,
                 paths: List[ReasoningPath], evidence: List[Dict[str, Any]] = None):
        self.query = query
        self.answer = answer
        self.confidence = confidence
        self.paths = paths
        self.evidence = evidence or []
        self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "query": self.query,
            "answer": self.answer,
            "confidence": self.confidence,
            "paths": [path.to_dict() for path in self.paths],
            "evidence": self.evidence,
            "created_at": self.created_at.isoformat()
        }

class MultimodalReasoningEngine:
    """多模态推理引擎类"""
    
    def __init__(self, knowledge_graph_builder: KnowledgeGraphBuilder = None):
        """
        初始化多模态推理引擎
        
        Args:
            knowledge_graph_builder: 知识图谱构建器
        """
        self.config = AdvancedRAGConfig()
        self.kg_builder = knowledge_graph_builder
        
        # 推理缓存
        self.reasoning_cache = {}
        self.cache_stats = {"hits": 0, "misses": 0}
        
        # 推理统计
        self.reasoning_stats = {
            "total_queries": 0,
            "successful_reasoning": 0,
            "failed_reasoning": 0,
            "average_confidence": 0.0
        }
        
        logger.info("多模态推理引擎初始化完成")
    
    def reason(self, query: str, start_entities: List[str] = None,
              reasoning_type: str = None, max_depth: int = None) -> ReasoningResult:
        """
        执行推理
        
        Args:
            query: 查询问题
            start_entities: 起始实体列表
            reasoning_type: 推理类型
            max_depth: 最大推理深度
            
        Returns:
            推理结果
        """
        try:
            self.reasoning_stats["total_queries"] += 1
            
            # 检查缓存
            cache_key = self._generate_cache_key(query, start_entities, reasoning_type)
            if cache_key in self.reasoning_cache:
                self.cache_stats["hits"] += 1
                return self.reasoning_cache[cache_key]
            
            self.cache_stats["misses"] += 1
            
            # 如果没有提供起始实体，从查询中提取
            if not start_entities:
                start_entities = self._extract_entities_from_query(query)
            
            if not start_entities:
                return ReasoningResult(query, "无法识别查询中的实体", 0.0, [])
            
            # 选择推理类型
            if not reasoning_type:
                reasoning_type = self.config.REASONING_CONFIG["default_reasoning"]
            
            # 设置最大深度
            if not max_depth:
                max_depth = self.config.REASONING_CONFIG["path_finding"]["max_depth"]
            
            # 执行推理
            if reasoning_type == "path_based":
                result = self._path_based_reasoning(query, start_entities, max_depth)
            elif reasoning_type == "subgraph_matching":
                result = self._subgraph_matching_reasoning(query, start_entities)
            elif reasoning_type == "random_walk":
                result = self._random_walk_reasoning(query, start_entities)
            else:
                result = self._simple_reasoning(query, start_entities)
            
            # 更新统计
            if result.confidence > 0.5:
                self.reasoning_stats["successful_reasoning"] += 1
            else:
                self.reasoning_stats["failed_reasoning"] += 1
            
            # 更新平均置信度
            total_successful = self.reasoning_stats["successful_reasoning"]
            if total_successful > 0:
                self.reasoning_stats["average_confidence"] = (
                    (self.reasoning_stats["average_confidence"] * (total_successful - 1) + result.confidence) 
                    / total_successful
                )
            
            # 缓存结果
            self.reasoning_cache[cache_key] = result
            self._cleanup_cache()
            
            return result
            
        except Exception as e:
            logger.error(f"推理失败: {e}")
            self.reasoning_stats["failed_reasoning"] += 1
            return ReasoningResult(query, f"推理过程中发生错误: {str(e)}", 0.0, [])
    
    def _path_based_reasoning(self, query: str, start_entities: List[str], 
                             max_depth: int) -> ReasoningResult:
        """基于路径的推理"""
        try:
            if not self.kg_builder or not self.kg_builder.graph:
                return ReasoningResult(query, "知识图谱不可用", 0.0, [])
            
            graph = self.kg_builder.graph
            all_paths = []
            
            # 为每个起始实体查找路径
            for start_entity in start_entities:
                if start_entity not in graph:
                    continue
                
                # 查找相关路径
                paths = self._find_reasoning_paths(graph, start_entity, max_depth)
                all_paths.extend(paths)
            
            if not all_paths:
                return ReasoningResult(query, "未找到相关推理路径", 0.1, [])
            
            # 路径排序和筛选
            all_paths.sort(key=lambda x: x.confidence, reverse=True)
            top_paths = all_paths[:self.config.REASONING_CONFIG["path_finding"]["max_paths"]]
            
            # 生成答案
            answer = self._generate_answer_from_paths(query, top_paths)
            
            # 计算整体置信度
            if top_paths:
                confidence = sum(path.confidence for path in top_paths) / len(top_paths)
            else:
                confidence = 0.0
            
            return ReasoningResult(query, answer, confidence, top_paths)
            
        except Exception as e:
            logger.error(f"基于路径的推理失败: {e}")
            return ReasoningResult(query, "路径推理失败", 0.0, [])
    
    def _find_reasoning_paths(self, graph: nx.Graph, start_node: str, 
                             max_depth: int) -> List[ReasoningPath]:
        """查找推理路径"""
        paths = []
        
        try:
            # 使用BFS查找路径
            queue = deque([(start_node, [start_node], [])])
            visited = set()
            
            while queue:
                current_node, path, edges = queue.popleft()
                
                if len(path) > max_depth:
                    continue
                
                if current_node in visited and len(path) > 1:
                    continue
                
                visited.add(current_node)
                
                # 如果路径长度大于1，创建推理路径
                if len(path) > 1:
                    confidence = self._calculate_path_confidence(path, edges, graph)
                    reasoning_path = ReasoningPath(path.copy(), edges.copy(), confidence, "path_based")
                    paths.append(reasoning_path)
                
                # 扩展路径
                if len(path) < max_depth:
                    for neighbor in graph.neighbors(current_node):
                        if neighbor not in path:  # 避免循环
                            edge_data = graph.get_edge_data(current_node, neighbor)
                            new_path = path + [neighbor]
                            new_edges = edges + [edge_data or {}]
                            queue.append((neighbor, new_path, new_edges))
            
        except Exception as e:
            logger.error(f"路径查找失败: {e}")
        
        return paths
    
    def _calculate_path_confidence(self, path: List[str], edges: List[Dict[str, Any]], 
                                  graph: nx.Graph) -> float:
        """计算路径置信度"""
        try:
            if len(path) <= 1:
                return 0.0
            
            factors = self.config.REASONING_CONFIG["confidence_calculation"]["factors"]
            weights = self.config.REASONING_CONFIG["confidence_calculation"]["weights"]
            
            scores = []
            
            # 路径长度因子（越短越好）
            if "path_length" in factors:
                length_score = 1.0 / len(path)
                scores.append(length_score)
            
            # 边权重因子
            if "edge_weights" in factors:
                edge_scores = []
                for edge in edges:
                    confidence = edge.get("confidence", 0.5)
                    edge_scores.append(confidence)
                
                edge_weight_score = sum(edge_scores) / len(edge_scores) if edge_scores else 0.5
                scores.append(edge_weight_score)
            
            # 节点重要性因子
            if "node_importance" in factors:
                node_scores = []
                for node in path:
                    # 使用节点度数作为重要性指标
                    degree = graph.degree(node)
                    importance = min(1.0, degree / 10.0)  # 归一化
                    node_scores.append(importance)
                
                node_importance_score = sum(node_scores) / len(node_scores) if node_scores else 0.5
                scores.append(node_importance_score)
            
            # 证据强度因子
            if "evidence_strength" in factors:
                # 简化的证据强度计算
                evidence_score = 0.7  # 默认值
                scores.append(evidence_score)
            
            # 加权平均
            if len(scores) == len(weights):
                confidence = sum(score * weight for score, weight in zip(scores, weights))
            else:
                confidence = sum(scores) / len(scores) if scores else 0.0
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"路径置信度计算失败: {e}")
            return 0.5
    
    def _subgraph_matching_reasoning(self, query: str, start_entities: List[str]) -> ReasoningResult:
        """基于子图匹配的推理"""
        try:
            if not self.kg_builder or not self.kg_builder.graph:
                return ReasoningResult(query, "知识图谱不可用", 0.0, [])
            
            # 简化实现：提取相关子图
            graph = self.kg_builder.graph
            subgraph_nodes = set()
            
            # 收集起始实体的邻居
            for entity in start_entities:
                if entity in graph:
                    subgraph_nodes.add(entity)
                    subgraph_nodes.update(graph.neighbors(entity))
            
            if not subgraph_nodes:
                return ReasoningResult(query, "未找到相关子图", 0.1, [])
            
            # 创建子图
            subgraph = graph.subgraph(subgraph_nodes)
            
            # 分析子图结构
            answer = self._analyze_subgraph(query, subgraph, start_entities)
            
            # 创建虚拟路径
            paths = [ReasoningPath(list(subgraph_nodes)[:5], [], 0.6, "subgraph_matching")]
            
            return ReasoningResult(query, answer, 0.6, paths)
            
        except Exception as e:
            logger.error(f"子图匹配推理失败: {e}")
            return ReasoningResult(query, "子图匹配失败", 0.0, [])
    
    def _random_walk_reasoning(self, query: str, start_entities: List[str]) -> ReasoningResult:
        """基于随机游走的推理"""
        try:
            if not self.kg_builder or not self.kg_builder.graph:
                return ReasoningResult(query, "知识图谱不可用", 0.0, [])
            
            graph = self.kg_builder.graph
            walk_config = self.config.REASONING_CONFIG["random_walk"]
            
            # 执行随机游走
            visited_nodes = defaultdict(int)
            
            for start_entity in start_entities:
                if start_entity not in graph:
                    continue
                
                for _ in range(walk_config["num_walks"]):
                    path = self._random_walk(graph, start_entity, walk_config)
                    for node in path:
                        visited_nodes[node] += 1
            
            if not visited_nodes:
                return ReasoningResult(query, "随机游走未发现相关信息", 0.1, [])
            
            # 按访问频率排序
            sorted_nodes = sorted(visited_nodes.items(), key=lambda x: x[1], reverse=True)
            
            # 生成答案
            top_nodes = [node for node, count in sorted_nodes[:5]]
            answer = f"通过随机游走发现的相关实体: {', '.join(top_nodes)}"
            
            # 创建虚拟路径
            paths = [ReasoningPath(top_nodes, [], 0.5, "random_walk")]
            
            return ReasoningResult(query, answer, 0.5, paths)
            
        except Exception as e:
            logger.error(f"随机游走推理失败: {e}")
            return ReasoningResult(query, "随机游走失败", 0.0, [])
    
    def _random_walk(self, graph: nx.Graph, start_node: str, config: Dict[str, Any]) -> List[str]:
        """执行单次随机游走"""
        path = [start_node]
        current_node = start_node
        
        for _ in range(config["walk_length"]):
            neighbors = list(graph.neighbors(current_node))
            
            if not neighbors:
                break
            
            # 重启概率
            if np.random.random() < config["restart_probability"]:
                current_node = start_node
            else:
                # 随机选择邻居
                current_node = np.random.choice(neighbors)
            
            path.append(current_node)
        
        return path
    
    def _simple_reasoning(self, query: str, start_entities: List[str]) -> ReasoningResult:
        """简单推理"""
        try:
            if not self.kg_builder:
                return ReasoningResult(query, "知识图谱构建器不可用", 0.0, [])
            
            # 查找实体的直接邻居
            related_entities = set()
            
            for entity in start_entities:
                neighbors = self.kg_builder.get_entity_neighbors(entity, max_depth=1)
                related_entities.update(neighbors)
            
            if related_entities:
                answer = f"与查询相关的实体: {', '.join(list(related_entities)[:5])}"
                confidence = 0.4
            else:
                answer = "未找到相关信息"
                confidence = 0.1
            
            # 创建简单路径
            paths = [ReasoningPath(start_entities + list(related_entities)[:3], [], confidence, "simple")]
            
            return ReasoningResult(query, answer, confidence, paths)
            
        except Exception as e:
            logger.error(f"简单推理失败: {e}")
            return ReasoningResult(query, "推理失败", 0.0, [])
    
    def _extract_entities_from_query(self, query: str) -> List[str]:
        """从查询中提取实体"""
        try:
            if not self.kg_builder:
                return []
            
            # 使用知识图谱构建器提取实体
            entities = self.kg_builder.extract_entities(query)
            return [entity.name for entity in entities]
            
        except Exception as e:
            logger.error(f"查询实体提取失败: {e}")
            return []
    
    def _generate_answer_from_paths(self, query: str, paths: List[ReasoningPath]) -> str:
        """从路径生成答案"""
        try:
            if not paths:
                return "未找到相关推理路径"
            
            # 收集路径中的关键信息
            key_entities = set()
            key_relations = set()
            
            for path in paths[:3]:  # 只考虑前3条路径
                key_entities.update(path.nodes)
                for edge in path.edges:
                    if "type" in edge:
                        key_relations.add(edge["type"])
            
            # 生成答案
            answer_parts = []
            
            if key_entities:
                entities_str = ", ".join(list(key_entities)[:5])
                answer_parts.append(f"相关实体: {entities_str}")
            
            if key_relations:
                relations_str = ", ".join(list(key_relations)[:3])
                answer_parts.append(f"相关关系: {relations_str}")
            
            if answer_parts:
                return "; ".join(answer_parts)
            else:
                return "基于推理路径未能生成具体答案"
                
        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            return "答案生成过程中发生错误"
    
    def _analyze_subgraph(self, query: str, subgraph: nx.Graph, start_entities: List[str]) -> str:
        """分析子图"""
        try:
            # 基本子图统计
            num_nodes = subgraph.number_of_nodes()
            num_edges = subgraph.number_of_edges()
            
            # 查找中心节点
            centrality = nx.degree_centrality(subgraph)
            central_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]
            
            answer_parts = [
                f"子图包含 {num_nodes} 个节点和 {num_edges} 条边"
            ]
            
            if central_nodes:
                central_entities = [node for node, _ in central_nodes]
                answer_parts.append(f"中心实体: {', '.join(central_entities)}")
            
            return "; ".join(answer_parts)
            
        except Exception as e:
            logger.error(f"子图分析失败: {e}")
            return "子图分析失败"
    
    def _generate_cache_key(self, query: str, start_entities: List[str], 
                           reasoning_type: str) -> str:
        """生成缓存键"""
        import hashlib
        content = f"{query}_{start_entities}_{reasoning_type}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _cleanup_cache(self):
        """清理缓存"""
        try:
            max_size = 100  # 最大缓存大小
            
            if len(self.reasoning_cache) > max_size:
                # 删除最旧的缓存项
                keys_to_remove = list(self.reasoning_cache.keys())[:-max_size]
                for key in keys_to_remove:
                    del self.reasoning_cache[key]
                
        except Exception as e:
            logger.error(f"缓存清理失败: {e}")
    
    def get_reasoning_stats(self) -> Dict[str, Any]:
        """获取推理统计信息"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        cache_hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            "cache_size": len(self.reasoning_cache),
            "cache_hit_rate": cache_hit_rate,
            "total_requests": total_requests,
            "reasoning_stats": self.reasoning_stats,
            "available_reasoning_types": self.config.REASONING_CONFIG["reasoning_types"]
        }
    
    def explain_reasoning(self, result: ReasoningResult) -> Dict[str, Any]:
        """解释推理过程"""
        try:
            explanation = {
                "query": result.query,
                "reasoning_steps": [],
                "confidence_breakdown": {},
                "evidence_summary": []
            }
            
            # 分析推理路径
            for i, path in enumerate(result.paths[:3]):
                step = {
                    "step": i + 1,
                    "path": " -> ".join(path.nodes),
                    "confidence": path.confidence,
                    "reasoning_type": path.reasoning_type
                }
                explanation["reasoning_steps"].append(step)
            
            # 置信度分解
            if result.paths:
                avg_confidence = sum(path.confidence for path in result.paths) / len(result.paths)
                explanation["confidence_breakdown"] = {
                    "overall_confidence": result.confidence,
                    "average_path_confidence": avg_confidence,
                    "path_count": len(result.paths)
                }
            
            # 证据摘要
            explanation["evidence_summary"] = result.evidence
            
            return explanation
            
        except Exception as e:
            logger.error(f"推理解释失败: {e}")
            return {"error": "推理解释生成失败"}
    
    def clear_cache(self):
        """清空缓存"""
        self.reasoning_cache.clear()
        self.cache_stats = {"hits": 0, "misses": 0}
        logger.info("推理缓存已清空")
