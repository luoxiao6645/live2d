"""
图查询处理器

负责自然语言到图查询的转换、复杂查询的分解和优化
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from collections import defaultdict

from .advanced_rag_config import AdvancedRAGConfig, EntityType, RelationType
from .knowledge_graph_builder import KnowledgeGraphBuilder
from .multimodal_reasoning_engine import MultimodalReasoningEngine

# 配置日志
logger = logging.getLogger(__name__)

class QueryIntent:
    """查询意图类"""

    def __init__(self, intent_type: str, entities: List[str], relations: List[str],
                 confidence: float, parameters: Dict[str, Any] = None):
        self.intent_type = intent_type
        self.entities = entities
        self.relations = relations
        self.confidence = confidence
        self.parameters = parameters or {}
        self.created_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "intent_type": self.intent_type,
            "entities": self.entities,
            "relations": self.relations,
            "confidence": self.confidence,
            "parameters": self.parameters,
            "created_at": self.created_at.isoformat()
        }

class GraphQuery:
    """图查询类"""

    def __init__(self, query_type: str, source_entities: List[str],
                 target_entities: List[str] = None, relations: List[str] = None,
                 constraints: Dict[str, Any] = None):
        self.query_type = query_type
        self.source_entities = source_entities
        self.target_entities = target_entities or []
        self.relations = relations or []
        self.constraints = constraints or {}
        self.created_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "query_type": self.query_type,
            "source_entities": self.source_entities,
            "target_entities": self.target_entities,
            "relations": self.relations,
            "constraints": self.constraints,
            "created_at": self.created_at.isoformat()
        }

class QueryResult:
    """查询结果类"""

    def __init__(self, query: str, results: List[Dict[str, Any]],
                 confidence: float, query_time: float):
        self.query = query
        self.results = results
        self.confidence = confidence
        self.query_time = query_time
        self.result_count = len(results)
        self.created_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "query": self.query,
            "results": self.results,
            "confidence": self.confidence,
            "query_time": self.query_time,
            "result_count": self.result_count,
            "created_at": self.created_at.isoformat()
        }

class GraphQueryProcessor:
    """图查询处理器类"""

    def __init__(self, kg_builder: KnowledgeGraphBuilder = None,
                 reasoning_engine: MultimodalReasoningEngine = None):
        """
        初始化图查询处理器

        Args:
            kg_builder: 知识图谱构建器
            reasoning_engine: 推理引擎
        """
        self.config = AdvancedRAGConfig()
        self.kg_builder = kg_builder
        self.reasoning_engine = reasoning_engine

        # 查询模式
        self.query_patterns = self._init_query_patterns()

        # 查询缓存
        self.query_cache = {}
        self.cache_stats = {"hits": 0, "misses": 0}

        # 查询统计
        self.query_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "average_query_time": 0.0
        }

        logger.info("图查询处理器初始化完成")

    def _init_query_patterns(self) -> Dict[str, List[str]]:
        """初始化查询模式"""
        return {
            "find_entity": [
                r"什么是\s*(.+)",
                r"(.+)\s*是什么",
                r"告诉我关于\s*(.+)\s*的信息",
                r"(.+)\s*的定义",
                r"what is\s*(.+)",
                r"tell me about\s*(.+)"
            ],
            "find_relation": [
                r"(.+)\s*和\s*(.+)\s*的关系",
                r"(.+)\s*与\s*(.+)\s*有什么关系",
                r"(.+)\s*如何影响\s*(.+)",
                r"relationship between\s*(.+)\s*and\s*(.+)",
                r"how does\s*(.+)\s*affect\s*(.+)"
            ],
            "find_path": [
                r"从\s*(.+)\s*到\s*(.+)\s*的路径",
                r"(.+)\s*如何连接到\s*(.+)",
                r"(.+)\s*和\s*(.+)\s*之间的联系",
                r"path from\s*(.+)\s*to\s*(.+)",
                r"connection between\s*(.+)\s*and\s*(.+)"
            ],
            "find_neighbors": [
                r"(.+)\s*的相关实体",
                r"与\s*(.+)\s*相关的",
                r"(.+)\s*的邻居",
                r"related to\s*(.+)",
                r"neighbors of\s*(.+)"
            ],
            "count_entities": [
                r"有多少\s*(.+)",
                r"(.+)\s*的数量",
                r"统计\s*(.+)",
                r"how many\s*(.+)",
                r"count of\s*(.+)"
            ]
        }

    def process_query(self, query: str) -> QueryResult:
        """
        处理自然语言查询

        Args:
            query: 自然语言查询

        Returns:
            查询结果
        """
        start_time = datetime.now()

        try:
            self.query_stats["total_queries"] += 1

            # 检查缓存
            cache_key = self._generate_cache_key(query)
            if cache_key in self.query_cache:
                self.cache_stats["hits"] += 1
                cached_result = self.query_cache[cache_key]
                return cached_result

            self.cache_stats["misses"] += 1

            # 查询理解
            intent = self._understand_query(query)

            if intent.confidence < 0.3:
                return QueryResult(query, [], 0.0, 0.0)

            # 转换为图查询
            graph_query = self._convert_to_graph_query(intent)

            # 执行查询
            results = self._execute_graph_query(graph_query)

            # 结果排序和过滤
            filtered_results = self._filter_and_rank_results(results, query)

            # 计算查询时间
            query_time = (datetime.now() - start_time).total_seconds()

            # 创建结果
            result = QueryResult(query, filtered_results, intent.confidence, query_time)

            # 更新统计
            if filtered_results:
                self.query_stats["successful_queries"] += 1
            else:
                self.query_stats["failed_queries"] += 1

            # 更新平均查询时间
            total_queries = self.query_stats["total_queries"]
            self.query_stats["average_query_time"] = (
                (self.query_stats["average_query_time"] * (total_queries - 1) + query_time)
                / total_queries
            )

            # 缓存结果
            self.query_cache[cache_key] = result
            self._cleanup_cache()

            return result

        except Exception as e:
            logger.error(f"查询处理失败: {e}")
            query_time = (datetime.now() - start_time).total_seconds()
            self.query_stats["failed_queries"] += 1
            return QueryResult(query, [], 0.0, query_time)

    def _understand_query(self, query: str) -> QueryIntent:
        """理解查询意图"""
        try:
            best_intent = None
            best_confidence = 0.0

            # 匹配查询模式
            for intent_type, patterns in self.query_patterns.items():
                for pattern in patterns:
                    match = re.search(pattern, query, re.IGNORECASE)

                    if match:
                        entities = [group.strip() for group in match.groups() if group]
                        confidence = 0.8  # 基础置信度

                        # 验证实体是否存在于知识图谱中
                        if self.kg_builder:
                            valid_entities = []
                            for entity in entities:
                                if self.kg_builder.find_entity(entity):
                                    valid_entities.append(entity)
                                    confidence += 0.1
                            entities = valid_entities

                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_intent = QueryIntent(
                                intent_type=intent_type,
                                entities=entities,
                                relations=[],
                                confidence=min(1.0, confidence)
                            )

            # 如果没有匹配的模式，尝试实体提取
            if not best_intent:
                entities = self._extract_entities_from_query(query)
                if entities:
                    best_intent = QueryIntent(
                        intent_type="find_entity",
                        entities=entities,
                        relations=[],
                        confidence=0.5
                    )

            return best_intent or QueryIntent("unknown", [], [], 0.0)

        except Exception as e:
            logger.error(f"查询理解失败: {e}")
            return QueryIntent("unknown", [], [], 0.0)

    def _convert_to_graph_query(self, intent: QueryIntent) -> GraphQuery:
        """将意图转换为图查询"""
        try:
            if intent.intent_type == "find_entity":
                return GraphQuery(
                    query_type="node_lookup",
                    source_entities=intent.entities
                )

            elif intent.intent_type == "find_relation":
                if len(intent.entities) >= 2:
                    return GraphQuery(
                        query_type="edge_lookup",
                        source_entities=[intent.entities[0]],
                        target_entities=[intent.entities[1]]
                    )
                else:
                    return GraphQuery(
                        query_type="node_lookup",
                        source_entities=intent.entities
                    )

            elif intent.intent_type == "find_path":
                if len(intent.entities) >= 2:
                    return GraphQuery(
                        query_type="path_finding",
                        source_entities=[intent.entities[0]],
                        target_entities=[intent.entities[1]]
                    )
                else:
                    return GraphQuery(
                        query_type="node_lookup",
                        source_entities=intent.entities
                    )

            elif intent.intent_type == "find_neighbors":
                return GraphQuery(
                    query_type="neighbor_lookup",
                    source_entities=intent.entities
                )

            elif intent.intent_type == "count_entities":
                return GraphQuery(
                    query_type="count",
                    source_entities=intent.entities
                )

            else:
                return GraphQuery(
                    query_type="general",
                    source_entities=intent.entities
                )

        except Exception as e:
            logger.error(f"图查询转换失败: {e}")
            return GraphQuery("general", intent.entities)

    def _execute_graph_query(self, graph_query: GraphQuery) -> List[Dict[str, Any]]:
        """执行图查询"""
        try:
            if not self.kg_builder or not self.kg_builder.graph:
                return []

            graph = self.kg_builder.graph
            results = []

            if graph_query.query_type == "node_lookup":
                results = self._execute_node_lookup(graph, graph_query)

            elif graph_query.query_type == "edge_lookup":
                results = self._execute_edge_lookup(graph, graph_query)

            elif graph_query.query_type == "path_finding":
                results = self._execute_path_finding(graph, graph_query)

            elif graph_query.query_type == "neighbor_lookup":
                results = self._execute_neighbor_lookup(graph, graph_query)

            elif graph_query.query_type == "count":
                results = self._execute_count_query(graph, graph_query)

            else:
                results = self._execute_general_query(graph, graph_query)

            return results

        except Exception as e:
            logger.error(f"图查询执行失败: {e}")
            return []

    def _execute_node_lookup(self, graph, graph_query: GraphQuery) -> List[Dict[str, Any]]:
        """执行节点查找"""
        results = []

        for entity in graph_query.source_entities:
            if entity in graph:
                node_data = graph.nodes[entity]
                result = {
                    "type": "entity",
                    "entity": entity,
                    "data": node_data,
                    "relevance": 1.0
                }
                results.append(result)

        return results

    def _execute_edge_lookup(self, graph, graph_query: GraphQuery) -> List[Dict[str, Any]]:
        """执行边查找"""
        results = []

        for source in graph_query.source_entities:
            for target in graph_query.target_entities:
                if graph.has_edge(source, target):
                    edge_data = graph.get_edge_data(source, target)
                    result = {
                        "type": "relation",
                        "source": source,
                        "target": target,
                        "data": edge_data,
                        "relevance": 1.0
                    }
                    results.append(result)

        return results

    def _execute_path_finding(self, graph, graph_query: GraphQuery) -> List[Dict[str, Any]]:
        """执行路径查找"""
        results = []

        try:
            import networkx as nx

            for source in graph_query.source_entities:
                for target in graph_query.target_entities:
                    if source in graph and target in graph:
                        try:
                            # 查找最短路径
                            path = nx.shortest_path(graph, source, target)

                            result = {
                                "type": "path",
                                "source": source,
                                "target": target,
                                "path": path,
                                "length": len(path) - 1,
                                "relevance": 1.0 / len(path)
                            }
                            results.append(result)

                        except nx.NetworkXNoPath:
                            # 没有路径
                            result = {
                                "type": "no_path",
                                "source": source,
                                "target": target,
                                "message": "未找到连接路径",
                                "relevance": 0.0
                            }
                            results.append(result)

        except Exception as e:
            logger.error(f"路径查找失败: {e}")

        return results

    def _execute_neighbor_lookup(self, graph, graph_query: GraphQuery) -> List[Dict[str, Any]]:
        """执行邻居查找"""
        results = []

        for entity in graph_query.source_entities:
            if entity in graph:
                neighbors = list(graph.neighbors(entity))

                for neighbor in neighbors:
                    edge_data = graph.get_edge_data(entity, neighbor)
                    result = {
                        "type": "neighbor",
                        "source": entity,
                        "neighbor": neighbor,
                        "edge_data": edge_data,
                        "relevance": 0.8
                    }
                    results.append(result)

        return results

    def _execute_count_query(self, graph, graph_query: GraphQuery) -> List[Dict[str, Any]]:
        """执行计数查询"""
        results = []

        # 统计节点数量
        node_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()

        result = {
            "type": "count",
            "node_count": node_count,
            "edge_count": edge_count,
            "relevance": 1.0
        }
        results.append(result)

        return results

    def _execute_general_query(self, graph, graph_query: GraphQuery) -> List[Dict[str, Any]]:
        """执行通用查询"""
        results = []

        # 使用推理引擎进行通用查询
        if self.reasoning_engine:
            reasoning_result = self.reasoning_engine.reason(
                query="通用查询",
                start_entities=graph_query.source_entities
            )

            result = {
                "type": "reasoning",
                "answer": reasoning_result.answer,
                "confidence": reasoning_result.confidence,
                "paths": [path.to_dict() for path in reasoning_result.paths],
                "relevance": reasoning_result.confidence
            }
            results.append(result)

        return results

    def _filter_and_rank_results(self, results: List[Dict[str, Any]],
                                query: str) -> List[Dict[str, Any]]:
        """过滤和排序结果"""
        try:
            config = self.config.QUERY_PROCESSING_CONFIG["result_filtering"]

            # 过滤低置信度结果
            min_confidence = config["min_confidence"]
            filtered_results = [
                result for result in results
                if result.get("relevance", 0) >= min_confidence
            ]

            # 按相关性排序
            filtered_results.sort(key=lambda x: x.get("relevance", 0), reverse=True)

            # 限制结果数量
            max_results = config["max_results"]
            filtered_results = filtered_results[:max_results]

            return filtered_results

        except Exception as e:
            logger.error(f"结果过滤排序失败: {e}")
            return results

    def _extract_entities_from_query(self, query: str) -> List[str]:
        """从查询中提取实体"""
        try:
            if self.kg_builder:
                entities = self.kg_builder.extract_entities(query)
                return [entity.name for entity in entities]
            else:
                return []

        except Exception as e:
            logger.error(f"实体提取失败: {e}")
            return []

    def _generate_cache_key(self, query: str) -> str:
        """生成缓存键"""
        import hashlib
        return hashlib.md5(query.encode()).hexdigest()[:16]

    def _cleanup_cache(self):
        """清理缓存"""
        try:
            max_size = 100  # 最大缓存大小

            if len(self.query_cache) > max_size:
                # 删除最旧的缓存项
                keys_to_remove = list(self.query_cache.keys())[:-max_size]
                for key in keys_to_remove:
                    del self.query_cache[key]

        except Exception as e:
            logger.error(f"缓存清理失败: {e}")

    def get_query_stats(self) -> Dict[str, Any]:
        """获取查询统计信息"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        cache_hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0

        return {
            "cache_size": len(self.query_cache),
            "cache_hit_rate": cache_hit_rate,
            "total_requests": total_requests,
            "query_stats": self.query_stats,
            "supported_query_types": list(self.query_patterns.keys())
        }

    def clear_cache(self):
        """清空缓存"""
        self.query_cache.clear()
        self.cache_stats = {"hits": 0, "misses": 0}
        logger.info("查询缓存已清空")