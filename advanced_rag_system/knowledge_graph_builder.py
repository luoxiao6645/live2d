"""
知识图谱构建器

负责从文档中提取实体和关系，构建动态的知识图谱
"""

import logging
import re
import networkx as nx
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime
import json
import pickle

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy不可用，实体识别功能将受限")

from .advanced_rag_config import AdvancedRAGConfig, EntityType, RelationType, GraphType

# 配置日志
logger = logging.getLogger(__name__)

class Entity:
    """实体类"""
    
    def __init__(self, name: str, entity_type: EntityType, confidence: float = 1.0,
                 attributes: Dict[str, Any] = None):
        self.name = name
        self.entity_type = entity_type
        self.confidence = confidence
        self.attributes = attributes or {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "type": self.entity_type.value,
            "confidence": self.confidence,
            "attributes": self.attributes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

class Relation:
    """关系类"""
    
    def __init__(self, source: str, target: str, relation_type: RelationType,
                 confidence: float = 1.0, attributes: Dict[str, Any] = None):
        self.source = source
        self.target = target
        self.relation_type = relation_type
        self.confidence = confidence
        self.attributes = attributes or {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "source": self.source,
            "target": self.target,
            "type": self.relation_type.value,
            "confidence": self.confidence,
            "attributes": self.attributes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

class KnowledgeGraphBuilder:
    """知识图谱构建器类"""
    
    def __init__(self):
        """初始化知识图谱构建器"""
        self.config = AdvancedRAGConfig()
        
        # 创建图结构
        if self.config.KNOWLEDGE_GRAPH_CONFIG["graph_type"] == GraphType.DIRECTED:
            self.graph = nx.DiGraph()
        elif self.config.KNOWLEDGE_GRAPH_CONFIG["graph_type"] == GraphType.MULTIGRAPH:
            self.graph = nx.MultiDiGraph()
        else:
            self.graph = nx.Graph()
        
        # NLP模型
        self.nlp_model = None
        
        # 实体和关系缓存
        self.entities = {}  # name -> Entity
        self.relations = []  # List[Relation]
        
        # 统计信息
        self.stats = {
            "entities_extracted": 0,
            "relations_extracted": 0,
            "documents_processed": 0
        }
        
        # 初始化NLP模型
        self._init_nlp_model()
        
        logger.info("知识图谱构建器初始化完成")
    
    def _init_nlp_model(self):
        """初始化NLP模型"""
        if not SPACY_AVAILABLE:
            logger.warning("spaCy不可用，使用基于规则的实体识别")
            return
        
        try:
            model_name = self.config.KNOWLEDGE_GRAPH_CONFIG["entity_extraction"]["model"]
            self.nlp_model = spacy.load(model_name)
            logger.info(f"spaCy模型加载成功: {model_name}")
        except Exception as e:
            logger.warning(f"spaCy模型加载失败: {e}，使用基于规则的实体识别")
            self.nlp_model = None
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        从文本中提取实体
        
        Args:
            text: 输入文本
            
        Returns:
            提取的实体列表
        """
        try:
            entities = []
            
            if self.nlp_model:
                # 使用spaCy进行实体识别
                entities.extend(self._extract_entities_with_spacy(text))
            
            # 使用基于规则的实体识别
            entities.extend(self._extract_entities_with_rules(text))
            
            # 去重和过滤
            entities = self._deduplicate_entities(entities)
            entities = self._filter_entities(entities)
            
            return entities
            
        except Exception as e:
            logger.error(f"实体提取失败: {e}")
            return []
    
    def _extract_entities_with_spacy(self, text: str) -> List[Entity]:
        """使用spaCy提取实体"""
        entities = []
        
        try:
            doc = self.nlp_model(text)
            
            for ent in doc.ents:
                # 映射spaCy实体类型到自定义类型
                entity_type = self._map_spacy_entity_type(ent.label_)
                
                if entity_type:
                    entity = Entity(
                        name=ent.text.strip(),
                        entity_type=entity_type,
                        confidence=0.8,  # spaCy实体的默认置信度
                        attributes={
                            "start": ent.start_char,
                            "end": ent.end_char,
                            "label": ent.label_
                        }
                    )
                    entities.append(entity)
            
        except Exception as e:
            logger.error(f"spaCy实体提取失败: {e}")
        
        return entities
    
    def _extract_entities_with_rules(self, text: str) -> List[Entity]:
        """使用规则提取实体"""
        entities = []
        
        try:
            for entity_type, patterns in self.config.ENTITY_PATTERNS.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text)
                    
                    for match in matches:
                        entity_name = match.group().strip()
                        
                        if len(entity_name) > 1:  # 过滤太短的实体
                            entity = Entity(
                                name=entity_name,
                                entity_type=entity_type,
                                confidence=0.6,  # 规则实体的默认置信度
                                attributes={
                                    "start": match.start(),
                                    "end": match.end(),
                                    "pattern": pattern
                                }
                            )
                            entities.append(entity)
            
        except Exception as e:
            logger.error(f"规则实体提取失败: {e}")
        
        return entities
    
    def _map_spacy_entity_type(self, spacy_label: str) -> Optional[EntityType]:
        """映射spaCy实体类型"""
        mapping = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,
            "LOC": EntityType.LOCATION,
            "EVENT": EntityType.EVENT,
            "DATE": EntityType.TIME,
            "TIME": EntityType.TIME
        }
        return mapping.get(spacy_label)
    
    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """
        从文本中提取关系
        
        Args:
            text: 输入文本
            entities: 已提取的实体列表
            
        Returns:
            提取的关系列表
        """
        try:
            relations = []
            
            # 使用规则提取关系
            relations.extend(self._extract_relations_with_rules(text, entities))
            
            # 使用依存句法分析提取关系（如果spaCy可用）
            if self.nlp_model:
                relations.extend(self._extract_relations_with_dependency(text, entities))
            
            # 过滤和去重
            relations = self._filter_relations(relations)
            
            return relations
            
        except Exception as e:
            logger.error(f"关系提取失败: {e}")
            return []
    
    def _extract_relations_with_rules(self, text: str, entities: List[Entity]) -> List[Relation]:
        """使用规则提取关系"""
        relations = []
        
        try:
            entity_names = [entity.name for entity in entities]
            
            for relation_type, patterns in self.config.RELATION_PATTERNS.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text)
                    
                    for match in matches:
                        groups = match.groups()
                        if len(groups) >= 2:
                            source = groups[0].strip()
                            target = groups[1].strip()
                            
                            # 检查是否为已识别的实体
                            if source in entity_names and target in entity_names:
                                relation = Relation(
                                    source=source,
                                    target=target,
                                    relation_type=relation_type,
                                    confidence=0.7,
                                    attributes={
                                        "pattern": pattern,
                                        "match_text": match.group()
                                    }
                                )
                                relations.append(relation)
            
        except Exception as e:
            logger.error(f"规则关系提取失败: {e}")
        
        return relations
    
    def _extract_relations_with_dependency(self, text: str, entities: List[Entity]) -> List[Relation]:
        """使用依存句法分析提取关系"""
        relations = []
        
        try:
            doc = self.nlp_model(text)
            entity_names = [entity.name for entity in entities]
            
            for token in doc:
                # 查找主谓宾结构
                if token.dep_ in ["nsubj", "dobj", "pobj"]:
                    head = token.head
                    
                    # 检查是否为实体
                    if token.text in entity_names and head.text in entity_names:
                        # 根据依存关系确定关系类型
                        relation_type = self._map_dependency_to_relation(token.dep_, head.pos_)
                        
                        if relation_type:
                            relation = Relation(
                                source=token.text,
                                target=head.text,
                                relation_type=relation_type,
                                confidence=0.6,
                                attributes={
                                    "dependency": token.dep_,
                                    "head_pos": head.pos_
                                }
                            )
                            relations.append(relation)
            
        except Exception as e:
            logger.error(f"依存关系提取失败: {e}")
        
        return relations
    
    def _map_dependency_to_relation(self, dep: str, head_pos: str) -> Optional[RelationType]:
        """映射依存关系到关系类型"""
        mapping = {
            ("nsubj", "VERB"): RelationType.RELATED_TO,
            ("dobj", "VERB"): RelationType.RELATED_TO,
            ("pobj", "ADP"): RelationType.RELATED_TO
        }
        return mapping.get((dep, head_pos))
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """实体去重"""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            key = (entity.name.lower(), entity.entity_type)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _filter_entities(self, entities: List[Entity]) -> List[Entity]:
        """过滤实体"""
        filtered = []
        
        confidence_threshold = self.config.KNOWLEDGE_GRAPH_CONFIG["entity_extraction"]["confidence_threshold"]
        max_entities = self.config.KNOWLEDGE_GRAPH_CONFIG["entity_extraction"]["max_entities_per_doc"]
        
        for entity in entities:
            # 置信度过滤
            if entity.confidence < confidence_threshold:
                continue
            
            # 长度过滤
            if len(entity.name) < 2 or len(entity.name) > 100:
                continue
            
            filtered.append(entity)
        
        # 限制数量
        if len(filtered) > max_entities:
            # 按置信度排序，取前N个
            filtered.sort(key=lambda x: x.confidence, reverse=True)
            filtered = filtered[:max_entities]
        
        return filtered
    
    def _filter_relations(self, relations: List[Relation]) -> List[Relation]:
        """过滤关系"""
        filtered = []
        
        confidence_threshold = self.config.KNOWLEDGE_GRAPH_CONFIG["relation_extraction"]["confidence_threshold"]
        
        for relation in relations:
            # 置信度过滤
            if relation.confidence < confidence_threshold:
                continue
            
            # 自环过滤
            if relation.source == relation.target:
                continue
            
            filtered.append(relation)
        
        return filtered
    
    def build_graph_from_document(self, text: str, doc_id: str = None) -> Dict[str, Any]:
        """
        从文档构建知识图谱
        
        Args:
            text: 文档文本
            doc_id: 文档ID
            
        Returns:
            构建结果
        """
        try:
            # 提取实体
            entities = self.extract_entities(text)
            
            # 提取关系
            relations = self.extract_relations(text, entities)
            
            # 添加到图中
            added_entities = 0
            added_relations = 0
            
            # 添加实体节点
            for entity in entities:
                if entity.name not in self.entities:
                    self.entities[entity.name] = entity
                    self.graph.add_node(entity.name, **entity.to_dict())
                    added_entities += 1
                else:
                    # 更新现有实体
                    self._update_entity(entity.name, entity)
            
            # 添加关系边
            for relation in relations:
                if self.graph.has_node(relation.source) and self.graph.has_node(relation.target):
                    self.graph.add_edge(
                        relation.source,
                        relation.target,
                        **relation.to_dict()
                    )
                    self.relations.append(relation)
                    added_relations += 1
            
            # 更新统计信息
            self.stats["entities_extracted"] += added_entities
            self.stats["relations_extracted"] += added_relations
            self.stats["documents_processed"] += 1
            
            result = {
                "success": True,
                "doc_id": doc_id,
                "entities_found": len(entities),
                "relations_found": len(relations),
                "entities_added": added_entities,
                "relations_added": added_relations,
                "graph_stats": self.get_graph_stats()
            }
            
            logger.info(f"文档图谱构建完成: {added_entities}个实体, {added_relations}个关系")
            return result
            
        except Exception as e:
            logger.error(f"文档图谱构建失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _update_entity(self, entity_name: str, new_entity: Entity):
        """更新实体信息"""
        existing_entity = self.entities[entity_name]
        
        # 更新置信度（取最大值）
        if new_entity.confidence > existing_entity.confidence:
            existing_entity.confidence = new_entity.confidence
        
        # 合并属性
        existing_entity.attributes.update(new_entity.attributes)
        existing_entity.updated_at = datetime.now()
        
        # 更新图节点
        self.graph.nodes[entity_name].update(existing_entity.to_dict())
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "is_connected": nx.is_connected(self.graph.to_undirected()) if self.graph.number_of_nodes() > 0 else False,
            "total_entities_extracted": self.stats["entities_extracted"],
            "total_relations_extracted": self.stats["relations_extracted"],
            "documents_processed": self.stats["documents_processed"]
        }
    
    def find_entity(self, entity_name: str) -> Optional[Entity]:
        """查找实体"""
        return self.entities.get(entity_name)
    
    def find_relations(self, source: str = None, target: str = None,
                      relation_type: RelationType = None) -> List[Relation]:
        """查找关系"""
        filtered_relations = []
        
        for relation in self.relations:
            if source and relation.source != source:
                continue
            if target and relation.target != target:
                continue
            if relation_type and relation.relation_type != relation_type:
                continue
            
            filtered_relations.append(relation)
        
        return filtered_relations
    
    def get_entity_neighbors(self, entity_name: str, max_depth: int = 1) -> List[str]:
        """获取实体的邻居"""
        try:
            if entity_name not in self.graph:
                return []
            
            if max_depth == 1:
                return list(self.graph.neighbors(entity_name))
            else:
                # 使用BFS获取多层邻居
                visited = set()
                queue = [(entity_name, 0)]
                neighbors = []
                
                while queue:
                    node, depth = queue.pop(0)
                    
                    if depth >= max_depth:
                        continue
                    
                    if node in visited:
                        continue
                    
                    visited.add(node)
                    
                    if depth > 0:  # 不包括起始节点
                        neighbors.append(node)
                    
                    for neighbor in self.graph.neighbors(node):
                        if neighbor not in visited:
                            queue.append((neighbor, depth + 1))
                
                return neighbors
                
        except Exception as e:
            logger.error(f"获取实体邻居失败: {e}")
            return []
    
    def save_graph(self, file_path: str = None):
        """保存图谱"""
        try:
            if not file_path:
                file_path = self.config.KNOWLEDGE_GRAPH_CONFIG["graph_storage"]["file_path"]
            
            # 确保目录存在
            import os
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 保存图结构
            format_type = self.config.KNOWLEDGE_GRAPH_CONFIG["graph_storage"]["format"]
            
            if format_type == "graphml":
                nx.write_graphml(self.graph, file_path)
            elif format_type == "gexf":
                nx.write_gexf(self.graph, file_path)
            elif format_type == "json":
                graph_data = nx.node_link_data(self.graph)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(graph_data, f, ensure_ascii=False, indent=2)
            else:
                # 使用pickle作为默认格式
                with open(file_path, 'wb') as f:
                    pickle.dump(self.graph, f)
            
            logger.info(f"图谱保存成功: {file_path}")
            
        except Exception as e:
            logger.error(f"图谱保存失败: {e}")
    
    def load_graph(self, file_path: str = None):
        """加载图谱"""
        try:
            if not file_path:
                file_path = self.config.KNOWLEDGE_GRAPH_CONFIG["graph_storage"]["file_path"]
            
            if not os.path.exists(file_path):
                logger.warning(f"图谱文件不存在: {file_path}")
                return
            
            format_type = self.config.KNOWLEDGE_GRAPH_CONFIG["graph_storage"]["format"]
            
            if format_type == "graphml":
                self.graph = nx.read_graphml(file_path)
            elif format_type == "gexf":
                self.graph = nx.read_gexf(file_path)
            elif format_type == "json":
                with open(file_path, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)
                self.graph = nx.node_link_graph(graph_data)
            else:
                # 使用pickle
                with open(file_path, 'rb') as f:
                    self.graph = pickle.load(f)
            
            # 重建实体和关系缓存
            self._rebuild_caches()
            
            logger.info(f"图谱加载成功: {file_path}")
            
        except Exception as e:
            logger.error(f"图谱加载失败: {e}")
    
    def _rebuild_caches(self):
        """重建缓存"""
        self.entities.clear()
        self.relations.clear()
        
        # 重建实体缓存
        for node, data in self.graph.nodes(data=True):
            if 'type' in data:
                entity = Entity(
                    name=node,
                    entity_type=EntityType(data['type']),
                    confidence=data.get('confidence', 1.0),
                    attributes=data.get('attributes', {})
                )
                self.entities[node] = entity
        
        # 重建关系缓存
        for source, target, data in self.graph.edges(data=True):
            if 'type' in data:
                relation = Relation(
                    source=source,
                    target=target,
                    relation_type=RelationType(data['type']),
                    confidence=data.get('confidence', 1.0),
                    attributes=data.get('attributes', {})
                )
                self.relations.append(relation)
