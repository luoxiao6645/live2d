"""
高级RAG系统配置文件

定义了向量化、知识图谱、内容融合、推理等核心配置信息
"""

from typing import Dict, List, Tuple, Any
from enum import Enum
import os

class VectorizerType(Enum):
    """向量化器类型"""
    SENTENCE_TRANSFORMER = "sentence_transformer"
    CLIP = "clip"
    MULTIMODAL_BERT = "multimodal_bert"
    CUSTOM = "custom"

class GraphType(Enum):
    """图类型"""
    DIRECTED = "directed"
    UNDIRECTED = "undirected"
    MULTIGRAPH = "multigraph"

class EntityType(Enum):
    """实体类型"""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    EVENT = "event"
    CONCEPT = "concept"
    OBJECT = "object"
    TIME = "time"
    CUSTOM = "custom"

class RelationType(Enum):
    """关系类型"""
    IS_A = "is_a"
    PART_OF = "part_of"
    RELATED_TO = "related_to"
    LOCATED_IN = "located_in"
    WORKS_FOR = "works_for"
    HAPPENED_AT = "happened_at"
    CAUSES = "causes"
    SIMILAR_TO = "similar_to"
    CUSTOM = "custom"

class AdvancedRAGConfig:
    """高级RAG系统配置类"""
    
    # 向量化配置
    VECTORIZER_CONFIG = {
        "text_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "image_model": "clip-ViT-B-32",
        "multimodal_model": "clip-ViT-B-32",
        "embedding_dimension": 384,
        "max_sequence_length": 512,
        "batch_size": 32,
        "device": "auto",  # auto, cpu, cuda
        "cache_embeddings": True,
        "cache_size": 10000
    }
    
    # 知识图谱配置
    KNOWLEDGE_GRAPH_CONFIG = {
        "graph_type": GraphType.DIRECTED,
        "max_nodes": 100000,
        "max_edges": 500000,
        "entity_extraction": {
            "enabled": True,
            "model": "zh_core_web_sm",  # spaCy模型
            "confidence_threshold": 0.8,
            "max_entities_per_doc": 50
        },
        "relation_extraction": {
            "enabled": True,
            "patterns": [
                {"pattern": r"(.+)\s+是\s+(.+)", "relation": "is_a"},
                {"pattern": r"(.+)\s+位于\s+(.+)", "relation": "located_in"},
                {"pattern": r"(.+)\s+属于\s+(.+)", "relation": "part_of"},
                {"pattern": r"(.+)\s+导致\s+(.+)", "relation": "causes"}
            ],
            "confidence_threshold": 0.7
        },
        "graph_storage": {
            "format": "graphml",  # graphml, gexf, json
            "file_path": "./knowledge_graphs/main_graph.graphml",
            "auto_save": True,
            "save_interval": 300  # 5分钟
        }
    }
    
    # 内容融合配置
    CONTENT_FUSION_CONFIG = {
        "fusion_strategies": [
            "weighted_average",
            "attention_based",
            "graph_based",
            "confidence_based"
        ],
        "default_strategy": "attention_based",
        "conflict_resolution": {
            "enabled": True,
            "strategy": "confidence_based",  # confidence_based, majority_vote, latest
            "threshold": 0.1
        },
        "quality_assessment": {
            "enabled": True,
            "factors": [
                "source_credibility",
                "content_freshness",
                "information_completeness",
                "consistency_score"
            ],
            "weights": [0.3, 0.2, 0.3, 0.2]
        },
        "fusion_cache": {
            "enabled": True,
            "ttl": 3600,  # 1小时
            "max_size": 1000
        }
    }
    
    # 推理引擎配置
    REASONING_CONFIG = {
        "reasoning_types": [
            "path_based",
            "subgraph_matching",
            "random_walk",
            "graph_neural_network"
        ],
        "default_reasoning": "path_based",
        "path_finding": {
            "max_depth": 5,
            "max_paths": 10,
            "algorithm": "dijkstra",  # dijkstra, bfs, dfs
            "weight_function": "semantic_similarity"
        },
        "subgraph_matching": {
            "max_subgraph_size": 20,
            "similarity_threshold": 0.8,
            "isomorphism_check": True
        },
        "random_walk": {
            "walk_length": 10,
            "num_walks": 100,
            "restart_probability": 0.15
        },
        "confidence_calculation": {
            "enabled": True,
            "factors": [
                "path_length",
                "edge_weights",
                "node_importance",
                "evidence_strength"
            ],
            "weights": [0.2, 0.3, 0.2, 0.3]
        }
    }
    
    # 查询处理配置
    QUERY_PROCESSING_CONFIG = {
        "query_understanding": {
            "enabled": True,
            "intent_recognition": True,
            "entity_linking": True,
            "query_expansion": True
        },
        "query_optimization": {
            "enabled": True,
            "query_rewriting": True,
            "query_decomposition": True,
            "parallel_execution": True
        },
        "result_ranking": {
            "enabled": True,
            "ranking_factors": [
                "relevance_score",
                "confidence_score",
                "freshness_score",
                "authority_score"
            ],
            "weights": [0.4, 0.3, 0.15, 0.15]
        },
        "result_filtering": {
            "enabled": True,
            "min_confidence": 0.5,
            "max_results": 20,
            "diversity_threshold": 0.8
        }
    }
    
    # 多模态配置
    MULTIMODAL_CONFIG = {
        "modalities": ["text", "image", "audio"],
        "cross_modal_alignment": {
            "enabled": True,
            "alignment_model": "clip",
            "alignment_threshold": 0.7
        },
        "modal_fusion": {
            "strategy": "late_fusion",  # early_fusion, late_fusion, hybrid
            "fusion_weights": {
                "text": 0.6,
                "image": 0.3,
                "audio": 0.1
            }
        },
        "modal_specific_processing": {
            "text": {
                "preprocessing": ["tokenization", "normalization", "segmentation"],
                "feature_extraction": ["embeddings", "keywords", "entities"]
            },
            "image": {
                "preprocessing": ["resize", "normalize", "augment"],
                "feature_extraction": ["visual_features", "objects", "scenes"]
            }
        }
    }
    
    # 性能配置
    PERFORMANCE_CONFIG = {
        "parallel_processing": {
            "enabled": True,
            "max_workers": 4,
            "chunk_size": 100
        },
        "caching": {
            "enabled": True,
            "cache_types": ["embeddings", "graph_queries", "reasoning_results"],
            "cache_backend": "memory",  # memory, redis, file
            "ttl": 3600
        },
        "optimization": {
            "lazy_loading": True,
            "batch_processing": True,
            "memory_management": True,
            "gpu_acceleration": True
        },
        "monitoring": {
            "enabled": True,
            "metrics": [
                "query_latency",
                "memory_usage",
                "cache_hit_rate",
                "error_rate"
            ],
            "log_level": "INFO"
        }
    }
    
    # 实体识别模式
    ENTITY_PATTERNS = {
        EntityType.PERSON: [
            r"[A-Z][a-z]+\s+[A-Z][a-z]+",  # 英文人名
            r"[\u4e00-\u9fff]{2,4}",       # 中文人名
        ],
        EntityType.ORGANIZATION: [
            r"[A-Z][a-zA-Z\s]+(?:Inc|Corp|Ltd|LLC|Company)",
            r"[\u4e00-\u9fff]+(?:公司|集团|企业|机构|组织)"
        ],
        EntityType.LOCATION: [
            r"[A-Z][a-zA-Z\s]+(?:City|State|Country|Province)",
            r"[\u4e00-\u9fff]+(?:市|省|县|区|国|州)"
        ],
        EntityType.TIME: [
            r"\d{4}年\d{1,2}月\d{1,2}日",
            r"\d{1,2}/\d{1,2}/\d{4}",
            r"\d{4}-\d{2}-\d{2}"
        ]
    }
    
    # 关系抽取模式
    RELATION_PATTERNS = {
        RelationType.IS_A: [
            r"(.+)\s+是\s+(.+)",
            r"(.+)\s+is\s+a\s+(.+)",
            r"(.+)\s+属于\s+(.+)"
        ],
        RelationType.LOCATED_IN: [
            r"(.+)\s+位于\s+(.+)",
            r"(.+)\s+在\s+(.+)",
            r"(.+)\s+is\s+located\s+in\s+(.+)"
        ],
        RelationType.PART_OF: [
            r"(.+)\s+是\s+(.+)\s+的一部分",
            r"(.+)\s+属于\s+(.+)",
            r"(.+)\s+is\s+part\s+of\s+(.+)"
        ],
        RelationType.CAUSES: [
            r"(.+)\s+导致\s+(.+)",
            r"(.+)\s+引起\s+(.+)",
            r"(.+)\s+causes\s+(.+)"
        ]
    }
    
    @classmethod
    def get_model_config(cls, model_type: VectorizerType) -> Dict[str, Any]:
        """获取模型配置"""
        configs = {
            VectorizerType.SENTENCE_TRANSFORMER: {
                "model_name": cls.VECTORIZER_CONFIG["text_model"],
                "dimension": cls.VECTORIZER_CONFIG["embedding_dimension"],
                "max_length": cls.VECTORIZER_CONFIG["max_sequence_length"]
            },
            VectorizerType.CLIP: {
                "model_name": cls.VECTORIZER_CONFIG["image_model"],
                "dimension": 512,
                "supports_text": True,
                "supports_image": True
            }
        }
        return configs.get(model_type, {})
    
    @classmethod
    def get_entity_patterns(cls, entity_type: EntityType) -> List[str]:
        """获取实体识别模式"""
        return cls.ENTITY_PATTERNS.get(entity_type, [])
    
    @classmethod
    def get_relation_patterns(cls, relation_type: RelationType) -> List[str]:
        """获取关系抽取模式"""
        return cls.RELATION_PATTERNS.get(relation_type, [])
    
    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        directories = [
            "./knowledge_graphs",
            "./embeddings_cache",
            "./fusion_cache",
            "./reasoning_cache"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def validate_config(cls) -> bool:
        """验证配置有效性"""
        try:
            # 检查必要的配置项
            required_configs = [
                cls.VECTORIZER_CONFIG,
                cls.KNOWLEDGE_GRAPH_CONFIG,
                cls.CONTENT_FUSION_CONFIG,
                cls.REASONING_CONFIG
            ]
            
            for config in required_configs:
                if not isinstance(config, dict) or not config:
                    return False
            
            # 检查数值范围
            if not (0 < cls.VECTORIZER_CONFIG["embedding_dimension"] <= 2048):
                return False
            
            if not (0 < cls.KNOWLEDGE_GRAPH_CONFIG["max_nodes"] <= 1000000):
                return False
            
            return True
            
        except Exception:
            return False
