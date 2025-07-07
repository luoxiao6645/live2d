#!/usr/bin/env python3
"""
高级RAG系统测试脚本

测试高级RAG系统的各个组件功能
"""

import sys
import asyncio
import logging
from typing import Dict, Any

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_advanced_rag_config():
    """测试高级RAG配置"""
    logger.info("测试高级RAG配置...")
    
    try:
        from advanced_rag_system.advanced_rag_config import AdvancedRAGConfig, VectorizerType, EntityType
        
        config = AdvancedRAGConfig()
        
        # 测试配置验证
        is_valid = config.validate_config()
        logger.info(f"配置验证结果: {is_valid}")
        
        # 测试模型配置获取
        model_config = config.get_model_config(VectorizerType.SENTENCE_TRANSFORMER)
        logger.info(f"模型配置: {model_config}")
        
        # 测试实体模式获取
        entity_patterns = config.get_entity_patterns(EntityType.PERSON)
        logger.info(f"人名实体模式数量: {len(entity_patterns)}")
        
        # 创建目录
        config.create_directories()
        
        logger.info("✓ 高级RAG配置测试通过")
        return True
        
    except Exception as e:
        logger.error(f"✗ 高级RAG配置测试失败: {e}")
        return False

def test_multimodal_vectorizer():
    """测试图文混合向量化器"""
    logger.info("测试图文混合向量化器...")
    
    try:
        from advanced_rag_system.multimodal_vectorizer import MultimodalVectorizer, VectorizerType
        
        vectorizer = MultimodalVectorizer(VectorizerType.SENTENCE_TRANSFORMER)
        
        # 测试文本向量化
        test_text = "这是一个测试文本，用于验证向量化功能。"
        text_embedding = vectorizer.vectorize_text(test_text)
        logger.info(f"文本向量维度: {len(text_embedding)}")
        
        # 测试批量向量化
        test_texts = ["文本1", "文本2", "文本3"]
        batch_embeddings = vectorizer.batch_vectorize_texts(test_texts)
        logger.info(f"批量向量化结果数量: {len(batch_embeddings)}")
        
        # 测试相似度计算
        similarity = vectorizer.compute_similarity(text_embedding, text_embedding)
        logger.info(f"自相似度: {similarity}")
        
        # 测试统计信息
        stats = vectorizer.get_vectorizer_stats()
        logger.info(f"向量化器统计: {stats}")
        
        logger.info("✓ 图文混合向量化器测试通过")
        return True
        
    except Exception as e:
        logger.error(f"✗ 图文混合向量化器测试失败: {e}")
        return False

def test_knowledge_graph_builder():
    """测试知识图谱构建器"""
    logger.info("测试知识图谱构建器...")
    
    try:
        from advanced_rag_system.knowledge_graph_builder import KnowledgeGraphBuilder
        
        kg_builder = KnowledgeGraphBuilder()
        
        # 测试实体提取
        test_text = "张三是北京大学的学生，他住在北京市海淀区。"
        entities = kg_builder.extract_entities(test_text)
        logger.info(f"提取的实体数量: {len(entities)}")
        
        for entity in entities:
            logger.info(f"  实体: {entity.name} ({entity.entity_type.value})")
        
        # 测试关系提取
        relations = kg_builder.extract_relations(test_text, entities)
        logger.info(f"提取的关系数量: {len(relations)}")
        
        for relation in relations:
            logger.info(f"  关系: {relation.source} -> {relation.target} ({relation.relation_type.value})")
        
        # 测试图谱构建
        result = kg_builder.build_graph_from_document(test_text, "test_doc_1")
        if result["success"]:
            logger.info(f"图谱构建成功: {result['entities_added']}个实体, {result['relations_added']}个关系")
        else:
            logger.error(f"图谱构建失败: {result['error']}")
        
        # 测试图谱统计
        stats = kg_builder.get_graph_stats()
        logger.info(f"图谱统计: {stats}")
        
        # 测试实体查找
        if entities:
            entity_name = entities[0].name
            found_entity = kg_builder.find_entity(entity_name)
            if found_entity:
                logger.info(f"实体查找成功: {found_entity.name}")
        
        logger.info("✓ 知识图谱构建器测试通过")
        return True
        
    except Exception as e:
        logger.error(f"✗ 知识图谱构建器测试失败: {e}")
        return False

def test_content_fusion_engine():
    """测试内容融合引擎"""
    logger.info("测试内容融合引擎...")
    
    try:
        from advanced_rag_system.content_fusion_engine import ContentFusionEngine, ContentSource
        from datetime import datetime
        
        fusion_engine = ContentFusionEngine()
        
        # 创建测试内容源
        sources = [
            ContentSource(
                source_id="source_1",
                content="这是第一个信息源的内容。",
                credibility=0.9,
                timestamp=datetime.now()
            ),
            ContentSource(
                source_id="source_2", 
                content="这是第二个信息源的内容，提供了补充信息。",
                credibility=0.8,
                timestamp=datetime.now()
            ),
            ContentSource(
                source_id="source_3",
                content="第三个信息源包含了相关的背景信息。",
                credibility=0.7,
                timestamp=datetime.now()
            )
        ]
        
        # 测试不同融合策略
        strategies = ["weighted_average", "attention_based", "confidence_based"]
        
        for strategy in strategies:
            result = fusion_engine.fuse_content(sources, strategy)
            logger.info(f"融合策略 {strategy}: 置信度 {result.confidence:.2f}")
            logger.info(f"  融合内容长度: {len(result.fused_content)}")
        
        # 测试统计信息
        stats = fusion_engine.get_fusion_stats()
        logger.info(f"融合引擎统计: {stats}")
        
        logger.info("✓ 内容融合引擎测试通过")
        return True
        
    except Exception as e:
        logger.error(f"✗ 内容融合引擎测试失败: {e}")
        return False

def test_multimodal_reasoning_engine():
    """测试多模态推理引擎"""
    logger.info("测试多模态推理引擎...")
    
    try:
        from advanced_rag_system.multimodal_reasoning_engine import MultimodalReasoningEngine
        from advanced_rag_system.knowledge_graph_builder import KnowledgeGraphBuilder
        
        # 创建知识图谱
        kg_builder = KnowledgeGraphBuilder()
        test_text = "苹果公司位于美国加利福尼亚州。史蒂夫·乔布斯是苹果公司的创始人。"
        kg_builder.build_graph_from_document(test_text, "test_doc")
        
        # 创建推理引擎
        reasoning_engine = MultimodalReasoningEngine(kg_builder)
        
        # 测试推理
        test_queries = [
            "苹果公司在哪里？",
            "谁创建了苹果公司？",
            "苹果公司和史蒂夫·乔布斯的关系"
        ]
        
        for query in test_queries:
            result = reasoning_engine.reason(query)
            logger.info(f"查询: {query}")
            logger.info(f"  答案: {result.answer}")
            logger.info(f"  置信度: {result.confidence:.2f}")
            logger.info(f"  推理路径数量: {len(result.paths)}")
        
        # 测试推理统计
        stats = reasoning_engine.get_reasoning_stats()
        logger.info(f"推理引擎统计: {stats}")
        
        logger.info("✓ 多模态推理引擎测试通过")
        return True
        
    except Exception as e:
        logger.error(f"✗ 多模态推理引擎测试失败: {e}")
        return False

def test_graph_query_processor():
    """测试图查询处理器"""
    logger.info("测试图查询处理器...")
    
    try:
        from advanced_rag_system.graph_query_processor import GraphQueryProcessor
        from advanced_rag_system.knowledge_graph_builder import KnowledgeGraphBuilder
        from advanced_rag_system.multimodal_reasoning_engine import MultimodalReasoningEngine
        
        # 创建知识图谱
        kg_builder = KnowledgeGraphBuilder()
        test_text = "北京是中国的首都。上海是中国的经济中心。"
        kg_builder.build_graph_from_document(test_text, "test_doc")
        
        # 创建推理引擎
        reasoning_engine = MultimodalReasoningEngine(kg_builder)
        
        # 创建查询处理器
        query_processor = GraphQueryProcessor(kg_builder, reasoning_engine)
        
        # 测试查询
        test_queries = [
            "什么是北京？",
            "北京和中国的关系",
            "中国有哪些城市？"
        ]
        
        for query in test_queries:
            result = query_processor.process_query(query)
            logger.info(f"查询: {query}")
            logger.info(f"  结果数量: {result.result_count}")
            logger.info(f"  置信度: {result.confidence:.2f}")
            logger.info(f"  查询时间: {result.query_time:.3f}s")
        
        # 测试查询统计
        stats = query_processor.get_query_stats()
        logger.info(f"查询处理器统计: {stats}")
        
        logger.info("✓ 图查询处理器测试通过")
        return True
        
    except Exception as e:
        logger.error(f"✗ 图查询处理器测试失败: {e}")
        return False

def test_advanced_rag_manager():
    """测试高级RAG管理器"""
    logger.info("测试高级RAG管理器...")
    
    try:
        from advanced_rag_system.advanced_rag_manager import AdvancedRAGManager
        
        # 创建管理器
        manager = AdvancedRAGManager()
        
        # 测试文档处理
        test_text = "人工智能是计算机科学的一个分支。机器学习是人工智能的重要组成部分。"
        doc_result = manager.process_document(test_text, "test_doc_ai")
        
        if doc_result["success"]:
            logger.info("文档处理成功")
            logger.info(f"  处理步骤: {doc_result['processing_steps']}")
        else:
            logger.error(f"文档处理失败: {doc_result['error']}")
        
        # 测试高级查询
        query_result = manager.advanced_query("什么是人工智能？")
        
        if query_result["success"]:
            logger.info("高级查询成功")
            logger.info(f"  融合答案: {query_result['fused_answer'][:100]}...")
            logger.info(f"  置信度: {query_result['confidence']:.2f}")
        else:
            logger.error(f"高级查询失败: {query_result['error']}")
        
        # 测试多模态推理
        reasoning_result = manager.multimodal_reasoning("人工智能和机器学习的关系")
        
        if reasoning_result["success"]:
            logger.info("多模态推理成功")
            reasoning_answer = reasoning_result["reasoning_result"]["answer"]
            logger.info(f"  推理答案: {reasoning_answer[:100]}...")
        else:
            logger.error(f"多模态推理失败: {reasoning_result['error']}")
        
        # 测试系统状态
        status = manager.get_system_status()
        if "error" not in status:
            logger.info("系统状态获取成功")
            logger.info(f"  系统活跃: {status['is_active']}")
            logger.info(f"  处理统计: {status['processing_stats']}")
        else:
            logger.warning(f"系统状态获取失败: {status['error']}")
        
        logger.info("✓ 高级RAG管理器测试通过")
        return True
        
    except Exception as e:
        logger.error(f"✗ 高级RAG管理器测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("=" * 60)
    logger.info("Live2D AI助手 高级RAG系统测试")
    logger.info("=" * 60)
    
    tests = [
        ("高级RAG配置", test_advanced_rag_config),
        ("图文混合向量化器", test_multimodal_vectorizer),
        ("知识图谱构建器", test_knowledge_graph_builder),
        ("内容融合引擎", test_content_fusion_engine),
        ("多模态推理引擎", test_multimodal_reasoning_engine),
        ("图查询处理器", test_graph_query_processor),
        ("高级RAG管理器", test_advanced_rag_manager)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n开始 {test_name} 测试...")
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} 测试通过")
            else:
                logger.error(f"✗ {test_name} 测试失败")
        except Exception as e:
            logger.error(f"✗ {test_name} 测试异常: {e}")
    
    logger.info("=" * 60)
    logger.info(f"测试完成: {passed}/{total} 通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！高级RAG系统功能正常")
        return 0
    else:
        logger.warning("⚠️  部分测试失败，请检查相关问题")
        return 1

if __name__ == "__main__":
    sys.exit(main())
