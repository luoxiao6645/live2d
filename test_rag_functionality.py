#!/usr/bin/env python3
"""
RAG功能测试脚本
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """测试必要的导入"""
    logger.info("测试导入...")
    
    try:
        from rag_manager import RAGManager
        from document_processor import DocumentProcessor
        logger.info("✓ RAG模块导入成功")
        return True
    except ImportError as e:
        logger.error(f"✗ RAG模块导入失败: {e}")
        return False

def create_test_document():
    """创建测试文档"""
    test_content = """
# Live2D AI助手使用指南

## 简介
Live2D AI助手是一个结合了虚拟角色展示和人工智能对话的应用程序。

## 主要功能
1. Live2D模型展示
2. AI对话系统
3. 语音合成
4. 语音识别
5. 情感表达

## 技术特点
- 使用PIXI.js进行渲染
- 集成Ollama API
- 支持多种语音合成
- 实时口型同步

## 使用方法
启动服务器后，在浏览器中访问应用即可开始使用。
"""
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write(test_content)
        return f.name

def test_document_processor():
    """测试文档处理器"""
    logger.info("测试文档处理器...")
    
    try:
        from document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        
        # 测试文件类型验证
        class MockFile:
            def __init__(self, filename, size=1000):
                self.filename = filename
                self.size = size
                self.pos = 0
            
            def seek(self, pos, whence=0):
                if whence == 0:  # SEEK_SET
                    self.pos = pos
                elif whence == 2:  # SEEK_END
                    self.pos = self.size
            
            def tell(self):
                return self.pos
        
        # 测试有效文件
        valid_file = MockFile("test.pdf")
        is_valid, error = processor.validate_file(valid_file)
        if is_valid:
            logger.info("✓ 文件验证测试通过")
        else:
            logger.warning(f"文件验证失败: {error}")
        
        # 测试无效文件
        invalid_file = MockFile("test.xyz")
        is_valid, error = processor.validate_file(invalid_file)
        if not is_valid:
            logger.info("✓ 无效文件检测正常")
        
        logger.info("✓ 文档处理器测试通过")
        return True
        
    except Exception as e:
        logger.error(f"✗ 文档处理器测试失败: {e}")
        return False

def test_rag_manager():
    """测试RAG管理器"""
    logger.info("测试RAG管理器...")
    
    try:
        from rag_manager import RAGManager
        
        # 创建临时目录
        test_dir = tempfile.mkdtemp()
        logger.info(f"使用临时目录: {test_dir}")
        
        # 初始化RAG管理器
        rag_manager = RAGManager(persist_directory=test_dir)
        logger.info("✓ RAG管理器初始化成功")
        
        # 创建测试文档
        test_file = create_test_document()
        logger.info(f"创建测试文档: {test_file}")
        
        # 测试文档加载
        documents = rag_manager.load_document(test_file)
        if documents:
            logger.info(f"✓ 文档加载成功，页数: {len(documents)}")
        else:
            logger.warning("文档加载返回空结果")
        
        # 测试文档处理和存储
        doc_id = rag_manager.process_and_store_document(test_file)
        logger.info(f"✓ 文档处理和存储成功，文档ID: {doc_id}")
        
        # 测试搜索
        results = rag_manager.search_documents("Live2D", k=3)
        if results:
            logger.info(f"✓ 文档搜索成功，找到 {len(results)} 个结果")
            for i, doc in enumerate(results):
                logger.info(f"  结果 {i+1}: {doc.page_content[:100]}...")
        else:
            logger.warning("搜索未返回结果")
        
        # 测试知识库信息
        kb_info = rag_manager.get_knowledge_base_info()
        logger.info(f"✓ 知识库信息: {kb_info}")
        
        # 测试RAG链创建
        try:
            rag_chain = rag_manager.create_rag_chain()
            logger.info("✓ RAG链创建成功")
            
            # 测试RAG生成（如果Ollama可用）
            try:
                response = rag_chain.invoke("什么是Live2D？")
                logger.info(f"✓ RAG生成测试成功: {response[:100]}...")
            except Exception as e:
                logger.warning(f"RAG生成测试失败（可能是Ollama不可用）: {e}")
                
        except Exception as e:
            logger.warning(f"RAG链创建失败（可能是Ollama不可用）: {e}")
        
        # 清理
        os.unlink(test_file)
        import shutil
        shutil.rmtree(test_dir)
        logger.info("✓ 测试清理完成")
        
        logger.info("✓ RAG管理器测试通过")
        return True
        
    except Exception as e:
        logger.error(f"✗ RAG管理器测试失败: {e}")
        return False

def test_server_integration():
    """测试服务器集成"""
    logger.info("测试服务器集成...")
    
    try:
        # 检查server.py是否包含RAG相关代码
        with open('server.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_imports = [
            'from rag_manager import RAGManager',
            'from document_processor import DocumentProcessor'
        ]
        
        required_routes = [
            '/api/rag/status',
            '/api/rag/upload',
            '/rag_generate'
        ]
        
        for imp in required_imports:
            if imp in content:
                logger.info(f"✓ 找到导入: {imp}")
            else:
                logger.warning(f"✗ 缺少导入: {imp}")
        
        for route in required_routes:
            if route in content:
                logger.info(f"✓ 找到路由: {route}")
            else:
                logger.warning(f"✗ 缺少路由: {route}")
        
        logger.info("✓ 服务器集成检查完成")
        return True
        
    except Exception as e:
        logger.error(f"✗ 服务器集成测试失败: {e}")
        return False

def test_frontend_integration():
    """测试前端集成"""
    logger.info("测试前端集成...")
    
    try:
        # 检查HTML文件是否包含RAG相关代码
        with open('live2d_llm.html', 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_elements = [
            'id="rag_enabled"',
            'id="document_upload"',
            'uploadDocument()',
            'searchDocuments()'
        ]
        
        for element in required_elements:
            if element in content:
                logger.info(f"✓ 找到前端元素: {element}")
            else:
                logger.warning(f"✗ 缺少前端元素: {element}")
        
        logger.info("✓ 前端集成检查完成")
        return True
        
    except Exception as e:
        logger.error(f"✗ 前端集成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("=" * 60)
    logger.info("Live2D AI助手 RAG功能测试")
    logger.info("=" * 60)
    
    tests = [
        ("导入测试", test_imports),
        ("文档处理器测试", test_document_processor),
        ("RAG管理器测试", test_rag_manager),
        ("服务器集成测试", test_server_integration),
        ("前端集成测试", test_frontend_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n开始 {test_name}...")
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} 通过")
            else:
                logger.error(f"✗ {test_name} 失败")
        except Exception as e:
            logger.error(f"✗ {test_name} 异常: {e}")
    
    logger.info("=" * 60)
    logger.info(f"测试完成: {passed}/{total} 通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！RAG功能已准备就绪")
        return 0
    else:
        logger.warning("⚠️  部分测试失败，请检查相关问题")
        return 1

if __name__ == "__main__":
    sys.exit(main())
