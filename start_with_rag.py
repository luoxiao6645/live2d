#!/usr/bin/env python3
"""
Live2D AI助手 RAG功能快速启动脚本
"""

import os
import sys
import subprocess
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_ollama_service():
    """检查Ollama服务是否运行"""
    try:
        import requests
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        if response.status_code == 200:
            logger.info("✓ Ollama服务运行正常")
            return True
        else:
            logger.warning("Ollama服务响应异常")
            return False
    except Exception as e:
        logger.warning(f"无法连接到Ollama服务: {e}")
        return False

def check_dependencies():
    """检查依赖是否安装"""
    logger.info("检查依赖...")
    
    required_packages = [
        'flask',
        'langchain',
        'chromadb',
        'sentence_transformers'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package}")
        except ImportError:
            logger.error(f"✗ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"缺少依赖包: {', '.join(missing_packages)}")
        logger.info("请运行: python install_rag_dependencies.py")
        return False
    
    logger.info("✓ 所有依赖检查通过")
    return True

def check_directories():
    """检查必要目录"""
    directories = [
        './models',
        './uploads',
        './knowledge_base'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"创建目录: {directory}")
            except Exception as e:
                logger.error(f"创建目录失败 {directory}: {e}")
                return False
        else:
            logger.info(f"✓ 目录存在: {directory}")
    
    return True

def test_rag_functionality():
    """测试RAG功能"""
    logger.info("测试RAG功能...")
    
    try:
        from rag_manager import RAGManager
        from document_processor import DocumentProcessor
        
        # 快速初始化测试
        rag_manager = RAGManager()
        document_processor = DocumentProcessor()
        
        logger.info("✓ RAG功能初始化成功")
        return True
        
    except Exception as e:
        logger.error(f"✗ RAG功能测试失败: {e}")
        return False

def start_server():
    """启动服务器"""
    logger.info("启动Live2D AI助手服务器...")
    
    try:
        # 启动服务器
        subprocess.run([sys.executable, "server.py"], check=True)
    except KeyboardInterrupt:
        logger.info("服务器已停止")
    except Exception as e:
        logger.error(f"启动服务器失败: {e}")

def print_welcome():
    """打印欢迎信息"""
    print("=" * 60)
    print("🎭 Live2D AI助手 with RAG功能")
    print("=" * 60)
    print()
    print("🆕 新功能:")
    print("  📚 知识库管理 - 上传文档，构建专属知识库")
    print("  🤖 RAG增强对话 - 基于文档内容的智能问答")
    print("  🔍 智能检索 - 自动找到相关文档片段")
    print("  📖 来源追踪 - 显示回答的文档来源")
    print()
    print("🚀 使用方法:")
    print("  1. 在浏览器中访问: http://localhost:5000")
    print("  2. 在'知识库管理'部分启用RAG功能")
    print("  3. 上传PDF、DOCX、TXT或Markdown文档")
    print("  4. 开始与AI助手对话，享受智能问答体验")
    print()
    print("=" * 60)

def main():
    """主函数"""
    print_welcome()
    
    # 检查依赖
    if not check_dependencies():
        logger.error("依赖检查失败，请先安装必要的依赖")
        logger.info("运行: python install_rag_dependencies.py")
        return 1
    
    # 检查目录
    if not check_directories():
        logger.error("目录检查失败")
        return 1
    
    # 测试RAG功能
    if not test_rag_functionality():
        logger.error("RAG功能测试失败")
        return 1
    
    # 检查Ollama服务
    if not check_ollama_service():
        logger.warning("Ollama服务未运行，AI对话功能可能不可用")
        logger.info("请确保Ollama服务正在运行: ollama serve")
        
        response = input("是否继续启动服务器? (y/N): ")
        if response.lower() != 'y':
            return 1
    
    logger.info("所有检查通过，启动服务器...")
    time.sleep(1)
    
    # 启动服务器
    start_server()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
