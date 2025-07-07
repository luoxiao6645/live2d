#!/usr/bin/env python3
"""
Live2D AI助手 RAG功能依赖安装脚本
"""

import subprocess
import sys
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    """运行命令并处理错误"""
    logger.info(f"正在{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"{description}成功")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"{description}失败: {e}")
        logger.error(f"错误输出: {e.stderr}")
        return False

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("需要Python 3.8或更高版本")
        return False
    logger.info(f"Python版本检查通过: {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    """安装依赖包"""
    logger.info("开始安装RAG功能依赖...")
    
    # 基础依赖
    base_packages = [
        "langchain==0.3.12",
        "langchain-community==0.3.12", 
        "langchain-core==0.3.28",
        "langchain-text-splitters==0.3.2",
        "langchain-chroma==0.1.4",
        "langchain-ollama==0.2.1"
    ]
    
    # 向量数据库
    vector_packages = [
        "chromadb==0.5.23"
    ]
    
    # 嵌入模型
    embedding_packages = [
        "sentence-transformers==3.3.1"
    ]
    
    # 文档处理
    document_packages = [
        "pypdf==5.1.0",
        "python-docx==1.1.2", 
        "python-multipart==0.0.12",
        "unstructured==0.16.9",
        "markdown==3.7"
    ]
    
    all_packages = base_packages + vector_packages + embedding_packages + document_packages
    
    # 分批安装以避免依赖冲突
    package_groups = [
        ("基础LangChain包", base_packages),
        ("向量数据库", vector_packages),
        ("嵌入模型", embedding_packages),
        ("文档处理", document_packages)
    ]
    
    for group_name, packages in package_groups:
        logger.info(f"安装{group_name}...")
        for package in packages:
            if not run_command(f"pip install {package}", f"安装 {package}"):
                logger.warning(f"安装 {package} 失败，继续安装其他包...")
    
    return True

def create_directories():
    """创建必要的目录"""
    directories = [
        "./knowledge_base",
        "./uploads"
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"创建目录: {directory}")
        except Exception as e:
            logger.error(f"创建目录失败 {directory}: {e}")

def test_imports():
    """测试导入"""
    logger.info("测试依赖包导入...")
    
    test_packages = [
        "langchain",
        "langchain_community", 
        "langchain_core",
        "langchain_text_splitters",
        "langchain_chroma",
        "langchain_ollama",
        "chromadb",
        "sentence_transformers",
        "pypdf",
        "docx",
        "markdown"
    ]
    
    failed_imports = []
    
    for package in test_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} 导入成功")
        except ImportError as e:
            logger.error(f"✗ {package} 导入失败: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        logger.warning(f"以下包导入失败: {', '.join(failed_imports)}")
        return False
    else:
        logger.info("所有依赖包导入测试通过！")
        return True

def main():
    """主函数"""
    logger.info("=" * 50)
    logger.info("Live2D AI助手 RAG功能依赖安装")
    logger.info("=" * 50)
    
    # 检查Python版本
    if not check_python_version():
        sys.exit(1)
    
    # 创建目录
    create_directories()
    
    # 安装依赖
    if not install_dependencies():
        logger.error("依赖安装失败")
        sys.exit(1)
    
    # 测试导入
    if not test_imports():
        logger.warning("部分依赖包导入失败，但安装过程已完成")
        logger.info("请检查错误信息并手动安装失败的包")
    
    logger.info("=" * 50)
    logger.info("RAG功能依赖安装完成！")
    logger.info("现在可以启动服务器并使用RAG功能了")
    logger.info("运行: python server.py")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()
