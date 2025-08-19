#!/usr/bin/env python3
"""
Live2D AI助手 问题修复脚本

自动检测和修复常见的配置和依赖问题
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

def setup_logging():
    """
    设置日志配置
    
    功能说明:
    - 配置日志级别为INFO，记录详细的修复过程
    - 同时输出到文件和控制台，便于调试和监控
    - 使用UTF-8编码确保中文日志正确显示
    
    Returns:
        logging.Logger: 配置好的日志记录器实例
        
    异常处理:
        - 如果日志文件创建失败，会降级到仅控制台输出
        - 确保日志系统始终可用，不影响主要功能
    """
    try:
        # 尝试创建完整的日志配置
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('fix_issues.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
        logger.info("日志系统初始化成功")
        return logger
    except Exception as e:
        # 如果文件日志失败，降级到控制台日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        logger = logging.getLogger(__name__)
        logger.warning(f"文件日志创建失败，使用控制台日志: {e}")
        return logger

def fix_ffmpeg_warning():
    """
    修复FFmpeg相关的警告信息
    
    功能说明:
    - 设置环境变量FFMPEG_HIDE_BANNER=1来隐藏FFmpeg的版权信息横幅
    - 减少日志输出中的冗余信息，提高日志可读性
    - 不影响FFmpeg的核心功能，仅优化输出显示
    
    Returns:
        bool: 修复是否成功
        
    异常处理:
        - 捕获环境变量设置过程中的任何异常
        - 记录详细的错误信息用于调试
        - 即使失败也不影响程序继续运行
    """
    try:
        # 设置环境变量隐藏FFmpeg横幅
        os.environ['FFMPEG_HIDE_BANNER'] = '1'
        
        # 验证环境变量是否设置成功
        if os.environ.get('FFMPEG_HIDE_BANNER') == '1':
            logging.info("FFmpeg警告横幅已成功隐藏")
            return True
        else:
            logging.warning("FFmpeg环境变量设置可能未生效")
            return False
            
    except Exception as e:
        logging.error(f"修复FFmpeg警告时发生异常: {type(e).__name__}: {e}")
        return False

def fix_spacy_models():
    """
    修复spaCy自然语言处理模型问题
    
    功能说明:
    - 检查spaCy库是否正确安装
    - 验证中文语言模型zh_core_web_sm的可用性
    - 如果模型缺失，自动尝试下载和安装
    - 提供详细的安装状态反馈和错误诊断
    
    Returns:
        bool: 修复是否成功
        
    异常处理:
    - ImportError: spaCy库未安装时的处理
    - OSError: 模型文件缺失时的处理
    - TimeoutExpired: 下载超时时的处理
    - 网络连接问题的优雅处理
    """
    logging.info("开始检查spaCy模型状态...")
    
    try:
        # 首先检查spaCy库是否可用
        import spacy
        logging.info("spaCy库导入成功")
        
        # 检查中文模型是否已安装
        try:
            nlp = spacy.load("zh_core_web_sm")
            logging.info("中文spaCy模型(zh_core_web_sm)已正确安装并可用")
            
            # 验证模型基本功能
            test_doc = nlp("测试文本")
            if len(test_doc) > 0:
                logging.info("spaCy模型功能验证通过")
                return True
            else:
                logging.warning("spaCy模型加载成功但功能异常")
                return False
                
        except OSError as e:
            logging.warning(f"中文spaCy模型未安装: {e}")
            logging.info("正在尝试自动安装zh_core_web_sm模型...")
            
            try:
                # 使用subprocess安装模型
                result = subprocess.run([
                    sys.executable, "-m", "spacy", "download", "zh_core_web_sm"
                ], capture_output=True, text=True, timeout=300, check=False)
                
                if result.returncode == 0:
                    logging.info("中文spaCy模型安装成功")
                    
                    # 验证安装结果
                    try:
                        nlp = spacy.load("zh_core_web_sm")
                        logging.info("模型安装验证通过")
                        return True
                    except OSError:
                        logging.error("模型安装后仍无法加载")
                        return False
                else:
                    logging.error(f"模型安装失败 - 返回码: {result.returncode}")
                    logging.error(f"错误输出: {result.stderr}")
                    logging.info("建议手动运行: python -m spacy download zh_core_web_sm")
                    return False
                    
            except subprocess.TimeoutExpired:
                logging.error("模型下载超时(300秒)，可能是网络连接问题")
                logging.info("建议检查网络连接或手动安装模型")
                return False
            except Exception as install_error:
                logging.error(f"模型安装过程中发生异常: {type(install_error).__name__}: {install_error}")
                return False
                
    except ImportError as e:
        logging.error(f"spaCy库未安装: {e}")
        logging.info("请先安装spaCy: pip install spacy")
        return False
    except Exception as e:
        logging.error(f"spaCy检查过程中发生未预期的异常: {type(e).__name__}: {e}")
        return False

def fix_directories():
    """
    创建Live2D AI助手项目所需的必要目录结构
    
    功能说明:
    - 创建项目运行所需的核心目录
    - 确保目录权限正确设置
    - 提供详细的目录创建状态反馈
    - 支持目录已存在的情况下的幂等操作
    
    目录说明:
    - logs: 存储应用程序日志文件
    - uploads: 用户上传文件的存储目录
    - temp: 临时文件存储目录
    - models: AI模型文件存储目录
    - data: 应用数据和配置文件目录
    - cache: 缓存文件存储目录
    
    Returns:
        bool: 所有目录创建是否成功
        
    异常处理:
    - PermissionError: 权限不足时的处理
    - OSError: 系统级错误的处理
    - 确保部分失败不影响其他目录的创建
    """
    logging.info("开始创建项目必要目录结构...")
    
    # 定义项目所需的目录列表
    directories = [
        'logs',      # 日志文件目录
        'uploads',   # 用户上传文件目录
        'temp',      # 临时文件目录
        'models',    # AI模型文件目录
        'data',      # 应用数据目录
        'cache'      # 缓存文件目录
    ]
    
    created_dirs = []    # 新创建的目录列表
    existing_dirs = []   # 已存在的目录列表
    failed_dirs = []     # 创建失败的目录列表
    
    for directory in directories:
        try:
            if not os.path.exists(directory):
                # 创建目录，设置适当的权限
                os.makedirs(directory, mode=0o755, exist_ok=True)
                created_dirs.append(directory)
                logging.info(f"成功创建目录: {directory}")
                
                # 验证目录是否真正创建成功
                if not os.path.exists(directory):
                    logging.error(f"目录创建验证失败: {directory}")
                    failed_dirs.append(directory)
                    
            else:
                existing_dirs.append(directory)
                logging.info(f"目录已存在: {directory}")
                
                # 检查目录权限
                if not os.access(directory, os.W_OK):
                    logging.warning(f"目录 {directory} 可能没有写入权限")
                    
        except PermissionError as e:
            logging.error(f"权限不足，无法创建目录 {directory}: {e}")
            failed_dirs.append(directory)
        except OSError as e:
            logging.error(f"系统错误，创建目录失败 {directory}: {e}")
            failed_dirs.append(directory)
        except Exception as e:
            logging.error(f"创建目录时发生未预期的异常 {directory}: {type(e).__name__}: {e}")
            failed_dirs.append(directory)
    
    # 输出创建结果统计
    if created_dirs:
        logging.info(f"成功创建 {len(created_dirs)} 个新目录: {', '.join(created_dirs)}")
    
    if existing_dirs:
        logging.info(f"发现 {len(existing_dirs)} 个已存在目录: {', '.join(existing_dirs)}")
    
    if failed_dirs:
        logging.error(f"创建失败 {len(failed_dirs)} 个目录: {', '.join(failed_dirs)}")
        logging.info("建议检查文件系统权限或手动创建失败的目录")
        return False
    
    logging.info("目录结构创建完成")
    return True

def fix_environment_file():
    """创建环境配置文件"""
    print("检查环境配置文件...")
    
    env_file = Path('.env')
    env_example = Path('.env.example')
    
    if not env_file.exists():
        if env_example.exists():
            # 复制示例文件
            with open(env_example, 'r', encoding='utf-8') as src:
                content = src.read()
            
            with open(env_file, 'w', encoding='utf-8') as dst:
                dst.write(content)
            
            print("✓ 从模板创建 .env 文件")
            print("请编辑 .env 文件配置您的设置")
        else:
            # 创建基础配置
            basic_config = """# Live2D AI助手基础配置
HOST=127.0.0.1
PORT=5000
DEBUG=False
SECRET_KEY=change-this-to-something-secure

# Ollama配置
OLLAMA_BASE_URL=http://127.0.0.1:11434
DEFAULT_MODEL=qwen2:0.5b

# 日志配置
LOG_LEVEL=INFO
"""
            with open(env_file, 'w', encoding='utf-8') as f:
                f.write(basic_config)
            
            print("✓ 创建基础 .env 文件")
    else:
        print("✓ .env 文件已存在")
    
    return True

def fix_requirements():
    """检查和修复依赖问题"""
    print("检查Python依赖...")
    
    critical_packages = [
        'flask',
        'flask-cors', 
        'requests',
        'edge-tts'
    ]
    
    missing_packages = []
    
    for package in critical_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package}")
    
    if missing_packages:
        print(f"缺少关键依赖: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True

def fix_ollama_connection():
    """检查Ollama连接"""
    print("检查Ollama服务...")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✓ Ollama服务运行正常")
            
            # 检查推荐模型
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            
            recommended = ['qwen2:0.5b', 'deepseek-r1:7b']
            for model in recommended:
                if any(model in m for m in models):
                    print(f"✓ 模型 {model} 已安装")
                else:
                    print(f"⚠ 推荐安装模型: ollama pull {model}")
            
            return True
        else:
            print("⚠ Ollama服务响应异常")
    except Exception as e:
        print(f"✗ Ollama服务连接失败: {e}")
    
    print("解决方案:")
    print("1. 从 https://ollama.ai 下载并安装Ollama")
    print("2. 启动Ollama服务")
    print("3. 安装推荐模型: ollama pull qwen2:0.5b")
    return False

def fix_python_compatibility():
    """检查Python兼容性"""
    print(f"检查Python版本: {sys.version}")
    
    if sys.version_info < (3, 8):
        print("✗ Python版本过低，需要3.8+")
        return False
    elif sys.version_info >= (3, 13):
        print("⚠ Python 3.13+可能存在兼容性问题")
        print("建议使用 run.py 启动脚本")
    else:
        print("✓ Python版本兼容")
    
    return True

def main():
    """主函数"""
    print("=" * 60)
    print("Live2D AI助手 问题修复工具")
    print("=" * 60)
    
    setup_logging()
    
    fixes = [
        ("Python兼容性", fix_python_compatibility),
        ("Python依赖", fix_requirements),
        ("目录结构", fix_directories),
        ("环境配置", fix_environment_file),
        ("FFmpeg", fix_ffmpeg_warning),
        ("spaCy模型", fix_spacy_models),
        ("Ollama服务", fix_ollama_connection),
    ]
    
    results = []
    
    for name, fix_func in fixes:
        print(f"\n--- 修复 {name} ---")
        try:
            success = fix_func()
            results.append((name, success))
        except Exception as e:
            print(f"✗ 修复 {name} 时出错: {e}")
            results.append((name, False))
    
    # 总结
    print("\n" + "=" * 60)
    print("修复结果总结:")
    print("=" * 60)
    
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {name}")
    
    failed_count = sum(1 for _, success in results if not success)
    
    if failed_count == 0:
        print("\n🎉 所有检查都通过了！")
        print("现在可以运行: python run.py")
    else:
        print(f"\n⚠ {failed_count} 个问题需要手动解决")
        print("请查看上面的详细信息和解决方案")
    
    return failed_count

if __name__ == "__main__":
    sys.exit(main())
