#!/usr/bin/env python3
"""
多模态系统测试脚本

测试多模态AI系统的各个组件功能
"""

import sys
import os
import logging
from typing import Dict, Any
from PIL import Image
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_image():
    """创建测试图像"""
    try:
        # 创建一个简单的测试图像
        width, height = 400, 300
        image = Image.new('RGB', (width, height), color='lightblue')
        
        # 添加一些简单的图形
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        
        # 画一个圆
        draw.ellipse([50, 50, 150, 150], fill='red', outline='darkred')
        
        # 画一个矩形
        draw.rectangle([200, 100, 350, 200], fill='green', outline='darkgreen')
        
        # 添加文字
        try:
            from PIL import ImageFont
            font = ImageFont.load_default()
            draw.text((50, 250), "Test Image", fill='black', font=font)
        except:
            draw.text((50, 250), "Test Image", fill='black')
        
        # 保存测试图像
        test_image_path = "./test_image.jpg"
        image.save(test_image_path, "JPEG")
        
        logger.info(f"测试图像创建成功: {test_image_path}")
        return test_image_path
        
    except Exception as e:
        logger.error(f"创建测试图像失败: {e}")
        return None

def test_image_processor():
    """测试图像处理器"""
    logger.info("测试图像处理器...")
    
    try:
        from multimodal_system.image_processor import ImageProcessor
        
        processor = ImageProcessor()
        
        # 创建测试图像
        test_image_path = create_test_image()
        if not test_image_path:
            return False
        
        # 读取图像数据
        with open(test_image_path, 'rb') as f:
            image_data = f.read()
        
        # 验证图像
        is_valid, error_msg = processor.validate_image(image_data, "test_image.jpg")
        if is_valid:
            logger.info("✓ 图像验证通过")
        else:
            logger.error(f"✗ 图像验证失败: {error_msg}")
            return False
        
        # 处理图像
        result = processor.process_image(image_data, "test_image.jpg")
        if result["success"]:
            logger.info("✓ 图像处理成功")
            logger.info(f"  处理后路径: {result['processed_path']}")
            logger.info(f"  缩略图路径: {result['thumbnail_path']}")
            logger.info(f"  特征数量: {len(result['features'])}")
        else:
            logger.error(f"✗ 图像处理失败: {result['error']}")
            return False
        
        # 清理测试文件
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
        
        logger.info("✓ 图像处理器测试通过")
        return True
        
    except Exception as e:
        logger.error(f"✗ 图像处理器测试失败: {e}")
        return False

def test_image_analyzer():
    """测试图像分析器"""
    logger.info("测试图像分析器...")
    
    try:
        from multimodal_system.image_analyzer import ImageAnalyzer
        
        analyzer = ImageAnalyzer()
        
        # 创建测试图像
        test_image_path = create_test_image()
        if not test_image_path:
            return False
        
        # 分析图像
        analysis_result = analyzer.analyze_image(test_image_path, "general")
        
        if analysis_result["success"]:
            logger.info("✓ 图像分析成功")
            logger.info(f"  描述: {analysis_result.get('description', '无描述')}")
            logger.info(f"  识别物体: {analysis_result.get('objects', [])}")
            logger.info(f"  情感: {analysis_result.get('emotions', [])}")
            logger.info(f"  场景: {analysis_result.get('scene', '无场景')}")
        else:
            logger.error(f"✗ 图像分析失败: {analysis_result['error']}")
            return False
        
        # 清理测试文件
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
        
        logger.info("✓ 图像分析器测试通过")
        return True
        
    except Exception as e:
        logger.error(f"✗ 图像分析器测试失败: {e}")
        return False

def test_vision_language_model():
    """测试视觉语言模型"""
    logger.info("测试视觉语言模型...")
    
    try:
        from multimodal_system.vision_language_model import VisionLanguageModel
        
        vlm = VisionLanguageModel()
        
        # 创建测试图像
        test_image_path = create_test_image()
        if not test_image_path:
            return False
        
        # 生成图像描述
        desc_result = vlm.generate_image_description(test_image_path)
        
        if desc_result["success"]:
            logger.info("✓ 图像描述生成成功")
            logger.info(f"  描述: {desc_result['description']}")
            logger.info(f"  模型: {desc_result.get('model', '未知')}")
        else:
            logger.error(f"✗ 图像描述生成失败: {desc_result['error']}")
            return False
        
        # 测试图像问答
        qa_result = vlm.answer_image_question(test_image_path, "这张图片里有什么？")
        
        if qa_result["success"]:
            logger.info("✓ 图像问答成功")
            logger.info(f"  问题: {qa_result['question']}")
            logger.info(f"  回答: {qa_result['answer']}")
        else:
            logger.warning(f"图像问答失败: {qa_result['error']}")
        
        # 清理测试文件
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
        
        logger.info("✓ 视觉语言模型测试通过")
        return True
        
    except Exception as e:
        logger.error(f"✗ 视觉语言模型测试失败: {e}")
        return False

def test_multimodal_rag():
    """测试多模态RAG"""
    logger.info("测试多模态RAG...")
    
    try:
        from multimodal_system.multimodal_rag import MultimodalRAG
        
        rag = MultimodalRAG()
        
        # 创建测试图像
        test_image_path = create_test_image()
        if not test_image_path:
            return False
        
        # 添加图像文档
        doc_id = rag.add_image_document(test_image_path, "这是一个测试图像，包含红色圆形和绿色矩形")
        logger.info(f"✓ 图像文档添加成功，ID: {doc_id}")
        
        # 搜索测试
        search_results = rag.search_multimodal("红色圆形", include_images=True, k=3)
        logger.info(f"✓ 搜索完成，找到 {len(search_results)} 个结果")
        
        for i, result in enumerate(search_results):
            logger.info(f"  结果 {i+1}: {result['type']} - {result['content'][:50]}...")
        
        # 生成回答
        response = rag.generate_multimodal_response("图片里有什么形状？", search_results)
        logger.info(f"✓ 回答生成: {response[:100]}...")
        
        # 获取统计信息
        stats = rag.get_multimodal_stats()
        logger.info(f"✓ 统计信息: {stats}")
        
        # 清理测试文件
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
        
        logger.info("✓ 多模态RAG测试通过")
        return True
        
    except Exception as e:
        logger.error(f"✗ 多模态RAG测试失败: {e}")
        return False

def test_multimodal_manager():
    """测试多模态管理器"""
    logger.info("测试多模态管理器...")
    
    try:
        from multimodal_system.multimodal_manager import MultimodalManager
        from werkzeug.datastructures import FileStorage
        import io
        
        manager = MultimodalManager()
        
        # 创建测试图像
        test_image_path = create_test_image()
        if not test_image_path:
            return False
        
        # 模拟文件上传
        with open(test_image_path, 'rb') as f:
            file_data = f.read()
        
        # 创建FileStorage对象
        file_obj = FileStorage(
            stream=io.BytesIO(file_data),
            filename="test_image.jpg",
            content_type="image/jpeg"
        )
        
        # 上传和处理图像
        upload_result = manager.upload_and_process_image(
            file_obj, 
            "测试图像描述",
            add_to_knowledge_base=True
        )
        
        if upload_result["success"]:
            logger.info("✓ 图像上传处理成功")
            image_id = upload_result["image_id"]
            logger.info(f"  图像ID: {image_id}")
            logger.info(f"  描述: {upload_result['description']}")
        else:
            logger.error(f"✗ 图像上传处理失败: {upload_result['error']}")
            return False
        
        # 测试图像问答
        qa_result = manager.analyze_image_with_question(image_id, "这张图片的主要颜色是什么？")
        if qa_result["success"]:
            logger.info("✓ 图像问答成功")
            logger.info(f"  回答: {qa_result['answer']}")
        else:
            logger.warning(f"图像问答失败: {qa_result['error']}")
        
        # 测试多模态搜索
        search_result = manager.search_multimodal_content("测试图像")
        if search_result["success"]:
            logger.info("✓ 多模态搜索成功")
            logger.info(f"  找到 {search_result['result_count']} 个结果")
        else:
            logger.warning(f"多模态搜索失败: {search_result['error']}")
        
        # 获取系统状态
        status = manager.get_system_status()
        if status["success"]:
            logger.info("✓ 系统状态获取成功")
            logger.info(f"  处理图像数: {status['processed_images_count']}")
        
        # 清理测试数据
        delete_result = manager.delete_image(image_id)
        if delete_result["success"]:
            logger.info("✓ 测试图像清理成功")
        
        # 清理测试文件
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
        
        logger.info("✓ 多模态管理器测试通过")
        return True
        
    except Exception as e:
        logger.error(f"✗ 多模态管理器测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("=" * 60)
    logger.info("Live2D AI助手 多模态系统测试")
    logger.info("=" * 60)
    
    tests = [
        ("图像处理器", test_image_processor),
        ("图像分析器", test_image_analyzer),
        ("视觉语言模型", test_vision_language_model),
        ("多模态RAG", test_multimodal_rag),
        ("多模态管理器", test_multimodal_manager)
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
        logger.info("🎉 所有测试通过！多模态系统功能正常")
        return 0
    else:
        logger.warning("⚠️  部分测试失败，请检查相关问题")
        return 1

if __name__ == "__main__":
    sys.exit(main())
