#!/usr/bin/env python3
"""
语音情感系统测试脚本

测试语音情感合成系统的各个组件功能
"""

import sys
import asyncio
import logging
from typing import Dict, Any

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_voice_emotion_config():
    """测试语音情感配置"""
    logger.info("测试语音情感配置...")
    
    try:
        from voice_emotion_system.voice_emotion_config import VoiceEmotionConfig, EmotionType, VoiceLanguage
        
        config = VoiceEmotionConfig()
        
        # 测试情感参数获取
        emotion_params = config.get_emotion_params(EmotionType.HAPPY, 0.8)
        logger.info(f"开心情感参数: {emotion_params}")
        
        # 测试语言检测
        chinese_text = "你好，今天天气很好"
        detected_language = config.detect_language(chinese_text)
        logger.info(f"语言检测结果: {detected_language}")
        
        # 测试语音配置获取
        chinese_voices = config.get_voice_by_language(VoiceLanguage.CHINESE)
        logger.info(f"中文语音数量: {len(chinese_voices)}")
        
        logger.info("✓ 语音情感配置测试通过")
        return True
        
    except Exception as e:
        logger.error(f"✗ 语音情感配置测试失败: {e}")
        return False

def test_emotion_voice_synthesizer():
    """测试情感语音合成器"""
    logger.info("测试情感语音合成器...")
    
    try:
        from voice_emotion_system.emotion_voice_synthesizer import EmotionVoiceSynthesizer
        from voice_emotion_system.voice_emotion_config import EmotionType, VoiceLanguage
        
        synthesizer = EmotionVoiceSynthesizer()
        
        # 测试语音合成
        test_text = "你好，这是语音合成测试。"
        
        async def test_synthesis():
            result = await synthesizer.synthesize_with_emotion(
                test_text, 
                EmotionType.HAPPY, 
                0.8,
                "xiaoxiao",
                VoiceLanguage.CHINESE
            )
            return result
        
        # 运行异步测试
        result = asyncio.run(test_synthesis())
        
        if result["success"]:
            logger.info("✓ 语音合成成功")
            logger.info(f"  文件路径: {result['file_path']}")
            logger.info(f"  情感: {result['emotion']}")
            logger.info(f"  强度: {result['intensity']}")
        else:
            logger.error(f"✗ 语音合成失败: {result['error']}")
            return False
        
        # 测试获取可用语音
        voices = synthesizer.get_available_voices()
        logger.info(f"可用语音数量: {len(voices['voices'])}")
        
        # 测试获取支持的情感
        emotions = synthesizer.get_supported_emotions()
        logger.info(f"支持的情感数量: {len(emotions)}")
        
        logger.info("✓ 情感语音合成器测试通过")
        return True
        
    except Exception as e:
        logger.error(f"✗ 情感语音合成器测试失败: {e}")
        return False

def test_voice_style_controller():
    """测试语音风格控制器"""
    logger.info("测试语音风格控制器...")
    
    try:
        from voice_emotion_system.voice_style_controller import VoiceStyleController
        from voice_emotion_system.voice_emotion_config import EmotionType, VoiceLanguage
        
        controller = VoiceStyleController()
        
        # 测试设置风格
        result = controller.set_style("gentle")
        if result["success"]:
            logger.info(f"✓ 风格设置成功: {result['description']}")
        else:
            logger.error(f"✗ 风格设置失败: {result['error']}")
            return False
        
        # 测试获取当前风格
        current_style = controller.get_current_style()
        logger.info(f"当前风格: {current_style['name']}")
        
        # 测试情感强度调整
        adjusted_intensity = controller.adjust_emotion_intensity(EmotionType.HAPPY, 1.0)
        logger.info(f"调整后的情感强度: {adjusted_intensity}")
        
        # 测试获取语音推荐
        voice_name = controller.get_voice_for_language(VoiceLanguage.CHINESE)
        logger.info(f"推荐语音: {voice_name}")
        
        # 测试获取所有风格
        all_styles = controller.get_all_styles()
        logger.info(f"可用风格数量: {len(all_styles['styles'])}")
        
        logger.info("✓ 语音风格控制器测试通过")
        return True
        
    except Exception as e:
        logger.error(f"✗ 语音风格控制器测试失败: {e}")
        return False

def test_multilingual_voice_manager():
    """测试多语言语音管理器"""
    logger.info("测试多语言语音管理器...")
    
    try:
        from voice_emotion_system.multilingual_voice_manager import MultilingualVoiceManager
        from voice_emotion_system.voice_emotion_config import EmotionType, VoiceLanguage
        
        manager = MultilingualVoiceManager()
        
        # 测试语言检测
        test_texts = [
            "你好，今天天气很好",
            "Hello, how are you today?",
            "你好 Hello 混合语言 mixed language"
        ]
        
        for text in test_texts:
            detection_result = manager.detect_language(text)
            logger.info(f"文本: '{text}' -> 语言: {detection_result['language'].value}, 置信度: {detection_result['confidence']:.2f}")
            
            if detection_result['mixed_language']:
                logger.info(f"  混合语言段落数: {len(detection_result['segments'])}")
        
        # 测试跨语言情感调整
        adjusted_intensity = manager.adjust_emotion_for_language(
            EmotionType.HAPPY, 1.0, VoiceLanguage.CHINESE, VoiceLanguage.ENGLISH
        )
        logger.info(f"跨语言情感强度调整: 1.0 -> {adjusted_intensity}")
        
        # 测试语音推荐
        recommendations = manager.get_voice_recommendations(VoiceLanguage.CHINESE, EmotionType.HAPPY)
        logger.info(f"语音推荐: {recommendations}")
        
        # 测试支持的语言
        languages = manager.get_supported_languages()
        logger.info(f"支持的语言数量: {len(languages)}")
        
        logger.info("✓ 多语言语音管理器测试通过")
        return True
        
    except Exception as e:
        logger.error(f"✗ 多语言语音管理器测试失败: {e}")
        return False

def test_realtime_voice_processor():
    """测试实时语音处理器"""
    logger.info("测试实时语音处理器...")
    
    try:
        from voice_emotion_system.realtime_voice_processor import RealtimeVoiceProcessor
        from voice_emotion_system.emotion_voice_synthesizer import EmotionVoiceSynthesizer
        from voice_emotion_system.voice_emotion_config import EmotionType
        import time
        
        # 创建合成器和处理器
        synthesizer = EmotionVoiceSynthesizer()
        processor = RealtimeVoiceProcessor(synthesizer)
        
        # 启动处理器
        processor.start()
        
        # 提交测试任务
        task_id = processor.submit_synthesis_task(
            "这是实时处理测试", 
            EmotionType.HAPPY, 
            0.8, 
            priority=1
        )
        
        if task_id:
            logger.info(f"✓ 任务提交成功: {task_id}")
        else:
            logger.error("✗ 任务提交失败")
            return False
        
        # 等待任务完成
        max_wait = 10  # 最多等待10秒
        wait_time = 0
        
        while wait_time < max_wait:
            status = processor.get_task_status(task_id)
            logger.info(f"任务状态: {status['status']}")
            
            if status['status'] in ['completed', 'failed']:
                break
            
            time.sleep(1)
            wait_time += 1
        
        # 获取最终状态
        final_status = processor.get_task_status(task_id)
        if final_status['status'] == 'completed':
            logger.info("✓ 任务完成成功")
        else:
            logger.warning(f"任务最终状态: {final_status['status']}")
        
        # 获取队列状态
        queue_status = processor.get_queue_status()
        logger.info(f"队列状态: {queue_status}")
        
        # 获取性能统计
        performance_stats = processor.get_performance_stats()
        logger.info(f"性能统计: {performance_stats}")
        
        # 停止处理器
        processor.stop()
        
        logger.info("✓ 实时语音处理器测试通过")
        return True
        
    except Exception as e:
        logger.error(f"✗ 实时语音处理器测试失败: {e}")
        return False

def test_voice_emotion_manager():
    """测试语音情感管理器"""
    logger.info("测试语音情感管理器...")
    
    try:
        from voice_emotion_system.voice_emotion_manager import VoiceEmotionManager
        from voice_emotion_system.voice_emotion_config import EmotionType, VoiceLanguage
        
        manager = VoiceEmotionManager()
        
        # 测试语音合成
        async def test_synthesis():
            result = await manager.synthesize_with_emotion(
                "你好，这是管理器测试", 
                EmotionType.EXCITED, 
                0.9
            )
            return result
        
        result = asyncio.run(test_synthesis())
        
        if result["success"]:
            logger.info("✓ 管理器语音合成成功")
            logger.info(f"  风格信息: {result.get('style_info', {})}")
        else:
            logger.error(f"✗ 管理器语音合成失败: {result['error']}")
            return False
        
        # 测试实时合成
        task_id = manager.synthesize_realtime("实时合成测试", EmotionType.CALM, 0.7)
        if task_id:
            logger.info(f"✓ 实时合成任务提交成功: {task_id}")
        else:
            logger.error("✗ 实时合成任务提交失败")
        
        # 测试设置
        style_result = manager.set_voice_style("energetic")
        if style_result["success"]:
            logger.info("✓ 风格设置成功")
        
        voice_result = manager.set_default_voice("yunxi", VoiceLanguage.CHINESE)
        if voice_result["success"]:
            logger.info("✓ 默认语音设置成功")
        
        # 测试获取信息
        voices = manager.get_available_voices()
        logger.info(f"可用语音: {voices['current_voice']}")
        
        styles = manager.get_available_styles()
        logger.info(f"当前风格: {styles['current_style']}")
        
        emotions = manager.get_supported_emotions()
        logger.info(f"支持情感数量: {len(emotions)}")
        
        languages = manager.get_supported_languages()
        logger.info(f"支持语言数量: {len(languages)}")
        
        # 测试系统状态
        status = manager.get_system_status()
        if "error" not in status:
            logger.info("✓ 系统状态获取成功")
            logger.info(f"  系统活跃: {status['is_active']}")
        else:
            logger.warning(f"系统状态获取失败: {status['error']}")
        
        # 测试语音测试功能
        async def test_voice_test():
            test_result = await manager.test_voice_synthesis("测试语音功能")
            return test_result
        
        test_result = asyncio.run(test_voice_test())
        if test_result["success"]:
            logger.info("✓ 语音测试功能正常")
        
        logger.info("✓ 语音情感管理器测试通过")
        return True
        
    except Exception as e:
        logger.error(f"✗ 语音情感管理器测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("=" * 60)
    logger.info("Live2D AI助手 语音情感系统测试")
    logger.info("=" * 60)
    
    tests = [
        ("语音情感配置", test_voice_emotion_config),
        ("情感语音合成器", test_emotion_voice_synthesizer),
        ("语音风格控制器", test_voice_style_controller),
        ("多语言语音管理器", test_multilingual_voice_manager),
        ("实时语音处理器", test_realtime_voice_processor),
        ("语音情感管理器", test_voice_emotion_manager)
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
        logger.info("🎉 所有测试通过！语音情感系统功能正常")
        return 0
    else:
        logger.warning("⚠️  部分测试失败，请检查相关问题")
        return 1

if __name__ == "__main__":
    sys.exit(main())
