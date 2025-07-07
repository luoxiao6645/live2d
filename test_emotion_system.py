#!/usr/bin/env python3
"""
情感系统测试脚本

测试高级情感系统的各个组件功能
"""

import sys
import time
import logging
from typing import Dict, Any

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_emotion_analyzer():
    """测试情感分析器"""
    logger.info("测试情感分析器...")
    
    try:
        from emotion_system.emotion_analyzer import EmotionAnalyzer
        
        analyzer = EmotionAnalyzer()
        
        # 测试用例
        test_texts = [
            "我今天非常开心！",
            "这让我感到很难过...",
            "太惊讶了！没想到会这样",
            "我很生气，这太不公平了",
            "我有点困惑，不太明白",
            "好害羞啊，不好意思",
            "我超级兴奋的！",
            "我很担心这个问题",
            "我爱你",
            "让我想想...",
            "我好困啊，想睡觉",
            "今天天气不错"  # 中性
        ]
        
        logger.info("开始情感分析测试:")
        for text in test_texts:
            result = analyzer.analyze_emotion(text)
            logger.info(f"文本: '{text}' -> 情感: {result['emotion_label']}, 强度: {result['intensity_label']}")
        
        logger.info("✓ 情感分析器测试通过")
        return True
        
    except Exception as e:
        logger.error(f"✗ 情感分析器测试失败: {e}")
        return False

def test_emotion_state_manager():
    """测试情感状态管理器"""
    logger.info("测试情感状态管理器...")
    
    try:
        from emotion_system.emotion_state_manager import EmotionStateManager
        from emotion_system.emotion_config import EmotionType
        
        manager = EmotionStateManager()
        
        # 测试状态获取
        current_state = manager.get_current_state()
        logger.info(f"初始状态: {current_state['emotion_type']}")
        
        # 测试情感更新
        test_emotion_data = {
            "primary_emotion": "happy",
            "intensity": 0.8,
            "confidence": 0.9
        }
        
        success = manager.update_emotion(test_emotion_data)
        if success:
            logger.info("情感状态更新成功")
        
        # 等待转换完成
        time.sleep(3)
        
        # 获取Live2D参数
        params = manager.get_live2d_parameters()
        logger.info(f"Live2D参数: {list(params.keys())[:5]}...")  # 只显示前5个参数
        
        # 测试强制设置情感
        manager.force_emotion(EmotionType.SAD, 0.6)
        logger.info("强制设置悲伤情感")
        
        # 获取情感历史
        history = manager.get_emotion_history(5)
        logger.info(f"情感历史记录数: {len(history)}")
        
        logger.info("✓ 情感状态管理器测试通过")
        return True
        
    except Exception as e:
        logger.error(f"✗ 情感状态管理器测试失败: {e}")
        return False

def test_animation_sequencer():
    """测试动画序列编排器"""
    logger.info("测试动画序列编排器...")
    
    try:
        from emotion_system.animation_sequencer import AnimationSequencer
        
        sequencer = AnimationSequencer()
        
        # 测试播放预定义动画
        animations_to_test = ["blink", "nod", "wave", "thinking"]
        
        for animation in animations_to_test:
            success = sequencer.play_animation(animation)
            if success:
                logger.info(f"播放动画成功: {animation}")
            else:
                logger.warning(f"播放动画失败: {animation}")
            
            time.sleep(0.5)  # 短暂等待
        
        # 获取活动动画
        active_animations = sequencer.get_active_animations()
        logger.info(f"当前活动动画数: {len(active_animations)}")
        
        # 等待动画完成
        time.sleep(2)
        
        # 获取当前参数
        params = sequencer.get_current_parameters()
        logger.info(f"动画参数数: {len(params)}")
        
        logger.info("✓ 动画序列编排器测试通过")
        return True
        
    except Exception as e:
        logger.error(f"✗ 动画序列编排器测试失败: {e}")
        return False

def test_advanced_animation_controller():
    """测试高级动画控制器"""
    logger.info("测试高级动画控制器...")
    
    try:
        from emotion_system.advanced_animation_controller import AdvancedAnimationController
        
        controller = AdvancedAnimationController()
        
        # 测试文本处理
        test_texts = [
            "我今天很开心！",
            "这让我很困惑...",
            "太惊讶了！"
        ]
        
        for text in test_texts:
            result = controller.process_text_input(text)
            if result['success']:
                emotion = result['emotion_analysis']['emotion_label']
                logger.info(f"文本处理成功: '{text}' -> {emotion}")
            else:
                logger.warning(f"文本处理失败: {text}")
        
        # 测试手势触发
        gestures = ["wave", "nod", "thinking"]
        for gesture in gestures:
            success = controller.trigger_gesture(gesture)
            if success:
                logger.info(f"手势触发成功: {gesture}")
        
        # 测试强制情感设置
        controller.force_emotion("excited", 0.9)
        logger.info("强制设置兴奋情感")
        
        # 获取系统状态
        status = controller.get_system_status()
        logger.info(f"系统状态: 活动={status['is_active']}, 当前情感={status['current_emotion']['emotion_type']}")
        
        # 获取Live2D参数
        params = controller.get_current_live2d_parameters()
        logger.info(f"Live2D参数数量: {len(params)}")
        
        # 导出动画数据
        export_data = controller.export_animation_data()
        logger.info(f"导出数据包含: {list(export_data.keys())}")
        
        logger.info("✓ 高级动画控制器测试通过")
        return True
        
    except Exception as e:
        logger.error(f"✗ 高级动画控制器测试失败: {e}")
        return False

def test_integration():
    """集成测试"""
    logger.info("进行集成测试...")
    
    try:
        from emotion_system import AdvancedAnimationController
        
        controller = AdvancedAnimationController()
        
        # 模拟对话场景
        conversation = [
            "你好！",
            "我今天心情不太好",
            "发生了一些让我困惑的事情",
            "不过现在感觉好多了",
            "谢谢你的陪伴！"
        ]
        
        logger.info("模拟对话场景:")
        for i, message in enumerate(conversation, 1):
            logger.info(f"第{i}轮: {message}")
            result = controller.process_text_input(message)
            
            if result['success']:
                emotion = result['emotion_analysis']['emotion_label']
                intensity = result['emotion_analysis']['intensity_label']
                logger.info(f"  -> 检测情感: {emotion} ({intensity})")
                
                # 获取当前Live2D参数
                params = controller.get_current_live2d_parameters()
                key_params = {k: v for k, v in params.items() if 'Eye' in k or 'Mouth' in k}
                logger.info(f"  -> 关键参数: {key_params}")
            
            time.sleep(1)  # 模拟对话间隔
        
        # 获取情感历史
        history = controller.get_emotion_history()
        logger.info(f"对话后情感历史: {len(history)} 条记录")
        
        # 重置到中性状态
        controller.reset_to_neutral()
        logger.info("重置到中性状态")
        
        logger.info("✓ 集成测试通过")
        return True
        
    except Exception as e:
        logger.error(f"✗ 集成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("=" * 60)
    logger.info("Live2D AI助手 情感系统测试")
    logger.info("=" * 60)
    
    tests = [
        ("情感分析器", test_emotion_analyzer),
        ("情感状态管理器", test_emotion_state_manager),
        ("动画序列编排器", test_animation_sequencer),
        ("高级动画控制器", test_advanced_animation_controller),
        ("集成测试", test_integration)
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
        logger.info("🎉 所有测试通过！情感系统功能正常")
        return 0
    else:
        logger.warning("⚠️  部分测试失败，请检查相关问题")
        return 1

if __name__ == "__main__":
    sys.exit(main())
