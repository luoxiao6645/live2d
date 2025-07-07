"""
情感分析器

负责分析文本内容中的情感信息，支持多种分析方法：
1. 基于关键词的快速分析
2. 基于机器学习模型的深度分析
3. 多维度情感分析
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import numpy as np

from .emotion_config import EmotionType, EmotionConfig

# 配置日志
logger = logging.getLogger(__name__)

class EmotionAnalyzer:
    """情感分析器类"""
    
    def __init__(self, use_ml_model: bool = False):
        """
        初始化情感分析器
        
        Args:
            use_ml_model: 是否使用机器学习模型进行分析
        """
        self.use_ml_model = use_ml_model
        self.config = EmotionConfig()
        
        # 初始化关键词权重
        self._init_keyword_weights()
        
        # 如果启用ML模型，尝试加载
        if use_ml_model:
            self._init_ml_model()
        
        logger.info("情感分析器初始化完成")
    
    def _init_keyword_weights(self):
        """初始化关键词权重"""
        self.keyword_weights = {}
        
        for emotion_type, keywords in self.config.EMOTION_KEYWORDS.items():
            self.keyword_weights[emotion_type] = {}
            for keyword in keywords:
                # 根据关键词长度和特征设置权重
                if len(keyword) >= 3:
                    weight = 1.0  # 长关键词权重高
                elif keyword in ['爱', '恨', 'love', 'hate']:
                    weight = 1.2  # 强情感词权重高
                else:
                    weight = 0.8  # 短关键词权重稍低
                
                self.keyword_weights[emotion_type][keyword] = weight
    
    def _init_ml_model(self):
        """初始化机器学习模型"""
        try:
            # 这里可以集成transformers等库的情感分析模型
            # 由于依赖较重，暂时使用占位符
            self.ml_model = None
            logger.info("ML模型初始化完成（占位符）")
        except Exception as e:
            logger.warning(f"ML模型初始化失败: {e}")
            self.ml_model = None
            self.use_ml_model = False
    
    def analyze_emotion(self, text: str) -> Dict[str, Any]:
        """
        分析文本情感
        
        Args:
            text: 要分析的文本
            
        Returns:
            情感分析结果字典
        """
        if not text or not text.strip():
            return self._create_neutral_result()
        
        # 文本预处理
        processed_text = self._preprocess_text(text)
        
        # 基于关键词的分析
        keyword_result = self._analyze_by_keywords(processed_text)
        
        # 如果启用ML模型，进行深度分析
        if self.use_ml_model and self.ml_model:
            ml_result = self._analyze_by_ml_model(processed_text)
            # 融合两种分析结果
            final_result = self._merge_analysis_results(keyword_result, ml_result)
        else:
            final_result = keyword_result
        
        # 后处理和规范化
        final_result = self._post_process_result(final_result)
        
        logger.debug(f"情感分析结果: {final_result}")
        return final_result
    
    def _preprocess_text(self, text: str) -> str:
        """文本预处理"""
        # 转换为小写
        text = text.lower()
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 移除特殊字符（保留中文、英文、数字和基本标点）
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s.,!?;:]', '', text)
        
        return text
    
    def _analyze_by_keywords(self, text: str) -> Dict[str, Any]:
        """基于关键词的情感分析"""
        emotion_scores = defaultdict(float)
        matched_keywords = defaultdict(list)
        
        # 遍历所有情感类型的关键词
        for emotion_type, keywords in self.config.EMOTION_KEYWORDS.items():
            for keyword in keywords:
                # 计算关键词在文本中的出现次数
                count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE))
                if count > 0:
                    weight = self.keyword_weights[emotion_type][keyword]
                    score = count * weight
                    emotion_scores[emotion_type] += score
                    matched_keywords[emotion_type].extend([keyword] * count)
        
        # 如果没有匹配到任何关键词，返回中性情感
        if not emotion_scores:
            return self._create_neutral_result()
        
        # 计算情感强度和主要情感
        total_score = sum(emotion_scores.values())
        emotion_probabilities = {
            emotion_type.value: score / total_score 
            for emotion_type, score in emotion_scores.items()
        }
        
        # 找到主要情感
        primary_emotion = max(emotion_scores.keys(), key=lambda x: emotion_scores[x])
        primary_intensity = min(emotion_scores[primary_emotion] / 5.0, 1.0)  # 归一化到0-1
        
        return {
            "primary_emotion": primary_emotion.value,
            "intensity": primary_intensity,
            "emotion_probabilities": emotion_probabilities,
            "matched_keywords": dict(matched_keywords),
            "analysis_method": "keyword_based",
            "confidence": min(primary_intensity * 1.2, 1.0)
        }
    
    def _analyze_by_ml_model(self, text: str) -> Dict[str, Any]:
        """基于机器学习模型的情感分析"""
        # 这里是ML模型分析的占位符
        # 实际实现中可以集成BERT、RoBERTa等预训练模型
        
        # 模拟ML模型输出
        mock_result = {
            "primary_emotion": EmotionType.NEUTRAL.value,
            "intensity": 0.5,
            "emotion_probabilities": {
                EmotionType.NEUTRAL.value: 0.7,
                EmotionType.HAPPY.value: 0.2,
                EmotionType.CONFUSED.value: 0.1
            },
            "analysis_method": "ml_model",
            "confidence": 0.8
        }
        
        return mock_result
    
    def _merge_analysis_results(self, keyword_result: Dict, ml_result: Dict) -> Dict:
        """融合关键词分析和ML模型分析结果"""
        # 简单的加权融合策略
        keyword_weight = 0.4
        ml_weight = 0.6
        
        # 融合情感概率
        merged_probabilities = defaultdict(float)
        
        for emotion, prob in keyword_result.get("emotion_probabilities", {}).items():
            merged_probabilities[emotion] += prob * keyword_weight
        
        for emotion, prob in ml_result.get("emotion_probabilities", {}).items():
            merged_probabilities[emotion] += prob * ml_weight
        
        # 找到主要情感
        primary_emotion = max(merged_probabilities.keys(), key=lambda x: merged_probabilities[x])
        
        # 计算融合后的强度和置信度
        merged_intensity = (
            keyword_result.get("intensity", 0) * keyword_weight +
            ml_result.get("intensity", 0) * ml_weight
        )
        
        merged_confidence = (
            keyword_result.get("confidence", 0) * keyword_weight +
            ml_result.get("confidence", 0) * ml_weight
        )
        
        return {
            "primary_emotion": primary_emotion,
            "intensity": merged_intensity,
            "emotion_probabilities": dict(merged_probabilities),
            "matched_keywords": keyword_result.get("matched_keywords", {}),
            "analysis_method": "hybrid",
            "confidence": merged_confidence
        }
    
    def _post_process_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """后处理分析结果"""
        # 确保强度在合理范围内
        result["intensity"] = max(0.0, min(1.0, result["intensity"]))
        result["confidence"] = max(0.0, min(1.0, result["confidence"]))
        
        # 如果强度太低，设为中性情感
        if result["intensity"] < 0.1:
            result["primary_emotion"] = EmotionType.NEUTRAL.value
            result["intensity"] = 0.1
        
        # 添加情感标签的中文名称
        emotion_labels = {
            "neutral": "中性", "happy": "开心", "sad": "悲伤", "angry": "生气",
            "surprised": "惊讶", "confused": "困惑", "shy": "害羞", "excited": "兴奋",
            "worried": "担心", "loving": "喜爱", "thinking": "思考", "sleepy": "困倦"
        }
        
        result["emotion_label"] = emotion_labels.get(result["primary_emotion"], "未知")
        
        # 添加强度描述
        intensity = result["intensity"]
        if intensity < 0.3:
            result["intensity_label"] = "轻微"
        elif intensity < 0.6:
            result["intensity_label"] = "中等"
        elif intensity < 0.8:
            result["intensity_label"] = "强烈"
        else:
            result["intensity_label"] = "非常强烈"
        
        return result
    
    def _create_neutral_result(self) -> Dict[str, Any]:
        """创建中性情感结果"""
        return {
            "primary_emotion": EmotionType.NEUTRAL.value,
            "emotion_label": "中性",
            "intensity": 0.1,
            "intensity_label": "轻微",
            "emotion_probabilities": {EmotionType.NEUTRAL.value: 1.0},
            "matched_keywords": {},
            "analysis_method": "default",
            "confidence": 1.0
        }
    
    def analyze_emotion_sequence(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        分析文本序列的情感变化
        
        Args:
            texts: 文本列表
            
        Returns:
            情感分析结果列表
        """
        results = []
        for text in texts:
            result = self.analyze_emotion(text)
            results.append(result)
        
        # 添加情感变化趋势分析
        if len(results) > 1:
            for i in range(1, len(results)):
                prev_emotion = results[i-1]["primary_emotion"]
                curr_emotion = results[i]["primary_emotion"]
                
                if prev_emotion != curr_emotion:
                    results[i]["emotion_change"] = f"{prev_emotion} -> {curr_emotion}"
                else:
                    results[i]["emotion_change"] = "stable"
        
        return results
    
    def get_emotion_statistics(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        获取情感分析统计信息
        
        Args:
            analysis_results: 情感分析结果列表
            
        Returns:
            统计信息字典
        """
        if not analysis_results:
            return {}
        
        # 统计各种情感的出现次数
        emotion_counts = defaultdict(int)
        total_intensity = 0
        
        for result in analysis_results:
            emotion_counts[result["primary_emotion"]] += 1
            total_intensity += result["intensity"]
        
        # 计算主导情感
        dominant_emotion = max(emotion_counts.keys(), key=lambda x: emotion_counts[x])
        
        # 计算平均强度
        avg_intensity = total_intensity / len(analysis_results)
        
        return {
            "total_analyses": len(analysis_results),
            "emotion_distribution": dict(emotion_counts),
            "dominant_emotion": dominant_emotion,
            "average_intensity": avg_intensity,
            "emotion_changes": sum(1 for r in analysis_results if r.get("emotion_change", "stable") != "stable")
        }
