"""
多模态系统配置文件

定义了图像处理、模型配置、API设置等核心配置信息
"""

from typing import Dict, List, Tuple, Any
from enum import Enum
import os

class ImageFormat(Enum):
    """支持的图像格式"""
    JPEG = "jpeg"
    JPG = "jpg"
    PNG = "png"
    GIF = "gif"
    WEBP = "webp"
    BMP = "bmp"

class VisionModelType(Enum):
    """视觉模型类型"""
    OPENAI_GPT4V = "openai_gpt4v"
    BLIP2 = "blip2"
    CLIP = "clip"
    SIMPLE_TAGS = "simple_tags"

class MultimodalConfig:
    """多模态系统配置类"""
    
    # 图像处理配置
    IMAGE_CONFIG = {
        "max_file_size": 10 * 1024 * 1024,  # 10MB
        "max_width": 2048,
        "max_height": 2048,
        "quality": 85,  # JPEG质量
        "supported_formats": [fmt.value for fmt in ImageFormat],
        "thumbnail_size": (256, 256),
        "upload_directory": "./uploads/images",
        "processed_directory": "./uploads/processed"
    }
    
    # 视觉模型配置
    VISION_MODEL_CONFIG = {
        "default_model": VisionModelType.SIMPLE_TAGS,
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "openai_model": "gpt-4-vision-preview",
        "openai_max_tokens": 300,
        "blip2_model": "Salesforce/blip2-opt-2.7b",
        "clip_model": "ViT-B/32",
        "device": "auto"  # auto, cpu, cuda
    }
    
    # 图像分析提示模板
    ANALYSIS_PROMPTS = {
        "general_description": """
请详细描述这张图片的内容，包括：
1. 主要物体和人物
2. 场景和环境
3. 颜色和光线
4. 情感和氛围
5. 任何值得注意的细节

请用中文回答，保持简洁明了。
""",
        
        "emotion_analysis": """
分析这张图片中的情感表达，包括：
1. 人物的表情和情绪
2. 整体氛围和感觉
3. 色彩给人的情感感受
4. 场景传达的情绪

请用中文回答。
""",
        
        "object_detection": """
识别图片中的所有物体，列出：
1. 主要物体名称
2. 物体的位置和大小
3. 物体的颜色和特征
4. 物体之间的关系

请用中文回答。
""",
        
        "scene_understanding": """
理解图片的场景内容：
1. 这是什么类型的场景？
2. 发生在什么地方？
3. 可能的时间（白天/夜晚/季节）
4. 场景中正在发生什么？

请用中文回答。
"""
    }
    
    # 简单标签识别配置
    SIMPLE_TAGS_CONFIG = {
        "common_objects": [
            "人", "男人", "女人", "孩子", "婴儿",
            "猫", "狗", "鸟", "动物",
            "汽车", "自行车", "摩托车", "飞机", "船",
            "房子", "建筑", "桥", "道路", "树",
            "花", "草", "山", "海", "河", "湖",
            "食物", "水果", "蔬菜", "饮料",
            "书", "电脑", "手机", "电视",
            "椅子", "桌子", "床", "沙发",
            "天空", "云", "太阳", "月亮", "星星",
            "雨", "雪", "风", "彩虹"
        ],
        
        "colors": [
            "红色", "蓝色", "绿色", "黄色", "紫色",
            "橙色", "粉色", "黑色", "白色", "灰色",
            "棕色", "金色", "银色"
        ],
        
        "emotions": [
            "开心", "悲伤", "愤怒", "惊讶", "恐惧",
            "厌恶", "平静", "兴奋", "紧张", "放松"
        ],
        
        "scenes": [
            "室内", "室外", "城市", "乡村", "海边",
            "山区", "森林", "公园", "街道", "商店",
            "学校", "医院", "餐厅", "办公室", "家"
        ]
    }
    
    # 多模态RAG配置
    MULTIMODAL_RAG_CONFIG = {
        "image_embedding_dim": 512,
        "text_embedding_dim": 384,
        "fusion_method": "concatenate",  # concatenate, attention, cross_modal
        "similarity_threshold": 0.7,
        "max_retrieved_items": 5,
        "rerank_enabled": True
    }
    
    # API配置
    API_CONFIG = {
        "max_concurrent_requests": 5,
        "request_timeout": 30,
        "retry_attempts": 3,
        "rate_limit": {
            "requests_per_minute": 60,
            "requests_per_hour": 1000
        }
    }
    
    # 缓存配置
    CACHE_CONFIG = {
        "enable_image_cache": True,
        "enable_analysis_cache": True,
        "cache_ttl": 3600,  # 1小时
        "max_cache_size": 1000,
        "cache_directory": "./cache/multimodal"
    }
    
    # 安全配置
    SECURITY_CONFIG = {
        "content_filter_enabled": True,
        "allowed_mime_types": [
            "image/jpeg", "image/jpg", "image/png", 
            "image/gif", "image/webp", "image/bmp"
        ],
        "blocked_extensions": [".exe", ".bat", ".sh", ".php"],
        "scan_for_malware": False,  # 需要额外的安全库
        "privacy_mode": True  # 不保存敏感图像
    }
    
    # 错误消息
    ERROR_MESSAGES = {
        "file_too_large": "图片文件太大，请选择小于10MB的图片",
        "invalid_format": "不支持的图片格式，请使用JPEG、PNG、GIF或WebP格式",
        "processing_failed": "图片处理失败，请重试",
        "analysis_failed": "图片分析失败，请检查图片内容",
        "model_not_available": "视觉模型不可用，请稍后重试",
        "api_key_missing": "API密钥缺失，请配置相应的API密钥",
        "network_error": "网络连接错误，请检查网络设置",
        "quota_exceeded": "API调用次数超限，请稍后重试"
    }
    
    # 成功消息
    SUCCESS_MESSAGES = {
        "upload_success": "图片上传成功",
        "analysis_success": "图片分析完成",
        "processing_success": "图片处理完成",
        "cache_hit": "从缓存获取结果"
    }
    
    @classmethod
    def get_upload_path(cls) -> str:
        """获取上传目录路径"""
        path = cls.IMAGE_CONFIG["upload_directory"]
        os.makedirs(path, exist_ok=True)
        return path
    
    @classmethod
    def get_processed_path(cls) -> str:
        """获取处理后图片目录路径"""
        path = cls.IMAGE_CONFIG["processed_directory"]
        os.makedirs(path, exist_ok=True)
        return path
    
    @classmethod
    def get_cache_path(cls) -> str:
        """获取缓存目录路径"""
        path = cls.CACHE_CONFIG["cache_directory"]
        os.makedirs(path, exist_ok=True)
        return path
    
    @classmethod
    def is_supported_format(cls, filename: str) -> bool:
        """检查文件格式是否支持"""
        ext = os.path.splitext(filename)[1].lower().lstrip('.')
        return ext in cls.IMAGE_CONFIG["supported_formats"]
    
    @classmethod
    def get_model_config(cls, model_type: VisionModelType) -> Dict[str, Any]:
        """获取指定模型的配置"""
        config = cls.VISION_MODEL_CONFIG.copy()
        config["model_type"] = model_type
        return config
    
    @classmethod
    def validate_image_size(cls, width: int, height: int) -> bool:
        """验证图片尺寸是否符合要求"""
        max_width = cls.IMAGE_CONFIG["max_width"]
        max_height = cls.IMAGE_CONFIG["max_height"]
        return width <= max_width and height <= max_height
    
    @classmethod
    def get_thumbnail_size(cls) -> Tuple[int, int]:
        """获取缩略图尺寸"""
        return cls.IMAGE_CONFIG["thumbnail_size"]
