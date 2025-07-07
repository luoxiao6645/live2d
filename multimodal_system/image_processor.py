"""
图像处理器

负责图像的预处理、格式转换、尺寸调整、质量优化等功能
"""

import os
import io
import base64
import hashlib
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from PIL import Image, ImageOps, ImageFilter
import cv2
import numpy as np
from datetime import datetime

from .multimodal_config import MultimodalConfig, ImageFormat

# 配置日志
logger = logging.getLogger(__name__)

class ImageProcessor:
    """图像处理器类"""
    
    def __init__(self):
        """初始化图像处理器"""
        self.config = MultimodalConfig()
        
        # 确保目录存在
        self.upload_path = self.config.get_upload_path()
        self.processed_path = self.config.get_processed_path()
        
        logger.info("图像处理器初始化完成")
    
    def validate_image(self, image_data: bytes, filename: str) -> Tuple[bool, str]:
        """
        验证图像文件
        
        Args:
            image_data: 图像数据
            filename: 文件名
            
        Returns:
            (是否有效, 错误信息)
        """
        try:
            # 检查文件大小
            if len(image_data) > self.config.IMAGE_CONFIG["max_file_size"]:
                return False, self.config.ERROR_MESSAGES["file_too_large"]
            
            # 检查文件格式
            if not self.config.is_supported_format(filename):
                return False, self.config.ERROR_MESSAGES["invalid_format"]
            
            # 尝试打开图像
            try:
                image = Image.open(io.BytesIO(image_data))
                width, height = image.size
                
                # 检查图像尺寸
                if not self.config.validate_image_size(width, height):
                    return False, f"图像尺寸过大，最大支持{self.config.IMAGE_CONFIG['max_width']}x{self.config.IMAGE_CONFIG['max_height']}"
                
                # 检查图像模式
                if image.mode not in ['RGB', 'RGBA', 'L', 'P']:
                    return False, "不支持的图像颜色模式"
                
                return True, ""
                
            except Exception as e:
                return False, f"图像文件损坏或格式不正确: {str(e)}"
                
        except Exception as e:
            logger.error(f"图像验证失败: {e}")
            return False, f"图像验证失败: {str(e)}"
    
    def process_image(self, image_data: bytes, filename: str, 
                     resize: bool = True, optimize: bool = True) -> Dict[str, Any]:
        """
        处理图像
        
        Args:
            image_data: 图像数据
            filename: 文件名
            resize: 是否调整尺寸
            optimize: 是否优化质量
            
        Returns:
            处理结果字典
        """
        try:
            # 验证图像
            is_valid, error_msg = self.validate_image(image_data, filename)
            if not is_valid:
                return {"success": False, "error": error_msg}
            
            # 打开图像
            image = Image.open(io.BytesIO(image_data))
            original_size = image.size
            original_mode = image.mode
            
            # 转换为RGB模式（如果需要）
            if image.mode in ['RGBA', 'P']:
                # 创建白色背景
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'RGBA':
                    background.paste(image, mask=image.split()[-1])
                else:
                    background.paste(image)
                image = background
            elif image.mode == 'L':
                image = image.convert('RGB')
            
            # 调整尺寸（如果需要）
            if resize:
                image = self._resize_image(image)
            
            # 优化质量（如果需要）
            if optimize:
                image = self._optimize_image(image)
            
            # 生成文件信息
            file_info = self._generate_file_info(filename, original_size, image.size)
            
            # 保存处理后的图像
            processed_path = self._save_processed_image(image, file_info["processed_filename"])
            
            # 生成缩略图
            thumbnail_path = self._generate_thumbnail(image, file_info["image_id"])
            
            # 计算图像特征
            features = self._extract_image_features(image)
            
            result = {
                "success": True,
                "file_info": file_info,
                "processed_path": processed_path,
                "thumbnail_path": thumbnail_path,
                "features": features,
                "original_size": original_size,
                "processed_size": image.size,
                "message": self.config.SUCCESS_MESSAGES["processing_success"]
            }
            
            logger.info(f"图像处理成功: {filename}")
            return result
            
        except Exception as e:
            logger.error(f"图像处理失败: {e}")
            return {
                "success": False,
                "error": f"图像处理失败: {str(e)}"
            }
    
    def _resize_image(self, image: Image.Image) -> Image.Image:
        """调整图像尺寸"""
        max_width = self.config.IMAGE_CONFIG["max_width"]
        max_height = self.config.IMAGE_CONFIG["max_height"]
        
        width, height = image.size
        
        # 如果图像已经在限制范围内，不需要调整
        if width <= max_width and height <= max_height:
            return image
        
        # 计算缩放比例
        ratio = min(max_width / width, max_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        # 使用高质量重采样
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        logger.debug(f"图像尺寸调整: {width}x{height} -> {new_width}x{new_height}")
        return resized_image
    
    def _optimize_image(self, image: Image.Image) -> Image.Image:
        """优化图像质量"""
        # 应用轻微的锐化滤镜
        try:
            enhanced_image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
            return enhanced_image
        except Exception as e:
            logger.warning(f"图像优化失败，使用原图: {e}")
            return image
    
    def _generate_file_info(self, filename: str, original_size: Tuple[int, int], 
                           processed_size: Tuple[int, int]) -> Dict[str, Any]:
        """生成文件信息"""
        # 生成唯一ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_hash = hashlib.md5(f"{filename}_{timestamp}".encode()).hexdigest()[:8]
        image_id = f"{timestamp}_{file_hash}"
        
        # 生成处理后的文件名
        name, ext = os.path.splitext(filename)
        processed_filename = f"{image_id}_{name}_processed.jpg"
        
        return {
            "image_id": image_id,
            "original_filename": filename,
            "processed_filename": processed_filename,
            "original_size": original_size,
            "processed_size": processed_size,
            "timestamp": timestamp,
            "file_hash": file_hash
        }
    
    def _save_processed_image(self, image: Image.Image, filename: str) -> str:
        """保存处理后的图像"""
        file_path = os.path.join(self.processed_path, filename)
        
        # 保存为JPEG格式，优化质量
        image.save(
            file_path, 
            "JPEG", 
            quality=self.config.IMAGE_CONFIG["quality"],
            optimize=True
        )
        
        return file_path
    
    def _generate_thumbnail(self, image: Image.Image, image_id: str) -> str:
        """生成缩略图"""
        thumbnail_size = self.config.get_thumbnail_size()
        thumbnail_filename = f"{image_id}_thumb.jpg"
        thumbnail_path = os.path.join(self.processed_path, thumbnail_filename)
        
        # 创建缩略图
        thumbnail = image.copy()
        thumbnail.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
        
        # 保存缩略图
        thumbnail.save(thumbnail_path, "JPEG", quality=80, optimize=True)
        
        return thumbnail_path
    
    def _extract_image_features(self, image: Image.Image) -> Dict[str, Any]:
        """提取图像特征"""
        try:
            # 转换为numpy数组
            img_array = np.array(image)
            
            # 基础统计特征
            features = {
                "width": image.size[0],
                "height": image.size[1],
                "aspect_ratio": image.size[0] / image.size[1],
                "mean_brightness": float(np.mean(img_array)),
                "std_brightness": float(np.std(img_array))
            }
            
            # 颜色特征
            if len(img_array.shape) == 3:  # RGB图像
                features.update({
                    "mean_red": float(np.mean(img_array[:, :, 0])),
                    "mean_green": float(np.mean(img_array[:, :, 1])),
                    "mean_blue": float(np.mean(img_array[:, :, 2])),
                    "dominant_color": self._get_dominant_color(img_array)
                })
            
            # 使用OpenCV提取更多特征（如果可用）
            try:
                cv_features = self._extract_cv_features(img_array)
                features.update(cv_features)
            except Exception as e:
                logger.debug(f"OpenCV特征提取失败: {e}")
            
            return features
            
        except Exception as e:
            logger.warning(f"特征提取失败: {e}")
            return {
                "width": image.size[0],
                "height": image.size[1],
                "aspect_ratio": image.size[0] / image.size[1]
            }
    
    def _get_dominant_color(self, img_array: np.ndarray) -> List[int]:
        """获取主导颜色"""
        try:
            # 重塑数组并计算颜色直方图
            pixels = img_array.reshape(-1, 3)
            
            # 使用K-means聚类找到主导颜色（简化版本）
            mean_color = np.mean(pixels, axis=0)
            return [int(c) for c in mean_color]
            
        except Exception as e:
            logger.debug(f"主导颜色计算失败: {e}")
            return [128, 128, 128]  # 默认灰色
    
    def _extract_cv_features(self, img_array: np.ndarray) -> Dict[str, Any]:
        """使用OpenCV提取高级特征"""
        features = {}
        
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # 边缘检测
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            features["edge_density"] = float(edge_density)
            
            # 纹理特征（简化版本）
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            features["texture_variance"] = float(laplacian_var)
            
            # 对比度
            features["contrast"] = float(gray.std())
            
        except Exception as e:
            logger.debug(f"OpenCV特征提取失败: {e}")
        
        return features
    
    def image_to_base64(self, image_path: str) -> str:
        """将图像转换为base64编码"""
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return encoded_string
        except Exception as e:
            logger.error(f"图像base64编码失败: {e}")
            return ""
    
    def base64_to_image(self, base64_string: str) -> Optional[Image.Image]:
        """将base64编码转换为图像"""
        try:
            image_data = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_data))
            return image
        except Exception as e:
            logger.error(f"base64解码失败: {e}")
            return None
    
    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        """获取图像信息"""
        try:
            with Image.open(image_path) as image:
                return {
                    "filename": os.path.basename(image_path),
                    "size": image.size,
                    "mode": image.mode,
                    "format": image.format,
                    "file_size": os.path.getsize(image_path)
                }
        except Exception as e:
            logger.error(f"获取图像信息失败: {e}")
            return {}
    
    def cleanup_old_files(self, days: int = 7) -> int:
        """清理旧文件"""
        try:
            import time
            current_time = time.time()
            cutoff_time = current_time - (days * 24 * 60 * 60)
            
            deleted_count = 0
            
            # 清理处理后的图像
            for filename in os.listdir(self.processed_path):
                file_path = os.path.join(self.processed_path, filename)
                if os.path.isfile(file_path):
                    file_mtime = os.path.getmtime(file_path)
                    if file_mtime < cutoff_time:
                        os.remove(file_path)
                        deleted_count += 1
            
            logger.info(f"清理完成，删除了 {deleted_count} 个旧文件")
            return deleted_count
            
        except Exception as e:
            logger.error(f"清理旧文件失败: {e}")
            return 0
