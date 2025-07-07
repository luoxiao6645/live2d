import os
import logging
import tempfile
import mimetypes
from typing import List, Dict, Any, Optional, Tuple
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """文档处理器"""
    
    # 支持的文件类型
    SUPPORTED_EXTENSIONS = {
        '.pdf': 'application/pdf',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.doc': 'application/msword',
        '.txt': 'text/plain',
        '.md': 'text/markdown',
        '.markdown': 'text/markdown'
    }
    
    # 最大文件大小 (50MB)
    MAX_FILE_SIZE = 50 * 1024 * 1024
    
    def __init__(self, upload_directory: str = "./uploads"):
        """
        初始化文档处理器
        
        Args:
            upload_directory: 上传文件存储目录
        """
        self.upload_directory = upload_directory
        os.makedirs(upload_directory, exist_ok=True)
        logger.info(f"文档处理器初始化完成，上传目录: {upload_directory}")
    
    def validate_file(self, file: FileStorage) -> Tuple[bool, str]:
        """
        验证上传的文件
        
        Args:
            file: 上传的文件对象
            
        Returns:
            (是否有效, 错误信息)
        """
        if not file or not file.filename:
            return False, "没有选择文件"
        
        # 检查文件名
        filename = secure_filename(file.filename)
        if not filename:
            return False, "文件名无效"
        
        # 检查文件扩展名
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in self.SUPPORTED_EXTENSIONS:
            supported = ', '.join(self.SUPPORTED_EXTENSIONS.keys())
            return False, f"不支持的文件类型。支持的类型: {supported}"
        
        # 检查文件大小
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)  # 重置文件指针
        
        if file_size > self.MAX_FILE_SIZE:
            size_mb = self.MAX_FILE_SIZE / (1024 * 1024)
            return False, f"文件太大，最大支持 {size_mb}MB"
        
        if file_size == 0:
            return False, "文件为空"
        
        return True, ""
    
    def save_uploaded_file(self, file: FileStorage) -> Tuple[bool, str, Dict[str, Any]]:
        """
        保存上传的文件
        
        Args:
            file: 上传的文件对象
            
        Returns:
            (是否成功, 文件路径或错误信息, 文件信息)
        """
        try:
            # 验证文件
            is_valid, error_msg = self.validate_file(file)
            if not is_valid:
                return False, error_msg, {}
            
            # 生成安全的文件名
            original_filename = file.filename
            secure_name = secure_filename(original_filename)
            
            # 生成唯一文件名（避免重名）
            base_name, ext = os.path.splitext(secure_name)
            counter = 1
            final_filename = secure_name
            
            while os.path.exists(os.path.join(self.upload_directory, final_filename)):
                final_filename = f"{base_name}_{counter}{ext}"
                counter += 1
            
            # 保存文件
            file_path = os.path.join(self.upload_directory, final_filename)
            file.save(file_path)
            
            # 获取文件信息
            file_info = self.get_file_info(file_path, original_filename)
            
            logger.info(f"文件保存成功: {file_path}")
            return True, file_path, file_info
            
        except Exception as e:
            logger.error(f"保存文件失败: {e}")
            return False, f"保存文件失败: {str(e)}", {}
    
    def get_file_info(self, file_path: str, original_filename: str = None) -> Dict[str, Any]:
        """
        获取文件信息
        
        Args:
            file_path: 文件路径
            original_filename: 原始文件名
            
        Returns:
            文件信息字典
        """
        try:
            stat = os.stat(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()
            
            return {
                "filename": os.path.basename(file_path),
                "original_filename": original_filename or os.path.basename(file_path),
                "file_path": file_path,
                "file_size": stat.st_size,
                "file_size_mb": round(stat.st_size / (1024 * 1024), 2),
                "file_type": file_ext,
                "mime_type": self.SUPPORTED_EXTENSIONS.get(file_ext, "unknown"),
                "created_time": stat.st_ctime,
                "modified_time": stat.st_mtime
            }
        except Exception as e:
            logger.error(f"获取文件信息失败: {e}")
            return {}
    
    def delete_file(self, file_path: str) -> bool:
        """
        删除文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否删除成功
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"文件删除成功: {file_path}")
                return True
            else:
                logger.warning(f"文件不存在: {file_path}")
                return False
        except Exception as e:
            logger.error(f"删除文件失败: {e}")
            return False
    
    def list_uploaded_files(self) -> List[Dict[str, Any]]:
        """
        列出所有上传的文件
        
        Returns:
            文件信息列表
        """
        try:
            files = []
            if os.path.exists(self.upload_directory):
                for filename in os.listdir(self.upload_directory):
                    file_path = os.path.join(self.upload_directory, filename)
                    if os.path.isfile(file_path):
                        file_info = self.get_file_info(file_path)
                        if file_info:
                            files.append(file_info)
            
            # 按修改时间排序（最新的在前）
            files.sort(key=lambda x: x.get('modified_time', 0), reverse=True)
            return files
            
        except Exception as e:
            logger.error(f"列出文件失败: {e}")
            return []
    
    def get_file_content_preview(self, file_path: str, max_chars: int = 500) -> str:
        """
        获取文件内容预览
        
        Args:
            file_path: 文件路径
            max_chars: 最大字符数
            
        Returns:
            文件内容预览
        """
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(max_chars)
                    if len(content) == max_chars:
                        content += "..."
                    return content
            elif file_ext in ['.md', '.markdown']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(max_chars)
                    if len(content) == max_chars:
                        content += "..."
                    return content
            else:
                return f"无法预览 {file_ext} 文件类型"
                
        except Exception as e:
            logger.error(f"获取文件预览失败: {e}")
            return "预览失败"
    
    def cleanup_old_files(self, days: int = 30) -> int:
        """
        清理旧文件
        
        Args:
            days: 保留天数
            
        Returns:
            删除的文件数量
        """
        try:
            import time
            current_time = time.time()
            cutoff_time = current_time - (days * 24 * 60 * 60)
            
            deleted_count = 0
            if os.path.exists(self.upload_directory):
                for filename in os.listdir(self.upload_directory):
                    file_path = os.path.join(self.upload_directory, filename)
                    if os.path.isfile(file_path):
                        file_mtime = os.path.getmtime(file_path)
                        if file_mtime < cutoff_time:
                            if self.delete_file(file_path):
                                deleted_count += 1
            
            logger.info(f"清理完成，删除了 {deleted_count} 个旧文件")
            return deleted_count
            
        except Exception as e:
            logger.error(f"清理旧文件失败: {e}")
            return 0
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """
        格式化文件大小
        
        Args:
            size_bytes: 字节数
            
        Returns:
            格式化的文件大小字符串
        """
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
    
    @staticmethod
    def get_file_type_description(file_ext: str) -> str:
        """
        获取文件类型描述
        
        Args:
            file_ext: 文件扩展名
            
        Returns:
            文件类型描述
        """
        descriptions = {
            '.pdf': 'PDF文档',
            '.docx': 'Word文档',
            '.doc': 'Word文档',
            '.txt': '文本文件',
            '.md': 'Markdown文档',
            '.markdown': 'Markdown文档'
        }
        return descriptions.get(file_ext.lower(), '未知类型')
