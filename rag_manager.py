import os
import logging
import tempfile
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

# 文档加载器
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGManager:
    """RAG (Retrieval Augmented Generation) 管理器"""
    
    def __init__(self, 
                 persist_directory: str = "./knowledge_base",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 ollama_base_url: str = "http://127.0.0.1:11434"):
        """
        初始化RAG管理器
        
        Args:
            persist_directory: 向量数据库持久化目录
            embedding_model: 嵌入模型名称
            ollama_base_url: Ollama服务地址
        """
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        self.ollama_base_url = ollama_base_url
        
        # 确保目录存在
        os.makedirs(persist_directory, exist_ok=True)
        os.makedirs("./uploads", exist_ok=True)
        
        # 初始化组件
        self._init_embeddings()
        self._init_text_splitter()
        self._init_vector_store()
        self._init_rag_chain()
        
        logger.info("RAG管理器初始化完成")
    
    def _init_embeddings(self):
        """初始化嵌入模型"""
        try:
            self.embeddings = SentenceTransformerEmbeddings(
                model_name=self.embedding_model_name
            )
            logger.info(f"嵌入模型 {self.embedding_model_name} 加载成功")
        except Exception as e:
            logger.error(f"嵌入模型加载失败: {e}")
            raise
    
    def _init_text_splitter(self):
        """初始化文本分割器"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
        logger.info("文本分割器初始化完成")
    
    def _init_vector_store(self):
        """初始化向量数据库"""
        try:
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name="live2d_knowledge_base"
            )
            logger.info("向量数据库初始化完成")
        except Exception as e:
            logger.error(f"向量数据库初始化失败: {e}")
            raise
    
    def _init_rag_chain(self):
        """初始化RAG链"""
        # 创建提示模板
        self.rag_prompt = ChatPromptTemplate.from_template("""
你是一个AI助手，请基于以下检索到的相关文档内容来回答用户的问题。
如果检索到的内容无法回答问题，请诚实地说明你不知道答案。
请保持回答简洁明了，最多使用3句话。

相关文档内容：
{context}

用户问题：{question}

回答：""")
        
        # 创建检索器
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        logger.info("RAG链初始化完成")
    
    def create_rag_chain(self, model_name: str = "qwen2:0.5b", temperature: float = 0.7):
        """创建完整的RAG链"""
        try:
            # 初始化Ollama模型
            llm = ChatOllama(
                model=model_name,
                base_url=self.ollama_base_url,
                temperature=temperature
            )
            
            # 文档格式化函数
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            # 构建RAG链
            rag_chain = (
                {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
                | self.rag_prompt
                | llm
                | StrOutputParser()
            )
            
            return rag_chain
            
        except Exception as e:
            logger.error(f"创建RAG链失败: {e}")
            raise
    
    def load_document(self, file_path: str, file_type: str = None) -> List[Document]:
        """
        加载文档
        
        Args:
            file_path: 文件路径
            file_type: 文件类型，如果为None则自动检测
            
        Returns:
            文档列表
        """
        if file_type is None:
            file_type = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_type == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_type in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            elif file_type == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_type in ['.md', '.markdown']:
                loader = UnstructuredMarkdownLoader(file_path)
            else:
                raise ValueError(f"不支持的文件类型: {file_type}")
            
            documents = loader.load()
            logger.info(f"成功加载文档: {file_path}, 页数: {len(documents)}")
            return documents
            
        except Exception as e:
            logger.error(f"加载文档失败 {file_path}: {e}")
            raise
    
    def process_and_store_document(self, file_path: str, metadata: Dict[str, Any] = None) -> str:
        """
        处理文档并存储到向量数据库
        
        Args:
            file_path: 文件路径
            metadata: 额外的元数据
            
        Returns:
            文档ID
        """
        try:
            # 加载文档
            documents = self.load_document(file_path)
            
            # 生成文档ID
            doc_id = self._generate_doc_id(file_path)
            
            # 添加元数据
            for doc in documents:
                doc.metadata.update({
                    "doc_id": doc_id,
                    "source_file": os.path.basename(file_path),
                    "upload_time": datetime.now().isoformat(),
                    **(metadata or {})
                })
            
            # 分割文档
            splits = self.text_splitter.split_documents(documents)
            
            # 存储到向量数据库
            self.vector_store.add_documents(splits)
            
            logger.info(f"文档处理完成: {file_path}, 分块数: {len(splits)}, 文档ID: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"处理文档失败 {file_path}: {e}")
            raise
    
    def _generate_doc_id(self, file_path: str) -> str:
        """生成文档ID"""
        file_name = os.path.basename(file_path)
        timestamp = datetime.now().isoformat()
        content = f"{file_name}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def search_documents(self, query: str, k: int = 5) -> List[Document]:
        """
        搜索相关文档
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            相关文档列表
        """
        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"搜索查询: {query}, 找到 {len(results)} 个相关文档")
            return results
        except Exception as e:
            logger.error(f"搜索文档失败: {e}")
            return []
    
    def get_knowledge_base_info(self) -> Dict[str, Any]:
        """获取知识库信息"""
        try:
            # 获取集合信息
            collection = self.vector_store._collection
            count = collection.count()
            
            # 获取所有文档的元数据
            if count > 0:
                results = collection.get(include=['metadatas'])
                metadatas = results.get('metadatas', [])
                
                # 统计文档信息
                doc_ids = set()
                sources = set()
                for metadata in metadatas:
                    if 'doc_id' in metadata:
                        doc_ids.add(metadata['doc_id'])
                    if 'source_file' in metadata:
                        sources.add(metadata['source_file'])
                
                return {
                    "total_chunks": count,
                    "total_documents": len(doc_ids),
                    "source_files": list(sources),
                    "doc_ids": list(doc_ids)
                }
            else:
                return {
                    "total_chunks": 0,
                    "total_documents": 0,
                    "source_files": [],
                    "doc_ids": []
                }
                
        except Exception as e:
            logger.error(f"获取知识库信息失败: {e}")
            return {"error": str(e)}
    
    def delete_document(self, doc_id: str) -> bool:
        """
        删除文档
        
        Args:
            doc_id: 文档ID
            
        Returns:
            是否删除成功
        """
        try:
            # 查找要删除的文档
            collection = self.vector_store._collection
            results = collection.get(
                where={"doc_id": doc_id},
                include=['ids']
            )
            
            if results['ids']:
                # 删除文档
                collection.delete(ids=results['ids'])
                logger.info(f"成功删除文档: {doc_id}, 删除了 {len(results['ids'])} 个分块")
                return True
            else:
                logger.warning(f"未找到要删除的文档: {doc_id}")
                return False
                
        except Exception as e:
            logger.error(f"删除文档失败 {doc_id}: {e}")
            return False
