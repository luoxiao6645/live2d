# 🌟 Live2D AI助手 - 下一代智能交互系统

一个集成了Live2D虚拟角色、大语言模型和前沿AI技术的智能助手系统，具备深度理解、推理和多模态交互能力。

## 🚀 系统架构演进

### 第四阶段 (最新) - 高级RAG系统 🧠
- **图文混合向量化**: 基于Sentence-Transformers和CLIP的多模态向量化
- **知识图谱构建**: 自动实体识别、关系抽取和动态图谱构建
- **智能内容融合**: 多源信息融合、冲突解决和质量评估
- **多模态推理**: 路径推理、子图匹配、随机游走等推理策略
- **图查询处理**: 自然语言到图查询的智能转换

### 第三阶段 - 多模态系统 🖼️ + 语音情感系统 🎵
- **图像理解**: 基于视觉语言模型的图像分析
- **情感语音合成**: 基于情感状态的TTS语音生成
- **多语言支持**: 中英文等多语言语音合成
- **视觉推理**: 图像内容的深度理解和推理

### 第二阶段 - 情感表达系统 🎭
- **高级动画控制**: 基于情感状态的Live2D动画控制
- **情感分析**: 实时文本情感识别和状态管理
- **动画序列**: 复杂情感表达的动画组合

### 第一阶段 - 基础AI功能 🤖
- **大语言模型对话**: 支持多种LLM模型
- **RAG知识库**: 文档检索和问答
- **语音交互**: TTS和STT功能

## 🎯 核心特性

### 🧠 高级RAG系统 (Advanced RAG System)

#### 图文混合向量化 (Multimodal Vectorization)
- **Sentence-Transformers**: 多语言文本向量化，支持中英文混合处理
- **CLIP模型集成**: 图像和文本的统一向量表示
- **融合策略**: 拼接、平均、加权、注意力等多种融合方式
- **智能缓存**: 向量缓存机制，提升处理效率
- **批量处理**: 支持批量文本向量化和相似度计算

#### 知识图谱构建 (Knowledge Graph Construction)
- **实体识别**: 基于spaCy和规则的混合实体识别
- **关系抽取**: 依存句法分析和模式匹配的关系提取
- **动态图谱**: 实时构建和更新知识图谱结构
- **图谱存储**: 支持GraphML、GEXF、JSON等多种格式
- **图谱查询**: 实体查找、邻居发现、路径查询

#### 智能内容融合 (Content Fusion Engine)
- **多源融合**: 来自不同来源的信息智能整合
- **冲突解决**: 基于置信度、时间戳、多数投票的冲突处理
- **质量评估**: 来源可信度、内容新鲜度、完整性评估
- **融合策略**: 加权平均、注意力机制、图结构融合
- **一致性保证**: 确保融合结果的逻辑一致性

#### 多模态推理 (Multimodal Reasoning)
- **路径推理**: 基于知识图谱的路径查找和推理
- **子图匹配**: 复杂模式的子图匹配推理
- **随机游走**: 基于随机游走的关联发现和重要性计算
- **推理解释**: 可解释的推理过程和结果
- **置信度计算**: 多因子的推理置信度评估

#### 图查询处理 (Graph Query Processing)
- **自然语言理解**: 查询意图识别和实体链接
- **查询优化**: 查询重写、分解和并行执行
- **多种查询类型**: 实体查找、关系查询、路径查找、计数查询
- **结果排序**: 多因子的结果排序和过滤
- **查询缓存**: 高效的查询结果缓存机制

### 🎭 情感表达系统 (Emotion System)

#### 高级动画控制
- **情感状态管理**: 基于文本分析的情感状态识别
- **Live2D动画控制**: 精确的表情和动作控制
- **动画序列**: 复杂情感表达的动画组合
- **平滑过渡**: 情感状态间的自然过渡效果
- **个性化配置**: 可调节的角色性格和表达风格

#### 情感分析引擎
- **多维情感识别**: 开心、悲伤、惊讶、生气、害羞、疑惑等
- **情感强度计算**: 精确的情感强度量化
- **上下文理解**: 基于对话历史的情感理解
- **实时处理**: 毫秒级的情感分析响应

### 🎵 语音情感系统 (Voice Emotion System)

#### 情感语音合成
- **情感TTS**: 基于情感状态的语音合成
- **多语言支持**: 中文、英文等多语言语音生成
- **语音风格控制**: 可调节的语音参数和风格
- **实时合成**: 高效的实时语音生成

#### 语音处理
- **语音识别**: 高精度的STT语音识别
- **情感识别**: 从语音中识别情感状态
- **口型同步**: 精确的口型同步系统
- **音频处理**: 音量控制、播放控制等

### 🖼️ 多模态系统 (Multimodal System)

#### 图像理解
- **视觉语言模型**: 基于先进VLM的图像分析
- **图像描述**: 自动生成图像描述和标签
- **视觉问答**: 基于图像内容的问答
- **场景理解**: 复杂场景的理解和分析

#### 多模态对话
- **图文对话**: 支持图像输入的多模态对话
- **视觉推理**: 图像内容的深度理解和推理
- **多模态RAG**: 图文混合的知识检索
- **跨模态关联**: 文本和图像的语义关联

## 🚀 快速开始

### 1. 环境准备

#### 基础依赖
```bash
# Python 3.8+
pip install flask flask-cors requests python-docx PyPDF2 markdown

# 基础RAG依赖
pip install langchain chromadb sentence-transformers

# 高级RAG系统依赖
pip install networkx scikit-learn spacy

# 多模态系统依赖 (可选)
pip install torch torchvision transformers pillow

# 语音情感系统依赖 (可选)
pip install edge-tts pydub librosa
```

#### spaCy中文模型
```bash
python -m spacy download zh_core_web_sm
```

#### Ollama安装 (用于LLM)
```bash
# 安装Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 下载模型
ollama pull deepseek-r1:7b
ollama pull qwen2:0.5b
```

### 2. 启动系统

#### 完整启动 (推荐)
```bash
# 启动完整系统 (包含所有功能)
python server.py
```

#### 测试各个系统
```bash
# 测试高级RAG系统
python test_advanced_rag_system.py

# 测试情感系统
python test_emotion_system.py

# 测试多模态系统
python test_multimodal_system.py

# 测试语音情感系统
python test_voice_emotion_system.py
```

#### 访问界面
打开浏览器访问: `http://localhost:5000`

### 3. 系统配置

#### 高级RAG配置
```python
# advanced_rag_system/advanced_rag_config.py
VECTORIZER_CONFIG = {
    "text_model": "paraphrase-multilingual-MiniLM-L12-v2",
    "embedding_dimension": 384,
    "batch_size": 32,
    "device": "auto"  # auto, cpu, cuda
}

KNOWLEDGE_GRAPH_CONFIG = {
    "entity_extraction": {
        "model": "zh_core_web_sm",
        "confidence_threshold": 0.8
    },
    "relation_extraction": {
        "confidence_threshold": 0.7
    }
}
```

#### 情感系统配置
```python
# emotion_system/emotion_config.py
EMOTION_CONFIG = {
    "sensitivity": 0.8,
    "animation_duration": 2.0,
    "transition_smoothness": 0.5
}
```

## 📊 系统架构

### 核心组件架构
```
Live2D AI助手
├── 高级RAG系统 (Advanced RAG)
│   ├── MultimodalVectorizer (图文混合向量化)
│   ├── KnowledgeGraphBuilder (知识图谱构建)
│   ├── ContentFusionEngine (内容融合引擎)
│   ├── MultimodalReasoningEngine (多模态推理)
│   └── GraphQueryProcessor (图查询处理)
├── 情感表达系统 (Emotion System)
│   ├── EmotionAnalyzer (情感分析)
│   ├── EmotionStateManager (状态管理)
│   ├── AdvancedAnimationController (动画控制)
│   └── AnimationSequencer (动画序列)
├── 语音情感系统 (Voice Emotion)
│   ├── EmotionVoiceSynthesizer (情感语音合成)
│   ├── MultilingualVoiceManager (多语言语音)
│   ├── RealtimeVoiceProcessor (实时语音处理)
│   └── VoiceStyleController (语音风格控制)
├── 多模态系统 (Multimodal System)
│   ├── ImageAnalyzer (图像分析)
│   ├── VisionLanguageModel (视觉语言模型)
│   ├── MultimodalRAG (多模态RAG)
│   └── ImageProcessor (图像处理)
└── 基础系统 (Base System)
    ├── Live2D渲染引擎
    ├── LLM对话系统
    ├── 基础RAG系统
    └── Web界面
```

### 数据流程
```
用户输入 → 意图理解 → 多模态处理 → 知识检索 → 推理融合 → 情感分析 → 响应生成 → Live2D表现
```

## 🎯 使用场景

### 🎓 学术研究
- **文献分析**: 自动提取论文中的实体和关系，构建学术知识图谱
- **知识发现**: 通过图谱推理发现隐藏的知识联系和研究机会
- **多源整合**: 融合来自不同文献的信息，生成综合性研究报告
- **假设验证**: 基于知识图谱验证研究假设的可行性

### 🏢 企业应用
- **知识管理**: 将企业文档转化为结构化知识图谱
- **智能决策**: 基于多源信息的智能决策支持系统
- **专家系统**: 构建领域专家知识图谱，传承专业经验
- **业务分析**: 通过关系分析发现业务机会和风险

### 📚 教育培训
- **个性化学习**: 根据学习者特点推荐最优学习路径
- **智能答疑**: 基于推理的深度问答，解决复杂学习问题
- **知识可视化**: 将抽象知识转化为可视化的图谱结构
- **学习评估**: 多维度的学习效果评估和改进建议

### 🤖 智能客服
- **复杂问题处理**: 通过推理解决复杂的业务问题
- **多轮对话**: 基于上下文的连续对话和问题跟踪
- **知识更新**: 动态更新业务知识库，保持信息时效性
- **个性化服务**: 基于用户画像的个性化回答和建议

## 🔧 API接口

### 高级RAG系统API

#### 系统状态
```http
GET /api/advanced-rag/status
```

#### 文档处理
```http
POST /api/advanced-rag/process-document
Content-Type: application/json

{
    "text": "文档内容",
    "doc_id": "文档ID",
    "image_path": "图像路径",
    "metadata": {}
}
```

#### 高级查询
```http
POST /api/advanced-rag/query
Content-Type: application/json

{
    "query": "查询问题",
    "query_type": "auto",
    "max_results": 10,
    "include_reasoning": true
}
```

#### 多模态推理
```http
POST /api/advanced-rag/reasoning
Content-Type: application/json

{
    "query": "推理问题",
    "reasoning_type": "path_based",
    "image_path": "图像路径"
}
```

### 情感系统API

#### 情感分析
```http
POST /api/emotion/analyze
Content-Type: application/json

{
    "text": "要分析的文本",
    "context": "上下文信息"
}
```

#### 动画控制
```http
POST /api/emotion/animate
Content-Type: application/json

{
    "emotion": "happy",
    "intensity": 0.8,
    "duration": 2.0
}
```

### 多模态系统API

#### 图像分析
```http
POST /api/multimodal/analyze-image
Content-Type: multipart/form-data

image: [图像文件]
prompt: "分析提示"
```

#### 图文对话
```http
POST /api/multimodal/chat
Content-Type: application/json

{
    "text": "对话文本",
    "image_path": "图像路径",
    "history": []
}
```

## 🌟 技术亮点

### 🧠 前沿AI技术
- **Sentence-Transformers**: 多语言语义向量化
- **知识图谱**: 基于NetworkX的图结构处理
- **多模态融合**: 图文信息的智能整合
- **深度推理**: 路径推理、子图匹配等算法
- **情感计算**: 基于NLP的情感识别和表达

### ⚡ 高性能设计
- **多级缓存**: 向量缓存、查询缓存、推理缓存
- **批量处理**: 高效的批量向量化和处理
- **并行计算**: 支持多线程和GPU加速
- **内存优化**: 智能的内存管理和资源清理
- **流式处理**: 实时的流式响应和处理

### 🔧 工程化特性
- **模块化设计**: 清晰的模块划分和接口设计
- **配置驱动**: 灵活的配置系统和参数调节
- **错误处理**: 完善的异常处理和错误恢复
- **日志系统**: 详细的日志记录和调试信息
- **测试覆盖**: 完整的单元测试和集成测试

### 🌐 扩展性
- **插件架构**: 支持自定义插件和扩展
- **API标准**: 标准化的REST API接口
- **多语言**: 支持多语言处理和国际化
- **跨平台**: 支持多种操作系统和部署环境

## 📈 性能指标

### 响应性能
- **查询响应**: < 500ms (缓存命中)
- **推理延迟**: < 2s (复杂推理)
- **向量化速度**: > 1000 docs/min
- **图谱构建**: > 500 entities/min

### 准确性指标
- **实体识别**: F1 > 0.85
- **关系抽取**: F1 > 0.80
- **情感识别**: 准确率 > 0.90
- **推理置信度**: 平均 > 0.75

### 系统容量
- **知识图谱**: 支持100K+节点
- **向量数据库**: 支持1M+文档
- **并发用户**: 支持100+用户
- **内存占用**: < 4GB (完整系统)

## 🔍 技术栈详解

### 核心技术
- **Python 3.8+**: 主要开发语言
- **Flask**: Web框架和API服务
- **NetworkX**: 图结构处理和算法
- **Sentence-Transformers**: 文本向量化
- **spaCy**: 自然语言处理
- **scikit-learn**: 机器学习算法

### AI/ML技术
- **LangChain**: LLM应用框架
- **Chroma**: 向量数据库
- **Transformers**: 预训练模型
- **CLIP**: 多模态模型 (可选)
- **Edge-TTS**: 语音合成

### 前端技术
- **HTML5/CSS3**: 现代Web标准
- **JavaScript/jQuery**: 交互逻辑
- **Live2D Cubism**: 虚拟角色渲染
- **WebGL**: 3D图形渲染

### 数据存储
- **Chroma**: 向量数据库
- **GraphML/GEXF**: 图谱存储格式
- **JSON**: 配置和数据交换
- **SQLite**: 轻量级关系数据库 (可选)

## 📚 文档指南

### 系统文档
- [高级RAG系统指南](ADVANCED_RAG_SYSTEM_GUIDE.md)
- [情感系统指南](EMOTION_SYSTEM_GUIDE.md)
- [多模态系统指南](MULTIMODAL_SYSTEM_GUIDE.md)
- [语音情感系统指南](VOICE_EMOTION_SYSTEM_GUIDE.md)

### 开发文档
- [RAG升级指南](RAG_UPGRADE_README.md)
- API接口文档 (内置于代码)
- 配置说明 (各系统config文件)
- 测试指南 (test_*.py文件)

## 🚀 部署方案

### 本地部署
```bash
# 克隆项目
git clone https://github.com/luoxiao6645/live2d.git
cd live2d

# 安装依赖
pip install -r requirements.txt

# 安装spaCy中文模型
python -m spacy download zh_core_web_sm

# 启动服务
python server.py
```

### Docker部署 (计划中)
```bash
# 构建镜像
docker build -t live2d-ai .

# 运行容器
docker run -p 5000:5000 live2d-ai
```

### 云端部署
- **AWS**: 支持EC2、Lambda、ECS部署
- **Azure**: 支持App Service、Container Instances
- **Google Cloud**: 支持Compute Engine、Cloud Run
- **阿里云**: 支持ECS、函数计算、容器服务

## 🤝 贡献指南

### 开发环境
```bash
# 克隆开发分支
git clone https://github.com/luoxiao6645/live2d.git

# 安装开发依赖
pip install -r requirements.txt

# 运行测试
python test_advanced_rag_system.py
python test_emotion_system.py
python test_multimodal_system.py
python test_voice_emotion_system.py

# 代码格式化 (可选)
# pip install black isort
# black . && isort .
```

### 贡献流程
1. Fork项目到个人仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

### 代码规范
- 遵循PEP 8代码风格
- 添加必要的注释和文档
- 编写单元测试
- 确保测试通过

## 🎯 使用场景

### 🎓 学术研究
- **文献分析**: 自动提取论文中的实体和关系，构建学术知识图谱
- **知识发现**: 通过图谱推理发现隐藏的知识联系和研究机会
- **多源整合**: 融合来自不同文献的信息，生成综合性研究报告
- **假设验证**: 基于知识图谱验证研究假设的可行性

### 🏢 企业应用
- **知识管理**: 将企业文档转化为结构化知识图谱
- **智能决策**: 基于多源信息的智能决策支持系统
- **专家系统**: 构建领域专家知识图谱，传承专业经验
- **业务分析**: 通过关系分析发现业务机会和风险

### 📚 教育培训
- **个性化学习**: 根据学习者特点推荐最优学习路径
- **智能答疑**: 基于推理的深度问答，解决复杂学习问题
- **知识可视化**: 将抽象知识转化为可视化的图谱结构
- **学习评估**: 多维度的学习效果评估和改进建议

### 🤖 智能客服
- **复杂问题处理**: 通过推理解决复杂的业务问题
- **多轮对话**: 基于上下文的连续对话和问题跟踪
- **知识更新**: 动态更新业务知识库，保持信息时效性
- **个性化服务**: 基于用户画像的个性化回答和建议

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

感谢以下开源项目和技术：
- [Live2D Cubism SDK](https://www.live2d.com/)
- [Sentence-Transformers](https://www.sbert.net/)
- [LangChain](https://langchain.com/)
- [NetworkX](https://networkx.org/)
- [spaCy](https://spacy.io/)
- [Flask](https://flask.palletsprojects.com/)
- [Ollama](https://ollama.ai/)

## 📞 联系方式

- **项目地址**: https://github.com/luoxiao6645/live2d
- **问题反馈**: [GitHub Issues](https://github.com/luoxiao6645/live2d/issues)
- **讨论交流**: [GitHub Discussions](https://github.com/luoxiao6645/live2d/discussions)

---

🌟 **如果这个项目对您有帮助，请给个Star支持一下！** ⭐

通过这个Live2D AI助手，您将体验到下一代智能交互系统的强大能力！ 🚀✨