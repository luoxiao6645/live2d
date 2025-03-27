# Live2D AI 助手

一个结合了Live2D模型展示和AI对话功能的Web应用程序。通过整合Live2D Cubism SDK和大语言模型，创建了一个可视化的AI对话助手。

## 功能特点

- 💬 支持与AI助手实时对话
- 🎭 支持多个Live2D模型切换展示
- 🗣️ 集成Edge TTS实现AI回复语音播放
- 🖼️ 支持自定义背景图片和颜色
- 💾 支持会话管理和历史记录
- 🎨 美观的用户界面和流畅的动画效果

## 技术栈

- 前端：HTML5, JavaScript, Live2D Cubism SDK
- 后端：Python Flask
- AI模型：Ollama (支持多种模型如Qwen)
- 语音合成：Microsoft Edge TTS
- 其他：WebSocket, CORS支持

## 安装要求

- Python 3.8+
- Ollama服务
- Edge TTS
- Live2D Cubism SDK

## 快速开始

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 确保Ollama服务运行在本地11434端口

3. 启动服务器：
```bash
python server.py
```

4. 访问应用：
打开浏览器访问 `http://localhost:5000/live2d_llm.html`

## 配置说明

### 模型配置
- 支持Live2D Cubism 4.0模型
- 模型文件需放置在`models`目录下
- 支持动态加载和切换模型

### AI对话配置
- 默认使用Qwen模型
- 可配置温度、采样等参数
- 支持多轮对话历史记录

### 语音配置
- 支持调整语音音量
- 可选择不同的语音角色
- 支持语音播放控制

## 目录结构

```
.
├── server.py           # 后端服务器
├── live2d_llm.html    # 前端页面
├── models/            # Live2D模型目录
├── js/               # JavaScript库
├── backgrounds/      # 背景图片资源
└── requirements.txt  # Python依赖
```

## 许可证

本项目基于MIT许可证开源。

## 致谢

- Live2D Cubism SDK
- Ollama项目
- Edge TTS
- 所有贡献者

## 注意事项

- 请确保有适当的模型使用许可
- 建议使用现代浏览器访问
- 需要稳定的网络连接 