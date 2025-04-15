import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import sys
import logging
import time
from typing import Optional, Dict, Any, Generator
from dataclasses import dataclass
from pathlib import Path

# 配置日志系统
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ollama_client.log")
    ]
)

# ANSI颜色代码
COLORS = {
    "user": "\033[94m",     # 蓝色
    "assistant": "\033[92m",# 绿色
    "system": "\033[93m",   # 黄色
    "reset": "\033[0m"
}

@dataclass
class ModelConfig:
    """模型配置类"""
    name: str
    context_length: int
    default_temperature: float
    default_top_p: float
    stop_sequences: list
    timeout: int
    num_predict: int
    supports_multimodal: bool = False

class ModelConfigManager:
    """模型配置管理器"""
    
    def __init__(self, config_file: str = "model_configs.json"):
        self.config_file = config_file
        self.configs = self._load_configs()

    def _load_configs(self) -> Dict[str, ModelConfig]:
        """从文件加载配置"""
        default_configs = {
            "deepseek-r1:7b": ModelConfig(
                name="deepseek-r1:7b",
                context_length=4096,
                default_temperature=0.7,
                default_top_p=0.9,
                stop_sequences=["</s>", "Human:", "Assistant:"],
                timeout=60,
                num_predict=512,
                supports_multimodal=False
            ),
            # 其他默认配置...
        }
        
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, "r") as f:
                    configs = json.load(f)
                    return {
                        name: ModelConfig(**data) 
                        for name, data in configs.items()
                    }
            return default_configs
        except Exception as e:
            logging.error(f"加载配置文件失败: {str(e)}")
            return default_configs

    def get_config(self, model_name: str) -> ModelConfig:
        """获取模型配置"""
        return self.configs.get(model_name, ModelConfig(
            name=model_name,
            context_length=4096,
            default_temperature=0.7,
            default_top_p=0.9,
            stop_sequences=[],
            timeout=30,
            num_predict=256
        ))

class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        max_retries: int = 3
    ):
        self.base_url = base_url.rstrip('/')
        self.config_manager = ModelConfigManager()
        self.session = requests.Session()
        self.history = []
        
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _print_colored(self, text: str, role: str = "system"):
        """带颜色打印"""
        print(f"{COLORS[role]}{text}{COLORS['reset']}")

    def get_available_models(self) -> list:
        """获取可用模型列表"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return [model["name"] for model in response.json().get("models", [])]
        except Exception as e:
            logging.error(f"获取模型列表失败: {str(e)}")
            return []

    def check_model_status(self, model_name: str) -> bool:
        """检查模型是否可用"""
        return model_name in self.get_available_models()

    def generate_stream(
        self,
        prompt: str,
        model_name: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        verbose: bool = False
    ) -> Generator[str, None, None]:
        """流式生成响应"""
        model_config = self.config_manager.get_config(model_name)
        
        data = {
            "model": model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature or model_config.default_temperature,
                "top_p": top_p or model_config.default_top_p,
                "num_ctx": model_config.context_length,
                "stop": model_config.stop_sequences,
                "num_predict": model_config.num_predict
            }
        }

        try:
            with self.session.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=model_config.timeout,
                stream=True
            ) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line.decode())
                        if "response" in chunk:
                            yield chunk["response"]
                        if chunk.get("done"):
                            break

        except requests.exceptions.RequestException as e:
            logging.error(f"请求失败: {str(e)}")
            yield f"请求错误: {str(e)}"

    def interactive_chat(self, model_name: str):
        """交互式聊天模式"""
        self._print_colored(f"=== 当前使用模型: {model_name} ===", "system")
        self._print_colored("输入内容开始对话（输入'/help'查看帮助）", "system")
        
        while True:
            try:
                user_input = input(f"{COLORS['user']}\n您：{COLORS['reset']}").strip()
                
                # 处理命令
                if user_input.startswith('/'):
                    if user_input == '/exit':
                        break
                    elif user_input == '/clear':
                        self.history.clear()
                        print("历史记录已清除")
                        continue
                    elif user_input == '/models':
                        available = self.get_available_models()
                        print(f"可用模型: {', '.join(available)}")
                        continue
                    elif user_input == '/help':
                        print("可用命令：\n/exit 退出\n/clear 清除历史\n/models 查看模型\n/help 帮助")
                        continue

                if not user_input:
                    continue
                
                # 记录历史（简单实现）
                self.history.append({"role": "user", "content": user_input})
                
                # 流式输出
                print(f"{COLORS['assistant']}助手：", end="", flush=True)
                full_response = ""
                start_time = time.time()
                
                for chunk in self.generate_stream(
                    prompt=user_input,
                    model_name=model_name
                ):
                    print(chunk, end="", flush=True)
                    full_response += chunk
                
                # 记录响应时间
                elapsed = time.time() - start_time
                self.history.append({
                    "role": "assistant", 
                    "content": full_response,
                    "time": f"{elapsed:.2f}s"
                })
                
                print(COLORS["reset"])  # 重置颜色

            except KeyboardInterrupt:
                self._print_colored("\n输入 '/exit' 退出程序", "system")
            except Exception as e:
                logging.error(f"运行时错误: {str(e)}")
                self._print_colored("发生错误，请查看日志", "system")

if __name__ == "__main__":
    client = OllamaClient()
    
    # 获取可用模型
    available_models = client.get_available_models()
    if not available_models:
        print("没有可用模型，请检查Ollama服务")
        sys.exit(1)

    # 模型选择
    print(f"{COLORS['system']}可用模型: {', '.join(available_models)}{COLORS['reset']}")
    selected_model = input("选择模型（默认deepseek-r1:7b）: ").strip() or "deepseek-r1:7b"
    
    if not client.check_model_status(selected_model):
        print(f"模型 {selected_model} 不可用")
        sys.exit(1)

    # 进入交互模式
    client.interactive_chat(selected_model)
    print("对话已结束")
