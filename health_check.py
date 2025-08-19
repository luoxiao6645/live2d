#!/usr/bin/env python3
"""
Live2D AI助手 健康检查脚本

检查系统各组件的运行状态
"""

import requests
import json
import sys
from datetime import datetime

def check_server_health(host="localhost", port=5000):
    """检查服务器健康状态"""
    try:
        url = f"http://{host}:{port}/api/health"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✓ 服务器运行正常")
            print(f"  状态: {data.get('status', 'unknown')}")
            print(f"  时间: {datetime.fromtimestamp(data.get('timestamp', 0))}")
            
            # 检查各个服务状态
            services = data.get('services', {})
            print("\n服务状态:")
            for service, status in services.items():
                icon = "✓" if status else "✗"
                print(f"  {icon} {service}: {'可用' if status else '不可用'}")
            
            return True
        else:
            print(f"✗ 服务器响应异常: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ 无法连接到服务器")
        print("请确保服务器正在运行")
        return False
    except requests.exceptions.Timeout:
        print("✗ 服务器响应超时")
        return False
    except Exception as e:
        print(f"✗ 健康检查失败: {e}")
        return False

def check_ollama_service():
    """检查Ollama服务"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            print(f"✓ Ollama服务正常 ({len(models)} 个模型)")
            
            for model in models[:3]:  # 显示前3个模型
                print(f"  - {model.get('name', 'unknown')}")
            
            return True
        else:
            print("✗ Ollama服务响应异常")
            return False
    except Exception:
        print("✗ Ollama服务不可用")
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("Live2D AI助手 健康检查")
    print("=" * 50)
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 检查主服务器
    server_ok = check_server_health()
    print()
    
    # 检查Ollama
    ollama_ok = check_ollama_service()
    print()
    
    # 总结
    if server_ok and ollama_ok:
        print("🎉 所有服务运行正常！")
        return 0
    elif server_ok:
        print("⚠ 主服务正常，但Ollama服务有问题")
        return 1
    else:
        print("❌ 主服务有问题")
        return 2

if __name__ == "__main__":
    sys.exit(main())
