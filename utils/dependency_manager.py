# -*- coding: utf-8 -*-
"""
依赖管理优化工具
提供依赖版本检查、安全漏洞扫描、兼容性分析和更新建议功能
"""

import os
import re
import json
import subprocess
import sys
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from packaging import version
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
try:
    from importlib.metadata import distributions
except ImportError:
    # Python < 3.8 fallback
    from importlib_metadata import distributions

@dataclass
class DependencyInfo:
    """
    依赖信息数据类
    """
    name: str
    current_version: str
    latest_version: str
    required_version: str
    is_outdated: bool
    security_issues: List[Dict[str, Any]]
    compatibility_issues: List[str]
    update_recommendation: str
    priority: str  # 'high', 'medium', 'low'
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        Returns:
            Dict[str, Any]: 字典格式的依赖信息
        """
        return asdict(self)

@dataclass
class SecurityVulnerability:
    """
    安全漏洞信息数据类
    """
    id: str
    severity: str
    title: str
    description: str
    affected_versions: str
    fixed_version: str
    published_date: str
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        Returns:
            Dict[str, Any]: 字典格式的漏洞信息
        """
        return asdict(self)

class DependencyManager:
    """
    依赖管理器
    
    功能：
    1. 解析requirements.txt文件
    2. 检查依赖版本状态
    3. 扫描安全漏洞
    4. 分析兼容性问题
    5. 提供更新建议
    6. 生成依赖报告
    """
    
    def __init__(self, requirements_file: str = "requirements.txt"):
        """
        初始化依赖管理器
        
        Args:
            requirements_file: requirements文件路径
        """
        self.requirements_file = requirements_file
        self.dependencies = {}
        self.security_db = {}  # 安全漏洞数据库缓存
        self.pypi_cache = {}   # PyPI API缓存
        self.cache_expiry = timedelta(hours=24)  # 缓存过期时间
        
        # 已知的安全漏洞数据库（简化版）
        self.known_vulnerabilities = {
            'flask': [
                {
                    'id': 'CVE-2023-30861',
                    'severity': 'medium',
                    'title': 'Flask Cookie Parsing Vulnerability',
                    'description': 'Flask可能受到cookie解析漏洞影响',
                    'affected_versions': '<2.3.2',
                    'fixed_version': '2.3.2',
                    'published_date': '2023-05-02'
                }
            ],
            'requests': [
                {
                    'id': 'CVE-2023-32681',
                    'severity': 'medium',
                    'title': 'Requests Proxy-Authorization Header Leak',
                    'description': 'Requests可能泄露代理授权头信息',
                    'affected_versions': '<2.31.0',
                    'fixed_version': '2.31.0',
                    'published_date': '2023-05-26'
                }
            ],
            'pillow': [
                {
                    'id': 'CVE-2023-50447',
                    'severity': 'high',
                    'title': 'Pillow Arbitrary Code Execution',
                    'description': 'Pillow存在任意代码执行漏洞',
                    'affected_versions': '<10.2.0',
                    'fixed_version': '10.2.0',
                    'published_date': '2024-01-02'
                }
            ],
            'torch': [
                {
                    'id': 'GHSA-47fc-vmh7-8w2q',
                    'severity': 'medium',
                    'title': 'PyTorch TorchScript Arbitrary Code Execution',
                    'description': 'PyTorch TorchScript存在代码执行风险',
                    'affected_versions': '<2.1.2',
                    'fixed_version': '2.1.2',
                    'published_date': '2023-12-05'
                }
            ]
        }
        
        # 兼容性规则
        self.compatibility_rules = {
            'python_version': {
                'torch>=2.1.0': '>=3.8',
                'transformers>=4.35.0': '>=3.8',
                'PySide6>=6.8.0': '>=3.8',
                'langchain>=0.3.0': '>=3.8'
            },
            'conflicting_packages': [
                ('tensorflow', 'torch'),  # 可能存在冲突
                ('opencv-python', 'opencv-contrib-python')  # 不应同时安装
            ],
            'deprecated_packages': [
                'imp',  # Python 3.12中已弃用
                'distutils'  # Python 3.12中已移除
            ]
        }
        
        logging.info("依赖管理器初始化完成")
    
    def parse_requirements(self) -> Dict[str, Requirement]:
        """
        解析requirements.txt文件
        
        Returns:
            Dict[str, Requirement]: 解析后的依赖字典
        """
        try:
            requirements = {}
            
            if not os.path.exists(self.requirements_file):
                logging.warning(f"Requirements文件不存在: {self.requirements_file}")
                return requirements
            
            with open(self.requirements_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                
                # 跳过注释和空行
                if not line or line.startswith('#'):
                    continue
                
                try:
                    # 解析依赖规范
                    req = Requirement(line)
                    requirements[req.name.lower()] = req
                    
                except Exception as e:
                    logging.warning(f"解析依赖失败 (行 {line_num}): {line} - {e}")
                    continue
            
            logging.info(f"成功解析 {len(requirements)} 个依赖")
            return requirements
            
        except Exception as e:
            logging.error(f"解析requirements文件失败: {e}")
            return {}
    
    def get_installed_packages(self) -> Dict[str, str]:
        """
        获取已安装的包及其版本
        使用现代的importlib.metadata替代弃用的pkg_resources
        
        Returns:
            Dict[str, str]: 包名到版本的映射
        """
        try:
            installed = {}
            
            # 使用importlib.metadata获取已安装的包（替代pkg_resources）
            for dist in distributions():
                try:
                    # 获取包名和版本，处理可能的异常
                    if dist.metadata is not None:
                        name = dist.metadata.get('Name', '').lower()
                        version = dist.version
                        if name and version:
                            installed[name] = version
                except Exception as e:
                    logging.warning(f"处理包信息时出错: {e}")
                    continue
            
            logging.info(f"检测到 {len(installed)} 个已安装的包")
            return installed
            
        except Exception as e:
            logging.error(f"获取已安装包列表失败: {e}")
            return {}
    
    def get_latest_version(self, package_name: str, max_retries: int = 3) -> Optional[str]:
        """
        从PyPI获取包的最新版本
        优化了网络超时处理和重试机制
        
        Args:
            package_name: 包名
            max_retries: 最大重试次数
            
        Returns:
            Optional[str]: 最新版本号，获取失败返回None
        """
        try:
            # 检查缓存
            cache_key = f"latest_{package_name}"
            if cache_key in self.pypi_cache:
                cached_data = self.pypi_cache[cache_key]
                if datetime.now() - cached_data['timestamp'] < self.cache_expiry:
                    return cached_data['version']
            
            # 从PyPI API获取信息，带重试机制
            url = f"https://pypi.org/pypi/{package_name}/json"
            
            for attempt in range(max_retries):
                try:
                    # 逐步增加超时时间
                    timeout = 5 + (attempt * 2)  # 5s, 7s, 9s
                    
                    response = requests.get(
                        url, 
                        timeout=timeout,
                        headers={
                            'User-Agent': 'dependency-manager/1.0',
                            'Accept': 'application/json'
                        }
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        latest_version = data['info']['version']
                        
                        # 缓存结果
                        self.pypi_cache[cache_key] = {
                            'version': latest_version,
                            'timestamp': datetime.now()
                        }
                        
                        return latest_version
                    elif response.status_code == 404:
                        logging.warning(f"包 {package_name} 在PyPI中不存在")
                        return None
                    else:
                        logging.warning(f"获取 {package_name} 版本信息失败: HTTP {response.status_code}")
                        if attempt == max_retries - 1:
                            return None
                        
                except requests.exceptions.Timeout:
                    logging.warning(f"获取 {package_name} 版本信息超时 (尝试 {attempt + 1}/{max_retries})")
                    if attempt == max_retries - 1:
                        return None
                except requests.exceptions.ConnectionError:
                    logging.warning(f"连接PyPI失败 {package_name} (尝试 {attempt + 1}/{max_retries})")
                    if attempt == max_retries - 1:
                        return None
                except Exception as e:
                    logging.warning(f"获取 {package_name} 版本信息时发生错误: {e} (尝试 {attempt + 1}/{max_retries})")
                    if attempt == max_retries - 1:
                        return None
                
                # 重试前等待
                if attempt < max_retries - 1:
                    import time
                    time.sleep(1 + attempt)  # 1s, 2s, 3s
            
            return None
                
        except Exception as e:
            logging.error(f"获取 {package_name} 最新版本失败: {e}")
            return None
    
    def check_security_vulnerabilities(self, package_name: str, current_version: str) -> List[SecurityVulnerability]:
        """
        检查包的安全漏洞
        
        Args:
            package_name: 包名
            current_version: 当前版本
            
        Returns:
            List[SecurityVulnerability]: 安全漏洞列表
        """
        try:
            vulnerabilities = []
            
            # 检查已知漏洞数据库
            package_vulns = self.known_vulnerabilities.get(package_name.lower(), [])
            
            for vuln_data in package_vulns:
                try:
                    # 检查当前版本是否受影响
                    affected_spec = SpecifierSet(vuln_data['affected_versions'])
                    current_ver = version.parse(current_version)
                    
                    if current_ver in affected_spec:
                        vulnerability = SecurityVulnerability(
                            id=vuln_data['id'],
                            severity=vuln_data['severity'],
                            title=vuln_data['title'],
                            description=vuln_data['description'],
                            affected_versions=vuln_data['affected_versions'],
                            fixed_version=vuln_data['fixed_version'],
                            published_date=vuln_data['published_date']
                        )
                        vulnerabilities.append(vulnerability)
                        
                except Exception as e:
                    logging.warning(f"检查漏洞 {vuln_data['id']} 失败: {e}")
                    continue
            
            return vulnerabilities
            
        except Exception as e:
            logging.error(f"检查 {package_name} 安全漏洞失败: {e}")
            return []
    
    def check_compatibility_issues(self, package_name: str, current_version: str) -> List[str]:
        """
        检查兼容性问题
        
        Args:
            package_name: 包名
            current_version: 当前版本
            
        Returns:
            List[str]: 兼容性问题列表
        """
        try:
            issues = []
            
            # 检查Python版本兼容性
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            
            for pkg_spec, required_python in self.compatibility_rules['python_version'].items():
                try:
                    req = Requirement(pkg_spec)
                    if req.name.lower() == package_name.lower():
                        current_ver = version.parse(current_version)
                        if current_ver in req.specifier:
                            required_spec = SpecifierSet(required_python)
                            python_ver = version.parse(python_version)
                            
                            if python_ver not in required_spec:
                                issues.append(
                                    f"Python版本不兼容: 需要 {required_python}, 当前 {python_version}"
                                )
                except Exception as e:
                    logging.warning(f"检查Python兼容性失败: {e}")
                    continue
            
            # 检查弃用包
            if package_name.lower() in self.compatibility_rules['deprecated_packages']:
                issues.append(f"包 {package_name} 已被弃用，建议寻找替代方案")
            
            return issues
            
        except Exception as e:
            logging.error(f"检查 {package_name} 兼容性失败: {e}")
            return []
    
    def generate_update_recommendation(self, dep_info: DependencyInfo) -> str:
        """
        生成更新建议
        
        Args:
            dep_info: 依赖信息
            
        Returns:
            str: 更新建议
        """
        try:
            recommendations = []
            
            # 安全漏洞建议
            if dep_info.security_issues:
                high_severity = any(vuln['severity'] == 'high' for vuln in dep_info.security_issues)
                if high_severity:
                    recommendations.append("🚨 立即更新以修复高危安全漏洞")
                else:
                    recommendations.append("⚠️ 建议更新以修复安全漏洞")
            
            # 版本更新建议
            if dep_info.is_outdated:
                try:
                    current_ver = version.parse(dep_info.current_version)
                    latest_ver = version.parse(dep_info.latest_version)
                    
                    # 判断更新类型
                    if latest_ver.major > current_ver.major:
                        recommendations.append("📈 主版本更新可用，请仔细测试兼容性")
                    elif latest_ver.minor > current_ver.minor:
                        recommendations.append("🔄 次版本更新可用，建议更新")
                    else:
                        recommendations.append("🔧 补丁版本更新可用，安全更新")
                        
                except Exception:
                    recommendations.append("📦 有新版本可用")
            
            # 兼容性问题建议
            if dep_info.compatibility_issues:
                recommendations.append("⚡ 存在兼容性问题，请检查")
            
            # 默认建议
            if not recommendations:
                recommendations.append("✅ 当前版本状态良好")
            
            return " | ".join(recommendations)
            
        except Exception as e:
            logging.error(f"生成更新建议失败: {e}")
            return "❓ 无法生成建议"
    
    def determine_priority(self, dep_info: DependencyInfo) -> str:
        """
        确定更新优先级
        
        Args:
            dep_info: 依赖信息
            
        Returns:
            str: 优先级 ('high', 'medium', 'low')
        """
        try:
            # 高优先级：安全漏洞
            if dep_info.security_issues:
                high_severity = any(vuln['severity'] in ['high', 'critical'] for vuln in dep_info.security_issues)
                if high_severity:
                    return 'high'
            
            # 高优先级：兼容性问题
            if dep_info.compatibility_issues:
                return 'high'
            
            # 中等优先级：版本过时
            if dep_info.is_outdated:
                try:
                    current_ver = version.parse(dep_info.current_version)
                    latest_ver = version.parse(dep_info.latest_version)
                    
                    # 主版本差异
                    if latest_ver.major > current_ver.major:
                        return 'medium'
                    # 次版本差异较大
                    elif latest_ver.minor - current_ver.minor > 2:
                        return 'medium'
                    else:
                        return 'low'
                except Exception:
                    return 'medium'
            
            return 'low'
            
        except Exception as e:
            logging.error(f"确定优先级失败: {e}")
            return 'medium'
    
    def analyze_dependencies(self) -> Dict[str, DependencyInfo]:
        """
        分析所有依赖
        
        Returns:
            Dict[str, DependencyInfo]: 依赖分析结果
        """
        try:
            logging.info("开始分析依赖...")
            
            # 解析requirements文件
            requirements = self.parse_requirements()
            
            # 获取已安装的包
            installed_packages = self.get_installed_packages()
            
            analysis_results = {}
            
            for pkg_name, req in requirements.items():
                try:
                    logging.info(f"分析依赖: {pkg_name}")
                    
                    # 获取当前版本
                    current_version = installed_packages.get(pkg_name, "未安装")
                    
                    # 获取最新版本
                    latest_version = self.get_latest_version(pkg_name)
                    if not latest_version:
                        latest_version = "未知"
                    
                    # 检查是否过时
                    is_outdated = False
                    if current_version != "未安装" and latest_version != "未知":
                        try:
                            current_ver = version.parse(current_version)
                            latest_ver = version.parse(latest_version)
                            is_outdated = current_ver < latest_ver
                        except Exception:
                            pass
                    
                    # 检查安全漏洞
                    security_issues = []
                    if current_version != "未安装":
                        vulnerabilities = self.check_security_vulnerabilities(pkg_name, current_version)
                        security_issues = [vuln.to_dict() for vuln in vulnerabilities]
                    
                    # 检查兼容性问题
                    compatibility_issues = []
                    if current_version != "未安装":
                        compatibility_issues = self.check_compatibility_issues(pkg_name, current_version)
                    
                    # 创建依赖信息对象
                    dep_info = DependencyInfo(
                        name=pkg_name,
                        current_version=current_version,
                        latest_version=latest_version,
                        required_version=str(req.specifier) if req.specifier else "任意版本",
                        is_outdated=is_outdated,
                        security_issues=security_issues,
                        compatibility_issues=compatibility_issues,
                        update_recommendation="",
                        priority="low"
                    )
                    
                    # 生成更新建议和优先级
                    dep_info.update_recommendation = self.generate_update_recommendation(dep_info)
                    dep_info.priority = self.determine_priority(dep_info)
                    
                    analysis_results[pkg_name] = dep_info
                    
                except Exception as e:
                    logging.error(f"分析依赖 {pkg_name} 失败: {e}")
                    continue
            
            logging.info(f"依赖分析完成，共分析 {len(analysis_results)} 个依赖")
            return analysis_results
            
        except Exception as e:
            logging.error(f"依赖分析失败: {e}")
            return {}
    
    def generate_dependency_report(self, analysis_results: Dict[str, DependencyInfo]) -> Dict[str, Any]:
        """
        生成依赖报告
        
        Args:
            analysis_results: 依赖分析结果
            
        Returns:
            Dict[str, Any]: 依赖报告
        """
        try:
            # 统计信息
            total_deps = len(analysis_results)
            outdated_deps = sum(1 for dep in analysis_results.values() if dep.is_outdated)
            security_issues = sum(len(dep.security_issues) for dep in analysis_results.values())
            compatibility_issues = sum(len(dep.compatibility_issues) for dep in analysis_results.values())
            
            # 按优先级分组
            priority_groups = {'high': [], 'medium': [], 'low': []}
            for dep in analysis_results.values():
                priority_groups[dep.priority].append(dep.name)
            
            # 安全漏洞详情
            security_details = []
            for dep in analysis_results.values():
                for issue in dep.security_issues:
                    security_details.append({
                        'package': dep.name,
                        'current_version': dep.current_version,
                        **issue
                    })
            
            # 更新建议
            update_suggestions = []
            for dep in analysis_results.values():
                if dep.is_outdated or dep.security_issues or dep.compatibility_issues:
                    update_suggestions.append({
                        'package': dep.name,
                        'current_version': dep.current_version,
                        'latest_version': dep.latest_version,
                        'priority': dep.priority,
                        'recommendation': dep.update_recommendation,
                        'security_issues_count': len(dep.security_issues),
                        'compatibility_issues_count': len(dep.compatibility_issues)
                    })
            
            # 生成报告
            report = {
                'generated_at': datetime.now().isoformat(),
                'summary': {
                    'total_dependencies': total_deps,
                    'outdated_dependencies': outdated_deps,
                    'security_issues': security_issues,
                    'compatibility_issues': compatibility_issues,
                    'high_priority_updates': len(priority_groups['high']),
                    'medium_priority_updates': len(priority_groups['medium']),
                    'low_priority_updates': len(priority_groups['low'])
                },
                'priority_groups': priority_groups,
                'security_details': security_details,
                'update_suggestions': sorted(
                    update_suggestions,
                    key=lambda x: {'high': 3, 'medium': 2, 'low': 1}[x['priority']],
                    reverse=True
                ),
                'detailed_analysis': {
                    pkg_name: dep.to_dict()
                    for pkg_name, dep in analysis_results.items()
                }
            }
            
            return report
            
        except Exception as e:
            logging.error(f"生成依赖报告失败: {e}")
            return {'error': str(e)}
    
    def generate_updated_requirements(self, analysis_results: Dict[str, DependencyInfo]) -> str:
        """
        生成更新后的requirements.txt内容
        
        Args:
            analysis_results: 依赖分析结果
            
        Returns:
            str: 更新后的requirements.txt内容
        """
        try:
            lines = []
            lines.append("# 优化后的依赖文件")
            lines.append(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append("")
            
            # 按类别分组
            categories = {
                '核心Web框架': ['flask', 'flask-cors', 'werkzeug', 'jinja2', 'markupsafe', 'itsdangerous'],
                'HTTP和网络': ['requests', 'urllib3', 'certifi', 'charset-normalizer', 'idna'],
                'AI/ML核心': ['torch', 'transformers', 'sentence-transformers', 'numpy', 'scikit-learn'],
                '语音处理': ['librosa', 'soundfile', 'pydub', 'edge-tts'],
                'LangChain RAG': ['langchain', 'langchain-community', 'langchain-core', 'langchain-text-splitters', 'langchain-chroma', 'langchain-ollama', 'chromadb'],
                '文档处理': ['pypdf', 'python-docx', 'python-multipart', 'unstructured', 'markdown'],
                '多模态AI': ['pillow', 'opencv-python'],
                'GUI框架': ['pyside6', 'pyside6-addons', 'pyside6-essentials', 'shiboken6'],
                '系统工具': ['click', 'colorama', 'blinker'],
                '高级RAG': ['networkx', 'spacy']
            }
            
            # 为每个类别生成依赖
            for category, packages in categories.items():
                category_deps = []
                for pkg_name in packages:
                    if pkg_name in analysis_results:
                        dep = analysis_results[pkg_name]
                        
                        # 决定使用的版本
                        if dep.security_issues or dep.priority == 'high':
                            # 有安全问题或高优先级，使用最新版本
                            version_spec = f">={dep.latest_version}"
                        elif dep.is_outdated and dep.priority == 'medium':
                            # 中等优先级更新，使用当前主版本的最新
                            try:
                                current_major = version.parse(dep.current_version).major
                                latest_major = version.parse(dep.latest_version).major
                                if current_major == latest_major:
                                    version_spec = f">={dep.latest_version}"
                                else:
                                    version_spec = f">={dep.current_version},<{latest_major}.0.0"
                            except Exception:
                                version_spec = f">={dep.current_version}"
                        else:
                            # 保持当前版本
                            version_spec = f">={dep.current_version}"
                        
                        category_deps.append(f"{dep.name}{version_spec}")
                
                if category_deps:
                    lines.append(f"# {category}")
                    lines.extend(category_deps)
                    lines.append("")
            
            # 添加其他未分类的依赖
            other_deps = []
            categorized_packages = set()
            for packages in categories.values():
                categorized_packages.update(packages)
            
            for pkg_name, dep in analysis_results.items():
                if pkg_name not in categorized_packages:
                    version_spec = f">={dep.current_version}"
                    other_deps.append(f"{dep.name}{version_spec}")
            
            if other_deps:
                lines.append("# 其他依赖")
                lines.extend(other_deps)
                lines.append("")
            
            # 添加安装说明
            lines.extend([
                "# ========================================",
                "# 安装说明",
                "# ========================================",
                "#",
                "# 1. 更新pip: python -m pip install --upgrade pip",
                "# 2. 安装依赖: pip install -r requirements.txt",
                "# 3. 安装spaCy模型: python -m spacy download zh_core_web_sm",
                "# 4. 安装Ollama: https://ollama.ai",
                "# 5. 下载模型: ollama pull qwen2:0.5b",
                "#",
                "# 注意：某些包可能需要系统级依赖（如FFmpeg）"
            ])
            
            return "\n".join(lines)
            
        except Exception as e:
            logging.error(f"生成更新后的requirements失败: {e}")
            return f"# 生成失败: {e}"
    
    def save_report(self, report: Dict[str, Any], filename: str = "dependency_report.json"):
        """
        保存依赖报告到文件
        
        Args:
            report: 依赖报告
            filename: 文件名
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logging.info(f"依赖报告已保存到: {filename}")
            
        except Exception as e:
            logging.error(f"保存依赖报告失败: {e}")
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """
        运行完整的依赖分析
        
        Returns:
            Dict[str, Any]: 完整的分析报告
        """
        try:
            logging.info("开始完整依赖分析...")
            
            # 分析依赖
            analysis_results = self.analyze_dependencies()
            
            if not analysis_results:
                return {'error': '依赖分析失败'}
            
            # 生成报告
            report = self.generate_dependency_report(analysis_results)
            
            # 生成优化后的requirements
            updated_requirements = self.generate_updated_requirements(analysis_results)
            report['updated_requirements'] = updated_requirements
            
            # 保存报告
            self.save_report(report)
            
            logging.info("完整依赖分析完成")
            return report
            
        except Exception as e:
            logging.error(f"完整依赖分析失败: {e}")
            return {'error': str(e)}

def main():
    """
    主函数 - 运行依赖分析
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 创建依赖管理器
    manager = DependencyManager()
    
    # 运行完整分析
    report = manager.run_full_analysis()
    
    # 打印摘要
    if 'error' not in report:
        summary = report['summary']
        print("\n" + "="*50)
        print("依赖分析摘要")
        print("="*50)
        print(f"总依赖数量: {summary['total_dependencies']}")
        print(f"过时依赖: {summary['outdated_dependencies']}")
        print(f"安全问题: {summary['security_issues']}")
        print(f"兼容性问题: {summary['compatibility_issues']}")
        print(f"高优先级更新: {summary['high_priority_updates']}")
        print(f"中优先级更新: {summary['medium_priority_updates']}")
        print(f"低优先级更新: {summary['low_priority_updates']}")
        print("="*50)
        
        # 显示高优先级更新建议
        high_priority = [s for s in report['update_suggestions'] if s['priority'] == 'high']
        if high_priority:
            print("\n🚨 高优先级更新建议:")
            for suggestion in high_priority[:5]:  # 只显示前5个
                print(f"  • {suggestion['package']}: {suggestion['current_version']} → {suggestion['latest_version']}")
                print(f"    {suggestion['recommendation']}")
    else:
        print(f"分析失败: {report['error']}")

if __name__ == "__main__":
    main()