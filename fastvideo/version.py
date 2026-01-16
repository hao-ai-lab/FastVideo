__version__ = "0.1.0"

from packaging import version
import torch
import re

# 判断PyTorch版本
def clean_torch_version(full_version):
    """清洗 torch 版本字符串，移除非数字和点号的部分。"""
    # 例如，将 “2.7.0+cu118” 匹配为 “2.7.0”
    match = re.match(r'^(\d+\.\d+\.\d+)', full_version)
    return match.group(1) if match else full_version

def is_torch_version_ge(target_version):
    """检查当前torch版本是否大于等于目标版本"""
    try:
        # 方法1: 如果安装了torchtune，可以使用官方工具函数
        from torchtune.utils import torch_version_ge
        return torch_version_ge(target_version)
    except ImportError:
        # 方法2: 回退到使用packaging.version进行通用比较
        current_clean = clean_torch_version(torch.__version__)
        current_version = version.parse(current_clean)
        target_version = version.parse(str(target_version))
        return current_version >= target_version

fsdp2_supported = is_torch_version_ge("2.7.0")
