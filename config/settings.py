"""项目配置 — 统一读取环境变量, 不在代码中硬编码密钥。"""
from __future__ import annotations

import os
import sys


def _get_setting(name: str, default: str = "") -> str:
    """环境变量优先; 若为空(常见于 setx 后终端未刷新), 回退读注册表用户环境变量。

    这让程序对"终端没继承 setx 的新值"鲁棒 —— 只要注册表里有值就能自动取到。
    """
    val = os.environ.get(name, "").strip()
    if val:
        return val
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
            val, _ = winreg.QueryValueEx(key, name)
            return val if val else default
    except Exception:
        return default


# DeepSeek API
DEEPSEEK_API_KEY = _get_setting("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = _get_setting("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = _get_setting("DEEPSEEK_MODEL", "deepseek-chat")

# 场景物体清单(单点配置, 见 config/objects.py)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from config.objects import AVAILABLE_OBJECTS  # noqa: E402