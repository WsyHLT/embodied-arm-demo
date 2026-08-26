"""项目配置 — 统一读取环境变量, 不在代码中硬编码密钥。"""
from __future__ import annotations

import os
import sys

# DeepSeek API
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# 场景物体清单(单点配置, 见 config/objects.py)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from config.objects import AVAILABLE_OBJECTS  # noqa: E402