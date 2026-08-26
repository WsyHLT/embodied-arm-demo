"""项目配置 — 统一读取环境变量, 不在代码中硬编码密钥。"""
from __future__ import annotations

import os

# DeepSeek API
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# 场景物体(与 ur5e_sim.TABLE_OBJECTS 一致)
AVAILABLE_OBJECTS = ["red_cube", "blue_cube", "green_cylinder"]