"""场景物体单点配置 — 颜色 / 形状 / 数量 / 初始位置 / 中文名 / CLIP 描述 全在此定义。

设计原则: 这是唯一需要修改的地方。改颜色、增删物体、换形状, 只需改下面的
OBJECTS 列表; src 与 config 其余模块都从这里派生所需信息, 不再硬编码。

字段说明:
  - name:      物体唯一规范名(英文, 全小写下划线)
  - shape:     几何形状 "box" | "cylinder" | "sphere"
  - half_size: box     -> (半长x, 半长y, 半长z)
               cylinder-> (半径, 半径, 半高)
               sphere  -> (半径, ...)  取第一个作半径
  - rgba:      颜色 (r,g,b,a), 0~1
  - mass:      质量(kg)
  - pos:       初始桌面位置 (x, y)
  - aliases:   中文别名 -> 映射到 name(解析/识别用)
  - clip:      CLIP 英文描述文本(零样本分类用)
"""
from __future__ import annotations

# 场景物体定义(唯一数据源)
OBJECTS: list[dict] = [
    {
        "name": "red_cube",
        "shape": "box",
        "half_size": (0.03, 0.03, 0.03),
        "rgba": (0.9, 0.1, 0.1, 1.0),
        "mass": 0.15,
        "pos": (0.45, -0.15),
        "aliases": ["红色方块", "红色的方块", "红色立方体", "红块", "红的", "红色的"],
        "clip": "a red cube",
    },
    {
        "name": "blue_cube",
        "shape": "box",
        "half_size": (0.03, 0.03, 0.03),
        "rgba": (0.1, 0.2, 0.9, 1.0),
        "mass": 0.15,
        "pos": (0.45, 0.15),
        "aliases": ["蓝色方块", "蓝色的方块", "蓝色立方体", "蓝块", "蓝的", "蓝色的"],
        "clip": "a blue cube",
    },
    {
        "name": "green_cylinder",
        "shape": "cylinder",
        "half_size": (0.03, 0.03, 0.04),
        "rgba": (0.1, 0.8, 0.2, 1.0),
        "mass": 0.15,
        "pos": (0.55, 0.0),
        "aliases": ["绿色圆柱", "绿色的圆柱", "绿色柱子", "绿圆柱", "绿的", "绿色的", "绿块", "圆柱"],
        "clip": "a green cylinder",
    },
    {
        "name": "yellow_cylinder",
        "shape": "cylinder",
        "half_size": (0.03, 0.03, 0.04),
        "rgba": (0.92, 0.78, 0.1, 1.0),
        "mass": 0.15,
        "pos": (0.2, 0.35),
        "aliases": ["黄色圆柱", "黄色的圆柱", "黄色柱子", "黄圆柱", "黄的", "黄色的"],
        "clip": "a yellow cylinder",
    },
    {
        "name": "white_cube",
        "shape": "box",
        "half_size": (0.03, 0.03, 0.03),
        "rgba": (0.92, 0.92, 0.92, 1.0),
        "mass": 0.15,
        "pos": (0.2, -0.35),
        "aliases": ["白色方块", "白色的方块", "白色立方体", "白方块", "白块", "白的", "白色的"],
        "clip": "a white cube",
    },
]

# ---------- 派生结构(由 OBJECTS 自动生成, 不要手动改下面) ----------

# 仿真场景用: {name: {half_size, rgba, shape, mass}}
TABLE_OBJECTS: dict[str, dict] = {
    o["name"]: {
        "half_size": tuple(o["half_size"]),
        "rgba": tuple(o["rgba"]),
        "shape": o["shape"],
        "mass": float(o.get("mass", 0.15)),
    }
    for o in OBJECTS
}

# 初始桌面位置: {name: (x, y)}
DEFAULT_OBJECT_XY: dict[str, tuple[float, float]] = {
    o["name"]: tuple(o["pos"]) for o in OBJECTS
}

# 中文别名 -> 规范名(指令解析用)
OBJECT_ALIASES: dict[str, str] = {
    alias: o["name"] for o in OBJECTS for alias in o["aliases"]
}

# 物体名 -> CLIP 英文描述(视觉分类用)
CLIP_DESCRIPTIONS: dict[str, str] = {o["name"]: o["clip"] for o in OBJECTS}

# 场景物体清单(LLM/视觉需要知道哪些物体可选)
AVAILABLE_OBJECTS: list[str] = [o["name"] for o in OBJECTS]


def object_half_z(name: str) -> float:
    """物体在 z 方向的半尺寸(用于叠放高度计算)。"""
    spec = TABLE_OBJECTS.get(name)
    if not spec:
        return 0.03
    hs = spec["half_size"]
    if spec["shape"] == "sphere":
        return float(hs[0])
    return float(hs[2])
