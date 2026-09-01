"""DeepSeek 指令解析 — 把自然语言指令解析为结构化抓取任务。

输入示例:
  "帮我拿起红色方块放到蓝色方块旁边"
输出(JSON):
  {
    "action": "pick_and_place",
    "target": "red_cube",     // 要抓的物体
    "destination": "near_blue_cube",  // 放置语义
    "objects": ["red_cube", "blue_cube"]  // 场景中识别的候选
  }

使用 DeepSeek API, 通过 system prompt 强制输出 JSON,
并让模型只从给定的物体清单里选物体(避免幻觉出不存在的物体)。
"""
from __future__ import annotations

import json
import os
import re
import sys

import requests

# 物体别名单点配置在 config/objects.py, 此处只导入, 不再各自硬编码。
_HERE = os.path.dirname(os.path.abspath(__file__))
_CONFIG_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "config"))
if _CONFIG_DIR not in sys.path:
    sys.path.insert(0, _CONFIG_DIR)
from config.objects import OBJECT_ALIASES  # noqa: E402

# 叠放语义关键词 → placement = "stack"(放正上方); 否则 "beside"(放旁边)
STACK_KEYWORDS = ["上面", "之上", "上头", "叠", "堆", "摞"]

# 位置语义 → 放置到默认落点
EDGE_ALIASES = ["棋盘边缘", "棋盘最边缘", "最边缘", "边缘", "角落"]

SYSTEM_PROMPT = """你是机械臂操作规划器。根据用户的中文指令和场景物体清单，
输出 JSON(不要输出其他任何文字), 格式:
{"action": "pick_and_place" | "pick" | "place" | "unknown",
 "target": "<物体规范名或 null>",
 "destination": "<放置目标物体规范名或 'original_position' 或 null>",
 "placement": "beside" | "stack" | "edge",
 "reason": "<一句话解释>"}

规则:
- 只能从 given objects 清单里选物体, 清单外的名字返回 null
- "拿/抓/取/放/移动/挪" 等词 → action 合理推断
- 放到某个物体旁边/附近 → destination 填该物体, placement="beside"
- 放到某个物体上面/之上/叠起来/堆起来 → destination 填该物体, placement="stack"
- 放到角落/边缘/远处 → destination=null, placement="edge"
- 无法理解 → action="unknown"
场景物体清单: {objects}"""


class InstructionParser:
    """DeepSeek API 指令解析器。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        available_objects: list[str] | None = None,
        use_llm: bool = True,
    ) -> None:
        self._api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self._base_url = base_url
        self._model = model
        self._available = available_objects or ["red_cube", "blue_cube", "green_cylinder"]
        self._use_llm = use_llm

    def parse(self, instruction: str) -> dict:
        """解析指令, 返回结构化动作。

        默认优先走 DeepSeek LLM(配了 DEEPSEEK_API_KEY 就调用), 失败或
        未配置 key 时回退到规则解析。可用 use_llm=False 强制只用规则。
        """
        if self._use_llm:
            if self._api_key:
                try:
                    result = self._llm_parse(instruction)
                    if result.get("reason"):
                        result["reason"] = f"DeepSeek: {result['reason']}"
                    return result
                except Exception as exc:
                    print(f"[提示] LLM 解析失败, 回退规则解析: {exc}")
            # 无 key 或 LLM 失败 → 兜底规则解析
            return self._rule_parse(instruction)
        return self._rule_parse(instruction)

    def _rule_parse(self, instruction: str) -> dict:
        """基于物体别名的轻量规则解析。

        按各物体在指令中的**出现位置**排序(句子里先提到的是 target,
        后提到的作为 destination)。每个物体取**最长命中别名**, 并做区间消除:
        若某短别名(如"圆柱")被更长别名(如"黄色圆柱")的子串覆盖, 则丢弃。
        否则,"把黄色圆柱放到白色方块上"会被误判出 green(因"圆柱"→green)。
        """
        # 按物体聚合别名
        groups: dict[str, list[str]] = {}
        for alias, canon in OBJECT_ALIASES.items():
            groups.setdefault(canon, []).append(alias)

        # 每个物体收集其在指令中命中的别名, 只保留最长者
        matched: list[tuple[int, int, str]] = []  # (start, end, canon)
        for canon, aliases in groups.items():
            found = [(instruction.find(a), a) for a in aliases if instruction.find(a) != -1]
            if not found:
                continue
            best_a = max(found, key=lambda x: len(x[1]))  # 最长命中别名
            idx = instruction.find(best_a[1])
            matched.append((idx, idx + len(best_a[1]), canon))

        # 区间消除: 长别名优先, 被覆盖的短别名丢弃(如"圆柱"嵌套在"黄色圆柱"内)
        matched.sort(key=lambda x: -(x[1] - x[0]))  # 长优先
        accepted: list[tuple[int, int, str]] = []
        for start, end, canon in matched:
            if any(c == canon for _, _, c in accepted):
                continue
            if any(start < ae and end > s for s, ae, _ in accepted):
                continue  # 与已接受区间重叠(被子串覆盖), 丢弃该短别名
            accepted.append((start, end, canon))
        accepted.sort(key=lambda x: x[0])

        seen = [c for _, _, c in accepted]
        if not seen:
            return {"action": "unknown", "target": None, "destination": None,
                    "placement": "beside", "reason": "规则未命中"}
        # target 取先提到的物体; destination 取第一个与 target 不同的物体
        target = seen[0]
        dest = None
        for canon in seen[1:]:
            if canon != target:
                dest = canon
                break
        action = "pick_and_place" if dest else "pick"
        placement = "stack" if any(k in instruction for k in STACK_KEYWORDS) else "beside"
        # 边缘语义: 放到默认落点(非具体物体)
        if dest is None and any(k in instruction for k in EDGE_ALIASES):
            placement = "edge"
        reason = "规则解析"
        return {
            "action": action,
            "target": target,
            "destination": dest,
            "placement": placement,
            "reason": reason,
        }

    def _llm_parse(self, instruction: str) -> dict:
        objects_json = json.dumps(self._available, ensure_ascii=False)
        # 用 replace 而非 format(系统提示里含 JSON 大括号, format 会误判占位符)
        system = SYSTEM_PROMPT.replace("{objects}", objects_json)
        resp = requests.post(
            f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": instruction},
                ],
                "temperature": 0.0,
                "max_tokens": 200,
                "response_format": {"type": "json_object"},
            },
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        # 提取 JSON(容错前后缀, 包括 ```json 代码块)
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise ValueError(f"LLM 未返回 JSON: {content[:100]}")
        result = json.loads(match.group(0))
        # 确保关键键存在(模型可能缺字段)
        for key in ("action", "target", "destination"):
            if key not in result:
                result[key] = None
        if "placement" not in result:
            result["placement"] = "beside"
        return result