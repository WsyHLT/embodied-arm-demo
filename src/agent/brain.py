"""具身大脑(Embodied Brain) — 让大模型自己"思考"如何完成抓取任务。

与 instruction_parser(规则/单步结构化)不同, 本模块:
  - 把实时场景状态(物体位置/尺寸/可执行技能)喂给 LLM
  - 让 LLM 自主输出: reasoning(思考过程) + steps(多步动作计划)
  - 每步动作是独立的原子操作, 由执行器按序翻译执行

关键设计:
  1. 大脑"看见"环境(物体位置、谁压在谁上面) → 自己推断该做什么
  2. 支持复杂任务: 叠放、交换上下顺序、多物体重排
  3. 执行器不写死任何业务逻辑 —— 完全听大脑的安排

动作原语(技能):
  - pick_place   {target, place: {type, obj?/pos?}}  抓取一个物体放到指定位置
  - place_type:
      "beside"  -> {obj}          放到某物体旁边
      "on_top"  -> {obj}          叠放到某物体正上方(高度自动算)
      "at"      -> {pos: [x,y,z]} 放到指定坐标
      "free"    -> {}             放到桌面空位
"""
from __future__ import annotations

import json
import os
import re

import requests

SYSTEM_PROMPT = """你是具身智能机械臂的操作规划大脑。
用户会给你一个任务指令和当前场景状态。你要自己"思考"怎么完成, 输出多步动作计划。

【场景状态】
物体清单(name: {half_size:[x,y,z], 当前中心位置 pos:[x,y,z]}):
{objects}

【桌面规则】
- 桌面顶面 z=0.08
- 物体可以放在桌面任意空位, 也可以叠放在其他物体正上方
- 同一物体同一时刻只有一个位置; 挪动前先看它现在在哪
- 如果某物体压在另一物体上面(z 更高且水平接近), 要先挪开上面的才能动下面的
- 用 "at" 指定坐标时, 必须选机械臂可达的区域: x/y 大致在 0.1~0.85 之间,
  且避开机械臂底座正下方(x=0, y=0 附近) —— 底座正下方是奇异区, 机械臂够不到。
  优先选桌面中部或靠前的开阔空位(如 x=0.6,y=0.3 或 x=0.2,y=0.4 附近)。

【你能做的动作(按顺序执行)】
1. pick_place: 抓取一个物体放到目标位置
   格式: {{"action":"pick_place","target":"<物体名>","place":{{"type":"beside","obj":"<目标物体>"}}}}
   或   {{"action":"pick_place","target":"<物体名>","place":{{"type":"on_top","obj":"<目标物体>"}}}}
   或   {{"action":"pick_place","target":"<物体名>","place":{{"type":"at","pos":[x,y,z]}}}}
   或   {{"action":"pick_place","target":"<物体名>","place":{{"type":"free"}}}}

【输出要求】
只输出一个 JSON(不要任何其他文字), 格式:
{{
  "reasoning": "用一两句话说明你的计划思路(你看到了什么、为什么这么做)",
  "steps": [
     <上面的动作对象, 一个或多个>,
     ...
  ]
}}
物体名必须严格等于清单里的名字。步骤数量按需, 复杂任务可以多步。

【关键: 先想清楚最终状态再定步骤】
- 用户的指令描述的是"最终要变成什么样子"(谁在谁上面、谁在谁旁边)
- 规划前先在脑海里模拟: 最终每个物体应该在哪
- 再反推步骤。特别地:
  * "A 放到 B 上面" → 最终 A 在上, B 在下
  * "交换 A 和 B 的位置/上下顺序" → 最终 A 和 B 互换方位, 务必让用户指定的在上
  * 若目标物体被另一物体压住, 先挪开压住它的物体(放到空位), 最后再放回正确位置
  * 放回时用 on_top 时注意: 它会让物体叠到目标之上 —— 想清楚谁该在上

【绝不能擅自替换物体】
- 只能操作场景物体清单里【确实存在】的物体。
- 若用户提到的物体(颜色/形状)在清单里找不到(比如用户说"黄色柱子"而场景没有黄色),
  **绝对不要用其他物体代替, 也不要擅自去抓别的物体**。
- 此时 reasoning 说明"场景中没有该物体, 未执行", steps 输出空列表 [] —— 什么都不做。
- 宁可少做、明确说做不到, 也不要把用户没指定的物体错拿成目标。"""


class EmbodiedBrain:
    """具身大脑: 观察场景 → LLM 推理 → 输出多步计划。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
    ) -> None:
        self._api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self._base_url = base_url
        self._model = model

    def plan(self, instruction: str, object_states: dict[str, dict]) -> dict:
        """根据指令与实时物体状态, 让 LLM 输出行动计划。

        object_states: {name: {"position": ndarray, "half_size": (x,y,z)}}
        Returns: {"reasoning": str, "steps": [...]}
        """
        resp = requests.post(
            f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": self._system_prompt(object_states)},
                    {"role": "user", "content": instruction},
                ],
                "temperature": 0.2,
                "max_tokens": 800,
                "response_format": {"type": "json_object"},
            },
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise ValueError(f"大脑未返回 JSON: {content[:200]}")
        plan = json.loads(match.group(0))
        if "steps" not in plan:
            plan["steps"] = []
        return plan

    def plan_next_step(
        self,
        instruction: str,
        object_states: dict[str, dict],
        *,
        history: list[dict] | None = None,
        completed: bool = False,
    ) -> dict:
        """逐步规划(ReAct): 每次只让大脑决定下一步做什么。

        history 传入已执行步骤的实际结果, 大脑据此修正判断。
        Returns: {"done": bool, "step": {...} | None, "reasoning": str}
        """
        history = history or []
        hist_desc = ""
        if history:
            hist_desc = "\n【已执行的步骤与实际结果】\n" + json.dumps(history, ensure_ascii=False)
        done_note = "\n(任务已由前面步骤完成, 无需更多动作)" if completed else ""

        prompt = (
            f"当前场景状态:\n{self._scene_desc(object_states)}\n"
            f"{hist_desc}\n"
            f"任务指令: {instruction}\n"
            f"{done_note}\n"
            f"现在只决定【下一步】。若任务已完成输出 {{\"done\": true, \"step\": null}}; "
            f"否则输出 {{\"done\": false, \"step\": <单个动作>}}。只输出 JSON。"
        )
        resp = requests.post(
            f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": self._system_prompt(object_states, step_mode=True)},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 500,
                "response_format": {"type": "json_object"},
            },
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise ValueError(f"大脑未返回 JSON: {content[:200]}")
        return json.loads(match.group(0))

    def _scene_desc(self, object_states: dict[str, dict]) -> str:
        return json.dumps(
            {
                name: {
                    "pos": [round(float(v), 3) for v in st["position"]],
                    "half_size": [round(float(v), 3) for v in st["half_size"]],
                }
                for name, st in object_states.items()
            },
            ensure_ascii=False,
        )

    def _system_prompt(self, object_states: dict[str, dict], *, step_mode: bool = False) -> str:
        objects_desc = self._scene_desc(object_states)
        sp = SYSTEM_PROMPT.replace("{objects}", objects_desc)
        if step_mode:
            sp += """
【逐步模式】
不要一次性规划全部步骤。只看当前状态和上一步结果, 决定下一步。
注意: 上一步可能移动了物体, 用最新的场景状态判断谁在哪、谁压在谁上。"""
        return sp

    def plan_fallback(self, instruction: str, object_states: dict[str, dict]) -> dict:
        """API 失败时的规则兜底: 单步抓放(靠别名识别)。

        仅保证不崩溃, 复杂任务仍需 LLM。
        """
        from agent.instruction_parser import OBJECT_ALIASES

        found = [a for a in OBJECT_ALIASES if a in instruction]
        if not found:
            return {"reasoning": "规则兜底: 无法理解", "steps": []}
        target = OBJECT_ALIASES[found[0]]
        return {
            "reasoning": "规则兜底: 单步抓放",
            "steps": [{"action": "pick_place", "target": target, "place": {"type": "free"}}],
        }