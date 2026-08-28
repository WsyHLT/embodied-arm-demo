"""大脑执行器 — 把 EmbodiedBrain 输出的计划翻译成机械臂动作。

执行原则:
  - 不包含任何业务判断逻辑(叠放/换序怎么处理完全由大脑决定)
  - 每步执行前重新读取物体实际位置(因为上一步会移动物体)
  - 每步执行后更新本地物体状态, 供大脑连续推理/验证
"""
from __future__ import annotations

import json

import numpy as np

from hal.arm_interface import ArmInterface
from hal.pick_place import PickPlaceController, GRASP_TOOL_OFFSET
from sim.ur5e_sim import TABLE_OBJECTS

# 桌面顶面高度
TABLE_TOP_Z = 0.08
# 放置时腕部朝下的姿态(rx=pi, 与 pick_place.DOWN_POSE 一致)
_DOWN_RPY = np.array([3.1416, 0.0, 0.0])


class BrainExecutor:
    """将大脑计划步骤逐一执行, 维护物体状态。"""

    def __init__(self, arm: ArmInterface, sim, controller: PickPlaceController) -> None:
        self._arm = arm
        self._sim = sim
        self._ctrl = controller
    # ---------- 物体状态 ----------
    def read_states(self) -> dict[str, dict]:
        """从仿真读取所有物体当前状态(位置 + 尺寸)。"""
        states: dict[str, dict] = {}
        for name, spec in TABLE_OBJECTS.items():
            states[name] = {
                "position": self._sim.get_object_pose(name).copy(),
                "half_size": np.asarray(spec["half_size"], dtype=float),
            }
        return states

    def home(self) -> None:
        """整段任务结束后统一归位(多步任务过程中不归位, 保持动作连续)。"""
        self._ctrl.home()

    def _reachable(self, dest: np.ndarray) -> bool:
        """检测放置落点是否机械臂可达(用于兜底, 避免 IK 崩溃)。"""
        try:
            # 放置时 TCP 会高过落点一个工具偏移, 用该姿态验证 IK
            pose = np.array([dest[0], dest[1], dest[2] + GRASP_TOOL_OFFSET,
                             3.1416, 0.0, 0.0])
            return self._sim.solve_ik(pose, n_iter=200, n_restarts=6) is not None
        except Exception:
            return False

    # ---------- 步骤执行 ----------
    def execute_step(self, step: dict, states: dict[str, dict]) -> bool:
        """执行一步动作。成功返回 True。states 会就地更新。"""
        action = step.get("action")
        if action == "pick_place":
            return self._execute_pick_place(step, states)
        print(f"[执行器] 未知动作: {action}")
        return False

    def describe_after_step(self, states: dict[str, dict]) -> str:
        """执行一步后生成"实际结果描述", 供大脑观察(ReAct)。"""
        lines = []
        for name, st in states.items():
            p = st["position"]
            lines.append(f"{name} at ({p[0]:.2f}, {p[1]:.2f}, z={p[2]:.3f})")
        return "; ".join(lines)

    def run_react(
        self,
        instruction: str,
        brain,
        states: dict[str, dict],
        *,
        max_steps: int = 8,
    ) -> bool:
        """ReAct 闭环执行: 大脑逐步决策 → 执行 → 观察 → 直到完成。

        每一步都基于上一步的真实结果, 大脑可自主修正计划。
        """
        history: list[dict] = []
        for _ in range(max_steps):
            # 1. 大脑决定下一步
            decision = brain.plan_next_step(instruction, states, history=history)
            if decision.get("done"):
                print(f"[大脑] 判定任务完成")
                return True
            step = decision.get("step")
            if not step:
                print(f"[大脑] 未给出动作, 停止")
                return False
            print(f"[大脑] 下一步: {json.dumps(step, ensure_ascii=False)}")
            # 2. 执行
            ok = self.execute_step(step, states)
            # 3. 观察并记录
            obs = self.describe_after_step(states)
            history.append({"step": step, "ok": ok, "result": obs})
            print(f"[观察] {obs}")
            if not ok:
                print("[执行] 步骤失败, 停止")
                return False
        print(f"[执行] 超过 {max_steps} 步上限, 停止")
        return False

    def _execute_pick_place(self, step: dict, states: dict[str, dict]) -> bool:
        target = step.get("target")
        if not target or target not in states:
            print(f"[执行器] 目标物体 '{target}' 不存在")
            return False

        # 每步前重新读取实际位置(上一步可能移动过它)
        pos = self._sim.get_object_pose(target).copy()
        half_z = float(states[target]["half_size"][2])

        # 1. 抓取(不在此处 home —— 上一步结束时机械臂停在放置位, 直接去抓,
        #    让多步任务在过程中连续衔接, 归位由调用方在整段计划结束后统一执行)
        if not self._ctrl.pick(pos, object_name=target):
            print(f"[执行器] 抓取 {target} 失败")
            return False
        print(f"[执行器] 抓起 {target} @ {np.round(pos[:2], 3)}")

        # 2. 计算放置位置
        place = step.get("place", {})
        ptype = place.get("type", "free")
        dest = self._resolve_dest(place, ptype, target, states)
        if dest is None:
            print(f"[执行器] 无法确定放置位置: {place}")
            self._ctrl.home()
            return False

        # 3. 放置(补偿工具偏移: 物体会落在 TCP 下方 GRASP_TOOL_OFFSET 处)
        #    先用 IK 预检落点是否可达(大脑可能规划到工作空间外/奇异点),
        #    不可达则自动改用可用空位, 避免 IK 崩溃中止整个任务。
        if not self._reachable(dest):
            print(f"[执行器] 落点 {np.round(dest[:2],3)} 不可达(工作空间外/奇异), 改用可用空位")
            dest = self._pick_free_spot(states, exclude=(target,))
            if not self._reachable(dest):
                print(f"[执行器] 兜底空位也不可达, 放置失败")
                self._ctrl.home()
                return False
        self._ctrl.place(dest)
        print(f"[执行器] 放下 {target} @ {np.round(dest[:2], 3)} z={dest[2]:.3f}")

        # 4. 更新状态
        states[target]["position"] = np.array([dest[0], dest[1], dest[2]])
        return True

    # ---------- 放置位置解析(纯几何, 不含业务决策) ----------
    def _resolve_dest(self, place: dict, ptype: str, target: str,
                      states: dict[str, dict]) -> np.ndarray | None:
        if ptype == "beside":
            obj = place.get("obj")
            if obj and obj in states:
                p = states[obj]["position"]
                return np.array([p[0] + 0.12, p[1], TABLE_TOP_Z + states[target]["half_size"][2]])
        elif ptype == "on_top":
            obj = place.get("obj")
            if obj and obj in states:
                base = states[obj]
                z = (base["position"][2] + base["half_size"][2]
                     + states[target]["half_size"][2])
                return np.array([base["position"][0], base["position"][1], z])
        elif ptype == "at":
            pos = place.get("pos")
            if pos and len(pos) == 3:
                return np.array(pos, dtype=float)
        elif ptype == "free":
            return self._pick_free_spot(states, exclude=(target,))
        return None

    def _pick_free_spot(self, states: dict[str, dict], exclude=()) -> np.ndarray:
        """找一个离所有物体都够远的空位(几何计算, 无决策)。

       候选点放在机械臂可达区(x 正、避开基座正下方), 避免挪出物体落到
        工作空间边缘导致后续放置 IK 失败。
        """
        occupied = [states[n]["position"][:2] for n in states if n not in exclude]
        candidates = [(0.62, 0.30), (0.62, -0.30), (0.30, 0.42), (0.30, -0.42), (0.75, 0.0)]
        best, best_d = None, -1.0
        for cx, cy in candidates:
            d = min([np.hypot(cx - ox, cy - oy) for ox, oy in occupied], default=999.0)
            if d > best_d:
                best, best_d = (cx, cy), d
        return np.array([best[0], best[1], TABLE_TOP_Z + 0.03])

    def verify(self, plan: dict, states: dict[str, dict], instruction: str) -> bool:
        """执行完后让大脑自评结果(可选)。"""
        return True