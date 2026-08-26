"""测试"交换叠放顺序" — 构造红在上蓝在下, 让大脑自己思考如何交换。

这个用例考验大脑是否理解:
  - 必须先挪开上面的(蓝? 红?)才能抓下面的
  - 交换 = 上面的挪到旁边, 下面的挪到原上面物体的位置/或按需求

场景: red_cube 叠在 blue_cube 上(蓝下红上)
任务: 交换上下顺序 → 变成 blue_cube 在 red_cube 上面
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "config"))

from config.settings import DEEPSEEK_API_KEY
from agent.brain import EmbodiedBrain
from agent.executor import BrainExecutor
from hal.arm_interface import MuJoCoArm
from hal.pick_place import PickPlaceController
from sim.ur5e_sim import UR5eSim


def build_stacked_scene(sim) -> None:
    """手动构造 red_cube 叠在 blue_cube 上的场景。"""
    # blue 在桌面
    sim.move_object_to("blue_cube", np.array([0.45, 0.0, 0.11]))
    # red 叠在 blue 上: z = blue中心(0.11) + blue半高(0.03) + red半高(0.03) = 0.17
    sim.move_object_to("red_cube", np.array([0.45, 0.0, 0.17]))
    # green 放一边
    sim.move_object_to("green_cylinder", np.array([0.55, 0.0, 0.12]))


def main() -> None:
    instruction = sys.argv[1] if len(sys.argv) > 1 else "交换红色方块和蓝色方块, 让蓝色方块在上"

    sim = UR5eSim(render=False)
    build_stacked_scene(sim)
    arm = MuJoCoArm(sim)
    controller = PickPlaceController(arm, sim=sim)
    executor = BrainExecutor(arm, sim, controller)
    brain = EmbodiedBrain(api_key=DEEPSEEK_API_KEY)

    print("=" * 60)
    print(f"指令: {instruction}")
    print("=" * 60)

    states = executor.read_states()
    print("\n[初始场景] 红叠蓝上:")
    for name, st in states.items():
        print(f"   {name}: {np.round(st['position'], 3)}")

    print("\n[思考] 大脑推理中 ...")
    try:
        plan = brain.plan(instruction, states)
    except Exception as exc:
        print(f"[思考] LLM 失败, 兜底: {exc}")
        plan = brain.plan_fallback(instruction, states)

    print(f"[思考] reasoning: {plan.get('reasoning', '')}")
    print(f"[计划] {len(plan.get('steps', []))} 步:")
    for i, s in enumerate(plan.get("steps", []), 1):
        print(f"   Step{i}: {json.dumps(s, ensure_ascii=False)}")

    print("\n[执行] ...")
    ok = True
    for i, step in enumerate(plan.get("steps", []), 1):
        print(f"  Step{i} ...")
        if not executor.execute_step(step, states):
            print(f"  Step{i} 失败")
            ok = False
            break

    print("\n[最终场景]")
    for name, st in states.items():
        print(f"   {name}: {np.round(st['position'], 3)}")

    # 验证: blue 应在 red 上方
    r, b = states["red_cube"]["position"], states["blue_cube"]["position"]
    on_top = b[2] > r[2] and np.linalg.norm(b[:2] - r[:2]) < 0.05
    print(f"[验证] blue({b[2]:.2f}) > red({r[2]:.2f}) 且水平接近: "
          f"{'交换成功' if on_top else '未交换'}")
    sim.close()


if __name__ == "__main__":
    main()