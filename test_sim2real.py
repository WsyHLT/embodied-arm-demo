"""Sim-to-Real 鲁棒性对比测试。

对比两种执行方式下抓取放置的最终位置误差:
  - 理想控制(无噪声): 直接 sim 控制
  - 真机特性(延迟+噪声+粘滞): 经 RealisticController

如果真机特性下物体仍能被放到目标附近, 说明规划链路对真实机械臂
的不完美是鲁棒的 —— 这是面试里 Sim-to-Real 的关键卖点。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "config"))

from sim.ur5e_sim import UR5eSim
from hal.arm_interface import MuJoCoArm
from hal.pick_place import PickPlaceController
from hal.realistic import RealisticController, SimToRealConfig


def _place_err(arm, controller, pick_pos, place_target):
    """执行一次抓取放置, 返回最终位置误差。"""
    tgt = np.asarray(place_target, dtype=float)
    pre = tgt + np.array([0, 0, 0.15])

    # 抓取(理想移动, 保证可复现)
    arm.move_to_pose(np.concatenate([pick_pos + np.array([0, 0, 0.15]), [3.1416, 0, 0]]))
    arm.move_to_pose(np.concatenate([[pick_pos[0], pick_pos[1], pick_pos[2] - 0.01], [3.1416, 0, 0]]), duration=1.5)
    arm.close_gripper("red_cube")
    arm.move_to_pose(np.concatenate([pick_pos + np.array([0, 0, 0.15]), [3.1416, 0, 0]]), duration=1.5)

    # 放置(经 controller: 理想 or 真机)
    controller.move_to_pose(np.concatenate([pre, [3.1416, 0, 0]]))
    controller.move_to_pose(np.concatenate([tgt, [3.1416, 0, 0]]), duration=1.5)
    arm.open_gripper()
    controller.move_to_pose(np.concatenate([pre, [3.1416, 0, 0]]), duration=1.5)

    final = arm.get_object_pose("red_cube")
    return float(np.linalg.norm(final[:2] - tgt[:2]))


def main() -> None:
    pick_pos = np.array([0.45, -0.15, 0.11])
    place_target = np.array([0.57, 0.15, 0.11])

    # 理想控制
    sim = UR5eSim(render=False)
    arm = MuJoCoArm(sim)
    err_ideal = _place_err(arm, arm, pick_pos, place_target)
    sim.close()

    # 真机特性
    sim2 = UR5eSim(render=False)
    arm2 = MuJoCoArm(sim2)
    rc = RealisticController(sim2, SimToRealConfig(enabled=True))
    err_real = _place_err(arm2, rc, pick_pos, place_target)
    sim2.close()

    print("=" * 50)
    print("Sim-to-Real 鲁棒性对比 (red_cube 放置位置误差)")
    print(f"  理想控制      : {err_ideal:.4f} m")
    print(f"  真机特性(带噪声): {err_real:.4f} m")
    print(f"  结论          : {'真机特性下仍能放准, 链路鲁棒 OK' if err_real < 0.03 else '真机特性下偏差过大, 需调容差/增加吸持余量'}")
    print("=" * 50)


if __name__ == "__main__":
    main()