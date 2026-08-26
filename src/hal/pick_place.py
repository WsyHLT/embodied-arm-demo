"""抓取控制器 — 在 HAL 之上实现"接近-抓取-抬起-放置"标准动作序列。

上层(LLM 指令/任务规划)只提供目标物体与放置位置, 本模块负责:
  1. 计算抓取点位姿(物体正上方下降)
  2. 下降 → 闭合夹爪 → 抬起
  3. 移动到放置点 → 松开

夹爪在仿真中未建模刚性手指, 这里用"位置吸附"模拟:
  - close_gripper 后把物体绑定到 TCP 位姿, 抬起时物体跟随移动
  - 真机上由实际夹爪 + 力传感器决定抓取是否成功
"""
from __future__ import annotations

import numpy as np

from hal.arm_interface import ArmInterface

# 抓取安全高度(物体上方)
GRASP_APPROACH_H = 0.15
GRASP_LIFT_H = 0.30
# 吸附偏移: 物体被固定在 TCP 正下方这段距离(_sync_attached 用的 0.10)。
# 抓取/放置时 TCP 的目标高度必须补偿该偏移, 否则物体会被放到桌面以下。
GRASP_TOOL_OFFSET = 0.10
# 腕部朝下姿态 (z 轴向下, UR 常用)
DOWN_POSE = np.array([0.0, 0.0, 0.0, 3.1416, 0.0, 0.0])


class PickPlaceController:
    """在 ArmInterface 之上实现 pick & place 动作原语。"""

    def __init__(self, arm: ArmInterface, sim=None) -> None:
        self._arm = arm
        self._sim = sim          # 仅仿真后端需要, 用于吸附物体
        self._grabbed: str | None = None

    def pick(self, object_pos: np.ndarray, object_name: str | None = None) -> bool:
        """在指定物体位置执行抓取。成功返回 True。"""
        obj = np.asarray(object_pos, dtype=float)
        pre_pos = obj + np.array([0.0, 0.0, GRASP_APPROACH_H])  # 物体正上方

        # 1. 移到物体正上方(安全高度)
        self._arm.move_to_pose(np.concatenate([pre_pos, DOWN_POSE[3:]]))
        # 2. 下降接近(TCP 高过物体中心一个吸附偏移, 使夹爪指尖贴物体顶面)
        grasp_z = obj[2] + GRASP_TOOL_OFFSET
        self._arm.move_to_pose(
            np.concatenate([[obj[0], obj[1], grasp_z], DOWN_POSE[3:]]), duration=1.5
        )
        # 3. 闭合夹爪 + 吸附物体(仅仿真)
        self._arm.close_gripper(object_name)
        self._grabbed = object_name
        # 4. 抬起
        self._arm.move_to_pose(np.concatenate([pre_pos, DOWN_POSE[3:]]), duration=1.5)
        return True

    def place(self, target_pos: np.ndarray) -> None:
        """在目标位置放下物体。"""
        tgt = np.asarray(target_pos, dtype=float)
        pre_pos = tgt + np.array([0.0, 0.0, GRASP_APPROACH_H])
        self._arm.move_to_pose(np.concatenate([pre_pos, DOWN_POSE[3:]]))
        # TCP 落到目标高度 + 吸附偏移, 使物体(位于 TCP 下方 offset 处)正好落在目标高度
        drop_pos = tgt + np.array([0.0, 0.0, GRASP_TOOL_OFFSET])
        self._arm.move_to_pose(np.concatenate([drop_pos, DOWN_POSE[3:]]), duration=1.5)
        self._arm.open_gripper()
        self._grabbed = None
        self._arm.move_to_pose(np.concatenate([pre_pos, DOWN_POSE[3:]]), duration=1.5)

    def home(self) -> None:
        """回到标准 home 姿态。

        用已知 home 关节角(move_to_joint)而非手算位姿 + IK —— 位姿 IK
        可能求解出横伸/展开的关节解(机械臂铺开), 而直接归位到 home 关节角
        一定能可靠地立回原位。
        """
        self._arm.move_to_joint(np.asarray(self._sim.HOME_Q, dtype=float), duration=2.0)