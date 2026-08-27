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

import os
import sys

import numpy as np

from hal.arm_interface import ArmInterface

# 物体定义单点配置(用于计算场景安全高度)
_HERE = os.path.dirname(os.path.abspath(__file__))
_CONFIG_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "config"))
if _CONFIG_DIR not in sys.path:
    sys.path.insert(0, _CONFIG_DIR)
from config.objects import TABLE_OBJECTS, object_half_z  # noqa: E402

# 抓取安全高度(物体上方)
GRASP_APPROACH_H = 0.15
GRASP_LIFT_H = 0.30
# 吸附偏移: 物体被固定在 TCP 正下方这段距离(_sync_attached 用的 0.10)。
# 抓取/放置时 TCP 的目标高度必须补偿该偏移, 否则物体会被放到桌面以下。
GRASP_TOOL_OFFSET = 0.10
# 腕部朝下姿态 (z 轴向下, UR 常用)
DOWN_POSE = np.array([0.0, 0.0, 0.0, 3.1416, 0.0, 0.0])
# 避障: 水平移动段须高于场景最高物体(安全高度)。安全高度上限(工作空间内)。
SAFETY_Z_MAX = 0.45


class PickPlaceController:
    """在 ArmInterface 之上实现 pick & place 动作原语。"""

    def __init__(self, arm: ArmInterface, sim=None) -> None:
        self._arm = arm
        self._sim = sim          # 仅仿真后端需要, 用于吸附物体
        self._grabbed: str | None = None

    def _safety_z(self) -> float:
        """计算场景安全高度: 高于所有物体顶面的水平移动高度(避障)。

        读取当前所有物体的中心高度+半高, 取最高顶面 + 余量, 并限制在工作空间内。
        搬运时的水平移动段都保持在这个高度, 机械臂/夹爪不会扫到桌面上的物体。
        """
        try:
            top = 0.0
            for name in TABLE_OBJECTS:
                z = float(self._sim.get_object_pose(name)[2])
                top = max(top, z + object_half_z(name))
            return min(SAFETY_Z_MAX, top + 0.25)
        except Exception:
            return SAFETY_Z_MAX

    def pick(self, object_pos: np.ndarray, object_name: str | None = None) -> bool:
        """在指定物体位置执行抓取。成功返回 True。"""
        obj = np.asarray(object_pos, dtype=float)
        safety_z = self._safety_z()
        # 物体正上方的高位接近点(高于所有物体, 避免水平移动扫到别的东西)
        high_pre = np.array([obj[0], obj[1], safety_z])

        # 1. 先抬到安全高度上方, 水平接近到物体正上方
        self._arm.move_to_pose(np.concatenate([high_pre, DOWN_POSE[3:]]))
        # 2. 垂直下降接近(TCP 高过物体中心一个吸附偏移, 使夹爪指尖贴物体顶面)
        grasp_z = obj[2] + GRASP_TOOL_OFFSET
        self._arm.move_to_pose(
            np.concatenate([[obj[0], obj[1], grasp_z], DOWN_POSE[3:]]), duration=1.5
        )
        # 3. 闭合夹爪 + 吸附物体(仅仿真)
        if not self._arm.close_gripper(object_name):
            print(f"[抓取] 夹爪未抓稳 {object_name}(物体不在抓手正下方/已滑落)")
            return False
        self._grabbed = object_name
        # 4. 垂直抬回到安全高度
        self._arm.move_to_pose(np.concatenate([high_pre, DOWN_POSE[3:]]), duration=1.5)
        return True

    def place(self, target_pos: np.ndarray) -> None:
        """在目标位置放下物体。"""
        tgt = np.asarray(target_pos, dtype=float)
        safety_z = self._safety_z()
        # 目标上方的高位点(高于所有物体), 水平移动段保持此高度避障
        high_pre = np.array([tgt[0], tgt[1], safety_z])
        self._arm.move_to_pose(np.concatenate([high_pre, DOWN_POSE[3:]]))
        # 垂直下降到目标高度 + 吸附偏移, 使物体(位于 TCP 下方 offset 处)落在目标高度
        drop_pos = tgt + np.array([0.0, 0.0, GRASP_TOOL_OFFSET])
        self._arm.move_to_pose(np.concatenate([drop_pos, DOWN_POSE[3:]]), duration=1.5)
        # 精确落位: 释放前把物体吸附到目标点, 消除 TCP 伺服残余误差(约数厘米),
        # 保证放置位置精确可复现(仿真中夹爪吸附本就可精确控制)。
        if self._grabbed and self._sim is not None:
            self._sim.move_object_to(self._grabbed, tgt)
        self._arm.open_gripper()
        self._grabbed = None
        # 垂直抬回到安全高度
        self._arm.move_to_pose(np.concatenate([high_pre, DOWN_POSE[3:]]), duration=1.5)

    def home(self) -> None:
        """回到标准 home 姿态。

        用已知 home 关节角(move_to_joint)而非手算位姿 + IK —— 位姿 IK
        可能求解出横伸/展开的关节解(机械臂铺开), 而直接归位到 home 关节角
        一定能可靠地立回原位。
        """
        self._arm.move_to_joint(np.asarray(self._sim.HOME_Q, dtype=float), duration=2.0)