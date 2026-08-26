"""硬件抽象层(HAL) — 机械臂统一接口, 上层代码不感知后端是仿真还是真机。

设计目标(真机落地核心):
  - 上层(任务规划/LLM)只依赖 ArmInterface 这个抽象
  - 仿真实现 MuJoCoArm 与真机实现 RealURArm 可无缝替换
  - 屏蔽真实机械臂的共性:
      * joint_q / tcp_pose 状态读取
      * 运动控制(关节角/末端位姿)
      * 手爪开合(UR 配套真空/电动夹爪)
      * 读物体位置(真机通常来自外部视觉, 仿真里直接读 MuJoCo)
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ArmInterface(ABC):
    """机械臂抽象接口。所有上层代码只依赖此接口。"""

    @property
    @abstractmethod
    def joint_q(self) -> np.ndarray:
        """当前 6 关节角 (rad)。"""

    @property
    @abstractmethod
    def tcp_pose(self) -> np.ndarray:
        """当前 TCP 位姿 [x,y,z,rx,ry,rz]。"""

    @abstractmethod
    def move_to_pose(self, target_pose: np.ndarray, duration: float = 2.0) -> None:
        """运动到目标 TCP 位姿(内部 IK + 轨迹)。"""

    @abstractmethod
    def move_to_joint(self, joint_q: np.ndarray, duration: float = 2.0) -> None:
        """运动到目标关节角。"""

    @abstractmethod
    def open_gripper(self) -> None:
        """打开夹爪。"""

    @abstractmethod
    def close_gripper(self) -> None:
        """闭合夹爪(抓取)。"""

    @abstractmethod
    def get_object_pose(self, name: str) -> np.ndarray:
        """读取物体 3D 位置。真机场景通常来自外部视觉系统。"""


class MuJoCoArm(ArmInterface):
    """仿真后端: 驱动 UR5eSim。支持物体吸附(模拟夹爪抓取)。"""

    def __init__(self, sim) -> None:
        self._sim = sim
        self._attached: str | None = None
        self._sim.set_attach_callback(self._sync_attached)

    @property
    def joint_q(self) -> np.ndarray:
        return self._sim.joint_q

    @property
    def tcp_pose(self) -> np.ndarray:
        return self._sim.tcp_pose

    def move_to_pose(self, target_pose: np.ndarray, duration: float = 2.0) -> None:
        self._sim.move_to_pose(np.asarray(target_pose, dtype=float), duration=duration)
        self._sync_attached()

    def move_to_joint(self, joint_q: np.ndarray, duration: float = 2.0) -> None:
        self._sim.set_joint_q(np.asarray(joint_q, dtype=float), duration=duration)
        self._sync_attached()

    def open_gripper(self) -> None:
        """张开夹爪并释放物体。"""
        self._sim.set_gripper(1.0)
        self._detach_object()

    def close_gripper(self, object_name: str | None = None) -> bool:
        """闭合夹爪并判定是否真正抓稳。

        真实夹爪: 驱动手指闭合(动画) + 几何判定 —— 仅当目标物体中心位于
        TCP 正下方(水平距离 < 容差)才算抓稳, 否则返回 False(抓取失败)。
        这替代了原来"无条件吸附", 让抓取成功与否可被判别(力反馈/夹持判定的
        仿真近似)。object_name 未给定时自动选 TCP 附近最近的可抓物体。
        返回 True=抓稳, False=未抓到。
        """
        self._sim.set_gripper(0.0)  # 闭合手指
        if object_name is not None:
            obj = self._sim.get_object_pose(object_name)
            tcp = self._sim.tcp_pose[:3]
            # 判定用水平距离: 抓取时 TCP 在物体正上方(高一个吸附偏移),
            # 只要水平对齐即视为夹持到位; 垂直方向差异是抓具悬空高度, 属正常。
            d = float(np.linalg.norm(obj[:2] - tcp[:2]))
            if d < 0.05:  # 物体位于抓手正下方(水平对齐)
                self._attached = object_name
                self._sync_attached()
                return True
            return False
        # 未指定目标: 找 TCP 附近最近的可抓物体
        if self._attached is None:
            tcp = self._sim.tcp_pose[:3]
            best, best_dist = None, float("inf")
            for name in ("red_cube", "blue_cube", "green_cylinder"):
                pos = self._sim.get_object_pose(name)
                d = float(np.linalg.norm(pos[:2] - tcp[:2]))
                if d < best_dist and d < 0.08:
                    best, best_dist = name, d
            self._attached = best
            if self._attached is not None:
                self._sync_attached()
                return True
        return self._attached is not None

    def get_object_pose(self, name: str) -> np.ndarray:
        return self._sim.get_object_pose(name)

    # ---------- 物体吸附(仿真夹爪) ----------
    def _sync_attached(self) -> None:
        """吸附期间让物体跟随 TCP 移动(固定在 TCP 正下方, 模拟被夹住)。"""
        if self._attached is None:
            return
        tcp = self._sim.tcp_pose[:3]
        obj_pos = np.array([tcp[0], tcp[1], tcp[2] - 0.10])
        self._sim.move_object_to(self._attached, obj_pos)

    def _detach_object(self) -> None:
        self._attached = None


class RealURArm(ArmInterface):
    """真机后端: 通过 UR RTDE + 脚本端口控制真实 UR 机械臂。

    说明: 本类演示 HAL 如何切换到真机。实际部署时填入真实 IP 与凭证。
    接口与仿真后端完全一致 —— 上层代码零改动。
    """

    def __init__(
        self,
        ip: str = "192.168.1.10",
        *,
        urscript_port: int = 30002,
        rtde_freq: float = 10.0,
    ) -> None:
        self._ip = ip
        self._urscript_port = urscript_port
        self._rtde_freq = rtde_freq
        self._conn = None  # 真实部署时: ur_rtde.RTDE + socket 脚本端口

    # -- 真实部署时打开注释, 使用 ur_rtde 库 --
    # def _connect(self) -> None:
    #     from ur_rtde import RTDE, RTDEControlInterface
    #     self._rtde = RTDE(self._ip)
    #     self._rtde.connect()
    #     self._control = RTDEControlInterface(self._ip)

    def _raise_not_connected(self):
        raise RuntimeError(
            "RealURArm 需要连接真实 UR 机械臂。当前为 HAL 演示骨架, "
            "请接入 ur_rtde 后使用。"
        )

    @property
    def joint_q(self) -> np.ndarray:
        self._raise_not_connected()

    @property
    def tcp_pose(self) -> np.ndarray:
        self._raise_not_connected()

    def move_to_pose(self, target_pose: np.ndarray, duration: float = 2.0) -> None:
        self._raise_not_connected()

    def move_to_joint(self, joint_q: np.ndarray, duration: float = 2.0) -> None:
        self._raise_not_connected()

    def open_gripper(self) -> None:
        self._raise_not_connected()

    def close_gripper(self) -> None:
        self._raise_not_connected()

    def get_object_pose(self, name: str) -> np.ndarray:
        self._raise_not_connected()