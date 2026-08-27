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

import os
import sys

from abc import ABC, abstractmethod

import numpy as np

# 物体名集(用于检测叠放链)
_HERE = os.path.dirname(os.path.abspath(__file__))
_CONFIG_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "config"))
if _CONFIG_DIR not in sys.path:
    sys.path.insert(0, _CONFIG_DIR)
from config.objects import TABLE_OBJECTS  # noqa: E402

# 叠放判定: 水平接近阈值 / 高度差阈值
STACK_XY_EPS = 0.06
STACK_Z_EPS = 0.02


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
        self._attached_group: dict[str, np.ndarray] = {}  # 整塔: name -> 相对底部偏移
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
        """张开夹爪并释放物体。

        用 settle=False 只设置张爪目标、不额外跑物理——避免在物体精确落位后
        再经物理步导致圆柱等自由物体滚动/偏移。
        """
        self._sim.set_gripper(1.0, settle=False)
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
                self._attach_object(object_name)
                return True
            return False
        # 未指定目标: 找 TCP 附近最近的可抓物体
        if self._attached is None:
            tcp = self._sim.tcp_pose[:3]
            best, best_dist = None, float("inf")
            for name in TABLE_OBJECTS:
                pos = self._sim.get_object_pose(name)
                d = float(np.linalg.norm(pos[:2] - tcp[:2]))
                if d < best_dist and d < 0.08:
                    best, best_dist = name, d
            if best is not None:
                self._attach_object(best)
                return True
        return self._attached is not None

    def _attach_object(self, name: str) -> None:
        """把物体及其上方叠放的整塔绑定到夹爪(整塔吸附, 支持带叠放物一起搬运)。"""
        self._attached = name
        self._attached_group = self._detect_stack(name)
        self._sync_attached()

    def _detect_stack(self, bottom: str) -> dict[str, np.ndarray]:
        """检测叠放在 bottom 之上的所有物体, 返回 {name: 相对 bottom 中心偏移}。

        判定: 水平接近 bottom 且 z 明显更高(在 bottom 正上方)。这样抓起底部
        物体时, 其上叠放的整塔(如绿叠蓝上)会一起绑定移动, 避免散落。
        """
        bp = np.asarray(self._sim.get_object_pose(bottom), dtype=float)
        group: dict[str, np.ndarray] = {bottom: np.zeros(3)}
        for name in TABLE_OBJECTS:
            if name == bottom:
                continue
            p = np.asarray(self._sim.get_object_pose(name), dtype=float)
            if (float(np.linalg.norm(p[:2] - bp[:2])) < STACK_XY_EPS
                    and p[2] > bp[2] + STACK_Z_EPS):
                group[name] = p - bp
        return group

    def get_object_pose(self, name: str) -> np.ndarray:
        return self._sim.get_object_pose(name)

    # ---------- 物体吸附(仿真夹爪) ----------
    def _sync_attached(self) -> None:
        """吸附期间让整塔(底部 + 其上方叠放物)跟随 TCP 移动, 保持相对位置。

        底部物体固定在 TCP 正下方 0.10(模拟被夹住), 塔成员按相对偏移跟随 ——
        实现"带着叠放物一起搬运"。"""
        if self._attached is None or not self._attached_group:
            return
        tcp = self._sim.tcp_pose[:3]
        bottom_pos = np.array([tcp[0], tcp[1], tcp[2] - 0.10])
        for name, off in self._attached_group.items():
            self._sim.move_object_to(name, bottom_pos + off)

    def _detach_object(self) -> None:
        self._attached = None
        self._attached_group = {}


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