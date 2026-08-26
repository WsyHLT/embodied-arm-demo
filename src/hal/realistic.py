"""真机特性模拟 — 让仿真行为更接近真实 UR 机械臂。

Sim-to-Real 的核心思路: 训练/验证在仿真里做的决策, 要能经受真机的不完美:
  1. 控制延迟: 上位机下指令到关节执行有 ~50-200ms 延迟(UR RTDE 典型 ~10ms 周期)
  2. 执行噪声: 关节角到位存在机械公差 / 编码器噪声(零点几度)
  3. 摩擦/粘滞: 低速时关节有静摩擦, 到位后会有微小漂移

本模块把这些"不完美"注入到 UR5eSim 的控制层, 用于验证:
  - 抓取点位姿的鲁棒性(噪声下是否仍能吸附)
  - 验证逻辑的容差是否合理
"""
from __future__ import annotations

import time

import numpy as np


class SimToRealConfig:
    """真机特性参数, 可动态调整。"""

    def __init__(
        self,
        *,
        control_delay_ms: float = 60.0,    # 控制延迟(ms), 真机典型 50-200
        joint_noise_deg: float = 0.3,      # 关节角到位噪声(度)
        stick_friction_rad: float = 0.02,  # 低速粘滞(rad), 到位后微小漂移
        enabled: bool = True,
    ) -> None:
        self.control_delay_s = control_delay_ms / 1000.0
        self.joint_noise_rad = np.deg2rad(joint_noise_deg)
        self.stick_friction_rad = stick_friction_rad
        self.enabled = enabled
        self._rng = np.random.default_rng(42)


class RealisticController:
    """包装 UR5eSim 的运动学控制, 注入真机特性。

    用法(替换直接调用 sim.move_to_pose):
        rc = RealisticController(sim, SimToRealConfig())
        rc.move_to_pose(pose)   # 带延迟/噪声/粘滞的"真机式"执行
    """

    def __init__(self, sim, config: SimToRealConfig | None = None) -> None:
        self._sim = sim
        self._cfg = config or SimToRealConfig()

    def move_to_pose(self, target_pose: np.ndarray, duration: float = 2.0) -> None:
        """运动到目标位姿, 加入真机延迟与执行噪声。"""
        # 1. 控制延迟: 指令发出后延迟生效
        if self._cfg.enabled and self._cfg.control_delay_s > 0:
            time.sleep(self._cfg.control_delay_s)

        # 2. 求解 IK(理想解)
        q_target = self._sim.solve_ik(np.asarray(target_pose, dtype=float))
        if q_target is None:
            raise RuntimeError(f"IK failed for {np.round(target_pose, 3)}")

        # 3. 执行: 加噪声(关节到位公差)+ 粘滞(微小漂移)
        q_exec = q_target.copy()
        if self._cfg.enabled:
            q_exec = q_exec + self._cfg._rng.normal(0.0, self._cfg.joint_noise_rad, 6)
            q_exec = q_exec + self._cfg._rng.uniform(
                -self._cfg.stick_friction_rad,
                self._cfg.stick_friction_rad,
                6,
            )
        self._sim.set_joint_q(q_exec, duration=duration)

    def get_object_pose(self, name: str) -> np.ndarray:
        return self._sim.get_object_pose(name)