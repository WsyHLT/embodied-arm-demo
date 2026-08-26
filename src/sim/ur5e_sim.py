"""MuJoCo UR5e 仿真环境 — 面向"真机落地"的仿真底座。

职责:
  - 加载 UR5e 模型, 组装桌面 + 物体场景
  - 提供两类底层控制原语: 关节角控制 / 末端位姿(IK)控制
  - 暴露真机风格接口(tcp_pose/joint_q), 使上层 HAL 可以无差别接入

设计说明:
  本模块刻意不引入 pybullet/moveit, 只用 MuJoCo + 自实现 IK,
  保持依赖轻量; 上层 HAL(RTDE 模拟器)通过本模块驱动"虚拟本体"。
"""
from __future__ import annotations

import os
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np

ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "ur5e")
UR5E_XML = os.path.join(ASSET_DIR, "ur5e.xml")

# 桌面物体表: {物体名: (半尺寸, 颜色, 位置x, 位置y, 高度)}
TABLE_OBJECTS: dict[str, dict] = {
    "red_cube": {"half_size": (0.03, 0.03, 0.03), "rgba": (0.9, 0.1, 0.1, 1.0)},
    "blue_cube": {"half_size": (0.03, 0.03, 0.03), "rgba": (0.1, 0.2, 0.9, 1.0)},
    "green_cylinder": {"half_size": (0.03, 0.03, 0.04), "rgba": (0.1, 0.8, 0.2, 1.0)},
}

DEFAULT_OBJECT_XY = {
    "red_cube": (0.45, -0.15),
    "blue_cube": (0.45, 0.15),
    "green_cylinder": (0.55, 0.0),
}


def _build_scene_xml(objects: dict[str, dict] | None = None) -> str:
    """在 UR5e 官方模型基础上注入桌面与物体, 生成完整场景 XML 字符串。"""
    objects = objects or DEFAULT_OBJECT_XY
    with open(UR5E_XML, "r", encoding="utf-8") as f:
        ur5e_xml = f.read()

    # 把 meshdir 改为绝对路径(from_xml_string 以进程 cwd 解析相对路径)
    mesh_abs = os.path.abspath(os.path.join(ASSET_DIR, "assets")).replace("\\", "/")
    ur5e_xml = ur5e_xml.replace('meshdir="assets"', f'meshdir="{mesh_abs}"')

    obj_blocks: list[str] = []
    TABLE_TOP_Z = 0.08  # 桌子顶面高度(0.04 桌厚 + 0.04 中心高)
    for name, (x, y) in objects.items():
        spec = TABLE_OBJECTS[name]
        hs = spec["half_size"]
        rgba = spec["rgba"]
        start_z = TABLE_TOP_Z + hs[2]  # 物体中心略高于桌面, 自由落体落稳
        obj_blocks.append(
            f'<body name="{name}" pos="{x} {y} {start_z:.3f}">'
            f'<freejoint name="{name}_joint"/>'
            f'<geom name="{name}_geom" type="box" size="{hs[0]} {hs[1]} {hs[2]}" '
            f'rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}" mass="0.15"/>'
            f"</body>"
        )
    world_extra = (
        '<geom name="floor" type="plane" size="2 2 0.1" pos="0 0 -0.01" '
        'rgba="0.75 0.75 0.75 1"/>'
        '<geom name="table" type="box" size="0.6 0.6 0.04" pos="0.35 0 0.04" '
        'rgba="0.6 0.55 0.5 1"/>'
        '<camera name="top_cam" pos="0.35 0 0.9" xyaxes="1 0 0 0 1 0" '
        'fovy="60" mode="fixed"/>'
        + "".join("    " + b for b in obj_blocks)
    )
    ur5e_xml = ur5e_xml.replace("</worldbody>", world_extra + "\n  </worldbody>")
    return ur5e_xml


class UR5eSim:
    """UR5e 仿真本体。

    对外暴露的接口刻意对齐真实 UR 机械臂:
      - joint_q:        6 关节角(rad)
      - tcp_pose:       末端位姿 [x,y,z,rx,ry,rz] (UR 标准姿势表示)
      - get_object_pose: 读取物体位姿(替代真机上的外部视觉)
    """

    JOINT_NAMES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    # UR 系列 TCP(工具中心点)相对 wrist_3_link 的偏移
    TCP_OFFSET = np.array([0.0, 0.1, 0.0])
    # 标准 home 关节角(机械臂立起)
    HOME_Q = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])

    def __init__(
        self,
        object_xy: dict[str, tuple[float, float]] | None = None,
        *,
        render: bool = False,
        time_scale: float = 1.0,
    ) -> None:
        # 直接 from_xml_string 加载(物体已注入 XML, 无需临时文件)
        self.model = mujoco.MjModel.from_xml_string(self._build_scene(object_xy))
        self.data = mujoco.MjData(self.model)
        self._render = render
        self._time_scale = time_scale
        self._render_lock = threading.Lock()
        self._attach_cb = None  # 夹爪吸附回调, 由 HAL 层注册, 每物理步跟随物体
        self._cache_joint_ids()
        # 先把物理推进到稳定状态(机械臂到home、物体落稳), 再打开 viewer。
        # 否则渲染模式会先弹窗、再逐帧播放"机械臂就位 + 物体下落"的过渡帧,
        # 用户看到的就是横伸机械臂 + 悬空物体的错误起始画面。
        self._setup_scene()
        self._init_windows()

    # ---------- 初始化辅助 ----------
    def _build_scene(self, object_xy) -> str:
        return _build_scene_xml(object_xy)

    def _init_windows(self) -> None:
        if not self._render:
            self._viewer = None
            return
        self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    def _cache_joint_ids(self) -> None:
        self._joint_ids: dict[str, int] = {}
        for name in self.JOINT_NAMES:
            self._joint_ids[name] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        self._body_wrist3 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")

    def _setup_scene(self) -> None:
        # 手动将机械臂关节设为 home 姿态并伺服保持(不用 mj_resetDataKeyframe,
        # 以免它把桌面物体的 freejoint 一并重置到原点导致掉落)
        self._serve_joint_q(self.HOME_Q)
        # 先把关节角摆到位(物理未启动), 再释放物理让伺服接管并让物体落稳
        self._apply_joint_q(self.HOME_Q, step=False)
        for _ in range(500):
            mujoco.mj_step(self.model, self.data)

    # ---------- 顶层接口(真机风格) ----------
    @property
    def joint_q(self) -> np.ndarray:
        """当前 6 关节角(rad)。"""
        return np.array([self.data.qpos[self._joint_ids[n]] for n in self.JOINT_NAMES])

    @property
    def tcp_pose(self) -> np.ndarray:
        """当前 TCP 位姿 [x,y,z,rx,ry,rz], 与 UR RTDE actual_tcp_pose 同格式。"""
        pos = self.data.site("attachment_site").xpos.copy()
        rot = self.data.site("attachment_site").xmat.reshape(3, 3)
        # TCP 点 = 末端 site 沿其本地 x 轴偏移 0.1(对齐 UR 工具)
        local_offset = rot @ np.array([0.1, 0.0, 0.0])
        tcp_pos = pos + local_offset
        rpy = self._rotation_matrix_to_rpy(rot)
        return np.concatenate([tcp_pos, rpy])

    def get_object_pose(self, name: str) -> np.ndarray:
        """读取桌面物体的 3D 位置(模拟真机场景中的视觉/外部坐标)。"""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.data.xpos[body_id].copy()

    def move_object_to(self, name: str, pos: np.ndarray) -> None:
        """直接把物体移动到指定位置(仿真夹爪吸附的简化实现)。

        通过修改 freejoint 的 qpos(位置+朝向)实现, 而非 xpos ——
        因为 mj_forward 会从 qpos 重新计算 xpos, 只改 xpos 会被覆盖回原位。
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint")
        if jnt_id < 0:
            raise RuntimeError(f"物体 {name} 没有 freejoint")
        qpos_adr = self.model.jnt_qposadr[jnt_id]
        p = np.asarray(pos, dtype=float)
        self.data.qpos[qpos_adr:qpos_adr + 3] = p
        self.data.qpos[qpos_adr + 3:qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qvel[qpos_adr:qpos_adr + 6] = 0.0
        self.data.xpos[body_id] = p
        mujoco.mj_forward(self.model, self.data)

    # ---------- 底层控制原语 ----------
    def set_joint_q(self, target: np.ndarray, duration: float = 2.0) -> None:
        """将机械臂平滑过渡到目标关节角(smoothstep 轨迹 + 物理伺服跟踪)。

        关键: 不能用"直接设 ctrl=最终目标"—— PD 伺服会尽快收敛,
        剩余时间全静止, 动画表现为"瞬间冲到位再停住"。
        正确做法是用 smoothstep 时间曲线生成整段目标关节轨迹, 每步把
        轨迹点作为 ctrl 喂给物理伺服跟踪。机械臂于是按 duration 精确、
        连续、平滑地运动, 全程无瞬移。
        """
        target = np.asarray(target, dtype=float)
        start = self.joint_q.copy()
        steps = max(1, int(duration / self.model.opt.timestep))
        for i in range(steps):
            t = (i + 1) / steps
            blend = self._smoothstep(t)
            q_traj = start + (target - start) * blend
            for j, name in enumerate(self.JOINT_NAMES):
                self.data.ctrl[j] = float(q_traj[j])
            mujoco.mj_step(self.model, self.data)
            if self._attach_cb is not None:
                self._attach_cb()   # 吸附物体沿轨迹平滑跟随 TCP
            if self._render:
                with self._render_lock:
                    self._viewer.sync()

    def move_to_pose(self, target_pose: np.ndarray, duration: float = 2.0) -> np.ndarray:
        """逆运动学: 从当前姿态运动到目标 TCP 位姿, 返回最终关节角。"""
        q_target = self.solve_ik(target_pose)
        if q_target is None:
            raise RuntimeError(f"IK failed for pose {np.round(target_pose, 3)}")
        self.set_joint_q(q_target, duration=duration)
        return q_target

    def solve_ik(self, target_pose: np.ndarray, n_iter: int = 300, n_restarts: int = 8) -> np.ndarray | None:
        """基于 scipy least_squares 的多起点数值 IK。

        跟踪 TCP 点(site + 0.1m x 偏移), 与 tcp_pose 定义一致。
        位置误差权重 1.0, 姿态误差权重 0.3(位置优先, 抓取场景足够)。
        单次求解容易陷入局部最优, 故从当前解 + 多个随机初始角出发,
        取位置误差最小且满足容差的解。返回关节角; 全部失败返回 None。
        """
        from scipy.optimize import least_squares

        target_pose = np.asarray(target_pose, dtype=float)
        target_pos = target_pose[:3]
        target_rpy = target_pose[3:]
        target_rot = self._rpy_to_rotation_matrix(target_rpy)
        rng = np.random.default_rng(seed=int(sum(target_pos) * 1000) % 2**31)

        # 记录求解前真实关节角, 结束时恢复 —— 否则 _apply_joint_q 会反复改写
        # qpos, 导致 move_to_pose 在动作开始前把机械臂瞬移到某个 IK 候选解,
        # 这正是"动作之间一帧一帧跳"的根源。
        orig_joint = self.joint_q.copy()

        def residual(q: np.ndarray) -> np.ndarray:
            self._apply_joint_q(q, step=False)
            site = self.data.site("attachment_site")
            pos = site.xpos.copy()
            rot = site.xmat.reshape(3, 3)
            tcp = pos + rot @ np.array([0.1, 0.0, 0.0])
            err_pos = target_pos - tcp
            dR = target_rot @ rot.T
            angle = np.arccos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0))
            if angle < 1e-6:
                err_rot = np.zeros(3)
            else:
                ax = np.array([
                    dR[2, 1] - dR[1, 2],
                    dR[0, 2] - dR[2, 0],
                    dR[1, 0] - dR[0, 1],
                ]) / (2.0 * np.sin(angle))
                err_rot = ax * angle
            return np.concatenate([err_pos, err_rot * 0.3])

        # 以模型真实关节限位作为求解边界
        lo_all = np.array([self.model.jnt_range[self._joint_ids[n]][0] for n in self.JOINT_NAMES])
        hi_all = np.array([self.model.jnt_range[self._joint_ids[n]][1] for n in self.JOINT_NAMES])
        bounds = (lo_all, hi_all)
        starts = [np.clip(self.joint_q.copy(), lo_all, hi_all)]
        for _ in range(max(0, n_restarts - 1)):
            starts.append(rng.uniform(lo_all, hi_all))

        best: tuple[float, np.ndarray] | None = None
        try:
            for q0 in starts:
                res = least_squares(
                    residual, q0, bounds=bounds, method="trf",
                    max_nfev=n_iter, ftol=1e-8, xtol=1e-8,
                )
                if res is None:
                    continue
                q = res.x
                self._apply_joint_q(q, step=False)
                site = self.data.site("attachment_site")
                pos = site.xpos.copy()
                rot = site.xmat.reshape(3, 3)
                tcp = pos + rot @ np.array([0.1, 0.0, 0.0])
                err = float(np.linalg.norm(tcp - target_pos))
                if best is None or err < best[0]:
                    best = (err, q.copy())
                if err < 0.01:
                    return q
        finally:
            self._apply_joint_q(orig_joint, step=False)

        if best is not None and best[0] < 0.015:
            return best[1]
        return None

    # ---------- 内部物理推进 ----------
    def _serve_joint_q(self, q: np.ndarray) -> None:
        """通过执行器伺服把关节驱动到目标角(位置控制, 保留物体物理)。"""
        for i, name in enumerate(self.JOINT_NAMES):
            self.data.ctrl[i] = float(q[i])

    def _apply_joint_q(self, q: np.ndarray, step: bool = True) -> None:
        """直接设置关节角并前向更新(仅用于 IK 数值求解, 不驱动物理)。

        钳制到模型关节限位内, 避免 least_squares 有限差分把 qpos 推到
        MuJoCo 范围之外导致前向计算出错/NaN。
        """
        for i, name in enumerate(self.JOINT_NAMES):
            jid = self._joint_ids[name]
            lo, hi = self.model.jnt_range[jid]
            qv = min(max(float(q[i]), lo), hi)
            self.data.qpos[jid] = qv
        mujoco.mj_forward(self.model, self.data)
        if step:
            self._step(1)

    def _step(self, frames: int) -> None:
        dt = self.model.opt.timestep * self._time_scale
        for _ in range(frames):
            mujoco.mj_step(self.model, self.data)
            if self._render:
                with self._render_lock:
                    self._viewer.sync()
        time.sleep(0.0)  # 让出 GIL

    @staticmethod
    def _smoothstep(t: float) -> float:
        t = min(max(t, 0.0), 1.0)
        return t * t * (3.0 - 2.0 * t)

    @staticmethod
    def _rpy_to_rotation_matrix(rpy) -> np.ndarray:
        rx, ry, rz = rpy
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        return np.array([
            [cy * cz, -cy * sz, sy],
            [cz * sx * sy + cx * sz, cx * cz - sx * sy * sz, -cy * sx],
            [-cx * cz * sy + sx * sz, cz * sx + cx * sy * sz, cx * cy],
        ])

    @staticmethod
    def _rotation_matrix_to_rpy(R) -> np.ndarray:
        R = np.asarray(R, dtype=float)
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        if sy > 1e-6:
            rx = np.arctan2(R[2, 1], R[2, 2])
            ry = np.arctan2(-R[2, 0], sy)
            rz = np.arctan2(R[1, 0], R[0, 0])
        else:
            rx = np.arctan2(-R[1, 2], R[1, 1])
            ry = np.arctan2(-R[2, 0], sy)
            rz = 0.0
        return np.array([rx, ry, rz])

    def set_attach_callback(self, cb) -> None:
        """注册夹爪吸附回调。每个物理步(过轨迹伺服时)都会被调用,
        用于让被抓物体沿 TCP 轨迹平滑跟随, 而非运动结束时瞬移。"""
        self._attach_cb = cb

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()


if __name__ == "__main__":
    sim = UR5eSim(render=False)
    print("home qpos:", np.round(sim.joint_q, 3))
    print("tcp pose:", np.round(sim.tcp_pose, 3))
    for name in TABLE_OBJECTS:
        print(f"object {name}:", np.round(sim.get_object_pose(name), 3))
    sim.close()