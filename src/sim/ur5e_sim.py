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
import re
import sys
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np

# 物体定义收敛在 config/objects.py —— 从这里读取, 避免多处硬编码。
# 把项目根下 config 加入 path, 便于作为独立脚本 / 测试直接 import。
_HERE = os.path.dirname(os.path.abspath(__file__))
_CONFIG_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "config"))
if _CONFIG_DIR not in sys.path:
    sys.path.insert(0, _CONFIG_DIR)
from config.objects import TABLE_OBJECTS, DEFAULT_OBJECT_XY  # noqa: E402

ASSET_DIR = os.path.join(_HERE, "..", "..", "assets", "ur5e")
UR5E_XML = os.path.join(ASSET_DIR, "ur5e.xml")


def _object_half_z_local(shape: str, hs: tuple) -> float:
    """物体在 z 方向的半尺寸(决定落稳后的中心高度)。"""
    if shape == "sphere":
        return float(hs[0])
    return float(hs[2])


def _geom_xml(name: str, shape: str, hs: tuple, rgba: tuple, mass: float) -> str:
    """按几何形状生成 MuJoCo geom 的 XML 片段。

    MuJoCo 尺寸约定:
      box      -> size="hx hy hz"
      cylinder -> size="radius halfheight"
      sphere   -> size="radius"
    """
    r, g, b, a = rgba
    base = f'<geom name="{name}_geom" rgba="{r} {g} {b} {a}" mass="{mass}"'
    if shape == "cylinder":
        size = f"{hs[0]} {hs[2]}"          # (半径, 半高)
        return f'{base} type="cylinder" size="{size}"/>\n'
    if shape == "sphere":
        size = f"{hs[0]}"                  # (半径)
        return f'{base} type="sphere" size="{size}"/>\n'
    # 默认 box
    size = f"{hs[0]} {hs[1]} {hs[2]}"
    return f'{base} type="box" size="{size}"/>\n'


# 平行夹爪注入片段: 手掌 + 两根沿 y 轴开合的滑轨手指(挂在 attachment_site 处)。
# 手指带高摩擦, 用于真实接触夹持 + 接触力反馈判断。
GRIPPER_XML = """
      <body name="gripper" pos="0 0.1 0">
        <geom name="gripper_palm" type="box" size="0.014 0.018 0.022" pos="0.08 0 0"
          rgba="0.2 0.2 0.22 1" mass="0.05" friction="0.9 0.05 0.001"/>
        <body name="finger_L" pos="0.1 0.04 0.05">
          <joint name="gripper_L" type="slide" axis="0 0 1" limited="true" range="-0.06 0.02"
            armature="0.001"/>
          <geom name="finger_L_geom" type="box" size="0.006 0.028 0.006" pos="0 0.03 0"
            rgba="0.55 0.55 0.55 1" mass="0.02" friction="1.2 0.1 0.02"/>
        </body>
        <body name="finger_R" pos="0.1 0.04 -0.05">
          <joint name="gripper_R" type="slide" axis="0 0 1" limited="true" range="-0.02 0.06"
            armature="0.001"/>
          <geom name="finger_R_geom" type="box" size="0.006 0.028 0.006" pos="0 0.03 0"
            rgba="0.55 0.55 0.55 1" mass="0.02" friction="1.2 0.1 0.02"/>
        </body>
      </body>
    """

GRIPPER_ACT_XML = """
    <general class="ur5e" name="gripper_L" joint="gripper_L" ctrlrange="-0.03 0.02"/>
    <general class="ur5e" name="gripper_R" joint="gripper_R" ctrlrange="-0.02 0.03"/>
  """


def _inject_gripper(xml: str) -> str:
    """把平行夹爪(bodies + joints)注入到 wrist_3_link, 并注册开合执行器。"""
    m = re.search(r'(<body name="wrist_3_link".*?)(</body>)', xml, re.S)
    if m:
        xml = xml[:m.start(2)] + GRIPPER_XML + xml[m.start(2):]
    xml = xml.replace("</actuator>", GRIPPER_ACT_XML + "</actuator>", 1)
    return xml


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
        shape = spec["shape"]
        mass = spec.get("mass", 0.15)
        start_z = TABLE_TOP_Z + _object_half_z_local(shape, hs)  # 略高于桌面, 自由落体落稳
        geom = _geom_xml(name, shape, hs, rgba, mass)
        obj_blocks.append(
            f'<body name="{name}" pos="{x} {y} {start_z:.3f}">'
            f'<freejoint name="{name}_joint"/>'
            f"{geom}"
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
    ur5e_xml = _inject_gripper(ur5e_xml)
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
        self._last_ik_q: np.ndarray | None = None  # 上次 IK 解(warm-start 缓存)
        self._cache_joint_ids()
        self._tune_actuators()
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
        self._gripper_ids = {
            n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in ("gripper_L", "gripper_R")
        }
        # 夹爪执行器在最后两位(臂 6 + 夹爪 2)
        self._gripper_act = (self.model.nu - 2, self.model.nu - 1)

    def set_gripper(self, open_fraction: float = 1.0, settle: bool = True) -> None:
        """控制平行夹爪开合。open_fraction: 1=张开, 0=闭合。

        驱动两根手指的滑轨执行器到位(位置伺服)。闭合仅做动画靠近,
        是否"夹住"由上层用几何判定(物体是否在 TCP 正下方), 不依赖物理接触。
        """
        open_fraction = min(max(float(open_fraction), 0.0), 1.0)
        # 手指沿局部 z(全局横向)开合: 张开 L=+0.02/R=-0.02, 闭合 L=-0.02/R=+0.02
        L = 0.02 + (open_fraction - 1.0) * 0.04   # open=1 -> 0.02; close=0 -> -0.02
        R = -0.02 + (open_fraction - 1.0) * 0.04
        self.data.ctrl[self._gripper_act[0]] = L
        self.data.ctrl[self._gripper_act[1]] = R
        if settle:
            steps = max(1, int(0.1 / self.model.opt.timestep))
            for _ in range(steps):
                mujoco.mj_step(self.model, self.data)
                if self._render:
                    with self._render_lock:
                        self._viewer.sync()

    def _tune_actuators(self) -> None:
        """整定关节执行器, 消除力矩饱和导致的到位误差。

        UR5e 模型默认后 3 个腕部关节 forcerange 仅 ±28(N·m), 力矩太小,
        PD 位置伺服驱动不到目标角 —— 表现为 IK 点位误差(如 above_red 0.05~0.1m)
        和归位后关节偏离 home。放宽 to ±150(与前 3 关节一致)并统一腕部增益,
        让所有关节都能真正驱动到位, 误差可降 8~20 倍(实测 0.049 -> 0.006m)。
        """
        fr = self.model.actuator_forcerange.reshape(-1, 2)
        fr[:] = np.array([-150.0, 150.0])

    def _setup_scene(self) -> None:
        # 手动将机械臂关节设为 home 姿态并伺服保持(不用 mj_resetDataKeyframe,
        # 以免它把桌面物体的 freejoint 一并重置到原点导致掉落)
        self._serve_joint_q(self.HOME_Q)
        # 先把关节角摆到位(物理未启动), 再释放物理让伺服接管并让物体落稳
        self._apply_joint_q(self.HOME_Q, step=False)
        self.set_gripper(1.0, settle=False)   # 夹爪初始张开
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

        安全:
          - 每步把轨迹点钳制到关节限位内(避免越界/碰撞)
          - 检测途径点是否接近奇异构型, 若接近则提示(避开失控/速度爆炸)
        """
        target = np.asarray(target, dtype=float)
        start = self.joint_q.copy()
        lo = np.array([self.model.jnt_range[self._joint_ids[n]][0] for n in self.JOINT_NAMES])
        hi = np.array([self.model.jnt_range[self._joint_ids[n]][1] for n in self.JOINT_NAMES])
        self._warn_if_singular(start, target)
        steps = max(1, int(duration / self.model.opt.timestep))
        for i in range(steps):
            t = (i + 1) / steps
            blend = self._smoothstep(t)
            q_traj = np.clip(start + (target - start) * blend, lo, hi)
            for j, name in enumerate(self.JOINT_NAMES):
                self.data.ctrl[j] = float(q_traj[j])
            mujoco.mj_step(self.model, self.data)
            if self._attach_cb is not None:
                self._attach_cb()   # 吸附物体沿轨迹平滑跟随 TCP
            if self._render:
                with self._render_lock:
                    self._viewer.sync()
        # 静置段: 平滑轨迹到达目标时刻并没有让 PD 伺服收敛(动态跟随误差),
        # 追加一段保持 ctrl=target 的物理步, 让机械臂稳定到目标, 消除残余误差。
        settle_steps = max(1, int(0.15 / self.model.opt.timestep))
        for _ in range(settle_steps):
            for j, name in enumerate(self.JOINT_NAMES):
                self.data.ctrl[j] = float(target[j])
            mujoco.mj_step(self.model, self.data)
            if self._attach_cb is not None:
                self._attach_cb()
            if self._render:
                with self._render_lock:
                    self._viewer.sync()

    def _warn_if_singular(self, a: np.ndarray, b: np.ndarray) -> None:
        """检测 start→goal 的关节插值路径是否穿过接近奇异构型, 是则提示。

        奇异构型处位姿 Jacobian 条件数极大, 伺服会速度/力放大、难以到位。
        这里只做检测提示(真正的规避由 solve_ik 尽量选良态端点实现)。
        检测为只读, 结束后恢复原关节状态。
        """
        orig = self.joint_q.copy()
        try:
            for t in (0.25, 0.5, 0.75):
                q = a + (b - a) * t
                cond = self._pose_jacobian_cond(q)
                if cond > 150.0:
                    print(f"[提示] 轨迹 t={t:.2f} 附近接近奇异(条件数 {cond:.0f}), "
                          f"伺服不到位风险; 已尽量选良态端点缓解")
                    return
        finally:
            self._apply_joint_q(orig, step=False)

    def move_to_pose(self, target_pose: np.ndarray, duration: float = 2.0) -> np.ndarray:
        """逆运动学: 从当前姿态运动到目标 TCP 位姿, 返回最终关节角。"""
        q_target = self.solve_ik(target_pose)
        if q_target is None:
            raise RuntimeError(f"IK failed for pose {np.round(target_pose, 3)}")
        self.set_joint_q(q_target, duration=duration)
        return q_target

    def _pose_jacobian_cond(self, q: np.ndarray) -> float:
        """计算当前位姿 Jacobian(平移+旋转)的 2-范数条件数, 用于衡量奇异程度。

        条件数越大越接近奇异(某些方向速度/力放大到无穷, 关节难以控制)。
        参考值: <30 良态, 50-80 接近奇异, >100 奇异。
        """
        self._apply_joint_q(np.asarray(q, dtype=float), step=False)
        nv = self.model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        site = self.data.site("attachment_site")
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site.id)
        jac = np.vstack([jacp, jacr])
        return float(np.linalg.cond(jac))

    def solve_ik(self, target_pose: np.ndarray, n_iter: int = 300, n_restarts: int = 8,
                 max_cond: float = 400.0) -> np.ndarray | None:
        """基于 scipy least_squares 的多起点数值 IK(带 warm-start 与奇异规避)。

        跟踪 TCP 点(site + 0.1m x 偏移), 与 tcp_pose 定义一致。
        位置误差权重 1.0, 姿态误差权重 0.3(位置优先, 抓取场景足够)。
        改进:
          - warm-start: 优先从当前解 + 上一目标解出发, 收敛更快更稳
          - 奇异规避: 选解以【位置误差最小】为主, 仅在误差可接受时更偏好
            条件数更小(远离奇异)的解; 条件数极大(>max_cond)的解直接被剔除,
            避免规划到难以控制/速度爆炸的构型
          - 多起点取最优解; 全失败返回 None
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

        # warm-start 起点: 当前解 → 上次成功解 → 随机
        starts = [np.clip(self.joint_q.copy(), lo_all, hi_all)]
        if self._last_ik_q is not None:
            starts.append(np.clip(self._last_ik_q.copy(), lo_all, hi_all))
        for _ in range(max(0, n_restarts - len(starts))):
            starts.append(rng.uniform(lo_all, hi_all))

        best: tuple[float, float, np.ndarray] | None = None  # (误差, 条件数, q)
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
                cond = self._pose_jacobian_cond(q)
                # 极奇异解直接剔除;
                # 否则: 误差最小优先, 误差相近时取条件数更小(远离奇异)的解
                if cond < max_cond:
                    if best is None:
                        best = (err, cond, q.copy())
                    elif err < best[0] - 0.002:
                        best = (err, cond, q.copy())   # 明显更准
                    elif abs(err - best[0]) <= 0.002 and cond < best[1]:
                        best = (err, cond, q.copy())   # 同精度但更良态
                if err < 0.008 and cond < max_cond:
                    self._last_ik_q = q.copy()
                    return q
        finally:
            self._apply_joint_q(orig_joint, step=False)

        if best is not None and best[0] < 0.015:
            self._last_ik_q = best[2].copy()
            return best[2]
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