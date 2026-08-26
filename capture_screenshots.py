"""生成 GitHub 展示截图 — 程序化渲染多个场景到 docs/screenshots/。

比屏幕截图更干净、可重复。用 MuJoCo 离屏 Renderer + 自定义相机视角,
渲染: 初始场景 / 叠放 / 三物体堆叠 / 抓取过程等关键画面。

用法:
  python capture_screenshots.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("src", "config"):
    sys.path.insert(0, os.path.join(HERE, p))

import mujoco
from sim.ur5e_sim import UR5eSim

OUT_DIR = os.path.join(HERE, "docs", "screenshots")
RES = (1280, 720)


def make_camera(model, *, pos, lookat, fov=50):
    """在模型上动态注册一个固定相机(离屏渲染用)。"""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "shot_cam")
    if cam_id < 0:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "top_cam")
    return cam_id


def render(sim, cam_id, out_path, *, width=RES[0], height=RES[1]):
    renderer = mujoco.Renderer(sim.model, height, width)
    renderer.update_scene(sim.data, camera=cam_id)
    img = renderer.render()
    try:
        from PIL import Image
        Image.fromarray(img).save(out_path)
    finally:
        renderer.close()
    return out_path


def add_camera(sim, pos, lookat, fov=55):
    """用 MuJoCo 的 MjModel 克隆代价高, 改为直接写 scene——这里用现有相机位置调整。"""
    # 直接设置 top_cam / 或利用 free camera: 通过 mjv 并不支持。
    # 简化: 用模型里已有的 top_cam, 但离屏视角受限。为保证图质量,
    # 改为在读 scene XML 层面不重写——这里仅调 top_cam 达不到透视效果。
    # 因此我们不渲染相机, 而是用 mj_forward + viewer 截图(见 render_live)。
    return sim.model.camera("top_cam").id


def render_from_camera(sim, camera_id, path):
    renderer = mujoco.Renderer(sim.model, 720, 1280)
    renderer.update_scene(sim.data, camera=camera_id)
    img = renderer.render()
    try:
        from PIL import Image
        Image.fromarray(img).save(path)
    finally:
        renderer.close()
    print(f"  saved {os.path.basename(path)}")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"输出目录: {OUT_DIR}")
    print("(注: 当前用 MuJoCo 内置 top_cam 俯视视角)")
    print("如需更美观透视视角, 建议用 live 窗口按 F7/F8 调 view 后截图。")

    # 场景 1: 初始状态
    sim = UR5eSim(render=False)
    cam = sim.model.camera("top_cam").id
    render_from_camera(sim, cam, os.path.join(OUT_DIR, "01_initial.png"))
    sim.close()

    # 场景 2: 叠放 red 叠 blue 上
    sim = UR5eSim(render=False)
    sim.move_object_to("red_cube", np.array([0.45, 0.15, 0.17]))
    render_from_camera(sim, cam, os.path.join(OUT_DIR, "02_stacked_cubes.png"))
    sim.close()

    # 场景 3: 三物体堆叠
    sim = UR5eSim(render=False)
    sim.move_object_to("blue_cube", np.array([0.45, 0.0, 0.11]))
    sim.move_object_to("red_cube", np.array([0.45, 0.0, 0.17]))
    sim.move_object_to("green_cylinder", np.array([0.45, 0.0, 0.24]))
    render_from_camera(sim, cam, os.path.join(OUT_DIR, "03_tower.png"))
    sim.close()

    print("\n完成。")


if __name__ == "__main__":
    main()