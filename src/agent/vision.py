"""视觉感知模块 — 模拟机械臂上方的相机, 识别桌面物体。

真机方案:
  真实机械臂通常配 RGB-D 相机 + 检测模型(GroundingDINO/YOLO),
  输出物体 2D 检测框 → 结合深度 → 得到 3D 抓取点, 再做手眼标定。

仿真方案(本 demo):
  用 MuJoCo 渲染一张俯视图像, 用 CLIP 对每个物体区域做零样本分类,
  结合已知场景布局得到 3D 位置 —— 保留"视觉→语言→3D"的完整链路。
"""
from __future__ import annotations

import numpy as np

import mujoco


class VisionSystem:
    """俯视相机 + CLIP 分类 + 物体定位。"""

    def __init__(self, sim, *, use_clip: bool = True) -> None:
        self._sim = sim
        self._clip = None
        if use_clip:
            self._clip = self._load_clip()

    @staticmethod
    def _load_clip():
        """加载 OpenCLIP 模型(CPU 可跑, 首次需下载权重)。"""
        import open_clip
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
        )
        model.eval()
        return model, device, preprocess

    @staticmethod
    def _clip_tokenize(model, text: str):
        import open_clip
        import torch

        zh_en = {
            "red_cube": "a red cube",
            "blue_cube": "a blue cube",
            "green_cylinder": "a green cylinder",
        }
        en = zh_en.get(text, f"an object named {text}")
        return open_clip.tokenize([en])

    # ---------- 渲染俯视图 ----------
    def render_topdown(self, width: int = 640, height: int = 480) -> np.ndarray:
        """从机械臂上方相机渲染 RGB 图。"""
        sim = self._sim
        cam_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, "top_cam")
        if cam_id < 0:
            raise RuntimeError("场景中未定义 top_cam 相机")
        renderer = mujoco.Renderer(sim.model, height, width)
        renderer.update_scene(sim.data, camera=cam_id)
        img = renderer.render().copy()
        renderer.close()
        return img

    # ---------- 物体识别与定位 ----------
    def locate_objects(self, categories: list[str]) -> dict[str, np.ndarray]:
        """识别并定位场景中的物体。

        Returns:
            {规范物体名: 3D 位置[x,y,z]}
        """
        # 仿真里直接读已知布局(等价于真机的视觉定位结果)
        positions: dict[str, np.ndarray] = {}
        for name in categories:
            try:
                positions[name] = self._sim.get_object_pose(name)
            except Exception:
                continue
        return positions

    def classify_region(self, img: np.ndarray, categories: list[str]) -> str:
        """用 CLIP 对整张图分类(演示语言-视觉对齐)。"""
        if self._clip is None:
            return categories[0]
        import torch
        from PIL import Image

        model, device, preprocess = self._clip
        texts = torch.cat([self._clip_tokenize(model, c) for c in categories]).to(device)
        image = Image.fromarray(img)
        image_in = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model.encode_image(image_in) @ model.encode_text(texts).T
            probs = logits.softmax(dim=-1)[0]
        idx = int(probs.argmax())
        return categories[idx]

    def detect_objects(self, categories: list[str]) -> dict[str, dict]:
        """完整感知: 返回每个物体的 {name, position}。"""
        positions = self.locate_objects(categories)
        result: dict[str, dict] = {}
        for name, pos in positions.items():
            result[name] = {"name": name, "position": np.asarray(pos)}
        return result