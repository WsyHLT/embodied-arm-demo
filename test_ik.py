"""验证 UR5eSim 的 IK 运动学功能。"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from sim.ur5e_sim import UR5eSim

sim = UR5eSim(render=False)
print("home tcp:", np.round(sim.tcp_pose, 3) if (np := __import__("numpy")) else "")

import numpy as np

targets = {
    "above_red":   [0.45, -0.15, 0.25, 3.1416, 0.0, 0.0],   # 红块正上方
    "above_blue":  [0.45, 0.15, 0.25, 3.1416, 0.0, 0.0],    # 蓝块正上方
    "above_green": [0.55, 0.00, 0.28, 3.1416, 0.0, 0.0],    # 绿圆柱正上方
}

for name, pose in targets.items():
    pose = np.array(pose, dtype=float)
    q = sim.move_to_pose(pose, duration=1.2)
    actual = sim.tcp_pose
    err = np.linalg.norm(actual[:3] - pose[:3])
    print(f"{name}: target={np.round(pose[:3],3)} actual={np.round(actual[:3],3)} "
          f"pos_err={err:.4f} q={np.round(q,3)}")

print("\nALL IK MOVES DONE")
sim.close()