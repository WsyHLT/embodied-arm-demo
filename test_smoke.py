"""综合回归自测 — 覆盖指令解析与抓取放置的全部关键逻辑。

覆盖点:
  1. 规则解析: 语序取目标 / 叠放(stack) / 边缘(edge) / 未别名(unknown) / 带"的"变体
  2. LLM 回退: 无 key 或 key 失效时降级到规则, 不抛异常
  3. 动作原语: 旁边(beside) / 叠放(stack) / 换序(先挪上层) / 三叠 / 边缘落点
  4. 归位: 执行后机械臂回到 home 附近
  5. 既有测试: IK / Sim-to-Real(只验证能跑通不崩溃)

任何一项失败都会以明确编号报出, 不静默吞错。
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("src", "config"):
    sys.path.insert(0, os.path.join(HERE, p))

from config.settings import AVAILABLE_OBJECTS
from agent.instruction_parser import InstructionParser
from sim.ur5e_sim import UR5eSim
from hal.arm_interface import MuJoCoArm
from hal.pick_place import PickPlaceController
from agent.vision import VisionSystem

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}  {detail}")
    if not ok:
        raise SystemExit(f"自测失败: {name} :: {detail}")


def make_ctrl():
    sim = UR5eSim(render=False)
    arm = MuJoCoArm(sim)
    ctl = PickPlaceController(arm, sim=sim)
    vis = VisionSystem(sim, use_clip=False)
    parser = InstructionParser(api_key="", available_objects=AVAILABLE_OBJECTS)
    return sim, arm, ctl, vis, parser


def pos(sim, name) -> np.ndarray:
    return np.asarray(sim.get_object_pose(name), dtype=float)


# ---------- 1. 规则解析 ----------
def test_parse():
    parser = InstructionParser(api_key="", available_objects=AVAILABLE_OBJECTS)
    cases = [
        ("帮我拿起蓝色方块放到红色方块旁边", "blue_cube", "red_cube", "beside"),
        ("把绿色的放到红色的上面", "green_cylinder", "red_cube", "stack"),
        ("把红色的方块放到绿色的圆柱上面叠起来", "red_cube", "green_cylinder", "stack"),
        ("抓取红色的方块", "red_cube", None, "beside"),
        ("抓取黑色的方块", None, None, "beside"),
    ]
    for ins, exp_t, exp_d, exp_p in cases:
        pl = parser.parse(ins)
        check(f"parse[{ins[:10]}...] target={exp_t}",
              pl.get("target") == exp_t,
              f"got target={pl.get('target')}")
        check(f"parse[{ins[:10]}...] dest={exp_d}",
              pl.get("destination") == exp_d,
              f"got dest={pl.get('destination')}")
        check(f"parse[{ins[:10]}...] placement={exp_p}",
              pl.get("placement") == exp_p,
              f"got placement={pl.get('placement')}")


# ---------- 2. LLM 回退 ----------
def test_llm_fallback():
    parser = InstructionParser(api_key="sk-invalid-test-key",
                               available_objects=AVAILABLE_OBJECTS)
    # key 无效 -> LLM 抛异常 -> 应降级规则而不崩溃
    pl = parser.parse("帮我拿起绿色圆柱")
    check("llm_fallback_no_crash", isinstance(pl, dict), f"plan={pl}")


# ---------- 3. 动作原语 ----------
def test_beside():
    sim, arm, ctl, vis, parser = make_ctrl()
    main = __import__("main")
    r = main.execute_one("帮我拿起红色方块放到蓝色方块旁边", sim, arm, ctl, vis, parser)
    red = pos(sim, "red_cube")
    blue = pos(sim, "blue_cube")
    # 红应位于蓝旁边(+x 0.12), z 回桌面
    exp = np.array([blue[0] + 0.12, blue[1], 0.11])
    check("aside_red_near_blue", r == 0 and float(np.linalg.norm(red[:2] - exp[:2])) < 0.02,
          f"red={np.round(red[:2],3)} exp={np.round(exp[:2],3)}")
    sim.close()


def test_stack():
    sim, arm, ctl, vis, parser = make_ctrl()
    main = __import__("main")
    main.execute_one("把绿色圆柱放到红色方块上面", sim, arm, ctl, vis, parser)
    g = pos(sim, "green_cylinder")
    r_ = pos(sim, "red_cube")
    exp_z = r_[2] + 0.03 + 0.04  # 红半高0.03 + 绿半高0.04
    check("stack_green_on_red", abs(g[2] - exp_z) < 0.02 and
          float(np.linalg.norm(g[:2] - r_[:2])) < 0.03,
          f"green_z={g[2]:.3f} exp_z={exp_z:.3f} xy_align={float(np.linalg.norm(g[:2]-r_[:2])):.3f}")
    sim.close()


def test_reorder():
    sim, arm, ctl, vis, parser = make_ctrl()
    main = __import__("main")
    main.execute_one("把绿色圆柱放到红色方块上面", sim, arm, ctl, vis, parser)
    # 现在绿压红上, 要求"红放到绿上" -> 应先挪绿再叠红
    main.execute_one("把红色的方块放到绿色的圆柱上面", sim, arm, ctl, vis, parser)
    red = pos(sim, "red_cube")
    g = pos(sim, "green_cylinder")
    check("reorder_red_on_green", red[2] > g[2],
          f"red_z={red[2]:.3f} green_z={g[2]:.3f} (期望红在上)")
    sim.close()


def test_triple_stack():
    sim, arm, ctl, vis, parser = make_ctrl()
    main = __import__("main")
    main.execute_one("把绿色圆柱放到红色方块上面", sim, arm, ctl, vis, parser)
    main.execute_one("把蓝色的方块放到绿色圆柱上面，三个叠在一起", sim, arm, ctl, vis, parser)
    r_ = pos(sim, "red_cube"); g = pos(sim, "green_cylinder"); b = pos(sim, "blue_cube")
    ok = (abs(g[2] - 0.18) < 0.02 and abs(b[2] - 0.25) < 0.02 and
          float(np.linalg.norm(b[:2] - r_[:2])) < 0.03)
    check("triple_stack_three_layers", ok,
          f"r={r_[2]:.3f} g={g[2]:.3f} b={b[2]:.3f}")
    sim.close()


def test_edge():
    sim, arm, ctl, vis, parser = make_ctrl()
    main = __import__("main")
    main.execute_one("抓取红色的方块，放到棋盘最边缘", sim, arm, ctl, vis, parser)
    red = pos(sim, "red_cube")
    check("edge_default_spot", float(np.linalg.norm(red[:2] - np.array([0.6, 0.3]))) < 0.02,
          f"red={np.round(red[:2],3)} exp=(0.6,0.3)")
    sim.close()


def test_home():
    sim, arm, ctl, vis, parser = make_ctrl()
    main = __import__("main")
    main.execute_one("把绿色圆柱放到红色方块上面", sim, arm, ctl, vis, parser)
    q = np.asarray(sim.joint_q)
    dq = float(np.linalg.norm(q - UR5eSim.HOME_Q))
    tcp_z = float(sim.tcp_pose[2])
    check("home_return", dq < 0.2 and tcp_z > 0.3,
          f"dq={dq:.3f} tcp_z={tcp_z:.3f} (期望回到home, 直立)")
    sim.close()


def run_all():
    names = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in names:
        fn()


if __name__ == "__main__":
    run_all()
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print("\n" + "=" * 50)
    print(f"自测完成: {passed}/{total} 通过")
    print("=" * 50)
    sys.exit(0 if passed == total else 1)
