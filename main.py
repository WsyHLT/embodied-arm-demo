"""具身智能机械臂 demo 主流程 — 自然语言指令 → 仿真机械臂抓取放置。

链路:
  中文指令 → DeepSeek 解析(结构化任务)
           → 视觉系统(物体定位 + CLIP 分类)
           → PickPlaceController(接近-抓取-抬起-放置)
           → HAL → MuJoCo 仿真 UR5e

可选 --rtde 参数: 同时启动 RTDE 协议服务器, 演示走"真机协议"。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("src", "config"):
    sys.path.insert(0, os.path.join(HERE, p))

from config.settings import DEEPSEEK_API_KEY, AVAILABLE_OBJECTS
from agent.instruction_parser import InstructionParser
from agent.vision import VisionSystem
from hal.arm_interface import MuJoCoArm
from hal.pick_place import PickPlaceController
from sim.ur5e_sim import UR5eSim, TABLE_OBJECTS

# 放置语义: 目标物体旁 0.12m(在桌面范围内)
PLACE_OFFSET = 0.12


def object_half_z(name: str | None) -> float:
    """物体在 z 方向的半尺寸(用于叠放高度计算)。"""
    if name and name in TABLE_OBJECTS:
        return float(TABLE_OBJECTS[name]["half_size"][2])
    return 0.03


def resolve_destination(
    dest_name: str | None,
    objects: dict[str, dict],
    *,
    placement: str = "beside",
    target_name: str | None = None,
) -> np.ndarray:
    """把放置语义解析成具体 3D 位置。

    - beside: 放到目标物体旁边(默认 +PLACE_OFFSET)
    - stack:  放到目标物体正上方(叠起来), z = 目标顶面 + 待放物体半高
    - edge/其它: 默认落点
    """
    if dest_name and dest_name in objects:
        pos = objects[dest_name]["position"]
        if placement == "stack":
            # 叠放: 放在目标物体中心正上方, 高度 = 目标中心 + 目标半高 + 待放半高
            z = pos[2] + object_half_z(dest_name) + object_half_z(target_name)
            return np.array([pos[0], pos[1], z])
        return np.array([pos[0] + PLACE_OFFSET, pos[1], 0.11])
    return np.array([0.60, 0.30, 0.11])  # 默认落点(边缘/角落/未指定)


def pick_free_spot(objects: dict[str, dict], exclude: tuple[str, ...] = ()) -> np.ndarray:
    """在桌面上找一个离所有物体都较远的空位(用于临时挪放/重排)。"""
    occupied = [objects[n]["position"][:2] for n in objects if n not in exclude]
    candidates = [(0.60, 0.30), (0.0, -0.45), (-0.10, 0.40), (0.20, 0.40), (0.0, 0.45)]
    best: tuple[float, tuple[float, float]] | None = None
    for cx, cy in candidates:
        d = min([np.hypot(cx - ox, cy - oy) for ox, oy in occupied], default=999.0)
        if best is None or d > best[0]:
            best = (d, (cx, cy))
    return np.array([best[1][0], best[1][1], 0.11])


def execute_one(
    instruction: str,
    sim,
    arm,
    controller,
    vision,
    parser,
    *,
    use_clip: bool = True,
) -> int:
    """执行单条指令(视觉感知 → LLM 解析 → 抓取放置 → 验证)。

    复用已构造的仿真/机械臂/控制器/视觉/解析器, 供单次与交互模式共用。
    返回 0 成功, 1 失败(指令无法理解 / 目标不存在 / 抓取失败)。
    """
    print("=" * 60)
    print(f"指令: {instruction}")
    print("=" * 60)

    # 1. 视觉感知(每轮重跑, 因为物体位置会被搬运改变)
    objects = vision.detect_objects(AVAILABLE_OBJECTS)
    print("\n[视觉] 识别到物体:")
    for name, info in objects.items():
        print(f"   {name}: {np.round(info['position'], 3)}")
    if use_clip:
        img = vision.render_topdown()
        cls = vision.classify_region(img, AVAILABLE_OBJECTS)
        print(f"   [CLIP] 整图分类: {cls}")

    # 2. LLM 指令解析
    plan = parser.parse(instruction)
    print(f"\n[规划] {json.dumps(plan, ensure_ascii=False)}")

    if plan.get("action") == "unknown":
        print("[结果] 无法理解指令, 未执行动作")
        return 1

    target = plan.get("target")
    if not target or target not in objects:
        print(f"[结果] 目标物体 '{target}' 未在场景中")
        return 1

    # 3. 执行抓取放置
    placement = plan.get("placement", "beside")
    dest_name = plan.get("destination")

    def do_pick(name: str, pos) -> bool:
        controller.home()
        ok = controller.pick(np.asarray(pos, dtype=float), object_name=name)
        if not ok:
            print("[结果] 抓取失败")
        return ok

    def do_place(pos) -> None:
        controller.place(np.asarray(pos, dtype=float))

    if placement == "stack" and dest_name and dest_name in objects:
        tz = objects[target]["position"][2]
        dz = objects[dest_name]["position"][2]
        xy_d = float(np.linalg.norm(
            objects[target]["position"][:2] - objects[dest_name]["position"][:2]))
        # 换序条件: 必须 target 真的被 dest 压在正下方(水平接近 + dest 更高)。
        # 否则若 dest 只是因叠在别的东西上而更高(绿叠红上)、或两者本来并排,
        # target 并没有被 dest 压住, 直接单步叠放即可, 不能误挪 dest。
        if dz > tz + 0.04 and xy_d < 0.06:
            # target 被 dest 压在下方: 先把上层 dest 挪开腾空位, 再叠放 target。
            print(f"[换序] {dest_name} 压在 {target} 之上, 先挪开 {dest_name} 再叠放")
            temp = pick_free_spot(objects, exclude=[target, dest_name])
            if not do_pick(dest_name, objects[dest_name]["position"]):
                return 1
            do_place(temp)
            objects[dest_name]["position"] = temp.copy()
            dest_xyz = np.array([
                temp[0], temp[1],
                temp[2] + object_half_z(dest_name) + object_half_z(target),
            ])
            if not do_pick(target, objects[target]["position"]):
                return 1
            do_place(dest_xyz)
            print(f"[执行] 已叠放: {target} 在 {dest_name} 上方 z={dest_xyz[2]:.3f}")
        else:
            # target 已在 dest 上方, 直接叠放
            dest_xyz = resolve_destination(dest_name, objects, placement=placement,
                                           target_name=target)
            if not do_pick(target, objects[target]["position"]):
                return 1
            do_place(dest_xyz)
            print(f"[执行] 已叠放: {target} 在 {dest_name} 上方 z={dest_xyz[2]:.3f}")
    else:
        dest_xyz = resolve_destination(dest_name, objects, placement=placement,
                                       target_name=target)
        if not do_pick(target, objects[target]["position"]):
            return 1
        do_place(dest_xyz)
        print(f"[执行] 已放到 {np.round(dest_xyz[:2], 3)}, z={dest_xyz[2]:.3f}")

    # 放置完成后机械臂自动归位
    print("[执行] 机械臂归位 ...")
    controller.home()

    # 4. 验证结果
    final_pos = sim.get_object_pose(target)
    dist = float(np.linalg.norm(final_pos[:2] - dest_xyz[:2]))
    print(f"\n[验证] {target} 最终位置 {np.round(final_pos[:2], 3)}, "
          f"距目标点 {dist:.3f}m")
    ok = dist < 0.02
    print(f"[结果] {'成功' if ok else '失败'}")
    return 0 if ok else 1


def run_demo(instruction: str, *, use_llm: bool = True, use_clip: bool = True,
             keep_open: bool = False) -> int:
    print("=" * 60)
    print("具身智能机械臂 demo")
    print("=" * 60)

    # 1. 仿真环境
    sim = UR5eSim(render=True)
    arm = MuJoCoArm(sim)
    controller = PickPlaceController(arm, sim=sim)

    # 2. 视觉感知 + LLM 解析器
    vision = VisionSystem(sim, use_clip=use_clip)
    parser = InstructionParser(
        api_key=DEEPSEEK_API_KEY,
        available_objects=AVAILABLE_OBJECTS,
        use_llm=use_llm,
    )

    result = execute_one(instruction, sim, arm, controller, vision, parser,
                         use_clip=use_clip)

    if keep_open:
        print("[保持] 演示完成, 画面保持显示, 按 Ctrl+C 或关闭窗口退出")
        try:
            import mujoco
            while True:
                mujoco.mj_step(sim.model, sim.data)
                sim._viewer.sync()
                time.sleep(0.01)
        except KeyboardInterrupt:
            pass
    sim.close()
    return result


def run_interactive(*, use_llm: bool = True, use_clip: bool = True) -> int:
    """交互式 REPL: 反复输入指令执行, 直到输入 q/quit/exit 或 Ctrl+C 退出。

    场景始终复用同一仿真, 物体位置随搬运持续更新, 窗口保持打开。
    """
    print("=" * 60)
    print("具身智能机械臂交互模式")
    print("输入中文指令(如: 帮我拿起红色方块放到蓝色方块旁边)")
    print("输入 q / quit / exit 退出")
    print("=" * 60)

    sim = UR5eSim(render=True)
    arm = MuJoCoArm(sim)
    controller = PickPlaceController(arm, sim=sim)
    vision = VisionSystem(sim, use_clip=use_clip)
    parser = InstructionParser(
        api_key=DEEPSEEK_API_KEY,
        available_objects=AVAILABLE_OBJECTS,
        use_llm=use_llm,
    )

    # 初始回到 home
    controller.home()

    exit_codes = []
    while True:
        try:
            instruction = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n收到中断, 退出交互模式")
            break
        if not instruction:
            continue
        if instruction.lower() in ("q", "quit", "exit"):
            print("退出交互模式")
            break
        try:
            exit_codes.append(
                execute_one(instruction, sim, arm, controller, vision, parser,
                            use_clip=use_clip)
            )
        except KeyboardInterrupt:
            print("\n当前动作被中断, 继续交互")
        except Exception as exc:
            print(f"[错误] 执行异常: {exc}")

    sim.close()
    return 0 if all(c == 0 for c in exit_codes) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="具身智能机械臂 demo")
    parser.add_argument("instruction", nargs="?", default="帮我拿起红色方块放到蓝色方块旁边",
                        help="中文指令")
    parser.add_argument("--no-llm", action="store_true", help="禁用 DeepSeek, 仅用规则解析")
    parser.add_argument("--no-clip", action="store_true", help="禁用 CLIP 视觉分类")
    parser.add_argument("--rtde", action="store_true", help="启动 RTDE 协议服务器")
    parser.add_argument("--keep-open", action="store_true",
                        help="执行完后保持演示画面, 不自动关闭")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="交互模式: 循环输入指令, 直到手动退出")
    args = parser.parse_args()

    if args.rtde:
        from hal.rtde_server import RTDESimServer
        sim = UR5eSim(render=False)
        srv = RTDESimServer(sim)
        srv.start()
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            srv.stop()
            sim.close()
        return

    if args.interactive:
        sys.exit(run_interactive(use_llm=not args.no_llm, use_clip=not args.no_clip))

    sys.exit(run_demo(args.instruction, use_llm=not args.no_llm, use_clip=not args.no_clip,
                      keep_open=args.keep_open))


if __name__ == "__main__":
    main()