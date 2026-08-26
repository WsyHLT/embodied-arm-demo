"""具身大脑演示入口 — 让大模型自己思考如何完成抓取任务。

用法:
  交互模式(推荐):  python brain_demo.py -i
  单次批处理:     python brain_demo.py "把红色方块放到蓝色方块上面"
  ReAct 逐步模式: python brain_demo.py "交换红色方块和蓝色方块的上下顺序" --react
                  或在交互模式内输入: react 指令

交互模式特性:
  - 持续输入指令, 场景物体位置跨轮次保留
  - 每轮重新感知场景 → 大脑自己思考 → 执行 → 显示结果
  - 支持连续叠放 / 交换 / 重排等复杂操作
  - 输入 q / quit / exit 退出

大脑 vs 旧流程:
  - 大脑把场景状态喂给 LLM, 让它自主输出 reasoning + 多步计划
  - 执行器只翻译动作, 不写任何叠放/换序的业务逻辑
  - ReAct 模式: 每步执行后观察实际结果再决定下一步, 可修正计划
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("src", "config"):
    sys.path.insert(0, os.path.join(HERE, p))

import numpy as np

from config.settings import DEEPSEEK_API_KEY
from agent.brain import EmbodiedBrain
from agent.executor import BrainExecutor
from hal.arm_interface import MuJoCoArm
from hal.pick_place import PickPlaceController
from sim.ur5e_sim import UR5eSim


def _fix_windows_console() -> None:
    """Windows 控制台默认 GBK, 把输入输出统一改为 UTF-8, 保证中文正常。"""
    if sys.platform == "win32":
        for stream in (sys.stdin, sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass


_fix_windows_console()


def show_scene(states: dict[str, dict]) -> None:
    """打印当前场景(大脑看到的世界)。"""
    for name, st in states.items():
        p = st["position"]
        print(f"   {name}: ({p[0]:.2f}, {p[1]:.2f}, z={p[2]:.3f})")


def run_instruction(
    instruction: str,
    sim,
    executor: BrainExecutor,
    brain: EmbodiedBrain,
    states: dict[str, dict],
    *,
    react: bool,
) -> int:
    """执行一条指令。返回 0 成功 / 1 失败。"""
    print(f"指令: {instruction}")
    print("=" * 60)

    # 每轮重新感知场景(物体位置会被上一轮搬运改变)
    states.update(executor.read_states())
    print("\n[环境] 大脑看到:")
    show_scene(states)

    print("\n[思考] 大脑正在推理 ...")
    if react:
        ok_all = executor.run_react(instruction, brain, states)
    else:
        try:
            plan = brain.plan(instruction, states)
        except Exception as exc:
            print(f"[思考] LLM 调用失败, 回退规则兜底: {exc}")
            plan = brain.plan_fallback(instruction, states)

        print(f"[思考] reasoning: {plan.get('reasoning', '')}")
        steps = plan.get("steps", [])
        print(f"[计划] 共 {len(steps)} 步:")
        for i, s in enumerate(steps, 1):
            print(f"   Step{i}: {json.dumps(s, ensure_ascii=False)}")

        ok_all = True
        print("\n[执行] ...")
        for i, step in enumerate(steps, 1):
            print(f"  Step{i} ...")
            if not executor.execute_step(step, states):
                print(f"  Step{i} 失败, 中止")
                ok_all = False
                break

    # 整段任务执行完(或中止)后统一归位: 过程中各步骤连续衔接不归位,
    # 只有最后一步完成(或中止)后把机械臂立回 home。
    print("[执行] 归位 ...")
    executor.home()

    print("\n[最终场景]")
    show_scene(states)
    print(f"[结果] {'任务完成' if ok_all else '存在失败步骤'}")
    return 0 if ok_all else 1


def run_interactive(*, react: bool = False, render: bool = False) -> int:
    """交互式 REPL: 反复输入指令, 场景持续保留。"""
    print("=" * 60)
    print("具身大脑 交互模式" + (" (ReAct)" if react else "") + (" (3D视图)" if render else ""))
    print("输入中文指令(如: 把红色方块放到蓝色方块上面)")
    print("输入 `react <指令>` 切换 ReAct 模式执行当前指令")
    print("输入 q / quit / exit 退出")
    print("=" * 60)

    sim = UR5eSim(render=render)
    arm = MuJoCoArm(sim)
    controller = PickPlaceController(arm, sim=sim)
    executor = BrainExecutor(arm, sim, controller)
    brain = EmbodiedBrain(api_key=DEEPSEEK_API_KEY)

    states = executor.read_states()
    print("\n[初始场景]")
    show_scene(states)

    # 初始回 home
    controller.home()

    exit_codes = []
    use_react = react
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n收到中断, 退出交互模式")
            break
        if not line:
            continue
        if line.lower() in ("q", "quit", "exit"):
            print("退出交互模式")
            break
        if line.lower().startswith("react "):
            use_react = True
            line = line[len("react "):].strip()
            print(f"(使用 ReAct 逐步模式执行)")
        elif line.lower() == "react":
            use_react = True
            print("已切换 ReAct 模式")
            continue
        elif line.lower() == "batch":
            use_react = False
            print("已切换批处理模式")
            continue
        if not line:
            continue
        try:
            exit_codes.append(
                run_instruction(line, sim, executor, brain, states, react=use_react)
            )
        except KeyboardInterrupt:
            print("\n当前动作被中断, 继续交互")
        except Exception as exc:
            print(f"[错误] 执行异常: {exc}")

    sim.close()
    return 0 if all(c == 0 for c in exit_codes) else 1


def main() -> None:
    args = sys.argv[1:]
    react_mode = "--react" in args
    interactive = "-i" in args or "--interactive" in args
    render = "-r" in args or "--render" in args
    args = [a for a in args if a not in ("--react", "-i", "--interactive", "-r", "--render")]

    if interactive:
        sys.exit(run_interactive(react=react_mode, render=render))

    instruction = args[0] if args else "把红色方块放到蓝色方块上面"

    sim = UR5eSim(render=render)
    arm = MuJoCoArm(sim)
    controller = PickPlaceController(arm, sim=sim)
    executor = BrainExecutor(arm, sim, controller)
    brain = EmbodiedBrain(api_key=DEEPSEEK_API_KEY)
    states = executor.read_states()

    sys.exit(run_instruction(instruction, sim, executor, brain, states, react=react_mode))


if __name__ == "__main__":
    main()