# Embodied Arm Demo — 具身智能机械臂抓取放置

自然语言指令 → DeepSeek 解析 → 视觉感知 → 机械臂 IK 抓取放置 → MuJoCo 仿真 UR5e

以"真机落地"为目标设计：代码通过硬件抽象层(HAL) + UR RTDE 协议驱动机械臂，
**换真机只需改一个后端实现，上层逻辑零改动**。

## 架构

```
用户中文指令
   │
   ▼
┌───────────────────────────────────────────────┐
│  src/agent/instruction_parser.py               │
│  DeepSeek API → 结构化 {action, target, dest}  │
└───────────────────────────────────────────────┘
   │
   ▼
┌───────────────────────────────────────────────┐
│  src/agent/vision.py                           │
│  MuJoCo 俯视渲染 + 物体定位 + CLIP 分类         │
└───────────────────────────────────────────────┘
   │
   ▼
┌───────────────────────────────────────────────┐
│  src/hal/pick_place.py  PickPlaceController     │
│  接近 → 抓取 → 抬起 → 放置 (动作原语)           │
└───────────────────────────────────────────────┘
   │
   ▼
┌───────────────────────────────────────────────┐
│  src/hal/arm_interface.py  ArmInterface (HAL)  │
│  ├─ MuJoCoArm    → 仿真后端                     │
│  └─ RealURArm    → 真机后端 (ur_rtde, 占位)     │
└───────────────────────────────────────────────┘
   │
   ▼
┌───────────────────────────────────────────────┐
│  src/hal/rtde_server.py  RTDE 协议模拟器        │
│  127.0.0.1:5900 — 用真机协议驱动仿真            │
└───────────────────────────────────────────────┘
   │
   ▼
┌───────────────────────────────────────────────┐
│  src/sim/ur5e_sim.py  UR5eSim (MuJoCo UR5e)    │
│  6-DOF 机械臂 + 桌面 3 物体 + IK 求解           │
└───────────────────────────────────────────────┘
```

## 快速开始

```bash
# 依赖
pip install mujoco scipy requests pillow open_clip_torch

# DeepSeek API key(不硬编码)
set DEEPSEEK_API_KEY=sk-xxxx

# 跑 demo(首次会下载 CLIP 权重, 可用 HF 镜像加速)
set HF_ENDPOINT=https://hf-mirror.com
python main.py "帮我拿起红色方块放到蓝色方块旁边"
python main.py "把绿色圆柱拿到红色方块旁边"
python main.py "抓取蓝色方块"
```

选项:
- `--no-llm` 只用规则解析(不调 API)
- `--no-clip` 关闭 CLIP 视觉分类(更快)
- `--rtde` 启动 RTDE 协议服务器演示真机协议对接

## 真机落地设计(面试重点)

### 1. 硬件抽象层 (HAL)
`ArmInterface` 定义机械臂统一接口(位姿/关节/夹爪/读物体)。上层代码只依赖接口,
仿真(`MuJoCoArm`)与真机(`RealURArm`)可无缝替换 —— 这是"换真机零改动"的关键。

### 2. UR RTDE 协议模拟器
真实 UR 机械臂用 RTDE 协议(端口 5900)与上位机通信。`rtde_server.py` 在本机
实现简化版 RTDE 服务器, 上层代码像连真机一样读写关节/TCP, 内部桥接 MuJoCo。
**同一套上位机代码, 换真机只改 IP。**

### 3. Sim-to-Real 鲁棒性
`src/hal/realistic.py` 给仿真注入真机特性:
- 控制延迟 (50-200ms)
- 关节到位噪声 (±0.3°)
- 低速粘滞漂移

对比测试: `python test_sim2real.py` —— 显示理想控制 vs 真机特性下的
抓取放置误差, 量化验证规划链路对真实机械臂不完美的鲁棒程度。

### 4. 真机部署路径 (从仿真到 UR 真机)
1. 保留 `ArmInterface` 抽象, 实现 `RealURArm`(用 `ur_rtde` 库连真实机械臂)
2. 视觉: 仿真读 MuJoCo 坐标 → 真机换成 RGB-D 相机 + 检测模型 + 手眼标定
3. 夹爪: 仿真用"位置吸附" → 真机换成电动/真空夹爪 + 力传感器判断抓取成功
4. 轨迹: 仿真运动学插值 → 真机用 MoveIt/UR 内置轨迹规划(避免奇异/碰撞)

## 具身大脑 (Embodied Brain)

`src/agent/brain.py` + `src/agent/executor.py` — **让大模型自己思考**如何完成任务，
代码不写任何业务决策逻辑（叠放/换序/重排全由大脑规划）。

```bash
# 批处理模式: 大脑一次输出完整计划(reasoning + steps)
python brain_demo.py "把红色方块放到蓝色方块上面"
python brain_demo.py "把三个物体都叠在一起"
python brain_demo.py "交换红色方块和蓝色方块, 让蓝色方块在上"

# ReAct 模式: 大脑逐步决策 → 执行 → 观察 → 修正, 直到完成
python brain_demo.py "把红色方块放到蓝色方块上面" --react

# 交互模式: 持续输入指令, 场景跨轮保留(支持连续叠放/交换)
python brain_demo.py -i

# 加 3D 可视化窗口, 看着机械臂实时动作
python brain_demo.py -i -r
```

**核心区别**：
- `main.py`(旧)：指令→结构化字段→代码里 `if placement=="stack"` 判断怎么动
- `brain_demo.py`(新)：场景状态喂给 LLM → 大脑自己输出 `reasoning` + 多步计划
  → 执行器只翻译动作，不判断业务 → ReAct 模式下每步观察实际结果再决定下一步

支持复杂任务: 叠放、交换上下顺序、多物体重排、压住时的挪开腾位。

## 项目结构

```
embodied-arm-demo/
├── main.py                 # 主流程入口(规则/结构化模式)
├── brain_demo.py           # 具身大脑演示入口(LLM 自主规划)
├── test_ik.py              # IK 运动学自测
├── test_sim2real.py        # Sim-to-Real 鲁棒性对比
├── test_brain_swap.py      # 大脑交换叠放顺序测试
├── config/settings.py      # 配置(读环境变量)
├── assets/ur5e/            # UR5e 官方模型(meshes + XML)
├── src/
│   ├── agent/
│   │   ├── brain.py                # 具身大脑(场景感知→推理→计划)
│   │   ├── executor.py             # 大脑执行器(翻译动作+ReAct闭环)
│   │   ├── instruction_parser.py   # DeepSeek 指令 → 结构化任务
│   │   └── vision.py               # 俯视渲染 + 物体定位 + CLIP
│   ├── hal/
│   │   ├── arm_interface.py        # HAL 抽象 + 仿真/真机后端
│   │   ├── pick_place.py           # 抓取放置动作原语
│   │   ├── rtde_server.py          # UR RTDE 协议模拟器
│   │   └── realistic.py            # 真机特性模拟 (Sim-to-Real)
│   └── sim/
│       └── ur5e_sim.py             # MuJoCo UR5e 仿真 + IK
└── docs/
    └── sim_to_real.md              # Sim-to-Real 设计文档
```

## 数据来源
- UR5e 模型: google-deepmind/mujoco_menagerie (universal_robots_ur5e)
- 仿真: MuJoCo 3.x
- LLM: DeepSeek API (deepseek-chat)
- 视觉: OpenCLIP ViT-B/32