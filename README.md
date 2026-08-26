# Embodied Arm Demo — 具身智能机械臂抓取放置

自然语言指令 → 大模型"大脑"思考 → 视觉感知 → 多步 IK 抓取放置 → MuJoCo 仿真 UR5e

一个以**真机落地**为目标、强调**硬件抽象 + 大模型决策 + Sim-to-Real 鲁棒性**的具身智能动手实践。
复用了 MuJoCo + DeepSeek + OpenCLIP 的轻量链路，可为面试/展示提供完整的"感知→决策→执行"闭环。

<table>
  <tr>
    <td align="center">
      <img width="400" height="410" alt="d85517c3c56c8599bc241af2d78abafc"
           src="https://github.com/user-attachments/assets/da51cd06-1254-499b-874a-543da339be3c" />
    </td>
    <td align="center">
      <img width="700" height="450" alt="cd91aadeaa118a0e04aa79f6126a8234"
           src="https://github.com/user-attachments/assets/a8265c5d-e59d-408e-b7b6-61ba30359754" />
    </td>
  </tr>
</table>

---

## 与实习经历的衔接

在天权机器人的实习中，我做的是**交互式服务机器人**：语音(ASR)→大模型/意图理解->RAG/POI 知识问答→语音(TTS)播放，外加声纹识别、Android 端和 RAG 网关。它让机器人"**听得懂、答得出**"。

本仓库是把同样的骨架**延伸向具身智能(Embodied AI)**的一次实践：机器人不只是"听+回答"，而是要"**看+思考+动手**"。两者共享一条主链路——

```
感知(语音 / 视觉) → 大模型理解与决策 → 行动(答复 / 机械臂动作)
```

| 层级 | 实习：天权机器人(语音交互) | 本 demo(具身智能) |
|------|---------------------------|-------------------|
| 感知 | STT 语音识别 + 声纹 | MuJoCo 俯视渲染 + OpenCLIP 视觉分类 |
| 大脑 | 意图识别 + RAG/POI 知识问答 | DeepSeek 多步任务规划(EmbodiedBrain) |
| 行动 | TTS 语音合成回复 | 机械臂 IK 抓取/叠放/换序 |
| 载体 | 服务机器人本体 | UR5e 机械臂 |

如果说实习完成的是机器人"**嘴**"，那这个 demo 正在补上它的"**手**"。

---

## 特性总览

- **大模型自主决策**：`EmbodiedBrain` 把实时场景状态喂给 DeepSeek，让它输出 `reasoning` + 多步计划，执行器只翻译动作、不写死业务逻辑——支持叠放、交换上下顺序、多物体重排。
- **ReAct 逐步模式**：每执行一步就观察实际结果再决定下一步，可自动修正计划。
- **硬件抽象层(HAL)**：统一机械臂接口，`MuJoCoArm`(仿真) 与 `RealURArm`(真机) 无缝替换，换真机上层零改动。
- **UR RTDE 协议模拟器**：在 127.0.0.1:5900 用真机协议驱动仿真，对接真机只改 IP。
- **Sim-to-Real 鲁棒性**：给仿真注入真机特性(延迟/噪声/粘滞漂移)，量化评估执行不完美的鲁棒程度。
- **连续/多步执行**：步骤间机械臂不归位、动作无缝衔接，任务完成后统一归位。
- **视觉无关执行**：夹爪用"位置吸附"模拟，抓取/叠放高度精确可复现。

---

## 架构

```
用户中文指令
   │
   ▼
┌───────────────────────────────────────────────┐
│  brain_demo.py  具身大脑入口(交互 / 单次 / ReAct)│
│  └─ agent/brain.py  EmbodiedBrain              │
│      DeepSeek API → reasoning + 多步动作计划    │
└───────────────────────────────────────────────┘
   │
   ▼
┌───────────────────────────────────────────────┐
│  agent/executor.py   BrainExecutor            │
│  翻译每一步 → 抓取/叠放/换序 → IK 可达性兜底     │
│  agent/vision.py     俯视渲染 + 物体定位 + CLIP  │
└───────────────────────────────────────────────┘
   │
   ▼
┌───────────────────────────────────────────────┐
│  hal/pick_place.py   PickPlaceController      │
│  接近 → 抓取 → 抬起 → 放置 (动作原语)           │
└───────────────────────────────────────────────┘
   │
   ▼
┌───────────────────────────────────────────────┐
│  hal/arm_interface.py  ArmInterface (HAL)     │
│  ├─ MuJoCoArm    → 仿真后端                    │
│  └─ RealURArm    → 真机后端 (ur_rtde, 占位)     │
└───────────────────────────────────────────────┘
   │
   ▼
┌───────────────────────────────────────────────┐
│  hal/rtde_server.py  RTDE 协议模拟器           │
│  sim/ur5e_sim.py     MuJoCo UR5e + IK 求解     │
└───────────────────────────────────────────────┘
```

---

## 快速开始

```bash
# 依赖
pip install mujoco scipy requests pillow open_clip_torch

# DeepSeek API key(不硬编码, 读环境变量)
setx DEEPSEEK_API_KEY sk-xxxx          # Windows 持久化; 或临时: set DEEPSEEK_API_KEY=sk-xxxx

# 首次会下载 CLIP 权重, 可用 HF 镜像加速
set HF_ENDPOINT=https://hf-mirror.com
```

### 方式一：具身大脑(推荐)

```bash
py brain_demo.py -i -r            # 交互模式 + 3D 视图
py brain_demo.py -i               # 交互模式(无 3D 视图)
py brain_demo.py "把红色的放到蓝色的上面，然后把绿色的放到边缘"   # 单次
py brain_demo.py "交换红色和蓝色的上下顺序" --react              # ReAct 逐步模式
```

交互模式内可输入 `react <指令>` 切换 ReAct、`batch` 切回批处理、`q` 退出。
适合连续叠放/交换/重排等复杂多步任务。

### 方式二：单步流程(旧入口)

```bash
py main.py "帮我拿起红色方块放到蓝色方块旁边" --no-clip   # 单次
py main.py -i --no-clip                                  # 交互式 REPL
```

选项：`--no-llm` 仅规则解析、`--no-clip` 关闭 CLIP、`--keep-open` 执行完保留画面、`-i` 交互。

---

## 指令示例

```
把蓝色的放到红色的上面，三个叠在一起
把绿色的换到下面，把红色的放到上面
把红色方块放到棋盘边缘
抓取绿色圆柱
把红色和蓝色交换位置
```

---

## 自定义场景物体

物体的颜色 / 形状 / 尺寸 / 数量 / 初始位置 / 中文名 / CLIP 描述**全部收敛在
`config/objects.py`**，改这一个文件即可，其余模块(场景生成、物体清单、指令别名、
视觉分类)都自动从它派生。不再需要到处同步。

```python
# config/objects.py
OBJECTS = [
    {
        "name": "red_cube",            # 物体唯一规范名(英文小写)
        "shape": "box",                # box | cylinder | sphere
        "half_size": (0.03, 0.03, 0.03),  # box: 半长xyz; cyl/sphere: 半径...
        "rgba": (0.9, 0.1, 0.1, 1.0),     # 颜色 (r,g,b,a)
        "mass": 0.15,                     # 质量(kg)
        "pos": (0.45, -0.15),             # 初始桌面位置 (x,y)
        "aliases": ["红色方块", "红的"],   # 中文名 -> 映射到 name
        "clip": "a red cube",              # CLIP 英文描述
    },
    ...
]
```

- **改颜色** → 改 `rgba`；**改形状** → 改 `shape`(box/cylinder/sphere)；
  **改大小** → 改 `half_size`；**改位置** → 改 `pos`。
- **加一个物体** → 在 `OBJECTS` 列表里复制并改一项即可。
- 改完直接跑 `py test_smoke.py` 回归即可。

---

## 测试

```bash
py test_smoke.py     # 综合回归: 解析/叠放/换序/三叠/边缘/归位/LLM 回退 (22 项)
py test_cases.py     # TC-01~TC-15 验收用例表 (15 项)
py test_ik.py        # IK 运动学自测
py test_sim2real.py  # Sim-to-Real 鲁棒性对比
```

---

## 项目结构

```
embodied-arm-demo/
├── brain_demo.py            # 具身大脑入口(交互/单次/ReAct)
├── main.py                  # 主流程入口(单步/交互)
├── test_ik.py               # IK 运动学自测
├── test_sim2real.py         # Sim-to-Real 鲁棒性对比
├── test_smoke.py            # 综合回归自测
├── test_cases.py            # TC-01~15 验收用例
├── config/settings.py       # 配置(读环境变量)
├── assets/ur5e/             # UR5e 官方模型(meshes + XML)
├── src/
│   ├── agent/
│   │   ├── brain.py               # EmbodiedBrain: 大模型多步规划/ReAct
│   │   ├── executor.py            # BrainExecutor: 计划翻译 + IK 兜底
│   │   ├── instruction_parser.py  # DeepSeek 指令 → 结构化任务(单步)
│   │   └── vision.py              # 俯视渲染 + 物体定位 + CLIP
│   ├── hal/
│   │   ├── arm_interface.py       # HAL 抽象 + 仿真/真机后端
│   │   ├── pick_place.py          # 抓取放置动作原语
│   │   ├── rtde_server.py         # UR RTDE 协议模拟器
│   │   └── realistic.py           # 真机特性模拟 (Sim-to-Real)
│   └── sim/
│       └── ur5e_sim.py            # MuJoCo UR5e 仿真 + IK
└── docs/
    └── sim_to_real.md             # Sim-to-Real 设计文档
```

---

## 数据来源
- UR5e 模型: google-deepmind/mujoco_menagerie (universal_robots_ur5e)
- 仿真: MuJoCo 3.x
- LLM: DeepSeek API (deepseek-chat)
- 视觉: OpenCLIP ViT-B/32
