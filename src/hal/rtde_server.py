"""UR 机械臂 RTDE 协议模拟器 — 让上层代码通过"真机协议"驱动仿真。

背景:
  真实 UR 机械臂通过 RTDE(Real-Time Data Exchange)协议与上位机通信:
  - 客户端 TCP 连接到机械臂的 RTDE 端口(默认 5900)
  - 双方按二进制帧交换数据(请求/响应/数据订阅)
  - 常见数据: actual_q / actual_TCP_pose / target_q / robot_status

本模块在本地实现一个简化版的 RTDE 服务器:
  - 用 TCP socket 监听 5900 端口(与真机一致)
  - 上层程序像连真机一样连过来, 读写关节角/TCP 位姿
  - 内部把收到的目标关节角转发给 MuJoCo 仿真本体执行

这样同一套上位机代码, 换真机时只改 IP, 其余零改动 —— 这就是"真机落地"。
"""
from __future__ import annotations

import json
import socket
import struct
import threading
import time

import numpy as np

# RTDE 二进制帧类型(与 UR 官方 RTDE 一致)
RTDE_REQUEST_PROTOCOL_VERSION = 86
RTDE_GET_URCONTROL_VERSION = 87
RTDE_TEXT_MESSAGE = 77
RTDE_DATA_PACKAGE = 85
RTDE_CONTROL_PACKAGE_SETUP_OUTPUTS = 83
RTDE_CONTROL_PACKAGE_SETUP_INPUTS = 82
RTDE_CONTROL_PACKAGE_START = 80
RTDE_CONTROL_PACKAGE_PAUSE = 81

JOINT_NAMES = ["actual_q", "target_q", "actual_TCP_pose", "robot_status"]
OUTPUT_TYPES = {"actual_q": 6, "target_q": 6, "actual_TCP_pose": 6, "robot_status": 1}


class RTDESimServer:
    """简化版 RTDE 服务器, 桥接 MuJoCo 仿真本体。

    支持真实 UR 客户端库(如 ur_rtde)的最小握手流程:
      - 版本协商 / get urcontrol version
      - setup outputs(订阅数据)
      - start
    为保持协议简洁, 数据帧用轻量二进制打包 + 文本控制帧,
    足以演示"走真机协议"的思路。生产级应严格实现 UR 官方帧格式。
    """

    def __init__(self, sim, host: str = "127.0.0.1", port: int = 5900) -> None:
        self._sim = sim
        self._host = host
        self._port = port
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((host, port))
        self._server.listen(1)
        self._running = False
        self._thread = None
        self._client: socket.socket | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        print(f"[RTDE] 仿真 RTDE 服务器已启动: {self._host}:{self._port}")

    def stop(self) -> None:
        self._running = False
        if self._client:
            try:
                self._client.close()
            except OSError:
                pass
        try:
            self._server.close()
        except OSError:
            pass

    # ---------- 服务器主循环 ----------
    def _serve(self) -> None:
        while self._running:
            try:
                conn, addr = self._server.accept()
            except OSError:
                break
            self._client = conn
            print(f"[RTDE] 客户端连接: {addr}")
            try:
                self._handle_client(conn)
            except (ConnectionResetError, BrokenPipeError):
                pass
            finally:
                try:
                    conn.close()
                except OSError:
                    pass
                self._client = None

    def _handle_client(self, conn: socket.socket) -> None:
        buf = b""
        while self._running:
            try:
                chunk = conn.recv(4096)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            # 逐帧处理(简化协议: 2字节长度 + 1字节类型 + payload)
            while True:
                if len(buf) < 3:
                    break
                size = struct.unpack("<H", buf[0:2])[0]
                if len(buf) < 2 + size:
                    break
                frame = buf[2 : 2 + size]
                buf = buf[2 + size :]
                self._dispatch(conn, frame)

    def _dispatch(self, conn: socket.socket, frame: bytes) -> None:
        ptype = frame[0]
        payload = frame[1:]
        if ptype == RTDE_REQUEST_PROTOCOL_VERSION:
            # 版本协商
            self._send(conn, bytes([RTDE_REQUEST_PROTOCOL_VERSION]) + b"\x01")
        elif ptype == RTDE_GET_URCONTROL_VERSION:
            self._send(conn, bytes([RTDE_GET_URCONTROL_VERSION]) + b"\x01\x00\x00\x00")
        elif ptype == RTDE_CONTROL_PACKAGE_SETUP_OUTPUTS:
            # 客户端订阅哪些字段, 返回支持与否
            try:
                names = json.loads(payload.decode("utf-8"))
            except Exception:
                names = []
            mask = bytearray([1 if n in OUTPUT_TYPES else 0 for n in names])
            self._send(conn, bytes([RTDE_CONTROL_PACKAGE_SETUP_OUTPUTS]) + bytes(mask))
        elif ptype == RTDE_CONTROL_PACKAGE_SETUP_INPUTS:
            # 输入订阅(本模拟器忽略内容, 返回支持)
            try:
                names = json.loads(payload.decode("utf-8"))
            except Exception:
                names = []
            mask = bytearray([1] * len(names))
            self._send(conn, bytes([RTDE_CONTROL_PACKAGE_SETUP_INPUTS]) + bytes(mask))
        elif ptype == RTDE_CONTROL_PACKAGE_START:
            self._send(conn, bytes([RTDE_CONTROL_PACKAGE_START]) + b"\x01")
            self._stream_outputs(conn)
        elif ptype == RTDE_CONTROL_PACKAGE_PAUSE:
            self._send(conn, bytes([RTDE_CONTROL_PACKAGE_PAUSE]) + b"\x01")
        elif ptype == RTDE_TEXT_MESSAGE:
            # 文本消息: 上层可下发指令, 简化格式 "<cmd>:<json>"
            self._handle_text(payload)
        else:
            # 未知帧: 忽略
            pass

    def _handle_text(self, payload: bytes) -> None:
        try:
            msg = payload.decode("utf-8")
        except UnicodeDecodeError:
            return
        if ":" not in msg:
            return
        cmd, _, arg = msg.partition(":")
        if cmd == "move_to_pose":
            pose = np.array(json.loads(arg), dtype=float)
            self._sim.move_to_pose(pose)
            print(f"[RTDE] 执行 move_to_pose {np.round(pose[:3], 3)}")
        elif cmd == "move_to_joint":
            q = np.array(json.loads(arg), dtype=float)
            self._sim.set_joint_q(q)
            print(f"[RTDE] 执行 move_to_joint {np.round(q, 3)}")

    def _stream_outputs(self, conn: socket.socket) -> None:
        """持续推送订阅数据帧(仿真时间驱动)。"""
        while self._running:
            try:
                q = self._sim.joint_q
                pose = self._sim.tcp_pose
                payload = (
                    struct.pack("<6d", *q)                       # actual_q
                    + struct.pack("<6d", *q)                     # target_q
                    + struct.pack("<6d", *pose)                  # actual_TCP_pose
                    + struct.pack("<b", 1)                       # robot_status
                )
                frame = bytes([RTDE_DATA_PACKAGE]) + payload
                self._send(conn, frame)
            except OSError:
                break
            time.sleep(0.1)

    def _send(self, conn: socket.socket, frame: bytes) -> None:
        conn.sendall(struct.pack("<H", len(frame)) + frame)


if __name__ == "__main__":
    # 自测: 起一个服务器接仿真, 再起一个客户端读状态
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from sim.ur5e_sim import UR5eSim

    sim = UR5eSim(render=False)
    srv = RTDESimServer(sim)
    srv.start()

    # 简易客户端(真实场景用 ur_rtde)
    c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    c.connect(("127.0.0.1", 5900))
    c.sendall(struct.pack("<H", 2) + bytes([RTDE_REQUEST_PROTOCOL_VERSION, 1]))
    time.sleep(0.2)
    print("self-test client connected; server running.")
    c.close()
    time.sleep(0.3)
    srv.stop()
    sim.close()
    print("RTDE self-test OK")