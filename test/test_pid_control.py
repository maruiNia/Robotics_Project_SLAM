import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가

sys.path.insert(0, str(Path(__file__).parent.parent))

from module.MyPlanning import smooth_path_nd, point_to_path_min_distance_nd
from module.MyRobot import Moblie_robot
from module.MyControl import SimpleBicycleModel

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 0) 로봇/모델 준비 ---
# robot은 이미 만들어져 있다고 가정 (Moblie_robot 인스턴스)
# 예: robot = Moblie_robot(dim=2, position=[0,0], velocity=[0,0], steering=[0,0], update_rate=50_000_000)
model = SimpleBicycleModel(wheelbase=1.2)
robot = Moblie_robot(dim=2, position=[0,0], velocity=[0,0], steering=[0,0], update_rate=50_000_000, dynamic_operation_mode=True)

# --- 1) 경로 생성 ---
def make_L_path():
    path = []
    for x in range(6):
        path.append([x, 0])
    for y in range(1, 6):
        path.append([5, y])
    return path

path = make_L_path()
s_path = smooth_path_nd(path, alpha=0.1, beta=0.3)

x_s = [p[0] for p in s_path]
y_s = [p[1] for p in s_path]

# --- 2) 플롯 초기화 ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect("equal", adjustable="box")
ax.grid(True)
ax.set_title("Path Tracking Animation (Bicycle Model + Queue Step)")
ax.set_xlabel("X")
ax.set_ylabel("Y")

(line_path,) = ax.plot(x_s, y_s, "r-", linewidth=2, label="Smoothed Path")
(robot_dot,) = ax.plot([], [], "bo", markersize=8, label="Robot")
(closest_dot,) = ax.plot([], [], "go", markersize=6, label="Closest point q")
(err_line,) = ax.plot([], [], "k--", linewidth=1.5, label="Error (p→q)")
text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")
ax.legend(loc="best")

min_x, max_x = min(x_s), max(x_s)
min_y, max_y = min(y_s), max(y_s)
pad = 1.0
ax.set_xlim(min_x - pad, max_x + pad)
ax.set_ylim(min_y - pad, max_y + pad)

# --- 3) 애니메이션 업데이트 함수 ---
def update(frame):
    # (A) 매 프레임: "명령을 큐에 쌓기"
    # 여기서는 테스트로:
    #   - 속도 v는 1.0 고정
    #   - 조향각 delta는 "최근접점 방향"으로 대충 따라가게 만든 예시
    p = robot.get_position()
    dmin, q, seg_i, t = point_to_path_min_distance_nd(p, s_path)

    # q 방향으로 헤딩을 맞추는 아주 단순한 steering(테스트용)
    import math
    dx, dy = q[0] - p[0], q[1] - p[1]
    desired_theta = math.atan2(dy, dx)

    # orientation이 (theta, ...)라고 가정
    theta = robot.get_orientation()[0]
    err_theta = (desired_theta - theta + math.pi) % (2 * math.pi) - math.pi

    v = 1.0
    delta = max(-0.6, min(0.6, 1.5 * err_theta))  # clamp

    # ✅ 큐에 쌓기: steering[0]을 "조향각(delta)"로 쓰는 모델이므로 set으로 넣음
    robot.ins_set_mov(velocity=[v, 0.0], steering=[delta, 0.0])

    # (B) 큐에서 "한 스텝만" 실행 (모델로 pose 갱신)
    robot.step_queue_with_model(model)  # dt_ns 기본=update_rate

    # (C) 실행 후 위치로 다시 q 계산해서 시각화
    p2 = robot.get_position()
    dmin2, q2, seg_i2, t2 = point_to_path_min_distance_nd(p2, s_path)

    robot_dot.set_data([p2[0]], [p2[1]])
    closest_dot.set_data([q2[0]], [q2[1]])
    err_line.set_data([p2[0], q2[0]], [p2[1], q2[1]])
    text.set_text(f"min dist = {dmin2:.3f}\nseg_i={seg_i2}, t={t2:.2f}\nv={v:.2f}, delta={delta:.2f}")

    return robot_dot, closest_dot, err_line, text

ani = FuncAnimation(fig, update, frames=300, interval=50, blit=True)
plt.show()
