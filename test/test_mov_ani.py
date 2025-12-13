import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))
from module.MyRobot import Moblie_robot

# ===============================
# 1. 로봇 생성
# ===============================
robot = Moblie_robot(
    dim=2,
    position=(0.0, 0.0),
    velocity=(0.0, 0.0),
    steering=(0.0, 0.0),
    update_rate=50_000_000  # 50ms = 0.05초
)

# ===============================
# 2. 명령 입력
# ===============================
# 총 5초 = 5,000,000,000 ns
total_time_ns = 5_000_000_000
half_time_ns = total_time_ns // 2

# (1) 2.5초 동안 +1 m/s까지 가속
robot.ins_acc_mov(
    velocity=(1.0, 0.0),  # x방향 속도 증가
    time=half_time_ns
)

# (2) 2.5초 동안 다시 0 m/s로 감속
robot.ins_acc_mov(
    velocity=(-1.0, 0.0),  # x방향 속도 감소
    time=half_time_ns
)

# ===============================
# 3. 시뮬레이션 실행
# ===============================
log = robot.start()

# ===============================
# 4. 로그에서 위치 추출
# ===============================
xs = [state["position"][0] for state in log]
ys = [state["position"][1] for state in log]
times = [state["t_ns"] * 1e-9 for state in log]  # ns → s

# ===============================
# 5. matplotlib 애니메이션
# ===============================
fig, ax = plt.subplots()
ax.set_title("Mobile Robot: Acceleration & Deceleration (5s)")
ax.set_xlabel("X position (m)")
ax.set_ylabel("Y position (m)")

ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
ax.set_ylim(-1.0, 1.0)
ax.grid(True)

robot_dot, = ax.plot([], [], "ro", markersize=8)
path_line, = ax.plot([], [], "b--", linewidth=1)

time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

def init():
    robot_dot.set_data([], [])
    path_line.set_data([], [])
    time_text.set_text("")
    return robot_dot, path_line, time_text

def update(frame):
    robot_dot.set_data(xs[frame], ys[frame])
    path_line.set_data(xs[:frame+1], ys[:frame+1])
    time_text.set_text(f"time = {times[frame]:.2f} s")
    return robot_dot, path_line, time_text

ani = FuncAnimation(
    fig,
    update,
    frames=len(xs),
    init_func=init,
    interval=50,   # ms
    blit=True
)

plt.show()
