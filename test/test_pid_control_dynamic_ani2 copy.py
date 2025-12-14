import sys
from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from module.MyPlanning import smooth_path_nd, point_to_path_min_distance_nd
from module.MyRobot import Moblie_robot
from module.MyControl import SimpleBicycleModel
from module.MyPlanning import PIDVec


# --- 1) 경로 생성 ---
def make_L_path():
    path = []
    for x in range(6):
        path.append([x, 0])
    for y in range(1, 6):
        path.append([5, y])
    return path


# --- 2) 로그 생성(로봇에게 일임) ---
path = make_L_path()
s_path = smooth_path_nd(path, alpha=0.2, beta=0.2)

robot = Moblie_robot(
    dim=2,
    position=s_path[0],
    velocity=(0.0, 0.0),
    steering=(0.0, 0.0),
    dynamic_operation_mode=False,
    sensing_mode=False,
    update_rate=50_000_000
)

robot.set_path(s_path)

# PID 설정(로봇 내부에서 쓰도록)
pid = PIDVec(dim=2, kp=3, ki=0.1, kd = 1.0, i_limit=1.0)
robot.set_pid(pid, v_ref=1.0)

model = SimpleBicycleModel(wheelbase=1.2)

log = robot.run_dynamic_path_follow(
    model,
    v_ref=1.0,
    goal_tolerance=0.3,
    max_steps=3000,
    steer_limit=2,
    use_heading_term=True,
    heading_gain=0.6
)

print("log steps =", len(log))
print("last =", log[-1])


# --- 3) log만으로 애니메이션 (시각화용 계산은 여기서만) ---
# (log에서 position/orientation만 꺼내서 그림)
xs = [st["position"][0] for st in log]
ys = [st["position"][1] for st in log]
thetas = [st["orientation"][0] for st in log]

# 원본 경로(예쁘게 같이 표시)
x_orig = [p[0] for p in path]
y_orig = [p[1] for p in path]
x_s = [p[0] for p in s_path]
y_s = [p[1] for p in s_path]

# Figure
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_aspect("equal", adjustable="box")
ax.grid(True)
ax.set_title("Log-based Animation (Robot did all control)")

# 전체 경로가 항상 보이도록 범위 설정
all_x = x_orig + x_s + xs
all_y = y_orig + y_s + ys
margin = 1.0
ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

# 정적 요소: 전체 경로 표시
ax.plot(x_orig, y_orig, "o--", linewidth=2, label="Original Path (L-shape)")
ax.plot(x_s, y_s, "r-", linewidth=2, label="Smoothed Path")

# 동적 요소
trace_line, = ax.plot([], [], "b-", linewidth=1.5, label="Robot Trace")
robot_point, = ax.plot([], [], "go", markersize=8, label="Current Position")

closest_point, = ax.plot([], [], "ro", markersize=8, label="Closest Point on Path")
min_line, = ax.plot([], [], "k--", linewidth=1.2, label="Min Distance")

heading = ax.quiver(
    0, 0, 0, 0,
    angles="xy",
    scale_units="xy",
    scale=1,
    color="black",
    width=0.006
)

info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

ax.legend(loc="upper left")


def init():
    trace_line.set_data([], [])
    robot_point.set_data(xs[0], ys[0])

    dmin, closest, seg_i, t = point_to_path_min_distance_nd([xs[0], ys[0]], s_path)
    closest_point.set_data(closest[0], closest[1])
    min_line.set_data([xs[0], closest[0]], [ys[0], closest[1]])

    heading.set_offsets([xs[0], ys[0]])
    heading.set_UVC(0.0, 0.0)

    info_text.set_text("")
    return trace_line, robot_point, closest_point, min_line, heading, info_text


def update(frame):
    x = xs[frame]
    y = ys[frame]
    theta = thetas[frame]

    # trace
    trace_line.set_data(xs[:frame+1], ys[:frame+1])

    # current position
    robot_point.set_data(x, y)

    # closest point (시각화용으로만 계산)
    dmin, closest, seg_i, t = point_to_path_min_distance_nd([x, y], s_path)
    closest_point.set_data(closest[0], closest[1])
    min_line.set_data([x, closest[0]], [y, closest[1]])

    # heading
    arrow_len = 0.35
    dx = arrow_len * math.cos(theta)
    dy = arrow_len * math.sin(theta)
    heading.set_offsets([x, y])
    heading.set_UVC(dx, dy)

    info_text.set_text(
        f"frame={frame}/{len(xs)-1}\n"
        f"min_dist={dmin:.3f}\n"
        f"seg={seg_i}, t={t:.3f}"
    )

    return trace_line, robot_point, closest_point, min_line, heading, info_text


ani = FuncAnimation(
    fig,
    update,
    frames=len(xs),
    init_func=init,
    interval=50,
    blit=False
)

plt.show()
