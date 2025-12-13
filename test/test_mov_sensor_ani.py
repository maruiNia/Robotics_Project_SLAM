import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from module.MySlam import SlamMap, Obstacle_obj
from module.MyRobot import Moblie_robot
from module.MySensor import Circle_Sensor

# ===============================
# 1. 맵 생성 + 장애물
# ===============================
m = SlamMap(dim=2, limit=((0, 0), (10, 10)))

m.add(Obstacle_obj(dim=2, min_corner=(2.0, 2.0), max_corner=(4.5, 2.8)))
m.add(Obstacle_obj(dim=2, min_corner=(6.0, 1.5), size=(1.2, 5.5)))
m.add(Obstacle_obj(dim=2, min_corner=(3.2, 6.2), size=(4.8, 1.0)))
m.add(Obstacle_obj(dim=2, min_corner=(1.2, 7.6), size=(1.2, 1.8)))

# ===============================
# 2. 로봇 생성
# ===============================
robot = Moblie_robot(
    dim=2,
    position=(1.0, 5.0),
    velocity=(0.0, 0.0),
    steering=(0.0, 0.0),
    update_rate=50_000_000  # 50ms = 0.05초
)

# ===============================
# 3. 명령 입력
# ===============================
# 총 5초 = 5,000,000,000 ns
total_time_ns = 5_000_000_000
half_time_ns = total_time_ns // 2

# (1) 2.5초 동안 +1 m/s까지 가속 (x방향)
robot.ins_acc_mov(
    velocity=(1.0, 0.0),
    time=half_time_ns
)

# (2) 2.5초 동안 다시 0 m/s로 감속
robot.ins_acc_mov(
    velocity=(-1.0, 0.0),
    time=half_time_ns
)

# ===============================
# 4. 시뮬레이션 실행
# ===============================
log = robot.start()

# ===============================
# 5. 로그에서 위치 추출
# ===============================
xs = [state["position"][0] for state in log]
ys = [state["position"][1] for state in log]
times = [state["t_ns"] * 1e-9 for state in log]  # ns → s

# ===============================
# 6. 각 시간 단계에서 센싱 수행
# ===============================
sensor = Circle_Sensor(dim=2, number=48, distance=4.0, slam_map=m, step=0.02)
all_readings = []

for pos in zip(xs, ys):
    readings = sensor.sensing(pos)
    all_readings.append(readings)

# ===============================
# 7. matplotlib 애니메이션
# ===============================
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title("Mobile Robot with Circle Sensor")
ax.set_aspect("equal")
ax.grid(True)

# 맵 경계
(x0, y0), (x1, y1) = m.limit
ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='black', linewidth=2))

# 장애물
for obs in m.get_obs_list():
    (ox0, oy0), (ox1, oy1) = obs.min_corner, obs.max_corner
    ax.add_patch(Rectangle((ox0, oy0), ox1 - ox0, oy1 - oy0, fill=True, facecolor='gray', alpha=0.5))

ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-0.5, 10.5)

# 애니메이션 요소
robot_dot, = ax.plot([], [], "ro", markersize=8, label="Robot")
path_line, = ax.plot([], [], "b--", linewidth=1, alpha=0.5, label="Path")

# 센서 레이 저장 (리스트)
ray_lines = []
hit_points_scatter = None

time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10)
ax.legend(loc='upper left')

def init():
    robot_dot.set_data([], [])
    path_line.set_data([], [])
    time_text.set_text("")
    return robot_dot, path_line, time_text

def update(frame):
    global hit_points_scatter
    
    # 로봇 위치 업데이트
    robot_dot.set_data(xs[frame], ys[frame])
    path_line.set_data(xs[:frame+1], ys[:frame+1])
    time_text.set_text(f"time = {times[frame]:.2f} s")
    
    # 이전 센서 레이 제거
    for line in ray_lines:
        line.remove()
    ray_lines.clear()
    
    if hit_points_scatter:
        hit_points_scatter.remove()
    
    # 현재 프레임의 센싱 결과 표시
    readings = all_readings[frame]
    sensor_pos = (xs[frame], ys[frame])
    N = len(readings)
    
    hit_x, hit_y = [], []
    
    for i, dist in enumerate(readings):
        angle = 2 * math.pi * i / N
        d = dist if dist is not None else sensor.get_distance()
        
        ex = sensor_pos[0] + math.cos(angle) * d
        ey = sensor_pos[1] + math.sin(angle) * d
        
        # 레이 그리기
        line, = ax.plot(
            [sensor_pos[0], ex],
            [sensor_pos[1], ey],
            color='tab:blue',
            alpha=0.2,
            linewidth=0.5
        )
        ray_lines.append(line)
        
        # 장애물에 닿았으면 저장
        if dist is not None:
            hit_x.append(ex)
            hit_y.append(ey)
    
    # 히트 지점 X 표시
    if hit_x:
        hit_points_scatter = ax.scatter(hit_x, hit_y, marker="x", color="red", s=30)
    
    return robot_dot, path_line, time_text, *ray_lines

ani = FuncAnimation(
    fig,
    update,
    frames=len(xs),
    init_func=init,
    interval=50,   # ms
    blit=False
)

plt.show()
