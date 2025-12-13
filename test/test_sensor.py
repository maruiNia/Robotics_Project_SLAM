import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import sys
from pathlib import Path
# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from module.MySlam import SlamMap, Obstacle_obj
from module.MySensor import Circle_Sensor

# ----------------------------
# 1) 맵 생성 + 장애물
# ----------------------------
m = SlamMap(dim=2, limit=((0, 0), (10, 10)))

m.add(Obstacle_obj(dim=2, min_corner=(2.0, 2.0), max_corner=(4.5, 2.8)))
m.add(Obstacle_obj(dim=2, min_corner=(6.0, 1.5), size=(1.2, 5.5)))
m.add(Obstacle_obj(dim=2, min_corner=(3.2, 6.2), size=(4.8, 1.0)))
m.add(Obstacle_obj(dim=2, min_corner=(1.2, 7.6), size=(1.2, 1.8)))

# ----------------------------
# 2) 센서 생성 + 센싱
# ----------------------------
sensor_pos = (5.0, 5.0)
sensor = Circle_Sensor(dim=2, number=48, distance=4.0, slam_map=m, step=0.02)
readings = sensor.sensing(sensor_pos)

# ----------------------------
# 3) matplotlib 시각화
# ----------------------------
fig, ax = plt.subplots()
ax.set_title("Circle Sensor Sensing")
ax.set_aspect("equal")
ax.grid(True)

# 맵 경계
(x0, y0), (x1, y1) = m.limit
ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False))

# 장애물
for obs in m.get_obs_list():
    (ox0, oy0), (ox1, oy1) = obs.min_corner, obs.max_corner
    ax.add_patch(Rectangle((ox0, oy0), ox1 - ox0, oy1 - oy0, fill=False))

# 센서 위치
ax.plot(sensor_pos[0], sensor_pos[1], "ko", markersize=5)

# 스타일 설정
ray_color = "tab:blue"
ray_alpha = 0.3

hit_x, hit_y = [], []

N = len(readings)
for i, dist in enumerate(readings):
    angle = 2 * math.pi * i / N
    d = dist if dist is not None else sensor.get_distance()

    ex = sensor_pos[0] + math.cos(angle) * d
    ey = sensor_pos[1] + math.sin(angle) * d

    # 레이 (연한 단색)
    ax.plot(
        [sensor_pos[0], ex],
        [sensor_pos[1], ey],
        color=ray_color,
        alpha=ray_alpha,
        linewidth=1
    )

    # 끝점 작은 점
    ax.plot(ex, ey, ".", color=ray_color, alpha=ray_alpha)

    # 장애물에 닿았으면 X
    if dist is not None:
        hit_x.append(ex)
        hit_y.append(ey)

# 히트 지점 X 표시
ax.scatter(hit_x, hit_y, marker="x", color="red", s=40)

ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-0.5, 10.5)

plt.show()
