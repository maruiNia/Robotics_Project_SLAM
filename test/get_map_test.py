import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import sys
from pathlib import Path
# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from module.MySlam import SlamMap, Obstacle_obj

# ===== 맵 생성 (제한 없는 맵) =====
slam_map = SlamMap(dim=2)

# ===== 장애물 2개 생성 =====
# 장애물 1 : (1, 1) ~ (4, 3)
obs1 = Obstacle_obj(
    dim=2,
    min_corner=(1, 1),
    max_corner=(4, 3)
)

# 장애물 2 : (5, 2) 에서 size (2, 4) → (5,2) ~ (7,6)
obs2 = Obstacle_obj(
    dim=2,
    min_corner=(5, 2),
    size=(2, 4)
)

slam_map.add(obs1)
slam_map.add(obs2)

# ===== get_map() 결과 =====
map_info = slam_map.get_map()
print("get_map() 결과:")
for k, v in map_info.items():
    print(f"{k} : {v}")

# ===== matplotlib으로 시각화 =====
fig, ax = plt.subplots()

for obs in slam_map.get_obs_list():
    (x_min, y_min), (x_max, y_max) = obs.bounds()
    width = x_max - x_min
    height = y_max - y_min

    rect = Rectangle(
        (x_min, y_min),
        width,
        height,
        edgecolor='black',
        facecolor='gray',
        alpha=0.6
    )
    ax.add_patch(rect)

# 보기 좋게 범위 자동 설정
bounds = map_info["obstacles_bounds"]
if bounds:
    (xmin, ymin), (xmax, ymax) = bounds
    margin = 1
    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(ymin - margin, ymax + margin)

ax.set_aspect('equal')
ax.set_title("SLAM Map (Unlimited, Human-view)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.grid(True)

plt.show()
