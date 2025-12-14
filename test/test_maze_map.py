import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가

sys.path.insert(0, str(Path(__file__).parent.parent))

from module.MySlam import SlamMap, Obstacle_obj, maze_map, print_map_plt, get_expanded_wall_points
import matplotlib.pyplot as plt

# ===== 테스트용 미로 맵 데이터 =====
maze = [
    [0,1,1,1,1,1],
    [0,1,0,0,0,1],
    [0,0,0,1,0,1],
]

m = maze_map(maze, (0,0), (4,2), cell_size=5)

print("미로 맵 정보:")
map_info = m.get_map()

for k, v in map_info.items():
    print(f"{k} : {v}")

# ===== matplotlib으로 시각화 =====
print_map_plt(m)

# ===== expansion 벽 좌표 생성 테스트 =====


expansion = 1  # 자연수 기준

expanded_walls = get_expanded_wall_points(m, expansion)

print(f"\n확장된 벽 좌표 개수 (expansion={expansion}): {len(expanded_walls)}")
print("일부 좌표 샘플:")
for p in expanded_walls[:20]:
    print(p)

# ===== 시각화: 기존 맵 + 확장 벽 =====
fig, ax = plt.subplots()
ax.set_aspect("equal")

# 기존 맵 그리기
# print_map_plt(m)

# 확장된 벽 좌표 scatter로 표시
xs = [p[0] for p in expanded_walls]
ys = [p[1] for p in expanded_walls]

ax.scatter(
    xs,
    ys,
    c="blue",
    s=20,
    alpha=0.6,
    label="Expanded Wall (A* blocked)"
)

ax.legend()
ax.set_title(f"Expanded Walls Visualization (expansion={expansion})")
plt.show()
