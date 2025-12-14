import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가

sys.path.insert(0, str(Path(__file__).parent.parent))

from module.MySlam import SlamMap, Obstacle_obj, maze_map, print_map_plt

# ===== 테스트용 미로 맵 데이터 =====
maze = [
    [0,1,1,1,1,1],
    [0,1,0,0,0,1],
    [0,0,0,1,0,1],
]

m = maze_map(maze, (0,0), (4,2), cell_size=1)

print("미로 맵 정보:")
map_info = m.get_map()

for k, v in map_info.items():
    print(f"{k} : {v}")

# ===== matplotlib으로 시각화 =====
print_map_plt(m)