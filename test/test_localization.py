import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from module.MySlam import maze_map, print_map_plt
from module.MyLocalization import localize_by_scan_grid
from module.MySensor import Circle_Sensor, localize_grid_by_circle_scan
import math


maze = [
    [0,1,1,1,1,1],
    [0,1,0,0,0,1],
    [0,0,0,1,0,1],
]
m = maze_map(maze, (0,0), (4,2), cell_size=3)

sensor = Circle_Sensor(dim=2, number=12, distance=7, slam_map=m, step=0.1)
true_pose = (0.8, 1.2)

z = sensor.sensing(true_pose)   # [d0, d1, d2] : 0°,120°,240° 방향 거리들

best_xy, prob, meta = localize_grid_by_circle_scan(m, sensor, z, resolution=0.25)
print("추정 위치:", best_xy)
