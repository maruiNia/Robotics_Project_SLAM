import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from module.MySlam import maze_map, print_map_plt

# 네가 만든 함수/ A*가 어디 있는지에 따라 import 경로는 조정해줘야 할 수 있어요.
# - get_expanded_wall_points: 너가 작성한 곳 (예: module.MyPlanning)
# - astar_find_path_center_nodes: A* 함수가 있는 곳

from module.MySlam import get_expanded_wall_points
from module.MyNavi import astar_find_path_center_nodes  # 네가 A*를 여기에 넣었다고 가정
from module.MyPlanning import smooth_path_nd
import math

def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ===== 테스트용 미로 맵 데이터 =====
maze = [
    [0,1,1,1,1,1],
    [0,1,0,0,0,1],
    [0,0,0,1,0,1],
]

m = maze_map(maze, (0,0), (4,2), cell_size=5)

# ===== A* 경로 생성 =====
expanded = get_expanded_wall_points(m, expansion=1)

start = m.start_point
goal  = m.end_point

# ✅ 중앙 좌표 간격은 cell_size=5일 필요 없음!
# 여기선 예시로 resolution=1.0 (더 촘촘한 탐색)
path = astar_find_path_center_nodes(
    expanded,
    heuristic_fn=euclidean,
    current_pos=start,
    goal_pos=goal,
    dim=2,
    resolution=1.0,   # ← 여기 바꾸면 “길이 넓다/좁다”가 조절됨
    center_offset=0.5
)

s_path = smooth_path_nd(path, alpha=0.3, beta=0.3, max_iter=500, tol=1e-4, fix_ends=True)

print("path length:", len(s_path))
print("path preview:", s_path[:10])

# ===== 이미지 1: 원본 맵 위에 path =====
# print_map_plt(m, s_path)
# ===== 이미지 2: 확장벽(Inflation) 표시한 맵 위에 path =====
print_map_plt(m, s_path, expanded_walls=expanded)
