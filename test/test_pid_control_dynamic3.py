import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from module.MySlam import maze_map, print_map_plt
from module.MyRobot import Moblie_robot
from module.MyControl import SimpleBicycleModel

maze = [
    [0,1,1,1,1,1],
    [0,1,0,0,0,1],
    [0,0,0,1,0,1],
]

cell_size = 5
m = maze_map(maze, (0,0), (4,2), cell_size=cell_size)

# robot = Moblie_robot(dim=2, position=m.start_point, end_point=m.end_point, update_rate=10_000_000)

robot = Moblie_robot(
    dim=2,
    position=m.start_point,
    velocity=(0.0, 0.0),
    steering=(0.0, 0.0),
    dynamic_operation_mode=False,
    sensing_mode=False,
    update_rate=50_000_000,  # 50ms
)

robot.set_test_map_mode(True, m)

# blocked = robot._build_blocked_indices_from_map(m, resolution=cell_size, expansion=1)
# print("blocked count =", len(blocked))
# print("sample blocked =", list(blocked)[:10])

print("set map:", m)

model = SimpleBicycleModel(wheelbase=1.2)

# log = robot.run_replanning_astar_follow(
#     model,
#     resolution=float(cell_size),  # ✅ 중요
#     expansion=1,                  # 벽을 1셀 팽창(로봇 반경 느낌)
#     smooth=True,
#     v_ref=1.0,
#     dt_ns=10_000_000,
#     goal_tolerance=0.5,
#     max_steps=3000,
# )

log = robot.run_replanning_astar_follow(
    model,
    resolution=float(1),  # ⭐️ maze_map의 cell_size와 동일하게
    expansion=1,
    smooth=True,
    smooth_alpha=0.2,
    smooth_beta=0.2,
    v_ref=1.0,
    dt_ns=50_000_000,
    goal_tolerance=0.5,
    max_steps=3000,
    steer_limit=2.0,
    use_heading_term=True,
    heading_gain=0.6,
    look_ahead=5,
)

print("log steps =", len(log))
for step in log :
    print(step)

# 마지막에 한 번 맵+대충 최종 경로 확인하고 싶으면:
# (원하면 매 스텝마다 path를 저장하도록도 확장 가능)
print_map_plt(m, robot=robot)


