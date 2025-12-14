import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가

sys.path.insert(0, str(Path(__file__).parent.parent))

from module.MyPlanning import smooth_path_nd, point_to_path_min_distance_nd
from module.MyRobot import Moblie_robot
from module.MyControl import SimpleBicycleModel
from module.MyPlanning import PIDVec

import math

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#--

def make_L_path():
    path = []
    for x in range(6):
        path.append([x, 0])
    for y in range(1, 6):
        path.append([5, y])
    return path

path = make_L_path()
s_path = smooth_path_nd(path, alpha=0.2, beta=0.2)

robot = Moblie_robot(
    dim=2,
    position=s_path[0],
    velocity=(0.0, 0.0),
    steering=(0.0, 0.0),
    dynamic_operation_mode=False,  # run_dynamic...가 켜줌
    sensing_mode=False,
    update_rate=50_000_000
)

robot.set_path(s_path)

model = SimpleBicycleModel(wheelbase=1.2)

log = robot.run_dynamic_path_follow(
    model,
    v_ref=1.0,
    goal_tolerance=0.3,
    max_steps=3000,
    steer_limit=3.0,
    use_heading_term=True,
    heading_gain=1.2
)

print("log steps =", len(log))
print("last =", log[-1])
