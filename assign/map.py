import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import module.MySlam as MySlam
from module.MySlam import maze_map
from module.MySensor import Circle_Sensor
from module.MyControl import SimpleBicycleModel
from module.MyPlanning import PIDVec
from module.MyRobot_with_localization import Moblie_robot
import numpy as np


def draw_obstacles(ax, slam_map : MySlam, *, alpha=0.20):
    for obs in slam_map.get_obs_list():
        (min_x, min_y), (max_x, max_y) = obs.bounds()
        ax.add_patch(Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, alpha=alpha))


def main():
    # -------------------------
    # 1) 맵/센서/로봇 세팅
    # -------------------------
    print("맵/센서/로봇 세팅 중...")
    # maze = [
    #     [0, 1, 1, 1, 1, 1],
    #     [0, 1, 0, 0, 0, 1],
    #     [0, 0, 0, 1, 0, 1],
    # ]
    # cell_size = 3
    # m = maze_map(maze, (0, 0), (4, 2), cell_size=cell_size)
    # vel = 1.0

    #과제 맵
    maze = [ ## 0 1  2 3  4  5 6  7  8 9 10 11 12 
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # 0
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # 1
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # 2
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], # 3
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 4
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0], # 5
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0], # 6
        [0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0], # 7
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # 8
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 9
    ]
    start = (0, 9)
    end_g1 = (11, 9)
    end_g2 = (11, 1)

    land_1 = (2, 6)
    land_2 = (3, 4)
    land_3 = (7, 2)

    cell_size = 5
    m = maze_map(maze, start, end_g2, cell_size=cell_size)
    
    # -------------------------
    # 맵 시각화
    # -------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 맵의 장애물 그리기
    draw_obstacles(ax, m, alpha=0.5)
    
    # 시작점 표시
    ax.plot(start[0] * cell_size, start[1] * cell_size, 'go', markersize=10, label='Start')
    
    # 끝점 표시
    ax.plot(end_g2[0] * cell_size, end_g2[1] * cell_size, 'r*', markersize=15, label='Goal')
    
    # 랜드마크 표시
    ax.plot(land_1[0] * cell_size, land_1[1] * cell_size, 'bs', markersize=8, label='Landmark 1')
    ax.plot(land_2[0] * cell_size, land_2[1] * cell_size, 'bs', markersize=8, label='Landmark 2')
    ax.plot(land_3[0] * cell_size, land_3[1] * cell_size, 'bs', markersize=8, label='Landmark 3')
    
    # 축 설정
    ax.set_xlim(-5, (len(maze[0]) + 1) * cell_size)
    ax.set_ylim(-5, (len(maze) + 1) * cell_size)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Maze Map')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('maze_map.png', dpi=150, bbox_inches='tight')
    print("맵이 'maze_map.png'로 저장되었습니다.")
    plt.show()

if __name__ == "__main__":
    main()