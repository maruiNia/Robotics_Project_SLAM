import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import module.MySlam as MySlam
from module.MySlam import maze_map
from module.MySensor import Circle_Sensor
from module.MyControl import SimpleBicycleModel
from module.MyPlanning import PIDVec
from module.MyRobot_with_localization import Moblie_robot


def draw_obstacles(ax, slam_map: MySlam, *, alpha=0.20):
    for obs in slam_map.get_obs_list():
        (min_x, min_y), (max_x, max_y) = obs.bounds()
        ax.add_patch(Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, alpha=alpha))


def grid_center_xy(cell_xy, cell_size: float):
    gx, gy = cell_xy
    return ((gx + 0.5) * cell_size, (gy + 0.5) * cell_size)


def main():
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
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # 9
    ]

    start = (1, 8)
    g1 = (11, 9)
    g2 = (11, 1)

    # ✅ 랜드마크 여러 개 (grid 좌표만 계속 추가하면 됨)
    landmarks_grid = [
        (2, 6),
        (3, 4),
        (7, 2),
    ]

    cell_size = 5
    m = maze_map(maze, start, g1, cell_size=cell_size)

    # ✅ SlamMap에 landmarks 딕셔너리를 “런타임으로” 붙여도 됨(파이썬이라 가능)
    m.landmarks = {}
    for i_lm, lm in enumerate(landmarks_grid, start=1):
        m.landmarks[i_lm] = grid_center_xy(lm, cell_size)

    v_ref = 2
    sensor = Circle_Sensor(dim=2, number=10, distance=12.0, slam_map=m, step=0.1)

    robot = Moblie_robot(
        dim=2,
        position=m.start_point,
        end_point=m.end_point,
        update_rate=50_000_000,
        sensing_mode=True,
        sensor=sensor,
    )

    robot.enable_estimated_pose(True)
    robot.set_localization_params(resolution=0.2, sigma=0.01, period=1)
    robot.set_pid(PIDVec(dim=2, kp=1.5, ki=0.1, kd=0.3), v_ref=v_ref)
    robot.set_fake_fast_sensing(True, sigma=0.5)

    # ✅ GraphSLAM(SE2+landmark) ON
    # robot.enable_ekf(True)
    
    # robot.enable_pgslam(True)
    
    robot.enable_gslam(True)

    model = SimpleBicycleModel(wheelbase=1.2)

    print("Running replanning with A* and SLAM...")
    log = robot.run_replanning_astar_follow(
        model,
        expansion=1,
        smooth=True,
        smooth_alpha=0.6,
        smooth_beta=0.3,
        v_ref=v_ref,
        dt_ns=50_000_000,
        goal_tolerance=0.5,
        max_steps=20000,
        steer_limit=1.2,
        use_heading_term=True,
        heading_gain=0.6,
        look_ahead=6,
    )

    if not log:
        print("log empty")
        return
    
    for i in log :
        print(i)

    pos = [st["position"] for st in log]
    est = [st.get("est_position", None) for st in log]
    lms_est = [st.get("slam_landmarks", {}) for st in log]
    pathing = [st.get("pathing", None) for st in log]


    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_title("GraphSLAM (SE2) : True vs Estimated Landmarks")

    if m.limit is not None:
        (x0, y0), (x1, y1) = m.limit
        ax.set_xlim(x0 - 1, x1 + 1)
        ax.set_ylim(y0 - 1, y1 + 1)

    draw_obstacles(ax, m, alpha=0.25)

    # start / goals
    sx, sy = m.start_point
    g1x, g1y = grid_center_xy(g1, cell_size)
    g2x, g2y = grid_center_xy(g2, cell_size)
    ax.plot([sx], [sy], "bs", markersize=8, label="Start")
    ax.plot([g1x], [g1y], "g^", markersize=8, label="Goal g1")
    ax.plot([g2x], [g2y], "c^", markersize=8, label="Goal g2")

    # true landmarks (yellow diamond)
    true_lm_xy = np.array(list(m.landmarks.values()), dtype=float)
    ax.scatter(true_lm_xy[:,0], true_lm_xy[:,1], marker="D", s=70, label="LM true (map)")

    # estimated landmarks (pink diamond)
    est_lm_sc = ax.scatter([], [], marker="D", s=70, label="LM estimated")

    true_trace, = ax.plot([], [], "b-", linewidth=2, label="true trace")
    est_trace,  = ax.plot([], [], "m-", linewidth=2, label="est trace")
    true_pt, = ax.plot([], [], "bo", markersize=6, label="true pos")
    est_pt,  = ax.plot([], [], "mo", markersize=6, label="est pos")
    # path (현재 tick 기준)
    path_line, = ax.plot([], [], "r-", linewidth=2, label="path")


    ax.legend(loc="upper left")

    def init():
        true_trace.set_data([], [])
        est_trace.set_data([], [])
        true_pt.set_data([], [])
        est_pt.set_data([], [])
        path_line.set_data([], [])
        est_lm_sc.set_offsets(np.empty((0, 2)))
        return [path_line, true_trace, est_trace, true_pt, est_pt, est_lm_sc]

    def update(i):
        x, y = pos[i][0], pos[i][1]
        ex, ey = (est[i][0], est[i][1]) if est[i] is not None else (x, y)

        # path
        pth = pathing[i]
        if pth is not None and len(pth) >= 2:
            px = [p[0] for p in pth]
            py = [p[1] for p in pth]
            path_line.set_data(px, py)
        else:
            path_line.set_data([], [])

        # traces
        true_trace.set_data([p[0] for p in pos[:i+1]], [p[1] for p in pos[:i+1]])
        est_trace.set_data(
            [p[0] for p in est[:i+1] if p is not None],
            [p[1] for p in est[:i+1] if p is not None],
        )
        true_pt.set_data([x], [y])
        est_pt.set_data([ex], [ey])

        # landmarks
        lms = lms_est[i]
        if isinstance(lms, dict) and len(lms) > 0:
            xy = np.array(list(lms.values()), dtype=float)
            est_lm_sc.set_offsets(xy)
        else:
            est_lm_sc.set_offsets(np.empty((0, 2)))

        return [path_line, true_trace, est_trace, true_pt, est_pt, est_lm_sc]

    ani = FuncAnimation(fig, update, frames=len(log), init_func=init, interval=50, blit=False, repeat=False)

    plt.show()


if __name__ == "__main__":
    main()
