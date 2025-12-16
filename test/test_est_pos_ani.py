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
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # 9
    ]
    start = (0, 9)
    end_g1 = (11, 9)
    end_g2 = (11, 1)

    land_1 = (2, 6)
    land_2 = (3, 4)
    land_3 = (7, 2)

    cell_size = 3
    m = maze_map(maze, start, end_g1, cell_size=cell_size)
    vel = 2

    sensor = Circle_Sensor(dim=2, number=10, distance=12.0, slam_map=m, step=0.1)

    robot = Moblie_robot(
        dim=2,
        position=m.start_point,
        end_point=m.end_point,
        update_rate=50_000_000,     # 50ms
        sensing_mode=True,
        sensor=sensor,
    )

    robot.enable_estimated_pose(True)
    robot.set_localization_params(resolution=0.2, sigma=0.5, period=1)
    pid = PIDVec(dim=2, kp=1.5, ki=0.1, kd=0.3)
    robot.set_pid(pid, v_ref=vel)

    robot.set_fake_fast_sensing(True, sigma=0.5)
    # robot.enable_ekf(True)
    robot.enable_ekf(False)
    robot.enable_pgslam(True)


    model = SimpleBicycleModel(wheelbase=1.2)

    # -------------------------
    # 2) 로봇이 모든 연산을 끝낸 뒤 log 확보 (애니는 log만 사용)
    # -------------------------
    print("로봇 실행 및 로그 확보 중...")
    log = robot.run_replanning_astar_follow(
        model,
        expansion=1,
        smooth=True,
        smooth_alpha=0.6,
        smooth_beta=0.3,
        v_ref=vel,
        dt_ns=50_000_000,
        goal_tolerance=0.5,
        max_steps=2000,
        steer_limit=1.2,
        use_heading_term=True,
        heading_gain=0.6,
        look_ahead= 6,
    )

    if not log:
        print("log가 비어있습니다.")
        return
    
    for i in range(10):
        print(f"  log[{i}]: {log[i]}")
        print('-----------\n')
    # else:
        # print(f"log에 {len(log)} 스냅샷이 기록되었습니다.")
        # for i in range(10):
        #     print(f"  log[{i}]: {log[i]}")

    # -------------------------
    # 3) log 파싱 (오직 snapshot 값만!)
    # -------------------------
    t_ns = [st["t_ns"] for st in log]
    pos = [st["position"] for st in log]
    est = [st.get("est_position", None) for st in log]
    vel = [st["velocity"] for st in log]
    steer = [st["steering"] for st in log]
    sensing = [st.get("sensing", None) for st in log]
    pathing = [st.get("pathing", None) for st in log]
    note = [st.get("note", "") for st in log]

    # -------------------------
    # 4) matplotlib 준비
    # -------------------------
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_title("Replanning A* Follow (log-only animation)")

    if m.limit is not None:
        (x0, y0), (x1, y1) = m.limit
        ax.set_xlim(x0 - 1, x1 + 1)
        ax.set_ylim(y0 - 1, y1 + 1)

    draw_obstacles(ax, m, alpha=0.25)

    # start/goal
    sx, sy = m.start_point
    gx, gy = m.end_point
    ax.plot([sx], [sy], "bs", markersize=8, label="Start")
    ax.plot([gx], [gy], "g^", markersize=8, label="Goal")

    # path(현재 tick)
    path_line, = ax.plot([], [], "r-", linewidth=2, label="pathing (from log)")

    # true/est trace
    true_trace, = ax.plot([], [], "b-", linewidth=2, label="true trace")
    est_trace,  = ax.plot([], [], "m-", linewidth=2, label="est trace")

    # true/est point
    true_pt, = ax.plot([], [], "bo", markersize=6, label="true pos")
    est_pt,  = ax.plot([], [], "mo", markersize=6, label="est pos")

    # velocity arrow (quiver)
    vel_q = ax.quiver([pos[0][0]], [pos[0][1]], [0], [0], angles="xy", scale_units="xy", scale=1)

    # steering/heading arrow: 여기서는 "orientation 로그"가 있으면 그걸로,
    # 없으면 steering[0] 적분한 추정 heading을 내부에서 만들어도 되지만,
    # 요청이 "log only"라서 orientation이 있으면 사용하고 없으면 화살표 생략.
    has_ori = ("orientation" in log[0])
    ori = [st.get("orientation", (0.0, 0.0)) for st in log] if has_ori else None
    head_q = ax.quiver([pos[0][0]], [pos[0][1]], [0], [0], angles="xy", scale_units="xy", scale=1)

    # ---- 센싱 시각화(2번째 방식: 레이 + 끝점 점 + hit이면 X) ----
    N = sensor.get_number()
    max_range = sensor.get_distance()

    # 레이 라인 N개
    ray_lines = [ax.plot([], [], alpha=0.25, linewidth=1)[0] for _ in range(N)]
    # 끝점 점 N개
    end_pts = [ax.plot([], [], ".", alpha=0.25)[0] for _ in range(N)]
    # hit X는 한 번에 scatter로 갱신
    hit_sc = ax.scatter([], [], marker="x", s=40)

    # 텍스트 HUD (실시간 상태 표시)
    hud = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10)

    ax.legend(loc="upper left")

    def init():
        path_line.set_data([], [])
        true_trace.set_data([], [])
        est_trace.set_data([], [])
        true_pt.set_data([], [])
        est_pt.set_data([], [])
        hud.set_text("")

        for ln in ray_lines:
            ln.set_data([], [])
        for pt in end_pts:
            pt.set_data([], [])
        hit_sc.set_offsets([[0.0, 0.0]])   # 임시 1점
        hit_sc.set_alpha(0.0)              # 안 보이게

        vel_q.set_offsets([pos[0][0], pos[0][1]])
        vel_q.set_UVC(0.0, 0.0)

        head_q.set_offsets([pos[0][0], pos[0][1]])
        head_q.set_UVC(0.0, 0.0)

        return [path_line, true_trace, est_trace, true_pt, est_pt, vel_q, head_q, hit_sc, hud] + ray_lines + end_pts

    def update(i):
        # --- 현재 상태 ---
        x, y = pos[i][0], pos[i][1]
        ex, ey = (est[i][0], est[i][1]) if est[i] is not None else (x, y)

        vx, vy = vel[i][0], vel[i][1]
        s0 = steer[i][0] if steer[i] is not None else 0.0

        # --- trace / point ---
        true_trace.set_data([p[0] for p in pos[:i+1]], [p[1] for p in pos[:i+1]])
        est_trace.set_data([p[0] for p in est[:i+1] if p is not None],
                           [p[1] for p in est[:i+1] if p is not None])

        true_pt.set_data([x], [y])
        est_pt.set_data([ex], [ey])

        # --- pathing (log에 있는 값 그대로) ---
        pth = pathing[i]
        if pth is not None and len(pth) >= 2:
            px = [p[0] for p in pth]
            py = [p[1] for p in pth]
            path_line.set_data(px, py)
        else:
            path_line.set_data([], [])

        # --- velocity arrow ---
        vel_q.set_offsets([x, y])
        vel_q.set_UVC(vx, vy)

        # --- heading arrow (가능하면 orientation 로그 기반) ---
        head_q.set_offsets([x, y])
        if has_ori:
            th = ori[i][0]
            L = 1.0
            head_q.set_UVC(L * math.cos(th), L * math.sin(th))
        else:
            head_q.set_UVC(0.0, 0.0)

        # --- sensing rays (2번째 방식 그대로) ---
        z = sensing[i]
        hit_xy = []

        if z is None:
            # 센싱 없으면 모두 비움
            for ln in ray_lines:
                ln.set_data([], [])
            for pt in end_pts:
                pt.set_data([], [])
            hit_sc.set_offsets(np.empty((0, 2)))
        else:
            for k in range(N):
                angle = 2 * math.pi * k / N
                dist = z[k] if k < len(z) else None
                d = dist if dist is not None else max_range

                ex2 = x + math.cos(angle) * d
                ey2 = y + math.sin(angle) * d

                # 레이
                ray_lines[k].set_data([x, ex2], [y, ey2])
                # 끝점 점
                end_pts[k].set_data([ex2], [ey2])

                # hit이면 X 좌표로 모음
                if dist is not None:
                    hit_xy.append([ex2, ey2])

            hit_sc.set_offsets(hit_xy)

        # --- HUD 텍스트(실시간 전체 표시) ---
        hud.set_text(
            f"step={i}/{len(log)-1}\n"
            f"note={note[i]}\n"
            f"t_ns={t_ns[i]}\n"
            f"true=({x:.2f},{y:.2f})\n"
            f"est =({ex:.2f},{ey:.2f})\n"
            f"v=({vx:.2f},{vy:.2f})  |v|={math.hypot(vx,vy):.2f}\n"
            f"steer(w)={s0:.3f} rad/s\n"
            f"sensing={'ON' if z is not None else 'None'}  rays={N}\n"
            f"pathing={'ON' if pth is not None else 'None'}"
        )

        return [path_line, true_trace, est_trace, true_pt, est_pt, vel_q, head_q, hit_sc, hud] + ray_lines + end_pts

    ani = FuncAnimation(
        fig, update,
        frames=len(log),
        init_func=init,
        interval=50,
        blit=False,
        repeat=False
    )

    plt.show()


if __name__ == "__main__":
    main()
