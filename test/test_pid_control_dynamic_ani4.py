import sys
from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from module.MySlam import maze_map
from module.MyRobot_with_localization import Moblie_robot
from module.MyControl import SimpleBicycleModel
from module.MyPlanning import point_to_path_min_distance_nd, PIDVec
from module.MySensor import Circle_Sensor


def draw_obstacles(ax, slam_map, *, alpha=0.25):
    """SlamMap의 obstacle을 사각형으로 그림(시각화용)."""
    for obs in slam_map.get_obs_list():
        (min_x, min_y), (max_x, max_y) = obs.bounds()
        w = max_x - min_x
        h = max_y - min_y
        ax.add_patch(Rectangle((min_x, min_y), w, h, alpha=alpha))


def localize_window_by_circle_scan(
    slam_map,
    sensor: Circle_Sensor,
    z_meas,
    *,
    prev_est=None,
    resolution: float = 1.0,
    sigma: float = 0.4,
    window_cells: int = 8,   # prev_est 주변 +/- window_cells 칸만 탐색
):
    """
    센서측정(z_meas)과 맵을 매칭해서 (x,y) 추정.
    - prev_est가 있으면 주변 window로만 탐색 -> 훨씬 빨라짐.
    - None(범위 내 장애물 없음)은 max_range(distance)로 취급.
    """
    if slam_map.limit is None:
        raise ValueError("slam_map.limit가 필요합니다. maze_map을 쓰거나 set_limit 하세요.")

    max_range = float(sensor.get_distance())
    (x0, y0), (x1, y1) = slam_map.limit

    # 탐색 범위 결정
    if prev_est is None:
        sx0, sy0, sx1, sy1 = x0, y0, x1, y1
    else:
        px, py = prev_est
        span = window_cells * resolution
        sx0, sy0 = px - span, py - span
        sx1, sy1 = px + span, py + span
        # limit에 클램프
        sx0 = max(sx0, x0)
        sy0 = max(sy0, y0)
        sx1 = min(sx1, x1)
        sy1 = min(sy1, y1)

    nx = int(math.ceil((sx1 - sx0) / resolution))
    ny = int(math.ceil((sy1 - sy0) / resolution))

    best_ll = -1e18
    best_xy = (sx0, sy0)

    for iy in range(ny):
        cy = sy0 + (iy + 0.5) * resolution
        for ix in range(nx):
            cx = sx0 + (ix + 0.5) * resolution

            if slam_map.is_wall((cx, cy), inclusive=True):
                continue

            z_pred = sensor.sensing((cx, cy))

            ll = 0.0
            for m, p in zip(z_meas, z_pred):
                m2 = max_range if m is None else float(m)
                p2 = max_range if p is None else float(p)
                err = m2 - p2
                ll += -0.5 * (err / sigma) ** 2

            if ll > best_ll:
                best_ll = ll
                best_xy = (cx, cy)

    return best_xy


def main():
    # -------------------------
    # 1) 맵 생성
    # -------------------------
    # maze = [
    #     [0, 1, 1, 1, 1, 1],
    #     [0, 1, 0, 0, 0, 1],
    #     [0, 0, 0, 1, 0, 1],
    # ]
    # 목표
    maze = [ ## 0 1  2 3  4  5 6  7  8 9 10 11 12 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 9
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # 8
        [0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0], # 7
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0], # 6
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0], # 5
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 4
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], # 3
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # 2
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # 1
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # 0
    ]
    cell_size = 3
    start = (0, 9)
    end_g1 = (11, 9)
    end_g2 = (11, 1)

    land_1 = (2, 6)
    land_2 = (3, 4)
    land_3 = (7, 2)

    m = maze_map(maze, start, end_g1, cell_size=cell_size)

    # -------------------------
    # 2) 로봇 + 센서 + 모델 세팅
    # -------------------------
    speed = 2.0

    # 원형 센싱: number개로 360도를 균등 분할(예: 3개면 120도 간격)
    sensor = Circle_Sensor(
        dim=2,
        number=20,        # 필요하면 3, 8, 16 등으로 바꿔도 됨
        distance=12,      # 최대 측정 거리(맵 크기에 맞게)
        slam_map=m,
        step=0.05,
    )

    robot = Moblie_robot(
        dim=2,
        position=m.start_point,
        end_point=m.end_point,
        update_rate=50_000_000,  # 50ms
        sensing_mode=True,
        sensor=sensor,
    )

    robot.set_test_map_mode(True, m)

    pid = PIDVec(dim=2, kp=5, ki=0.5, kd=0.5, i_limit=1.0)
    robot.set_pid(pid, v_ref=speed)

    # 로컬라이제이션 파라미터(로봇 내부에서도 추정치 사용)
    robot.set_localization_params(resolution=1.0, sigma=0.4, period=5)
    robot.enable_estimated_pose(True)

    model = SimpleBicycleModel(wheelbase=1.0)

    # -------------------------
    # 3) “매 tick 재계획 + 스무딩 + 1스텝 추종” 실행
    # -------------------------
    log = robot.run_replanning_astar_follow(
        model,
        resolution=1,
        expansion=1,
        smooth=True,
        smooth_alpha=0.6,
        smooth_beta=0.3,
        v_ref=speed,
        dt_ns=50_000_000,
        goal_tolerance=0.5,
        max_steps=5000,
        steer_limit=1.7,
        use_heading_term=True,
        heading_gain=0.6,
        look_ahead=10,
    )

    if not log:
        print("log가 비어있습니다.")
        return

    # -------------------------
    # 4) 로그 파싱 (true trajectory)
    # -------------------------
    xs = [st["position"][0] for st in log]
    ys = [st["position"][1] for st in log]
    thetas = [st.get("orientation", (0.0, 0.0))[0] for st in log]
    notes = [st.get("note", "") for st in log]
    paths = [st.get("pathing", None) for st in log]

    # -------------------------
    # 5) (중요) 추정 위치(est) 시계열 만들기
    # - 로봇 내부에서 추정치를 썼더라도 log에는 true만 들어있어서,
    #   애니메이션용으로 센서측정 -> 맵매칭을 다시 돌려 est를 만든다.
    # -------------------------
    loc_period = 5        # 5프레임마다 1번만 재추정(가벼움)
    resolution = 1.0
    sigma = 0.4
    window_cells = 8

    est_xs, est_ys = [], []
    est = (xs[0], ys[0])   # 초기값은 start 근처로 두기
    for i in range(len(xs)):
        if i % loc_period == 0:
            z = sensor.sensing((xs[i], ys[i]))
            est = localize_window_by_circle_scan(
                m, sensor, z,
                prev_est=est,
                resolution=resolution,
                sigma=sigma,
                window_cells=window_cells,
            )
        est_xs.append(est[0])
        est_ys.append(est[1])

    # start / goal 마커
    sx, sy = m.start_point[0], m.start_point[1]
    gx, gy = m.end_point[0], m.end_point[1]

    # -------------------------
    # 6) 플롯/애니메이션 세팅
    # -------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    if m.limit is not None:
        (x0, y0), (x1, y1) = m.limit
        ax.set_xlim(x0 - 1, x1 + 1)
        ax.set_ylim(y0 - 1, y1 + 1)

    draw_obstacles(ax, m, alpha=0.25)

    ax.plot([sx], [sy], "bs", markersize=8, label="Start")
    ax.plot([gx], [gy], "g^", markersize=8, label="Goal")

    # 동적 재계획 경로
    path_line, = ax.plot([], [], "r-", linewidth=2, label="Replanned Smooth Path")

    # true / est 궤적
    true_trace, = ax.plot([], [], "b-", linewidth=2, label="True Trace")
    est_trace,  = ax.plot([], [], "m-", linewidth=2, label="Estimated Trace")

    # 현재 위치 점(진짜/추정)
    true_point, = ax.plot([], [], "go", markersize=7, label="True Position")
    est_point,  = ax.plot([], [], "mo", markersize=7, label="Estimated Position")

    # closest point (추정 위치 기준으로 계산해보자)
    closest_point, = ax.plot([], [], "ro", markersize=7, label="Closest (to path, using est)")
    min_line, = ax.plot([], [], "k--", linewidth=1, label="Min Distance (est->path)")

    # heading(화살표) - true heading 표시
    heading = ax.quiver([xs[0]], [ys[0]], [1], [0], angles="xy", scale_units="xy", scale=1)

    info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")
    ax.legend(loc="upper left")

    current_path = None

    def init():
        nonlocal current_path
        current_path = None

        true_trace.set_data([], [])
        est_trace.set_data([], [])
        path_line.set_data([], [])
        true_point.set_data([], [])
        est_point.set_data([], [])
        closest_point.set_data([], [])
        min_line.set_data([], [])
        info_text.set_text("")

        heading.set_offsets([xs[0], ys[0]])
        heading.set_UVC(1.0, 0.0)

        return (true_trace, est_trace, path_line, true_point, est_point,
                closest_point, min_line, heading, info_text)

    def update(frame):
        nonlocal current_path

        x, y = xs[frame], ys[frame]
        ex, ey = est_xs[frame], est_ys[frame]
        theta = thetas[frame]

        # trace
        true_trace.set_data(xs[:frame + 1], ys[:frame + 1])
        est_trace.set_data(est_xs[:frame + 1], est_ys[:frame + 1])

        # points
        true_point.set_data([x], [y])
        est_point.set_data([ex], [ey])

        # 최신 pathing 갱신 (있으면 교체, 없으면 이전 유지)
        p = paths[frame]
        if p is not None and len(p) >= 2:
            current_path = p

        if current_path is not None and len(current_path) >= 2:
            px = [pt[0] for pt in current_path]
            py = [pt[1] for pt in current_path]
            path_line.set_data(px, py)

            dmin, closest, seg_i, t = point_to_path_min_distance_nd([ex, ey], current_path)
            closest_point.set_data([closest[0]], [closest[1]])
            min_line.set_data([ex, closest[0]], [ey, closest[1]])

            info_text.set_text(
                f"frame={frame}/{len(xs)-1}\n"
                f"note={notes[frame]}\n"
                f"min_dist(est->path)={dmin:.3f}\n"
                f"seg={seg_i}, t={t:.3f}\n"
                f"loc_period={loc_period}, window_cells={window_cells}"
            )
        else:
            path_line.set_data([], [])
            closest_point.set_data([], [])
            min_line.set_data([], [])
            info_text.set_text(
                f"frame={frame}/{len(xs)-1}\n"
                f"note={notes[frame]}\n"
                f"(no pathing in log)"
            )

        # heading (true theta)
        arrow_len = 0.8
        dx = arrow_len * math.cos(theta)
        dy = arrow_len * math.sin(theta)
        heading.set_offsets([x, y])
        heading.set_UVC(dx, dy)

        return (true_trace, est_trace, path_line, true_point, est_point,
                closest_point, min_line, heading, info_text)

    ani = FuncAnimation(
        fig,
        update,
        frames=len(xs),
        init_func=init,
        interval=50,
        blit=False,
        repeat=False,
    )

    plt.show()


if __name__ == "__main__":
    main()
