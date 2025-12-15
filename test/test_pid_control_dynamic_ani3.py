import sys
from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from module.MySlam import maze_map
from module.MyRobot import Moblie_robot
from module.MyControl import SimpleBicycleModel
from module.MyPlanning import point_to_path_min_distance_nd, PIDVec


def draw_obstacles(ax, slam_map, *, alpha=0.25):
    """SlamMap의 obstacle을 사각형으로 그림(시각화용)."""
    for obs in slam_map.get_obs_list():
        (min_x, min_y), (max_x, max_y) = obs.bounds()
        w = max_x - min_x
        h = max_y - min_y
        ax.add_patch(Rectangle((min_x, min_y), w, h, alpha=alpha))


def main():
    # -------------------------
    # 1) 맵 생성
    # -------------------------
    maze = [
        [0, 1, 1, 1, 1, 1],
        [0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 1],
    ]
    cell_size = 5
    m = maze_map(maze, (0, 0), (4, 2), cell_size=cell_size)

    # -------------------------
    # 2) 로봇 + 모델 세팅
    # -------------------------
    speed = 2

    robot = Moblie_robot(
        dim=2,
        position=m.start_point,
        end_point=m.end_point,
        update_rate=50_000_000,  # 50ms
    )
    robot.set_test_map_mode(True, m)
    pid = PIDVec(dim=2, kp=1.5, ki=0.5, kd=0.5, i_limit=1.0)
    robot.set_pid(pid, v_ref=speed)
    model = SimpleBicycleModel(wheelbase=1.0)

    # -------------------------
    # 3) “매 tick 재계획 + 스무딩 + 1스텝 추종” 실행
    # -------------------------
    log = robot.run_replanning_astar_follow(
        model,
        resolution=1,       # ✅ 나노가 원하는 계획 격자
        expansion=1,          # 필요시 0/1로 튜닝
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
        look_ahead=10,  # 너무 작으면 재계획 자주, 너무 크면 반응 느림
    )

    print(f"log steps = {len(log)}")
    if log:
        print("last =", log[-1])

    # -------------------------
    # 4) 로그 파싱
    # -------------------------
    xs = [st["position"][0] for st in log]
    ys = [st["position"][1] for st in log]
    thetas = [st.get("orientation", (0.0, 0.0))[0] for st in log]
    notes = [st.get("note", "") for st in log]
    paths = [st.get("pathing", None) for st in log]  # ✅ 핵심: 실시간 경로

    if not xs:
        print("log가 비어있습니다.")
        return

    # start / goal 마커
    sx, sy = m.start_point[0], m.start_point[1]
    gx, gy = m.end_point[0], m.end_point[1]

    # -------------------------
    # 5) 플롯/애니메이션 세팅
    # -------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    # 맵 경계
    if m.limit is not None:
        (x0, y0), (x1, y1) = m.limit
        ax.set_xlim(x0 - 1, x1 + 1)
        ax.set_ylim(y0 - 1, y1 + 1)

    draw_obstacles(ax, m, alpha=0.25)

    # start/goal
    ax.plot([sx], [sy], "bs", markersize=8, label="Start")
    ax.plot([gx], [gy], "g^", markersize=8, label="Goal")

    # ✅ 동적 재계획 경로(스무딩 포함)
    path_line, = ax.plot([], [], "r-", linewidth=2, label="Replanned Smooth Path")

    # 로봇 궤적
    trace_line, = ax.plot([], [], "b-", linewidth=2, label="Robot Trace")

    # 현재 위치 / closest
    robot_point, = ax.plot([], [], "go", markersize=8, label="Current Position")
    closest_point, = ax.plot([], [], "ro", markersize=8, label="Closest Point (pathing)")

    # 최소거리 선
    min_line, = ax.plot([], [], "k--", linewidth=1, label="Min Distance")

    # heading(화살표)
    heading = ax.quiver([xs[0]], [ys[0]], [1], [0], angles="xy", scale_units="xy", scale=1)

    # 텍스트
    info_text = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes,
        va="top"
    )

    ax.legend(loc="upper left")

    # “현재 사용 중인 pathing”을 프레임 간 유지하기 위한 상태
    current_path = None

    def init():
        nonlocal current_path
        current_path = None

        trace_line.set_data([], [])
        path_line.set_data([], [])
        robot_point.set_data([], [])
        closest_point.set_data([], [])
        min_line.set_data([], [])
        info_text.set_text("")

        heading.set_offsets([xs[0], ys[0]])
        heading.set_UVC(1.0, 0.0)

        return trace_line, path_line, robot_point, closest_point, min_line, heading, info_text

    def update(frame):
        nonlocal current_path

        x = xs[frame]
        y = ys[frame]
        theta = thetas[frame]

        # trace
        trace_line.set_data(xs[:frame + 1], ys[:frame + 1])

        # current position
        robot_point.set_data([x], [y])

        # ✅ 최신 pathing 갱신 (있으면 교체, 없으면 이전 유지)
        p = paths[frame]
        if p is not None and len(p) >= 2:
            # tuple(tuple(...)) 형태일 수도 있으니 list로 써도 무방
            current_path = p

        # ✅ pathing 기반 시각화 + closest point
        if current_path is not None and len(current_path) >= 2:
            px = [pt[0] for pt in current_path]
            py = [pt[1] for pt in current_path]
            path_line.set_data(px, py)

            dmin, closest, seg_i, t = point_to_path_min_distance_nd([x, y], current_path)
            closest_point.set_data([closest[0]], [closest[1]])
            min_line.set_data([x, closest[0]], [y, closest[1]])

            info_text.set_text(
                f"frame={frame}/{len(xs)-1}\n"
                f"note={notes[frame]}\n"
                f"min_dist={dmin:.3f}\n"
                f"seg={seg_i}, t={t:.3f}"
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

        # heading
        arrow_len = 0.8
        dx = arrow_len * math.cos(theta)
        dy = arrow_len * math.sin(theta)
        heading.set_offsets([x, y])
        heading.set_UVC(dx, dy)

        return trace_line, path_line, robot_point, closest_point, min_line, heading, info_text

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

