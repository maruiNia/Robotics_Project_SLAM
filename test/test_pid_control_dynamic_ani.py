import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =========================
# 0) 프로젝트 루트 경로 추가
# =========================
sys.path.insert(0, str(Path(__file__).parent.parent))

from module.MyPlanning import smooth_path_nd, point_to_path_min_distance_nd
from module.MyRobot import Moblie_robot
from module.MyControl import SimpleBicycleModel
from module.MyPlanning import PIDVec  # 네 PIDVec가 MyPlanning.py에 있다면


# =========================
# 1) 테스트용 ㄱ자 경로 생성
# =========================
def make_L_path():
    path = []
    for x in range(6):
        path.append([x, 0])
    for y in range(1, 6):
        path.append([5, y])
    for x in range(5, 11):
        path.append([x, 5])

    return path


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def signed_cte_2d(pos, closest, seg_i, path):
    """2D cross-track error에 부호 부여"""
    x, y = pos
    cx, cy = closest

    i = max(0, min(seg_i, len(path) - 2))
    x0, y0 = path[i]
    x1, y1 = path[i + 1]

    sx, sy = (x1 - x0, y1 - y0)      # segment 방향
    ex, ey = (x - cx, y - cy)        # closest -> pos

    cross_z = sx * ey - sy * ex
    sign = 1.0 if cross_z >= 0 else -1.0

    return sign * math.hypot(x - cx, y - cy)


# =========================
# 2) 메인: dynamic 제어 루프를 애니메이션 프레임에서 수행
# =========================
def main():
    # --- 경로 생성 & 스무딩 ---
    path = make_L_path()
    s_path = smooth_path_nd(
        path,
        alpha=0.2, beta=0.2,
        tol=1e-6, max_iter=10000,
        fix_ends=True
    )

    x_orig = [p[0] for p in path]
    y_orig = [p[1] for p in path]
    x_s = [p[0] for p in s_path]
    y_s = [p[1] for p in s_path]

    # --- 로봇 생성 ---
    robot = Moblie_robot(
        dim=2,
        position=s_path[0],
        velocity=(0.0, 0.0),
        steering=(0.0, 0.0),
        dynamic_operation_mode=True,   # dynamic 쓰겠다는 선언
        sensing_mode=False,
        update_rate=50_000_000         # 50ms
    )

    robot.set_path(s_path)

    # --- 모델 ---
    model = SimpleBicycleModel(wheelbase=1.2)
    robot.set_motion_model(model)

    # --- PID (중요: dim=2로 만들고 e=[cte,0] 넣을 것) ---
    pid = PIDVec(dim=2, kp=7, ki=0.05, kd=4, i_limit=1.0)
    robot.set_pid(pid, v_ref=1.0, control_dt_ns=robot.get_update_rate())

    # --- 제어 파라미터 ---
    dt_ns = robot.get_update_rate()
    dt_s = dt_ns * 1e-9

    v_ref = 1.0
    steer_limit = 3.0

    use_heading_term = True
    heading_gain = 1.2

    goal = s_path[-1]
    goal_tol = 0.30

    # 로봇 trace 저장용
    trace_x, trace_y = [], []

    # =========================
    # 시각화 세팅
    # =========================
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_title("Dynamic Path Tracking (Closest Point + PID + Bicycle Model)")

    # 화면 범위(전체 경로가 항상 보이게!)
    all_x = x_orig + x_s
    all_y = y_orig + y_s
    margin = 1.0
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    # --- 정적 요소: 전체 경로 항상 표시 ---
    ax.plot(x_orig, y_orig, "o--", linewidth=2, label="Original Path (L-shape)")
    ax.plot(x_s, y_s, "r-", linewidth=2, label="Smoothed Path")

    # --- 동적 요소 ---
    trace_line, = ax.plot([], [], "b-", linewidth=1.5, label="Robot Trace")

    cur_scatter = ax.scatter([], [], c="green", s=90, label="Current Position")
    close_scatter = ax.scatter([], [], c="red", s=90, label="Closest Point on Path")
    min_line, = ax.plot([], [], "k--", linewidth=1.2, label="Min Distance")

    heading = ax.quiver(
        0, 0, 0, 0,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="black",
        width=0.006
    )

    info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")
    ax.legend(loc="upper left")

    done = {"flag": False}  # 애니메이션 종료 플래그(클로저에서 변경)

    # =========================
    # 초기화
    # =========================
    def init():
        x, y = robot.get_position()
        trace_x.clear()
        trace_y.clear()
        trace_x.append(x)
        trace_y.append(y)

        trace_line.set_data(trace_x, trace_y)
        cur_scatter.set_offsets([[x, y]])

        dmin, closest, seg_i, tparam = point_to_path_min_distance_nd([x, y], s_path)
        close_scatter.set_offsets([[closest[0], closest[1]]])
        min_line.set_data([x, closest[0]], [y, closest[1]])

        heading.set_offsets([x, y])
        heading.set_UVC(0.0, 0.0)
        info_text.set_text("")

        return trace_line, cur_scatter, close_scatter, min_line, heading, info_text

    # =========================
    # 프레임 업데이트 = "dynamic 제어 1 tick"
    # =========================
    def update(frame):
        if done["flag"]:
            return trace_line, cur_scatter, close_scatter, min_line, heading, info_text

        # 1) 현재 위치에서 closest point
        x, y = robot.get_position()
        dmin, closest, seg_i, tparam = point_to_path_min_distance_nd([x, y], s_path)

        # 2) signed cross-track error
        cte = signed_cte_2d((x, y), closest, seg_i, s_path)

        # 3) PIDVec 입력은 반드시 벡터!
        #    e = [cte, 0.0]
        u = pid.step([cte, 0.0], dt_s)
        steer_pid = float(u[0])

        # 4) heading 보정(선택)
        steer_heading = 0.0
        if use_heading_term:
            i = max(0, min(seg_i, len(s_path) - 2))
            x0, y0 = s_path[i]
            x1, y1 = s_path[i + 1]
            target_theta = math.atan2(y1 - y0, x1 - x0)

            cur_theta = robot.get_orientation()[0]
            heading_err = wrap_pi(target_theta - cur_theta)
            steer_heading = heading_gain * heading_err

        steer = -(steer_pid + steer_heading)
        steer = max(-steer_limit, min(steer_limit, steer))

        # 5) 현재 tick의 입력을 robot 상태로 적용
        robot._velocity = (v_ref, 0.0)
        robot._steering = (steer, 0.0)

        # 6) 모델로 1 step 적분
        robot._step(dt_ns, note="dynamic_control")

        # 7) 업데이트된 상태로 그래픽 갱신
        x2, y2 = robot.get_position()
        trace_x.append(x2)
        trace_y.append(y2)
        trace_line.set_data(trace_x, trace_y)

        cur_scatter.set_offsets([[x2, y2]])

        dmin2, closest2, seg_i2, tparam2 = point_to_path_min_distance_nd([x2, y2], s_path)
        close_scatter.set_offsets([[closest2[0], closest2[1]]])
        min_line.set_data([x2, closest2[0]], [y2, closest2[1]])

        theta = robot.get_orientation()[0]
        arrow_len = 0.35
        dx = arrow_len * math.cos(theta)
        dy = arrow_len * math.sin(theta)
        heading.set_offsets([x2, y2])
        heading.set_UVC(dx, dy)

        info_text.set_text(
            f"t = {robot._time_ns*1e-9:.2f}s\n"
            f"cte = {cte:.3f}\n"
            f"min_dist = {dmin2:.3f}\n"
            f"steer = {steer:.3f}"
        )

        # 8) goal 체크
        if math.hypot(x2 - goal[0], y2 - goal[1]) <= goal_tol:
            done["flag"] = True
            info_text.set_text(info_text.get_text() + "\nGOAL!")
            # 프레임을 멈추고 싶으면 아래처럼:
            # ani.event_source.stop()

        return trace_line, cur_scatter, close_scatter, min_line, heading, info_text

    ani = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=5000,     # 넉넉히(도달하면 done 플래그로 사실상 멈춤)
        interval=50,     # ms
        blit=False
    )

    plt.show()


if __name__ == "__main__":
    main()
