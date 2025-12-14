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
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =========================
# 0) 프로젝트 루트를 Python 경로에 추가
#    (현재 파일이 test/ 아래에 있다고 가정: test/test_xxx.py)
# =========================
sys.path.insert(0, str(Path(__file__).parent.parent))

from module.MyPlanning import smooth_path_nd, point_to_path_min_distance_nd
from module.MyRobot import Moblie_robot
from module.MyControl import SimpleBicycleModel


# =========================
# 1) 테스트용 ㄱ자 경로 생성
# =========================
def make_L_path():
    path = []
    for x in range(6):
        path.append([x, 0])
    for y in range(1, 6):
        path.append([5, y])
    return path


def wrap_pi(a: float) -> float:
    """각도를 [-pi, pi] 범위로 감기"""
    return (a + math.pi) % (2 * math.pi) - math.pi


# =========================
# 2) 로봇이 경로를 "대충" 따라가도록 명령 만들기 (waypoint 기반)
#    - 정확한 PID path tracking 전 단계의 단순 추종
# =========================
def build_commands_follow_path(robot: Moblie_robot, s_path, *, v_ref=1.0, dt_ns=200_000_000, steer_gain=1.2):
    """
    s_path를 따라가도록 robot 큐에 ins_acc_mov들을 쌓는다.
    - steering은 dim=2이므로 (yaw_rate, 0.0) 형태로 넣는다.
    """
    for i in range(1, len(s_path)):
        tx, ty = s_path[i]

        # 현재 로봇 위치 기준으로 다음 타겟 방향 계산
        px, py = robot.get_position()
        dx = tx - px
        dy = ty - py

        target_theta = math.atan2(dy, dx)
        cur_theta = robot.get_orientation()[0]
        heading_err = wrap_pi(target_theta - cur_theta)

        steer_cmd = (steer_gain * heading_err, 0.0)

        # 1) 조향 먼저 살짝
        robot.ins_acc_mov(steering=steer_cmd, time=dt_ns // 3)

        # 2) 전진
        robot.ins_acc_mov(velocity=(v_ref, 0.0), time=dt_ns)


# =========================
# 3) 메인: 전체 플롯 + 애니메이션
# =========================
def main():
    # --- (A) 원본 경로 생성 ---
    path = make_L_path()

    # --- (B) 스무딩 적용 ---
    s_path = smooth_path_nd(
        path,
        alpha=0.2,
        beta=0.2,
        tol=1e-6,
        max_iter=10000,
        fix_ends=True
    )

    # plot용 분리
    x_orig = [p[0] for p in path]
    y_orig = [p[1] for p in path]
    x_s = [p[0] for p in s_path]
    y_s = [p[1] for p in s_path]

    # --- (C) 로봇 생성 ---
    robot = Moblie_robot(
        dim=2,
        position=s_path[0],
        velocity=(0.0, 0.0),
        steering=(0.0, 0.0),
        error_control=PIDVec(dim = 2, kp = 1, ki = 1, kd = 1),
        update_rate=50_000_000  # 50ms
    )

    # --- (D) 모델 설정 ---
    model = SimpleBicycleModel(wheelbase=1.5)

    # --- (E) 경로 따라가기 명령 만들기 ---
    build_commands_follow_path(
        robot,
        s_path,
        v_ref=1.0,
        dt_ns=200_000_000,
        steer_gain=1.2
    )

    # --- (F) 실행해서 로그 얻기 ---
    log = robot.run_queue_with_model(model)

    # 로그에서 위치/자세 뽑기
    xs = [st["position"][0] for st in log]
    ys = [st["position"][1] for st in log]
    thetas = [st["orientation"][0] for st in log]
    ts = [st["t_ns"] * 1e-9 for st in log]  # sec

    # =========================
    # (G) 애니메이션용 Figure 세팅
    # =========================
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_title("2D Path Smoothing + Closest Point + Robot Animation")

    # 범위(원본+스무딩+로봇 모두 고려)
    all_x = x_orig + x_s + xs
    all_y = y_orig + y_s + ys
    margin = 0.8
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    # --- 항상 보여야 하는 것들(정적 요소) ---
    # 원본 경로
    ax.plot(x_orig, y_orig, "o--", linewidth=2, label="Original Path (L-shape)")
    # 스무딩 경로
    ax.plot(x_s, y_s, "r-", linewidth=2, label="Smoothed Path")

    # --- 애니메이션으로 바뀌는 것들(동적 요소) ---
    # 로봇 궤적
    trace_line, = ax.plot([], [], "b-", linewidth=1.5, label="Robot Trace")

    # 현재 로봇 위치 (초록)
    cur_scatter = ax.scatter([], [], c="green", s=80, label="Current Position")

    # closest point (빨강)
    close_scatter = ax.scatter([], [], c="red", s=80, label="Closest Point on Path")

    # 최소거리 연결선
    min_line, = ax.plot([], [], "k--", linewidth=1.2)

    # heading 화살표
    heading = ax.quiver(
        0, 0, 0, 0,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="black",
        width=0.006
    )

    # 거리 텍스트
    dist_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    ax.legend(loc="upper left")

    # =========================
    # (H) 애니메이션 함수들
    # =========================
    def init():
        trace_line.set_data([], [])
        cur_scatter.set_offsets([[xs[0], ys[0]]])

        dmin, closest, seg_i, t = point_to_path_min_distance_nd([xs[0], ys[0]], s_path)
        close_scatter.set_offsets([[closest[0], closest[1]]])
        min_line.set_data([xs[0], closest[0]], [ys[0], closest[1]])

        heading.set_offsets([xs[0], ys[0]])
        heading.set_UVC(0.0, 0.0)

        dist_text.set_text("")
        return trace_line, cur_scatter, close_scatter, min_line, heading, dist_text

    def update(frame):
        x = xs[frame]
        y = ys[frame]
        theta = thetas[frame]

        # 1) trace 업데이트
        trace_line.set_data(xs[:frame+1], ys[:frame+1])

        # 2) current pos
        cur_scatter.set_offsets([[x, y]])

        # 3) closest point 계산
        dmin, closest, seg_i, tparam = point_to_path_min_distance_nd([x, y], s_path)
        close_scatter.set_offsets([[closest[0], closest[1]]])

        # 4) min distance line
        min_line.set_data([x, closest[0]], [y, closest[1]])

        # 5) heading 화살표
        arrow_len = 0.35
        dx = arrow_len * math.cos(theta)
        dy = arrow_len * math.sin(theta)
        heading.set_offsets([x, y])
        heading.set_UVC(dx, dy)

        # 6) 텍스트
        dist_text.set_text(
            f"t = {ts[frame]:.2f}s\n"
            f"min_dist = {dmin:.3f}\n"
            f"seg = {seg_i}, t = {tparam:.3f}"
        )

        return trace_line, cur_scatter, close_scatter, min_line, heading, dist_text

    ani = FuncAnimation(
        fig,
        update,
        frames=len(log),
        init_func=init,
        interval=50,   # ms
        blit=False     # scatter+text는 blit=False가 더 안정적일 때가 많음
    )

    plt.show()


if __name__ == "__main__":
    main()
