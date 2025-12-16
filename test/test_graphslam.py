import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from module.MyGraphSlam import PoseGraphSLAM2D

def simulate_and_test(seed=7, N=200, dt=0.1):
    rng = np.random.default_rng(seed)

    # --- 1) "정답" 궤적 만들기 (직진 + 살짝 회전) ---
    true = []
    x, y, th = 0.0, 0.0, 0.0
    v_cmd = 1.0

    for k in range(N):
        # 중간부터 천천히 회전한다고 가정
        w_cmd = 0.0 if k < 80 else 0.25  # rad/s

        x += v_cmd * np.cos(th) * dt
        y += v_cmd * np.sin(th) * dt
        th += w_cmd * dt
        true.append((x, y, th))

    true_xy = np.array([(p[0], p[1]) for p in true])

    # --- 2) Dead-reckoning(오도메트리 누적) 생성 (노이즈 있음) ---
    dr_xy = []
    x, y, th = 0.0, 0.0, 0.0

    odo_sigma = 0.08  # 오도메트리 이동량 노이즈(표준편차)
    for k in range(N):
        w_cmd = 0.0 if k < 80 else 0.25

        # 실제 이동량(명령 기반) + 노이즈
        dx = v_cmd * np.cos(th) * dt + rng.normal(0, odo_sigma)
        dy = v_cmd * np.sin(th) * dt + rng.normal(0, odo_sigma)

        x += dx
        y += dy
        th += w_cmd * dt
        dr_xy.append((x, y))

    dr_xy = np.array(dr_xy)

    # --- 3) "절대 위치 측정" 만들기 (가끔만, 노이즈 큼) ---
    # 너의 scan-grid localization이 반환하는 z_xy 같은 역할
    z_every = 10
    meas_sigma = 0.45  # scan-grid는 튈 수 있으니 크게
    measurements = {}  # k -> (zx, zy)
    for k in range(N):
        if k % z_every == 0:
            zx = true_xy[k, 0] + rng.normal(0, meas_sigma)
            zy = true_xy[k, 1] + rng.normal(0, meas_sigma)
            measurements[k] = (zx, zy)

    # --- 4) Pose-Graph SLAM 구성 ---
    pg = PoseGraphSLAM2D(prior_info=1e6)
    pg.add_first_pose((0.0, 0.0), meas_var=1e-6)

    # 그래프에 노드/엣지 추가
    # 엣지는 "오도메트리 이동량(u_ij)"를 dr에서 역으로 계산해서 쓰는 방식
    # (실제로는 robot의 v,theta에서 dx,dy를 만들어 넣으면 됨)
    for k in range(1, N):
        pg.add_pose()

        # dr 이동량을 u로 사용(오도 기반)
        u = (dr_xy[k, 0] - dr_xy[k-1, 0], dr_xy[k, 1] - dr_xy[k-1, 1])
        pg.add_odometry(k-1, k, u_ij=u, odo_var=odo_sigma**2)

        # 절대 측정이 있는 시점엔 엣지 추가
        if k in measurements:
            pg.add_abs_position(k, measurements[k], meas_var=meas_sigma**2)

    opt_xy = np.array(pg.solve())

    # --- 5) 숫자로 성능 확인 (RMSE) ---
    def rmse(a, b):
        return float(np.sqrt(np.mean(np.sum((a-b)**2, axis=1))))

    rmse_dr = rmse(dr_xy, true_xy)
    rmse_opt = rmse(opt_xy, true_xy)

    print(f"RMSE dead-reckoning : {rmse_dr:.3f}")
    print(f"RMSE pose-graph SLAM: {rmse_opt:.3f}")

    # --- 6) 시각화 ---
    plt.figure(figsize=(7,7))
    plt.plot(true_xy[:,0], true_xy[:,1], label="True")
    plt.plot(dr_xy[:,0], dr_xy[:,1], label="Dead-reckoning")
    plt.plot(opt_xy[:,0], opt_xy[:,1], label="Pose-Graph SLAM")

    # 측정점도 찍기
    z = np.array(list(measurements.values()))
    plt.scatter(z[:,0], z[:,1], s=15, label="Abs meas (scan-grid-like)", alpha=0.7)

    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title("Pose-Graph SLAM test (odometry + occasional abs position)")
    plt.show()

simulate_and_test()
