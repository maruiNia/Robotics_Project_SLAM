# -*- coding: utf-8 -*-
"""
test__slam.py
================
✅ 프로젝트 본체는 건드리지 않고, "로봇 로그(log)만" 이용해서 GraphSLAM(poses + landmarks)을
오프라인으로 검증하는 단일 테스트 스크립트.

- 입력: Moblie_robot.run_replanning_astar_follow()로 생성한 log
- 측정 생성(시뮬레이션):
    1) 오도메트리(relative dx,dy,dtheta) : log의 true pose로부터 생성 + 노이즈
    2) 랜드마크 관측(range, bearing)      : (true pose, landmark GT)로부터 생성 + 노이즈
- 최적화: Gauss-Newton + (약한) Levenberg damping
- 출력: True / Initial / Optimized trajectory + landmark 비교 플롯

※ 주의:
- 이 파일은 "GraphSLAM 개념 검증용"이라 데이터 어소시에이션(어떤 관측이 어떤 랜드마크인지)은
  이미 정답(landmark id)을 안다고 가정합니다.
"""

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
from pathlib import Path

# 현재 파일 위치(/mnt/data)를 PYTHONPATH에 추가해 module 패키지 import 가능하게 함
sys.path.insert(0, str(Path(__file__).parent.parent))

from module.MySlam import maze_map
from module.MySensor import Circle_Sensor
from module.MyControl import SimpleBicycleModel
from module.MyPlanning import PIDVec
from module.MyRobot_with_localization import Moblie_robot


# -----------------------------
# 유틸
# -----------------------------
def wrap_pi(a: float) -> float:
    """각도를 [-pi, pi)로 래핑"""
    return (a + math.pi) % (2 * math.pi) - math.pi


def v2(x):
    return np.asarray(x, dtype=float).reshape(-1)


# -----------------------------
# 측정 모델
# -----------------------------
def relative_pose(pose_i, pose_j):
    """
    world pose_i -> pose_j 를 pose_i 로컬 프레임에서 표현한 상대이동 (dx,dy,dtheta)
    pose: (x,y,theta)
    """
    xi, yi, thi = pose_i
    xj, yj, thj = pose_j

    dxw = xj - xi
    dyw = yj - yi

    c = math.cos(thi)
    s = math.sin(thi)

    # world -> i frame
    dx =  c * dxw + s * dyw
    dy = -s * dxw + c * dyw
    dth = wrap_pi(thj - thi)
    return np.array([dx, dy, dth], dtype=float)


def compose_pose(pose, d):
    """
    pose ⊕ d  (pose frame에서의 dx,dy,dtheta를 world로 합성)
    """
    x, y, th = pose
    dx, dy, dth = d

    c = math.cos(th)
    s = math.sin(th)

    x2 = x + c * dx - s * dy
    y2 = y + s * dx + c * dy
    th2 = wrap_pi(th + dth)
    return np.array([x2, y2, th2], dtype=float)


def predict_range_bearing(pose, lm):
    """pose에서 landmark를 봤을 때 (range, bearing) 예측"""
    x, y, th = pose
    lx, ly = lm
    dx = lx - x
    dy = ly - y
    r = math.hypot(dx, dy)
    b = wrap_pi(math.atan2(dy, dx) - th)
    return np.array([r, b], dtype=float)


# -----------------------------
# GraphSLAM(poses + landmarks) : Gauss-Newton
# -----------------------------
class PoseLandmarkGraphSLAM:
    """
    state vector:
      [pose0(3), pose1(3), ... poseN-1(3), lm0(2), lm1(2), ... lmM-1(2)]
    """

    def __init__(self, num_poses: int, num_landmarks: int):
        self.N = int(num_poses)
        self.M = int(num_landmarks)
        self.dim = 3 * self.N + 2 * self.M

        self.odom_edges = []   # (i, i+1, z(3), Omega(3x3))
        self.lm_edges = []     # (i, j, z(2), Omega(2x2))
        self.prior = None      # (i, z(3), Omega(3x3))  게이지 고정

    def idx_pose(self, i: int):
        base = 3 * i
        return slice(base, base + 3)

    def idx_lm(self, j: int):
        base = 3 * self.N + 2 * j
        return slice(base, base + 2)

    def set_prior(self, i: int, z, Omega):
        self.prior = (int(i), v2(z), np.asarray(Omega, dtype=float))

    def add_odom(self, i: int, z, Omega):
        self.odom_edges.append((int(i), int(i) + 1, v2(z), np.asarray(Omega, dtype=float)))

    def add_lm_obs(self, i: int, j: int, z, Omega):
        self.lm_edges.append((int(i), int(j), v2(z), np.asarray(Omega, dtype=float)))

    def unpack(self, X):
        """X -> poses(N,3), lms(M,2)"""
        X = v2(X)
        poses = X[: 3 * self.N].reshape(self.N, 3)
        lms = X[3 * self.N :].reshape(self.M, 2)
        return poses, lms

    def pack(self, poses, lms):
        return np.concatenate([np.asarray(poses).reshape(-1), np.asarray(lms).reshape(-1)])

    def build_linear_system(self, X, damping=1e-6):
        """
        H dx = -g  형태로 누적 (여기선 b = J^T Ω e 를 g로 두고 dx = -(H)^{-1} g)
        """
        H = np.zeros((self.dim, self.dim), dtype=float)
        g = np.zeros((self.dim,), dtype=float)

        poses, lms = self.unpack(X)

        # ---- prior (게이지 고정) ----
        if self.prior is not None:
            i, z, Omega = self.prior
            pi = poses[i]
            e = np.array([pi[0] - z[0], pi[1] - z[1], wrap_pi(pi[2] - z[2])], dtype=float)

            Ji = np.eye(3)
            si = self.idx_pose(i)
            H[si, si] += Ji.T @ Omega @ Ji
            g[si] += Ji.T @ Omega @ e

        # ---- odometry edges ----
        for (i, j, z, Omega) in self.odom_edges:
            pi = poses[i]
            pj = poses[j]

            # 예측: z_hat = relative_pose(pi, pj)
            zhat = relative_pose(pi, pj)
            e = zhat - z
            e[2] = wrap_pi(e[2])

            # Jacobian (수치 미분: 안정적이고 구현이 간단)
            # state = [pi(3), pj(3)] -> zhat(3)
            eps = 1e-6
            Ji = np.zeros((3, 3), dtype=float)
            Jj = np.zeros((3, 3), dtype=float)

            for k in range(3):
                dpi = np.zeros(3); dpi[k] = eps
                z1 = relative_pose(pi + dpi, pj)
                z0 = relative_pose(pi - dpi, pj)
                Ji[:, k] = (z1 - z0) / (2 * eps)

                dpj = np.zeros(3); dpj[k] = eps
                z1 = relative_pose(pi, pj + dpj)
                z0 = relative_pose(pi, pj - dpj)
                Jj[:, k] = (z1 - z0) / (2 * eps)

            si = self.idx_pose(i)
            sj = self.idx_pose(j)

            H[si, si] += Ji.T @ Omega @ Ji
            H[si, sj] += Ji.T @ Omega @ Jj
            H[sj, si] += Jj.T @ Omega @ Ji
            H[sj, sj] += Jj.T @ Omega @ Jj

            g[si] += Ji.T @ Omega @ e
            g[sj] += Jj.T @ Omega @ e

        # ---- landmark observation edges (range,bearing) ----
        for (i, j, z, Omega) in self.lm_edges:
            p = poses[i]
            lm = lms[j]

            zhat = predict_range_bearing(p, lm)
            e = zhat - z
            e[1] = wrap_pi(e[1])

            # Jacobian wrt pose(3) and lm(2) : 수치 미분
            eps = 1e-6
            Jp = np.zeros((2, 3), dtype=float)
            Jl = np.zeros((2, 2), dtype=float)

            for k in range(3):
                dp = np.zeros(3); dp[k] = eps
                z1 = predict_range_bearing(p + dp, lm)
                z0 = predict_range_bearing(p - dp, lm)
                Jp[:, k] = (z1 - z0) / (2 * eps)

            for k in range(2):
                dl = np.zeros(2); dl[k] = eps
                z1 = predict_range_bearing(p, lm + dl)
                z0 = predict_range_bearing(p, lm - dl)
                Jl[:, k] = (z1 - z0) / (2 * eps)

            sp = self.idx_pose(i)
            sl = self.idx_lm(j)

            H[sp, sp] += Jp.T @ Omega @ Jp
            H[sp, sl] += Jp.T @ Omega @ Jl
            H[sl, sp] += Jl.T @ Omega @ Jp
            H[sl, sl] += Jl.T @ Omega @ Jl

            g[sp] += Jp.T @ Omega @ e
            g[sl] += Jl.T @ Omega @ e

        # damping (Levenberg)
        H += damping * np.eye(self.dim)
        return H, g

    def solve(self, X0, iters=10, damping=1e-6, step_scale=1.0, verbose=True):
        X = v2(X0).copy()

        for it in range(iters):
            H, g = self.build_linear_system(X, damping=damping)

            # H dx = -g
            try:
                dx = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                # 혹시 수치 불안정하면 최소제곱으로
                dx, *_ = np.linalg.lstsq(H, -g, rcond=None)

            X = X + step_scale * dx

            if verbose:
                err = float(np.linalg.norm(g))
                print(f"[iter {it:02d}] ||g|| = {err:.6e}, ||dx|| = {float(np.linalg.norm(dx)):.6e}")

            if np.linalg.norm(dx) < 1e-6:
                break

        return X


# -----------------------------
# 1) EKF 테스트 때와 유사한 환경으로 log 생성
# -----------------------------
def make_log():
    # (test_est_pos_ani.py에서 가져온 맵/파라미터들)
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
    end_g2 = (11, 1)

    # ✅ 테스트 랜드마크(셀 좌표)
    land_cells = [(2, 6), (3, 4), (7, 2), (9, 8)]
    cell_size = 5

    m = maze_map(maze, start, end_g2, cell_size=cell_size)

    # 셀 -> 월드 좌표(센터)
    def cell_to_world(c):
        cx, cy = c
        return ((cx + 0.5) * cell_size, (cy + 0.5) * cell_size)

    landmarks = np.array([cell_to_world(c) for c in land_cells], dtype=float)

    vel_ref = 2.0
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
    robot.set_pid(pid, v_ref=vel_ref)
    robot.set_fake_fast_sensing(True, sigma=0.1)
    robot.enable_ekf(True)

    model = SimpleBicycleModel(wheelbase=1.2)

    print("로봇 실행 및 로그 확보 중...")
    log = robot.run_replanning_astar_follow(
        model,
        expansion=1,
        smooth=True,
        smooth_alpha=0.6,
        smooth_beta=0.3,
        v_ref=vel_ref,
        dt_ns=50_000_000,
        goal_tolerance=0.5,
        max_steps=1500,        # ✅ 너무 길면 느려져서 테스트용으로 줄임
        steer_limit=1.2,
        use_heading_term=True,
        heading_gain=0.6,
        look_ahead=6,
    )
    
    print("로봇 실행 완료.")
    print(f"로그 길이: {len(log)}")
    print(f"로그 속성 키 : {list(log[0].keys())}")
    print(f"로그 예시 : {log[0]}")
    print(f"랜드마크 GT:\n{landmarks}")

    if not log:
        raise RuntimeError("log가 비어있습니다. 로봇 실행이 실패했을 수 있어요.")

    return log, m, landmarks


# -----------------------------
# 2) log -> (poses_gt, poses_init) 뽑기 + 측정 생성
# -----------------------------
def extract_poses_from_log(log, *, init_noise=(0.30, 0.30, 0.12), seed=0):
    """
    returns:
      poses_gt : (T,3) [x,y,theta]
      poses_init : (T,3) 초기값(디버깅용으로 gt + noise)
    """
    pos = np.array([st["position"] for st in log], dtype=float)

    def _theta_from_orientation_value(v):
        # 스칼라면 theta로 간주
        if isinstance(v, (int, float, np.floating)):
            return float(v)

        # 튜플/리스트/ndarray면 (ox, oy, ...) 방향벡터 가능성
        arr = np.array(v, dtype=float).reshape(-1)
        if arr.size >= 2:
            return math.atan2(arr[1], arr[0])   # oy, ox
        if arr.size == 1:
            return float(arr[0])
        return 0.0

    # 1) theta 추출
    if "orientation" in log[0]:
        th = np.array([_theta_from_orientation_value(st["orientation"]) for st in log], dtype=float)
        th = np.array([wrap_pi(a) for a in th], dtype=float)

    elif "theta" in log[0]:
        th = np.array([float(st["theta"]) for st in log], dtype=float)
        th = np.array([wrap_pi(a) for a in th], dtype=float)

    else:
        # 없으면 위치 변화로 heading 추정
        th = np.zeros(len(log), dtype=float)
        for i in range(1, len(log)):
            dx = pos[i, 0] - pos[i - 1, 0]
            dy = pos[i, 1] - pos[i - 1, 1]
            if abs(dx) + abs(dy) > 1e-9:
                th[i] = math.atan2(dy, dx)
            else:
                th[i] = th[i - 1]
        th = np.array([wrap_pi(a) for a in th], dtype=float)

    poses_gt = np.column_stack([pos[:, 0], pos[:, 1], th])

    # 2) 초기값 만들기 (gt + noise)
    rng = np.random.default_rng(seed)
    sx, sy, sth = init_noise
    noise = np.column_stack([
        rng.normal(0, sx, size=len(log)),
        rng.normal(0, sy, size=len(log)),
        rng.normal(0, sth, size=len(log)),
    ])
    poses_init = poses_gt + noise
    poses_init[:, 2] = np.array([wrap_pi(a) for a in poses_init[:, 2]], dtype=float)

    return poses_gt, poses_init


def make_measurements(poses_gt, landmarks_gt, *,
                      stride=5,
                      sigma_odom=(0.08, 0.08, 0.03),
                      sigma_lm=(0.25, 0.04),
                      max_range=35.0,
                      p_obs=0.8,
                      seed=42):
    """
    poses_gt(전체)에서 stride로 서브샘플링하여 pose graph를 만들고,
    그 pose들에 대해 odom / landmark 측정을 생성.

    반환:
      poses_gt_sub (N,3), landmarks_gt (M,2),
      odom_edges: list[(i, z(3))]
      lm_edges:   list[(i, j, z(2))]
    """
    rng = np.random.default_rng(seed)

    idx = np.arange(0, len(poses_gt), stride, dtype=int)
    poses = poses_gt[idx].copy()
    N = len(poses)
    M = len(landmarks_gt)

    # odom edges
    odoms = []
    sx, sy, sth = sigma_odom
    for i in range(N - 1):
        z = relative_pose(poses[i], poses[i + 1])
        z_noisy = z + np.array([rng.normal(0, sx), rng.normal(0, sy), rng.normal(0, sth)], dtype=float)
        z_noisy[2] = wrap_pi(z_noisy[2])
        odoms.append((i, z_noisy))

    # landmark obs
    lms = []
    sr, sb = sigma_lm
    for i in range(N):
        p = poses[i]
        for j in range(M):
            lm = landmarks_gt[j]
            z = predict_range_bearing(p, lm)
            if z[0] > max_range:
                continue
            if rng.random() > p_obs:
                continue
            z_noisy = z + np.array([rng.normal(0, sr), rng.normal(0, sb)], dtype=float)
            z_noisy[1] = wrap_pi(z_noisy[1])
            lms.append((i, j, z_noisy))

    return poses, landmarks_gt.copy(), odoms, lms


def init_landmarks_from_first_obs(poses_init, lm_edges, M):
    """
    초기 랜드마크 추정치를 관측 1개 이상 있는 것들은 '역측정'으로 초기화.
    관측이 없는 랜드마크는 (0,0)에 두지 말고 평균 근처에 배치.
    """
    lms = np.full((M, 2), np.nan, dtype=float)

    # 첫 관측으로 초기화
    for (i, j, z) in lm_edges:
        if not np.any(np.isnan(lms[j])):
            continue
        r, b = z
        x, y, th = poses_init[i]
        ang = th + b
        lms[j] = np.array([x + r * math.cos(ang), y + r * math.sin(ang)], dtype=float)

    # nan 채우기
    valid = lms[~np.isnan(lms[:, 0])]
    if len(valid) == 0:
        center = np.array([0.0, 0.0])
    else:
        center = valid.mean(axis=0)

    rng = np.random.default_rng(7)
    for j in range(M):
        if np.any(np.isnan(lms[j])):
            lms[j] = center + rng.normal(0, 2.0, size=(2,))
    return lms


# -----------------------------
# 3) 실행
# -----------------------------
def main():
    log, m, landmarks_gt = make_log()
    poses_gt_full, poses_init_full = extract_poses_from_log(log)

    # ✅ Graph 크기 줄이기(수렴 + 속도 안정)
    stride = 5
    poses_gt, landmarks_gt, odom_edges, lm_edges = make_measurements(
        poses_gt_full, landmarks_gt,
        stride=stride,
        sigma_odom=(0.08, 0.08, 0.03),
        sigma_lm=(0.25, 0.04),
        max_range=40.0,
        p_obs=0.85,
        seed=3,
    )

    # poses 초기값도 동일 stride 적용
    poses_init = poses_init_full[np.arange(0, len(poses_init_full), stride)].copy()
    N = len(poses_gt)
    M = len(landmarks_gt)

    # landmark 초기값
    lms_init = init_landmarks_from_first_obs(poses_init, lm_edges, M)

    # GraphSLAM 구성
    slam = PoseLandmarkGraphSLAM(num_poses=N, num_landmarks=M)

    # 게이지 고정: 첫 포즈 prior (강하게)
    Omega_prior = np.diag([1e6, 1e6, 1e6])
    slam.set_prior(0, poses_init[0], Omega_prior)

    # 정보행렬(= inverse covariance)
    sx, sy, sth = 0.08, 0.08, 0.03
    Omega_odom = np.diag([1/(sx*sx), 1/(sy*sy), 1/(sth*sth)])

    sr, sb = 0.25, 0.04
    Omega_lm = np.diag([1/(sr*sr), 1/(sb*sb)])

    for (i, z) in odom_edges:
        slam.add_odom(i, z, Omega_odom)

    for (i, j, z) in lm_edges:
        slam.add_lm_obs(i, j, z, Omega_lm)

    X0 = slam.pack(poses_init, lms_init)

    print("\n=== GraphSLAM optimize start ===")
    Xopt = slam.solve(X0, iters=12, damping=1e-6, step_scale=1.0, verbose=True)
    poses_opt, lms_opt = slam.unpack(Xopt)

    # -----------------------------
    # 4) 결과 시각화
    # -----------------------------
    plt.figure(figsize=(8, 8))
    plt.axis("equal")
    plt.grid(True)
    plt.title("GraphSLAM Test (poses + landmarks)")

    # traj
    plt.plot(poses_gt[:, 0], poses_gt[:, 1], label="GT traj")
    plt.plot(poses_init[:, 0], poses_init[:, 1], label="Init traj (EKF/noisy)")
    plt.plot(poses_opt[:, 0], poses_opt[:, 1], label="Optimized traj")

    # landmarks
    plt.scatter(landmarks_gt[:, 0], landmarks_gt[:, 1], marker="D", label="GT landmarks")
    plt.scatter(lms_init[:, 0], lms_init[:, 1], marker="x", label="Init landmarks")
    plt.scatter(lms_opt[:, 0], lms_opt[:, 1], marker="o", label="Optimized landmarks")

    # 간단히 관측 일부를 선으로 표시(너무 많으면 지저분해서 일부만)
    for (i, j, z) in lm_edges[:: max(1, len(lm_edges)//50)]:
        p = poses_opt[i]
        lm = lms_opt[j]
        plt.plot([p[0], lm[0]], [p[1], lm[1]], linewidth=0.5)

    plt.legend()
    plt.tight_layout()
    out_png = Path(__file__).with_suffix('')
    out_png = str(out_png) + '_result.png'
    plt.savefig(out_png, dpi=160)
    print(f'결과 그림 저장: {out_png}')

    # 오차 출력
    traj_rmse = np.sqrt(np.mean((poses_opt[:, :2] - poses_gt[:, :2])**2))
    lm_rmse = np.sqrt(np.mean((lms_opt - landmarks_gt)**2))
    print(f"\nRMSE traj(x,y): {traj_rmse:.4f}")
    print(f"RMSE landmarks(x,y): {lm_rmse:.4f}")


if __name__ == "__main__":
    main()
