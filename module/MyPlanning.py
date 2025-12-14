from __future__ import annotations
from typing import List, Sequence, Tuple, Union
import math

Number = Union[int, float]


def smooth_path_nd(
    points: Sequence[Sequence[Number]],
    *,
    alpha: float = 0.1,
    beta: float = 0.3,
    tol: float = 1e-6, # 수렴 허용오차
    max_iter: int = 10000,
    fix_ends: bool = True,
) -> List[List[float]]:
    """
    N차원 경로 스무딩 (Gradient Descent)

    points: 길이 N의 경로 점들, 각 점은 dim차원 벡터
            예) 2D: [[x0,y0], [x1,y1], ...]
                3D: [[x0,y0,z0], ...]
                6D: [[q1,q2,q3,q4,q5,q6], ...]

    업데이트(슬라이드 방식 일반화):
      Y_i = Y_i + alpha*(X_i - Y_i)                      # data term
      Y_i = Y_i + beta*(Y_{i+1} + Y_{i-1} - 2*Y_i)       # smoothness term

    반환: 스무딩된 점 리스트 (List[List[float]])
    """
    if not points:
        return []

    n = len(points)
    dim = len(points[0])
    if dim == 0:
        raise ValueError("각 점의 차원(dim)이 0이에요.")

    # 모든 점이 동일 차원인지 검사
    for i, p in enumerate(points):
        if len(p) != dim:
            raise ValueError(f"{i}번째 점의 차원이 달라요. expected dim={dim}, got={len(p)}")

    if n < 2:
        return [[float(v) for v in points[0]]]

    # 원본 X, 초기 Y
    X = [[float(v) for v in p] for p in points]
    Y = [p[:] for p in X]  # 시작은 원본 그대로

    start_i = 1 if fix_ends else 0
    end_i = n - 1 if fix_ends else n

    for _ in range(max_iter):
        total_change = 0.0

        for i in range(start_i, end_i):
            im1 = max(0, i - 1)
            ip1 = min(n - 1, i + 1)

            old = Y[i][:]

            # data term + smoothness term를 "모든 차원"에 대해 적용
            for d in range(dim):
                y = Y[i][d]
                y += alpha * (X[i][d] - y)
                y += beta * (Y[ip1][d] + Y[im1][d] - 2.0 * y)
                Y[i][d] = y

            total_change += sum(abs(old[d] - Y[i][d]) for d in range(dim))

        if total_change < tol:
            break

    return Y


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x*y for x, y in zip(a, b))


def sub(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]


def add(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]


def mul(a: Sequence[float], s: float) -> List[float]:
    return [x * s for x in a]


def norm(a: Sequence[float]) -> float:
    return math.sqrt(dot(a, a))


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def point_to_segment_distance_nd(
    p: Sequence[Number],
    a: Sequence[Number],
    b: Sequence[Number],
) -> Tuple[float, List[float], float]:
    """
    점 p 와 선분 a-b 사이의 최소거리.
    반환: (거리 d, 가장 가까운 점 q(리스트), t)
      - q = a + t*(b-a), t는 [0,1]로 클램프됨
    """
    p = [float(x) for x in p]
    a = [float(x) for x in a]
    b = [float(x) for x in b]

    if len(p) != len(a) or len(p) != len(b):
        raise ValueError("p, a, b의 차원(dim)이 서로 달라요.")

    ab = sub(b, a)
    ap = sub(p, a)
    ab2 = dot(ab, ab)

    if ab2 == 0.0:
        # a==b (길이 0 선분)
        q = a
        return norm(sub(p, q)), q, 0.0

    t = dot(ap, ab) / ab2
    t = clamp(t, 0.0, 1.0)
    q = add(a, mul(ab, t))
    d = norm(sub(p, q))
    return d, q, t


def point_to_path_min_distance_nd(
    p: Sequence[Number],
    path: Sequence[Sequence[Number]],
) -> Tuple[float, List[float], int, float]:
    """
    점 p 와 polyline 경로(path) 사이의 최소거리.
    path는 [[x0,y0,...], [x1,y1,...], ...] 형태 (smooth_path_nd 출력과 호환)

    반환: (min_dist, closest_point, seg_index, t)
      - seg_index: 최소거리 발생한 선분의 시작 인덱스 i (선분: path[i] -> path[i+1])
      - t: 그 선분 위의 보간 파라미터 [0,1]
    """
    if not path:
        raise ValueError("path가 비어있어요.")
    if len(path) == 1:
        # 점 하나면 그 점까지 거리
        q = [float(x) for x in path[0]]
        d = norm(sub([float(x) for x in p], q))
        return d, q, 0, 0.0

    best_d = float("inf")
    best_q: List[float] = []
    best_i = 0
    best_t = 0.0

    for i in range(len(path) - 1):
        d, q, t = point_to_segment_distance_nd(p, path[i], path[i + 1])
        if d < best_d:
            best_d = d
            best_q = q
            best_i = i
            best_t = t

    return best_d, best_q, best_i, best_t

class PIDVec:
    """
    다차원(벡터) PID 제어기.

    - dim 차원의 오차 벡터 e = [e0, e1, ..., e_{dim-1}]를 입력으로 받아
    - 각 차원별로 독립적인 PID 제어를 수행한다.
    - 결과 u = [u0, u1, ..., u_{dim-1}] 역시 dim 차원의 제어 입력 벡터로 반환된다.

    주 용도:
    - 로봇의 조향, 속도, 위치 오차 등을 벡터 형태로 동시에 제어
    """

    def __init__(self, dim: int, kp: float, ki: float, kd: float, i_limit: float = 1.0):
        # 제어할 벡터의 차원 수
        self.dim = dim

        # PID 게인
        self.kp = kp  # Proportional gain (비례 항)
        self.ki = ki  # Integral gain (적분 항)
        self.kd = kd  # Derivative gain (미분 항)

        # 적분 항 제한값 (anti-windup용)
        self.i_limit = i_limit

        # 적분 항 누적값 (차원별)
        self.i = [0.0] * dim

        # 이전 오차값 (미분 계산용)
        self.prev_e = [0.0] * dim

        # 첫 step 호출 여부 플래그
        # 첫 호출 시에는 미분항을 계산하지 않기 위해 사용
        self.init = False

    def step(self, e: Sequence[float], dt: float) -> List[float]:
        """
        PID 제어를 한 step 수행한다.

        Parameters
        ----------
        e : Sequence[float]
            dim 차원의 오차 벡터.
            예) [cross_track_error, 0.0]  (조향만 제어할 경우)

        dt : float
            제어 주기(초 단위).
            이전 step 이후 경과 시간.

        Returns
        -------
        u : List[float]
            dim 차원의 제어 출력 벡터.
        """

        # 입력 오차를 float 리스트로 변환 (안전성 확보)
        e = [float(x) for x in e]

        # 첫 step에서는 이전 오차가 없으므로
        # 현재 오차를 그대로 prev_e로 초기화
        if not self.init:
            self.prev_e = e[:]
            self.init = True

        # -------------------------
        # I (Integral) term
        # -------------------------
        # 적분 항 누적: i = ∫ e dt
        # anti-windup: i_limit 범위로 클램핑
        for d in range(self.dim):
            self.i[d] += e[d] * dt
            self.i[d] = max(-self.i_limit, min(self.i_limit, self.i[d]))

        # -------------------------
        # D (Derivative) term
        # -------------------------
        # 미분 항: de/dt
        # dt가 0일 경우 분모 보호
        de = [
            (e[d] - self.prev_e[d]) / dt if dt > 0 else 0.0
            for d in range(self.dim)
        ]

        # 다음 step을 위해 현재 오차를 prev_e로 저장
        self.prev_e = e[:]

        # -------------------------
        # PID 출력 계산
        # -------------------------
        # u = Kp * e + Ki * ∫e dt + Kd * de/dt
        u = [
            self.kp * e[d] +
            self.ki * self.i[d] +
            self.kd * de[d]
            for d in range(self.dim)
        ]

        return u

def compute_e_perp_nd(p, path, seg_i, q):
    """
    p: 현재 위치(dim)
    path[seg_i] -> path[seg_i+1] 선분의 접선으로 e를 분해
    q: 최근접점
    """
    a = [float(x) for x in path[seg_i]]
    b = [float(x) for x in path[seg_i+1]]
    p = [float(x) for x in p]
    q = [float(x) for x in q]

    ab = sub(b, a)
    ab_len = norm(ab)
    if ab_len == 0:
        return sub(p, q)  # 선분 길이 0이면 그냥 오차벡터

    t_hat = mul(ab, 1.0/ab_len)
    e = sub(p, q)
    e_par = mul(t_hat, dot(e, t_hat))
    e_perp = sub(e, e_par)
    return e_perp

def follow_path_pid(robot, smooth_path, pid: PIDVec, *, v_ref: float = 1.0, dt_ns: int = 10000000):
    """
    robot: ins_acc_mov를 가진 객체
    smooth_path: smooth_path_nd의 출력
    pid: 벡터 PID
    v_ref: 원하는 전진(혹은 진행) 속도 크기
    dt_ns: 제어 주기(ns)
    """
    dt = dt_ns * 1e-9  # ns -> s

    # 1) 현재 위치 얻기 (예: robot.pose()나 robot.state()에 맞춰 수정)
    p = robot.position()  # dim 벡터라고 가정

    # 2) 경로까지 최소거리/최근접점 계산 (너가 만든 함수)
    dmin, q, seg_i, t = point_to_path_min_distance_nd(p, smooth_path)

    # 3) steering용 오차는 e_perp
    e_perp = compute_e_perp_nd(p, smooth_path, seg_i, q)

    # 4) PID로 steering 변화량(가속 목표 변화량) 생성
    ds = pid.step(e_perp, dt)   # dim 벡터

    # 5) velocity는 “경로 진행”에 대한 별도 제어가 이상적이지만, 간단히는 상수/스케줄
    #    예: 속도는 일정, 조향만 PID로
    dv = [0.0]*len(ds)  # 또는 특정 축만 속도 제어한다면 그 축만 채우기

    # 6) 로봇에 적용 (time은 dt_ns 만큼)
    robot.ins_acc_mov(velocity=dv, steering=ds, time=dt_ns)