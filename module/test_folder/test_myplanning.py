import math
import pytest

from module.MyPlanning import (
    smooth_path_nd,
    point_to_segment_distance_nd,
    point_to_path_min_distance_nd,
    PIDVec,
    compute_e_perp_nd,
    follow_path_pid,
)

EPS = 1e-9


def make_L_path():
    # (0,0)->(5,0)->(5,5)
    path = []
    for x in range(6):
        path.append([x, 0])
    for y in range(1, 6):
        path.append([5, y])
    return path


# -------------------------
# smooth_path_nd
# -------------------------
def test_smooth_path_empty():
    assert smooth_path_nd([]) == []


def test_smooth_path_single_point():
    out = smooth_path_nd([[1, 2]])
    assert out == [[1.0, 2.0]]


def test_smooth_path_dim_mismatch_raises():
    with pytest.raises(ValueError):
        smooth_path_nd([[0, 0], [1, 2, 3]])


def test_smooth_path_fix_ends_keeps_endpoints():
    path = make_L_path()
    out = smooth_path_nd(path, alpha=0.1, beta=0.3, tol=1e-10, max_iter=5000, fix_ends=True)

    assert len(out) == len(path)
    assert out[0] == [float(x) for x in path[0]]
    assert out[-1] == [float(x) for x in path[-1]]


def test_smooth_path_changes_inner_points_for_corner():
    path = make_L_path()
    out = smooth_path_nd(path, alpha=0.1, beta=0.3, tol=1e-10, max_iter=5000, fix_ends=True)

    # 코너 주변 점이 그대로면 스무딩 의미가 없음 → 보통 y가 0에서 살짝 떠야 함
    # (5,0)은 path[5]
    assert out[5] != [5.0, 0.0]


# -------------------------
# point_to_segment_distance_nd
# -------------------------
def test_point_to_segment_distance_basic():
    d, q, t = point_to_segment_distance_nd(p=[5, 3], a=[0, 0], b=[10, 0])
    assert abs(d - 3.0) < 1e-9
    assert abs(q[0] - 5.0) < 1e-9 and abs(q[1] - 0.0) < 1e-9
    assert abs(t - 0.5) < 1e-9


def test_point_to_segment_distance_clamps_to_a():
    d, q, t = point_to_segment_distance_nd(p=[-5, 0], a=[0, 0], b=[10, 0])
    assert abs(d - 5.0) < 1e-9
    assert q == [0.0, 0.0]
    assert abs(t - 0.0) < 1e-9


def test_point_to_segment_distance_clamps_to_b():
    d, q, t = point_to_segment_distance_nd(p=[15, 0], a=[0, 0], b=[10, 0])
    assert abs(d - 5.0) < 1e-9
    assert q == [10.0, 0.0]
    assert abs(t - 1.0) < 1e-9


def test_point_to_segment_distance_zero_length_segment():
    d, q, t = point_to_segment_distance_nd(p=[1, 1], a=[0, 0], b=[0, 0])
    assert abs(d - math.sqrt(2)) < 1e-9
    assert q == [0.0, 0.0]
    assert abs(t - 0.0) < 1e-9


# -------------------------
# point_to_path_min_distance_nd
# -------------------------
def test_point_to_path_min_distance_raises_on_empty():
    with pytest.raises(ValueError):
        point_to_path_min_distance_nd([0, 0], [])


def test_point_to_path_min_distance_single_point_path():
    d, q, seg_i, t = point_to_path_min_distance_nd([3, 4], [[0, 0]])
    assert abs(d - 5.0) < 1e-9
    assert q == [0.0, 0.0]
    assert seg_i == 0
    assert abs(t - 0.0) < 1e-9


def test_point_to_path_min_distance_L_shape():
    path = make_L_path()
    # (5,2) 근처면 세로 구간에 붙어야 함: 최근접점 q는 x=5, y≈2
    d, q, seg_i, t = point_to_path_min_distance_nd([4.0, 2.0], path)
    assert abs(d - 1.0) < 1e-9
    assert abs(q[0] - 5.0) < 1e-9
    assert abs(q[1] - 2.0) < 1e-6  # 보간/세그먼트 선택에 따라 아주 미세 오차 허용


# -------------------------
# PIDVec
# -------------------------
def test_pidvec_first_step_derivative_is_zero_like():
    pid = PIDVec(dim=2, kp=2.0, ki=0.0, kd=10.0)
    u = pid.step([1.0, -1.0], dt=0.1)

    # 첫 step은 prev_e를 e로 세팅하므로 D항이 0으로 나와야 함
    assert u == [2.0, -2.0]


def test_pidvec_integral_clamps():
    pid = PIDVec(dim=1, kp=0.0, ki=1.0, kd=0.0, i_limit=0.5)

    # e=1, dt=1이면 적분 i는 1이지만 i_limit=0.5로 clamp
    u1 = pid.step([1.0], dt=1.0)
    assert abs(u1[0] - 0.5) < 1e-9

    # 계속 더해도 i는 0.5에 고정
    u2 = pid.step([1.0], dt=10.0)
    assert abs(u2[0] - 0.5) < 1e-9


# -------------------------
# compute_e_perp_nd
# -------------------------
def test_compute_e_perp_nd_horizontal_segment():
    # 선분 a(0,0)->b(10,0), 최근접점 q(5,0), 현재점 p(5,3)
    p = [5.0, 3.0]
    path = [[0.0, 0.0], [10.0, 0.0]]
    seg_i = 0
    q = [5.0, 0.0]

    e_perp = compute_e_perp_nd(p, path, seg_i, q)

    # 접선 방향(x축) 성분 제거되면 (0,3)만 남아야 함
    assert abs(e_perp[0] - 0.0) < 1e-9
    assert abs(e_perp[1] - 3.0) < 1e-9


# -------------------------
# follow_path_pid
# -------------------------
class DummyRobot:
    """follow_path_pid가 요구하는 최소 인터페이스만 구현."""
    def __init__(self, pos):
        self._pos = list(pos)
        self.last_call = None

    def position(self):
        return self._pos

    def ins_acc_mov(self, *, velocity, steering, time):
        self.last_call = {"velocity": velocity, "steering": steering, "time": time}


def test_follow_path_pid_calls_robot_with_expected_shapes():
    robot = DummyRobot(pos=[5.0, 3.0])
    path = [[0.0, 0.0], [10.0, 0.0]]  # x축 직선
    pid = PIDVec(dim=2, kp=1.0, ki=0.0, kd=0.0)

    follow_path_pid(robot, path, pid, dt_ns=10_000_000)  # 0.01s

    assert robot.last_call is not None
    assert robot.last_call["time"] == 10_000_000

    # dv는 ds 길이와 같게 0으로 채워짐
    assert robot.last_call["velocity"] == [0.0, 0.0]

    # e_perp는 (0,3) => kp=1이면 steering = (0,3)
    st = robot.last_call["steering"]
    assert abs(st[0] - 0.0) < 1e-9
    assert abs(st[1] - 3.0) < 1e-9
