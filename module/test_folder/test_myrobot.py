# module/test_folder/test_myrobot.py
import math
import importlib

# 1) MyRobot가 import할 compute_pid_control이 MyPlanning에 없으므로
#    테스트에서 먼저 주입해주고 MyRobot를 import한다.
import module.MyPlanning as MyPlanning

if not hasattr(MyPlanning, "compute_pid_control"):
    def compute_pid_control(*args, **kwargs):
        """
        MyRobot.py가 요구하는 이름을 테스트에서 임시로 제공.
        실제 로봇 주행 PID 제어는 follow_path_pid 등을 쓰면 되지만,
        여기서는 import 에러 방지 + 단위 테스트 목적이라 '더미'로 둔다.
        """
        return None

    MyPlanning.compute_pid_control = compute_pid_control

# 주입 후 MyRobot import
import module.MyRobot as MyRobot
importlib.reload(MyRobot)  # 혹시 캐시된 import가 있으면 반영

from module.MyRobot import Moblie_robot
from module.MySlam import SlamMap, Obstacle_obj
from module.MySensor import Circle_Sensor


def _find_log_at_t(log, t_ns: int):
    for row in log:
        if row.get("t_ns") == t_ns:
            return row
    return None


def test_static_acc_linear_velocity_steps():
    """
    MyRobot._apply_acc는 update_rate 단위로 나눠서 선형으로 속도를 증가시킴.
    update_rate=500ns, time=1000ns, dv=(1,0) 이면
      t=500ns -> vx=0.5
      t=1000ns -> vx=1.0
    """
    r = Moblie_robot(dim=2, update_rate=500, dynamic_operation_mode=False)

    r.ins_set_mov(position=(0, 0), velocity=(0, 0), steering=(0, 0))
    r.ins_acc_mov(velocity=(1, 0), time=1000)

    log = r.start()

    row_500 = _find_log_at_t(log, 500)
    row_1000 = _find_log_at_t(log, 1000)
    assert row_500 is not None and row_1000 is not None

    assert math.isclose(row_500["velocity"][0], 0.5, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(row_1000["velocity"][0], 1.0, rel_tol=1e-12, abs_tol=1e-12)


def test_static_acc_position_integration():
    """
    _apply_acc는 각 스텝에서 velocity를 갱신한 뒤 _step(dt)로 position을 적분.
    위 테스트와 같은 조건에서,
      dt = 500ns = 5e-7 s
      step1: vx=0.5 -> dx=0.5 * 5e-7 = 2.5e-7
      step2: vx=1.0 -> dx=1.0 * 5e-7 = 5.0e-7
      total dx = 7.5e-7
    """
    r = Moblie_robot(dim=2, update_rate=500, dynamic_operation_mode=False)
    r.ins_set_mov(position=(0, 0), velocity=(0, 0), steering=(0, 0))
    r.ins_acc_mov(velocity=(1, 0), time=1000)

    log = r.start()
    row_1000 = _find_log_at_t(log, 1000)
    assert row_1000 is not None

    x = row_1000["position"][0]
    assert math.isclose(x, 7.5e-7, rel_tol=1e-9, abs_tol=1e-12)


def test_circle_sensor_raycast_hits_obstacle():
    """
    Circle_Sensor는 angle=0 방향( +x ) 레이를 쏨.
    로봇이 (0,0)에 있고, 장애물이 x∈[1,2], y∈[-0.5,0.5]이면
    첫 hit는 대략 d≈1.0 근처가 나와야 함.
    """
    m = SlamMap(dim=2)
    m.add(Obstacle_obj(dim=2, min_corner=(1.0, -0.5), max_corner=(2.0, 0.5)))

    sensor = Circle_Sensor(dim=2, number=8, distance=5.0, slam_map=m, step=0.05)
    results = sensor.sensing((0.0, 0.0))

    d0 = results[0]  # angle=0
    assert d0 is not None
    assert abs(d0 - 1.0) <= 0.06  # step=0.05라서 오차 허용


def test_is_collision_true_when_close_enough():
    """
    is_collision은 sensing 결과 중 dist <= collision_dist 이면 True.
    Circle_Sensor는 d를 step부터 검사하므로,
    장애물 내부에 있어도 최소 반환은 step(예:0.05)일 수 있음.
    그래서 collision_dist를 step보다 살짝 크게 잡아야 안정적으로 True.
    """
    m = SlamMap(dim=2)
    m.add(Obstacle_obj(dim=2, min_corner=(0.0, 0.0), max_corner=(1.0, 1.0)))

    sensor = Circle_Sensor(dim=2, number=16, distance=5.0, slam_map=m, step=0.05)

    r = Moblie_robot(
        dim=2,
        position=(0.5, 0.5),      # 장애물 내부
        velocity=(0, 0),
        steering=(0, 0),
        sensing_mode=True,
        dynamic_operation_mode=False,
        update_rate=1000,
        sensor=sensor,
        collision_dist=0.051,     # step(0.05)보다 약간 크게
    )

    assert r.is_collision() is True


def test_is_collision_false_when_far():
    m = SlamMap(dim=2)
    m.add(Obstacle_obj(dim=2, min_corner=(10.0, 10.0), max_corner=(11.0, 11.0)))

    sensor = Circle_Sensor(dim=2, number=16, distance=5.0, slam_map=m, step=0.05)

    r = Moblie_robot(
        dim=2,
        position=(0.0, 0.0),
        velocity=(0, 0),
        steering=(0, 0),
        sensing_mode=True,
        dynamic_operation_mode=False,
        update_rate=1000,
        sensor=sensor,
        collision_dist=0.051,
    )

    assert r.is_collision() is False
