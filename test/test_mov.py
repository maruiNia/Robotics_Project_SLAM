import math
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from module.MyRobot import Moblie_robot

def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol

def assert_vec_close(v1, v2, tol=1e-9):
    assert len(v1) == len(v2), f"벡터 길이 다름: {len(v1)} vs {len(v2)}"
    for i, (a, b) in enumerate(zip(v1, v2)):
        assert approx(a, b, tol), f"[{i}] {a} != {b} (tol={tol})"

def print_last(log, title=""):
    last = log[-1]
    print(f"\n== {title} ==")
    print("t_ns:", last["t_ns"])
    print("pos :", last["position"])
    print("vel :", last["velocity"])
    print("ste :", last["steering"])
    print("ori :", last["orientation"])
    print("note:", last["note"])


def test_set_mov_basic():
    r = Moblie_robot(dim=2, update_rate=1000)
    r.ins_set_mov(position=(1, 2), velocity=(3, 4), steering=(0.1, 0.2))
    log = r.start()

    st = log[-1]
    assert_vec_close(st["position"], (1.0, 2.0))
    assert_vec_close(st["velocity"], (3.0, 4.0))
    assert_vec_close(st["steering"], (0.1, 0.2))

    print_last(log, "test_set_mov_basic PASS")


def test_acc_velocity_linear_example():
    # 예시: update_rate=500ns, time=1000ns, dv=(1,0), 초기 v=(0,0)
    r = Moblie_robot(dim=2, update_rate=500, position=(0, 0), velocity=(0, 0), steering=(0, 0))
    r.ins_acc_mov(velocity=(1, 0), time=1000)
    log = r.start()

    # start 스냅샷 포함이라 로그가 여러 개
    # acc 동안 2스텝(500ns, 500ns)
    # 첫 acc 스텝 끝: v=(0.5, 0)
    # 둘째 acc 스텝 끝: v=(1.0, 0)
    acc_entries = [x for x in log if x["note"] == "acc"]
    assert len(acc_entries) == 2, f"acc 스텝 수가 예상과 다름: {len(acc_entries)}"

    assert_vec_close(acc_entries[0]["velocity"], (0.5, 0.0))
    assert_vec_close(acc_entries[1]["velocity"], (1.0, 0.0))

    print_last(log, "test_acc_velocity_linear_example PASS")


def test_acc_steering_updates_orientation():
    # steering을 각속도(rad/s)로 보고 orientation에 적분되는지 테스트
    # update_rate=1000ns(=1e-6s), time=3000ns => 3스텝
    r = Moblie_robot(dim=1, update_rate=1000, position=(0,), velocity=(0,), steering=(0,))
    r.ins_acc_mov(steering=(3.0,), time=3000)  # 최종 steering이 3rad/s가 되도록 선형 증가
    log = r.start()

    acc_entries = [x for x in log if x["note"] == "acc"]
    assert len(acc_entries) == 3

    # 각 스텝 끝의 steering:
    # t=1000/3000 => 1.0 rad/s
    # t=2000/3000 => 2.0 rad/s
    # t=3000/3000 => 3.0 rad/s
    assert_vec_close(acc_entries[0]["steering"], (1.0,))
    assert_vec_close(acc_entries[1]["steering"], (2.0,))
    assert_vec_close(acc_entries[2]["steering"], (3.0,))

    # orientation 적분(Euler): ori += steering * dt
    # dt=1e-6s
    # ori = 1*1e-6 + 2*1e-6 + 3*1e-6 = 6e-6 rad
    final_ori = log[-1]["orientation"][0]
    assert approx(final_ori, 6e-6, tol=1e-12), f"orientation={final_ori}, expected=6e-6"

    print_last(log, "test_acc_steering_updates_orientation PASS")


def test_position_integrates_velocity():
    # velocity를 set해서 position이 dt만큼 이동하는지 테스트
    # update_rate=1000ns(1e-6s)로 5번 step을 만들려면 acc명령을 time=5000ns로 주고,
    # dv=(0,0)이라 velocity는 변하지 않게 만들자.
    r = Moblie_robot(dim=2, update_rate=1000, position=(0, 0), velocity=(2, 0), steering=(0, 0))
    r.ins_acc_mov(velocity=(0, 0), time=5000)  # 5스텝 동안 velocity 유지하며 적분만 발생
    log = r.start()

    # 이동거리 = v * T,  T = 5000ns = 5e-6s
    # x = 2 * 5e-6 = 1e-5
    final_pos = log[-1]["position"]
    assert_vec_close(final_pos, (1e-5, 0.0), tol=1e-12)

    print_last(log, "test_position_integrates_velocity PASS")


def test_dim_mismatch_raises():
    r = Moblie_robot(dim=2, update_rate=1000)
    try:
        r.ins_set_mov(position=(1, 2, 3))  # dim mismatch
        assert False, "dim mismatch인데 예외가 안 났음"
    except ValueError:
        pass

    try:
        r.ins_acc_mov(velocity=(1,))  # dim mismatch
        assert False, "dim mismatch인데 예외가 안 났음"
    except ValueError:
        pass

    print("\n== test_dim_mismatch_raises PASS ==")


def run_all_tests():
    test_set_mov_basic()
    test_acc_velocity_linear_example()
    test_acc_steering_updates_orientation()
    test_position_integrates_velocity()
    test_dim_mismatch_raises()
    print("\n🎉 ALL TESTS PASSED 🎉")


if __name__ == "__main__":
    run_all_tests()
