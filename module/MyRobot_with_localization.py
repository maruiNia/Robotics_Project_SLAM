from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union, Dict, Any
import math

from .MySensor import Circle_Sensor
from .MyPlanning import PIDVec, point_to_path_min_distance_nd, smooth_path_nd
from .MyControl import MotionModel, SimpleBicycleModel
from .MyUtils import _vec, _check_dim, _wrap_pi
from .MySlam import SlamMap
from .MyNavi import astar_find_path_center_nodes, euclidean

from collections import deque
import math
import random  # ✅ (추가) 센서 측정에 가우시안 노이즈 넣을 때 사용

import numpy as np  # 파일 상단에 없으면 추가


Number = Union[int, float]

def _check_ns(ns: int, name: str):
    if not isinstance(ns, int) or ns <= 0:
        raise ValueError(f"{name}는 1 이상의 정수(ns)여야 합니다.")

@dataclass
class _Command:
    kind: str                                         # "set" or "acc"
    position: Optional[Tuple[float, ...]] = None        # 현 위치
    est_position: Optional[Tuple[float, ...]] = None    # 추정 위치
    velocity: Optional[Tuple[float, ...]] = None        # 현 속도
    steering: Optional[Tuple[float, ...]] = None        # 현 조향
    sensing: Optional[Tuple[float, ...]] = None         # 센서 측정값
    pathing: Optional[Tuple[float, ...]] = None         # 계산한 path
    time_ns: int = 0                                  # acc에서만 사용

@dataclass
class _ActiveAcc:
    dv: Tuple[float, ...]          # 총 변화량(Δv)
    ds: Tuple[float, ...]          # 총 변화량(Δsteering)
    total_ns: int                  # 전체 지속시간
    elapsed_ns: int                # 누적 진행시간
    v0: Tuple[float, ...]          # 시작 velocity
    s0: Tuple[float, ...]          # 시작 steering

class Moblie_robot:
    """
    맵에서 slam을 동작할 로봇의 객체이다.

    상태(단위):
      - position: dim차원 좌표 (m로 가정)
      - velocity: dim차원 속도 벡터 (m/s)
      - steering: dim차원 조향(각속도) 벡터 (rad/s)  (간단히 orientation을 적분)
      - update_rate: 상태 갱신 주기 (ns)

    동작:
      - ins_set_mov: 즉시 상태 설정(입력 None이면 유지)
      - ins_acc_mov: 주어진 시간 동안 목표 가속을 걸어 velocity/steering을 선형으로 변화시킴
        예시처럼 time < update_rate인 경우에도 '부분 스텝'으로 처리 가능하도록 구현.
      - start: 누적된 명령들을 순서대로 실행하며 로그 반환
    """

    def __init__(
        self,
        dim: int,
        position: Optional[Sequence[Number]] = None,
        velocity: Optional[Sequence[Number]] = None,
        steering: Optional[Sequence[Number]] = None,
        dynamic_operation_mode : bool = False,
        sensing_mode : bool = False,
        update_rate: int = 1000,
        sensor: Optional[Circle_Sensor] = None,
        collision_dist : float = 1e-6,
        error_control : PIDVec = None, # 오차 제어용 PID(기본값 내부 구현)
        end_point : Optional[Tuple[Number, ...]] = None,
        ):
        _check_dim(dim)
        _check_ns(update_rate, "update_rate")

        self.dim: int = dim
        self._position: Tuple[float, ...] = _vec(dim, position, "position")
        
        # --- Localization / state-estimation (sensor-based) ---
        # _position: 'true/simulated' state (motion model updates this)
        # _pos_est:  'estimated' state used by planning/control when sensing_mode=True
        # ✅ _position/_orientation  : true(simulated) state (motion model integrates this)
        # ✅ _pos_est/_ori_est       : estimated state used by planning/control when enabled
        
        self._pos_est: Tuple[float, ...] = self._position
        self._ori_est: Tuple[float, ...] = tuple(0.0 for _ in range(dim))  # ✅ (추가) 추정 heading/방향

        self._use_estimated_pose: bool = True  # get_position/get_orientation에서 est를 쓸지

        # (Localization schedule / mode)
        self._loc_enabled: bool = True         # ✅ 로컬라이제이션 전체 ON/OFF
        self._loc_use_sensor_update: bool = True  # ✅ N틱마다 센싱+보정(update) 할지
        self._loc_predict_only: bool = False   # ✅ True면 update 없이 predict만 (디버깅용)

        self._fake_fast_sensing: bool = None  # ✅ True면 현 위치에서 가우시안 분포 노이즈 추가 
        self._fake_fast_sensing_sigma: float = 0.5  # ✅ 가우시안 노이즈 표준편차

        # (Grid matching params)
        self._loc_resolution: float = 1.0
        self._loc_sigma: float = 0.5
        self._loc_period: int = 10             # ✅ 기본을 10틱으로(원하면 set_localization_params로 바꾸기)

        # (Measurement noise; 나중에 칼만/파티클로 갈 때 그대로 확장 가능)
        self._meas_noise_enabled: bool = False
        self._meas_noise_sigma: float = 0.0    # 가우시안 표준편차(거리 단위)

        #충돌 감지
        self._collision_detected: bool = False

        self._tick_count: int = 0

        self._velocity: Tuple[float, ...] = _vec(dim, velocity, "velocity")
        self._steering: Tuple[float, ...] = _vec(dim, steering, "steering")
        self._sensor: Optional[Circle_Sensor] = sensor
        self._collision_dist: float = collision_dist
        self._dynamic_operation_mode : bool = dynamic_operation_mode
        self._sensing_mode : bool = sensing_mode 
        self._active_acc: Optional[_ActiveAcc] = None # 동적 모드에서 "진행중인 acc 명령"을 들고 있을 상태
        self._last_sensing: Optional[Tuple[float, ...]] = None # 마지막 센싱(로그/디버깅용)
        self.pid_control : PIDVec = PIDVec(dim = dim, kp = 1.0, ki = 0.0, kd = 0.1) if error_control is None else error_control
        self._motion_model : Optional[MotionModel] = None #운동 모델
        self._end_point : Optional[Tuple[Number, ...]] = _vec(dim, end_point, "end_point") if end_point is not None else None

        # replan 경로 블렌딩용
        self._prev_replan_path = None        # 이전 tick에서 사용한 path
        self._prev_replan_seg = 0            # 이전 tick의 진행 seg index(뒤로 점프 방지)

        # 동적 명령 처리
        self._path = None #추적할 경로


        # --- EKF on/off ---
        self._ekf_enabled = True
        # 상태 x = [x, y, theta]
        self._ekf_x = np.array([self._pos_est[0], self._pos_est[1], self._ori_est[0]], dtype=float)
        # 공분산 P
        self._ekf_P = np.diag([0.5**2, 0.5**2, (5.0 * math.pi/180.0)**2])  # 초기 불확실성(예시)
        # 프로세스 노이즈 Q (모델 예측이 얼마나 틀릴 수 있나)
        self._ekf_Q = np.diag([0.02**2, 0.02**2, (1.0 * math.pi/180.0)**2])
        # 측정 노이즈 R (그리드매칭 측정이 얼마나 시끄럽나)
        self._ekf_R = np.diag([0.50**2, 0.50**2])  # (x,y)만 측정한다고 가정


        # orientation은 "방향"이니까 steering을 적분해서 얻는 값으로 하나 둠(벡터로)
        self._orientation: Tuple[float, ...] = tuple(0.0 for _ in range(dim))

        self._update_rate_ns: int = update_rate
        self._time_ns: int = 0

        self._queue: List[_Command] = []
        self._log: List[Dict[str, Any]] = []

    # ----------------------
    # enablers
    # ----------------------
    def enable_estimated_pose(self, enable: bool = True) -> None:   
        """planning/control에서 추정 위치를 쓸지 선택"""
        self._use_estimated_pose = bool(enable)

    def enable_localization(self, enable: bool = True) -> None:
        """✅ 로컬라이제이션 전체 ON/OFF"""
        self._loc_enabled = bool(enable)

    def enable_sensor_update(self, enable: bool = True) -> None:
        """
        ✅ N틱마다 센싱+보정(update) ON/OFF
        - False면: 센서는 안 쓰고 predict(제어 적분)로만 추정 유지
        """
        self._loc_use_sensor_update = bool(enable)

    def enable_ekf(self, enable: bool = True) -> None:
        """EKF 사용 on/off"""
        self._ekf_enabled = bool(enable)
        
    # ----------------------
    # getters
    # ----------------------
    def get_update_rate(self) -> int:
        return self._update_rate_ns

    def get_position(self) -> Tuple[float, ...]:
        """현재 위치 반환.
        - sensing_mode=True AND enable_estimated_pose(True)이면 추정 위치(_pos_est)
        - 그 외에는 실제/시뮬레이션 위치(_position)
        """
        if self._sensing_mode and self._use_estimated_pose:
            # print("real position:", self._position, "-> estimated position:", self._pos_est)
            return self._pos_est
        return self._position

    def get_true_position(self) -> Tuple[float, ...]:
        """시뮬레이션(모션모델) 기준 실제 상태"""
        return self._position

    def get_estimated_position(self) -> Tuple[float, ...]:
        """센서 기반 추정 상태"""
        return self._pos_est

    def get_velocity(self) -> Tuple[float, ...]:
        return self._velocity

    def get_steering(self) -> Tuple[float, ...]:
        return self._steering

    def get_state(self) -> Dict[str, Tuple[float, ...]]:
        return {
            "position": self._position,
            "velocity": self._velocity,
            "steering": self._steering,
            "orientation": self._orientation,
        }
    
    def get_orientation(self) -> Tuple[float, ...]:
        """
        현재 orientation 반환.
        - sensing_mode=True AND enable_estimated_pose(True)이면 추정 orientation(_ori_est)
        - 그 외에는 실제/시뮬레이션 orientation(_orientation)
        """
        if self._sensing_mode and self._use_estimated_pose:
            return self._ori_est
        return self._orientation
    
    def get_end_point(self) -> Optional[Tuple[float, ...]]:
        return self._end_point
    # ----------------------
    # setters
    # ----------------------
    def set_sensor(self, sensor: Circle_Sensor) -> None:
        self._sensor = sensor
    
    def set_path(self, path):
        """smooth_path_nd 결과를 저장."""
        self._path = path

    def set_pid(self, pid, *, v_ref: float = 1.0, control_dt_ns: Optional[int] = None):
        """PIDVec, 속도 목표, 제어주기 설정."""
        self.pid_control = pid
        self._v_ref = float(v_ref)
        if control_dt_ns is not None:
            self._control_dt_ns = int(control_dt_ns)

    def set_motion_model(self, model: Optional[MotionModel]) -> None:
        """조향/이동 갱신을 담당할 motion model을 설정. None이면 기존 Euler 방식."""
        self._motion_model = model

    def set_localization_params(self, *, resolution: float = 1.0, sigma: float = 0.5, period: int = 5) -> None:
        """그리드 기반 로컬라이제이션 파라미터."""
        if resolution <= 0:
            raise ValueError("resolution은 0보다 커야 합니다.")
        if sigma <= 0:
            raise ValueError("sigma는 0보다 커야 합니다.")
        if period <= 0:
            raise ValueError("period는 1 이상이어야 합니다.")
        self._loc_resolution = float(resolution)
        self._loc_sigma = float(sigma)
        self._loc_period = int(period)

    def set_fake_fast_sensing(self, enable: bool = True, sigma: float = 0.5) -> None:
        """✅ 페이크 빠른 센싱 모드 설정.
        - enable=True면, 센서 없이 현 위치에 가우시안 노이즈 추가한 값으로 센싱 흉내냄
        - sigma: 가우시안 표준편차(거리 단위)
        """
        self._fake_fast_sensing = bool(enable)
        self._fake_fast_sensing_sigma = float(sigma)
    # ----------------------
    # instruction APIs
    # ----------------------
    def ins_set_mov(
        self,
        position: Optional[Sequence[Number]] = None,
        velocity: Optional[Sequence[Number]] = None,
        steering: Optional[Sequence[Number]] = None,
    ) -> None:
        """즉시 설정 명령을 큐에 넣음. None은 유지."""
        # 동적 명령 처리 모드에서는 센싱값도 포함
        sensing = None
        if self._sensing_mode: # 센싱 모드가 켜져 있다면 활성화
            sensing = self.sensing()

        cmd = _Command(
            kind="set",
            position=None if position is None else _vec(self.dim, position, "position"),
            velocity=None if velocity is None else _vec(self.dim, velocity, "velocity"),
            steering=None if steering is None else _vec(self.dim, steering, "steering"),
            sensing=None if sensing is None else _vec(self.dim, sensing, "sensing"),
        )

        #큐에 등록
        self._queue.append(cmd)
        
    def ins_acc_mov(
        self,
        velocity: Optional[Sequence[Number]] = None,
        steering: Optional[Sequence[Number]] = None,
        time: int = 1000,
    ) -> None:
        """
        주어진 time(ns) 동안 velocity/steering을 '가속'시킴.
        여기서 velocity/steering 인자는 '최종적으로 더해질 변화량'이 아니라,
        **time 동안 선형으로 도달할 목표 변화량**으로 처리:
          - 현재 v에서 v + velocity 까지 time동안 선형 증가
          - 현재 s에서 s + steering 까지 time동안 선형 증가

        예시:
          update_rate=500ns, time=1000ns, velocity=1(m/s 증가 목표), 초기 v=0이면
            0 -> (500ns) 0.5 -> (1000ns) 1.0  처럼 선형으로 증가
        """

        #센싱 모드
        sensing = None
        if self._sensing_mode:
            sensing = self.sensing()
        
        _check_ns(time, "time")
        cmd = _Command(
            kind="acc",
            velocity=None if velocity is None else _vec(self.dim, velocity, "velocity"),
            steering=None if steering is None else _vec(self.dim, steering, "steering"),
            sensing=None if sensing is None else _vec(self.dim, sensing, "sensing"),
            time_ns=time,
        )
        #명령 등록
        self._queue.append(cmd)
            
    def sensing(self) -> Any:
        """센서로 현재 위치에서 주변 환경을 센싱."""
        if not hasattr(self, '_sensor') or self._sensor is None:
            raise RuntimeError("센서가 설정되지 않았습니다.")
        
        return self._sensor.sensing(self._position)

    def _begin_acc(self, cmd: _Command) -> None:
        """acc 명령을 '진행중' 상태로 등록한다. (동적 모드에서 사용)"""
        dv = cmd.velocity if cmd.velocity is not None else tuple(0.0 for _ in range(self.dim))
        ds = cmd.steering if cmd.steering is not None else tuple(0.0 for _ in range(self.dim))

        self._active_acc = _ActiveAcc(
            dv=dv,
            ds=ds,
            total_ns=cmd.time_ns,
            elapsed_ns=0,
            v0=self._velocity,
            s0=self._steering,
        )

    def _tick(self, dt_ns: int, *, note: str = "tick") -> None:
        """
        ✅ tick 기반 '전체 프레임' 함수.
        한 번 호출 = 한 tick.
        순서:
        1) tick_count 증가
        2) (필요시) acc 진행으로 v/steer 갱신
        3) 실제 상태 적분(_step)
        4) 센싱(매 tick)
        5) 추정(predict 매 tick) + 보정(update는 주기적으로)
        6) snapshot 기록
        """
        self._tick_count += 1
        dt_s = dt_ns * 1e-9

        # ----------------------
        # 0) acc 진행 처리 (동적 모드에서만 의미)
        # ----------------------
        if self._active_acc is not None:
            a = self._active_acc
            total = a.total_ns
            dt = min(dt_ns, total - a.elapsed_ns)

            t_next = a.elapsed_ns + dt
            ratio = t_next / total

            self._velocity = tuple(a.v0[i] + a.dv[i] * ratio for i in range(self.dim))
            self._steering = tuple(a.s0[i] + a.ds[i] * ratio for i in range(self.dim))

            a.elapsed_ns += dt
            if a.elapsed_ns >= total:
                self._active_acc = None

            # acc에서는 dt_ns 대신 dt로 실제 적분해야 함
            dt_ns = dt
            dt_s = dt_ns * 1e-9
            note = "acc_step"

        # ----------------------
        # 1) 실제 상태 적분 (true state)
        # ----------------------
        self._step(dt_ns)

        # ----------------------
        # 2) 센싱은 매 tick마다 (true position 기준)
        # ----------------------
        if self._sensing_mode and (self._sensor is not None):
            sensing = self._sensor.sensing(self._position)
            sensing = self._apply_measurement_noise(sensing)
            # self._last_sensing = _vec(self.dim, sensing, "sensing")  # type: ignore
            self._last_sensing = sensing  # type: ignore
        else:
            self._last_sensing = None

        # ----------------------
        # 3) 추정 상태(est state): predict는 매 tick, update는 주기적으로
        # ----------------------
        if self._sensing_mode and self._loc_enabled:
            # (1) Predict: dead-reckoning
            if self._motion_model is not None:
                est_pos, est_ori = self._motion_model.step(
                    position=self._pos_est,
                    orientation=self._ori_est,
                    velocity=self._velocity,
                    steering=self._steering,
                    dt_s=dt_s,
                )
                self._pos_est = est_pos
                self._ori_est = est_ori
            else:
                self._pos_est = tuple(self._pos_est[i] + self._velocity[i] * dt_s for i in range(self.dim))
                self._ori_est = tuple(self._ori_est[i] + self._steering[i] * dt_s for i in range(self.dim))

            # (2) Update: N tick마다만 센서로 보정
            if (not self._loc_predict_only) and self._loc_use_sensor_update and (self._tick_count % self._loc_period == 0):
                if self._last_sensing is not None and self._sensor is not None:
                    self._pos_est = self._localize_grid_by_circle_scan(self._last_sensing)

                    # 센싱 후 추정 시 스냅샷 출력
                    print(f"[Tick {self._tick_count}] t={self._time_ns}ns, pos={self._position}, est_pos={self._pos_est}, vel={self._velocity}, steer={self._steering}")

        # 5) 충돌 판정
        # ----------------------
        if self._sensor is not None and self._last_sensing is not None:
            if self.is_collision():
                self._collision_detected = True
                self._snapshot(note="collision_detected")


        # ----------------------
        # 4) snapshot 기록 (요청한 모든 항목이 여기서 한 번에 저장됨)
        # ----------------------

        # 기록
        self._snapshot(note=note)
        


    # ----------------------
    # simulation core
    # ----------------------
    def _snapshot(self, note: str = "", pathing = None) -> None:
        self._log.append(
            {
                "t_ns": self._time_ns,
                "position": self._position,              # true position
                "est_position": self._pos_est,            # ✅ estimated position
                "velocity": self._velocity,
                "steering": self._steering,
                "orientation": self._orientation,
                "est_orientation": self._ori_est,         # ✅ (확장 대비)
                "sensing": self._last_sensing,             # ✅ 마지막 센싱
                "pathing": pathing,
                "note": note,
            }
        )

    def _step(self, dt_ns: int) -> None:
        """
        ✅ dt_ns 만큼 '실제 상태(true state)'만 적분하는 함수.
        - 여기서는 물리 적분 + 시간 증가만 수행한다.
        - 센싱/로컬라이제이션/스냅샷은 _tick()에서 처리한다. (tick 기반 설계)
        """
        dt_s = dt_ns * 1e-9

        # 실제(true) 상태 적분
        if self._motion_model is not None:
            new_pos, new_ori = self._motion_model.step(
                position=self._position,
                orientation=self._orientation,
                velocity=self._velocity,
                steering=self._steering,
                dt_s=dt_s,
            )
            self._position = new_pos
            self._orientation = new_ori
        else:
            # 기존 Euler 적분
            self._position = tuple(self._position[i] + self._velocity[i] * dt_s for i in range(self.dim))
            self._orientation = tuple(self._orientation[i] + self._steering[i] * dt_s for i in range(self.dim))

        # 시간 누적
        self._time_ns += dt_ns

    def step_queue_with_model(self, model, dt_ns: int = None) -> bool:
        """
        애니메이션/실시간용: 큐에서 명령을 '조금만' 처리한다.
        - set: 즉시 적용하고 끝
        - acc: dt_ns만큼만 진행(남은 시간은 커맨드에 남겨둠)
        - 이동 적분은 model(예: SimpleBicycleModel)로 수행
        반환: 이번 호출에서 뭔가 처리했으면 True, 큐가 비었으면 False
        """
        if dt_ns is None:
            dt_ns = self._update_rate_ns

        # 시작 스냅샷
        if not self._log:
            self._snapshot(note="start(model-step)")

        if not self._queue:
            return False

        self.set_motion_model(model)

        cmd = self._queue[0]

        if cmd.kind == "set":
            # set은 즉시 반영 후 pop
            self._apply_set(cmd)
            self._queue.pop(0)
            self._snapshot(note="set")
            return True

        if cmd.kind == "acc":
            # acc는 dt_ns만큼만 잘라서 처리
            remain = cmd.time_ns
            dt = min(dt_ns, remain)

            # 목표 변화량 (없으면 0)
            dv = cmd.velocity if cmd.velocity is not None else tuple(0.0 for _ in range(self.dim))
            ds = cmd.steering if cmd.steering is not None else tuple(0.0 for _ in range(self.dim))

            # acc 시작 시점의 기준(v0,s0)을 커맨드에 저장하지 않았으니,
            # "현재 상태를 v0/s0로 보고" 남은 시간 동안 선형 변화시키는 방식으로 단순화
            # (애니메이션 테스트용으로는 충분히 자연스럽게 움직임)
            v0 = self._velocity
            s0 = self._steering

            total = remain  # 남은 시간 기준으로 비율 계산
            ratio = dt / total if total > 0 else 1.0

            self._velocity = tuple(v0[i] + dv[i] * ratio for i in range(self.dim))
            self._steering = tuple(s0[i] + ds[i] * ratio for i in range(self.dim))

            # ✅ 모델로 pose 갱신
            dt_s = dt * 1e-9
            new_pos, new_ori = self._motion_model.step(
                self._position, self._orientation, self._velocity, self._steering, dt_s
            )
            self._position = new_pos
            self._orientation = new_ori
            self._time_ns += dt
            self._snapshot(note="acc(model-step)")

            cmd.time_ns -= dt
            if cmd.time_ns <= 0:
                self._queue.pop(0)

            return True

        raise RuntimeError(f"알 수 없는 명령: {cmd.kind}")

    def _apply_set(self, cmd: _Command) -> None:
        if cmd.position is not None:
            self._position = cmd.position
        if cmd.velocity is not None:
            self._velocity = cmd.velocity
        if cmd.steering is not None:
            self._steering = cmd.steering
        self._snapshot(note="set")

    def _apply_acc(self, cmd: _Command) -> None:
        total = cmd.time_ns
        dt_base = self._update_rate_ns

        # 목표 변화량 (없으면 0)
        dv = cmd.velocity if cmd.velocity is not None else tuple(0.0 for _ in range(self.dim))
        ds = cmd.steering if cmd.steering is not None else tuple(0.0 for _ in range(self.dim))

        v0 = self._velocity
        s0 = self._steering

        elapsed = 0
        while elapsed < total:
            dt = min(dt_base, total - elapsed)

            # 선형 보간으로 "현재 시점의 목표 v/s"를 만들고 그 값으로 설정 후 step
            #   v(t) = v0 + dv * (t/total)
            #   s(t) = s0 + ds * (t/total)
            # 여기서 t는 elapsed+dt 시점(스텝 끝)을 기준으로 반영
            t_next = elapsed + dt
            ratio = t_next / total

            self._velocity = tuple(v0[i] + dv[i] * ratio for i in range(self.dim))
            self._steering = tuple(s0[i] + ds[i] * ratio for i in range(self.dim))

            elapsed += dt
            self._tick(dt, note="acc_step")

    def is_collision(self) -> bool:
        """현재 위치에서 센서로 충돌 판정."""
        sensing_results = self.sensing()

        # sensing 결과 중 하나라도 0에 매우 가까우면 충돌로 간주
        for dist in sensing_results:
            if dist is not None and dist <= self._collision_dist:
                return True
        return False

    def start(self) -> List[Dict[str, Any]]:
        # 시작 스냅샷
        if not self._log:
            self._snapshot(note="start")

        if not self._dynamic_operation_mode:
            while self._queue:
                cmd = self._queue.pop(0)
                if cmd.kind == "set":
                    self._apply_set(cmd)
                elif cmd.kind == "acc":
                    self._apply_acc(cmd)
                else:
                    raise RuntimeError(f"알 수 없는 명령: {cmd.kind}")
            return self._log

        # -------------------------
        # 동적 명령 처리 모드
        # -------------------------
        dt = self._update_rate_ns

        while True:
            # 1) 진행중 acc가 없고 큐에 명령이 있으면 하나 꺼내서 시작/적용
            if self._active_acc is None and self._queue:
                cmd = self._queue.pop(0)

                if cmd.kind == "set":
                    # set은 즉시 적용(그리고 같은 tick에 움직일지는 정책인데, 여기선 다음 tick에서 이동)
                    self._apply_set(cmd)
                elif cmd.kind == "acc":
                    # acc는 begin만 하고 tick에서 분할 실행
                    self._begin_acc(cmd)
                else:
                    raise RuntimeError(f"알 수 없는 명령: {cmd.kind}")

            # 2) 한 tick 진행(진행중 acc면 acc_step, 없으면 hold)
            self._tick(dt)

            # 3) 충돌 판정 (센서 기반)
            #    sensing_mode가 꺼져있으면 is_collision에서 sensing() 호출이 터질 수 있으니 주의.
            if self._sensing_mode and self.is_collision():
                self._snapshot(note="collision")
                break

            # 4) 종료 조건(임시): 더 처리할 명령도 없고 진행중 acc도 없으면 종료
            #    (나중에 end_point 도달 판정으로 바꾸면 됨)
            if (self._active_acc is None) and (not self._queue):
                break

        return self._log

    def run_queue_with_model(self, model: MotionModel) -> List[Dict[str, Any]]:
        """
        ✅ 요청한 기능:
        1) 조향 처리 객체(model)를 받는다.
        2) ins_acc_mov / ins_set_mov로 쌓인 큐를
        3) model 변환을 통해 state를 갱신하며 실행한다.
        """
        self.set_motion_model(model)

        # 시작 스냅샷
        if not self._log:
            self._snapshot(note="start(model)")

        # start()가 내부적으로 _step()을 쓰므로,
        # 여기서는 그냥 start를 호출하면 "모델 기반 step"으로 자동 실행됨.
        return self.start()

    def dynamic_operation(self) -> None:
        """
        동적 명령 처리 모드로 전환.
        주어진 갱신 주기마다 _setp으로 상태를 갱신한다.
        ins_명령어는 큐에 추가하고, 실행하지는 않는다.
        dynamic_operation 모드에서는 start()가 호출되면 다음을 수행함:
        - sensing_mode 가 True일 때 주기적으로 센싱값을 받아옴.
        - 명령이 큐에 있으면 하나씩 꺼내 즉시 처리. 명령이 없으면 현 상태 유지 명령.
            이때, 명령 지속 시간이 update_rate보다 길면 여러 스텝에 걸쳐 처리.
            ex) time=140ms, update_rate=50ms이면 3(50, 50, 40) 스텝에 걸쳐 처리.
            
        - 맵 충돌 판정 수행.
        - 맵 end_point 도달 판정 수행.

        - 센싱값에 따라 실시간으로 명령을 처리함.
        
        """
        self._dynamic_operation_mode = True
        self._active_acc = None
    
    def run_replanning_astar_follow(
        self,
        model,
        *,
        resolution: float = 1.0,
        expansion: int = 1,
        smooth: bool = True,
        smooth_alpha: float = 0.1,
        smooth_beta: float = 0.3,
        v_ref: float = 1.0,
        dt_ns: int = None,
        goal_tolerance: float = 0.25,
        max_steps: int = 5000,
        steer_limit: float = 2.5,
        use_heading_term: bool = True,
        heading_gain: float = 1.0,
        look_ahead: int = 5,
        keep_near_points: int = 12,
        join_window: int = 25,
    ):
        """
        [용도]
        ✅ "매 tick마다 경로를 다시 계획(replan)하면서" 목표까지 가는 실행 루프.
        - 테스트 맵 모드(True)일 때:
            SlamMap(고정 맵) 기반으로 A* 경로를 매 스텝 다시 계산하고,
            그 경로로 1스텝만 움직이고, 다음 스텝에서 다시 A*를 수행.
        - 나중에 테스트 모드(False) + SLAM(실시간 맵 업데이트)에서도 구조가 그대로 먹힘:
            slam_map이 변하면 blocked가 바뀌고, 그럼 경로도 자동으로 바뀜.

        [입력]
        - model:
            motion model 객체 (예: SimpleBicycleModel)
        - resolution: float
            A* 격자 크기(셀 크기).
            maze_map(cell_size) = 5면 resolution=5.0 강력 추천.
        - expansion: int
            장애물 팽창 셀 수. 로봇 크기/안전거리 반영.
        - smooth: bool
            A* 경로는 지그재그/각진 형태가 많아서,
            smooth_path_nd로 부드럽게 만들지 여부.
        - smooth_alpha, smooth_beta: float
            스무딩 강도 파라미터 (MyPlanning.smooth_path_nd에서 사용)
        - v_ref: float
            전진 속도 목표.
        - dt_ns: int
            tick 시간. None이면 self._update_rate_ns 사용.
        - goal_tolerance: float
            목표 도달 판정 거리(원 반경).
        - max_steps: int
            최대 반복 횟수.
        - steer_limit: float
            조향 제한.
        - use_heading_term: bool
            heading 항(look-ahead 방향 오차) 사용 여부.
        - heading_gain: float
            heading 항 가중치.
        - look_ahead: int
            경로에서 몇 점 앞을 보고 heading을 맞출지.

        [출력]
        - self._log:
            로봇이 기록한 로그 리스트를 반환 (기존 구조를 그대로 따름).
            (내부에서 _snapshot 등을 호출하는 구조라면 그 포맷 그대로 유지)

        [동작 흐름(중요)]
        for step in range(max_steps):
            1) 현재 slam_map으로 A* 실행 -> path 생성
            2) (선택) path 스무딩
            3) path 기반 제어를 1 tick 수행하여 상태 업데이트
            4) 목표 도달하면 종료
        """
        # dt 설정
        if dt_ns is None:
            dt_ns = self._update_rate_ns

        
        # ✅ replanning A*는 "센서가 가진 slam_map"을 사용한다.
        # (테스트용으로 map만 따로 들고 있던 _test_map_mode/_test_SlamMap은 삭제됨)
        # print(self._sensor)
        sensor_map = self._sensor.get_map() if self._sensor is not None else None
        if (self._sensor is None) or sensor_map is None:
            raise RuntimeError(
                "A* replanning을 하려면 sensor와 sensor._slam_map이 필요합니다. "
                "Circle_Sensor(dim, ..., slam_map=...) 형태로 센서를 먼저 설정하세요."
            )

        slam_map = sensor_map

        # goal 결정 우선순위:
        # 1) robot이 들고 있는 end_point
        # 2) slam_map.end_point (maze_map에서 제공)
        goal = self._end_point
        if goal is None and getattr(slam_map, "end_point", None) is not None:
            goal = tuple(float(x) for x in slam_map.end_point)

        if goal is None:
            raise RuntimeError(
                "goal이 없습니다. robot 생성 시 end_point를 주거나, "
                "maze_map의 end_point를 사용하세요."
            )

        # 로그 시작 스냅샷 (로봇이 로그 구조를 갖고 있다는 가정)
        if not self._log:
            self._snapshot(note="start(replanning-astar-follow)")

        # 메인 루프: 매 tick마다 계획 + 1스텝 이동
        for _ in range(max_steps):

            # 1) A* 재계획
            path = self._plan_astar_path(
                slam_map,
                goal_pos=goal,
                resolution=resolution,
                expansion=expansion,
            )
#디버깅 path 출력 
            # print("Replanned path:", path)
            
            # 경로 실패 처리
            if not path:
                self._snapshot(note="replan_failed(no_path)")
                break

            # 2) (선택) 스무딩
            #    path 길이가 너무 짧으면 스무딩 의미가 적거나 오히려 깨질 수 있으니 >=3 조건
            if smooth and len(path) >= 3:
                path = smooth_path_nd(path, alpha=smooth_alpha, beta=smooth_beta)

            # ✅ 2.5) 경로 블렌딩: 가까운 구간은 이전 유지, 먼 구간은 새 경로 반영
            if self._prev_replan_path is not None:
                path = self._merge_paths_keep_near(
                    self._prev_replan_path,
                    path,
                    keep_points=keep_near_points,
                    join_window=join_window,
                )

            # ✅ 이번 tick 경로를 저장(다음 tick에서 near 유지용)
            self._prev_replan_path = path
            
            # ✅ 현재 tick에서 사용한 path를 로그에 남김
            self._snapshot(
                note="replan_path",
                pathing=tuple(tuple(p) for p in path)  # 직렬화 안전
            )

            # 3) 1 스텝 제어/이동
            self._control_one_step_on_path(
                path,
                model=model,
                v_ref=v_ref,
                dt_ns=dt_ns,
                steer_limit=steer_limit,
                use_heading_term=use_heading_term,
                heading_gain=heading_gain,
                look_ahead=look_ahead,
            )

            # 충돌 시 즉시 종료
            if self._collision_detected:
                break


            # 4) 목표 도달 체크
            gx, gy = goal[0], goal[1]
            if math.hypot(self._position[0] - gx, self._position[1] - gy) <= goal_tolerance:
                self._snapshot(note="goal_reached")
                break

        return self._log

    def run_dynamic_path_follow(
        self,
        model,
        *,
        v_ref: float = 1.0,
        dt_ns: int = None,
        goal_tolerance: float = 0.25,
        max_steps: int = 5000,
        steer_limit: float = 2.5,
        use_heading_term: bool = True,
        heading_gain: float = 1.0,
    ):
        """
        ✅ dynamic_operation을 실제로 쓰는 '제어 루프' 실행기.
        - 매 tick마다:
          1) 현재 위치 -> 경로에서 closest point 계산
          2) cross-track error(부호 포함) 계산
          3) PID로 steering(각속도) 생성
          4) (선택) heading error도 같이 섞어서 안정화
          5) 모델로 1 step 적분
        - 목표: path의 마지막 점 근처(goal_tolerance) 도달하면 종료
        """
        if self._path is None:
            raise RuntimeError("추적할 경로가 없습니다. 먼저 robot.set_path(s_path)를 호출하세요.")

        if dt_ns is None:
            dt_ns = self._update_rate_ns

        self.dynamic_operation()         # 모드 ON :contentReference[oaicite:3]{index=3}
        self.set_motion_model(model)     # 모델 사용 :contentReference[oaicite:4]{index=4}

        # 시작 스냅샷
        if not self._log:
            self._snapshot(note="start(dynamic-path-follow)")

        goal = self._path[-1]
        
        last_seg = 0
        L = 5

        for _ in range(max_steps):
            x, y = self.get_position()[0], self.get_position()[1]

            # 1) closest point
            dmin, closest, seg_i, tparam = point_to_path_min_distance_nd([x, y], self._path)
            seg_i = max(seg_i, last_seg)
            last_seg = seg_i
            
            # 2) signed cross-track error
            cte = self._signed_cross_track_error_2d((x, y), closest, seg_i, self._path)

            # 3) PID -> steering rate(스칼라)
            dt_s = dt_ns * 1e-9
            steer_pid = self._pid_scalar(cte, dt_s)

            # 4) heading term(선택): look-ahead 목표점을 바라보게 해서 코너 튐 완화
            steer_heading = 0.0
            if use_heading_term:
                look_idx = min(seg_i + L, len(self._path) - 1)
                tx, ty = self._path[look_idx]

                target_theta = math.atan2(ty - y, tx - x)  # ✅ 현재 위치 -> look-ahead 점
                cur_theta = self._orientation[0]
                heading_err = _wrap_pi(target_theta - cur_theta)

                steer_heading = heading_gain * heading_err

            # dt_s 0 방어
            dt_s = max(1e-9, dt_ns * 1e-9)

            # # 4) heading term(선택): segment 방향을 바라보게 만들어서 코너에서 덜 흔들리게
            # steer_heading = 0.0
            # if use_heading_term:
            #     i = max(0, min(seg_i, len(self._path) - 2))
            #     x0, y0 = self._path[i]
            #     x1, y1 = self._path[i + 1]
            #     target_theta = math.atan2(y1 - y0, x1 - x0)
            #     cur_theta = self._orientation[0]
            #     heading_err = _wrap_pi(target_theta - cur_theta)
            #     steer_heading = heading_gain * heading_err

            steer = steer_pid + steer_heading

            # 5) steering clamp
            steer = max(-steer_limit, min(steer_limit, steer))

            # 6) 명령 적용(여기서는 큐를 쌓지 않고 "현재 tick 제어값"을 직접 적용)
            # velocity는 dim=2라 (v_ref, 0.0) 형태로 통일
            self._velocity = (float(v_ref), 0.0)
            self._steering = (float(steer), 0.0)

            # 7) 1 tick 적분
            self._tick(dt_ns, note="dynamic_control")

            # 8) goal 체크
            gx, gy = goal[0], goal[1]
            if math.hypot(self._position[0] - gx, self._position[1] - gy) <= goal_tolerance:
                self._snapshot(note="goal_reached")
                break

        return self._log

    # ----------------------
    # 계산
    # ----------------------
    def _pid_scalar(self, err: float, dt_s: float) -> float:
        """
        PIDVec는 e를 'Sequence[float]'로 받는다.
        따라서 스칼라 err를 (err, 0, 0, ...) 벡터로 확장해서 넣고
        출력 u[0]만 사용한다.
        """
        pid = self.pid_control

        # PIDVec.dim 기준으로 에러 벡터 구성
        e_vec = [0.0] * getattr(pid, "dim", 1)
        e_vec[0] = float(err)

        # PIDVec는 step(e, dt)가 정식 인터페이스
        out = pid.step(e_vec, float(dt_s))

        # out도 벡터이므로 첫 성분만 사용
        return float(out[0])

    def _signed_cross_track_error_2d(self, pos, closest, seg_i, path):
        """
        2D에서 cross-track error에 부호 부여:
        segment 방향 벡터와 (closest->pos) 벡터의 z-크로스 부호로 좌/우를 판단.
        """
        x, y = pos
        cx, cy = closest

        # segment 방향: path[seg_i] -> path[seg_i+1]
        i = max(0, min(seg_i, len(path) - 2))
        x0, y0 = path[i]
        x1, y1 = path[i + 1]
        sx, sy = (x1 - x0, y1 - y0)

        # normal 판정용 벡터: closest -> pos
        ex, ey = (x - cx, y - cy)

        # 2D cross product z = s x e = sx*ey - sy*ex
        cross_z = sx * ey - sy * ex

        # 부호: cross_z > 0이면 왼쪽, <0이면 오른쪽(좌표계 기준)
        sign = -1.0 if cross_z >= 0 else 1.0

        # 거리 크기
        dx = x - cx
        dy = y - cy
        dist = math.hypot(dx, dy)

        return sign * dist
    
    def _infer_bounds_from_map_limit(self, slam_map, resolution: float):
        """
        [용도]
        - A* 탐색에 사용할 '그리드 인덱스 bounds(경계)'를 SlamMap.limit에서 계산하는 도우미.
        - A*는 (ix, iy) 같은 '셀 인덱스' 공간에서 탐색하는데,
        SlamMap.limit는 보통 (미터/좌표계) 기준의 월드 경계로 저장되어 있음.
        따라서 limit -> (min_idx, max_idx) 형태로 변환해줌.

        [입력]
        - slam_map:
            테스트 모드에서 사용하는 SlamMap 객체.
            slam_map.limit가 존재한다는 가정 하에 처리.
            예: limit = ((0,0), (W,H))   (월드 좌표 경계)
        - resolution: float
            A*에서 사용하는 격자 한 칸의 크기(월드 좌표에서의 cell size).
            maze_map(cell_size=5)이면 보통 resolution=5.0이 맞음.

        [출력]
        - bounds: tuple | None
            (min_idx, max_idx) 형태:
            - min_idx = (0,0)
            - max_idx = (max_x, max_y)  # 인덱스 최대값
            slam_map.limit가 None이면 None 반환 (A* 내부가 bounds 없이 동작하게 할 수 있음)

        [주의]
        - slam_map.limit가 ((0,0),(W,H)) 형태라고 가정했음.
        - max_x/max_y 계산은 "해당 영역을 resolution로 나눈 셀 개수 - 1" 형태.

        --------------------------------------------------------------------------
        slam_map.limit = ((x0,y0),(x1,y1)) 월드 경계 -> A* 인덱스 bounds로 변환.
        여기서 resolution은 '계획 격자 한 칸의 크기' (cell_size랑 독립!)
        """
        if slam_map.limit is None:
            return None

        (x0, y0), (x1, y1) = slam_map.limit

        # 전체 길이
        W = float(x1) - float(x0)
        H = float(y1) - float(y0)

        # 셀 개수 (딱 나눠떨어질 때도 정확히 잡히게)
        nx = int(round(W / resolution))
        ny = int(round(H / resolution))

        # 인덱스는 0..nx-1
        if nx <= 0 or ny <= 0:
            return None

        return ((0, 0), (nx - 1, ny - 1))

    def _build_blocked_indices_from_map(self, slam_map, *, resolution: float, expansion: int):
        """
        [용도]
        - SlamMap 내부 장애물 목록을 A*가 이해하는 "blocked cell index set"으로 변환.
        - 그리고 로봇의 크기/안전거리 반영을 위해 expansion(팽창)을 적용.
        즉, 장애물 주변 셀까지 막아버려서 로봇이 벽에 비비지 않게 함.

        [입력]
        - slam_map:
            SlamMap 객체. get_obs_list()로 장애물 리스트를 꺼낼 수 있어야 함.
            각 장애물은 bounds()로 (min_xy, max_xy)를 반환한다고 가정.
        - resolution: float
            월드 좌표 -> 셀 인덱스 변환 스케일.
            셀 하나가 resolution 크기라고 보고 index를 계산.
        - expansion: int
            셀 단위 팽창 반경.
            - 0이면 장애물 셀만 막음
            - 1이면 장애물 주변 1칸까지 막음
            - 2이면 주변 2칸까지 막음
            보통 로봇 반지름/안전거리 느낌으로 쓰는 값.

        [출력]
        - blocked: set[tuple[int,int]]
            A*에서 통과 불가능한 셀 인덱스들의 집합.
            예: {(0,1), (0,2), (1,2), ...}

        [내부 동작]
        1) 각 장애물 bounds() -> (min_x, min_y), (max_x, max_y)
        2) 이를 셀 인덱스 영역으로 변환 (ix0..ix1, iy0..iy1)
        3) expansion만큼 주변 index까지 blocked에 추가

        [주의]
        - bounds가 "장애물의 실제 월드 좌표 범위"라면,
        index 변환에서 floor/ceil 조합이 중요함.
        """
        blocked = set()

        for obs in slam_map.get_obs_list():
            (min_x, min_y), (max_x, max_y) = obs.bounds()

            # 장애물이 걸치는 셀 인덱스 범위를 계산
            # min은 floor로 시작 index를 잡고,
            # max는 ceil-1로 끝 index를 잡는 방식 (경계 포함 처리)
            ix0 = int(math.floor(min_x / resolution + 1e-12))
            iy0 = int(math.floor(min_y / resolution + 1e-12))
            ix1 = int(math.ceil (max_x / resolution - 1e-12)) - 1
            iy1 = int(math.ceil (max_y / resolution - 1e-12)) - 1

            # expansion 적용: 장애물 셀 범위 주변까지 막는다
            for ix in range(ix0 - expansion, ix1 + expansion + 1):
                for iy in range(iy0 - expansion, iy1 + expansion + 1):
                    blocked.add((ix, iy))

        return blocked

    def _plan_astar_path(self, slam_map, *, goal_pos, resolution: float, expansion: int):
        """
        [용도]
        - "현재 로봇 위치(self.get_position()) → 목표(goal_pos)"까지
        A*로 경로를 계산해 "셀 중심 좌표들"로 이루어진 path를 반환.
        - 이 함수는 "경로 계획(Planning)"만 담당하고,
        "제어(Control)"은 _control_one_step_on_path에서 담당.

        [입력]
        - slam_map:
            SlamMap 객체. 장애물 리스트와 limit을 포함한다고 가정.
        - goal_pos: tuple/list (x, y)
            목표 위치(월드 좌표).
            보통 maze_map에서 end_point로 들어오는 (중심좌표) 사용.
        - resolution: float
            A* 그리드 셀 크기.
            maze_map(cell_size)와 동일하게 주는 게 가장 안전함.
        - expansion: int
            장애물 팽창 셀 수. _build_blocked_indices_from_map으로 전달.

        [출력]
        - path: list[list[float,float]] | None
            A* 결과 경로 (월드 좌표의 셀 중심점들).
            예: [[2.5, 2.5], [7.5, 2.5], [12.5, 2.5], ...]
            경로가 없으면 None 또는 빈 리스트가 올 수 있음(사용하는 A* 구현에 따라).

        [내부 동작]
        1) slam_map 장애물 -> blocked cell set 생성
        2) slam_map.limit -> bounds 생성(가능하면)
        3) astar_find_path_center_nodes 호출해서 경로 생성
        - center_offset=0.5로 "셀 중심"을 좌표로 반환하도록 설정
        """
        blocked = self._build_blocked_indices_from_map(slam_map, resolution=resolution, expansion=expansion)
        bounds  = self._infer_bounds_from_map_limit(slam_map, resolution)

        # start/goal을 "셀 중심좌표"로 스냅 (이건 유지 추천)
        start = self._snap_to_grid_center(self.get_position(), resolution, center_offset=0.5)
        # start = self.get_position()
        goal  = self._snap_to_grid_center(goal_pos, resolution, center_offset=0.5)

        sidx = self._pos_to_idx(start, resolution)
        gidx = self._pos_to_idx(goal,  resolution)

        # ✅ 변경점 1: start는 절대 옮기지 않는다.
        # 대신 start가 blocked면, A*에서만 start 셀을 임시로 free로 취급한다.
        if sidx in blocked:
            blocked = set(blocked)
            blocked.discard(sidx)

        # (선택) goal이 blocked면 이건 목표 자체가 벽 안이라서 보정이 필요하긴 함
        # 튐의 원인은 goal이 아니라 start라서 goal 보정은 유지해도 OK
        # 단, goal도 "점프"가 싫으면 goal도 동일 방식으로 discard만 해도 됨.
        if gidx in blocked:
            blocked = set(blocked)
            blocked.discard(gidx)
#debug
        # sx, sy = start
        # gx, gy = goal
        # sidx = (int(sx / resolution), int(sy / resolution))
        # gidx = (int(gx / resolution), int(gy / resolution))

        # print("start=", start, "sidx=", sidx, "blocked?", sidx in blocked)
        # print("goal =", goal,  "gidx=", gidx, "blocked?", gidx in blocked)
        # print("bounds=", bounds)
#/debug

        path = astar_find_path_center_nodes(
            blocked_map=blocked,
            heuristic_fn=euclidean,
            current_pos=start,
            goal_pos=goal,
            dim=2,
            resolution=resolution,
            center_offset=0.5,
            bounds=bounds,
        )
        return path

    def _merge_paths_keep_near(
        self,
        prev_path,
        new_path,
        *,
        keep_points: int = 12,
        join_window: int = 25,
    ):
        """
        prev_path의 '가까운 부분'은 유지하고, new_path의 '먼 부분'로 자연스럽게 이어붙임.

        - keep_points: prev_path에서 현재 진행(seg) 기준으로 몇 점까지 유지할지
        - join_window: new_path에서 이어붙일 시작점을 찾을 때 탐색할 최대 길이
        """
        if not prev_path:
            return new_path
        if not new_path:
            return prev_path

        x, y = self.get_position()[0], self.get_position()[1]

        # 1) prev_path에서 현재 위치의 closest seg 찾기
        dmin_p, closest_p, seg_p, t_p = point_to_path_min_distance_nd([x, y], prev_path)

        # 뒤로 점프 방지(진행도 고정)
        seg_p = max(seg_p, self._prev_replan_seg)
        self._prev_replan_seg = seg_p

        # 2) prev_path에서 가까운 구간(로컬) 유지: [0 .. seg_p + keep_points]
        cut_i = min(seg_p + keep_points, len(prev_path) - 1)
        kept = prev_path[: cut_i + 1]

        # 3) new_path에서 'kept의 마지막 점'과 가장 가까운 지점을 찾아 그 이후를 붙임
        anchor = kept[-1]
        ax, ay = anchor[0], anchor[1]

        # new_path 전체에서 찾으면 과하게 점프할 수 있어서 앞쪽 join_window만 탐색
        search_end = min(len(new_path), join_window)
        best_j = 0
        best_d = 1e18
        for j in range(search_end):
            nx, ny = new_path[j][0], new_path[j][1]
            d = (nx - ax) ** 2 + (ny - ay) ** 2
            if d < best_d:
                best_d = d
                best_j = j

        stitched = kept + new_path[best_j + 1 :]

        # 중복점 제거(연속 같은 점이 생기면 제어가 흔들릴 수 있음)
        out = [stitched[0]]
        for p in stitched[1:]:
            if (p[0] - out[-1][0]) ** 2 + (p[1] - out[-1][1]) ** 2 > 1e-12:
                out.append(p)

        return out

    def _control_one_step_on_path(
        self,
        path,
        *,
        model,
        v_ref: float,
        dt_ns: int,
        steer_limit: float,
        use_heading_term: bool,
        heading_gain: float,
        look_ahead: int,
    ):
        """
        [용도]
        - 주어진 path를 따라가도록 "제어 입력(steering)"을 만들고,
        motion model을 이용해 딱 1 tick(dt_ns)만 적분해서 로봇 상태를 업데이트.
        - 즉, 이 함수는 "한 스텝 제어 + 한 스텝 이동"만 수행.
        - replan 구조에서 핵심: 매 tick마다 새 path를 만들고, 이걸로 1스텝만 움직인 뒤 다시 계획.

        [입력]
        - path: list[[x,y], ...]
            따라갈 경로(월드 좌표).
            최소 2~3개 점이 있을수록 안정적.
        - model:
            로봇 motion model 객체(예: SimpleBicycleModel 등)
            self._step(dt_ns)에서 사용될 모델.
        - v_ref: float
            전진 속도 목표값(제어 목표).
            (테스트에서는 상수로 두는게 흔함)
        - dt_ns: int
            한 스텝 시간 (나노초).
            dt_s = dt_ns * 1e-9로 초 단위 변환해서 PID에 사용.
        - steer_limit: float
            조향각/각속도 입력 제한.
            큰 값이면 급회전 가능하지만 흔들릴 수 있음.
        - use_heading_term: bool
            cross-track error(경로 수직 오차)만으로 조향하면,
            급커브/노이즈에서 흔들릴 수 있어서 heading 항을 추가할지 여부.
        - heading_gain: float
            heading term 가중치.
            너무 크면 목표 방향으로 과민 반응, 너무 작으면 효과 미미.
        - look_ahead: int
            "경로를 몇 점 앞"을 목표로 heading을 잡을지.
            - 작으면 타이트하게 따라가지만 흔들릴 수 있음
            - 크면 부드럽지만 코너를 깎을 수 있음

        [출력]
        - 없음(None)
            내부 상태(self._position, self._orientation, self._velocity, self._steering, log)가 업데이트됨.

        [내부 흐름]
        1) 현재 위치에서 path까지의 최소거리 점(closest) 계산
        2) signed cross-track error(cte) 계산 (오른쪽/왼쪽 부호 포함)
        3) PID로 steer_pid 생성
        4) (옵션) look-ahead 방향 기반 heading_err 계산 -> steer_heading 생성
        5) steer = steer_pid + steer_heading, 제한 적용
        6) self._velocity / self._steering 업데이트 후, model로 1 step 적분
        """
        # 현재 위치
        c_pose = self.get_position()
        x, y = c_pose[0], c_pose[1]

        # 1) closest point (path 위에서 현재 위치와 가장 가까운 점)
        # point_to_path_min_distance_nd:
        #   dmin: 최소거리
        #   closest: path 위의 closest point 좌표
        #   seg_i: closest가 속한 선분 index
        #   tparam: 선분 내 보간 파라미터(0~1)
        dmin, closest, seg_i, tparam = point_to_path_min_distance_nd([x, y], path)

        # 2) signed cross-track error (좌/우 부호 포함)
        #    경로의 접선 방향을 기준으로,
        #    로봇이 경로의 왼쪽/오른쪽 어디에 있는지에 따라 +/-
        cte = self._signed_cross_track_error_2d((x, y), closest, seg_i, path)

        # 3) PID 기반 조향 입력
        dt_s = max(1e-9, dt_ns * 1e-9)
        steer_pid = self._pid_scalar(cte, dt_s)

        # 4) heading term (look-ahead 목표점으로 향하는 방향 오차)
        steer_heading = 0.0
        if use_heading_term:
            look_idx = min(seg_i + look_ahead, len(path) - 1)
            tx, ty = path[look_idx]
            target_theta = math.atan2(ty - y, tx - x)

            cur_theta = self.get_orientation()[0]
            heading_err = _wrap_pi(target_theta - cur_theta)  # -pi~pi로 정규화
            steer_heading = heading_gain * heading_err

        # 5) 두 항 합치고 제한
        steer = steer_pid + steer_heading
        steer = max(-steer_limit, min(steer_limit, steer))

        # 6) motion model에 들어갈 입력값 업데이트
        #    (여기서는 v_ref를 그대로 사용: 속도 제어는 단순화)
        self._velocity = (float(v_ref), 0.0)
        self._steering = (float(steer), 0.0)

        # 7) 모델 설정 + 1 tick 수행 (tick 엔진으로!)
        self.set_motion_model(model)
        self._tick(dt_ns, note="replan_step(dynamic_control)")


    def _snap_to_grid_center(self, pos, resolution: float, center_offset: float = 0.5):
        """
        연속 좌표 pos를, A*가 기대하는 'resolution 격자'의 셀 중심좌표로 스냅.
        예) resolution=1이면 (i+0.5, j+0.5) 형태로 맞춰줌.
        """
        x, y = float(pos[0]), float(pos[1])
        ix = int(math.floor(x / resolution - center_offset + 1e-12))
        iy = int(math.floor(y / resolution - center_offset + 1e-12))
        return ((ix + center_offset) * resolution, (iy + center_offset) * resolution)

    def _pos_to_idx(self, pos, resolution: float):
        x, y = float(pos[0]), float(pos[1])
        return (int(math.floor(x / resolution + 1e-12)),
                int(math.floor(y / resolution + 1e-12)))

    def _idx_to_center(self, idx, resolution: float, center_offset: float = 0.5):
        ix, iy = idx
        return ((ix + center_offset) * resolution, (iy + center_offset) * resolution)

    def _nearest_free_idx(self, start_idx, blocked: set, bounds, max_r: int = 30):
        """
        start_idx가 blocked면, bounds 안에서 가장 가까운 free idx를 BFS로 찾음.
        """
        if start_idx not in blocked:
            return start_idx

        (minx, miny), (maxx, maxy) = bounds if bounds is not None else ((-10**9, -10**9), (10**9, 10**9))

        q = deque([start_idx])
        seen = {start_idx}

        def in_bounds(a, b):
            return (minx <= a <= maxx) and (miny <= b <= maxy)

        steps = 0
        while q and steps < max_r * max_r:
            ix, iy = q.popleft()
            # 4-neighbor
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nx, ny = ix + dx, iy + dy
                if (nx, ny) in seen:
                    continue
                if not in_bounds(nx, ny):
                    continue
                if (nx, ny) not in blocked:
                    return (nx, ny)
                seen.add((nx, ny))
                q.append((nx, ny))
            steps += 1

        return start_idx  # 못 찾으면 원래 값(실패 유지)

    def _localize_grid_by_circle_scan(self, z_meas) -> Tuple[float, ...]:
        """Circle_Sensor 측정(z_meas)으로 맵 위 (x,y) 추정."""

        # 페이크 빠른 측정 모드
        if self._fake_fast_sensing:
            c_pose = self._position
            x = random.gauss(c_pose[0], self._fake_fast_sensing_sigma)
            y = random.gauss(c_pose[1], self._fake_fast_sensing_sigma)
            return (x, y)
        
        if self._sensor is None:
            return self._pos_est
        slam_map = self._sensor.get_map()
        if slam_map is None or getattr(slam_map, "limit", None) is None:
            return self._pos_est
        
        print("Localizing...")
        (x0, y0), (x1, y1) = slam_map.limit
        res = self._loc_resolution
        nx = int(math.ceil((x1 - x0) / res))
        ny = int(math.ceil((y1 - y0) / res))

        best_ll = -1e18
        best_xy: Tuple[float, ...] = self._pos_est
        max_range = getattr(self._sensor, "_distance", 0.0)

        for iy in range(ny):
            cy = y0 + (iy + 0.5) * res
            for ix in range(nx):
                cx = x0 + (ix + 0.5) * res
                if slam_map.is_wall((cx, cy), inclusive=True):
                    continue

                z_pred = self._sensor.sensing((cx, cy))
                ll = 0.0
                for m, p in zip(z_meas, z_pred):
                    m2 = max_range if m is None else m
                    p2 = max_range if p is None else p
                    err = m2 - p2
                    ll += -0.5 * (err / self._loc_sigma) ** 2

                if ll > best_ll:
                    best_ll = ll
                    best_xy = (cx, cy)

        return best_xy

    #----------------------
    # localization, noize
    #----------------------
    def set_measurement_noise(self, *, enable: bool = True, sigma: float = 0.1) -> None:
        """
        ✅ 센서 측정에 가우시안 노이즈 적용 ON/OFF
        - 지금은 간단히 z에 N(0, sigma^2) 추가
        - 나중에 EKF/Particle로 갈 때는 여기 대신 "센서모델"로 분리하면 됨
        """
        if sigma < 0:
            raise ValueError("sigma는 0 이상이어야 합니다.")
        self._meas_noise_enabled = bool(enable)
        self._meas_noise_sigma = float(sigma)

    def _apply_measurement_noise(self, z):
        """내부용: z(list/tuple) 형태의 거리측정에 가우시안 노이즈 추가."""
        if (not self._meas_noise_enabled) or self._meas_noise_sigma <= 0:
            return z
        out = []
        for v in z:
            if v is None:
                out.append(None)
            else:
                out.append(float(v) + random.gauss(0.0, self._meas_noise_sigma))
        return out

    #----------------------
    # 칼만 필터
    #----------------------
    def _ekf_sync_to_est(self) -> None:
        """EKF 상태 -> (pos_est, ori_est)로 반영"""
        x, y, th = float(self._ekf_x[0]), float(self._ekf_x[1]), float(self._ekf_x[2])
        self._pos_est = (x, y)
        # orientation이 (theta, 0.0) 같은 형태라면 첫 축만 갱신
        self._ori_est = (th,) + tuple(0.0 for _ in range(self.dim - 1))

    def _ekf_predict(self, dt_s: float) -> None:
        """제어(velocity/steering)로 예측 단계"""
        if not self._ekf_enabled:
            return

        x, y, th = self._ekf_x
        v = float(self._velocity[0]) if len(self._velocity) > 0 else 0.0
        w = float(self._steering[0]) if len(self._steering) > 0 else 0.0

        # 비선형 상태전이
        nx = x + v * math.cos(th) * dt_s
        ny = y + v * math.sin(th) * dt_s
        nth = th + w * dt_s
        self._ekf_x = np.array([nx, ny, nth], dtype=float)

        # Jacobian F = df/dx
        F = np.array([
            [1.0, 0.0, -v * math.sin(th) * dt_s],
            [0.0, 1.0,  v * math.cos(th) * dt_s],
            [0.0, 0.0,  1.0],
        ], dtype=float)

        self._ekf_P = F @ self._ekf_P @ F.T + self._ekf_Q
        self._ekf_sync_to_est()

    def _ekf_update_xy(self, z_xy: Tuple[float, float]) -> None:
        """측정 z=(x,y)로 update 단계"""
        if not self._ekf_enabled:
            return

        z = np.array([float(z_xy[0]), float(z_xy[1])], dtype=float)

        H = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=float)

        x = self._ekf_x
        P = self._ekf_P

        y = z - (H @ x)                        # innovation
        S = H @ P @ H.T + self._ekf_R          # innovation cov
        K = P @ H.T @ np.linalg.inv(S)         # Kalman gain

        self._ekf_x = x + K @ y
        I = np.eye(3, dtype=float)
        self._ekf_P = (I - K @ H) @ P

        self._ekf_sync_to_est()
