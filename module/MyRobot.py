from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union, Dict, Any

from .MySensor import Circle_Sensor
from .MyPlanning import PIDVec
from .MyControl import MotionModel, SimpleBicycleModel
from .MyUtils import _vec, _check_dim

Number = Union[int, float]

def _check_ns(ns: int, name: str):
    if not isinstance(ns, int) or ns <= 0:
        raise ValueError(f"{name}는 1 이상의 정수(ns)여야 합니다.")

@dataclass
class _Command:
    kind: str  # "set" or "acc"
    position: Optional[Tuple[float, ...]] = None
    velocity: Optional[Tuple[float, ...]] = None
    steering: Optional[Tuple[float, ...]] = None
    sensing: Optional[Tuple[float, ...]] = None
    time_ns: int = 0  # acc에서만 사용

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
        ):
        _check_dim(dim)
        _check_ns(update_rate, "update_rate")

        self.dim: int = dim
        self._position: Tuple[float, ...] = _vec(dim, position, "position")
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

        #동적 명령 처리
        self._path = None #추적할 경로

        # orientation은 "방향"이니까 steering을 적분해서 얻는 값으로 하나 둠(벡터로)
        self._orientation: Tuple[float, ...] = tuple(0.0 for _ in range(dim))

        self._update_rate_ns: int = update_rate
        self._time_ns: int = 0

        self._queue: List[_Command] = []
        self._log: List[Dict[str, Any]] = []

    # ----------------------
    # getters
    # ----------------------
    def get_update_rate(self) -> int:
        return self._update_rate_ns

    def get_position(self) -> Tuple[float, ...]:
        return self._position

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
        return self._orientation
    
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

    def _tick(self, dt_ns: int) -> None:
        """
        update_rate마다 한 번 호출되는 '동적 스텝' 함수.
        - 진행중 acc가 있으면 dt_ns만큼 선형으로 v/s를 진행
        - 없으면 hold 상태로 dt_ns만큼 진행
        """
        # 센싱 모드: tick 시작 시점에 한 번 갱신(원하면 acc 내부에서도 갱신 가능)
        if self._sensing_mode:
            sensing = self.sensing()
            self._last_sensing = _vec(self.dim, sensing, "sensing")  # type: ignore

        # 진행중 acc가 없으면 그냥 유지하며 step
        if self._active_acc is None:
            self._step(dt_ns, note="hold")
            return

        a = self._active_acc
        total = a.total_ns

        # 마지막 조각(예: 40ms)을 위해 남은 시간만큼만 수행
        dt = min(dt_ns, total - a.elapsed_ns)

        # 이번 tick 끝 시점의 ratio로 선형 보간
        t_next = a.elapsed_ns + dt
        ratio = t_next / total

        self._velocity = tuple(a.v0[i] + a.dv[i] * ratio for i in range(self.dim))
        self._steering = tuple(a.s0[i] + a.ds[i] * ratio for i in range(self.dim))

        self._step(dt, note="acc_step")
        a.elapsed_ns += dt

        # acc 끝났으면 해제
        if a.elapsed_ns >= total:
            self._active_acc = None

    # ----------------------
    # simulation core
    # ----------------------
    def _snapshot(self, note: str = "") -> None:
        self._log.append(
            {
                "t_ns": self._time_ns,
                "position": self._position,
                "velocity": self._velocity,
                "steering": self._steering,
                "orientation": self._orientation,
                "note": note,
            }
        )

    def _step(self, dt_ns: int, note: str = "") -> None:
        """dt_ns만큼 상태 적분."""
        dt_s = dt_ns * 1e-9

        # ✅ motion model이 있으면 그걸로 position/orientation 갱신
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
            # 기존 방식(각 성분별 Euler)
            self._position = tuple(self._position[i] + self._velocity[i] * dt_s for i in range(self.dim))
            self._orientation = tuple(self._orientation[i] + self._steering[i] * dt_s for i in range(self.dim))

        self._time_ns += dt_ns
        self._snapshot(note=note)

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
            self._step(dt, note="acc")

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
        
