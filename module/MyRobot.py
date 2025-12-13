from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union, Dict, Any

Number = Union[int, float]


def _vec(dim: int, value: Optional[Sequence[Number]], name: str) -> Tuple[float, ...]:
    """None이면 0벡터, 아니면 dim길이 튜플로 변환."""
    if value is None:
        return tuple(0.0 for _ in range(dim))
    if len(value) != dim:
        raise ValueError(f"{name}의 길이가 dim({dim})과 다릅니다. (len={len(value)})")
    return tuple(float(x) for x in value)


def _check_dim(dim: int):
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError("dim은 1 이상의 정수여야 합니다.")


def _check_ns(ns: int, name: str):
    if not isinstance(ns, int) or ns <= 0:
        raise ValueError(f"{name}는 1 이상의 정수(ns)여야 합니다.")


@dataclass
class _Command:
    kind: str  # "set" or "acc"
    position: Optional[Tuple[float, ...]] = None
    velocity: Optional[Tuple[float, ...]] = None
    steering: Optional[Tuple[float, ...]] = None
    time_ns: int = 0  # acc에서만 사용


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
        update_rate: int = 1000,
    ):
        _check_dim(dim)
        _check_ns(update_rate, "update_rate")

        self.dim: int = dim
        self._position: Tuple[float, ...] = _vec(dim, position, "position")
        self._velocity: Tuple[float, ...] = _vec(dim, velocity, "velocity")
        self._steering: Tuple[float, ...] = _vec(dim, steering, "steering")

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
        cmd = _Command(
            kind="set",
            position=None if position is None else _vec(self.dim, position, "position"),
            velocity=None if velocity is None else _vec(self.dim, velocity, "velocity"),
            steering=None if steering is None else _vec(self.dim, steering, "steering"),
        )
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
        _check_ns(time, "time")
        cmd = _Command(
            kind="acc",
            velocity=None if velocity is None else _vec(self.dim, velocity, "velocity"),
            steering=None if steering is None else _vec(self.dim, steering, "steering"),
            time_ns=time,
        )
        self._queue.append(cmd)

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
        """dt_ns만큼 상태 적분(단순 Euler)."""
        dt_s = dt_ns * 1e-9

        # position += velocity * dt
        self._position = tuple(self._position[i] + self._velocity[i] * dt_s for i in range(self.dim))

        # orientation += steering * dt  (조향을 각속도로 가정)
        self._orientation = tuple(self._orientation[i] + self._steering[i] * dt_s for i in range(self.dim))

        self._time_ns += dt_ns
        self._snapshot(note=note)

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

    def start(self) -> List[Dict[str, Any]]:
        """
        로봇을 작동시킨다.
        - 큐에 쌓인 명령을 순서대로 처리
        - 처리 과정 전체 로그를 반환
        """
        # 시작 스냅샷
        if not self._log:
            self._snapshot(note="start")

        while self._queue:
            cmd = self._queue.pop(0)
            if cmd.kind == "set":
                self._apply_set(cmd)
            elif cmd.kind == "acc":
                self._apply_acc(cmd)
            else:
                raise RuntimeError(f"알 수 없는 명령: {cmd.kind}")

        return self._log
