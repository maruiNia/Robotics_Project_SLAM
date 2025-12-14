from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union
import math
import random

from .MyUtils import _vec, _check_dim

Number = Union[int, float]


class Sensor:
    """센서 베이스 클래스(최소 인터페이스)."""
    def sensing(self, position: Sequence[Number]):
        raise NotImplementedError


class Circle_Sensor(Sensor):
    """
    2D 원형 센서(간단 LiDAR 스타일).
    - number개의 각도로 레이를 쏘고
    - distance(최대 감지 거리) 안에서 첫 장애물까지의 거리를 반환
    - 장애물 없으면 None
    """

    def __init__(
        self,
        dim: int,
        number: int = 10,
        distance: Number = 5,
        slam_map=None,
        step: Number = 0.05,  # 레이 전진 샘플 간격(m) - 작을수록 정밀, 느려짐
    ):
        if dim != 2:
            raise NotImplementedError("Circle_Sensor는 현재 dim=2만 지원합니다.")
        if number <= 0:
            raise ValueError("number는 1 이상이어야 합니다.")
        if distance <= 0:
            raise ValueError("distance는 0보다 커야 합니다.")
        if step <= 0:
            raise ValueError("step은 0보다 커야 합니다.")
        if slam_map is None:
            raise ValueError("slam_map(SlamMap 인스턴스)이 필요합니다.")

        self._dim = dim
        self._number = int(number)
        self._distance = float(distance)
        self._step = float(step)
        self._map = slam_map

        # 노이즈 함수(없으면 None)
        # signature: f(measure: Optional[float], angle_rad: float) -> Optional[float]
        self._noise_fn: Optional[Callable[[Optional[float], float], Optional[float]]] = None

    def set_noise(self, noise_fn: Optional[Callable[[Optional[float], float], Optional[float]]]):
        """노이즈 함수 설정. None이면 노이즈 제거."""
        self._noise_fn = noise_fn

    def get_number(self) -> int:
        return self._number

    def get_dim(self) -> int:
        return self._dim

    def get_distance(self) -> float:
        return self._distance

    def sensing(self, position: Sequence[Number]) -> List[Optional[float]]:
        """
        position: (x, y)
        return: [d0, d1, ..., d_{number-1}]  (각도 0..2pi 균등)
                di는 float(첫 장애물까지 거리) 또는 None(없음)
        """
        if len(position) != 2:
            raise ValueError("position은 (x, y) 형태의 길이 2여야 합니다.")
        ox, oy = float(position[0]), float(position[1])

        results: List[Optional[float]] = []
        for k in range(self._number):
            angle = 2.0 * math.pi * k / self._number
            dx = math.cos(angle)
            dy = math.sin(angle)

            hit_dist = self._raycast_first_hit(ox, oy, dx, dy)

            # 노이즈 적용
            if self._noise_fn is not None:
                hit_dist = self._noise_fn(hit_dist, angle)

            results.append(hit_dist)

        return results

    def _raycast_first_hit(self, ox: float, oy: float, dx: float, dy: float) -> Optional[float]:
        """(ox,oy)에서 (dx,dy) 방향으로 최대 distance까지 전진 샘플링, 첫 hit 거리 반환."""
        max_d = self._distance
        step = self._step

        # 0은 자기 위치(벽 위에 서있을 수도)라서 보통 아주 살짝 앞으로부터 체크
        d = step
        while d <= max_d + 1e-12:
            x = ox + dx * d
            y = oy + dy * d
            if self._map.is_wall((x, y), inclusive=True):
                return d
            d += step

        return None


# ---------------------------
# (선택) 노이즈 함수 예시들
# ---------------------------

def gaussian_noise(sigma: float = 0.02):
    """
    측정값이 있을 때만 가우시안 노이즈 추가.
    sigma 단위는 'm'.
    """
    def _f(measure: Optional[float], angle_rad: float) -> Optional[float]:
        if measure is None:
            return None
        noisy = measure + random.gauss(0.0, sigma)
        return max(0.0, noisy)
    return _f

def dropout(p: float = 0.05):
    """측정값이 있어도 확률 p로 None으로 날려버리는 드랍아웃."""
    def _f(measure: Optional[float], angle_rad: float) -> Optional[float]:
        if measure is None:
            return None
        if random.random() < p:
            return None
        return measure
    return _f
