from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union, Dict, Any

from .MySensor import Circle_Sensor
from .MyUtils import _vec, _check_dim

Number = Union[int, float]
from typing import Protocol

class MotionModel(Protocol):
    """
    로봇 상태를 motion model로 1스텝 갱신하는 인터페이스.
    - 반환: (new_position, new_orientation)
    """
    def step(
        self,
        position: Tuple[float, ...],
        orientation: Tuple[float, ...],
        velocity: Tuple[float, ...],
        steering: Tuple[float, ...],
        dt_s: float,
    ) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        ...


class SimpleBicycleModel:
    """
    2D용 간단 Bicycle Model 예시.
    상태:
      - position = (x, y)
      - orientation = (theta, 0) 또는 (theta,) 형태를 허용 (첫 원소를 theta로 사용)
    입력:
      - velocity[0] = v (m/s)  (벡터지만 첫 원소를 전진속도로 사용)
      - steering[0] = delta (rad) (앞바퀴 조향각)
    """
    def __init__(self, wheelbase: float = 1.0):
        self.L = float(wheelbase)

    def step(self, position, orientation, velocity, steering, dt_s):
        if len(position) < 2:
            raise ValueError("SimpleBicycleModel은 position dim>=2(최소 x,y) 필요")
        x, y = position[0], position[1]

        theta = orientation[0] if len(orientation) > 0 else 0.0
        v = velocity[0] if len(velocity) > 0 else 0.0
        delta = steering[0] if len(steering) > 0 else 0.0

        # Bicycle kinematics
        # x' = v cos(theta), y' = v sin(theta), theta' = v/L * tan(delta)
        import math
        x_new = x + v * math.cos(theta) * dt_s
        y_new = y + v * math.sin(theta) * dt_s
        theta_new = theta + (v / self.L) * math.tan(delta) * dt_s

        # 기존 튜플 길이 유지(나머지 차원은 그대로)
        new_pos = (x_new, y_new) + tuple(position[2:])
        if len(orientation) == 0:
            new_ori = (theta_new,)
        else:
            new_ori = (theta_new,) + tuple(orientation[1:])

        return new_pos, new_ori

# class MotionModel(Protocol):
#     def step(self, position, orientation, velocity, steering, dt_s):
#         ...

# class SimpleBicycleModel:
#     def __init__(self, wheelbase: float = 1.0):
#         self.L = float(wheelbase)

#     def step(self, position, orientation, velocity, steering, dt_s):
#         import math
#         x, y = position[0], position[1]
#         theta = orientation[0] if len(orientation) > 0 else 0.0
#         v = velocity[0] if len(velocity) > 0 else 0.0
#         delta = steering[0] if len(steering) > 0 else 0.0

#         x_new = x + v * math.cos(theta) * dt_s
#         y_new = y + v * math.sin(theta) * dt_s
#         theta_new = theta + (v / self.L) * math.tan(delta) * dt_s

#         new_pos = (x_new, y_new) + tuple(position[2:])
#         if len(orientation) == 0:
#             new_ori = (theta_new,)
#         else:
#             new_ori = (theta_new,) + tuple(orientation[1:])
#         return new_pos, new_ori