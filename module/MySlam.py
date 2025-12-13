from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

Number = Union[int, float]
Point = Union[Tuple[Number, ...], List[Number]]


def _to_tuple(seq: Sequence[Number], dim: int, name: str) -> Tuple[Number, ...]:
    if len(seq) != dim:
        raise ValueError(f"{name} 길이가 dim({dim})과 다릅니다. (len={len(seq)})")
    return tuple(seq)


@dataclass
class Obstacle_obj:
    """
    최소 좌표 + 최대 좌표(=max_corner) 또는
    최소 좌표 + size(각 축의 길이들)로 생성하는 '축에 평행한 사각형 장애물' 클래스.
    dim차원에서 동작하며, 내부적으로 (min_corner, max_corner)로 보관.

    포함 판정(contains):
      - 기본은 '경계 포함'입니다. (min <= p <= max)
    """
    dim: int
    min_corner: Tuple[Number, ...]
    max_corner: Tuple[Number, ...]

    def __init__(
        self,
        dim: int,
        min_corner: Sequence[Number],
        max_corner: Optional[Sequence[Number]] = None,
        size: Optional[Sequence[Number]] = None,
    ):
        if dim <= 0:
            raise ValueError("dim은 1 이상의 정수여야 합니다.")

        if (max_corner is None) == (size is None):
            raise ValueError("max_corner 또는 size 중 정확히 하나만 제공해야 합니다.")

        mn = _to_tuple(min_corner, dim, "min_corner")

        if max_corner is not None:
            mx = _to_tuple(max_corner, dim, "max_corner")
        else:
            sz = _to_tuple(size, dim, "size")
            mx = tuple(mn[i] + sz[i] for i in range(dim))

        # min/max가 뒤집혀 들어오면 자동 정렬해서 저장
        fixed_min = tuple(min(mn[i], mx[i]) for i in range(dim))
        fixed_max = tuple(max(mn[i], mx[i]) for i in range(dim))

        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "min_corner", fixed_min)
        object.__setattr__(self, "max_corner", fixed_max)

    def contains(self, point: Sequence[Number], inclusive: bool = True) -> bool:
        p = _to_tuple(point, self.dim, "point")
        if inclusive:
            return all(self.min_corner[i] <= p[i] <= self.max_corner[i] for i in range(self.dim))
        else:
            return all(self.min_corner[i] < p[i] < self.max_corner[i] for i in range(self.dim))

    def bounds(self) -> Tuple[Tuple[Number, ...], Tuple[Number, ...]]:
        return self.min_corner, self.max_corner

    def __repr__(self) -> str:
        return f"Obstacle_obj(dim={self.dim}, min={self.min_corner}, max={self.max_corner})"

class SlamMap:
    """
    지도 객체.
    - 좌표계에 한계(limit)가 없을 수도 있고, 있을 수도 있음.
    - 장애물(Obstacle_obj)을 추가/조회/충돌(벽) 판정 가능.

    init(dim, limit=None):
      - dim: 차원
      - limit: ((start...), (end...)) 형태. dim에 맞춰야 함.
               start/end가 뒤집혀도 자동으로 정렬 저장.

    get_map():
      - 사람 보기용 요약 정보 반환 (맵 limit + 장애물 전체 bounding box + 리스트)

    add(obstacle):
      - dim 맞으면 추가, 아니면 예외

    get_obs_list():
      - tuple 형태로 장애물 리스트 반환

    is_wall(point):
      - 해당 좌표가 어떤 장애물 내부(경계 포함)에 있으면 True
      - limit이 있을 경우, limit 밖이면 True로 볼지 정책이 필요하지만
        여기서는 "limit은 지도 범위 정보"로만 두고, 벽 판정은 장애물만 기준으로 함.
        (원하면 limit 밖을 wall로 처리하도록 옵션 추가 가능)

    limit_outside_is_wall:
      - True  : limit 밖 좌표는 벽(True)으로 처리
      - False : limit 밖이라도 장애물만 기준으로 벽 판단(기존 동작)
    """

    def __init__(
        self,
        dim: int,
        limit: Optional[Tuple[Sequence[Number], Sequence[Number]]] = None,
        limit_outside_is_wall: bool = False,
    ):
        if dim <= 0:
            raise ValueError("dim은 1 이상의 정수여야 합니다.")
        self.dim: int = dim
        self._obs: List[Obstacle_obj] = []
        self.limit: Optional[Tuple[Tuple[Number, ...], Tuple[Number, ...]]] = None
        self.limit_outside_is_wall: bool = limit_outside_is_wall

        if limit is not None:
            self.set_limit(limit)

    def set_limit(self, limit: Tuple[Sequence[Number], Sequence[Number]]) -> None:
        start, end = limit
        st = _to_tuple(start, self.dim, "limit.start")
        ed = _to_tuple(end, self.dim, "limit.end")

        fixed_start = tuple(min(st[i], ed[i]) for i in range(self.dim))
        fixed_end = tuple(max(st[i], ed[i]) for i in range(self.dim))
        self.limit = (fixed_start, fixed_end)

    def add(self, obstacle: Obstacle_obj) -> None:
        if not isinstance(obstacle, Obstacle_obj):
            raise TypeError("add()는 Obstacle_obj 타입만 받을 수 있습니다.")
        if obstacle.dim != self.dim:
            raise ValueError(f"장애물 dim({obstacle.dim})이 맵 dim({self.dim})과 다릅니다.")
        self._obs.append(obstacle)

    def get_obs_list(self) -> Tuple[Obstacle_obj, ...]:
        return tuple(self._obs)

    def _is_outside_limit(self, p: Tuple[Number, ...], inclusive: bool = True) -> bool:
        """
        p가 limit 밖인지 판정.
        inclusive=True이면 경계 포함: start <= p <= end 는 inside
        inclusive=False이면 경계 제외: start < p < end 만 inside
        """
        if self.limit is None:
            return False

        start, end = self.limit
        if inclusive:
            inside = all(start[i] <= p[i] <= end[i] for i in range(self.dim))
        else:
            inside = all(start[i] < p[i] < end[i] for i in range(self.dim))
        return not inside

    def is_wall(self, point: Point, inclusive: bool = True) -> bool:
        p = _to_tuple(point, self.dim, "point")

        # ✅ 추가된 기능: limit 밖이면 wall 처리
        if self.limit_outside_is_wall and self._is_outside_limit(p, inclusive=inclusive):
            return True

        # 장애물 충돌 판정
        return any(obs.contains(p, inclusive=inclusive) for obs in self._obs)

    def get_map(self) -> dict:
        return {
            "dim": self.dim,
            "limit": self.limit,
            "limit_outside_is_wall": self.limit_outside_is_wall,
            "obstacle_count": len(self._obs),
            "obstacles": [repr(o) for o in self._obs],
        }