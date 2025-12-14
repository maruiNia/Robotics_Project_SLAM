from __future__ import annotations
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Set
import heapq
import math

Number = float
Index = Tuple[int, ...]
Point = Tuple[Number, ...]

def euclidean(p: Sequence[float], q: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(p, q)))

def _to_index_from_center(pt: Sequence[Number], dim: int, cell_size: float, center_offset: float) -> Index:
    """
    중심좌표 (i+0.5)*cell_size 형태 -> 정수 셀 인덱스 i 로 변환
    """
    if len(pt) != dim:
        raise ValueError(f"point dim mismatch: expected {dim}, got {len(pt)}")
    idx = []
    for v in pt:
        v = float(v)
        # (i + center_offset)*cell_size = v  => i = floor(v/cell_size - center_offset)
        i = math.floor(v / cell_size - center_offset + 1e-12)
        idx.append(int(i))
    return tuple(idx)

def _to_center_from_index(idx: Index, cell_size: float, center_offset: float) -> Point:
    return tuple((i + center_offset) * cell_size for i in idx)

def _neighbors_axis_aligned(idx: Index) -> List[Index]:
    """
    n차원에서 축 정렬 이동만 허용: 각 축에 대해 +1/-1 (2*dim개)
    """
    nbrs = []
    dim = len(idx)
    for d in range(dim):
        for step in (-1, 1):
            lst = list(idx)
            lst[d] += step
            nbrs.append(tuple(lst))
    return nbrs

def astar_find_path_center_nodes(
    blocked_map: Iterable[Index],
    heuristic_fn: Callable[[Point, Point], float],
    current_pos: Sequence[Number],
    goal_pos: Sequence[Number],
    *,
    dim: int,
    resolution: float = 1.0,
    center_offset: float = 0.5,
    bounds: Optional[Tuple[Index, Index]] = None,
) -> List[Point]:
    """
    ✅ 요청 스펙:
    - 입력: (확장 처리된) 맵, 휴리스틱, 현 위치(중앙좌표), 목표 위치(중앙좌표)
    - 휴리스틱: 목표까지 직선거리 (사용자가 전달하는 함수로 일반화)
    - 비용(g): 이동거리 (축 이동이면 기본 1*resolution)
    - 출력: 경로(중앙좌표) 리스트

    blocked_map:
      - get_expanded_wall_points 결과를 (i,j,...) 정수 인덱스 집합으로 만든 것
    bounds:
      - (min_idx, max_idx) 형태로 탐색 범위를 제한하고 싶을 때 사용 (선택)
      - 각 축에서 min_idx <= idx <= max_idx 로 제한
    """
    blocked: Set[Index] = set(tuple(map(int, b)) for b in blocked_map)

    start_idx = _to_index_from_center(current_pos, dim, resolution, center_offset)
    goal_idx  = _to_index_from_center(goal_pos, dim, resolution, center_offset)

    # 시작/목표가 벽이면 바로 실패
    if start_idx in blocked or goal_idx in blocked:
        return []

    def in_bounds(i: Index) -> bool:
        if bounds is None:
            return True
        mn, mx = bounds
        return all(mn[d] <= i[d] <= mx[d] for d in range(dim))

    # A* 자료구조
    open_heap: List[Tuple[float, float, Index]] = []
    heapq.heappush(open_heap, (0.0, 0.0, start_idx))

    came_from: Dict[Index, Index] = {}
    g_score: Dict[Index, float] = {start_idx: 0.0}

    goal_center = _to_center_from_index(goal_idx, resolution, center_offset)

    while open_heap:
        _, g_cur, cur = heapq.heappop(open_heap)

        if cur == goal_idx:
            # 경로 복원
            path_idx = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path_idx.append(cur)
            path_idx.reverse()
            return [_to_center_from_index(i, resolution, center_offset) for i in path_idx]

        # 오래된 heap entry skip
        if g_cur > g_score.get(cur, float("inf")) + 1e-12:
            continue

        cur_center = _to_center_from_index(cur, resolution, center_offset)

        for nb in _neighbors_axis_aligned(cur):
            if nb in blocked:
                continue
            if not in_bounds(nb):
                continue

            nb_center = _to_center_from_index(nb, resolution, center_offset)

            # 축 이동 비용: 정확히 resolution (연속공간 거리)
            step_cost = euclidean(cur_center, nb_center)  # 보통 resolution
            tentative_g = g_score[cur] + step_cost

            if tentative_g + 1e-12 < g_score.get(nb, float("inf")):
                came_from[nb] = cur
                g_score[nb] = tentative_g
                h = heuristic_fn(nb_center, goal_center)
                f = tentative_g + h
                heapq.heappush(open_heap, (f, tentative_g, nb))

    return []  # 경로 없음
