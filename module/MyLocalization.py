# MyLocalization.py
from __future__ import annotations
from typing import List, Tuple, Optional
import math

def _raycast_first_hit(
    slam_map,
    origin: Tuple[float, float],
    angle_rad: float,
    max_range: float,
    step: float,
) -> float:
    """
    origin에서 angle 방향으로 레이를 쏴서, 처음 벽을 만날 때까지의 거리를 반환.
    벽을 못 만나면 max_range 반환.
    - slam_map.is_wall((x,y)) 사용
    """
    ox, oy = origin
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)

    dist = 0.0
    # step 간격으로 전진하면서 충돌 체크
    while dist <= max_range:
        x = ox + dx * dist
        y = oy + dy * dist
        if slam_map.is_wall((x, y)):  # SlamMap 판정 사용
            return dist
        dist += step

    return max_range


def _log_gaussian(err: float, sigma: float) -> float:
    # 정규분포 log-likelihood (상수항은 비교만 할 거면 생략 가능)
    return -0.5 * (err / sigma) ** 2


def localize_by_scan_grid(
    slam_map,
    z_meas: List[float],            # 센서 실제 측정 거리들(빔별)
    angles_rad: List[float],        # 센서 빔 각도들(라디안)
    max_range: float,
    *,
    resolution: float = 1.0,        # 격자 간격
    ray_step: Optional[float] = None,  # 레이 전진 step (기본 resolution/5 추천)
    sigma: float = 0.5,             # 센서 노이즈(거리) 표준편차
    theta_rad: float = 0.0,         # (선택) 로봇 heading. angles가 로봇 기준이면 여기로 회전 보정
):
    """
    반환:
      - best_xy: (x,y) 가장 확률 큰 격자 중심
      - belief: 2D grid 확률 (list of list)
      - meta: (xmin, ymin, nx, ny) 등 정보
    """
    if ray_step is None:
        ray_step = resolution / 5.0

    # 맵 영역을 limit로부터 가져옴 (maze_map이면 항상 있음)
    if slam_map.limit is None:
        raise ValueError("slam_map.limit가 필요해요. maze_map을 쓰거나 set_limit 하세용.")

    (x0, y0), (x1, y1) = slam_map.limit

    nx = int(math.ceil((x1 - x0) / resolution))
    ny = int(math.ceil((y1 - y0) / resolution))

    # log-belief로 누적 (언더플로 방지)
    log_belief = [[-1e18 for _ in range(nx)] for _ in range(ny)]

    best = (-1e18, (x0, y0))

    for iy in range(ny):
        cy = y0 + (iy + 0.5) * resolution
        for ix in range(nx):
            cx = x0 + (ix + 0.5) * resolution

            # 후보 위치가 벽 내부면 패스
            if slam_map.is_wall((cx, cy)):
                continue

            ll = 0.0
            for k, a in enumerate(angles_rad):
                pred = _raycast_first_hit(
                    slam_map,
                    origin=(cx, cy),
                    angle_rad=a + theta_rad,   # 로봇 자세 반영하고 싶으면 theta로 회전
                    max_range=max_range,
                    step=ray_step,
                )
                err = (z_meas[k] if z_meas[k] is not None else max_range) - pred
                ll += _log_gaussian(err, sigma)

            log_belief[iy][ix] = ll
            if ll > best[0]:
                best = (ll, (cx, cy))

    # log -> prob 정규화(softmax)
    max_ll = max(max(row) for row in log_belief)
    probs = []
    s = 0.0
    for iy in range(ny):
        row = []
        for ix in range(nx):
            v = log_belief[iy][ix]
            if v < -1e17:
                p = 0.0
            else:
                p = math.exp(v - max_ll)
            row.append(p)
            s += p
        probs.append(row)

    if s > 0:
        for iy in range(ny):
            for ix in range(nx):
                probs[iy][ix] /= s

    best_xy = best[1]
    meta = (x0, y0, nx, ny, resolution)
    return best_xy, probs, meta
