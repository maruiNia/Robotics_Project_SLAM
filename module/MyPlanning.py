from __future__ import annotations
from typing import List, Sequence, Tuple, Union

Number = Union[int, float]


def smooth_path_nd(
    points: Sequence[Sequence[Number]],
    *,
    alpha: float = 0.1,
    beta: float = 0.3,
    tol: float = 1e-6,
    max_iter: int = 10000,
    fix_ends: bool = True,
) -> List[List[float]]:
    """
    N차원 경로 스무딩 (Gradient Descent)

    points: 길이 N의 경로 점들, 각 점은 dim차원 벡터
            예) 2D: [[x0,y0], [x1,y1], ...]
                3D: [[x0,y0,z0], ...]
                6D: [[q1,q2,q3,q4,q5,q6], ...]

    업데이트(슬라이드 방식 일반화):
      Y_i = Y_i + alpha*(X_i - Y_i)                      # data term
      Y_i = Y_i + beta*(Y_{i+1} + Y_{i-1} - 2*Y_i)       # smoothness term

    반환: 스무딩된 점 리스트 (List[List[float]])
    """
    if not points:
        return []

    n = len(points)
    dim = len(points[0])
    if dim == 0:
        raise ValueError("각 점의 차원(dim)이 0이에요.")

    # 모든 점이 동일 차원인지 검사
    for i, p in enumerate(points):
        if len(p) != dim:
            raise ValueError(f"{i}번째 점의 차원이 달라요. expected dim={dim}, got={len(p)}")

    if n < 2:
        return [[float(v) for v in points[0]]]

    # 원본 X, 초기 Y
    X = [[float(v) for v in p] for p in points]
    Y = [p[:] for p in X]  # 시작은 원본 그대로

    start_i = 1 if fix_ends else 0
    end_i = n - 1 if fix_ends else n

    for _ in range(max_iter):
        total_change = 0.0

        for i in range(start_i, end_i):
            im1 = max(0, i - 1)
            ip1 = min(n - 1, i + 1)

            old = Y[i][:]

            # data term + smoothness term를 "모든 차원"에 대해 적용
            for d in range(dim):
                y = Y[i][d]
                y += alpha * (X[i][d] - y)
                y += beta * (Y[ip1][d] + Y[im1][d] - 2.0 * y)
                Y[i][d] = y

            total_change += sum(abs(old[d] - Y[i][d]) for d in range(dim))

        if total_change < tol:
            break

    return Y
