# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from module.MySlam import maze_map, print_map_plt

def cell_to_world(c, cell_size: float):
    cx, cy = c
    return ((cx + 0.5) * cell_size, (cy + 0.5) * cell_size)

def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

def simulate_landmark_ranges(true_xy, landmarks_xy, *, max_range=4.0, std=0.4, seed=0):
    """
    랜드마크는 거리만 측정한다고 가정(요구사항 std=0.4m).
    거리 > max_range면 관측 없음(None).
    """
    rng = np.random.default_rng(seed)
    x, y = true_xy
    z = []
    for (lx, ly) in landmarks_xy:
        r = math.hypot(lx - x, ly - y)
        if r > max_range:
            z.append(None)
        else:
            z.append(r + float(rng.normal(0, std)))
    return z

def estimate_xy_by_grid_search(m, z_meas, landmarks_xy, *, res=0.25, max_range=4.0, std=0.4):
    """
    맵 free 공간에서 (x,y) 후보를 그리드로 훑으면서
    랜드마크 거리 관측 likelihood가 최대인 점을 고르는 단순 위치추정.
    (과제용으로 직관적 + 튼튼)
    """
    (x0, y0), (x1, y1) = m.limit
    nx = int(math.ceil((x1 - x0) / res))
    ny = int(math.ceil((y1 - y0) / res))

    best_ll = -1e18
    best_xy = (m.start_point[0], m.start_point[1])

    for iy in range(ny):
        cy = y0 + (iy + 0.5) * res
        for ix in range(nx):
            cx = x0 + (ix + 0.5) * res

            # 벽/장애물 위는 후보 제외
            if m.is_wall((cx, cy), inclusive=True):
                continue

            ll = 0.0
            for meas, (lx, ly) in zip(z_meas, landmarks_xy):
                pred = math.hypot(lx - cx, ly - cy)

                # 관측 없음(None)은 "max_range로 클리핑된 값"처럼 취급(간단 버전)
                m2 = max_range if meas is None else float(meas)
                p2 = max_range if pred > max_range else float(pred)

                err = (m2 - p2)
                ll += -0.5 * (err / std) ** 2

            if ll > best_ll:
                best_ll = ll
                best_xy = (cx, cy)

    return best_xy, best_ll

def main():
    # -------------------------
    # 1) 과제 맵 + 랜드마크(장애물)
    # -------------------------
    maze = [  # x: 0..12
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # y=0
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # y=1
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # y=2
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # y=3
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # y=4
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0],  # y=5
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0],  # y=6
        [0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0],  # y=7
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # y=8
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # y=9
    ]
    start = (0, 9)
    end_g1 = (11, 9)
    end_g2 = (11, 1)

    land_cells = [(2, 6), (3, 4), (7, 2)]  # <= 최대 4개까지 가능
    cell_size = 5

    # ✅ 랜드마크 셀은 장애물: grid에서 1로 바꿈
    for (lx, ly) in land_cells:
        maze[ly][lx] = 1

    # 맵 생성(maze_map은 1을 장애물로 추가) :contentReference[oaicite:3]{index=3}
    m = maze_map(maze, start, end_g2, cell_size=cell_size)

    # 셀→월드(센터) 좌표 변환은 test__slam.py와 동일 방식 :contentReference[oaicite:4]{index=4}
    landmarks_xy = np.array([cell_to_world(c, cell_size) for c in land_cells], dtype=float)
    start_xy = m.start_point
    g1_xy = cell_to_world(end_g1, cell_size)
    g2_xy = m.end_point

    # -------------------------
    # 2) 맵 이미지 1장 저장
    # -------------------------
    print_map_plt(m, show=False)
    plt.scatter(landmarks_xy[:, 0], landmarks_xy[:, 1], marker="D", s=70, label="Landmarks")
    plt.scatter([g1_xy[0]], [g1_xy[1]], marker="x", s=70, label="G1")
    # plt.legend()
    out1 = Path(__file__).with_suffix("").as_posix() + "_map.png"
    plt.savefig(out1, dpi=160)
    plt.close()
    print(f"[OK] map saved: {out1}")

    # -------------------------
    # 3) (예시) 위치추정: 랜드마크 거리 관측으로 grid-search
    # -------------------------
    max_range = 4.0          # 요구사항(필요시 조정)
    std = 0.4                # 요구사항(가우시안 std)
    res = 0.25               # 후보 탐색 해상도(작을수록 정확↑/느림↑)

    # 예시로 "진짜 위치"를 start 근처 한 점으로 둠(원하면 로봇 주행 로그의 한 프레임으로 바꿔도 됨)
    true_xy = (start_xy[0] + 2.0, start_xy[1] - 1.0)

    z = simulate_landmark_ranges(true_xy, landmarks_xy, max_range=max_range, std=std, seed=2)
    est_xy, ll = estimate_xy_by_grid_search(m, z, landmarks_xy, res=res, max_range=max_range, std=std)

    # -------------------------
    # 4) 결과 이미지 1장 저장(진짜/추정 위치 표시)
    # -------------------------
    print_map_plt(m, show=False)
    plt.scatter(landmarks_xy[:, 0], landmarks_xy[:, 1], marker="D", s=70, label="Landmarks")
    plt.scatter([true_xy[0]], [true_xy[1]], marker="o", s=80, label="True pose")
    plt.scatter([est_xy[0]], [est_xy[1]], marker="x", s=80, label="Estimated pose")
    plt.title(f"Landmark localization (max_range={max_range}, std={std}, res={res})")
    # plt.legend()

    out2 = Path(__file__).with_suffix("").as_posix() + "_est.png"
    plt.savefig(out2, dpi=160)
    plt.close()
    print(f"[OK] est saved: {out2}")

    # 콘솔에도 좌표 출력
    print("\n--- 좌표 출력(월드 좌표) ---")
    print("start:", start_xy)
    print("g1   :", g1_xy)
    print("g2   :", g2_xy)
    print("landmarks:\n", landmarks_xy)
    print("true_xy:", true_xy)
    print("z(meas):", z)
    print("est_xy :", est_xy, "  (loglik=", ll, ")")

if __name__ == "__main__":
    main()
