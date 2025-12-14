import matplotlib.pyplot as plt

import sys
from pathlib import Path
# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from module.MyPlanning import smooth_path_nd, point_to_path_min_distance_nd

# ===== 테스트용 ㄱ자 경로 생성 =====
def make_L_path():
    """
    2D ㄱ자(직각) 경로 생성
    """
    path = []

    # (0,0) -> (5,0) : x 방향 직진
    for x in range(6):
        path.append([x, 0])

    # (5,0) -> (5,5) : y 방향 직진
    for y in range(1, 6):
        path.append([5, y])

    return path


def test_smoothing_2d():
    # 원본 경로
    path = make_L_path()

    # 스무딩 적용
    smooth_path = smooth_path_nd(
        path,
        alpha=0.2,
        beta=0.2,
        tol=1e-6,
        max_iter=10000,
        fix_ends=True
    )

    # x, y 분리 (plot용)
    x_orig = [p[0] for p in path]
    y_orig = [p[1] for p in path]

    x_s = [p[0] for p in smooth_path]
    y_s = [p[1] for p in smooth_path]

    #현 위치
    current_pos = [1.3, 0.8]  # (x,y)
    dmin, closest, seg_i, t = point_to_path_min_distance_nd(current_pos, smooth_path)


    # ===== 시각화 =====
    plt.figure(figsize=(6, 6))

    plt.scatter(
        [current_pos[0]],
        [current_pos[1]],
        c='green',
        s=100,
        label="Current Position"
    )
    plt.scatter(
        [closest[0]],
        [closest[1]],
        c='red',
        s=100,
        label="Closest Point on Path"
    )
    plt.plot(
        [current_pos[0], closest[0]],
        [current_pos[1], closest[1]],
        'k--',
        label=f"Min Distance = {dmin:.2f}"
    )
    plt.plot(
        x_orig, y_orig,
        'o--',
        label="Original Path (L-shape)",
        linewidth=2
    )

    plt.plot(
        x_s, y_s,
        'r-',
        label="Smoothed Path",
        linewidth=2
    )

    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title("2D Path Smoothing Test (L-shaped Path)")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.show()


# ===== 실행 =====
if __name__ == "__main__":
    test_smoothing_2d()
