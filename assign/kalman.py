# -*- coding: utf-8 -*-
import math
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가

sys.path.insert(0, str(Path(__file__).parent.parent))
from module.MySlam import maze_map, print_map_plt
from module.MySensor import Circle_Sensor, gaussian_noise


# -------------------------
# 유틸
# -------------------------
def cell_to_world(cell, cell_size: float):
    cx, cy = cell
    return ((cx + 0.5) * cell_size, (cy + 0.5) * cell_size)


def systematic_resample(particles, weights, rng):
    """Systematic resampling (분산 작고 안정적)"""
    N = len(particles)
    positions = (rng.random() + np.arange(N)) / N
    indexes = np.zeros(N, dtype=int)

    cumsum = np.cumsum(weights)
    i = 0
    j = 0
    while i < N:
        if positions[i] < cumsum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    new_particles = particles[indexes].copy()
    return new_particles


def sensor_loglik(z_meas, z_pred, sigma=0.4, max_range=4.0):
    """
    Circle_Sensor 출력: 각 ray의 거리 or None
    간단 likelihood:
      - None을 max_range로 클리핑해서 비교
      - 가우시안 오차 모델
    """
    ll = 0.0
    inv2 = 1.0 / (2.0 * sigma * sigma)

    for m, p in zip(z_meas, z_pred):
        m2 = max_range if m is None else float(m)
        p2 = max_range if (p is None or p > max_range) else float(p)
        err = m2 - p2
        ll += -(err * err) * inv2
    return ll


# -------------------------
# Particle Filter (MCL)
# -------------------------
class ParticleFilter2D:
    def __init__(self, slam_map, sensor: Circle_Sensor, *,
                 N=1200,
                 motion_std=0.1,     # ✅ 모션모델 불완전(std=0.1)
                 meas_std=0.4,       # 센싱 가우시안 std (과제 조건)
                 max_range=4.0,
                 seed=0):
        self.map = slam_map
        self.sensor = sensor
        self.N = int(N)
        self.motion_std = float(motion_std)
        self.meas_std = float(meas_std)
        self.max_range = float(max_range)
        self.rng = np.random.default_rng(seed)

        self.particles = self._init_uniform_free()

    def _init_uniform_free(self):
        """로봇이 위치를 모른다 → free-space에 균등 초기화"""
        (x0, y0), (x1, y1) = self.map.limit

        pts = []
        # rejection sampling
        while len(pts) < self.N:
            x = float(self.rng.uniform(x0, x1))
            y = float(self.rng.uniform(y0, y1))
            if not self.map.is_wall((x, y), inclusive=True):
                pts.append((x, y))
        return np.array(pts, dtype=float)

    def predict(self, u_dx, u_dy):
        """
        ✅ 1m 단위 명령(전후좌우): (u_dx, u_dy) ∈ {(±1,0),(0,±1)}
        모션 노이즈: N(0, motion_std)
        """
        noise = self.rng.normal(0.0, self.motion_std, size=(self.N, 2))
        prop = self.particles + np.array([u_dx, u_dy], dtype=float) + noise

        # 벽이면 이동 무효(그 자리 유지) 처리
        for i in range(self.N):
            if self.map.is_wall((prop[i, 0], prop[i, 1]), inclusive=True):
                prop[i] = self.particles[i]
        self.particles = prop

    def update(self, z_meas):
        """센서 likelihood로 weight 계산 후 resample"""
        logw = np.empty(self.N, dtype=float)

        for i in range(self.N):
            z_pred = self.sensor.sensing((self.particles[i, 0], self.particles[i, 1]))
            logw[i] = sensor_loglik(z_meas, z_pred, sigma=self.meas_std, max_range=self.max_range)

        # softmax 안정화
        m = float(np.max(logw))
        w = np.exp(logw - m)
        s = float(np.sum(w))
        if s <= 1e-12:
            # 최악의 경우: 다시 초기화
            self.particles = self._init_uniform_free()
            return

        w /= s
        self.particles = systematic_resample(self.particles, w, self.rng)

    def estimate(self):
        """추정 위치: 파티클 평균"""
        mean = np.mean(self.particles, axis=0)
        return float(mean[0]), float(mean[1])


# -------------------------
# 데모 실행 + 시각화
# -------------------------
def main():
    # -------------------------
    # 1) 과제 맵 (랜드마크가 있으면 그 셀은 1로 바꿔 장애물화)
    # -------------------------
    maze = [  # y=0..9, x=0..12
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 0
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 1
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 2
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # 3
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0],  # 5
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0],  # 6
        [0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0],  # 7
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 8
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 9
    ]

    # P1 = start (로봇은 이 좌표를 "모른다"는 설정이지만, 시뮬레이션에서는 true로 씀)
    start_cell = (0, 9)
    end_g2 = (11, 1)

    # ✅ 1m 단위 명령을 쓰려면 cell_size=1이 가장 직관적
    cell_size = 1.0

    m = maze_map(maze, start_cell, end_g2, cell_size=cell_size)

    # -------------------------
    # 2) Circle_Sensor 준비 (fake_sensor 말고 이거!)
    # -------------------------
    max_range = 4.0
    n_rays = 36
    sensor = Circle_Sensor(dim=2, number=n_rays, distance=max_range, slam_map=m, step=0.05, resolution=1)
    # 센서 노이즈(가우시안 std=0.4m) 적용
    sensor.set_noise(gaussian_noise(sigma=0.4))

    # -------------------------
    # 3) Particle Filter 생성
    # -------------------------
    pf = ParticleFilter2D(
        m, sensor,
        N=1400,
        motion_std=0.1,  # ✅ 모션 노이즈 std=0.1
        meas_std=0.4,
        max_range=max_range,
        seed=7
    )

    # -------------------------
    # 4) 시뮬레이션: "진짜 로봇"은 P1에서 시작
    #    (명령은 1m씩: 전후좌우)
    # -------------------------
    true_xy = list(m.start_point)

    # 예시 명령 시퀀스(필요하면 네 플래너 출력으로 대체 가능)
    # (dx, dy): right = (+1,0), left=(-1,0), up=(0,-1), down=(0,+1)
    commands = []
    commands += [(+1, 0)] * 6
    commands += [(0, -1)] * 3
    commands += [(+1, 0)] * 4
    commands += [(0, -1)] * 2
    commands += [(-1, 0)] * 2
    commands += [(0, -1)] * 1
    commands += [(+1, 0)] * 5

    # -------------------------
    # 5) 애니메이션 준비
    # -------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_aspect("equal", adjustable="box")

    # 배경 맵 1번만 그려두고, 위에 scatter만 업데이트
    print_map_plt(m, show=False)
    ax = plt.gca()

    scat_particles = ax.scatter([], [], s=4)
    scat_true = ax.scatter([], [], s=80, marker="o", label="True")
    scat_est = ax.scatter([], [], s=80, marker="x", label="Estimated")
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    ax.legend(loc="upper right")

    history_true = []
    history_est = []

    def step_true_motion(u_dx, u_dy):
        """진짜 로봇도 모션노이즈(std=0.1)로 움직인다고 가정"""
        nx = true_xy[0] + u_dx + random.gauss(0.0, 0.1)
        ny = true_xy[1] + u_dy + random.gauss(0.0, 0.1)
        if not m.is_wall((nx, ny), inclusive=True):
            true_xy[0], true_xy[1] = nx, ny

    def update(frame):
        nonlocal history_true, history_est
        if frame < len(commands):
            u_dx, u_dy = commands[frame]

            # (1) true motion
            step_true_motion(u_dx, u_dy)

            # (2) PF predict
            pf.predict(u_dx, u_dy)

            # (3) sensing from Circle_Sensor at true position
            z = sensor.sensing((true_xy[0], true_xy[1]))

            # (4) PF update
            pf.update(z)

        ex, ey = pf.estimate()
        history_true.append((true_xy[0], true_xy[1]))
        history_est.append((ex, ey))

        # 파티클 표시
        scat_particles.set_offsets(pf.particles)

        # true / est 표시
        scat_true.set_offsets([[true_xy[0], true_xy[1]]])
        scat_est.set_offsets([[ex, ey]])

        txt.set_text(
            f"step={frame}/{len(commands)}\n"
            f"true=({true_xy[0]:.2f},{true_xy[1]:.2f})\n"
            f"est =({ex:.2f},{ey:.2f})"
        )
        return scat_particles, scat_true, scat_est, txt

    anim = FuncAnimation(fig, update, frames=len(commands) + 1, interval=120, blit=False)

    out_gif = "pf_circle_localization.gif"
    anim.save(out_gif, writer="pillow", fps=8)
    plt.close(fig)

    print(f"[OK] saved: {out_gif}")


if __name__ == "__main__":
    main()
