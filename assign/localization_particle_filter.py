"""
Particle Filter 기반 로컬라이제이션
- 로봇이 자신의 위치를 모른 상태에서 시작
- Circle_Sensor로 센싱하며 지도와 매칭
- Particle Filter를 사용하여 위치 추정
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from scipy.spatial.distance import cdist

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import module.MySlam as MySlam
from module.MySlam import maze_map
from module.MySensor import Circle_Sensor
from module.MyControl import SimpleBicycleModel
from module.MyPlanning import PIDVec
from module.MyRobot_with_localization import Moblie_robot
import numpy as np


def draw_obstacles(ax, slam_map: MySlam, *, alpha=0.20):
    for obs in slam_map.get_obs_list():
        (min_x, min_y), (max_x, max_y) = obs.bounds()
        ax.add_patch(Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, alpha=alpha))


class ParticleFilterLocalizer:
    """Particle Filter를 사용한 로컬라이제이션"""
    
    def __init__(self, slam_map, num_particles=500, motion_noise_std=0.1, sensor=None):
        self.slam_map = slam_map
        self.num_particles = num_particles
        self.motion_noise_std = motion_noise_std
        self.sensor = sensor
        
        # 파티클 초기화 (자신의 위치를 모르므로 지도 내 자유 공간에 균등 분포)
        self.particles = self._init_particles()
        self.weights = np.ones(num_particles) / num_particles
        
        # 파티클 히스토리
        self.particle_history = [self.particles.copy()]
        self.weight_history = [self.weights.copy()]
        self.estimated_pose_history = [self._estimate_pose()]
        
    def _init_particles(self):
        """자유 공간에 파티클 초기화 (위치를 모른 상태)"""
        particles = []
        obs_list = self.slam_map.get_obs_list()
        
        # 지도의 한계
        if self.slam_map.limit is not None:
            (x_min, y_min), (x_max, y_max) = self.slam_map.limit
        else:
            x_min, y_min, x_max, y_max = -10, -10, 100, 100
        
        # 충돌 없는 위치에 파티클 배치
        attempts = 0
        while len(particles) < self.num_particles and attempts < self.num_particles * 10:
            x = np.random.uniform(x_min + 1, x_max - 1)
            y = np.random.uniform(y_min + 1, y_max - 1)
            
            # 장애물과의 충돌 확인
            in_collision = False
            for obs in obs_list:
                (ox_min, oy_min), (ox_max, oy_max) = obs.bounds()
                if ox_min - 0.5 <= x <= ox_max + 0.5 and oy_min - 0.5 <= y <= oy_max + 0.5:
                    in_collision = True
                    break
            
            if not in_collision:
                theta = np.random.uniform(-math.pi, math.pi)
                particles.append([x, y, theta])
            
            attempts += 1
        
        return np.array(particles)
    
    def predict(self, command_dx, command_dy):
        """모션 모델 기반 예측 (노이즈 포함)"""
        for i in range(self.num_particles):
            x, y, theta = self.particles[i]
            
            # 모션 노이즈 추가
            dx = command_dx + np.random.normal(0, self.motion_noise_std)
            dy = command_dy + np.random.normal(0, self.motion_noise_std)
            
            # 위치 업데이트
            self.particles[i][0] = x + dx
            self.particles[i][1] = y + dy
            
            # 방향 업데이트
            if abs(command_dx) > 0.01 or abs(command_dy) > 0.01:
                new_theta = math.atan2(dy, dx)
                self.particles[i][2] = new_theta + np.random.normal(0, 0.05)
    
    def update(self, sensor_measurements):
        """센서 측정값 기반 가중치 업데이트"""
        if sensor_measurements is None or len(sensor_measurements) == 0:
            return
        
        sensor_dist = self.sensor.get_distance()
        num_rays = self.sensor.get_number()
        
        # 각 파티클의 가중치 계산
        for i in range(self.num_particles):
            px, py, ptheta = self.particles[i]
            
            # 해당 위치에서의 예상 센서 측정값 계산
            expected_measurements = self._compute_expected_measurement(px, py, ptheta, num_rays, sensor_dist)
            
            # 실제 측정값과의 차이로 가중치 계산 (가우시안 모델)
            likelihood = 0.0
            measurement_noise_std = 0.3  # 센서 측정 노이즈
            
            for j in range(len(sensor_measurements)):
                if j < len(expected_measurements):
                    expected = expected_measurements[j]
                    actual = sensor_measurements[j] if sensor_measurements[j] is not None else sensor_dist
                    
                    # 가우시안 likelihood
                    diff = abs(actual - expected)
                    likelihood += math.exp(-0.5 * (diff / measurement_noise_std) ** 2)
            
            self.weights[i] *= likelihood + 1e-6  # 분자 영점 방지
        
        # 가중치 정규화
        self.weights /= (np.sum(self.weights) + 1e-6)
    
    def _compute_expected_measurement(self, px, py, ptheta, num_rays, max_range):
        """주어진 위치에서의 예상 센서 측정값 계산"""
        expected = []
        obs_list = self.slam_map.get_obs_list()
        
        for k in range(num_rays):
            angle = 2 * math.pi * k / num_rays + ptheta
            
            # 레이 추적 (ray casting)
            min_dist = max_range
            dx = math.cos(angle)
            dy = math.sin(angle)
            
            # 각 장애물과의 교점 확인
            for obs in obs_list:
                (ox_min, oy_min), (ox_max, oy_max) = obs.bounds()
                
                # 간단한 거리 계산 (박스와의 거리)
                dist = self._ray_box_distance(px, py, dx, dy, ox_min, oy_min, ox_max, oy_max)
                if dist < min_dist:
                    min_dist = dist
            
            expected.append(min_dist)
        
        return expected
    
    def _ray_box_distance(self, px, py, dx, dy, box_x_min, box_y_min, box_x_max, box_y_max):
        """레이와 박스 사이의 거리"""
        # 간단한 구현 (더 정확한 ray-box intersection이 필요하면 추가)
        # 박스의 각 면과의 교차점 확인
        
        min_dist = float('inf')
        
        # 박스의 네 개 면 확인
        # 오른쪽 면 (x = box_x_max)
        if abs(dx) > 1e-6:
            t = (box_x_max - px) / dx
            if t > 0:
                hit_y = py + t * dy
                if box_y_min <= hit_y <= box_y_max:
                    min_dist = min(min_dist, t)
        
        # 왼쪽 면 (x = box_x_min)
        if abs(dx) > 1e-6:
            t = (box_x_min - px) / dx
            if t > 0:
                hit_y = py + t * dy
                if box_y_min <= hit_y <= box_y_max:
                    min_dist = min(min_dist, t)
        
        # 위쪽 면 (y = box_y_max)
        if abs(dy) > 1e-6:
            t = (box_y_max - py) / dy
            if t > 0:
                hit_x = px + t * dx
                if box_x_min <= hit_x <= box_x_max:
                    min_dist = min(min_dist, t)
        
        # 아래쪽 면 (y = box_y_min)
        if abs(dy) > 1e-6:
            t = (box_y_min - py) / dy
            if t > 0:
                hit_x = px + t * dx
                if box_x_min <= hit_x <= box_x_max:
                    min_dist = min(min_dist, t)
        
        return min_dist if min_dist < float('inf') else 15.0
    
    def resample(self):
        """리샘플링 (중요도 리샘플링)"""
        # 누적 가중치
        cumsum = np.cumsum(self.weights)
        
        # 새로운 파티클 인덱스 선택
        indices = np.searchsorted(cumsum, np.random.rand(self.num_particles))
        indices = np.clip(indices, 0, self.num_particles - 1)
        
        # 파티클 리샘플링
        self.particles = self.particles[indices].copy()
        self.weights = np.ones(self.num_particles) / self.num_particles
    
    def _estimate_pose(self):
        """파티클들로부터 위치 추정 (가중 평균)"""
        weighted_x = np.sum(self.weights * self.particles[:, 0])
        weighted_y = np.sum(self.weights * self.particles[:, 1])
        weighted_theta = np.arctan2(
            np.sum(self.weights * np.sin(self.particles[:, 2])),
            np.sum(self.weights * np.cos(self.particles[:, 2]))
        )
        return np.array([weighted_x, weighted_y, weighted_theta])
    
    def get_estimated_pose(self):
        """현재 추정 위치"""
        return self._estimate_pose()
    
    def record_snapshot(self):
        """현재 상태 저장"""
        self.particle_history.append(self.particles.copy())
        self.weight_history.append(self.weights.copy())
        self.estimated_pose_history.append(self._estimate_pose())


def run_localization_scenario(maze_map, start_pos, commands, sensor):
    """로컬라이제이션 시뮬레이션 실행"""
    
    pf = ParticleFilterLocalizer(maze_map, num_particles=300, motion_noise_std=0.1, sensor=sensor)
    
    # 실제 로봇 상태 (시뮬레이션용)
    true_pos = np.array(start_pos, dtype=float)
    true_theta = 0.0
    
    # 로그
    log = []
    
    print(f"로컬라이제이션 시작...")
    print(f"실제 시작 위치 (비공개): {true_pos}")
    print(f"파티클 초기 위치: 맵 내 자유공간에 균등 분포")
    
    for step, (cmd_dx, cmd_dy) in enumerate(commands):
        print(f"\n=== Step {step}: Command ({cmd_dx}, {cmd_dy}) ===")
        
        # 1) 실제 로봇 움직임 (노이즈 포함)
        actual_dx = cmd_dx + np.random.normal(0, 0.1)
        actual_dy = cmd_dy + np.random.normal(0, 0.1)
        true_pos[0] += actual_dx
        true_pos[1] += actual_dy
        
        # 2) 파티클 예측
        pf.predict(actual_dx, actual_dy)
        
        # 3) 센싱 (실제 위치에서)
        sensor_measurements = sensor.sensing(true_pos)
        
        # 4) 파티클 가중치 업데이트
        pf.update(sensor_measurements)
        
        # 5) 리샘플링
        if step % 3 == 0:  # 3 스텝마다 리샘플링
            pf.resample()
        
        # 6) 위치 추정
        estimated_pose = pf.get_estimated_pose()
        error = np.linalg.norm(true_pos - estimated_pose[:2])
        
        print(f"실제 위치: ({true_pos[0]:.2f}, {true_pos[1]:.2f})")
        print(f"추정 위치: ({estimated_pose[0]:.2f}, {estimated_pose[1]:.2f})")
        print(f"위치오차: {error:.3f}m")
        print(f"센싱 광선: {len([z for z in sensor_measurements if z < 14.9])}개 hit")
        
        # 7) 스냅샷 저장
        pf.record_snapshot()
        log.append({
            "step": step,
            "true_pos": true_pos.copy(),
            "est_pos": estimated_pose.copy(),
            "particles": pf.particles.copy(),
            "weights": pf.weights.copy(),
            "error": error,
            "sensor_meas": sensor_measurements.copy() if sensor_measurements is not None else None,
        })
    
    return pf, log


def visualize_localization(maze_map, pf, log, sensor):
    """로컬라이제이션 과정 시각화"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    
    # ===== 첫 번째 그래프: 전체 로컬라이제이션 과정 =====
    ax1.set_title("Localization Process (Particle Filter)")
    ax1.set_aspect("equal")
    
    if maze_map.limit is not None:
        (x0, y0), (x1, y1) = maze_map.limit
        ax1.set_xlim(x0 - 2, x1 + 2)
        ax1.set_ylim(y0 - 2, y1 + 2)
    
    draw_obstacles(ax1, maze_map, alpha=0.3)
    
    # 시작점
    sx, sy = maze_map.start_point
    ax1.plot([sx], [sy], "g*", markersize=15, label="Start (known)", zorder=5)
    
    # 실제 궤적
    true_positions = np.array([entry["true_pos"] for entry in log])
    ax1.plot(true_positions[:, 0], true_positions[:, 1], "r-", linewidth=2, label="True trajectory", zorder=3)
    ax1.plot(true_positions[-1, 0], true_positions[-1, 1], "rs", markersize=8, zorder=5)
    
    # 추정 궤적
    est_positions = np.array([entry["est_pos"][:2] for entry in log])
    ax1.plot(est_positions[:, 0], est_positions[:, 1], "b--", linewidth=2, label="Estimated trajectory", zorder=3)
    ax1.plot(est_positions[-1, 0], est_positions[-1, 1], "bs", markersize=8, zorder=5)
    
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # ===== 두 번째 그래프: 마지막 스텝의 파티클 분포 =====
    ax2.set_title(f"Particle Distribution (Step {len(log)-1})")
    ax2.set_aspect("equal")
    
    if maze_map.limit is not None:
        ax2.set_xlim(x0 - 2, x1 + 2)
        ax2.set_ylim(y0 - 2, y1 + 2)
    
    draw_obstacles(ax2, maze_map, alpha=0.3)
    
    last_particles = log[-1]["particles"]
    last_weights = log[-1]["weights"]
    
    # 파티클 시각화 (가중치에 따라 색상과 크기 결정)
    scatter = ax2.scatter(
        last_particles[:, 0],
        last_particles[:, 1],
        c=last_weights,
        s=last_weights * 1000 + 10,
        alpha=0.6,
        cmap="viridis"
    )
    plt.colorbar(scatter, ax=ax2, label="Weight")
    
    # 추정 위치
    est_pose = log[-1]["est_pos"]
    ax2.plot([est_pose[0]], [est_pose[1]], "r*", markersize=20, label="Estimated pose", zorder=5)
    
    # 실제 위치
    true_pos = log[-1]["true_pos"]
    ax2.plot([true_pos[0]], [true_pos[1]], "g+", markersize=15, markeredgewidth=2, label="True pose", zorder=5)
    
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # ===== 세 번째 그래프: 위치 오차 추이 =====
    ax3.set_title("Localization Error over Time")
    errors = np.array([entry["error"] for entry in log])
    ax3.plot(errors, "b-o", linewidth=2, markersize=6)
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Error (m)")
    ax3.grid(True, alpha=0.3)
    
    # ===== 네 번째 그래프: 파티클 가중치 분포 =====
    ax4.set_title("Particle Weights Distribution")
    weights = log[-1]["weights"]
    ax4.hist(weights, bins=30, color="blue", alpha=0.7, edgecolor="black")
    ax4.set_xlabel("Weight")
    ax4.set_ylabel("Number of Particles")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("localization_pf_result.png", dpi=150, bbox_inches="tight")
    print("\n✓ 결과 이미지 저장: localization_pf_result.png")
    plt.show()


def main():
    print("=" * 60)
    print("Particle Filter 기반 로봇 로컬라이제이션")
    print("=" * 60)
    
    # 1) 맵 설정
    maze = [
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    
    start = (0, 9)
    end = (11, 1)
    cell_size = 5
    m = maze_map(maze, start, end, cell_size=cell_size)
    
    print(f"맵 생성 완료: {len(maze)}x{len(maze[0])} 그리드")
    print(f"시작점: {start}, 끝점: {end}")
    
    # 2) 센서 설정
    sensor = Circle_Sensor(dim=2, number=16, distance=15.0, slam_map=m, step=0.1)
    print(f"센서 설정: {sensor.get_number()}개 광선, 최대거리 {sensor.get_distance()}m")
    
    # 3) 로봇 명령 (1m 단위 이동)
    commands = [
        (1.0, 0.0),   # 오른쪽
        (1.0, 0.0),   # 오른쪽
        (1.0, 0.0),   # 오른쪽
        (0.0, -1.0),  # 위쪽
        (0.0, -1.0),  # 위쪽
        (1.0, 0.0),   # 오른쪽
        (1.0, 0.0),   # 오른쪽
        (0.0, -1.0),  # 위쪽
    ]
    
    print(f"\n명령 수열: {len(commands)}개 이동")
    
    # 4) 로컬라이제이션 시뮬레이션
    pf, log = run_localization_scenario(m, start, commands, sensor)
    
    # 5) 결과 시각화
    print("\n시각화 중...")
    visualize_localization(m, pf, log, sensor)
    
    print("\n" + "=" * 60)
    print("로컬라이제이션 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
