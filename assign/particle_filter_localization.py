"""
Particle Filter 기반 로봇 위치 추정 (Localization)
- 센서: 반경 3m, 10도 단위 360도 거리 측정 (std=0.2m)
- 모션모델: 1m 단위 명령, 전후좌우 이동 (std=0.2)
- 로봇은 자신의 위치를 모르고 지도정보만 가짐
- Particle Filter로 위치 추정
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Polygon
from matplotlib.patches import Circle as PltCircle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from module.MySlam import maze_map


class ParticleFilterLocalizer:
    """Particle Filter 기반 로컬라이제이션"""
    
    def __init__(self, maze, cell_size, num_particles=500, sensor_range=3.0, 
                 sensor_std=0.2, motion_std=0.2):
        self.maze = maze
        self.cell_size = cell_size
        self.num_particles = num_particles
        self.sensor_range = sensor_range
        self.sensor_std = sensor_std
        self.motion_std = motion_std
        
        # 맵 경계
        self.map_width = len(maze[0]) * cell_size
        self.map_height = len(maze) * cell_size
        
        # 입자 초기화 (균등 분포)
        self.particles = self._initialize_particles()
        self.weights = np.ones(num_particles) / num_particles
        
        # 추정 위치
        self.estimated_pos = self._get_weighted_mean()
    
    def _initialize_particles(self):
        """장애물이 없는 영역에 입자 균등 분포"""
        particles = []
        attempts = 0
        max_attempts = self.num_particles * 10
        
        while len(particles) < self.num_particles and attempts < max_attempts:
            x = np.random.uniform(0, self.map_width)
            y = np.random.uniform(0, self.map_height)
            
            # 장애물 확인
            grid_x = int(x / self.cell_size)
            grid_y = int(y / self.cell_size)
            
            if (0 <= grid_x < len(self.maze[0]) and 
                0 <= grid_y < len(self.maze) and 
                self.maze[grid_y][grid_x] == 0):
                particles.append(np.array([x, y]))
            
            attempts += 1
        
        return np.array(particles)
    
    def predict(self, dx, dy):
        """모션 모델: 각 입자 이동 (노이즈 포함)"""
        for i in range(self.num_particles):
            # 실제 운동 명령
            noise_x = np.random.normal(0, self.motion_std)
            noise_y = np.random.normal(0, self.motion_std)
            
            new_x = self.particles[i, 0] + dx + noise_x
            new_y = self.particles[i, 1] + dy + noise_y
            
            # 맵 경계 내로 제한
            new_x = np.clip(new_x, 0, self.map_width)
            new_y = np.clip(new_y, 0, self.map_height)
            
            self.particles[i] = np.array([new_x, new_y])
    
    def update(self, sensor_data, true_pos):
        """센서 업데이트: 각 입자의 가중치 계산"""
        self.weights = np.ones(self.num_particles)
        
        # 각 입자에 대해 센서 데이터와의 유사도 계산
        for i in range(self.num_particles):
            # 이 입자에서 센서가 관측할 것으로 예상되는 데이터
            expected_sensor = self._simulate_sensor(self.particles[i])
            
            # 관측된 센서 데이터와의 차이로 가중치 계산
            likelihood = self._calculate_likelihood(sensor_data, expected_sensor)
            self.weights[i] *= likelihood
        
        # 가중치 정규화
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            self.weights = np.ones(self.num_particles) / self.num_particles
    
    def resample(self):
        """중요도 재샘플링 (Importance Resampling)"""
        # 누적 확률로 재샘플링
        indices = np.random.choice(self.num_particles, size=self.num_particles, 
                                   p=self.weights, replace=True)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles
    
    def _simulate_sensor(self, pos):
        """주어진 위치에서 센서가 측정할 값 시뮬레이션"""
        readings = []
        num_rays = 36  # 10도 단위: 360도 / 10 = 36개
        
        for k in range(num_rays):
            angle = 2 * math.pi * k / num_rays
            
            # 광선 추적으로 거리 계산
            distance = self._ray_cast(pos, angle)
            readings.append(distance)
        
        return np.array(readings)
    
    def _ray_cast(self, pos, angle):
        """광선 추적으로 거리 계산"""
        x, y = pos
        dx = math.cos(angle)
        dy = math.sin(angle)
        
        # 작은 스텝으로 광선 추적
        step_size = self.cell_size / 10
        distance = 0
        
        for _ in range(int(self.sensor_range / step_size) + 1):
            test_x = x + dx * distance
            test_y = y + dy * distance
            
            # 맵 범위 확인
            if not (0 <= test_x < self.map_width and 0 <= test_y < self.map_height):
                return self.sensor_range
            
            # 장애물 확인
            grid_x = int(test_x / self.cell_size)
            grid_y = int(test_y / self.cell_size)
            
            if self.maze[grid_y][grid_x] == 1:
                return distance
            
            distance += step_size
        
        return self.sensor_range
    
    def _calculate_likelihood(self, observed, expected):
        """가우시안 분포로 가능도 계산"""
        # 관측값과 예상값의 차이
        diff = np.array(observed) - np.array(expected)
        
        # 더 민감한 가능도 계산 (더 작은 분산 사용)
        variance = (self.sensor_std ** 2) * 2  # 더 작은 분산으로 더 강한 시그널
        likelihood = np.exp(-np.sum(diff ** 2) / (2 * variance))
        
        return max(likelihood, 1e-100)
    
    def _get_weighted_mean(self):
        """가중치를 고려한 평균 위치"""
        weighted_pos = np.average(self.particles, axis=0, weights=self.weights)
        return weighted_pos
    
    def update_estimated_pos(self):
        """추정 위치 업데이트"""
        self.estimated_pos = self._get_weighted_mean()
        return self.estimated_pos
    
    def get_estimated_pos(self):
        """추정 위치 반환"""
        return self.estimated_pos
    
    def get_uncertainty(self):
        """추정 불확실성 (입자의 분산)"""
        if len(self.particles) == 0:
            return 0
        variance = np.var(self.particles, axis=0)
        return math.sqrt(np.sum(variance))


def draw_obstacles(ax, maze, cell_size, alpha=0.3):
    """맵의 장애물 그리기"""
    for y in range(len(maze)):
        for x in range(len(maze[0])):
            if maze[y][x] == 1:
                min_x = x * cell_size
                min_y = y * cell_size
                max_x = (x + 1) * cell_size
                max_y = (y + 1) * cell_size
                ax.add_patch(Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                       alpha=alpha, facecolor='gray', edgecolor='black', linewidth=1))


def main():
    print("\n" + "="*70)
    print("Particle Filter 기반 로봇 위치 추정 (Localization)")
    print("="*70)
    
    # 맵 설정
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
    
    cell_size = 5
    
    # 센서 및 모션 설정
    sensor_range = 3.0
    sensor_std = 0.2
    motion_std = 0.2
    num_particles = 500
    
    print(f"\n설정:")
    print(f"  맵 크기: {len(maze[0])}x{len(maze)} 그리드 ({len(maze[0])*cell_size}x{len(maze)*cell_size}m)")
    print(f"  입자 개수: {num_particles}")
    print(f"  센서 반경: {sensor_range}m, 측정 오차: {sensor_std}m")
    print(f"  모션 오차: {motion_std}m, 명령 단위: 1m")
    
    # Particle Filter 초기화
    pf = ParticleFilterLocalizer(maze, cell_size, num_particles=num_particles,
                                 sensor_range=sensor_range, sensor_std=sensor_std,
                                 motion_std=motion_std)
    
    # 실제 로봇 위치 (P1: 좌상단)
    true_pos = np.array([2.5, 47.5])
    
    # 센서에서 측정할 범위 (10도 단위)
    num_rays = 36
    
    print(f"\n초기 상태:")
    print(f"  실제 위치 (P1): ({true_pos[0]:.1f}, {true_pos[1]:.1f})")
    print(f"  입자 범위: 전체 맵 (균등 분포)")
    print(f"  센서 광선: {num_rays}개 (10도 단위)")
    
    # 시뮬레이션 데이터 저장
    log = []
    
    # 움직임 명령: 우향 3m, 하향 10m
    commands = [
        (1.0, 0.0),   # 우향 1m
        (1.0, 0.0),   # 우향 1m
        (1.0, 0.0),   # 우향 1m
        (0.0, -1.0),  # 하향 1m
        (0.0, -1.0),  # 하향 1m
        (0.0, -1.0),  # 하향 1m
        (0.0, -1.0),  # 하향 1m
        (0.0, -1.0),  # 하향 1m
        (0.0, -1.0),  # 하향 1m
        (0.0, -1.0),  # 하향 1m
        (1.0, 0.0),   # 추가: 우향 1m
        (1.0, 0.0),   # 추가: 우향 1m
        (1.0, 0.0),   # 추가: 우향 1m
        (0.0, -1.0),  # 추가: 하향 1m
        (0.0, -1.0),  # 추가: 하향 1m
    ]
    
    print(f"\n시뮬레이션 시작...")
    print(f"{'Step':<6} {'True Pos':<20} {'Est Pos':<20} {'Error':<10} {'Unc':<10}")
    print("="*70)
    
    for step, (cmd_dx, cmd_dy) in enumerate(commands):
        # 실제 로봇 이동 (노이즈 포함)
        actual_noise_x = np.random.normal(0, motion_std)
        actual_noise_y = np.random.normal(0, motion_std)
        true_pos = true_pos + np.array([cmd_dx + actual_noise_x, cmd_dy + actual_noise_y])
        true_pos = np.clip(true_pos, [0, 0], [len(maze[0])*cell_size, len(maze)*cell_size])
        
        # 1. Predict: 입자들을 운동 모델로 이동
        pf.predict(cmd_dx, cmd_dy)
        
        # 2. 센서 데이터 생성 (실제 위치에서)
        sensor_readings = pf._simulate_sensor(true_pos)
        # 센서 노이즈 추가
        sensor_readings = sensor_readings + np.random.normal(0, sensor_std, len(sensor_readings))
        sensor_readings = np.clip(sensor_readings, 0, sensor_range)
        
        # 3. Update: 센서 데이터로 입자 가중치 업데이트
        pf.update(sensor_readings, true_pos)
        
        # 4. Resample: 가중치에 따라 입자 재샘플링
        pf.resample()
        
        # 5. 추정 위치 업데이트
        est_pos = pf.update_estimated_pos()
        
        # 불확실성 계산
        uncertainty = pf.get_uncertainty()
        error = np.linalg.norm(true_pos - est_pos)
        
        log.append({
            "step": step,
            "true_pos": true_pos.copy(),
            "est_pos": est_pos.copy(),
            "particles": pf.particles.copy(),
            "weights": pf.weights.copy(),
            "error": error,
            "uncertainty": uncertainty,
            "sensor_readings": sensor_readings.copy(),
        })
        
        print(f"{step:<6} ({true_pos[0]:6.2f}, {true_pos[1]:6.2f})   ({est_pos[0]:6.2f}, {est_pos[1]:6.2f})   "
              f"{error:6.3f}m   {uncertainty:6.3f}m")
    
    print("\n" + "="*70)
    print("시뮬레이션 완료, 애니메이션 생성 중...")
    
    # 애니메이션
    visualize_localization(maze, cell_size, log, sensor_range, num_rays)
    
    print("[OK] 애니메이션 저장: particle_filter_localization.gif")
    print("\n" + "="*70)


def visualize_localization(maze, cell_size, log, sensor_range, num_rays):
    """Particle Filter 로컬라이제이션 시각화"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 왼쪽: 지도 + 입자 + 위치
    ax1 = axes[0]
    ax1.set_title("Particle Filter Localization", fontsize=14, fontweight='bold')
    ax1.set_aspect("equal")
    ax1.set_xlim(-2, len(maze[0]) * cell_size + 2)
    ax1.set_ylim(-2, len(maze) * cell_size + 2)
    ax1.invert_yaxis()
    
    draw_obstacles(ax1, maze, cell_size)
    
    # 입자들 (산점도)
    particle_scatter = ax1.scatter([], [], c='lightblue', s=10, alpha=0.5, label='Particles')
    
    # 센서 범위 (원)
    sensor_circle = PltCircle((0, 0), sensor_range, fill=False, edgecolor='green', 
                              linestyle=':', linewidth=2, alpha=0.5)
    ax1.add_patch(sensor_circle)
    
    # 센서 광선
    ray_lines = [ax1.plot([], [], 'g-', linewidth=0.5, alpha=0.3)[0] for _ in range(num_rays)]
    
    # 추정 위치
    est_pt, = ax1.plot([], [], 'ro', markersize=12, label='Estimated pos', zorder=5)
    
    # 실제 위치
    true_pt, = ax1.plot([], [], 'b*', markersize=20, label='True pos', zorder=5)
    
    # 추정 궤적
    est_trace, = ax1.plot([], [], 'r--', linewidth=1.5, alpha=0.7, label='Est trajectory')
    
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    
    # 오른쪽: 에러 그래프
    ax2 = axes[1]
    ax2.set_title("Localization Error & Uncertainty", fontsize=14, fontweight='bold')
    error_line, = ax2.plot([], [], 'r-o', linewidth=2, markersize=4, label='Estimation Error')
    uncertainty_line, = ax2.plot([], [], 'b--', linewidth=2, label='Uncertainty')
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Distance (m)")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    def init():
        particle_scatter.set_offsets(np.empty((0, 2)))
        est_pt.set_data([], [])
        true_pt.set_data([], [])
        est_trace.set_data([], [])
        error_line.set_data([], [])
        uncertainty_line.set_data([], [])
        sensor_circle.set_center((0, 0))
        
        for line in ray_lines:
            line.set_data([], [])
        
        return [particle_scatter, est_pt, true_pt, est_trace, error_line, 
                uncertainty_line, sensor_circle] + ray_lines
    
    def update(frame):
        data = log[frame]
        
        # 입자 표시
        particle_scatter.set_offsets(data["particles"])
        
        # 위치 표시
        true_x, true_y = data["true_pos"]
        est_x, est_y = data["est_pos"]
        
        true_pt.set_data([true_x], [true_y])
        est_pt.set_data([est_x], [est_y])
        
        # 추정 궤적
        est_positions = np.array([log[i]["est_pos"] for i in range(frame + 1)])
        est_trace.set_data(est_positions[:, 0], est_positions[:, 1])
        
        # 센서 범위 업데이트
        sensor_circle.set_center((est_x, est_y))
        
        # 센서 광선 표시 (추정 위치에서)
        sensor_readings = data["sensor_readings"]
        for k in range(num_rays):
            angle = 2 * math.pi * k / num_rays
            distance = sensor_readings[k]
            
            end_x = est_x + math.cos(angle) * distance
            end_y = est_y + math.sin(angle) * distance
            
            ray_lines[k].set_data([est_x, end_x], [est_y, end_y])
        
        # 에러 그래프
        errors = np.array([log[i]["error"] for i in range(frame + 1)])
        uncertainties = np.array([log[i]["uncertainty"] for i in range(frame + 1)])
        steps = np.arange(frame + 1)
        
        error_line.set_data(steps, errors)
        uncertainty_line.set_data(steps, uncertainties)
        
        ax2.set_xlim(-0.5, len(log) - 0.5)
        ax2.set_ylim(0, max(np.max(errors), np.max(uncertainties)) * 1.2)
        
        return [particle_scatter, est_pt, true_pt, est_trace, error_line, 
                uncertainty_line, sensor_circle] + ray_lines
    
    ani = FuncAnimation(fig, update, frames=len(log), init_func=init,
                       interval=500, blit=True, repeat=False)
    
    plt.tight_layout()
    ani.save("particle_filter_localization.gif", writer="pillow", fps=2)
    plt.close()


if __name__ == "__main__":
    main()
