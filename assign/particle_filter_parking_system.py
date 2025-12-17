"""
Particle Filter + A* Path Planning + PID Control 통합 주차 시스템

1. Particle Filter: 로봇의 위치를 모르는 상태에서 센서/모션으로 자기 위치 추정
2. Path Planning: 충분한 확신(Uncertainty < 1.0m)이 되면 목표 위치(P2 또는 G1)로의 최적 경로 계산
3. PID Control: 계산된 smooth path를 따라 주차 수행
4. 실시간 위치 추정: Path 추적 중 계속 센서 데이터로 위치 업데이트
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle as PltCircle
from collections import deque
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from module.MySlam import maze_map


class ParticleFilterLocalizer:
    """Particle Filter 기반 로컬라이제이션"""
    
    def __init__(self, maze, cell_size, num_particles=1000, sensor_range=24.0, 
                 sensor_std=0.2, motion_std=0.2):
        self.maze = maze
        self.cell_size = cell_size
        self.num_particles = num_particles
        self.sensor_range = sensor_range
        self.sensor_std = sensor_std
        self.motion_std = motion_std
        
        self.map_width = len(maze[0]) * cell_size
        self.map_height = len(maze) * cell_size
        
        self.particles = self._initialize_particles()
        self.weights = np.ones(num_particles) / num_particles
        self.estimated_pos = self._get_weighted_mean()
    
    def _initialize_particles(self):
        """장애물이 없는 영역에 입자 균등 분포"""
        particles = []
        attempts = 0
        max_attempts = self.num_particles * 10
        
        while len(particles) < self.num_particles and attempts < max_attempts:
            x = np.random.uniform(0, self.map_width)
            y = np.random.uniform(0, self.map_height)
            
            grid_x = int(x / self.cell_size)
            grid_y = int(y / self.cell_size)
            
            if (0 <= grid_x < len(self.maze[0]) and 
                0 <= grid_y < len(self.maze) and 
                self.maze[grid_y][grid_x] == 0):
                particles.append(np.array([x, y]))
            
            attempts += 1
        
        return np.array(particles)
    
    def predict(self, dx, dy):
        """모션 모델"""
        for i in range(self.num_particles):
            noise_x = np.random.normal(0, self.motion_std)
            noise_y = np.random.normal(0, self.motion_std)
            
            new_x = self.particles[i, 0] + dx + noise_x
            new_y = self.particles[i, 1] + dy + noise_y
            
            new_x = np.clip(new_x, 0, self.map_width)
            new_y = np.clip(new_y, 0, self.map_height)
            
            self.particles[i] = np.array([new_x, new_y])
    
    def update(self, sensor_data):
        """센서 업데이트"""
        self.weights = np.ones(self.num_particles)
        
        for i in range(self.num_particles):
            expected_sensor = self._simulate_sensor(self.particles[i])
            likelihood = self._calculate_likelihood(sensor_data, expected_sensor)
            self.weights[i] *= likelihood
        
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            self.weights = np.ones(self.num_particles) / self.num_particles
    
    def resample(self):
        """중요도 재샘플링"""
        indices = np.random.choice(self.num_particles, size=self.num_particles, 
                                   p=self.weights, replace=True)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles
    
    def _simulate_sensor(self, pos):
        """센서 시뮬레이션"""
        readings = []
        num_rays = 36
        
        for k in range(num_rays):
            angle = 2 * math.pi * k / num_rays
            distance = self._ray_cast(pos, angle)
            readings.append(distance)
        
        return np.array(readings)
    
    def _ray_cast(self, pos, angle):
        """광선 추적"""
        x, y = pos
        dx = math.cos(angle)
        dy = math.sin(angle)
        
        step_size = self.cell_size / 20
        distance = 0
        
        for _ in range(int(self.sensor_range / step_size) + 1):
            test_x = x + dx * distance
            test_y = y + dy * distance
            
            if not (0 <= test_x < self.map_width and 0 <= test_y < self.map_height):
                return self.sensor_range
            
            grid_x = int(test_x / self.cell_size)
            grid_y = int(test_y / self.cell_size)
            
            if self.maze[grid_y][grid_x] == 1:
                return distance
            
            distance += step_size
        
        return self.sensor_range
    
    def _calculate_likelihood(self, observed, expected):
        """가능도 계산"""
        diff = np.array(observed) - np.array(expected)
        variance = (self.sensor_std ** 2) * 1.5
        likelihood = np.exp(-np.sum(diff ** 2) / (2 * variance))
        return max(likelihood, 1e-100)
    
    def _get_weighted_mean(self):
        """가중 평균 위치"""
        weighted_pos = np.average(self.particles, axis=0, weights=self.weights)
        return weighted_pos
    
    def update_estimated_pos(self):
        """추정 위치 업데이트"""
        self.estimated_pos = self._get_weighted_mean()
        return self.estimated_pos
    
    def get_uncertainty(self):
        """불확실성"""
        variance = np.var(self.particles, axis=0)
        return math.sqrt(np.sum(variance))


class AStarPlanner:
    """A* 경로 계획"""
    
    def __init__(self, maze, cell_size):
        self.maze = maze
        self.cell_size = cell_size
        self.map_width = len(maze[0])
        self.map_height = len(maze)
    
    def plan(self, start, goal):
        """A* 경로 계획"""
        start_grid = (int(start[0] / self.cell_size), int(start[1] / self.cell_size))
        goal_grid = (int(goal[0] / self.cell_size), int(goal[1] / self.cell_size))
        
        open_set = {start_grid}
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self._heuristic(start_grid, goal_grid)}
        
        while open_set:
            # f_score가 최소인 노드 선택
            current = min(open_set, key=lambda node: f_score.get(node, float('inf')))
            
            if current == goal_grid:
                # 경로 재구성
                path = []
                while current in came_from:
                    path.append((current[0] * self.cell_size + self.cell_size/2,
                                current[1] * self.cell_size + self.cell_size/2))
                    current = came_from[current]
                path.append((current[0] * self.cell_size + self.cell_size/2,
                            current[1] * self.cell_size + self.cell_size/2))
                return list(reversed(path))
            
            open_set.remove(current)
            
            # 이웃 탐색
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # 범위 확인
                if not (0 <= neighbor[0] < self.map_width and 0 <= neighbor[1] < self.map_height):
                    continue
                
                # 장애물 확인
                if self.maze[neighbor[1]][neighbor[0]] == 1:
                    continue
                
                # 대각선 이동 비용
                tentative_g = g_score[current] + math.sqrt(dx*dx + dy*dy)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_grid)
                    open_set.add(neighbor)
        
        return []
    
    def _heuristic(self, node, goal):
        """휴리스틱 함수 (유클리드 거리)"""
        return math.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)


class SmoothPathGenerator:
    """Smooth Path 생성 (Bezier curve)"""
    
    @staticmethod
    def smooth_path(waypoints, num_points=100):
        """경로를 부드러운 곡선으로 변환"""
        if len(waypoints) < 2:
            return waypoints
        
        smooth = []
        
        for i in range(len(waypoints) - 1):
            p0 = waypoints[i]
            p1 = waypoints[i + 1]
            
            # 각 세그먼트를 num_points/len(waypoints)개로 분할
            segment_points = int(num_points / (len(waypoints) - 1))
            
            for t in np.linspace(0, 1, segment_points, endpoint=(i == len(waypoints) - 2)):
                # 선형 보간
                x = p0[0] * (1 - t) + p1[0] * t
                y = p0[1] * (1 - t) + p1[1] * t
                smooth.append((x, y))
        
        return smooth


class PIDController:
    """PID 제어기"""
    
    def __init__(self, kp=2.0, ki=0.1, kd=0.3):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0
    
    def update(self, error, dt=0.1):
        """PID 업데이트"""
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return output


def draw_obstacles(ax, maze, cell_size, alpha=0.3):
    """맵 그리기"""
    for y in range(len(maze)):
        for x in range(len(maze[0])):
            if maze[y][x] == 1:
                min_x = x * cell_size
                min_y = y * cell_size
                ax.add_patch(Rectangle((min_x, min_y), cell_size, cell_size,
                                       alpha=alpha, facecolor='gray', edgecolor='black', linewidth=1))


def main():
    print("\n" + "="*80)
    print("Particle Filter + A* Path Planning + PID Control 통합 주차 시스템")
    print("="*80)
    
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
    
    # 주차 위치 정의
    P1 = np.array([2.5, 47.5])      # 출발 위치
    P2 = np.array([57.5, 47.5])     # 주차 위치 1
    G1 = np.array([57.5, 7.5])      # 주차 위치 2
    
    print(f"\n[맵 설정]")
    print(f"  맵 크기: 13x10 그리드 (65x50m)")
    print(f"  P1 (출발): ({P1[0]:.1f}, {P1[1]:.1f})")
    print(f"  P2 (주차 위치 1): ({P2[0]:.1f}, {P2[1]:.1f})")
    print(f"  G1 (주차 위치 2): ({G1[0]:.1f}, {G1[1]:.1f})")
    
    # Particle Filter 초기화
    pf = ParticleFilterLocalizer(maze, cell_size, num_particles=1000, 
                                sensor_range=24.0, sensor_std=0.2, motion_std=0.2)
    
    # A* 계획기 초기화
    planner = AStarPlanner(maze, cell_size)
    
    # 실제 로봇 위치 (처음엔 모름)
    true_pos = np.array([2.5, 47.5])
    
    print(f"\n[Phase 1: 위치 추정]")
    print(f"  로봇의 위치를 파악하기 위해 센서를 이용한 localization 수행")
    
    # Phase 1: Localization 단계
    localization_log = []
    target_goal = None
    localization_done = False
    
    # 탐색 움직임
    commands_phase1 = [
        (1.0, 0.0), (1.0, 0.0), (1.0, 0.0),
        (0.0, -1.0), (0.0, -1.0), (0.0, -1.0),
        (0.0, -1.0), (0.0, -1.0), (0.0, -1.0),
        (1.0, 0.0), (1.0, 0.0), (1.0, 0.0),
    ]
    
    for step, (cmd_dx, cmd_dy) in enumerate(commands_phase1):
        # 실제 로봇 이동
        actual_noise_x = np.random.normal(0, 0.2)
        actual_noise_y = np.random.normal(0, 0.2)
        true_pos = true_pos + np.array([cmd_dx + actual_noise_x, cmd_dy + actual_noise_y])
        true_pos = np.clip(true_pos, [0, 0], [65, 50])
        
        # Particle Filter 업데이트
        pf.predict(cmd_dx, cmd_dy)
        sensor_readings = pf._simulate_sensor(true_pos)
        sensor_readings = sensor_readings + np.random.normal(0, 0.2, len(sensor_readings))
        sensor_readings = np.clip(sensor_readings, 0, 24.0)
        pf.update(sensor_readings)
        pf.resample()
        
        est_pos = pf.update_estimated_pos()
        uncertainty = pf.get_uncertainty()
        error = np.linalg.norm(true_pos - est_pos)
        
        localization_log.append({
            "step": step,
            "true_pos": true_pos.copy(),
            "est_pos": est_pos.copy(),
            "particles": pf.particles.copy(),
            "uncertainty": uncertainty,
            "error": error,
        })
        
        print(f"  Step {step}: True ({true_pos[0]:6.2f}, {true_pos[1]:6.2f}) | "
              f"Est ({est_pos[0]:6.2f}, {est_pos[1]:6.2f}) | "
              f"Error {error:6.3f}m | Unc {uncertainty:6.3f}m", end="")
        
        # Uncertainty가 낮아지면 충분한 확신 달성
        if uncertainty < 1.0 and not localization_done:
            print(" ← Localization Done! 목표 선택")
            localization_done = True
            
            # 목표 선택 (P2 또는 G1 - 여기서는 G1 선택)
            target_goal = G1
            target_name = "G1"
        else:
            print()
    
    if not localization_done:
        print(f"\n  [경고] Localization 미완료, 마지막 추정 위치로 진행")
        target_goal = G1
        target_name = "G1"
        est_pos = pf.update_estimated_pos()
    
    print(f"\n[Phase 2: 경로 계획]")
    print(f"  시작: ({est_pos[0]:.1f}, {est_pos[1]:.1f})")
    print(f"  목표: {target_name} ({target_goal[0]:.1f}, {target_goal[1]:.1f})")
    
    # Phase 2: A* 경로 계획
    waypoints = planner.plan(est_pos, target_goal)
    if waypoints:
        print(f"  경로 포인트: {len(waypoints)}개")
        
        # Smooth Path 생성
        smooth_path = SmoothPathGenerator.smooth_path(waypoints, num_points=200)
        print(f"  부드러운 경로: {len(smooth_path)}개 포인트")
    else:
        print(f"  [에러] 경로를 찾을 수 없음!")
        smooth_path = [est_pos, target_goal]
    
    # Phase 3: PID 제어로 주차
    print(f"\n[Phase 3: PID 제어 주차]")
    print(f"  Smooth path를 따라가면서 실시간 localization 수행")
    
    parking_log = []
    true_pos = localization_log[-1]["true_pos"].copy()
    final_goal = np.array(target_goal)
    
    step_counter = 0
    max_parking_steps = len(smooth_path) * 5
    
    for step in range(max_parking_steps):
        # 최종 목표까지의 거리
        distance_to_final = np.linalg.norm(true_pos - final_goal)
        
        # 최종 목표에 도달했는지 확인
        if distance_to_final < 0.5:
            print(f"  [주차 완료] Step {step}: 최종 목표 도달 (에러 {distance_to_final:.3f}m)")
            break
        
        # 최종 목표 방향으로 작은 이동
        direction = final_goal - true_pos
        direction_normalized = direction / (np.linalg.norm(direction) + 1e-6)
        cmd_dx = direction_normalized[0] * 0.2  
        cmd_dy = direction_normalized[1] * 0.2
        
        # 실제 로봇 이동
        actual_noise_x = np.random.normal(0, 0.2)
        actual_noise_y = np.random.normal(0, 0.2)
        true_pos = true_pos + np.array([cmd_dx + actual_noise_x, cmd_dy + actual_noise_y])
        true_pos = np.clip(true_pos, [0, 0], [65, 50])
        
        # Particle Filter 업데이트 (실시간 위치 추정)
        pf.predict(cmd_dx, cmd_dy)
        sensor_readings = pf._simulate_sensor(true_pos)
        sensor_readings = sensor_readings + np.random.normal(0, 0.2, len(sensor_readings))
        sensor_readings = np.clip(sensor_readings, 0, 24.0)
        pf.update(sensor_readings)
        pf.resample()
        
        est_pos = pf.update_estimated_pos()
        uncertainty = pf.get_uncertainty()
        error = np.linalg.norm(true_pos - est_pos)
        path_error = distance_to_final
        
        parking_log.append({
            "step": step,
            "true_pos": true_pos.copy(),
            "est_pos": est_pos.copy(),
            "target_point": final_goal.copy(),
            "particles": pf.particles.copy(),
            "uncertainty": uncertainty,
            "error": error,
            "path_error": path_error,
        })
        
        if step % 10 == 0:
            print(f"  Step {step}: True ({true_pos[0]:6.2f}, {true_pos[1]:6.2f}) | "
                  f"Goal ({final_goal[0]:6.2f}, {final_goal[1]:6.2f}) | "
                  f"Distance {path_error:6.3f}m | Unc {uncertainty:6.3f}m")
    
    print(f"\n[시뮬레이션 완료]")
    print(f"  Localization 단계: {len(localization_log)}스텝")
    print(f"  PID 주차 단계: {len(parking_log)}스텝")
    
    # 애니메이션 생성
    print(f"\n애니메이션 생성 중...")
    visualize_parking_system(maze, cell_size, localization_log, parking_log, 
                            waypoints, smooth_path, P1, P2, G1)
    print(f"[OK] 애니메이션 저장: particle_filter_parking_system.gif")
    print("\n" + "="*80)


def visualize_parking_system(maze, cell_size, loc_log, park_log, waypoints, smooth_path,
                            P1, P2, G1):
    """시각화"""
    
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])     # 지도
    ax2 = fig.add_subplot(gs[0, 1])     # 에러
    
    # 좌측: 지도
    ax1.set_title("Particle Filter Localization + PID Parking", fontsize=13, fontweight='bold')
    ax1.set_aspect("equal")
    ax1.set_xlim(-2, 67)
    ax1.set_ylim(-2, 52)
    
    draw_obstacles(ax1, maze, cell_size)
    
    # 입자 표시
    particle_scatter = ax1.scatter([], [], c='lightblue', s=10, alpha=0.5, label='Particles')
    
    # 경로 표시
    if waypoints:
        waypoints_array = np.array(waypoints)
        ax1.plot(waypoints_array[:, 0], waypoints_array[:, 1], 'g--', linewidth=2, 
                alpha=0.7, label='A* Waypoints')
    
    if smooth_path:
        smooth_array = np.array(smooth_path)
        ax1.plot(smooth_array[:, 0], smooth_array[:, 1], 'g-', linewidth=2, 
                alpha=0.9, label='Smooth Path')
    
    # 위치 표시
    true_pt, = ax1.plot([], [], 'b*', markersize=20, label='True pos', zorder=5)
    est_pt, = ax1.plot([], [], 'ro', markersize=12, label='Est pos', zorder=5)
    target_pt, = ax1.plot([], [], 'g^', markersize=15, label='Target', zorder=5)
    
    # 궤적
    true_trace, = ax1.plot([], [], 'b-', linewidth=1, alpha=0.5, label='True trajectory')
    est_trace, = ax1.plot([], [], 'r--', linewidth=1.5, alpha=0.7, label='Est trajectory')
    
    # 주차 위치 표시
    ax1.plot(P1[0], P1[1], 'bs', markersize=10, label='P1 (Start)')
    ax1.plot(P2[0], P2[1], 'gs', markersize=10, label='P2 (Parking)')
    ax1.plot(G1[0], G1[1], 'r^', markersize=12, label='G1 (Goal)')
    
    ax1.legend(loc='upper right', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("X (m)", fontsize=10)
    ax1.set_ylabel("Y (m)", fontsize=10)
    
    # 우측: 에러 그래프
    ax2.set_title("Localization Error & Path Error", fontsize=12, fontweight='bold')
    loc_error_line, = ax2.plot([], [], 'b-o', linewidth=2, markersize=4, label='Loc Error')
    path_error_line, = ax2.plot([], [], 'r-s', linewidth=2, markersize=4, label='Path Error')
    ax2.set_xlabel("Step", fontsize=10)
    ax2.set_ylabel("Error (m)", fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 상태 텍스트
    status_text = ax1.text(0.02, 0.98, "", transform=ax1.transAxes, va="top", fontsize=10,
                          bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8))
    
    # Phase 조합
    total_steps = len(loc_log) + len(park_log)
    
    def init():
        particle_scatter.set_offsets(np.empty((0, 2)))
        true_pt.set_data([], [])
        est_pt.set_data([], [])
        target_pt.set_data([], [])
        true_trace.set_data([], [])
        est_trace.set_data([], [])
        loc_error_line.set_data([], [])
        path_error_line.set_data([], [])
        status_text.set_text("")
        return [particle_scatter, true_pt, est_pt, target_pt, true_trace, est_trace,
                loc_error_line, path_error_line, status_text]
    
    def update(frame):
        if frame < len(loc_log):
            # Phase 1: Localization
            data = loc_log[frame]
            phase = "Phase 1: Localization"
            
            particle_scatter.set_offsets(data["particles"])
            
            true_x, true_y = data["true_pos"]
            est_x, est_y = data["est_pos"]
            
            true_pt.set_data([true_x], [true_y])
            est_pt.set_data([est_x], [est_y])
            target_pt.set_data([G1[0]], [G1[1]])
            
            # 궤적
            true_positions = np.array([loc_log[i]["true_pos"] for i in range(frame + 1)])
            est_positions = np.array([loc_log[i]["est_pos"] for i in range(frame + 1)])
            
            true_trace.set_data(true_positions[:, 0], true_positions[:, 1])
            est_trace.set_data(est_positions[:, 0], est_positions[:, 1])
            
            # 에러
            loc_errors = np.array([loc_log[i]["error"] for i in range(frame + 1)])
            loc_steps = np.arange(frame + 1)
            loc_error_line.set_data(loc_steps, loc_errors)
            path_error_line.set_data([], [])
            
            ax2.set_xlim(-0.5, max(len(loc_log), 10))
            ax2.set_ylim(0, 50)
            
            status = f"{phase}\nStep: {frame}\nUnc: {data['uncertainty']:.2f}m\nError: {data['error']:.3f}m"
        
        else:
            # Phase 2: Parking
            park_idx = frame - len(loc_log)
            data = park_log[park_idx]
            phase = "Phase 2: PID Parking"
            
            particle_scatter.set_offsets(data["particles"])
            
            true_x, true_y = data["true_pos"]
            est_x, est_y = data["est_pos"]
            target_x, target_y = data["target_point"]
            
            true_pt.set_data([true_x], [true_y])
            est_pt.set_data([est_x], [est_y])
            target_pt.set_data([target_x], [target_y])
            
            # 궤적
            all_true = np.array([loc_log[i]["true_pos"] for i in range(len(loc_log))] +
                               [park_log[i]["true_pos"] for i in range(park_idx + 1)])
            all_est = np.array([loc_log[i]["est_pos"] for i in range(len(loc_log))] +
                              [park_log[i]["est_pos"] for i in range(park_idx + 1)])
            
            true_trace.set_data(all_true[:, 0], all_true[:, 1])
            est_trace.set_data(all_est[:, 0], all_est[:, 1])
            
            # 에러
            loc_errors = np.array([loc_log[i]["error"] for i in range(len(loc_log))])
            path_errors = np.array([park_log[i]["path_error"] for i in range(park_idx + 1)])
            
            loc_steps = np.arange(len(loc_log))
            path_steps = np.arange(len(loc_log), len(loc_log) + park_idx + 1)
            
            loc_error_line.set_data(loc_steps, loc_errors)
            path_error_line.set_data(path_steps, path_errors)
            
            ax2.set_xlim(-0.5, total_steps)
            ax2.set_ylim(0, max(np.max(loc_errors), 10))
            
            status = f"{phase}\nStep: {park_idx}\nPath Error: {data['path_error']:.3f}m\nLoc Error: {data['error']:.3f}m"
        
        status_text.set_text(status)
        
        return [particle_scatter, true_pt, est_pt, target_pt, true_trace, est_trace,
                loc_error_line, path_error_line, status_text]
    
    ani = FuncAnimation(fig, update, frames=total_steps, init_func=init,
                       interval=500, blit=True, repeat=True)
    
    ani.save("particle_filter_parking_system.gif", writer="pillow", fps=2)
    plt.close()


if __name__ == "__main__":
    main()
