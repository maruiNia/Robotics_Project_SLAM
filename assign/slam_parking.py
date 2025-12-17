"""
SLAM + EKF 위치 확신 후 주차 프로그램
- SLAM/EKF로 위치 추정 및 불확실성 감소
- 불확실성이 threshold 이하가 되면 목표 위치로 최적경로 계획
- PID 제어기로 smooth path 따라 주차
- 실시간 위치 측정 및 시각화
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Polygon, Circle as PltCircle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import module.MySlam as MySlam
from module.MySlam import maze_map
from module.MySensor import Circle_Sensor
from module.MyControl import SimpleBicycleModel
from module.MyPlanning import PIDVec, smooth_path_nd, point_to_path_min_distance_nd
from module.MyRobot_with_localization import Moblie_robot


def draw_obstacles(ax, slam_map: MySlam, *, alpha=0.20):
    """맵의 장애물 그리기"""
    for obs in slam_map.get_obs_list():
        (min_x, min_y), (max_x, max_y) = obs.bounds()
        ax.add_patch(Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, 
                               alpha=alpha, color='gray', edgecolor='black', linewidth=1))


class Landmark:
    """랜드마크 정의"""
    def __init__(self, landmark_id: int, position: tuple, max_range: float = 4.0, std: float = 0.4):
        self.id = landmark_id
        self.position = np.array(position, dtype=float)
        self.max_range = max_range
        self.std = std
    
    def measure(self, robot_pos):
        """로봇이 랜드마크를 센싱했을 때의 거리 반환 (노이즈 포함)"""
        true_distance = np.linalg.norm(robot_pos - self.position)
        
        if true_distance > self.max_range:
            return None
        
        measured_distance = true_distance + np.random.normal(0, self.std)
        return max(0.0, measured_distance)
    
    def draw(self, ax, color='yellow', markersize=12):
        """랜드마크 시각화"""
        x, y = self.position
        diamond = Polygon([
            [x, y + 0.3],
            [x + 0.3, y],
            [x, y - 0.3],
            [x - 0.3, y],
        ], closed=True, fill=True, facecolor=color, alpha=0.8, edgecolor='black', linewidth=2)
        ax.add_patch(diamond)
        ax.text(x, y - 0.6, f'L{self.id}', ha='center', fontsize=8, fontweight='bold')


class SLAMwithEKF:
    """SLAM + EKF 통합 로컬라이제이션"""
    
    def __init__(self, landmarks):
        self.landmarks = landmarks
        
        # EKF 상태: [x, y, theta]
        self.x = np.array([0.0, 0.0, 0.0], dtype=float)
        self.P = np.eye(3) * 1.0
        
        # 프로세스 및 측정 노이즈
        self.Q = np.diag([0.005, 0.005, 0.005])
        self.R = np.eye(len(landmarks)) * 0.16
    
    def predict(self, dx, dy, dt=1.0):
        """EKF 예측"""
        x, y, theta = self.x
        
        x_new = x + dx
        y_new = y + dy
        
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            theta_new = math.atan2(dy, dx)
        else:
            theta_new = theta
        
        self.x = np.array([x_new, y_new, theta_new])
        
        F = np.eye(3)
        self.P = F @ self.P @ F.T + self.Q
    
    def update(self, landmark_measurements):
        """EKF 업데이트"""
        z = []
        H_list = []
        
        for lm_id, meas in landmark_measurements:
            if meas is not None:
                z.append(meas)
                
                lm = next((l for l in self.landmarks if l.id == lm_id), None)
                if lm is not None:
                    dx = lm.position[0] - self.x[0]
                    dy = lm.position[1] - self.x[1]
                    dist = math.sqrt(dx**2 + dy**2)
                    
                    if dist > 1e-6:
                        h = [-dx/dist, -dy/dist, 0]
                    else:
                        h = [0, 0, 0]
                    H_list.append(h)
        
        if len(z) == 0:
            return
        
        z = np.array(z, dtype=float)
        H = np.array(H_list)
        
        z_pred = []
        for lm_id, meas in landmark_measurements:
            if meas is not None:
                lm = next((l for l in self.landmarks if l.id == lm_id), None)
                if lm is not None:
                    dist = np.linalg.norm(lm.position - self.x[:2])
                    z_pred.append(dist)
        
        z_pred = np.array(z_pred)
        y = z - z_pred
        
        R_reduced = self.R[:len(z), :len(z)]
        S = H @ self.P @ H.T + R_reduced
        
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            self.P = (np.eye(3) - K @ H) @ self.P
        except:
            pass
    
    def get_estimated_pos(self):
        """추정 위치"""
        return self.x[:2]
    
    def get_uncertainty(self):
        """불확실성"""
        return np.sqrt(np.trace(self.P[:2, :2]))


def run_slam_phase(landmarks, slam_map, num_steps=20):
    """
    Phase 1: SLAM 단계 - 로봇이 좌상단에서 시작하여 위치 확신도를 높임
    불확실성이 threshold 이하가 되면 종료
    """
    # 좌상단 (0, 0)에서 시작
    start_pos = (2.5, 47.5)  # 그리드 (0, 9)의 중심
    
    slam_ekf = SLAMwithEKF(landmarks)
    slam_ekf.x = np.array([start_pos[0], start_pos[1], 0.0])
    
    log = []
    
    print(f"\n{'='*70}")
    print("Phase 1: SLAM + EKF 로컬라이제이션 (위치 확신도 증가)")
    print(f"{'='*70}")
    print(f"시작 위치: ({start_pos[0]:.2f}, {start_pos[1]:.2f}) [좌상단]")
    
    # 우향 + 하향 이동 (목표 위치로 직진)
    directions = [
        (0.5, 0.0),
        (0.5, 0.0),
        (0.0, -0.5),
        (0.0, -0.5),
    ]
    
    uncertainty_threshold = 0.5
    confirmed_step = -1
    confirmed_pos = None
    
    # 현재 위치 추적
    current_pos = np.array(start_pos, dtype=float)
    
    commands = []
    for _ in range(num_steps // 4):
        commands.extend(directions)
    
    for step, (cmd_dx, cmd_dy) in enumerate(commands):
        # 실제 운동 (노이즈 포함)
        actual_dx = cmd_dx + np.random.normal(0, 0.05)
        actual_dy = cmd_dy + np.random.normal(0, 0.05)
        current_pos += np.array([actual_dx, actual_dy])
        current_pos = np.clip(current_pos, [0, 0], [65, 50])
        
        # 랜드마크 측정
        landmark_meas = [(lm.id, lm.measure(current_pos)) for lm in landmarks]
        
        slam_ekf.predict(actual_dx, actual_dy)
        slam_ekf.update(landmark_meas)
        
        est_pos = slam_ekf.get_estimated_pos()
        uncertainty = slam_ekf.get_uncertainty()
        error = np.linalg.norm(current_pos - est_pos)
        num_detected = sum(1 for _, meas in landmark_meas if meas is not None)
        
        log.append({
            "step": step,
            "true_pos": current_pos.copy(),
            "est_pos": est_pos.copy(),
            "error": error,
            "uncertainty": uncertainty,
            "num_detected": num_detected,
        })
        
        print(f"[Step {step}] Pos: ({current_pos[0]:.2f}, {current_pos[1]:.2f}), "
              f"Unc: {uncertainty:.3f}m, Error: {error:.3f}m, Detected: {num_detected}/4")
        
        # 불확실성이 threshold 이하가 되면 확신 상태
        if uncertainty <= uncertainty_threshold and confirmed_step == -1:
            confirmed_step = step
            confirmed_pos = est_pos.copy()
            print(f"\n[OK] 위치 확신 달성! (Step {step}, 불확실성: {uncertainty:.3f}m)")
            print(f"  확인된 위치: ({confirmed_pos[0]:.2f}, {confirmed_pos[1]:.2f})")
            break
    
    if confirmed_pos is None:
        confirmed_pos = slam_ekf.get_estimated_pos()
        print(f"\n[OK] 시뮬레이션 종료, 위치 확신: ({confirmed_pos[0]:.2f}, {confirmed_pos[1]:.2f})")
    
    return slam_ekf, confirmed_pos, current_pos, log


def simple_astar_path(maze, start, goal):
    """
    A* 경로 탐색 (그리드 기반)
    maze: 2D 그리드 (0: 통행 가능, 1: 장애물)
    start, goal: (x, y) 튜플 (그리드 좌표)
    """
    from heapq import heappush, heappop
    
    print(f"A* 탐색: {start} → {goal}")
    print(f"미로 크기: {len(maze[0])} x {len(maze)}")
    
    open_set = []
    closed_set = set()
    parent_map = {}
    g_score = {start: 0}
    
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    h = heuristic(start, goal)
    heappush(open_set, (h, 0, start))  # (f, g, node) 형태로 변경해서 tie-breaking 개선
    
    iterations = 0
    max_iterations = 10000
    
    while open_set and iterations < max_iterations:
        iterations += 1
        f, g, current = heappop(open_set)
        
        if current == goal:
            # 경로 복원
            path = []
            node = goal
            while node in parent_map:
                path.append(node)
                node = parent_map[node]
            path.append(start)
            path = list(reversed(path))
            print(f"[OK] 경로 찾음! 길이: {len(path)}, 반복: {iterations}")
            return path
        
        if current in closed_set:
            continue
        closed_set.add(current)
        
        x, y = current
        
        # 4 방향 탐색 (먼저 기본 방향만)
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            
            # 맵 범위 확인
            if not (0 <= nx < len(maze[0]) and 0 <= ny < len(maze)):
                continue
            
            # 장애물 확인
            if maze[ny][nx] == 1:
                continue
            
            neighbor = (nx, ny)
            
            if neighbor in closed_set:
                continue
            
            tentative_g = g_score[current] + 1
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                parent_map[neighbor] = current
                g_score[neighbor] = tentative_g
                h = heuristic(neighbor, goal)
                f = tentative_g + h
                heappush(open_set, (f, tentative_g, neighbor))
    
    print(f"[FAIL] 경로를 찾을 수 없습니다! (반복: {iterations})")
    return None


def plan_path_to_parking(maze, confirmed_pos, parking_pos, cell_size=5):
    """
    Phase 2: 경로 계획 - 확인된 위치에서 주차 위치로 A* 경로 탐색 및 스무딩
    """
    print(f"\n{'='*70}")
    print("Phase 2: 최적경로 계획 (A* + Smooth)")
    print(f"{'='*70}")
    print(f"현재 위치: ({confirmed_pos[0]:.2f}, {confirmed_pos[1]:.2f})")
    print(f"주차 위치: ({parking_pos[0]:.2f}, {parking_pos[1]:.2f})")
    
    # 맵 좌표 -> 그리드 좌표 변환
    start_grid_x = int(confirmed_pos[0] / cell_size)
    start_grid_y = int(confirmed_pos[1] / cell_size)
    goal_grid_x = int(parking_pos[0] / cell_size)
    goal_grid_y = int(parking_pos[1] / cell_size)
    
    # 범위 확인
    start_grid_x = max(0, min(start_grid_x, len(maze[0]) - 1))
    start_grid_y = max(0, min(start_grid_y, len(maze) - 1))
    goal_grid_x = max(0, min(goal_grid_x, len(maze[0]) - 1))
    goal_grid_y = max(0, min(goal_grid_y, len(maze) - 1))
    
    start_grid = (start_grid_x, start_grid_y)
    goal_grid = (goal_grid_x, goal_grid_y)
    
    print(f"그리드 좌표: {start_grid} → {goal_grid}")
    print(f"미로 크기: {len(maze[0])} x {len(maze)}")
    print(f"셀 크기: {cell_size}m")
    
    # A* 경로 탐색
    astar_path_result = simple_astar_path(maze, start_grid, goal_grid)
    
    if not astar_path_result:
        print("[FAIL] A* 경로를 찾을 수 없습니다!")
        print(f"시작 그리드: {start_grid}, 미로[{start_grid_y}][{start_grid_x}] = {maze[start_grid_y][start_grid_x]}")
        print(f"목표 그리드: {goal_grid}, 미로[{goal_grid_y}][{goal_grid_x}] = {maze[goal_grid_y][goal_grid_x]}")
        # 직선 경로로 대체
        print("직선 경로로 대체합니다.")
        world_path = [list(confirmed_pos), list(parking_pos)]
    else:
        # 그리드 -> 월드 좌표 변환
        world_path = []
        for (gx, gy) in astar_path_result:
            wx = gx * cell_size + cell_size / 2
            wy = gy * cell_size + cell_size / 2
            world_path.append([wx, wy])
        
        print(f"[OK] A* 경로: {len(world_path)}개 점")
        for i, pt in enumerate(world_path):
            print(f"  [{i}] ({pt[0]:.1f}, {pt[1]:.1f})")
    
    # 스무딩
    print(f"\n경로 스무딩 중...")
    smooth_path = smooth_path_nd(world_path, alpha=0.1, beta=0.3, max_iter=500)
    print(f"[OK] Smooth 경로: {len(smooth_path)}개 점")
    for i, pt in enumerate(smooth_path):
        print(f"  [{i}] ({pt[0]:.1f}, {pt[1]:.1f})")
    
    return world_path, smooth_path


def run_parking_phase(slam_ekf, smooth_path, landmarks, slam_map, current_pos_start, log_base):
    """
    Phase 3: 주차 단계 - PID 제어기로 smooth path 따라 주차
    """
    print(f"\n{'='*70}")
    print("Phase 3: PID 제어기로 주차 경로 추적")
    print(f"{'='*70}")
    print(f"경로 점 개수: {len(smooth_path)}")
    for i, pt in enumerate(smooth_path):
        print(f"  [{i}] ({pt[0]:.1f}, {pt[1]:.1f})")
    
    pid = PIDVec(dim=2, kp=2.0, ki=0.1, kd=0.3)  # 더 강한 제어
    
    log = log_base.copy()
    base_step = len(log_base)
    dt = 0.05  # 50ms
    goal_tolerance = 1.5
    max_steps = 500
    
    current_pos = np.array(current_pos_start, dtype=float)
    path_index = 0
    
    print(f"초기 위치: ({current_pos[0]:.2f}, {current_pos[1]:.2f})")
    print(f"목표 위치: ({smooth_path[-1][0]:.2f}, {smooth_path[-1][1]:.2f})")
    print(f"초기 거리: {np.linalg.norm(np.array(smooth_path[-1]) - current_pos):.2f}m\n")
    
    for step in range(max_steps):
        # 목표에 도달했는지 확인
        dist_to_goal = np.linalg.norm(np.array(smooth_path[-1]) - current_pos)
        
        if dist_to_goal < goal_tolerance:
            print(f"\n[OK] 주차 완료! (Step {step})")
            print(f"  최종 위치: ({current_pos[0]:.2f}, {current_pos[1]:.2f})")
            print(f"  목표 위치: ({smooth_path[-1][0]:.2f}, {smooth_path[-1][1]:.2f})")
            print(f"  도착 오차: {dist_to_goal:.2f}m")
            break
        
        # 현재 위치에서 smooth path까지의 최소거리 계산
        dmin, q, seg_i, t = point_to_path_min_distance_nd(current_pos, smooth_path)
        
        # 경로 추적 오차 계산 (경로에 수직인 벡터)
        if seg_i < len(smooth_path) - 1:
            a = np.array(smooth_path[seg_i])
            b = np.array(smooth_path[seg_i + 1])
            ab = b - a
            ab_len = np.linalg.norm(ab)
            
            if ab_len > 1e-6:
                # 접선 방향
                t_hat = ab / ab_len
                # 현재 위치에서 경로의 최근접점으로의 벡터
                e = np.array(q) - current_pos
                # 접선 성분 (전진 방향)
                e_par = np.dot(e, t_hat) * t_hat
                # 수직 성분 (오차)
                e_perp = e - e_par
                
                # 다음 경로점 방향으로 가속
                direction_to_next = np.array(smooth_path[seg_i + 1]) - current_pos
                direction_to_next = direction_to_next / (np.linalg.norm(direction_to_next) + 1e-6)
            else:
                e_perp = np.array(smooth_path[seg_i]) - current_pos
                direction_to_next = e_perp / (np.linalg.norm(e_perp) + 1e-6)
        else:
            e_perp = np.array(smooth_path[-1]) - current_pos
            direction_to_next = e_perp / (np.linalg.norm(e_perp) + 1e-6)
        
        # PID 제어로 조향 생성
        steering = pid.step(e_perp, dt)
        
        # 전진 속도 (목표까지의 거리에 따라 조정)
        velocity_mag = min(0.5, 0.1 + dist_to_goal * 0.01)
        
        # 로봇 이동: 전진 + 조향
        motion_direction = direction_to_next + np.array(steering) * 0.1
        motion_direction = motion_direction / (np.linalg.norm(motion_direction) + 1e-6)
        motion = motion_direction * velocity_mag
        motion += np.random.normal(0, 0.02, 2)  # 작은 노이즈
        
        current_pos = current_pos + motion * dt
        current_pos = np.clip(current_pos, [0, 0], [65, 50])
        
        # SLAM/EKF 업데이트
        landmark_meas = [(lm.id, lm.measure(current_pos)) for lm in landmarks]
        slam_ekf.predict(motion[0] * dt, motion[1] * dt)
        slam_ekf.update(landmark_meas)
        
        est_pos = slam_ekf.get_estimated_pos()
        uncertainty = slam_ekf.get_uncertainty()
        error = np.linalg.norm(current_pos - est_pos)
        
        log.append({
            "step": base_step + step,
            "true_pos": current_pos.copy(),
            "est_pos": est_pos.copy(),
            "error": error,
            "uncertainty": uncertainty,
            "num_detected": sum(1 for _, meas in landmark_meas if meas is not None),
            "path_error": dmin,
        })
        
        if step % 30 == 0:
            print(f"[Parking Step {step:3d}] Pos: ({current_pos[0]:6.2f}, {current_pos[1]:6.2f}), "
                  f"PathErr: {dmin:6.3f}m, EstErr: {error:6.3f}m, Goal: {dist_to_goal:6.2f}m")
    
    return log, smooth_path


def visualize_parking(slam_map, log, smooth_path, landmarks, world_path=None):
    """주차 과정 시각화"""
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # ===== 왼쪽: 지도 + 궤적 =====
    ax1 = axes[0]
    ax1.set_title("Parking Path Following (Smooth Path Tracking)", fontsize=14, fontweight='bold')
    ax1.set_aspect("equal")
    ax1.set_xlim(-2, 67)
    ax1.set_ylim(-2, 52)
    ax1.invert_yaxis()  # 좌상단이 (0,0)이 되도록
    
    draw_obstacles(ax1, slam_map, alpha=0.3)
    
    for lm in landmarks:
        lm.draw(ax1)
    
    # A* 경로 표시 (있으면)
    if world_path is not None:
        astar_x = [p[0] for p in world_path]
        astar_y = [p[1] for p in world_path]
        ax1.plot(astar_x, astar_y, 'r^--', linewidth=2, markersize=8, 
                label='A* path (waypoints)', alpha=0.7, zorder=1)
    
    # Smooth 경로 표시
    smooth_x = [p[0] for p in smooth_path]
    smooth_y = [p[1] for p in smooth_path]
    ax1.plot(smooth_x, smooth_y, 'g-', linewidth=3, label='Smooth path (target)', 
            alpha=0.8, zorder=2)
    ax1.plot(smooth_x, smooth_y, 'go', markersize=6, zorder=3)
    
    # 궤적
    true_trace, = ax1.plot([], [], 'b-', linewidth=2, label='True trajectory', zorder=4)
    est_trace, = ax1.plot([], [], 'm--', linewidth=1.5, label='Est trajectory', zorder=4, alpha=0.7)
    true_pt, = ax1.plot([], [], 'bo', markersize=12, zorder=5, label='Current pos')
    est_pt, = ax1.plot([], [], 'ms', markersize=10, zorder=5)
    
    # 불확실성 원
    uncertainty_circle = PltCircle((0, 0), 0, fill=False, edgecolor='blue', 
                                   linestyle=':', linewidth=2, alpha=0.5, zorder=6)
    ax1.add_patch(uncertainty_circle)
    
    # 시작점과 목표점
    ax1.plot([log[0]['true_pos'][0]], [log[0]['true_pos'][1]], 'go', markersize=15, 
            label='Start', zorder=7, markeredgecolor='darkgreen', markeredgewidth=2)
    ax1.plot([smooth_path[-1][0]], [smooth_path[-1][1]], 'r*', markersize=25, 
            label='Goal', zorder=7)
    
    ax1.legend(loc='upper left', fontsize=10, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    
    # ===== 오른쪽: 오차 및 경로 추적 오차 =====
    ax2 = axes[1]
    ax2.set_title("Error Metrics During Parking", fontsize=14, fontweight='bold')
    error_line, = ax2.plot([], [], 'r-o', linewidth=2, markersize=4, label='Position Error')
    uncertainty_line, = ax2.plot([], [], 'b--', linewidth=2, label='Uncertainty')
    path_error_line, = ax2.plot([], [], 'g-.', linewidth=2, label='Path Error')
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Error (m)")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    def init():
        true_trace.set_data([], [])
        est_trace.set_data([], [])
        true_pt.set_data([], [])
        est_pt.set_data([], [])
        error_line.set_data([], [])
        uncertainty_line.set_data([], [])
        path_error_line.set_data([], [])
        return [true_trace, est_trace, true_pt, est_pt, error_line, uncertainty_line, path_error_line]
    
    def update(frame):
        true_positions = np.array([log[i]['true_pos'] for i in range(frame + 1)])
        est_positions = np.array([log[i]['est_pos'] for i in range(frame + 1)])
        errors = np.array([log[i]['error'] for i in range(frame + 1)])
        uncertainties = np.array([log[i]['uncertainty'] for i in range(frame + 1)])
        path_errors = np.array([log[i].get('path_error', 0.0) for i in range(frame + 1)])
        
        true_trace.set_data(true_positions[:, 0], true_positions[:, 1])
        est_trace.set_data(est_positions[:, 0], est_positions[:, 1])
        true_pt.set_data([true_positions[-1, 0]], [true_positions[-1, 1]])
        est_pt.set_data([est_positions[-1, 0]], [est_positions[-1, 1]])
        
        uncertainty_circle.set_center(est_positions[-1])
        uncertainty_circle.set_radius(uncertainties[-1] * 2)
        
        error_line.set_data(range(len(errors)), errors)
        uncertainty_line.set_data(range(len(uncertainties)), uncertainties)
        path_error_line.set_data(range(len(path_errors)), path_errors)
        
        ax2.set_xlim(-0.5, max(len(errors), 5))
        ax2.set_ylim(0, max(np.max(errors), np.max(uncertainties), np.max(path_errors), 1) * 1.2)
        
        return [true_trace, est_trace, true_pt, est_pt, error_line, uncertainty_line, path_error_line]
    
    ani = FuncAnimation(fig, update, frames=len(log), init_func=init, 
                       interval=100, blit=True, repeat=False)
    
    plt.tight_layout()
    ani.save("slam_parking_animation.gif", writer="pillow", fps=10)
    print("\n[OK] 애니메이션 저장: slam_parking_animation.gif")
    plt.close()
    # plt.show()


def main():
    print("\n" + "="*70)
    print("SLAM + EKF 기반 자율 주차 시스템")
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
    start = (0, 9)
    end = (11, 1)
    slam_map = maze_map(maze, start, end, cell_size=cell_size)
    
    # 주차 위치 (G1 또는 P2)
    G1 = (55.0, 5.0)  # 오른쪽 위
    P2 = (10.0, 5.0)  # 왼쪽 위
    parking_pos = G1
    
    print(f"\n주차 위치: {parking_pos}")
    
    # 랜드마크 정의
    landmarks = [
        Landmark(1, (25.0, 20.0), max_range=50.0, std=0.4),
        Landmark(2, (40.0, 15.0), max_range=50.0, std=0.4),
        Landmark(3, (15.0, 30.0), max_range=50.0, std=0.4),
        Landmark(4, (35.0, 35.0), max_range=50.0, std=0.4),
    ]
    
    # Phase 1: SLAM 단계
    slam_ekf, confirmed_pos, current_pos, log_slam = run_slam_phase(landmarks, slam_map, num_steps=50)
    
    # Phase 2: 경로 계획
    world_path, smooth_path = plan_path_to_parking(maze, confirmed_pos, parking_pos, cell_size)
    
    # Phase 3: 주차 단계
    log_parking, final_smooth_path = run_parking_phase(slam_ekf, smooth_path, landmarks, slam_map, current_pos, log_slam)
    
    # 시각화 (world_path 포함)
    print("\n시각화 중...")
    visualize_parking(slam_map, log_parking, final_smooth_path, landmarks, world_path)
    
    print("\n" + "="*70)
    print("자율 주차 완료!")
    print("="*70)


if __name__ == "__main__":
    main()
