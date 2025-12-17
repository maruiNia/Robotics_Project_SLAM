"""
SLAM + EKF 기반 자율 주차 시스템 (단순화 버전)
1. SLAM 단계 (5단계)
2. 경로 계획 단계
3. 주차 제어 단계 (20단계)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Polygon, Circle as PltCircle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import module.MySlam as MySlam
from module.MySlam import maze_map
from module.MyPlanning import smooth_path_nd, point_to_path_min_distance_nd
from heapq import heappush, heappop
import math

class PIDVec:
    """간단한 2D PID 제어기"""
    def __init__(self, dim=2, kp=1.0, ki=0.0, kd=0.0):
        self.dim = dim
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = np.zeros(dim)
        self.prev_error = np.zeros(dim)
    
    def step(self, error, dt):
        """PID 업데이트"""
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else np.zeros(self.dim)
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error.copy()
        return output

def draw_obstacles(ax, maze, cell_size=5, alpha=0.20):
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

class Landmark:
    """랜드마크 정의"""
    def __init__(self, landmark_id, position, max_range=50.0, std=0.4):
        self.id = landmark_id
        self.position = np.array(position, dtype=float)
        self.max_range = max_range
        self.std = std
    
    def measure(self, robot_pos):
        """로봇이 랜드마크를 센싱했을 때의 거리 반환"""
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
        ax.text(x, y - 0.6, 'L' + str(self.id), ha='center', fontsize=8, fontweight='bold')

class SLAMwithEKF:
    """SLAM + EKF"""
    def __init__(self, landmarks):
        self.landmarks = landmarks
        self.x = np.array([0.0, 0.0, 0.0], dtype=float)
        self.P = np.eye(3) * 1.0
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
        return self.x[:2]
    
    def get_uncertainty(self):
        return np.sqrt(np.trace(self.P[:2, :2]))

def simple_astar_path(maze, start, goal):
    """A* 경로 탐색"""
    open_set = []
    closed_set = set()
    parent_map = {}
    g_score = {start: 0}
    
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    h = heuristic(start, goal)
    heappush(open_set, (h, 0, start))
    
    iterations = 0
    while open_set and iterations < 10000:
        iterations += 1
        f, g, current = heappop(open_set)
        
        if current == goal:
            path = []
            node = goal
            while node in parent_map:
                path.append(node)
                node = parent_map[node]
            path.append(start)
            print("[OK] 경로 찾음! 길이: %d, 반복: %d" % (len(path), iterations))
            return list(reversed(path))
        
        if current in closed_set:
            continue
        closed_set.add(current)
        
        x, y = current
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < len(maze[0]) and 0 <= ny < len(maze)):
                continue
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
    
    print("[FAIL] 경로를 찾을 수 없습니다!")
    return None

def run_slam_phase(landmarks, num_steps=5):
    """Phase 1: SLAM 위치 확신도 증가"""
    start_pos = (2.5, 47.5)  # 좌상단
    slam_ekf = SLAMwithEKF(landmarks)
    slam_ekf.x = np.array([start_pos[0], start_pos[1], 0.0])
    
    print("\n" + "="*70)
    print("Phase 1: SLAM + EKF (위치 확신도 증가)")
    print("="*70)
    print("시작 위치: (%.2f, %.2f) [좌상단]" % (start_pos[0], start_pos[1]))
    
    current_pos = np.array(start_pos, dtype=float)
    log = []
    
    directions = [(0.5, 0), (0.5, 0), (0, -0.5), (0, -0.5)]
    for step in range(num_steps):
        cmd_dx, cmd_dy = directions[step % len(directions)]
        actual_dx = cmd_dx + np.random.normal(0, 0.05)
        actual_dy = cmd_dy + np.random.normal(0, 0.05)
        current_pos += np.array([actual_dx, actual_dy])
        current_pos = np.clip(current_pos, [0, 0], [65, 50])
        
        landmark_meas = [(lm.id, lm.measure(current_pos)) for lm in landmarks]
        slam_ekf.predict(actual_dx, actual_dy)
        slam_ekf.update(landmark_meas)
        
        est_pos = slam_ekf.get_estimated_pos()
        uncertainty = slam_ekf.get_uncertainty()
        error = np.linalg.norm(current_pos - est_pos)
        num_detected = sum(1 for _, m in landmark_meas if m is not None)
        
        log.append({"step": step, "true_pos": current_pos.copy(), "est_pos": est_pos.copy(),
                    "error": error, "uncertainty": uncertainty, "num_detected": num_detected})
        
        print("[Step %d] Pos: (%.2f, %.2f), Unc: %.3f, Error: %.3f, Detected: %d/4"
              % (step, current_pos[0], current_pos[1], uncertainty, error, num_detected))
        
        if uncertainty <= 0.5:
            print("[OK] 위치 확신 달성! (Step %d, 불확실성: %.3f)" % (step, uncertainty))
            break
    
    confirmed_pos = slam_ekf.get_estimated_pos()
    print("[OK] 확인된 위치: (%.2f, %.2f)" % (confirmed_pos[0], confirmed_pos[1]))
    
    return slam_ekf, confirmed_pos, current_pos, log

def run_parking_phase(slam_ekf, smooth_path, landmarks, current_pos_start, log_base):
    """Phase 3: PID 제어로 주차"""
    print("\n" + "="*70)
    print("Phase 3: PID 제어로 주차 경로 추적")
    print("="*70)
    
    pid = PIDVec(dim=2, kp=2.0, ki=0.1, kd=0.3)
    log = log_base.copy()
    base_step = len(log_base)
    current_pos = np.array(current_pos_start, dtype=float)
    goal_tolerance = 1.5
    max_steps = 2000  # 목표 도달까지 충분한 시뮬레이션
    
    print("경로 점: %d개" % len(smooth_path))
    print("초기 위치: (%.2f, %.2f)" % (current_pos[0], current_pos[1]))
    print("목표 위치: (%.2f, %.2f)" % (smooth_path[-1][0], smooth_path[-1][1]))
    
    for step in range(max_steps):
        dist_to_goal = np.linalg.norm(np.array(smooth_path[-1]) - current_pos)
        
        if dist_to_goal < goal_tolerance:
            print("\n[OK] 주차 완료! (Step %d)" % step)
            print("  최종 위치: (%.2f, %.2f)" % (current_pos[0], current_pos[1]))
            print("  도착 오차: %.2f m" % dist_to_goal)
            # 목표 도달 후 로그에 추가하고 종료
            landmark_meas = [(lm.id, lm.measure(current_pos)) for lm in landmarks]
            log.append({"step": base_step + step, "true_pos": current_pos.copy(),
                        "est_pos": slam_ekf.get_estimated_pos().copy(),
                        "error": np.linalg.norm(current_pos - slam_ekf.get_estimated_pos()),
                        "uncertainty": slam_ekf.get_uncertainty(),
                        "num_detected": sum(1 for _, m in landmark_meas if m is not None),
                        "path_error": 0.0})
            return log, smooth_path
        
        dmin, q, seg_i, t = point_to_path_min_distance_nd(current_pos, smooth_path)
        
        if seg_i < len(smooth_path) - 1:
            a = np.array(smooth_path[seg_i])
            b = np.array(smooth_path[seg_i + 1])
            ab = b - a
            ab_len = np.linalg.norm(ab)
            
            if ab_len > 1e-6:
                t_hat = ab / ab_len
                e = np.array(q) - current_pos
                e_par = np.dot(e, t_hat) * t_hat
                e_perp = e - e_par
                direction_to_next = np.array(smooth_path[seg_i + 1]) - current_pos
                direction_to_next = direction_to_next / (np.linalg.norm(direction_to_next) + 1e-6)
            else:
                e_perp = np.array(smooth_path[seg_i]) - current_pos
                direction_to_next = e_perp / (np.linalg.norm(e_perp) + 1e-6)
        else:
            e_perp = np.array(smooth_path[-1]) - current_pos
            direction_to_next = e_perp / (np.linalg.norm(e_perp) + 1e-6)
        
        steering = pid.step(e_perp, 0.05)
        velocity_mag = min(2.0, 0.5 + dist_to_goal * 0.03)  # 더 빠른 속도
        
        motion_direction = direction_to_next + np.array(steering) * 0.1
        motion_direction = motion_direction / (np.linalg.norm(motion_direction) + 1e-6)
        motion = motion_direction * velocity_mag
        motion += np.random.normal(0, 0.02, 2)
        
        current_pos = current_pos + motion * 0.05
        current_pos = np.clip(current_pos, [0, 0], [65, 50])
        
        landmark_meas = [(lm.id, lm.measure(current_pos)) for lm in landmarks]
        slam_ekf.predict(motion[0] * 0.05, motion[1] * 0.05)
        slam_ekf.update(landmark_meas)
        
        est_pos = slam_ekf.get_estimated_pos()
        uncertainty = slam_ekf.get_uncertainty()
        error = np.linalg.norm(current_pos - est_pos)
        
        log.append({"step": base_step + step, "true_pos": current_pos.copy(),
                    "est_pos": est_pos.copy(), "error": error, "uncertainty": uncertainty,
                    "num_detected": sum(1 for _, m in landmark_meas if m is not None),
                    "path_error": dmin})
        
        if step % 5 == 0 or step == max_steps - 1:
            print("[Parking Step %d] Pos: (%.2f, %.2f), PathErr: %.3f, Goal: %.2f"
                  % (step, current_pos[0], current_pos[1], dmin, dist_to_goal))
    
    return log, smooth_path

def visualize_parking(maze, log, smooth_path, landmarks, world_path=None):
    """주차 과정 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    ax1 = axes[0]
    ax1.set_title("Parking Path Following", fontsize=14, fontweight='bold')
    ax1.set_aspect("equal")
    ax1.set_xlim(-2, 67)
    ax1.set_ylim(-2, 52)
    
    draw_obstacles(ax1, maze)
    
    for lm in landmarks:
        lm.draw(ax1)
    
    if world_path is not None:
        astar_x = [p[0] for p in world_path]
        astar_y = [p[1] for p in world_path]
        ax1.plot(astar_x, astar_y, 'r^--', linewidth=2, markersize=8,
                label='A* path (waypoints)', alpha=0.7, zorder=1)
    
    smooth_x = [p[0] for p in smooth_path]
    smooth_y = [p[1] for p in smooth_path]
    ax1.plot(smooth_x, smooth_y, 'g-', linewidth=3, label='Smooth path (target)',
            alpha=0.8, zorder=2)
    ax1.plot(smooth_x, smooth_y, 'go', markersize=6, zorder=3)
    
    true_trace, = ax1.plot([], [], 'b-', linewidth=2, label='True trajectory', zorder=4)
    est_trace, = ax1.plot([], [], 'm--', linewidth=1.5, label='Est trajectory', zorder=4, alpha=0.7)
    true_pt, = ax1.plot([], [], 'bo', markersize=12, zorder=5, label='Current pos')
    est_pt, = ax1.plot([], [], 'ms', markersize=10, zorder=5)
    
    uncertainty_circle = PltCircle((0, 0), 0, fill=False, edgecolor='blue',
                                   linestyle=':', linewidth=2, alpha=0.5, zorder=6)
    ax1.add_patch(uncertainty_circle)
    
    ax1.plot([log[0]['true_pos'][0]], [log[0]['true_pos'][1]], 'go', markersize=15,
            label='Start', zorder=7, markeredgecolor='darkgreen', markeredgewidth=2)
    ax1.plot([smooth_path[-1][0]], [smooth_path[-1][1]], 'r*', markersize=25,
            label='Goal', zorder=7)
    
    ax1.legend(loc='upper right', fontsize=10, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    
    ax2 = axes[1]
    ax2.set_title("Error Metrics", fontsize=14, fontweight='bold')
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
        uncertainty_circle.set_radius(max(uncertainties) * 0.5 if len(uncertainties) > 0 else 0)
        
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

def main():
    print("\n" + "="*70)
    print("SLAM + EKF 기반 자율 주차 시스템")
    print("="*70)
    
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
    start_grid = (0, 9)
    end_grid = (11, 1)
    slam_map = maze_map(maze, start_grid, end_grid, cell_size=cell_size)
    
    G1 = (55.0, 5.0)
    
    landmarks = [
        Landmark(1, (25.0, 20.0), max_range=50.0, std=0.4),
        Landmark(2, (40.0, 15.0), max_range=50.0, std=0.4),
        Landmark(3, (15.0, 30.0), max_range=50.0, std=0.4),
        Landmark(4, (35.0, 35.0), max_range=50.0, std=0.4),
    ]
    
    # Phase 1: SLAM
    slam_ekf, confirmed_pos, current_pos, log_slam = run_slam_phase(landmarks, num_steps=5)
    
    # Phase 2: 경로 계획
    print("\n" + "="*70)
    print("Phase 2: 경로 계획 (A* + Smooth)")
    print("="*70)
    
    start_grid_x = max(0, min(int(confirmed_pos[0] / cell_size), len(maze[0]) - 1))
    start_grid_y = max(0, min(int(confirmed_pos[1] / cell_size), len(maze) - 1))
    goal_grid_x = max(0, min(int(G1[0] / cell_size), len(maze[0]) - 1))
    goal_grid_y = max(0, min(int(G1[1] / cell_size), len(maze) - 1))
    
    start_grid = (start_grid_x, start_grid_y)
    goal_grid = (goal_grid_x, goal_grid_y)
    
    print("확인된 위치: (%.2f, %.2f)" % (confirmed_pos[0], confirmed_pos[1]))
    print("그리드: %s -> %s" % (start_grid, goal_grid))
    
    astar_path_result = simple_astar_path(maze, start_grid, goal_grid)
    
    if astar_path_result:
        world_path = []
        for gx, gy in astar_path_result:
            wx = gx * cell_size + cell_size / 2
            wy = gy * cell_size + cell_size / 2
            world_path.append([wx, wy])
        print("[OK] A* 경로: %d개 점" % len(world_path))
    else:
        world_path = [list(confirmed_pos), list(G1)]
        print("[OK] 직선 경로로 대체")
    
    print("경로 스무딩 중...")
    smooth_path = smooth_path_nd(world_path, alpha=0.1, beta=0.3, max_iter=500)
    print("[OK] Smooth 경로: %d개 점" % len(smooth_path))
    
    # Phase 3: 주차
    log_parking, final_smooth_path = run_parking_phase(slam_ekf, smooth_path, landmarks, current_pos, log_slam)
    
    # 시각화
    print("\n시각화 중...")
    visualize_parking(maze, log_parking, final_smooth_path, landmarks, world_path)
    
    print("\n" + "="*70)
    print("자율 주차 완료!")
    print("="*70)

if __name__ == "__main__":
    main()
