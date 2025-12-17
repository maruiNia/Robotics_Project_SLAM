"""
MyRobot 기반 실시간 SLAM + Kalman Filter 로컬라이제이션
- 랜드마크 기반 SLAM (최대 4개)
- EKF로 움직이면서 위치 추정 구체화
- 실시간 애니메이션으로 시각화
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
from module.MyPlanning import PIDVec
from module.MyRobot_with_localization import Moblie_robot
import numpy as np


def draw_obstacles(ax, slam_map: MySlam, *, alpha=0.20):
    """맵의 장애물 그리기"""
    for obs in slam_map.get_obs_list():
        (min_x, min_y), (max_x, max_y) = obs.bounds()
        ax.add_patch(Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, alpha=alpha, color='gray'))


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
            return None  # 범위 밖
        
        # 가우시안 노이즈 추가
        measured_distance = true_distance + np.random.normal(0, self.std)
        return max(0.0, measured_distance)
    
    def draw(self, ax, color='yellow', markersize=12):
        """랜드마크 시각화 (마름모 모양)"""
        x, y = self.position
        diamond = Polygon([
            [x, y + 0.3],        # 위
            [x + 0.3, y],        # 오른쪽
            [x, y - 0.3],        # 아래
            [x - 0.3, y],        # 왼쪽
        ], closed=True, fill=True, color=color, alpha=0.8, edgecolor='black', linewidth=2)
        ax.add_patch(diamond)
        ax.text(x, y - 0.6, f'L{self.id}', ha='center', fontsize=8, fontweight='bold')


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
            return None  # 범위 밖
        
        # 가우시안 노이즈 추가
        measured_distance = true_distance + np.random.normal(0, self.std)
        return max(0.0, measured_distance)
    
    def draw(self, ax, color='yellow', markersize=12):
        """랜드마크 시각화 (마름모 모양)"""
        x, y = self.position
        diamond = Polygon([
            [x, y + 0.3],        # 위
            [x + 0.3, y],        # 오른쪽
            [x, y - 0.3],        # 아래
            [x - 0.3, y],        # 왼쪽
        ], closed=True, fill=True, color=color, alpha=0.8, edgecolor='black', linewidth=2)
        ax.add_patch(diamond)
        ax.text(x, y - 0.6, f'L{self.id}', ha='center', fontsize=8)


def get_landmark_measurements(landmarks, robot_pos):
    """모든 랜드마크의 측정값 반환"""
    measurements = []
    for lm in landmarks:
        dist = lm.measure(robot_pos)
        measurements.append((lm.id, dist))
    return measurements


class SLAMwithEKF:
    """SLAM + EKF 통합 로컬라이제이션"""
    
    def __init__(self, landmarks):
        self.landmarks = landmarks
        
        # EKF 상태: [x, y, theta]
        self.x = np.array([0.0, 0.0, 0.0], dtype=float)  # 초기 추정값
        self.P = np.eye(3) * 1.0  # 초기 불확실성 (줄임)
        
        # 프로세스 및 측정 노이즈
        self.Q = np.diag([0.005, 0.005, 0.005])  # 프로세스 노이즈 (줄임)
        self.R = np.eye(len(landmarks)) * 0.16  # 0.4^2
    
    def predict(self, dx, dy, dt=1.0):
        """EKF 예측: 모션 모델로 상태 전이"""
        x, y, theta = self.x
        
        # 간단한 모션 모델
        x_new = x + dx
        y_new = y + dy
        
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            theta_new = math.atan2(dy, dx)
        else:
            theta_new = theta
        
        # 상태 업데이트
        self.x = np.array([x_new, y_new, theta_new])
        
        # 공분산 예측 (프로세스 노이즈 추가)
        F = np.eye(3)
        self.P = F @ self.P @ F.T + self.Q
    
    def update(self, landmark_measurements):
        """EKF 업데이트: 랜드마크 측정값으로 보정"""
        # 측정값 벡터 구성
        z = []
        H_list = []
        
        for lm_id, meas in landmark_measurements:
            if meas is not None:
                z.append(meas)
                
                # 해당 랜드마크 찾기
                lm = next((l for l in self.landmarks if l.id == lm_id), None)
                if lm is not None:
                    dx = lm.position[0] - self.x[0]
                    dy = lm.position[1] - self.x[1]
                    dist = math.sqrt(dx**2 + dy**2)
                    
                    # Jacobian 행
                    if dist > 1e-6:
                        h = [-dx/dist, -dy/dist, 0]
                    else:
                        h = [0, 0, 0]
                    H_list.append(h)
        
        if len(z) == 0:
            return
        
        z = np.array(z, dtype=float)
        H = np.array(H_list)
        
        # 예상 측정값
        z_pred = []
        for lm_id, meas in landmark_measurements:
            if meas is not None:
                lm = next((l for l in self.landmarks if l.id == lm_id), None)
                if lm is not None:
                    dist = np.linalg.norm(lm.position - self.x[:2])
                    z_pred.append(dist)
        
        z_pred = np.array(z_pred)
        
        # 혁신
        y = z - z_pred
        
        # 혁신 공분산
        R_reduced = self.R[:len(z), :len(z)]
        S = H @ self.P @ H.T + R_reduced
        
        # Kalman 이득
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
            
            # 상태 보정
            self.x = self.x + K @ y
            
            # 공분산 보정
            self.P = (np.eye(3) - K @ H) @ self.P
        except:
            pass  # 특이 행렬 시 업데이트 스킵
    
    def get_estimated_pos(self):
        """추정 위치 반환"""
        return self.x[:2]
    
    def get_uncertainty(self):
        """불확실성 반환 (공분산의 trace)"""
        return np.sqrt(np.trace(self.P[:2, :2]))


def run_slam_with_animation(landmarks, slam_map, commands):
    """MyRobot 기반 SLAM 시뮬레이션 + 실시간 애니메이션"""
    
    # 1) 로봇 생성
    sensor = Circle_Sensor(dim=2, number=12, distance=15.0, slam_map=slam_map, step=0.1)
    
    robot = Moblie_robot(
        dim=2,
        position=slam_map.start_point,
        end_point=slam_map.end_point,
        update_rate=50_000_000,  # 50ms
        sensing_mode=False,  # 기본은 센싱 비활성화
        sensor=sensor,
    )
    
    # 2) SLAM + EKF 초기화
    slam_ekf = SLAMwithEKF(landmarks)
    
    # 3) 로그 저장
    log = []
    
    print(f"\n{'='*70}")
    print(f"MyRobot 기반 SLAM + Kalman Filter 로컬라이제이션")
    print(f"{'='*70}")
    print(f"시작점: {slam_map.start_point}")
    print(f"랜드마크: {len(landmarks)}개")
    for lm in landmarks:
        print(f"  L{lm.id}: {lm.position}")
    
    # 4) 명령 실행
    for step, (cmd_dx, cmd_dy) in enumerate(commands):
        print(f"\n[Step {step}] 명령: ({cmd_dx:+.1f}m, {cmd_dy:+.1f}m)")
        
        # 4-1) 로봇 이동 명령
        robot.ins_acc_mov(
            velocity=(cmd_dx, cmd_dy),
            steering=(0, 0),
            time=50_000_000  # 50ms
        )
        
        # 4-2) 모션 적분 (시뮬레이션)
        # 간단한 적분: position = position + velocity * dt
        true_pos = np.array(robot._position)
        actual_dx = cmd_dx + np.random.normal(0, 0.1)  # 모션 노이즈
        actual_dy = cmd_dy + np.random.normal(0, 0.1)
        new_pos = true_pos + np.array([actual_dx, actual_dy])
        new_pos = np.clip(new_pos, [0, 0], [65, 50])
        robot._position = tuple(new_pos)
        
        # 4-3) 랜드마크 센싱
        landmark_meas = get_landmark_measurements(landmarks, new_pos)
        
        # 4-4) SLAM EKF 업데이트
        slam_ekf.predict(actual_dx, actual_dy)
        slam_ekf.update(landmark_meas)
        
        # 4-5) 추정 위치
        est_pos = slam_ekf.get_estimated_pos()
        uncertainty = slam_ekf.get_uncertainty()
        error = np.linalg.norm(new_pos - est_pos)
        
        # 센싱된 랜드마크 수
        num_detected = sum(1 for _, meas in landmark_meas if meas is not None)
        
        print(f"  실제 위치: ({new_pos[0]:.2f}, {new_pos[1]:.2f})")
        print(f"  추정 위치: ({est_pos[0]:.2f}, {est_pos[1]:.2f})")
        print(f"  위치오차: {error:.3f}m, 불확실성: {uncertainty:.3f}m")
        print(f"  감지 랜드마크: {num_detected}/{len(landmarks)}개")
        
        # 4-6) 로그 저장
        log.append({
            "step": step,
            "true_pos": new_pos.copy(),
            "est_pos": est_pos.copy(),
            "error": error,
            "uncertainty": uncertainty,
            "num_detected": num_detected,
            "measurements": landmark_meas,
            "covariance": slam_ekf.P.copy(),
        })
    
    return slam_map, slam_ekf, landmarks, log


def animate_slam_localization(slam_map, slam_ekf, landmarks, log):
    """SLAM 로컬라이제이션 애니메이션"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # ===== 메인 지도 (좌상단) =====
    ax1.set_title("Real-time SLAM Localization with Landmarks", fontsize=12, fontweight='bold')
    ax1.set_aspect("equal")
    ax1.set_xlim(-2, 67)
    ax1.set_ylim(-2, 52)
    
    # 맵 그리기
    draw_obstacles(ax1, slam_map, alpha=0.3)
    
    # 랜드마크 그리기
    for lm in landmarks:
        lm.draw(ax1)
    
    # 궤적 라인
    true_trace, = ax1.plot([], [], 'r-', linewidth=2, label='True trajectory', zorder=2)
    est_trace, = ax1.plot([], [], 'b--', linewidth=2, label='Estimated trajectory', zorder=2)
    
    # 현재 위치
    true_pt, = ax1.plot([], [], 'ro', markersize=10, label='True position', zorder=5)
    est_pt, = ax1.plot([], [], 'bs', markersize=10, label='Estimated position', zorder=5)
    
    # 불확실성 원
    uncertainty_circle = PltCircle((0, 0), 0, fill=False, edgecolor='blue', linestyle=':', linewidth=2, alpha=0.5)
    ax1.add_patch(uncertainty_circle)
    
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    
    # ===== 오차 그래프 (우상단) =====
    ax2.set_title("Localization Error", fontsize=12, fontweight='bold')
    error_line, = ax2.plot([], [], 'r-o', linewidth=2, markersize=4, label='Position Error')
    uncertainty_line, = ax2.plot([], [], 'b--', linewidth=2, label='Uncertainty (1σ)')
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Error (m)")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # ===== 랜드마크 감지 (좌하단) =====
    ax3.set_title("Detected Landmarks per Step", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Number of Landmarks")
    ax3.set_ylim(0, len(landmarks) + 1)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ===== 상태 정보 (우하단) =====
    ax4.axis('off')
    status_text = ax4.text(0.05, 0.95, "", transform=ax4.transAxes, fontsize=11,
                           verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def init():
        true_trace.set_data([], [])
        est_trace.set_data([], [])
        true_pt.set_data([], [])
        est_pt.set_data([], [])
        error_line.set_data([], [])
        uncertainty_line.set_data([], [])
        status_text.set_text("")
        return true_trace, est_trace, true_pt, est_pt, error_line, uncertainty_line, status_text
    
    def update(frame):
        # 데이터 추출
        true_positions = np.array([log[i]['true_pos'] for i in range(frame + 1)])
        est_positions = np.array([log[i]['est_pos'] for i in range(frame + 1)])
        errors = np.array([log[i]['error'] for i in range(frame + 1)])
        uncertainties = np.array([log[i]['uncertainty'] for i in range(frame + 1)])
        num_detected = np.array([log[i]['num_detected'] for i in range(frame + 1)])
        
        # 궤적
        true_trace.set_data(true_positions[:, 0], true_positions[:, 1])
        est_trace.set_data(est_positions[:, 0], est_positions[:, 1])
        
        # 현재 위치
        true_pt.set_data([true_positions[-1, 0]], [true_positions[-1, 1]])
        est_pt.set_data([est_positions[-1, 0]], [est_positions[-1, 1]])
        
        # 불확실성 원
        uncertainty_circle.set_center(est_positions[-1])
        uncertainty_circle.set_radius(uncertainties[-1] * 2)
        
        # 오차 그래프
        error_line.set_data(range(len(errors)), errors)
        uncertainty_line.set_data(range(len(uncertainties)), uncertainties)
        
        # 축 자동 조정
        ax2.set_xlim(-0.5, max(len(errors), 5))
        ax2.set_ylim(0, max(np.max(errors), np.max(uncertainties), 1) * 1.2)
        
        # 랜드마크 감지 바
        ax3.clear()
        ax3.bar(range(len(num_detected)), num_detected, color='green', alpha=0.7)
        ax3.set_xlabel("Step")
        ax3.set_ylabel("Number of Landmarks")
        ax3.set_ylim(0, len(landmarks) + 1)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_title("Detected Landmarks per Step", fontsize=12, fontweight='bold')
        
        # 상태 정보
        current = log[frame]
        info_text = f"""
Step: {frame}
True Pos:  ({current['true_pos'][0]:6.2f}, {current['true_pos'][1]:6.2f})
Est Pos:   ({current['est_pos'][0]:6.2f}, {current['est_pos'][1]:6.2f})
Error:     {current['error']:6.3f} m
Unc:       {current['uncertainty']:6.3f} m
Detected:  {current['num_detected']}/{len(landmarks)}
        """
        status_text.set_text(info_text)
        
        return true_trace, est_trace, true_pt, est_pt, error_line, uncertainty_line, status_text
    
    ani = FuncAnimation(
        fig, update,
        frames=len(log),
        init_func=init,
        interval=100,  # 100ms per frame (초당 10fps)
        blit=False,
        repeat=True
    )
    
    plt.tight_layout()
    ani.save("slam_kal_animation.gif", writer="pillow", fps=10)
    print("\n✓ 애니메이션 저장: slam_kal_animation.gif")
    plt.show()


def animate_map_trajectory(slam_map, slam_ekf, landmarks, log):
    """지도와 궤적만 표시하는 애니메이션 (ax1)"""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # ===== 메인 지도 =====
    ax.set_title("SLAM Localization - Map Trajectory", fontsize=14, fontweight='bold')
    ax.set_aspect("equal")
    ax.set_xlim(-2, 67)
    ax.set_ylim(-2, 52)
    
    # 맵 그리기
    draw_obstacles(ax, slam_map, alpha=0.3)
    
    # 랜드마크 그리기
    for lm in landmarks:
        lm.draw(ax)
    
    # 궤적 라인
    true_trace, = ax.plot([], [], 'r-', linewidth=2, label='True trajectory', zorder=2)
    est_trace, = ax.plot([], [], 'b--', linewidth=2, label='Estimated trajectory', zorder=2)
    
    # 현재 위치
    true_pt, = ax.plot([], [], 'ro', markersize=10, label='True position', zorder=5)
    est_pt, = ax.plot([], [], 'bs', markersize=10, label='Estimated position', zorder=5)
    
    # 불확실성 원
    uncertainty_circle = PltCircle((0, 0), 0, fill=False, edgecolor='blue', linestyle=':', linewidth=2, alpha=0.5)
    ax.add_patch(uncertainty_circle)
    
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    
    def init():
        true_trace.set_data([], [])
        est_trace.set_data([], [])
        true_pt.set_data([], [])
        est_pt.set_data([], [])
        return true_trace, est_trace, true_pt, est_pt
    
    def update(frame):
        true_positions = np.array([log[i]['true_pos'] for i in range(frame + 1)])
        est_positions = np.array([log[i]['est_pos'] for i in range(frame + 1)])
        uncertainties = np.array([log[i]['uncertainty'] for i in range(frame + 1)])
        
        true_trace.set_data(true_positions[:, 0], true_positions[:, 1])
        est_trace.set_data(est_positions[:, 0], est_positions[:, 1])
        true_pt.set_data([true_positions[-1, 0]], [true_positions[-1, 1]])
        est_pt.set_data([est_positions[-1, 0]], [est_positions[-1, 1]])
        
        uncertainty_circle.set_center(est_positions[-1])
        uncertainty_circle.set_radius(uncertainties[-1] * 2)
        
        return true_trace, est_trace, true_pt, est_pt
    
    ani = FuncAnimation(fig, update, frames=len(log), init_func=init, interval=100, blit=True, repeat=True)
    plt.tight_layout()
    ani.save("slam_map_trajectory.gif", writer="pillow", fps=10)
    print("✓ 궤적 애니메이션 저장: slam_map_trajectory.gif")
    plt.close()


def animate_error_graph(log):
    """오차 그래프만 표시하는 애니메이션 (ax2)"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # ===== 오차 그래프 =====
    ax.set_title("Localization Error Over Time", fontsize=14, fontweight='bold')
    error_line, = ax.plot([], [], 'r-o', linewidth=2, markersize=4, label='Position Error')
    uncertainty_line, = ax.plot([], [], 'b--', linewidth=2, label='Uncertainty (1σ)')
    ax.set_xlabel("Step")
    ax.set_ylabel("Error (m)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    def init():
        error_line.set_data([], [])
        uncertainty_line.set_data([], [])
        return error_line, uncertainty_line
    
    def update(frame):
        errors = np.array([log[i]['error'] for i in range(frame + 1)])
        uncertainties = np.array([log[i]['uncertainty'] for i in range(frame + 1)])
        
        error_line.set_data(range(len(errors)), errors)
        uncertainty_line.set_data(range(len(uncertainties)), uncertainties)
        
        ax.set_xlim(-0.5, max(len(errors), 5))
        ax.set_ylim(0, max(np.max(errors), np.max(uncertainties), 1) * 1.2)
        
        return error_line, uncertainty_line
    
    ani = FuncAnimation(fig, update, frames=len(log), init_func=init, interval=100, blit=True, repeat=True)
    plt.tight_layout()
    ani.save("slam_error_graph.gif", writer="pillow", fps=10)
    print("✓ 오차 그래프 애니메이션 저장: slam_error_graph.gif")
    plt.close()


def main():
    print("\n" + "="*70)
    print("MyRobot 기반 SLAM + EKF 실시간 로컬라이제이션")
    print("="*70)
    
    # 1) 맵 생성
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
    print(f"맵 생성 완료: {len(maze)}x{len(maze[0])} 그리드 (셀크기: {cell_size}m)")
    print(f"시작점: {slam_map.start_point}, 끝점: {slam_map.end_point}")
    
    # 2) 랜드마크 정의 (4개) - 맵 중앙 근처에 밀집 배치
    landmarks = [
        Landmark(1, (25.0, 20.0), max_range=50.0, std=0.4),
        Landmark(2, (40.0, 15.0), max_range=50.0, std=0.4),
        Landmark(3, (15.0, 30.0), max_range=50.0, std=0.4),
        Landmark(4, (35.0, 35.0), max_range=50.0, std=0.4),
    ]
    
    print(f"\n랜드마크: {len(landmarks)}개")
    for lm in landmarks:
        print(f"  L{lm.id}: {lm.position} (max_range={lm.max_range}m, std={lm.std}m)")
    
    # 3) 로봇 명령 (더 큰 원형 경로 - 70개 스텝)
    commands = []
    # 8방향으로 더 크게 원을 그리며 이동
    directions = [
        (1.5, 0.0),    # 우측 (크기 증가)
        (1.0, -1.0),   # 우상향 대각선 (크기 증가)
        (0.0, -1.5),   # 상측 (크기 증가)
        (-1.0, -1.0),  # 좌상향 대각선 (크기 증가)
        (-1.5, 0.0),   # 좌측 (크기 증가)
        (-1.0, 1.0),   # 좌하향 대각선 (크기 증가)
        (0.0, 1.5),    # 하측 (크기 증가)
        (1.0, 1.0),    # 우하향 대각선 (크기 증가)
    ]
    
    # 원형 경로 반복 (8 방향 × 8 바퀴 + 6 추가 = 70개)
    for _ in range(8):
        commands.extend(directions)
    commands.extend(directions[:6])  # 6개 추가
    
    print(f"\n명령 수열: {len(commands)}개")
    
    # 4) SLAM 시뮬레이션 + 애니메이션
    slam_map, slam_ekf, landmarks, log = run_slam_with_animation(landmarks, slam_map, commands)
    
    # 5) 애니메이션 생성
    print("\n애니메이션 생성 중...")
    animate_slam_localization(slam_map, slam_ekf, landmarks, log)
    animate_map_trajectory(slam_map, slam_ekf, landmarks, log)
    animate_error_graph(log)
    
    print("\n" + "="*70)
    print("SLAM + EKF 로컬라이제이션 완료!")
    print("="*70)


if __name__ == "__main__":
    main()
