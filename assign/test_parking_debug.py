"""
SLAM + EKF 기반 자율 주차 시스템 - 디버그 버전
A* 경로 탐색 및 smooth path 추적 확인용
"""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from module.MyPlanning import smooth_path_nd, point_to_path_min_distance_nd

def simple_astar_path(maze, start, goal):
    """A* 경로 탐색"""
    from heapq import heappush, heappop
    
    print(f"\n{'='*70}")
    print(f"A* 경로 탐색: {start} → {goal}")
    print(f"미로 크기: {len(maze[0])} x {len(maze)}")
    print(f"{'='*70}")
    
    # 시작점과 목표점 확인
    print(f"시작점: maze[{start[1]}][{start[0]}] = {maze[start[1]][start[0]]}")
    print(f"목표점: maze[{goal[1]}][{goal[0]}] = {maze[goal[1]][goal[0]]}")
    
    open_set = []
    closed_set = set()
    parent_map = {}
    g_score = {start: 0}
    
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    h = heuristic(start, goal)
    heappush(open_set, (h, 0, start))
    
    iterations = 0
    max_iterations = 10000
    
    while open_set and iterations < max_iterations:
        iterations += 1
        f, g, current = heappop(open_set)
        
        if current == goal:
            path = []
            node = goal
            while node in parent_map:
                path.append(node)
                node = parent_map[node]
            path.append(start)
            path = list(reversed(path))
            print(f"\n✓ 경로 찾음!")
            print(f"  길이: {len(path)}")
            print(f"  반복: {iterations}")
            return path
        
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
    
    print(f"\n✗ 경로를 찾을 수 없습니다! (반복: {iterations})")
    return None

def test_astar():
    """A* 경로 탐색 테스트"""
    print("\n" + "="*70)
    print("테스트 1: A* 경로 탐색")
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
    
    # 좌상단 (0, 9) -> 우하단 (11, 1)
    start_grid = (0, 9)
    goal_grid = (11, 1)
    
    path = simple_astar_path(maze, start_grid, goal_grid)
    
    if path:
        print(f"\n경로 상세:")
        for i, (gx, gy) in enumerate(path):
            wx = gx * cell_size + cell_size / 2
            wy = gy * cell_size + cell_size / 2
            print(f"  [{i:2d}] Grid ({gx:2d}, {gy:2d}) → World ({wx:5.1f}, {wy:5.1f})")

def test_smooth_path():
    """경로 스무딩 테스트"""
    print("\n" + "="*70)
    print("테스트 2: 경로 스무딩")
    print("="*70)
    
    # 간단한 경로
    waypoints = [
        [2.5, 47.5],   # 시작
        [32.5, 47.5],  # 중간
        [55.0, 5.0],   # 목표
    ]
    
    print(f"원본 경로 ({len(waypoints)}개 점):")
    for i, pt in enumerate(waypoints):
        print(f"  [{i}] ({pt[0]:.1f}, {pt[1]:.1f})")
    
    smooth_path = smooth_path_nd(waypoints, alpha=0.1, beta=0.3, max_iter=500)
    
    print(f"\n스무딩 후 경로 ({len(smooth_path)}개 점):")
    for i, pt in enumerate(smooth_path):
        print(f"  [{i}] ({pt[0]:.1f}, {pt[1]:.1f})")

def test_path_distance():
    """경로 거리 계산 테스트"""
    print("\n" + "="*70)
    print("테스트 3: 경로 거리 계산")
    print("="*70)
    
    smooth_path = [
        [2.5, 47.5],
        [18.75, 47.5],
        [35.0, 47.5],
        [51.25, 47.5],
        [55.0, 26.25],
        [55.0, 5.0],
    ]
    
    test_points = [
        (2.5, 47.5),    # 경로 시작점
        (10.0, 47.5),   # 경로 위
        (10.0, 40.0),   # 경로 위쪽
        (55.0, 5.0),    # 경로 끝점
    ]
    
    for pos in test_points:
        d, q, seg_i, t = point_to_path_min_distance_nd(pos, smooth_path)
        print(f"위치 ({pos[0]:.1f}, {pos[1]:.1f})")
        print(f"  → 최소거리: {d:.3f}m")
        print(f"  → 최근접점: ({q[0]:.1f}, {q[1]:.1f})")
        print(f"  → 선분: [{seg_i}] ({smooth_path[seg_i][0]:.1f}, {smooth_path[seg_i][1]:.1f}) "
              f"→ ({smooth_path[seg_i+1][0]:.1f}, {smooth_path[seg_i+1][1]:.1f})")
        print(f"  → t: {t:.3f}")
        print()

if __name__ == "__main__":
    test_astar()
    test_smooth_path()
    test_path_distance()
    
    print("\n" + "="*70)
    print("모든 테스트 완료!")
    print("="*70)
