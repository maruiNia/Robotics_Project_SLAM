"""
SLAM + EKF 주차 프로그램 - 빠른 테스트 버전 (30 스텝만 실행)
"""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import module.MySlam as MySlam
from module.MySlam import maze_map

# 짧은 테스트용 main
if __name__ == "__main__":
    print("\n" + "="*70)
    print("SLAM + EKF 기반 자율 주차 시스템 - 빠른 테스트")
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
    
    # 주차 위치
    G1 = (55.0, 5.0)
    
    print(f"미로 크기: {len(maze[0])} x {len(maze)}")
    print(f"셀 크기: {cell_size}m")
    print(f"시작점: {start} (world: ({start[0]*cell_size+cell_size/2:.1f}, {start[1]*cell_size+cell_size/2:.1f}))")
    print(f"종료점: {end} (world: ({end[0]*cell_size+cell_size/2:.1f}, {end[1]*cell_size+cell_size/2:.1f}))")
    print(f"주차 위치: {G1}")
    
    # 간단한 A* 테스트
    from heapq import heappush, heappop
    
    def simple_astar(maze, start, goal):
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
                return list(reversed(path)), iterations
            
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
        
        return None, iterations
    
    # A* 실행
    print("\n" + "="*70)
    print("A* 경로 탐색")
    print("="*70)
    path, iterations = simple_astar(maze, start, end)
    
    if path:
        print(f"[OK] 경로 찾음! 길이: {len(path)}, 반복: {iterations}")
        print("경로 상세:")
        for i in range(0, len(path), max(1, len(path)//10)):
            gx, gy = path[i]
            wx = gx * cell_size + cell_size / 2
            wy = gy * cell_size + cell_size / 2
            print(f"  [{i:2d}] Grid ({gx:2d}, {gy:2d}) -> World ({wx:6.1f}, {wy:6.1f})")
        print(f"  ...{len(path)-1}까지")
    else:
        print(f"[FAIL] 경로를 찾을 수 없습니다! (반복: {iterations})")
    
    print("\n" + "="*70)
    print("기본 컴포넌트 테스트 완료")
    print("="*70)
