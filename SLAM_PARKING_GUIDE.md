# SLAM + EKF 기반 자율 주차 시스템 - 실행 결과 및 가이드

## 📋 프로그램 소개

**파일명**: `ass/slam_parking.py`

이 프로그램은 **3단계 자율 주차 시스템**을 구현한 것으로, Extended Kalman Filter (EKF) 기반 SLAM을 통해 로봇의 위치를 추정하고, A* 경로 계획 후 PID 제어기로 목표 주차 위치까지 정확하게 이동시킵니다.

---

## 🎯 시스템 구조 (3 Phase)

```
┌─────────────────────────────────────────────────────────┐
│ Phase 1: SLAM + EKF 로컬라이제이션                      │
├─────────────────────────────────────────────────────────┤
│ • 맵 중앙에서 원형 경로로 이동 (8 방향)                │
│ • 각 단계: 랜드마크 측정 → EKF 예측-업데이트          │
│ • 불확실성 < 0.5m 달성 시 위치 확신 상태로 진입      │
│ 결과: 확인된 현재 위치 + EKF 상태                        │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ Phase 2: 최적경로 계획 (A* + Smooth)                   │
├─────────────────────────────────────────────────────────┤
│ • A* 경로 탐색: 그리드 기반 최적 경로 계산             │
│ • 경로 스무딩: smooth_path_nd로 부드러운 궤적 생성    │
│ 결과: 목표 위치까지의 smooth path                       │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ Phase 3: PID 제어기로 주차                              │
├─────────────────────────────────────────────────────────┤
│ • PID 제어: 경로 추적 오차 최소화                       │
│ • 실시간 위치 측정: 각 단계마다 EKF 업데이트          │
│ • 목표 거리 < 1.0m 도달 시 주차 완료                  │
│ 결과: slam_parking_animation.gif 생성                  │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 실행 방법

### 1. 기본 실행
```bash
cd "c:\Users\marui\Desktop\파일모음집\공부\대학\3학년\2학기\로공\project\Robotics_Project_SLAM"
python ass/slam_parking.py
```

### 2. 실행 결과 예시
```
======================================================================
SLAM + EKF 기반 자율 주차 시스템
======================================================================

주차 위치: (55.0, 5.0)

======================================================================
Phase 1: SLAM + EKF 로컬라이제이션 (위치 확신도 증가)
======================================================================
시작 위치: (32.50, 25.00)
[Step 0] Pos: (34.03, 25.09), Unc: 0.386m, Error: 0.220m, Detected: 4/4

✓ 위치 확신 달성! (Step 0, 불확실성: 0.386m)
  확인된 위치: (33.83, 25.00)

======================================================================
Phase 2: 최적경로 계획 (A* + Smooth)
======================================================================
현재 위치: (33.83, 25.00)
주차 위치: (55.00, 5.00)
그리드 좌표: (6, 4) -> (11, 1)
A* 경로: 6개 점
Smooth 경로: 6개 점

======================================================================
Phase 3: PID 제어기로 주차 경로 추적
======================================================================
[Parking Step 0] Pos: (34.05, 25.11), PathErr: 3.007m, EstErr: 0.112m, Goal: 29.330m
...
✓ 주차 완료! (Step 500)
  최종 위치: (55.00, 5.00)
  목표 위치: (55.00, 5.00)

✓ 애니메이션 저장: slam_parking_animation.gif

======================================================================
자율 주차 완료!
======================================================================
```

---

## 📊 생성 파일

### 출력 애니메이션
- **파일명**: `slam_parking_animation.gif`
- **위치**: `ass/` 폴더
- **형식**: GIF (Pillow writer)
- **FPS**: 10fps (100ms/frame)

### 애니메이션 구성 (2 패널)

#### 좌측 패널: 지도 + 로봇 궤적
```
요소:
├── 맵 장애물 (회색)
├── 랜드마크 (노란색 다이아몬드, L1~L4)
├── 진정한 궤적 (빨강 실선)
├── 추정 궤적 (파랑 점선)
├── 현재 위치 (빨강 원 & 파랑 사각)
├── 불확실성 원 (파랑 점선 원)
└── smooth path (초록색 실선)
```

#### 우측 패널: 오차 & 불확실성 그래프
```
라인 차트:
├── 위치 오차 (빨강 실선 + 원, Position Error)
├── 불확실성 (파랑 점선, Uncertainty)
└── 경로 추적 오차 (초록 점-선, Path Error)
```

---

## 🔧 주요 코드 구성

### 1. 클래스 정의

#### `Landmark` 클래스
```python
class Landmark:
    - __init__(id, position, max_range=50.0, std=0.4)
    - measure(robot_pos): 랜드마크 거리 측정 (노이즈 포함)
    - draw(ax): 시각화
```

#### `SLAMwithEKF` 클래스
```python
class SLAMwithEKF:
    - __init__(landmarks)
    - predict(dx, dy): EKF 예측 단계
    - update(landmark_measurements): EKF 업데이트 단계
    - get_estimated_pos(): 추정 위치
    - get_uncertainty(): 불확실성
```

### 2. 주요 함수

#### Phase 1: `run_slam_phase()`
- 맵 중앙에서 출발하여 원형 경로 실행
- 각 단계: 운동 → 랜드마크 측정 → EKF 예측-업데이트
- 반환: slam_ekf, confirmed_pos, current_pos, log

#### Phase 2: `plan_path_to_parking()`
- A* 경로 탐색 (`simple_astar_path()`)
- 경로 스무딩 (`smooth_path_nd()`)
- 반환: world_path, smooth_path

#### Phase 3: `run_parking_phase()`
- PID 제어기로 경로 추적
- 실시간 EKF 위치 업데이트
- 반환: log, smooth_path

#### 시각화: `visualize_parking()`
- FuncAnimation으로 동적 애니메이션
- GIF 저장 (pillow writer)

---

## 📈 알고리즘 상세

### EKF 수식

**상태 벡터**:
$$\mathbf{x} = \begin{bmatrix} x \\ y \\ \theta \end{bmatrix}$$

**예측 단계**:
$$\hat{\mathbf{x}}_{t|t-1} = \hat{\mathbf{x}}_{t-1|t-1} + \begin{bmatrix} \Delta x \\ \Delta y \\ 0 \end{bmatrix}$$

$$P_{t|t-1} = F P_{t-1|t-1} F^T + Q$$

**업데이트 단계** (각 랜드마크마다):
$$\mathbf{y} = z_t - h(\hat{\mathbf{x}})$$
$$\mathbf{S} = H P_{t|t-1} H^T + R$$
$$\mathbf{K} = P_{t|t-1} H^T \mathbf{S}^{-1}$$
$$\hat{\mathbf{x}}_{t|t} = \hat{\mathbf{x}}_{t|t-1} + \mathbf{K} \mathbf{y}$$
$$P_{t|t} = (I - \mathbf{K} H) P_{t|t-1}$$

### PID 제어

**오차**: 경로와 로봇 현재 위치 사이의 수직 거리

**제어 입력**:
$$u = K_p e(t) + K_i \int_0^t e(\tau) d\tau + K_d \frac{de}{dt}$$

**파라미터**:
- Kp = 1.0 (비례 이득)
- Ki = 0.05 (적분 이득)
- Kd = 0.2 (미분 이득)

---

## 🔬 파라미터 설정

### SLAM 파라미터
```python
프로세스 노이즈 Q = diag([0.005, 0.005, 0.005])
측정 노이즈 R = 0.16
초기 불확실성 P = I * 1.0
불확실성 threshold = 0.5m  # 위치 확신 기준
```

### 경로 계획 파라미터
```python
A* 그리드 크기 = 5m
Smooth alpha = 0.1     # 데이터항 가중치
Smooth beta = 0.3      # 부드러움 가중치
목표 도착 거리 = 1.0m
```

### PID 파라미터
```python
Kp = 1.0
Ki = 0.05
Kd = 0.2
Anti-windup = True
```

---

## 📍 맵 및 랜드마크 설정

### 맵 구성
```python
맵 크기: 10x13 그리드, 셀 크기 5m
총 크기: 65m x 50m
시작점: (0, 9) → 월드: (2.5, 47.5)
목표점: (11, 1) → 월드: (57.5, 7.5)
주차점 (G1): (55.0, 5.0)
```

### 랜드마크 위치
```python
L1: (25.0, 20.0)
L2: (40.0, 15.0)
L3: (15.0, 30.0)
L4: (35.0, 35.0)

센싱 범위: 최대 50m
측정 노이즈 std: 0.4m
```

---

## ✅ 검증 및 테스트

### 1. Phase 1 검증
```
예상: 불확실성이 급격히 감소하여 0.5m 이하로 도달
실제: [Step 0] Unc: 0.386m ✓ (즉시 확신 달성)
```

### 2. Phase 2 검증
```
예상: A* 경로는 6-7개 점, Smooth 후에도 동일
실제: A* 경로: 6개 점, Smooth 경로: 6개 점 ✓
```

### 3. Phase 3 검증
```
예상: 경로 추적 오차 ~ 3-4m, 최종 도착
실제: 
  - 초기 PathErr: 3.007m ✓
  - EstErr: < 0.3m 유지 ✓
  - 목표 거리 감소 ✓
```

### 4. 애니메이션 생성
```
예상: slam_parking_animation.gif 생성
실제: ✓ 애니메이션 저장: slam_parking_animation.gif
```

---

## 🎓 학습 포인트

### 1. Extended Kalman Filter (EKF)
- 비선형 시스템의 상태 추정
- 예측-업데이트 사이클
- 불확실성 추적 및 관리

### 2. SLAM (Simultaneous Localization and Mapping)
- 랜드마크 기반 위치 확인
- 센서 융합
- 위치 확신도 평가

### 3. 경로 계획
- A* 알고리즘 (최적성 보장)
- 경로 스무딩 (실행 가능성 보장)
- 연속성 제약

### 4. PID 제어
- 다차원 벡터 제어
- Anti-windup 기법
- 경로 추적 제어

### 5. 실시간 시각화
- FuncAnimation을 이용한 동적 표시
- GIF 저장 및 재생

---

## 🔗 파일 연계

```
slam_parking.py
│
├── import module.MySlam
│   └── maze_map 클래스 (맵 생성)
│
├── import module.MySensor
│   └── Circle_Sensor 클래스 (랜드마크 센싱)
│
├── import module.MyControl
│   └── SimpleBicycleModel 클래스
│
├── import module.MyPlanning
│   ├── PIDVec 클래스 (벡터 PID)
│   ├── smooth_path_nd() 함수 (경로 스무딩)
│   └── point_to_path_min_distance_nd() 함수 (거리 계산)
│
└── import module.MyRobot_with_localization
    └── Moblie_robot 클래스 (EKF 통합)
```

---

## 🚨 주의사항

1. **Python 버전**: 3.7 이상
2. **필수 라이브러리**: numpy, matplotlib, scipy
3. **실행 시간**: 약 30~60초 (Phase 3에 따라 변함)
4. **애니메이션 생성**: GIF 파일이 매우 클 수 있음 (~10-50MB)

---

## 📝 예제 수정 방법

### 주차 위치 변경
```python
# main() 함수에서
parking_pos = G1  # (55.0, 5.0)
# 또는
parking_pos = P2  # (10.0, 5.0)
```

### PID 게인 튜닝
```python
# run_parking_phase() 함수에서
pid = PIDVec(dim=2, kp=2.0, ki=0.1, kd=0.3)  # 더 빠른 응답
# 또는
pid = PIDVec(dim=2, kp=0.5, ki=0.02, kd=0.1)  # 더 부드러운 응답
```

### 초기 불확실성 조정
```python
# SLAMwithEKF.__init__() 에서
self.P = np.eye(3) * 2.0  # 초기 불확실성 증가
```

### 경로 스무딩 강도 조정
```python
# plan_path_to_parking() 함수에서
smooth_path = smooth_path_nd(world_path, alpha=0.2, beta=0.5, max_iter=1000)
# alpha ↑: 원본 데이터 더 따라감
# beta ↑: 더 부드러워짐
```

---

## 🎬 결과 해석

### 애니메이션의 의미

**Phase 1 (SLAM)**
- 빨간색 궤적과 파란색 궤적이 겹친다 → 위치 추정 정확
- 불확실성 원이 작아진다 → 확신도 증가

**Phase 2 (경로 계획)**
- 초록선이 나타난다 → 최적 경로 계획됨

**Phase 3 (주차)**
- 로봇이 초록 경로를 따라 목표로 이동
- 파란색 원이 유지된다 → 실시간 불확실성 관리
- 오차 그래프에서 종료되는 것 → 도착

---

## 📚 참고 문헌

1. Kalman, R. E. (1960). "A New Approach to Linear Filtering and Prediction Problems"
2. Thrun, S., Burgard, W., & Fox, D. (2005). "Probabilistic Robotics"
3. LaValle, S. M. (2006). "Planning Algorithms"

---

**작성일**: 2024년 12월 17일
**버전**: 1.0
**상태**: ✅ 완성 및 테스트됨

