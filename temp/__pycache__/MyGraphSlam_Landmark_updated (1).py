# MyGraphSlam_Landmark.py
import numpy as np
from typing import Dict, Tuple, List

Vec2 = Tuple[float, float]



class GraphSLAM2D:
    """
    Full Graph SLAM (Pose + Landmark)
    State:
      [x0,y0, x1,y1, ..., l0x,l0y, l1x,l1y, ...]
    """

    def __init__(self, prior_info: float = 1e6):
        self.prior_info = prior_info
        self.pose_cnt = 0
        self.landmarks: Dict[int, int] = {}  # lm_id -> state index
        self.Omega = np.zeros((0, 0))
        self.Xi = np.zeros((0, 1))

    # ---------- utilities ----------
    def _expand(self, n: int):
        old = self.Omega.shape[0]
        new = old + n
        O = np.zeros((new, new))
        X = np.zeros((new, 1))
        O[:old, :old] = self.Omega
        X[:old] = self.Xi
        self.Omega, self.Xi = O, X

    def _pose_idx(self, i: int):
        return slice(2*i, 2*i+2)

    def _lm_idx(self, lm_id: int):
        return slice(self.landmarks[lm_id], self.landmarks[lm_id]+2)

    # ---------- pose ----------
    def add_first_pose(self, p0: Vec2):
        self._expand(2)
        self.pose_cnt = 1
        s = self._pose_idx(0)
        self.Omega[s, s] += np.eye(2) * self.prior_info
        self.Xi[s] += np.array(p0).reshape(2,1) * self.prior_info

    def add_pose(self):
        self._expand(2)
        self.pose_cnt += 1
        return self.pose_cnt - 1

    def add_odometry(self, i: int, j: int, u: Vec2, var=0.05):
        W = np.eye(2) / var
        si, sj = self._pose_idx(i), self._pose_idx(j)

        self.Omega[si, si] += W
        self.Omega[sj, sj] += W
        self.Omega[si, sj] -= W
        self.Omega[sj, si] -= W

        u = np.array(u).reshape(2,1)
        self.Xi[si] -= W @ u
        self.Xi[sj] += W @ u

    # ---------- landmark ----------
    def add_landmark(self, lm_id: int):
        if lm_id in self.landmarks:
            return
        idx = self.Omega.shape[0]
        self.landmarks[lm_id] = idx
        self._expand(2)

    def observe_landmark(
        self,
        pose_i: int,
        lm_id: int,
        measured_range: float,
        var: float = 0.5
    ):
        """
        range-only observation
        """
        self.add_landmark(lm_id)

        si = self._pose_idx(pose_i)
        sl = self._lm_idx(lm_id)

        # current estimate (needed for linearization)
        mu = self.solve_vector()
        px, py = mu[si].flatten()
        lx, ly = mu[sl].flatten()

        dx = lx - px
        dy = ly - py
        dist = np.sqrt(dx*dx + dy*dy) + 1e-9

        # Jacobians
        H_p = np.array([[-dx/dist, -dy/dist]])
        H_l = np.array([[ dx/dist,  dy/dist]])

        W = np.array([[1.0/var]])

        # Omega updates
        self.Omega[si, si] += H_p.T @ W @ H_p
        self.Omega[sl, sl] += H_l.T @ W @ H_l
        self.Omega[si, sl] += H_p.T @ W @ H_l
        self.Omega[sl, si] += H_l.T @ W @ H_p

        # Xi update
        z = np.array([[measured_range - dist]])
        self.Xi[si] += H_p.T @ W @ z
        self.Xi[sl] += H_l.T @ W @ z

    # ---------- solve ----------
    def solve_vector(self):
        try:
            return np.linalg.solve(self.Omega, self.Xi)
        except np.linalg.LinAlgError:
            mu, *_ = np.linalg.lstsq(self.Omega, self.Xi, rcond=None)
            return mu

    def solve(self):
        mu = self.solve_vector()
        poses = []
        for i in range(self.pose_cnt):
            s = self._pose_idx(i)
            poses.append(tuple(mu[s].flatten()))
        lms = {}
        for k, idx in self.landmarks.items():
            lms[k] = tuple(mu[idx:idx+2].flatten())
        return poses, lms


# ============================================================
# 업그레이드: SE(2) pose (x,y,theta) + Landmark (x,y)
# range + bearing 관측 지원
# state:
#   [x0,y0,th0, x1,y1,th1, ..., l0x,l0y, l1x,l1y, ...]
# ============================================================

def _wrap_pi(a: float) -> float:
    """wrap angle to (-pi, pi]"""
    return (a + np.pi) % (2 * np.pi) - np.pi


class GraphSLAM_SE2:
    """
    Full Graph SLAM with SE(2) robot poses and 2D landmarks.
    - Pose node: (x, y, theta)
    - Landmark node: (lx, ly)

    Constraints:
      1) Odometry: p_j = p_i ⊕ u_ij  (we linearize using current estimate)
         Here u_ij is given in world-frame increments (dx, dy, dtheta) for simplicity.
      2) Landmark observation: range + bearing
         z = [r, b], with b = atan2(ly - y, lx - x) - theta
    """

    def __init__(self, prior_info: float = 1e6):
        self.prior_info = prior_info
        self.pose_cnt = 0
        self.landmarks: Dict[int, int] = {}   # lm_id -> state index (start of 2-dim block)
        self.Omega = np.zeros((0, 0), dtype=float)
        self.Xi = np.zeros((0, 1), dtype=float)

    # ----- indexing -----
    def _pose_idx(self, i: int) -> slice:
        base = 3 * i
        return slice(base, base + 3)

    def _lm_idx(self, lm_id: int) -> slice:
        base = self.landmarks[lm_id]
        return slice(base, base + 2)

    def _expand(self, n: int):
        old = self.Omega.shape[0]
        new = old + n
        O = np.zeros((new, new), dtype=float)
        X = np.zeros((new, 1), dtype=float)
        O[:old, :old] = self.Omega
        X[:old, :] = self.Xi
        self.Omega, self.Xi = O, X

    # ----- pose nodes -----
    def add_first_pose(self, p0_xyth: Tuple[float, float, float]):
        self._expand(3)
        self.pose_cnt = 1
        s0 = self._pose_idx(0)

        # strong prior to fix gauge
        self.Omega[s0, s0] += np.eye(3) * self.prior_info
        self.Xi[s0, :] += np.array(p0_xyth, dtype=float).reshape(3, 1) * self.prior_info

    def add_pose(self) -> int:
        self._expand(3)
        self.pose_cnt += 1
        return self.pose_cnt - 1

    # ----- landmark nodes -----
    def add_landmark(self, lm_id: int):
        if lm_id in self.landmarks:
            return
        idx = self.Omega.shape[0]
        self.landmarks[lm_id] = idx
        self._expand(2)

    # ----- constraints -----
    def add_odometry(self, i: int, j: int, u_world: Tuple[float, float, float], var_xy: float = 0.05, var_th: float = 0.02):
        """
        Odometry in world increments (dx, dy, dtheta).
        Constraint: x_j - x_i = dx, y_j - y_i = dy, th_j - th_i = dth (wrapped)
        This is a simple linear constraint (works well for small dt).
        """
        if not (0 <= i < self.pose_cnt and 0 <= j < self.pose_cnt):
            raise IndexError("pose index out of range")

        si, sj = self._pose_idx(i), self._pose_idx(j)

        W = np.diag([1.0/max(var_xy, 1e-12), 1.0/max(var_xy, 1e-12), 1.0/max(var_th, 1e-12)])
        u = np.array(u_world, dtype=float).reshape(3, 1)

        # Omega blocks (like pose graph)
        self.Omega[si, si] += W
        self.Omega[sj, sj] += W
        self.Omega[si, sj] -= W
        self.Omega[sj, si] -= W

        self.Xi[si, :] -= W @ u
        self.Xi[sj, :] += W @ u

    def observe_landmark_rb(
        self,
        pose_i: int,
        lm_id: int,
        measured_range: float,
        measured_bearing: float,
        var_r: float = 0.5,
        var_b: float = 0.05,
    ):
        """
        range+bearing observation:
          r = sqrt(dx^2 + dy^2)
          b = atan2(dy, dx) - theta
        We linearize around current estimate mu.
        """
        self.add_landmark(lm_id)

        si = self._pose_idx(pose_i)
        sl = self._lm_idx(lm_id)

        mu = self.solve_vector()
        px, py, pth = mu[si, :].flatten()
        lx, ly = mu[sl, :].flatten()

        dx = lx - px
        dy = ly - py
        q = dx*dx + dy*dy + 1e-12
        r = math.sqrt(q)
        b = math.atan2(dy, dx) - pth
        b = _wrap_pi(b)

        # measurement residual
        z = np.array([measured_range - r, _wrap_pi(measured_bearing - b)], dtype=float).reshape(2, 1)

        # Jacobian H = [H_p | H_l] for h(x)
        # h1=r, h2=b
        # dr/dx = -dx/r, dr/dy = -dy/r, dr/dth = 0
        # db/dx =  dy/q, db/dy = -dx/q, db/dth = -1
        H_p = np.array([
            [-dx/r, -dy/r, 0.0],
            [ dy/q, -dx/q, -1.0],
        ], dtype=float)

        # for landmark
        # dr/dlx = dx/r, dr/dly = dy/r
        # db/dlx = -dy/q, db/dly = dx/q
        H_l = np.array([
            [ dx/r,  dy/r],
            [-dy/q,  dx/q],
        ], dtype=float)

        W = np.diag([1.0/max(var_r, 1e-12), 1.0/max(var_b, 1e-12)])

        # Omega updates
        self.Omega[si, si] += H_p.T @ W @ H_p
        self.Omega[sl, sl] += H_l.T @ W @ H_l
        self.Omega[si, sl] += H_p.T @ W @ H_l
        self.Omega[sl, si] += H_l.T @ W @ H_p

        # Xi updates
        self.Xi[si, :] += H_p.T @ W @ z
        self.Xi[sl, :] += H_l.T @ W @ z

    # ----- solve -----
    def solve_vector(self):
        if self.Omega.shape[0] == 0:
            return np.zeros((0, 1))
        try:
            return np.linalg.solve(self.Omega, self.Xi)
        except np.linalg.LinAlgError:
            mu, *_ = np.linalg.lstsq(self.Omega, self.Xi, rcond=None)
            return mu

    def solve(self):
        mu = self.solve_vector()

        poses: List[Tuple[float, float, float]] = []
        for i in range(self.pose_cnt):
            s = self._pose_idx(i)
            x, y, th = mu[s, :].flatten()
            poses.append((float(x), float(y), float(th)))

        lms: Dict[int, Tuple[float, float]] = {}
        for lm_id, idx in self.landmarks.items():
            lx, ly = mu[idx:idx+2, :].flatten()
            lms[lm_id] = (float(lx), float(ly))

        return poses, lms
