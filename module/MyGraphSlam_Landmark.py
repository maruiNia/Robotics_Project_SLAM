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
