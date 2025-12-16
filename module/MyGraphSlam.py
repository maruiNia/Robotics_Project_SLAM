# MyGraphSlam.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

Vec2 = Tuple[float, float]

@dataclass
class PoseGraphSLAM2D:
    """
    Pose-Graph SLAM (2D):
      state = [x0, y0, x1, y1, ..., xN, yN]^T

    Constraints:
      1) Odometry edge: p_j - p_i = u_ij
      2) Absolute position measurement: p_i = z_i  (from scan-grid localization, GPS, etc.)

    Solve:
      mu = inv(Omega) @ Xi
    """
    # Prior strength for first pose (bigger => more fixed)
    prior_info: float = 1e6

    def __post_init__(self):
        self._N: int = 0  # number of poses
        self.Omega = np.zeros((0, 0), dtype=float)
        self.Xi = np.zeros((0, 1), dtype=float)

    # -------- utilities --------
    def _idx(self, i: int) -> slice:
        """pose i -> slice for [xi, yi]"""
        return slice(2 * i, 2 * i + 2)

    def _expand_for_new_pose(self):
        """append a new pose to state"""
        new_dim = 2 * (self._N + 1)
        if self._N == 0:
            self.Omega = np.zeros((new_dim, new_dim), dtype=float)
            self.Xi = np.zeros((new_dim, 1), dtype=float)
        else:
            Omega2 = np.zeros((new_dim, new_dim), dtype=float)
            Xi2 = np.zeros((new_dim, 1), dtype=float)
            Omega2[: 2 * self._N, : 2 * self._N] = self.Omega
            Xi2[: 2 * self._N, :] = self.Xi
            self.Omega, self.Xi = Omega2, Xi2

        self._N += 1

        # prior for pose0 (fix the gauge / coordinate drift)
        if self._N == 1:
            s0 = self._idx(0)
            self.Omega[s0, s0] += np.eye(2) * self.prior_info

    # -------- public API --------
    def add_first_pose(self, p0: Vec2, meas_var: float = 1e-6):
        """
        Initialize with first pose measurement.
        meas_var small => strong constraint.
        """
        if self._N != 0:
            raise RuntimeError("First pose already exists.")
        self._expand_for_new_pose()
        self.add_abs_position(0, p0, meas_var=meas_var)

    def add_pose(self) -> int:
        """append a new pose node and return its index"""
        self._expand_for_new_pose()
        return self._N - 1

    def add_odometry(
        self,
        i: int,
        j: int,
        u_ij: Vec2,
        odo_var: float = 0.05,
    ):
        """
        Constraint: p_j - p_i = u_ij
        odo_var: variance (same for x,y) of odometry noise
        """
        if not (0 <= i < self._N and 0 <= j < self._N):
            raise IndexError("pose index out of range")

        W = np.eye(2) * (1.0 / max(odo_var, 1e-12))  # information = inv(cov)

        si = self._idx(i)
        sj = self._idx(j)

        # Omega blocks
        self.Omega[si, si] += W
        self.Omega[sj, sj] += W
        self.Omega[si, sj] -= W
        self.Omega[sj, si] -= W

        # Xi
        u = np.array(u_ij, dtype=float).reshape(2, 1)
        self.Xi[si, 0:1] -= (W @ u)
        self.Xi[sj, 0:1] += (W @ u)

    def add_abs_position(
        self,
        i: int,
        z_i: Vec2,
        meas_var: float = 0.25,
    ):
        """
        Absolute measurement: p_i = z_i
        meas_var: variance (same for x,y) of measurement noise
        """
        if not (0 <= i < self._N):
            raise IndexError("pose index out of range")

        W = np.eye(2) * (1.0 / max(meas_var, 1e-12))
        s = self._idx(i)

        self.Omega[s, s] += W
        z = np.array(z_i, dtype=float).reshape(2, 1)
        self.Xi[s, 0:1] += (W @ z)

    def solve(self) -> List[Vec2]:
        """
        Solve for all poses.
        Returns list of (x,y) for each pose.
        """
        if self._N == 0:
            return []

        # numerical stability: use solve instead of inv
        try:
            mu = np.linalg.solve(self.Omega, self.Xi)  # (2N,1)
        except np.linalg.LinAlgError:
            # fallback (least-squares) if singular (shouldn't happen w/ prior)
            mu, *_ = np.linalg.lstsq(self.Omega, self.Xi, rcond=None)

        poses: List[Vec2] = []
        for i in range(self._N):
            s = self._idx(i)
            poses.append((float(mu[s][0]), float(mu[s][1])))
        return poses

    def last_pose(self) -> Optional[Vec2]:
        if self._N == 0:
            return None
        return self.solve()[-1]
