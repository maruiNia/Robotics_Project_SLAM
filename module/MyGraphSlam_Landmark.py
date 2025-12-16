# module/MyGraphSlam_Landmark.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

Pose3 = Tuple[float, float, float]  # (x, y, theta)
Vec2 = Tuple[float, float]


def _wrap_pi(a: float) -> float:
    """wrap angle to [-pi, pi)"""
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class OdoFactor:
    i: int
    j: int
    dx: float
    dy: float
    dtheta: float
    W: np.ndarray  # (3,3) information


@dataclass
class RBFactor:
    i: int
    lm_id: int
    rng: float
    bearing: float
    W: np.ndarray  # (2,2) information


class GraphSLAM_SE2:
    """
    Full Graph SLAM (poses + landmarks) with SE2 pose and range+bearing landmark observations.
    - State: [x0,y0,th0, x1,y1,th1, ..., L0x,L0y, L1x,L1y, ...]
    - Solve: Gauss-Newton (batch), rebuild normal equation each solve()

    NOTE: This is intentionally simple & robust for your course project.
    """

    def __init__(self, prior_info: float = 1e6):
        self.prior_info = float(prior_info)

        self.poses: List[Pose3] = []
        self.landmarks: Dict[int, Vec2] = {}  # lm_id -> (x,y) initial/estimate

        self._odo: List[OdoFactor] = []
        self._rb: List[RBFactor] = []

    # ---------- init / add nodes ----------
    def add_first_pose(self, p0: Vec2, theta0: float = 0.0):
        if self.poses:
            raise RuntimeError("First pose already added.")
        self.poses.append((float(p0[0]), float(p0[1]), float(theta0)))

    def add_pose(self, initial: Optional[Pose3] = None) -> int:
        if not self.poses:
            raise RuntimeError("Call add_first_pose first.")
        if initial is None:
            # default guess: copy last pose
            initial = self.poses[-1]
        self.poses.append((float(initial[0]), float(initial[1]), float(initial[2])))
        return len(self.poses) - 1

    def add_landmark(self, lm_id: int, initial_xy: Vec2):
        if lm_id not in self.landmarks:
            self.landmarks[lm_id] = (float(initial_xy[0]), float(initial_xy[1]))

    # ---------- add factors ----------
    def add_odometry(self, i: int, j: int, dxyth: Pose3, cov: np.ndarray | None = None):
        dx, dy, dth = float(dxyth[0]), float(dxyth[1]), float(dxyth[2])
        if cov is None:
            cov = np.diag([0.05, 0.05, 0.02])
        cov = np.asarray(cov, dtype=float)
        W = np.linalg.inv(cov)
        self._odo.append(OdoFactor(i=i, j=j, dx=dx, dy=dy, dtheta=dth, W=W))

    def observe_landmark_rb(
        self,
        pose_i: int,
        lm_id: int,
        rng_bearing: Tuple[float, float],
        cov: np.ndarray | None = None,
        *,
        lm_init: Optional[Vec2] = None,
    ):
        rng, bearing = float(rng_bearing[0]), float(rng_bearing[1])
        if cov is None:
            cov = np.diag([0.5, 0.15])  # range variance, bearing variance
        cov = np.asarray(cov, dtype=float)
        W = np.linalg.inv(cov)

        # ensure landmark node exists
        if lm_id not in self.landmarks:
            if lm_init is not None:
                self.add_landmark(lm_id, lm_init)
            else:
                # crude init: place landmark in front of robot using measurement
                x, y, th = self.poses[pose_i]
                lx = x + rng * math.cos(th + bearing)
                ly = y + rng * math.sin(th + bearing)
                self.add_landmark(lm_id, (lx, ly))

        self._rb.append(RBFactor(i=pose_i, lm_id=lm_id, rng=rng, bearing=bearing, W=W))

    # ---------- solver helpers ----------
    def _pack_state(self) -> Tuple[np.ndarray, Dict[int, int]]:
        """return state vector and landmark index map"""
        N = len(self.poses)
        lm_ids = sorted(self.landmarks.keys())
        M = len(lm_ids)

        x = np.zeros((3 * N + 2 * M,), dtype=float)
        for i, (px, py, th) in enumerate(self.poses):
            x[3 * i + 0] = px
            x[3 * i + 1] = py
            x[3 * i + 2] = th

        lm_index: Dict[int, int] = {}
        base = 3 * N
        for k, lm_id in enumerate(lm_ids):
            lm_index[lm_id] = base + 2 * k
            lx, ly = self.landmarks[lm_id]
            x[lm_index[lm_id] + 0] = lx
            x[lm_index[lm_id] + 1] = ly

        return x, lm_index

    def _unpack_state(self, x: np.ndarray, lm_index: Dict[int, int]):
        N = len(self.poses)
        poses: List[Pose3] = []
        for i in range(N):
            poses.append((float(x[3 * i + 0]), float(x[3 * i + 1]), float(x[3 * i + 2])))

        lms: Dict[int, Vec2] = {}
        for lm_id, idx in lm_index.items():
            lms[lm_id] = (float(x[idx + 0]), float(x[idx + 1]))

        self.poses = poses
        self.landmarks = lms

    def _add_prior(self, H: np.ndarray, b: np.ndarray, x: np.ndarray):
        # Strong prior on first pose: fix gauge
        # residual r = (p0 - p0_current) ~ 0
        W = np.eye(3) * self.prior_info
        r = np.zeros((3, 1))
        # we want p0 to stay where it is initially; using current x as target is fine if add_first_pose is correct
        # so r = 0; still adds strong diagonal (stabilize)
        J = np.zeros((3, H.shape[0]))
        J[:, 0:3] = np.eye(3)

        # H += J^T W J ; b += J^T W r (r=0)
        H[:3, :3] += W

    # ---------- solve ----------
    def solve(self, iters: int = 5, damping: float = 1e-6) -> Tuple[List[Pose3], Dict[int, Vec2]]:
        """
        Gauss-Newton batch optimization.
        Returns: (poses, landmarks)
        """
        if not self.poses:
            return [], {}

        x, lm_index = self._pack_state()
        dim = x.shape[0]

        for _ in range(max(1, iters)):
            H = np.zeros((dim, dim), dtype=float)
            b = np.zeros((dim, 1), dtype=float)

            # prior on first pose
            self._add_prior(H, b, x)

            # ---- odometry factors ----
            for f in self._odo:
                i, j = f.i, f.j
                xi, yi, thi = x[3 * i + 0], x[3 * i + 1], x[3 * i + 2]
                xj, yj, thj = x[3 * j + 0], x[3 * j + 1], x[3 * j + 2]

                # model: p_j - p_i = (dx,dy), theta_j - theta_i = dtheta
                r = np.array([
                    [ (xj - xi) - f.dx ],
                    [ (yj - yi) - f.dy ],
                    [ _wrap_pi((thj - thi) - f.dtheta) ],
                ], dtype=float)

                # Jacobians wrt pose i and j (3x3 each)
                Ji = np.array([
                    [-1, 0, 0],
                    [ 0,-1, 0],
                    [ 0, 0,-1],
                ], dtype=float)
                Jj = np.array([
                    [ 1, 0, 0],
                    [ 0, 1, 0],
                    [ 0, 0, 1],
                ], dtype=float)

                W = f.W
                # assemble into H,b
                si = slice(3 * i, 3 * i + 3)
                sj = slice(3 * j, 3 * j + 3)

                H[si, si] += Ji.T @ W @ Ji
                H[si, sj] += Ji.T @ W @ Jj
                H[sj, si] += Jj.T @ W @ Ji
                H[sj, sj] += Jj.T @ W @ Jj

                b[si] += Ji.T @ W @ r
                b[sj] += Jj.T @ W @ r

            # ---- range+bearing landmark factors ----
            for f in self._rb:
                i = f.i
                lm_id = f.lm_id
                li = lm_index[lm_id]

                px, py, th = x[3 * i + 0], x[3 * i + 1], x[3 * i + 2]
                lx, ly = x[li + 0], x[li + 1]

                dx = lx - px
                dy = ly - py
                q = dx * dx + dy * dy
                dist = math.sqrt(q) + 1e-12

                # predicted measurements
                zhat_r = dist
                zhat_b = _wrap_pi(math.atan2(dy, dx) - th)

                # residual (measured - predicted)
                r = np.array([
                    [ f.rng - zhat_r ],
                    [ _wrap_pi(f.bearing - zhat_b) ],
                ], dtype=float)

                # Jacobians:
                # range:
                dr_dpx = -dx / dist
                dr_dpy = -dy / dist
                dr_dth = 0.0
                dr_dlx =  dx / dist
                dr_dly =  dy / dist

                # bearing:
                db_dpx =  dy / q
                db_dpy = -dx / q
                db_dth = -1.0
                db_dlx = -dy / q
                db_dly =  dx / q

                Jp = np.array([
                    [dr_dpx, dr_dpy, dr_dth],
                    [db_dpx, db_dpy, db_dth],
                ], dtype=float)  # (2x3)

                Jl = np.array([
                    [dr_dlx, dr_dly],
                    [db_dlx, db_dly],
                ], dtype=float)  # (2x2)

                W = f.W  # (2x2)

                sp = slice(3 * i, 3 * i + 3)
                sl = slice(li, li + 2)

                H[sp, sp] += Jp.T @ W @ Jp
                H[sp, sl] += Jp.T @ W @ Jl
                H[sl, sp] += Jl.T @ W @ Jp
                H[sl, sl] += Jl.T @ W @ Jl

                b[sp] += Jp.T @ W @ r
                b[sl] += Jl.T @ W @ r

            # damping (LM-lite)
            H += np.eye(dim) * damping

            # Solve: H * dx = b  (note: b is J^T W r ; typical GN uses -b, but we used residual as (z - zhat),
            # which makes step = solve(H, b) valid here.
            try:
                dx = np.linalg.solve(H, b).flatten()
            except np.linalg.LinAlgError:
                dx, *_ = np.linalg.lstsq(H, b, rcond=None)
                dx = dx.flatten()

            x = x + dx

            # wrap theta components
            for i in range(len(self.poses)):
                x[3 * i + 2] = _wrap_pi(x[3 * i + 2])

            # small step stop
            if float(np.linalg.norm(dx)) < 1e-6:
                break

        self._unpack_state(x, lm_index)
        return self.poses, self.landmarks
