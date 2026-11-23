"""Pose helpers for CamMC: first-camera alignment and resampling to frame times."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def relative_poses(c2w):
    """Re-express camera-to-world poses relative to the first camera: inv(T0) @ Ti."""
    c = np.asarray(c2w, dtype=float)
    if c.ndim != 3 or c.shape[1:] != (4, 4) or not np.isfinite(c).all():
        raise ValueError("Invalid poses")
    return np.linalg.inv(c[0])[None] @ c


def resample_poses(times, c2w, query):
    """Interpolate poses at `query` times: Slerp for rotations, linear for translations. No extrapolation."""
    t = np.asarray(times, dtype=float)
    q = np.asarray(query, dtype=float)
    c = np.asarray(c2w, dtype=float)
    if (len(t) < 2 or len(c) != len(t) or np.any(np.diff(t) <= 0)
            or np.any(q < t[0] - 1e-8) or np.any(q > t[-1] + 1e-8)):
        raise ValueError("Invalid time range or extrapolation")
    q = np.clip(q, t[0], t[-1])
    out = np.repeat(np.eye(4)[None], len(q), axis=0)
    out[:, :3, :3] = Slerp(t, Rotation.from_matrix(c[:, :3, :3]))(q).as_matrix()
    for k in range(3):
        out[:, k, 3] = np.interp(q, t, c[:, k, 3])
    return out
