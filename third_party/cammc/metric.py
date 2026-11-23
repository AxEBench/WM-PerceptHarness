"""CamMC between two camera trajectories.

Adapted from `calc_cammc` / `metric` (relative variant) in CamI2V `evaluation/glomap_evaluation.py`
(https://github.com/ZGCTroy/CamI2V, MIT License). Changes: per-frame values are averaged instead of
summed, and each trajectory is normalized by its own maximum distance from the first camera.
"""
import numpy as np

EPSILON = 1e-9


def _normalized(c2w):
    """Express camera-to-world poses relative to the first camera and divide translations by their max norm."""
    relative = np.linalg.inv(c2w[0]) @ c2w
    scale = np.linalg.norm(relative[:, :3, 3], axis=1).max() + EPSILON
    relative[:, :3, 3] /= scale
    return relative[:, :3, :4]


def per_frame_cammc(reference, estimate):
    """Frobenius norm of the 3x4 [R|t] difference per frame, after first-camera alignment and
    independent positive scale normalization. No scale or rotation is fitted."""
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(estimate, dtype=np.float64)
    if (a.shape != b.shape or a.ndim != 3 or a.shape[1:] != (4, 4) or len(a) < 2
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("Expected matched finite Nx4x4 trajectories, N>=2")
    difference = _normalized(a) - _normalized(b)
    return np.linalg.norm(difference.reshape(len(a), 12), axis=1)


def mean_cammc(reference, estimate):
    """Mean CamMC over frames (dimensionless, lower is better)."""
    return float(per_frame_cammc(reference, estimate).mean())
