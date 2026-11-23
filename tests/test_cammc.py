"""CamMC metric and frame-planning checks. Skipped where NumPy/SciPy are not installed; no GPU or DROID-SLAM."""
import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

PACKAGE = Path(__file__).resolve().parents[1] / "third_party" / "cammc"


def load(name):
    spec = importlib.util.spec_from_file_location(f"cammc_{name}", PACKAGE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


metric = load("metric")
mean_cammc = metric.mean_cammc


def poses():
    a = np.repeat(np.eye(4)[None], 3, axis=0)
    a[:, 0, 3] = [0, 1, 2]
    return a


def test_identical():
    assert mean_cammc(poses(), poses()) == pytest.approx(0)


def test_positive_scale_is_ignored():
    a = poses()
    b = a.copy()
    b[:, :3, 3] *= 4
    assert mean_cammc(a, b) == pytest.approx(0, abs=1e-8)


def test_reversed_direction():
    a = poses()
    b = a.copy()
    b[:, :3, 3] *= -1
    assert mean_cammc(a, b) == pytest.approx(1, abs=1e-8)


def test_rotation():
    a = poses()
    a[:, :3, 3] = 0
    b = a.copy()
    b[1:, :3, :3] = np.diag([-1, -1, 1])
    assert mean_cammc(a, b) == pytest.approx(2 * np.sqrt(2) * 2 / 3)


def test_static():
    a = poses()
    a[:, :3, 3] = 0
    assert mean_cammc(a, a) == 0


def test_invalid():
    a = poses()
    a[1, 0, 0] = np.nan
    with pytest.raises(ValueError):
        mean_cammc(a, poses())


def test_per_frame_values_and_first_camera_alignment():
    a = poses()
    b = a.copy()
    b[:, :3, 3] *= -1
    values = metric.per_frame_cammc(a, b)
    assert values == pytest.approx([0.0, 1.0, 2.0], abs=1e-8)
    # A rigid offset applied to every camera of one path is removed by the first-camera alignment.
    offset = np.eye(4)
    offset[:3, 3] = [5, -2, 7]
    assert mean_cammc(a, offset @ a) == pytest.approx(0, abs=1e-8)


def test_resample_and_relative_poses():
    pytest.importorskip("scipy")
    adapter = load("adapter")
    a = poses()
    out = adapter.resample_poses([0, 1, 2], a, [0.5, 1.5])
    assert out[:, 0, 3] == pytest.approx([0.5, 1.5])
    with pytest.raises(ValueError, match="extrapolation"):
        adapter.resample_poses([0, 1, 2], a, [2.5])
    shifted = a.copy()
    shifted[:, 1, 3] = 3
    assert adapter.relative_poses(shifted)[:, :3, 3] == pytest.approx(a[:, :3, 3])


def test_plan_frames_and_geometry():
    run_one = load("run_one")
    pts = np.arange(0, 3, 1 / 24)  # 3 s at 24 fps
    plan = run_one.plan_frames(pts, [0.5, 2.5], 24, video_start_s=0.0)
    assert plan["start"] == 0.5 and plan["end"] == 2.5 and len(plan["index"]) == 25
    assert plan["index"][:3].tolist() == [12, 14, 16] and plan["delta"] < 1e-9
    with pytest.raises(ValueError, match="outside the video/reference overlap"):
        run_one.plan_frames(pts, [0.5, 2.5], 24, start_s=0.0)
    with pytest.raises(ValueError, match="Fewer usable video frames"):
        run_one.plan_frames(np.arange(0, 3, 1 / 8), [0, 3], 8)
    K = run_one.load_intrinsics([800, 800, 640, 360])
    resized, cropped, intrinsics = run_one.tracking_geometry(1280, 720, K)
    assert resized == (682, 384) and cropped == (680, 384)
    assert intrinsics == pytest.approx([800 * 682 / 1280, 800 * 384 / 720, 640 * 682 / 1280, 360 * 384 / 720])
    with pytest.raises(ValueError, match="not normalized"):
        run_one.load_intrinsics([1, 1, .5, .5])
