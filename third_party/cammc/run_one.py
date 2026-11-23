#!/usr/bin/env python3
"""Score one generated video with CamMC against the camera trajectory it was conditioned on.

The video is sampled on a 12 Hz grid over the common time window, DROID-SLAM (the copy pinned by
WorldScore) estimates its camera path with the conditioning intrinsics, and both paths are compared
with metric.mean_cammc. Writes one JSON file with per-frame and mean CamMC (unitless, lower is better).

The reference is an .npz with `cam_c2w` (N x 4 x 4 camera-to-world, OpenCV convention) and
`source_times_s` (N increasing seconds). `--video-start-s` places the first video frame on that clock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
FPS = 12.0
MIN_FRAMES = 16
SEED = 20261112
TRACKING_AREA = 512 ** 2
WORLDSCORE_REVISION = "096c75fc4ada9c7d92e03140c3c1f0c8f383b61f"
DROID_SLAM_REVISION = "8016d2b9b72b101a3e9ac804ebf20b4c654dc291"
DROID_SUBDIR = "worldscore/benchmark/metrics/third_party/droid_slam"
# WorldScore's DROID-SLAM settings; `buffer` and `image_size` are set per video.
DROID_SETTINGS = dict(t0=0, stride=1, beta=.3, filter_thresh=.01, warmup=8, keyframe_thresh=4.,
                      frontend_thresh=16., frontend_window=25, frontend_radius=2, frontend_nms=1,
                      backend_thresh=22., backend_radius=2, backend_nms=3, upsample=True, stereo=False)


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_video(path: Path):
    """Presentation timestamps of every decoded frame, native size and average frame rate."""
    import numpy as np
    stream = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate", "-of", "json", str(path)], timeout=300))["streams"][0]
    frames = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "frame=best_effort_timestamp_time", "-of", "json", str(path)], timeout=300))["frames"]
    pts = np.array([float(frame["best_effort_timestamp_time"]) for frame in frames])
    if len(pts) < 2 or not np.all(np.diff(pts) > 0):
        raise ValueError("Video needs at least two frames with increasing timestamps")
    numerator, denominator = map(float, stream["avg_frame_rate"].split("/"))
    if not numerator or not denominator:
        raise ValueError(f"Invalid average frame rate {stream['avg_frame_rate']}")
    return pts, int(stream["width"]), int(stream["height"]), numerator / denominator


def load_reference(path: Path):
    import numpy as np
    with np.load(path) as data:
        missing = {"cam_c2w", "source_times_s"} - set(data.files)
        if missing:
            raise ValueError(f"Reference {path} lacks {', '.join(sorted(missing))}")
        c2w, times = data["cam_c2w"].astype(float), data["source_times_s"].astype(float)
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4) or len(c2w) != len(times) or len(times) < 2:
        raise ValueError("Reference needs N x 4 x 4 cam_c2w and N source_times_s, N >= 2")
    if not np.isfinite(c2w).all() or not np.all(np.diff(times) > 0):
        raise ValueError("Reference poses must be finite with increasing times")
    return times, c2w


def load_intrinsics(values=None, path: Path | None = None):
    """3x3 K at the video's native resolution, from fx fy cx cy or a .npy/.json file (3x3, Nx3x3 or 4 values)."""
    import numpy as np
    if path is not None:
        data = np.load(path) if path.suffix == ".npy" else np.asarray(json.loads(path.read_text()), dtype=float)
        values = data[0] if data.ndim == 3 else data
    K = np.asarray(values, dtype=float)
    if K.shape == (4,):
        fx, fy, cx, cy = K
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 10 or K[1, 1] <= 10:
        raise ValueError("Intrinsics must be pixel-unit fx fy cx cy or a 3x3 K (not normalized)")
    return K


def plan_frames(pts, reference_times, video_fps, *, video_start_s=0.0, start_s=None, end_s=None, fps=FPS):
    """Choose the video frame nearest to each time of the fps grid over [start, end].

    Default window: the overlap of the video and the reference. A frame is used at most once and
    must lie within one video frame period of its grid time; there is no extrapolation.
    """
    import numpy as np
    pts = np.asarray(pts, dtype=float)
    pts = pts - pts[0] + video_start_s
    ref = np.asarray(reference_times, dtype=float)
    low, high = max(ref[0], pts[0]), min(ref[-1], pts[-1])
    start = low if start_s is None else float(start_s)
    end = high if end_s is None else float(end_s)
    if not (low - 1e-7 <= start < end <= high + 1e-7):
        raise ValueError(f"Window [{start}, {end}] s is outside the video/reference overlap [{low}, {high}] s")
    count = int(np.floor((end - start) * fps)) + 1
    if count < MIN_FRAMES:
        raise ValueError(f"Window gives {count} frames at {fps} Hz; at least {MIN_FRAMES} are needed")
    query = start + np.arange(count) / fps
    valid = np.where((pts >= ref[0] - 1e-7) & (pts <= ref[-1] + 1e-7))[0]
    if len(valid) < count:
        raise ValueError("Fewer usable video frames than grid times")
    index = np.array([valid[np.argmin(abs(pts[valid] - t))] for t in query])
    delta = float(np.max(abs(pts[index] - query)))
    if len(set(index.tolist())) != len(index):
        raise ValueError("Grid selects a video frame twice; the video frame rate is below the grid rate")
    if delta > 1 / video_fps + 1e-5:
        raise ValueError(f"Nearest frame is {delta:.4f} s from its grid time")
    return {"query": query, "index": index, "times": pts[index], "delta": delta, "start": start, "end": end}


def tracking_geometry(width: int, height: int, K):
    """Resize to about 512x512 pixels keeping the aspect ratio, crop to multiples of 8 at the top left."""
    factor = (TRACKING_AREA / (width * height)) ** .5
    resized_w, resized_h = int(width * factor), int(height * factor)
    out_w, out_h = resized_w - resized_w % 8, resized_h - resized_h % 8
    K = K.copy()
    K[0] *= resized_w / width
    K[1] *= resized_h / height
    return (resized_w, resized_h), (out_w, out_h), [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]


def decode_frames(video: Path, index, resized, cropped):
    import numpy as np
    selection = "+".join("eq(n\\,%d)" % i for i in index)
    command = ["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-filter_threads", "1", "-i", str(video),
               "-vf", f"select={selection},scale={resized[0]}:{resized[1]},crop={cropped[0]}:{cropped[1]}:0:0",
               "-vsync", "0", "-frames:v", str(len(index)), "-threads", "1", "-f", "rawvideo",
               "-pix_fmt", "bgr24", "pipe:1"]
    raw = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=600).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, cropped[1], cropped[0], 3)
    if len(frames) != len(index):
        raise ValueError(f"Decoded {len(frames)} of {len(index)} selected frames")
    return frames


def worldscore_revision(root: Path, expected: str | None):
    if not (root / ".git").exists():
        return None
    actual = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    if expected is not None and actual != expected:
        raise ValueError(f"WorldScore checkout is at {actual}, expected {expected} (pass --any-revision to allow)")
    return actual


def track(frames, intrinsics, *, worldscore_root: Path, weights: Path, seed: int):
    """Run DROID-SLAM over the whole sequence without resets; return camera-to-world poses."""
    import numpy as np
    sys.path.insert(0, str(worldscore_root / DROID_SUBDIR))
    import torch
    from lietorch import SE3
    from droid import Droid

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    settings = dict(DROID_SETTINGS, weights=str(weights),
                    buffer=max(256, int(math.ceil((len(frames) + 16) / 64) * 64)),
                    image_size=list(frames.shape[1:3]))

    def stream():
        for i, frame in enumerate(frames):
            yield i, torch.from_numpy(frame.copy()).permute(2, 0, 1)[None], torch.tensor(intrinsics, dtype=torch.float32)

    tracker = Droid(argparse.Namespace(**settings))
    for t, image, k in stream():
        tracker.track(t, image, intrinsics=k)
    if not tracker.frontend.is_initialized:
        raise RuntimeError("DROID-SLAM did not initialize")
    trajectory, residuals = tracker.terminate(stream())
    if residuals is None or residuals.numel() == 0 or not torch.isfinite(residuals).all():
        raise RuntimeError("Invalid DROID-SLAM backend residuals")
    c2w = SE3(torch.as_tensor(trajectory)).matrix().double().numpy()
    if len(c2w) != len(frames) or not np.isfinite(c2w).all():
        raise RuntimeError("DROID-SLAM returned an incomplete trajectory")
    rotation = c2w[:, :3, :3]
    if (np.max(abs(rotation @ rotation.transpose(0, 2, 1) - np.eye(3))) >= 1e-3
            or np.max(abs(np.linalg.det(rotation) - 1)) >= 1e-3):
        raise RuntimeError("DROID-SLAM returned invalid rotations")
    details = {"settings": {k: v for k, v in settings.items() if k != "weights"},
               "keyframes": int(tracker.video.counter.value),
               "backend_residual_mean": float(residuals.mean()),
               "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
    return c2w, details


def score(args) -> dict:
    import numpy as np
    sys.path.insert(0, str(HERE))
    from adapter import relative_poses, resample_poses
    from metric import per_frame_cammc

    started = time.time()
    root = Path(args.worldscore_root).expanduser().resolve(strict=True)
    weights = Path(args.droid_weights).expanduser().resolve(strict=True)
    if not (root / DROID_SUBDIR / "droid.py").is_file():
        raise FileNotFoundError(f"No DROID-SLAM at {root / DROID_SUBDIR}")
    revision = worldscore_revision(root, None if args.any_revision else WORLDSCORE_REVISION)
    video = args.video.expanduser().resolve(strict=True)
    reference_path = args.reference.expanduser().resolve(strict=True)

    reference_times, reference_c2w = load_reference(reference_path)
    K = load_intrinsics(args.intrinsics, args.intrinsics_file)
    pts, width, height, video_fps = probe_video(video)
    plan = plan_frames(pts, reference_times, video_fps, video_start_s=args.video_start_s,
                       start_s=args.start_s, end_s=args.end_s, fps=args.fps)
    resized, cropped, intrinsics = tracking_geometry(width, height, K)
    frames = decode_frames(video, plan["index"], resized, cropped)
    reference = relative_poses(resample_poses(reference_times, reference_c2w, plan["times"]))
    prepared = time.time()

    c2w, droid = track(frames, intrinsics, worldscore_root=root, weights=weights, seed=args.seed)
    estimate = relative_poses(c2w)
    values = per_frame_cammc(reference, estimate)
    if not np.isfinite(values).all():
        raise RuntimeError("Non-finite CamMC")
    radius = float(np.linalg.norm(reference[:, :3, 3], axis=1).max())
    cosine = (np.einsum("nij,nij->n", estimate[:, :3, :3], reference[:, :3, :3]) - 1) / 2
    if args.trajectory_output is not None:
        with open(args.trajectory_output, "xb") as stream:
            np.savez_compressed(stream, c2w=c2w, relative_c2w=estimate, reference_c2w=reference,
                                sample_times=plan["query"], selected_video_times=plan["times"],
                                selected_indices=plan["index"], intrinsics=intrinsics)
    return {
        "metric": "cammc", "status": "complete", "higher_is_better": False,
        "mean_cammc": float(values.mean()), "per_frame_cammc": values.tolist(), "frames": len(values),
        "window": {"start_s": plan["start"], "end_s": plan["end"], "fps": args.fps},
        "sample_times_s": plan["query"].tolist(), "selected_video_times_s": plan["times"].tolist(),
        "selected_frame_indices": plan["index"].tolist(), "max_frame_time_offset_s": plan["delta"],
        "video_fps": video_fps, "native_size": [width, height], "tracking_size": list(cropped),
        "intrinsics_native": [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])],
        "intrinsics_tracking": intrinsics,
        "reference_radius": radius, "numerically_static_reference": radius < 1e-9,
        "diagnostic_rotation_deg": float(np.degrees(np.arccos(np.clip(cosine, -1, 1))).mean()),
        "droid": droid, "seed": args.seed,
        "inputs": {"video": str(video), "video_sha256": sha256(video), "reference": str(reference_path),
                   "reference_sha256": sha256(reference_path), "droid_weights_sha256": sha256(weights)},
        "source": {"worldscore_root": str(root), "worldscore_revision": revision,
                   "expected_worldscore_revision": WORLDSCORE_REVISION, "droid_slam_revision": DROID_SLAM_REVISION,
                   "code_sha256": {name: sha256(HERE / name) for name in ("run_one.py", "metric.py", "adapter.py")}},
        "prepare_seconds": prepared - started, "tracking_seconds": time.time() - prepared,
    }


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True, help="generated video")
    parser.add_argument("--reference", type=Path, required=True,
                        help=".npz with cam_c2w (N x 4 x 4) and source_times_s (N) of the conditioning trajectory")
    intrinsics = parser.add_mutually_exclusive_group(required=True)
    intrinsics.add_argument("--intrinsics", type=float, nargs=4, metavar=("FX", "FY", "CX", "CY"),
                            help="conditioning intrinsics in pixels at the video's native resolution")
    intrinsics.add_argument("--intrinsics-file", type=Path, help=".npy or .json 3x3 K (or N x 3 x 3; the first is used)")
    parser.add_argument("--video-start-s", type=float, default=0.0,
                        help="reference-clock time of the first video frame (default 0)")
    parser.add_argument("--start-s", type=float, help="window start on the reference clock (default: overlap start)")
    parser.add_argument("--end-s", type=float, help="window end on the reference clock (default: overlap end)")
    parser.add_argument("--fps", type=float, default=FPS, help="sampling grid rate (default 12)")
    parser.add_argument("--worldscore-root", type=Path, default=os.environ.get("CAMMC_WORLDSCORE_ROOT"),
                        help="WorldScore checkout holding DROID-SLAM (default $CAMMC_WORLDSCORE_ROOT)")
    parser.add_argument("--droid-weights", type=Path, default=os.environ.get("CAMMC_DROID_WEIGHTS"),
                        help="droid.pth (default $CAMMC_DROID_WEIGHTS)")
    parser.add_argument("--any-revision", action="store_true",
                        help=f"allow a WorldScore git checkout other than {WORLDSCORE_REVISION[:7]}")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--gpu", help="physical GPU ID (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--output", type=Path, required=True, help="new JSON file")
    parser.add_argument("--trajectory-output", type=Path, help="optional new .npz with both trajectories")
    return parser


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.worldscore_root is None or args.droid_weights is None:
        print("CamMC failed: pass --worldscore-root and --droid-weights "
              "(or set CAMMC_WORLDSCORE_ROOT and CAMMC_DROID_WEIGHTS)", file=sys.stderr)
        return 2
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    try:
        result, code = score(args), 0
    except Exception as error:
        result, code = {"metric": "cammc", "status": "failed", "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc()}, 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if code:
        print(f"CamMC failed: {result['error']}; see {args.output}", file=sys.stderr)
    else:
        print(json.dumps({key: result[key] for key in ("metric", "mean_cammc", "frames")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
