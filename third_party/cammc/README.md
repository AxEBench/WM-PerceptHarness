# CamMC: camera motion error

CamMC measures how far the camera path of a generated video is from the camera
trajectory the model was conditioned on. It is unitless; lower is better and 0
is a perfect match. Run it with `percept score cammc` (or `python run_one.py`
from this directory).

## Definition

1. **Frames.** The video is sampled on a 12 Hz grid over the common time window
   (by default the overlap of the video and the reference trajectory; pass
   `--start-s`/`--end-s` to score every model of a task on the same window).
   Each grid time takes the nearest decoded frame, which must be within one
   video frame period and is used at most once.
2. **Estimated path.** DROID-SLAM, the copy pinned by WorldScore, tracks the
   whole sequence monocularly without resets, with WorldScore's settings
   (warmup 8, keyframe threshold 4, frontend window 25, backend threshold 22,
   upsampling), a buffer sized to the frame count and seed 20261112. Frames are
   resized to about 512×512 pixels keeping the aspect ratio and cropped to a
   multiple of 8 at the top left. The intrinsics are the ones the model was
   conditioned with, rescaled and cropped the same way.
3. **Reference path.** The conditioning trajectory, interpolated to the
   timestamps of the selected frames (Slerp for rotations, linear for
   translations, no extrapolation).
4. **Alignment.** Both camera-to-world paths are expressed relative to their
   first camera, `inv(T0) @ Ti`, and each path's translations are divided by
   that path's maximum distance from the start (+1e-9). No scale, rotation or
   time alignment is fitted.
5. **Score.** Per frame, the Frobenius norm of the difference of the two 3×4
   `[R|t]` matrices; **CamMC is the mean over frames**.

As a control, the real recordings scored against their own reference
trajectories average about 0.18 in the paper's runs: the error floor of
DROID-SLAM and of the reference estimate itself. A static reference has no defined
translation scale (`numerically_static_reference` in the output), so the paper
reports CamMC on moving-camera recordings only.

## Inputs and output

- `--video`: the generated video.
- `--reference`: an `.npz` with `cam_c2w` (N×4×4 camera-to-world, OpenCV
  convention: x right, y down, z forward) and `source_times_s` (N increasing
  seconds). `--video-start-s` gives the time on that clock at which the first
  video frame is shown (default 0).
- `--intrinsics FX FY CX CY` in pixels at the video's native resolution, or
  `--intrinsics-file` with a 3×3 `K` (`.npy` or `.json`).
- `--output`: a new JSON file with `mean_cammc`, `per_frame_cammc`, the sample
  and selected frame times, the intrinsics used, a diagnostic mean rotation
  angle (not part of the score), DROID-SLAM keyframes and residuals, and the
  SHA-256 of the video, reference, weights and code. A failed run writes
  `status: failed` with the error and exits non-zero; count failures in the
  coverage of a model comparison. `--trajectory-output` also saves both paths.

## Environment

Linux with a CUDA GPU, Python 3.12, a CUDA build of PyTorch, `torch-scatter`,
NumPy, SciPy, OpenCV, and FFmpeg/ffprobe on `PATH`. The paper's runs used
torch 2.10.0+cu128 on RTX 5090 GPUs. DROID-SLAM comes from a WorldScore
checkout at `096c75fc4ada9c7d92e03140c3c1f0c8f383b61f`, whose
`thirdparty/DROID-SLAM` submodule (`8016d2b9b72b101a3e9ac804ebf20b4c654dc291`)
builds the `droid_backends` and `lietorch` extensions:

```bash
git clone https://github.com/haoyi-duan/WorldScore.git /path/to/WorldScore
cd /path/to/WorldScore
git checkout 096c75fc4ada9c7d92e03140c3c1f0c8f383b61f
git submodule update --init --recursive thirdparty/DROID-SLAM
# in the CamMC environment, with PyTorch already installed:
pip install torch-scatter -f https://data.pyg.org/whl/torch-<torch version>+<cuda tag>.html
pip install numpy scipy opencv-python gdown
(cd thirdparty/DROID-SLAM && python setup.py install)
gdown 1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh   # droid.pth
```

`run_one.py` imports DROID-SLAM from
`worldscore/benchmark/metrics/third_party/droid_slam` inside that checkout and
refuses another WorldScore revision unless `--any-revision` is passed.

## Usage

```bash
percept score cammc --python /path/to/cammc-env/bin/python \
  --worldscore-root /path/to/WorldScore --droid-weights /path/to/droid.pth \
  --video generated.mp4 --reference reference.npz \
  --intrinsics 912.3 912.3 640 360 --output results/cammc/generated.json
```

`--worldscore-root` and `--droid-weights` default to `$CAMMC_WORLDSCORE_ROOT`
and `$CAMMC_DROID_WEIGHTS`. Each run loads DROID-SLAM once and scores one video.

## Files

| File | Content |
| --- | --- |
| `metric.py` | `per_frame_cammc`, `mean_cammc` |
| `adapter.py` | first-camera alignment and pose resampling |
| `run_one.py` | frame selection, DROID-SLAM tracking and output for one video |

## Attribution and licenses

These files are project code under the repository's MIT License. They adapt
or call the following works, which keep their own licenses:

- **CamI2V** ([ZGCTroy/CamI2V](https://github.com/ZGCTroy/CamI2V), MIT):
  the CamMC formula follows `calc_cammc` and the relative variant of `metric`
  in `evaluation/glomap_evaluation.py` (at `2d6644c`). Differences: the
  per-frame values are averaged instead of summed, and the camera path is
  estimated with DROID-SLAM instead of GLOMAP.
- **WorldScore** ([haoyi-duan/WorldScore](https://github.com/haoyi-duan/WorldScore), MIT):
  the pinned DROID-SLAM copy and its tracking settings. Not included; obtained
  separately as above.
- **DROID-SLAM** ([princeton-vl/DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM), BSD-3-Clause):
  the tracker, its CUDA extensions and the `droid.pth` weights. Not included.
