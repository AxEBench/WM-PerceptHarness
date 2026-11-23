# Video metrics

Event scores do not measure visual quality, so the paper also reports the metrics
below. Each runs in its own environment and reports its own score; scores are
reported side by side and never combined into one number.

| Metric | Command | Direction | Implementation |
|---|---|---|---|
| PSNR (vs. reference recording) | `percept score psnr` | higher | [`third_party/psnr`](../third_party/psnr/README.md) |
| Dynamic Degree | `percept score dynamic-degree` | diagnostic | [`third_party/vbench`](../third_party/vbench/README.md) |
| Motion Smoothness | `percept score motion-smoothness` | higher | [`third_party/vbench`](../third_party/vbench/README.md) |
| Semantic Adherence (1–5) | `percept score semantic-adherence` | higher | [`third_party/videophy2`](../third_party/videophy2/README.md) |
| Physical Commonsense (1–5) | `percept score physical-commonsense` | higher | [`third_party/videophy2`](../third_party/videophy2/README.md) |
| GeCo structure error | `percept score geco-depth` | lower | [below](#geco-depth-error) |
| CamMC (camera motion error) | `percept score cammc` | lower | [below](#cammc-camera-motion-error), [`third_party/cammc`](../third_party/cammc/README.md) |

Use `percept score <metric> --help` for all options.

## Third-party evaluators

PSNR, Dynamic Degree, Motion Smoothness, Semantic Adherence and Physical
Commonsense run code kept under [`third_party/`](../third_party/README.md): the
selected upstream files, unchanged and hash-checked, plus documented adaptations.
VBench is loaded from the official wheel bundled in the repository, so neither
`pip install vbench` nor a separate VBench clone is needed.

`percept score <metric>` locates the runner under `third_party/` and starts it
with `--python` (default: the current interpreter), so each evaluator keeps its
own environment; VideoPhy-2, for example, uses Python 3.10. Pass
`--third-party-root` (or set `PERCEPT_THIRD_PARTY_ROOT`) if the checkout is not
next to the installed package, and `--dry-run` to print the exact command.
Sampling, model loading and scoring stay in the runner. Example inputs are in
[`examples/`](../examples). Model weights are never downloaded automatically;
each package README lists the expected files and hashes.

## GeCo Depth Error

For a generated RGB video, this command reports the official GeCo-Eval CSV
column `depth` (lower is better, 0 is ideal, no upper bound). It is the
structure term of the GeCo paper: the depth of one frame, reprojected into
another frame with the estimated camera motion, compared with that frame's own
depth, relative to the source depth. VGGT-1B estimates depth and cameras.
UFM-Base only provides the covisibility mask that decides which pixels count.
No prompt, known camera trajectory, or depth ground truth is required. The
paper's headline number is the `fused` column, so these scores are not
comparable with the paper's tables.

Install FFmpeg and ffprobe on the harness PATH. In a separate environment follow
the [upstream installation and weight instructions](https://github.com/ShixuanGu/GeCo/blob/b66d1c68647ba75b5e09d05755cedac79a435bd5/README.md)
for VGGT and UFM. Upstream documents Linux, Python 3.11 and PyTorch 2.8/cu128.
The upstream runtime may download weights on first use.

```bash
git clone https://github.com/ShixuanGu/GeCo.git /path/to/GeCo
git -C /path/to/GeCo checkout b66d1c68647ba75b5e09d05755cedac79a435bd5
percept score geco-depth --geco-root /path/to/GeCo \
  --geco-python /path/to/geco-env/bin/python --videos-path clip.mp4 \
  --output-dir results/geco/clip
```

**Windows.** Upstream scores 3 s windows sampled at min(8, native) fps and pairs
every frame of a window with every other. It allows at most four windows and, on
clips over 12 s, stretches each window to a quarter of the clip, which is
hundreds of frames per window on long videos. The default `--windows two`
keeps windows at 3 s. Clips up to 6 s use the upstream windows unchanged and
score identically to upstream. Longer clips use two windows that start at 1/3
and 2/3 of (duration − 3 s). The wrapper decodes only those frames from the
first video stream at its reported average frame rate (variable-rate clips are
resampled onto this grid), hands each window to the unmodified pinned evaluator
as one clip, and combines the windows as upstream does: a frame-weighted mean
over windows whose three official scores are finite. `--windows official`
passes every frame and lets upstream choose.

On machines without Hugging Face or GitHub access, pass a local UFM-Base
directory (`config.json` and `model.safetensors`) with `--ufm-checkpoint`, keep
VGGT-1B in `HF_HOME` and DINOv2 ViT-L in `TORCH_HOME`, and set
`HF_HUB_OFFLINE=1`. UFM builds its encoder through `torch.hub`, whose GitHub
lookup has no timeout.

A successful run writes `wrapper_metadata.json` with the score and the start
time, frame count and score of each window, alongside the original CSV and
logs. A clip with no valid window fails the command. Keep these failures in the
coverage denominator when comparing models. Use a new output directory for each
run. Each call loads both models; for a large batch, place the windows of many
videos under one frames root so that one evaluator process scores them.

**Reading the score.** A pilot on 108 tasks, 17 models and the matching real
videos (1080 clips) found the following.

- Legitimate motion counts as error. With a static camera the background is near
  zero and the error sits on moving hands, arms and liquids. A real video whose
  first frame was repeated scored 2.5% of the original, and generated videos
  that Dynamic Degree flags as static had about a quarter of the error of
  moving ones. Report the score next to Dynamic Degree, or as a ratio to the
  real video of the same task; a ratio below 1 means less motion than the real
  video, not better geometry.
- Content matters more than the model: tasks explained 37–50% of the variance
  and models 5–14%. Real videos ranged from 0.004 (simulation) to 0.07–0.10
  (head-mounted camera).
- The distribution is heavy-tailed (the top 5% of clips carried about 30% of the
  summed error). Aggregate with a geometric mean or median.
- Input height matters below 480: the same real clips scored 23% lower at 256
  than at 480, while 720 matched 480. Sampling at 6 fps matched 8 fps.
- Extreme values mark visible failures such as a scene cut or a frame filled by
  a textureless surface, where the geometry estimate itself is unstable.

## CamMC (camera motion error)

CamMC compares the camera path of a generated video with the camera trajectory
the model was conditioned on. It is unitless, lower is better, and 0 is a
perfect match.

**Definition.** The video is sampled at 12 Hz over the common time window (the
nearest decoded frame to each grid time, at most one frame period away).
DROID-SLAM, the copy pinned by WorldScore, tracks these frames monocularly over
the whole sequence with the conditioning intrinsics (rescaled to the tracking
resolution of about 512×512 pixels) and WorldScore's settings. The conditioning
trajectory is interpolated to the same frame times. Both camera-to-world paths
are expressed relative to their first camera, and each path's translations are
divided by its own maximum distance from the start. No scale, rotation or time
alignment is fitted. Per frame, the score is the Frobenius norm of the
difference of the two 3×4 `[R|t]` matrices; CamMC is the mean over frames.

The formula follows CamI2V's relative CamMC (`evaluation/glomap_evaluation.py`),
averaged over frames instead of summed, with DROID-SLAM instead of GLOMAP. As a
control, real recordings scored against their own reference trajectories
average about 0.18 in the paper's runs, the error floor of the tracker and of
the reference estimate. A static reference has no translation scale, so the
paper reports CamMC on moving-camera recordings only; compare models on the
same window and the same set of successfully scored videos.

**Dependencies.** Linux, a CUDA GPU, Python 3.12 with a CUDA build of PyTorch,
`torch-scatter`, NumPy, SciPy and OpenCV, FFmpeg/ffprobe on `PATH`, a WorldScore
checkout at `096c75fc4ada9c7d92e03140c3c1f0c8f383b61f` with its DROID-SLAM
submodule (`8016d2b`) built, and the `droid.pth` weights. The paper's runs used
torch 2.10.0+cu128.

**Install.**

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

**Run.** The reference is an `.npz` with `cam_c2w` (N×4×4 camera-to-world,
OpenCV convention) and `source_times_s`; the intrinsics are in pixels at the
video's native resolution.

```bash
percept score cammc --python /path/to/cammc-env/bin/python \
  --worldscore-root /path/to/WorldScore --droid-weights /path/to/droid.pth \
  --video generated.mp4 --reference reference.npz \
  --intrinsics 912.3 912.3 640 360 --output results/cammc/generated.json
```

The JSON holds `mean_cammc`, `per_frame_cammc`, the selected frame times and
the hashes of all inputs. Pass `--start-s`/`--end-s` to score all models of a
task on their common window. See [`third_party/cammc`](../third_party/cammc/README.md)
for every option and the attribution.
