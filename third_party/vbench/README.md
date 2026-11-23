# VBench 0.1.5: Dynamic Degree and Motion Smoothness

- **Dynamic Degree** measures whether a video has motion above the official optical-flow threshold (RAFT). A single video scores `0.0` or `1.0`. Camera motion also produces optical flow, so the value is not action correctness or quality.
- **Motion Smoothness** interpolates between adjacent kept frames with AMT-S and compares the result with the original intermediate frames. Scores are in `[0,1]`; higher means lower interpolation error. It does not judge action semantics or physical correctness.

## Official baseline and modifications

The baseline is the original PyPI `vbench==0.1.5` wheel, shipped in `official/distributions/` (SHA256 `13266ed4715d91a7669ab58484b9e010bc91bb96e129d414fb512d4937890023`, [release file](https://files.pythonhosted.org/packages/06/82/13e401ec97d2686397201dfe74e2dda88b3695192de1b25a5ab267ab8960/vbench-0.1.5-py3-none-any.whl)). `official/pypi/` holds files extracted unchanged from it (`dynamic_degree.py`, `motion_smoothness.py`, `AMT-S.yaml`, license and metadata).

`modified/vbench/` keeps decoded frame tensors on CPU and sends only the current frame pair to the GPU, reducing GPU memory for long videos. Frame sampling, models, thresholds and score formulas are unchanged. Each runner requires an explicit `--mode official` (unmodified) or `--mode cpu-staging`; both use `torch.device('cuda')` like the official CLI.

## Environment

Linux, one CUDA GPU, Python 3.12 and a compatible PyTorch/TorchVision (known working: torch 2.13.0, torchvision 0.28.0, numpy 2.5.2, Pillow 11.3.0). Install the shipped wheel without letting pip change your GPU environment:

```bash
python -m pip install --no-deps official/distributions/vbench-0.1.5-py3-none-any.whl
```

Weights are not included and are verified by hash before use:

| Metric | Path under `--cache-dir` | SHA256 | Source |
|---|---|---|---|
| Dynamic Degree | `raft_model/models/raft-things.pth` | `fcfa4125…a9a7e1` | [RAFT models.zip](https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip) |
| Motion Smoothness | `amt_model/amt-s.pth` | `07e7e034…d3cf77` | [AMT-S](https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth) |

## Usage

```bash
percept score dynamic-degree --mode cpu-staging \
  --video /data/example.mp4 --cache-dir /data/vbench-cache \
  --gpu 0 --output /data/results/dynamic_degree.json

percept score motion-smoothness --mode cpu-staging \
  --video /data/example.mp4 --cache-dir /data/vbench-cache \
  --gpu 0 --output /data/results/motion_smoothness.json
```

The runners can also be called directly from this directory (`python run_dynamic_degree.py --help`). Only MP4 is accepted and the output file must not exist. Each run checks the installed version, source, config and weight hashes, and records the video, code, weights and runtime versions.

## Limitations

CPU staging does not reduce the memory the model itself needs and is not streaming decoding; long videos still use a lot of CPU memory. For Motion Smoothness, padding also runs on CPU, so bilinear resizing of frames with different sizes may differ numerically from GPU. Static scenes need not score as dynamic, optical flow does not separate object from camera motion, and a smooth but incorrect action can still score highly. Bit-identical results across environments are not guaranteed.
