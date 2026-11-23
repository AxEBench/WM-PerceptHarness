# PSNR

Pairs each generated video with its original source video and computes a diagnostic RGB PSNR against the source as reference. The scoring kernel comes from a pinned IQA-PyTorch; the project adaptation handles temporal and native-geometry alignment. This protocol is not a human ground-truth evaluation and not a general perceptual-quality metric.

## Protocol and modifications

- Official PSNR (`psnr`): RGB, `crop_border=0`, `data_range=1`, `eps=1e-8`. Each frame is converted to dB first and then averaged within the video; identical images give 80 dB with this eps.
- Source and generated video hashes are pinned. FFmpeg extracts the first displayed source frame as an RGB PNG, which must match the SHA-256 of the real conditioning PNG.
- Both timelines subtract their first displayed PTS. The generated video must have a constant-frame-rate timeline: the manifest row's `fps`, or the stream's average frame rate if absent. For each generated timestamp the nearest source PTS is chosen, preferring the earlier frame on ties. A match is accepted only inside the source time range and within half a generated frame interval (1/48 s at 24 fps); there is no extrapolation, speed change or optical-flow alignment.
- Source frames are transformed to the generated video's native size according to the row's `geometry`, which works for any generator:
  - `cover` (default, any model): PIL LANCZOS aspect-preserving resize to cover the generated size, then floor center-crop.
  - `wan`: first checks the official `best_output_size` (stride 32, 1280×704 area preset), then the same LANCZOS cover and floor center-crop.
  - `lingbot`: official uint8 Torch bilinear resize and rounded center-crop to 832×480.
  - Older manifests with model `wan` / `lingbot-dense` and no `geometry` keep `wan` / `lingbot` respectively.
- The main score excludes generated frame 0; the first-frame score, unmatched frames and the per-frame mapping are saved separately. Model summaries should weight successful videos equally; cross-model comparisons must use the common set of successful samples. This entry point outputs per-video values and count status.

The entry point `modified/run_psnr.py` scores on CPU only and has no learned weights. Scoring and geometry functions are loaded unchanged; only the registry decorator is removed when loading the PSNR class, to avoid importing all of PyIQA. No generative model is run.

`official/` contains selected source files, not complete installable repositories. IQA-PyTorch is pinned to `18dd7a19694e94aac21019170e3f5e63d6b4e19e`, LingBot to `f348dca454c8f1e5707721614dbac10fa5290fed` and Wan to `42bf4cfaa384bc21833865abc2f9e6c0e67233dc`. Per-file sources and hashes are in `OFFICIAL_MANIFEST.json`.

## Run via `percept score`

```bash
percept score psnr --manifest /data/manifests/psnr.jsonl \
  --output-dir /data/results/psnr_source_reference_v1 --workers 3
```

`percept` locates `modified/run_psnr.py` in `third_party/` and keeps the same hashing, PTS alignment and output protocol. `--python` selects this package's CPU Torch environment, and `--dry-run` only prints the command. Running `python modified/run_psnr.py ...` directly still works for a standalone copy of this package.

## Environment and running

Linux, Python 3.12 and a CPU build of `torch==2.10.0`; other dependencies are in `requirements.txt`. FFmpeg and ffprobe must be on `PATH`. The conditioning-PNG hash is sensitive to FFmpeg version and decoding differences; the reference setup used FFmpeg 4.4.2. This is not a complete lockfile from a fresh-install check, and no generative models need to be downloaded.

With CPU Torch installed, from this package directory (for long remote runs, use a persistent session such as `tmux`):

```bash
python -m pip install -r requirements.txt
python modified/run_psnr.py --manifest /data/manifests/psnr.jsonl \
  --output-dir /data/results/psnr_source_reference_v1 --workers 3
```

[`examples/psnr_manifest.jsonl`](../../examples/psnr_manifest.jsonl) is only a format example; replace all paths, frame counts and all-zero hashes. Relative paths resolve against the manifest's directory. `model` may be any filename-safe identifier (letters, digits, `_`, `-`, `.`), and results are written to `results/<model>/`. Optional fields are `geometry` (`cover` / `wan` / `lingbot`) and `fps`. Any number of models with the same `sample_id` must share the same source video, source hash and real conditioning-PNG hash; a single model is also fine. Generated frame 0 must not be used in place of the original conditioning PNG.

Outputs: per-video scores in `results/`, PTS mappings and per-frame values in `evidence/`, and count status in `progress.json` and `batch_finished.json`. Failures are recorded separately and never replace scores. Use a new output directory whenever the manifest, protocol or code changes; resumed runs re-check the actual source and generated video hashes.

## Limitations and licenses

The `wan` / `lingbot` geometries reproduce those models' official input preprocessing; other models use `cover`, and if a model's official preprocessing differs (for example stretching instead of cropping), scores will be lower because of geometric misalignment. Plausible changes in action timing or trajectory can also lower PSNR, and differences in native resolution, frame rate and cropping limit cross-model comparison. The value does not replace physical plausibility or task success rates. If input identity, the first frame or geometry conditions do not hold, scoring stops for that row.

The Wan and LingBot source files are used only for the `wan` / `lingbot` geometries.

This release only checked the standalone source paths, the CLI help and scoring-function consistency; Torch scoring, re-scoring of real videos and fresh-environment installation were not re-run.

IQA-PyTorch's `LICENSE` (PolyForm Noncommercial) and `LICENSE-S-Lab` are kept unchanged, as are the PIQ/PIQA attributions in the PSNR file. The Apache-2.0 licenses of Wan and LingBot are kept in their directories. The project adaptation is not an official release and does not relicense third-party code.
