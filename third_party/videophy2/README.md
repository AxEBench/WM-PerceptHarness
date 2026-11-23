# VideoPhy-2: Semantic Adherence (SA) and Physical Commonsense (PC)

Both metrics use the same pinned VideoPhy-2 model and source; only the question differs.

- **Semantic Adherence (SA)** judges the overall semantic match between a video and its input caption.
- **Physical Commonsense (PC)** judges whether a video follows physical commonsense, using the fixed physical-commonsense template; it does not read captions.

Each produces the official 1–5 score (the official 0 is kept when the answer cannot be parsed).

## Protocol and modifications

- The official VideoPhy-2 model, prompt templates, generation parameters and numeric answer parsing are unchanged.
- Default frame sampling is `min(max(ceil(N/2),32),256)`. With `K=ceil(N/2)`, where `N` is the number of source frames: if `K<32`, the official 32-frame indices are used; if `32≤K≤256`, source frames `0,2,4,...` are taken; if `K>256`, 256 frames are sampled uniformly over the whole clip, including the first and last frames. `--sampling official` uses the official fixed 32 frames.
- The modified runner (`modified/sapc/run_videophy.py`) adds local weights, CSV input, and per-video raw answers, scores and sampled-frame records. The visual-token hook only records. `run_sa.py` and `run_pc.py` are thin entry points that fix the task.
- Example inputs are in [`examples/`](../../examples): `semantic_adherence.csv` (`videopath,caption`; captions must be frozen in advance and passed verbatim) and `physical_commonsense.csv` (`videopath` only). Both are synthetic placeholders.

## Sources and environment

Official code: [Hritikbansal/videophy](https://github.com/Hritikbansal/videophy/tree/01d366c979e4a82875852adbf427ccc036e7373a), commit `01d366c979e4a82875852adbf427ccc036e7373a`.
Weights: [videophysics/videophy_2_auto](https://huggingface.co/videophysics/videophy_2_auto/tree/ca38370e1dcc79c99c7017dd8755d0a087299afc), revision `ca38370e1dcc79c99c7017dd8755d0a087299afc`.

`official/` contains the selected upstream files and their license, not the complete repository. `official_sources.json` lists the source and hash of each file; `MANIFEST.json` lists every file in this directory. Known working versions: Python 3.10, Torch 2.10 / CUDA 12.8, Transformers 4.28.1, Accelerate 0.28.0, PEFT 0.3.0, NumPy 1.26.4, Decord 0.6.0 (see `environment/`). You need the complete pinned checkpoint and dependencies; the runner works offline and downloads nothing.

## Usage

From an environment where `percept` is installed, pointing `--python` at the VideoPhy-2 environment:

```bash
percept score semantic-adherence --python /path/to/videophy-env/bin/python \
  --input-csv /absolute/path/sa_input.csv \
  --checkpoint /absolute/path/videophy_2_auto \
  --output /absolute/path/sa_results

percept score physical-commonsense --python /path/to/videophy-env/bin/python \
  --input-csv /absolute/path/pc_input.csv \
  --checkpoint /absolute/path/videophy_2_auto \
  --output /absolute/path/pc_results
```

Add `--sampling official` for the official 32 frames, or `--dry-run` to print the command. The runners can also be called directly, e.g. `python modified/sapc/run_sa.py --help`. Outputs are `protocol.json`, `rows.json` and `complete.json` in a new directory. There is no resume; failures are recorded and stop the run.

## Limitations

- SA measures overall semantic match and cannot replace task success rates; caption accuracy affects the result. PC is a vision-language judgment and cannot replace physical simulation or measurement. Neither has been certified against human annotation.
- Sampling uses source frame indices, so frames are not equally spaced in time for variable-frame-rate videos; short clips may repeat frames. The visual abstractor still uses 64 latents plus 1 end token, and adding frames has no confirmed accuracy benefit.
- Numeric parsing keeps the official substring matching and returns 0 when nothing matches; the parse status is additionally recorded.
- Only loading with the complete `text_config` in the pinned checkpoint is supported; the upstream `text_config=None` branch references an unreleased module and is not patched.
