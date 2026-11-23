# WM-PerceptHarness

[![tests](https://github.com/AxEBench/WM-PerceptHarness/actions/workflows/tests.yml/badge.svg)](https://github.com/AxEBench/WM-PerceptHarness/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)

Evaluate generated world-model videos by the **events** they show. WM-PerceptHarness
turns each video into a timeline of grounded events with a vision-language model
and targeted SAM3.1 checks, then compares generated and reference timelines with
temporal and semantic matching: **Timing**, **Content** and **Fidelity**. Paired
variants of the same task add **Stability** and the **AxE Score**.

## 📢 News

- **2026-11**: Code released.

## ✨ Overview

| Component | What it does | Command |
|---|---|---|
| Event annotation | VLM drafts events, an independent review raises concrete questions, SAM3.1 tracks the relevant object, and a verified revision produces `{duration, events: [{start, end, description}]}` | `percept annotate` |
| Event scoring | One-to-one temporal matching plus an LLM semantic judge; reports Timing, Content and Fidelity | `percept score events` |
| Generalization | paired Stability and AxE Score | `percept score stability` |
| Complementary metrics | PSNR, Dynamic Degree, Motion Smoothness, VideoPhy-2 Semantic Adherence and Physical Commonsense, GeCo structure error, CamMC camera motion error | `percept score <metric>` |

## 🚀 Quick Start

### Install

Requires **Python 3.12+** and **FFmpeg/FFprobe**.

```bash
git clone https://github.com/AxEBench/WM-PerceptHarness.git
cd WM-PerceptHarness
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install -e .
python tools/verify_install.py
```

Annotation also needs local SAM3.1 assets and a GPU
([setup](docs/annotation.md#targeted-sam31-grounding)). Model weights are never
downloaded automatically.

### Annotate a video

Any vision-capable OpenAI-compatible endpoint works; see [`.env.example`](.env.example)
for provider presets.

```bash
export PERCEPT_OPENAI_BASE_URL="https://your-provider.example/v1"
export PERCEPT_OPENAI_MODEL="your-vision-model-id"
export PERCEPT_OPENAI_API_KEY="your-api-key"

percept annotate --videos ./videos/sample_001.mp4 --output ./outputs/generated/sample_001
```

This writes `sample_001.json` (only `duration` and `events`, in seconds on the
original video clock), `sample_001.audit.json` (review and evidence) and
`results.jsonl`. Pass a directory to `--videos`, or use `--manifest` to spread a
corpus over several GPUs ([multi-GPU annotation](docs/annotation.md#multi-gpu-manifest-annotation)).

### Score against a reference

```bash
percept annotate --videos ./videos/real/sample_001.mp4 --output outputs/reference/sample_001

export PERCEPT_JUDGE_BASE_URL="https://your-judge.example/v1"
export PERCEPT_JUDGE_MODEL="your-judge-model-id"
export PERCEPT_JUDGE_API_KEY="your-judge-key"
percept score events --reference outputs/reference \
  --system model=outputs/generated --out outputs/event_scores
```

Human-written `{duration, events}` references are accepted as well. For WBench
multi-turn generations, build instruction references with
`tools/build_wbench_references.py` and add `--judge-mode instruction`
([judge modes](docs/annotation.md#judge-modes)).

### Complementary metrics

Each metric runs in its own environment; scores are reported side by side and
never combined.

| Metric | Command | Details |
|---|---|---|
| PSNR | `percept score psnr` | [third_party/psnr](third_party/psnr/README.md) |
| Dynamic Degree | `percept score dynamic-degree` | [third_party/vbench](third_party/vbench/README.md) |
| Motion Smoothness | `percept score motion-smoothness` | [third_party/vbench](third_party/vbench/README.md) |
| Semantic Adherence | `percept score semantic-adherence` | [third_party/videophy2](third_party/videophy2/README.md) |
| Physical Commonsense | `percept score physical-commonsense` | [third_party/videophy2](third_party/videophy2/README.md) |
| GeCo structure error | `percept score geco-depth` | [docs/metrics.md](docs/metrics.md#geco-depth-error) |
| CamMC (camera motion error) | `percept score cammc` | [third_party/cammc](third_party/cammc/README.md) |

## 📁 Repository Structure

```text
src/percept_harness/
├── cli.py            # `percept` command
├── annotation/       # annotation workflow, batching, caching, results
├── models/           # OpenAI-compatible VLM client
├── cv/               # SAM3.1 grounding
├── evaluation/       # event scores, Stability and AxE Score
└── metrics/          # launchers for the complementary metrics
third_party/          # pinned upstream evaluator code (VideoPhy-2, VBench, PSNR) and CamMC
tools/                # installation check and WBench reference builder
examples/             # example metric inputs
docs/                 # annotation, metrics and third-party documentation
tests/                # CPU-only tests: python -m pytest
```

## 📖 Documentation

- [Annotation and event scoring](docs/annotation.md): configuration, workflow, outputs and resume rules
- [Metrics](docs/metrics.md): complementary metrics, GeCo and CamMC protocols
- [Third-party code and licenses](docs/third_party.md)

## 📝 Citation

The paper citation will be added on publication. Until then:

```bibtex
@misc{wmperceptharness2026,
  title        = {WM-PerceptHarness: Grounded Event Evaluation for World-Model Videos},
  author       = {WM-PerceptHarness contributors},
  year         = {2026},
  howpublished = {\url{https://github.com/AxEBench/WM-PerceptHarness}}
}
```

## 🙏 Acknowledgement

We build on [SAM 3](https://github.com/facebookresearch/sam3),
[VBench](https://github.com/Vchitect/VBench),
[VideoPhy](https://github.com/Hritikbansal/videophy),
[IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch),
[GeCo](https://github.com/ShixuanGu/GeCo),
[CamI2V](https://github.com/ZGCTroy/CamI2V),
[WorldScore](https://github.com/haoyi-duan/WorldScore),
[DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) and
[WBench](https://github.com/meituan-longcat/WBench). We thank their authors.

## 📄 License

The code is released under the [MIT License](LICENSE). Third-party code in
`third_party/`, external tools and model weights keep their own licenses, some
of which are non-commercial; see [docs/third_party.md](docs/third_party.md).
