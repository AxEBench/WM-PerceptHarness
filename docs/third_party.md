# Third-party code and licenses

The MIT License of this repository covers only its own code. It does not
relicense any component below, and several are limited to non-commercial use.
No model checkpoints, evaluation videos or private results are distributed here.

## Code included in `third_party/`

Each directory keeps the selected upstream files unchanged in `official/`, with
their license text, and lists every file with its SHA-256 in `MANIFEST.json`.
Project adaptations live in `modified/` (or the runner scripts) and are
described in each README.

| Directory | Upstream source | License |
| --- | --- | --- |
| [`videophy2`](../third_party/videophy2/README.md) | [VideoPhy](https://github.com/Hritikbansal/videophy) at `01d366c` | MIT (retained in `official/videophy/LICENSE`); checkpoint terms are separate |
| [`vbench`](../third_party/vbench/README.md) | Original `vbench==0.1.5` wheel from PyPI | Apache-2.0 (retained in the wheel metadata); RAFT (BSD-3-Clause) and AMT (CC BY-NC 4.0, non-commercial) weights carry their own terms |
| [`cammc`](../third_party/cammc/README.md) | Project code adapting the CamMC formula of [CamI2V](https://github.com/ZGCTroy/CamI2V) `evaluation/glomap_evaluation.py` | MIT (this repository; CamI2V is MIT) |
| [`psnr`](../third_party/psnr/README.md) | Selected [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch), [Wan2.2](https://github.com/Wan-Video/Wan2.2) and [LingBot-Video](https://github.com/Robbyant/lingbot-video) files | IQA-PyTorch: PolyForm Noncommercial 1.0.0 and NTU S-Lab license, both retained; Wan and LingBot: Apache-2.0, retained |

## Sources you obtain separately

| Component | Used by | License |
| --- | --- | --- |
| [SAM 3 / SAM 3.1](https://github.com/facebookresearch/sam3) source, checkpoint and tokenizer | `percept annotate` (targeted grounding) | [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE) (Meta). Publications of results obtained with SAM Materials must acknowledge their use, and certain end uses are restricted. |
| [GeCo](https://github.com/ShixuanGu/GeCo) with VGGT and UFM | `percept score geco-depth` | See the upstream repository and model cards |
| [WorldScore](https://github.com/haoyi-duan/WorldScore) at `096c75f` with its [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) submodule and `droid.pth` | `percept score cammc` | WorldScore: MIT; DROID-SLAM code and weights: BSD-3-Clause |
| [WBench](https://github.com/meituan-longcat/WBench) prompt builder and case data | `tools/build_wbench_references.py` | MIT (code and Hugging Face dataset `meituan-longcat/WBench`) |
| VideoPhy-2 checkpoint [`videophysics/videophy_2_auto`](https://huggingface.co/videophysics/videophy_2_auto) | Semantic Adherence, Physical Commonsense | See the model card |

The VLM and judge endpoints you configure are governed by your provider's terms.
