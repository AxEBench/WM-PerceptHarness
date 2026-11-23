# Third-party evaluators

Code for the paper's complementary metrics. Each directory keeps the selected
upstream files unchanged in `official/` with their licenses, adds the project
adaptations next to them, and lists every file with its SHA-256 in
`MANIFEST.json`. Runners check these hashes before scoring.

| Directory | Metrics | Command |
| --- | --- | --- |
| [`videophy2/`](videophy2/README.md) | Semantic Adherence, Physical Commonsense | `percept score semantic-adherence`, `percept score physical-commonsense` |
| [`vbench/`](vbench/README.md) | Dynamic Degree, Motion Smoothness | `percept score dynamic-degree`, `percept score motion-smoothness` |
| [`psnr/`](psnr/README.md) | PSNR against the reference recording | `percept score psnr` |
| [`cammc/`](cammc/README.md) | CamMC camera motion error (project code; DROID-SLAM from a separate WorldScore checkout) | `percept score cammc` |

`percept score <metric>` finds the runner here and starts it with the interpreter
given by `--python`, so each evaluator keeps its own environment. Use
`--dry-run` to print the exact command, or run the scripts in each directory
directly. VBench is imported from the bundled official wheel; no `pip install
vbench` or separate VBench clone is used.

No model weights, videos or results are included. Each README lists the weights
to prepare, the expected hashes and the known working environment. Licenses are
summarized in [`docs/third_party.md`](../docs/third_party.md); several components
are limited to non-commercial use.
