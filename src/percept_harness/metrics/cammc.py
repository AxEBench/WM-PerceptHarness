"""Score one generated video with CamMC against its conditioning camera trajectory.

Runs third_party/cammc/run_one.py in the CamMC environment (--python), with DROID-SLAM taken from
an external WorldScore checkout (--worldscore-root) and its droid.pth weights (--droid-weights).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from percept_harness.metrics import standalone

PACKAGE = "cammc"
RUNNER = "run_one.py"


def arguments(namespace: argparse.Namespace) -> list[str]:
    forwarded = ["--video", str(namespace.video), "--reference", str(namespace.reference)]
    if namespace.intrinsics is not None:
        forwarded += ["--intrinsics", *(repr(value) for value in namespace.intrinsics)]
    else:
        forwarded += ["--intrinsics-file", str(namespace.intrinsics_file)]
    for option in ("worldscore_root", "droid_weights", "video_start_s", "start_s", "end_s", "fps", "seed",
                   "gpu", "trajectory_output"):
        value = getattr(namespace, option)
        if value is not None:
            forwarded += ["--" + option.replace("_", "-"), str(value)]
    if namespace.any_revision:
        forwarded.append("--any-revision")
    return [*forwarded, "--output", str(namespace.output)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="percept score cammc", description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="generated video")
    parser.add_argument("--reference", type=Path, required=True,
                        help=".npz with cam_c2w (N x 4 x 4) and source_times_s (N) of the conditioning trajectory")
    intrinsics = parser.add_mutually_exclusive_group(required=True)
    intrinsics.add_argument("--intrinsics", type=float, nargs=4, metavar=("FX", "FY", "CX", "CY"),
                            help="conditioning intrinsics in pixels at the video's native resolution")
    intrinsics.add_argument("--intrinsics-file", type=Path, help=".npy or .json 3x3 K")
    parser.add_argument("--worldscore-root", type=Path,
                        help="WorldScore checkout holding DROID-SLAM (default $CAMMC_WORLDSCORE_ROOT)")
    parser.add_argument("--droid-weights", type=Path, help="droid.pth (default $CAMMC_DROID_WEIGHTS)")
    parser.add_argument("--any-revision", action="store_true", help="allow another WorldScore revision")
    parser.add_argument("--video-start-s", type=float, help="reference-clock time of the first video frame (default 0)")
    parser.add_argument("--start-s", type=float, help="window start on the reference clock")
    parser.add_argument("--end-s", type=float, help="window end on the reference clock")
    parser.add_argument("--fps", type=float, help="sampling grid rate (default 12)")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpu", help="physical GPU ID")
    parser.add_argument("--output", type=Path, required=True, help="new JSON file")
    parser.add_argument("--trajectory-output", type=Path, help="optional new .npz with both trajectories")
    standalone.add_arguments(parser)
    namespace = parser.parse_args(argv)
    try:
        return standalone.run(package=PACKAGE, script=RUNNER, arguments=arguments(namespace),
                              packages_root=namespace.third_party_root,
                              python=namespace.python, dry_run=namespace.dry_run)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"CamMC failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
