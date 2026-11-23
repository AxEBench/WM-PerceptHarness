"""Paired Stability and AxE Score from an event score report.

Each pair links a variant sample to its original sample in the same event report. With f(x) the
sample's Fidelity (an empty sample, whose Fidelity is undefined, counts as 0):

    Stability = 1 - sum_k |f(v_k) - f(o_k)| / sum_k (f(v_k) + f(o_k))
    AxE Score = Fidelity x Stability

where Fidelity is the system's micro (pooled) Fidelity from the event report.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROTOCOL = {
    "name": "AxE paired Stability and AxE Score",
    "version": 1,
    "stability": "1 - sum|f(variant) - f(original)| / sum(f(variant) + f(original)); f = per-sample Fidelity, undefined = 0",
    "axe_score": "micro Fidelity x Stability",
}


def read_pairs(path: Path) -> list[tuple[str, str]]:
    """Read (variant, original) sample ID pairs from a CSV with `variant` and `original` columns."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = {"variant", "original"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"pairs CSV {path} lacks column(s): {', '.join(sorted(missing))}")
        pairs = []
        for line, row in enumerate(reader, start=2):
            variant, original = ((row.get(key) or "").strip() for key in ("variant", "original"))
            if not variant or not original:
                raise ValueError(f"pairs CSV {path} line {line}: blank variant or original ID")
            pairs.append((variant, original))
    if not pairs:
        raise ValueError(f"pairs CSV {path} has no pairs")
    return pairs


def _fidelity(samples: dict, sample_id: str) -> float:
    if sample_id not in samples:
        raise ValueError(f"sample {sample_id!r} is not in the event report")
    sample = samples[sample_id]
    if sample.get("status") != "valid":
        raise ValueError(f"sample {sample_id!r} is not valid: {sample.get('error', sample.get('status'))}")
    value = sample.get("fidelity")
    return 0.0 if value is None else float(value)


def stability(samples: dict, pairs: list[tuple[str, str]]) -> dict:
    difference = total = 0.0
    for variant, original in pairs:
        fv, fo = _fidelity(samples, variant), _fidelity(samples, original)
        difference += abs(fv - fo)
        total += fv + fo
    return {"pairs": len(pairs), "stability": 1 - difference / total if total else None}


def score_report(report: dict, pairs: list[tuple[str, str]]) -> dict:
    systems = {}
    for name, system in report["systems"].items():
        if system.get("aggregate") is None:
            raise ValueError(f"system {name!r} has no aggregate scores (invalid samples in the event report)")
        fidelity = system["aggregate"]["fidelity"]["micro"]["f1"]
        result = stability(system["samples"], pairs)
        result["fidelity"] = fidelity
        result["axe_score"] = None if fidelity is None or result["stability"] is None else fidelity * result["stability"]
        systems[name] = result
    return systems


def _format(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events", required=True, type=Path, help="events.json written by `percept score events`")
    parser.add_argument("--pairs", required=True, type=Path, help="CSV with variant,original sample ID columns")
    parser.add_argument("--out", required=True, type=Path, help="report output directory")
    args = parser.parse_args(argv)
    try:
        report = json.loads(args.events.read_text(encoding="utf-8"))
        systems = score_report(report, read_pairs(args.pairs))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"stability failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "stability.json"
    result = {"protocol": PROTOCOL, "events": str(args.events), "pairs": str(args.pairs), "systems": systems}
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    for name, scores in systems.items():
        print(f"{name}: Fidelity={_format(scores['fidelity'])} Stability={_format(scores['stability'])} "
              f"AxE Score={_format(scores['axe_score'])} ({scores['pairs']} pairs)")
    print(f"stability report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
