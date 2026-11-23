"""Dispatch and argument forwarding for the standalone ``percept score`` commands.

These tests never load a metric model: they check command registration, help,
argument forwarding and exit-code propagation only.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from percept_harness import cli
from percept_harness.metrics import (
    cammc,
    dynamic_degree,
    motion_smoothness,
    physical_commonsense,
    psnr,
    semantic_adherence,
    standalone,
)

MODULES = {
    "semantic-adherence": semantic_adherence,
    "physical-commonsense": physical_commonsense,
    "psnr": psnr,
    "dynamic-degree": dynamic_degree,
    "motion-smoothness": motion_smoothness,
    "cammc": cammc,
}

REQUIRED = {
    "semantic-adherence": ["--input-csv", "in.csv", "--checkpoint", "ckpt", "--output", "out"],
    "physical-commonsense": ["--input-csv", "in.csv", "--checkpoint", "ckpt", "--output", "out"],
    "psnr": ["--manifest", "manifest.jsonl", "--output-dir", "out"],
    "dynamic-degree": ["--mode", "official", "--video", "v.mp4", "--cache-dir", "cache",
                       "--gpu", "0", "--output", "out.json"],
    "motion-smoothness": ["--mode", "cpu-staging", "--video", "v.mp4", "--cache-dir", "cache",
                          "--gpu", "0", "--output", "out.json"],
    "cammc": ["--video", "v.mp4", "--reference", "ref.npz", "--intrinsics", "500", "500", "320", "240",
              "--worldscore-root", "WorldScore", "--droid-weights", "droid.pth", "--output", "out.json"],
}

RUNNERS = {
    "semantic-adherence": "videophy2/modified/sapc/run_sa.py",
    "physical-commonsense": "videophy2/modified/sapc/run_pc.py",
    "psnr": "psnr/modified/run_psnr.py",
    "dynamic-degree": "vbench/run_dynamic_degree.py",
    "motion-smoothness": "vbench/run_motion_smoothness.py",
    "cammc": "cammc/run_one.py",
}


@pytest.mark.parametrize("name", MODULES)
def test_score_registry_points_at_standalone_module(name):
    assert cli._SCORE_MODULES[name] == f".metrics.{name.replace('-', '_')}"


@pytest.mark.parametrize("name", MODULES)
def test_help_names_the_harness_command(name, capsys):
    with pytest.raises(SystemExit) as raised:
        MODULES[name].main(["--help"])
    assert raised.value.code == 0
    assert f"usage: percept score {name}" in capsys.readouterr().out


def test_score_dispatch_reaches_the_module(capsys):
    with pytest.raises(SystemExit) as raised:
        cli.main(["score", "psnr", "--help"])
    assert raised.value.code == 0
    assert "usage: percept score psnr" in capsys.readouterr().out


@pytest.mark.parametrize("name", MODULES)
def test_dry_run_prints_the_packaged_runner(name, capsys):
    assert MODULES[name].main([*REQUIRED[name], "--dry-run"]) == 0
    error = capsys.readouterr().err.replace("\\", "/")
    assert RUNNERS[name] in error
    assert "Running: " in error


@pytest.mark.parametrize("name", MODULES)
def test_runner_exit_code_is_forwarded(name, tmp_path):
    module = MODULES[name]
    script = tmp_path / "third_party" / module.PACKAGE / module.RUNNER
    script.parent.mkdir(parents=True)
    script.write_text("raise SystemExit(7)\n", encoding="utf-8")
    assert module.main([*REQUIRED[name], "--third-party-root", str(tmp_path),
                        "--python", sys.executable]) == 7


def test_missing_runner_is_reported(tmp_path, capsys):
    assert psnr.main([*REQUIRED["psnr"], "--third-party-root", str(tmp_path),
                      "--python", sys.executable]) == 1
    assert "Missing standalone runner" in capsys.readouterr().err


def test_root_accepts_checkout_or_third_party(tmp_path):
    (tmp_path / "third_party").mkdir()
    assert standalone.resolve_root(tmp_path) == (tmp_path / "third_party").resolve()


def test_psnr_forwards_limit_and_workers(tmp_path, monkeypatch):
    output = tmp_path / "arguments.json"
    script = tmp_path / "third_party" / psnr.PACKAGE / psnr.RUNNER
    script.parent.mkdir(parents=True)
    script.write_text(
        "import json, os, sys\n"
        "with open(os.environ['ARGS_OUT'], 'w', encoding='utf-8') as stream:\n"
        "    json.dump(sys.argv[1:], stream)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ARGS_OUT", str(output))
    arguments = [*REQUIRED["psnr"], "--limit", "2", "--workers", "5"]
    assert psnr.main([*arguments, "--third-party-root", str(tmp_path),
                      "--python", sys.executable]) == 0
    forwarded = json.loads(Path(output).read_text(encoding="utf-8"))
    assert forwarded == ["--manifest", "manifest.jsonl", "--output-dir", "out",
                         "--workers", "5", "--limit", "2"]


def test_cammc_forwards_arguments(tmp_path, monkeypatch):
    output = tmp_path / "arguments.json"
    script = tmp_path / "third_party" / cammc.PACKAGE / cammc.RUNNER
    script.parent.mkdir(parents=True)
    script.write_text(
        "import json, os, sys\n"
        "with open(os.environ['ARGS_OUT'], 'w', encoding='utf-8') as stream:\n"
        "    json.dump(sys.argv[1:], stream)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ARGS_OUT", str(output))
    arguments = ["--video", "v.mp4", "--reference", "ref.npz", "--intrinsics-file", "K.npy",
                 "--start-s", "1.5", "--end-s", "9", "--gpu", "3", "--any-revision", "--output", "out.json"]
    assert cammc.main([*arguments, "--third-party-root", str(tmp_path), "--python", sys.executable]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == [
        "--video", "v.mp4", "--reference", "ref.npz", "--intrinsics-file", "K.npy",
        "--start-s", "1.5", "--end-s", "9.0", "--gpu", "3", "--any-revision", "--output", "out.json"]
