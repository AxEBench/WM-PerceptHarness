#!/usr/bin/env python3
"""Check a Percept installation without downloading weights or calling APIs.

Run from the repository with the Python environment you intend to use:
    python tools/verify_install.py
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import math
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig


def check_python():
    if sys.version_info < (3, 12):
        raise RuntimeError('Python 3.12+ is required')
    return sys.version.split()[0]


def check_core():
    for name in ('httpx', 'pydantic', 'pydantic_settings'):
        importlib.import_module(name)
    package = importlib.import_module('percept_harness')
    return f'{package.__version__} ({Path(package.__file__).resolve()})'


def check_program(name):
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f'{name} is missing from PATH; install FFmpeg/FFprobe')
    result = subprocess.run([path, '-version'], capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise RuntimeError(f'{name} failed to execute (exit {result.returncode})')
    return result.stdout.splitlines()[0] if result.stdout.strip() else path


def check_cli():
    launcher = shutil.which('percept')
    expected = Path(sysconfig.get_path('scripts')) / ('percept.exe' if sys.platform == 'win32' else 'percept')
    if launcher is None:
        raise RuntimeError('percept is missing from PATH; activate the installed Python environment')
    if Path(launcher).resolve() != expected.resolve():
        raise RuntimeError('percept on PATH belongs to another environment; activate the environment shown above')
    commands = [['--help'], ['annotate', '--help'], ['score', 'events', '--help']]
    for args in commands:
        result = subprocess.run([launcher, *args],
                                capture_output=True, text=True, timeout=30)
        if result.returncode or 'usage:' not in result.stdout:
            raise RuntimeError(f'percept {" ".join(args)} failed; reinstall the package in this environment')
    return f'{len(commands)} command parsers available'


def check_smoke():
    from percept_harness.evaluation.event_matching import match_weight_matrix, temporal_iou
    from percept_harness.evaluation.events import score_pair
    from percept_harness.annotation.pipelines.event_schemas import GroundedEvents
    from percept_harness.models.base import ModelOutputError, parse_strict_json

    if parse_strict_json('{"duration": 1, "events": []}') != {'duration': 1, 'events': []}:
        raise RuntimeError('Structured output parsing failed')
    try:
        parse_strict_json('{"score": NaN}')
    except ModelOutputError:
        pass
    else:
        raise RuntimeError('Structured output accepted a non-finite number')
    reference, prediction = GroundedEvents.model_validate({'duration': 3, 'events': [
        {'start': 0, 'end': 2, 'description': 'A hand moves a cup.'},
        {'start': 1, 'end': 3, 'description': 'A hand moves a cup.'},
    ]}).events
    if not math.isclose(temporal_iou(reference, prediction), 1 / 3):
        raise RuntimeError('Temporal overlap smoke check failed')
    matches = match_weight_matrix([[temporal_iou(reference, prediction)]], threshold=0.3)
    if len(matches) != 1 or (matches[0].reference_index, matches[0].prediction_index) != (0, 0):
        raise RuntimeError('Event matching smoke check failed')
    grounded = GroundedEvents.model_validate({'duration': 2.0, 'events': [
        {'start': 0.0, 'end': 2.0, 'description': 'A hand moves a cup.'}]})
    score = score_pair(grounded, grounded, lambda a, b: {'match': a == b, 'reason': 'same action'})
    if score['timing'] != 1 or score['content'] != 1 or score['fidelity'] != 1:
        raise RuntimeError('Grounded event scoring smoke check failed')
    return 'JSON parsing, temporal matching and grounded event scoring passed'


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    checks = [
        ('Python', check_python),
        ('Core package', check_core),
        ('FFmpeg', lambda: check_program('ffmpeg')),
        ('FFprobe', lambda: check_program('ffprobe')),
        ('CLI', check_cli),
        ('Local smoke', check_smoke),
    ]
    print(f'Percept installation check\nPython: {sys.executable}')
    failed = 0
    for name, function in checks:
        try:
            detail = function()
            print(f'[PASS] {name}: {detail}')
        except Exception as error:
            failed += 1
            print(f'[FAIL] {name}: {error}')
    print(f'\n{len(checks) - failed}/{len(checks)} checks passed.')
    print('Local checks only; API connectivity and real-model inference were not tested.')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
