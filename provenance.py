"""Verify the frozen submission sources and record the actual run environment."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources(root: Path) -> dict[str, str]:
    with (root / "source_manifest.json").open() as handle:
        manifest = json.load(handle)
    expected = manifest["sha256"]
    changed = [name for name, value in expected.items()
               if not (root / name).is_file() or digest(root / name) != value]
    if changed:
        raise ValueError("Frozen submission files changed: " + ", ".join(changed))
    return expected


def command_output(command: list[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=10, check=False)
        return (result.stdout or result.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return str(error)


def record_run(root: Path, directory: Path, targets: list[str],
               source_hashes: dict[str, str]) -> None:
    binaries = {name: digest(root / name) for name in targets}
    path = directory / "provenance.json"
    if path.exists():
        with path.open() as handle:
            previous = json.load(handle)
        if previous["source_sha256"] != source_hashes \
                or previous["binary_sha256"] != binaries:
            raise ValueError("Source or executable differs from the existing run. "
                             "Use a new experiment name.")
        return
    data = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_hashes,
        "binary_sha256": binaries,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": command_output(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "memory_bytes": command_output(["sysctl", "-n", "hw.memsize"]),
        "python": sys.version,
        "python_executable": sys.executable,
        "compiler": command_output(shlex.split(os.environ.get("CXX", "c++"))
                                   + ["--version"]),
        "build_recipe": command_output(["make", "-n", "-B", *targets]),
        "command": sys.argv,
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def record_instances(directory: Path) -> None:
    instances = directory / "instances"
    current = {p.name: digest(p) for p in sorted(instances.iterdir())
               if p.suffix in {".dat", ".csv"}}
    path = directory / "instance_sha256.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != current:
            raise ValueError("Generated instances differ from the existing run. "
                             "Use a new experiment name.")
    else:
        path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
