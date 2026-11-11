#!/usr/bin/env python3
"""Complete the submission sensitivity study with reproducible paired runs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
METRICS_SOURCE = ROOT / "sensitivity" / "sensitivity_metrics.cpp"
GENERATOR_SOURCE = ROOT / "sensitivity" / "generate_paired_committed.cpp"
METRICS_BINARY = ROOT / "bin" / "sensitivity_metrics"
GENERATOR_BINARY = ROOT / "bin" / "generate_paired_committed"
PROPOSED_BINARY = ROOT / "bin" / "proposed_solver"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subparsers = result.add_subparsers(dest="mode", required=True)

    metrics = subparsers.add_parser(
        "metrics", help="calculate policy and cost metrics for completed runs")
    metrics.add_argument("--run-dirs", nargs="+", required=True)
    metrics.add_argument("--output", required=True)
    metrics.add_argument("--summary", required=True)
    metrics.add_argument("--time-limit", type=float, default=1800.0)

    paired = subparsers.add_parser(
        "paired", help="run the paired committed-fraction experiment")
    paired.add_argument("--run-dir", default="runs/sensitivity_committed_paired_v2")
    paired.add_argument("--name", default="sensitivity_committed_paired_v2")
    paired.add_argument("--target-jobs", type=int, default=60)
    paired.add_argument("--scenarios", type=int, default=400)
    paired.add_argument("--processing-max", type=int, default=25)
    paired.add_argument("--weight-max", type=int, default=10)
    paired.add_argument("--resource-share", type=float, default=0.25)
    paired.add_argument("--pool-multiplier", type=int, default=100)
    paired.add_argument("--arrival-heterogeneity", type=float, default=0.5)
    paired.add_argument(
        "--committed-counts", nargs="+", type=int,
        default=[10, 20, 30, 40, 50, 60])
    paired.add_argument("--repetitions", type=int, default=10)
    paired.add_argument("--lambda-rate", type=int, default=10)
    paired.add_argument("--mu-rate", type=int, default=30)
    paired.add_argument("--seed-base", type=int, default=37000000)
    paired.add_argument("--time-limit", type=float, default=1800.0)
    paired.add_argument("--threads", type=int, default=1)
    return result


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_tools(include_generator: bool) -> None:
    (ROOT / "bin").mkdir(exist_ok=True)
    common = [
        "c++", "-O3", "-DNDEBUG", "-std=c++20", "-Wall", "-Wextra",
        "-Wpedantic", "-I", str(ROOT / "src")
    ]
    subprocess.run(
        [*common, str(METRICS_SOURCE), "-o", str(METRICS_BINARY)],
        cwd=ROOT, check=True)
    if include_generator:
        subprocess.run(
            [*common, str(GENERATOR_SOURCE), "-o", str(GENERATOR_BINARY)],
            cwd=ROOT, check=True)
        subprocess.run(
            ["make", "bin/proposed_solver"], cwd=ROOT, check=True)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def instance_jobs(path: Path) -> list[tuple[int, int, bool]]:
    lines = path.read_text().splitlines()
    job_line = next(index for index, line in enumerate(lines)
                    if line.startswith("jobs "))
    count = int(lines[job_line].split()[1])
    jobs: list[tuple[int, int, bool]] = []
    for line in lines[job_line + 1:job_line + 1 + count]:
        _, processing, weight, resource, _ = map(int, line.split())
        jobs.append((processing, weight, bool(resource)))
    return jobs


def verify_paired_instances(rows: list[dict[str, str]], directory: Path,
                            block_size: int) -> None:
    by_repetition: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        by_repetition.setdefault(int(row["repetition"]), []).append(row)
    for repetition, group in by_repetition.items():
        group.sort(key=lambda row: int(row["committed"]))
        for lower, upper in zip(group, group[1:]):
            lower_count = int(lower["committed"])
            upper_count = int(upper["committed"])
            lower_jobs = instance_jobs(directory / lower["path"])
            upper_jobs = instance_jobs(directory / upper["path"])
            if lower_jobs[:lower_count] != upper_jobs[:lower_count]:
                raise ValueError(
                    f"Committed jobs are not nested in repetition {repetition}")
            for block in range(lower_count, upper_count):
                lower_id = lower_count + (block - lower_count) * block_size
                if lower_jobs[lower_id] != upper_jobs[block]:
                    raise ValueError(
                        f"Revealed job data changed in repetition {repetition}")
            target_jobs = int(lower["target_jobs"])
            for block in range(upper_count, target_jobs):
                lower_start = lower_count + (block - lower_count) * block_size
                upper_start = upper_count + (block - upper_count) * block_size
                if lower_jobs[lower_start:lower_start + block_size] \
                        != upper_jobs[upper_start:upper_start + block_size]:
                    raise ValueError(
                        f"Potential job data changed in repetition {repetition}")


def locate_instance(run_directory: Path, name: str) -> Path:
    path = run_directory / "instances" / f"{name}.dat"
    if not path.exists():
        raise FileNotFoundError(f"Missing instance {path}")
    return path


def append_metrics(run_directories: Iterable[Path], output: Path,
                   time_limit: float) -> None:
    completed = {row["instance"] for row in read_rows(output)}
    output.parent.mkdir(parents=True, exist_ok=True)
    log_directory = output.parent / f"{output.stem}_logs"
    log_directory.mkdir(exist_ok=True)
    for run_directory in run_directories:
        results = read_rows(run_directory / "results.csv")
        for row in results:
            if row["method"] != "proposed" or row["status"] != "optimal":
                continue
            if row["instance"] in completed:
                continue
            instance = locate_instance(run_directory, row["instance"])
            command = [
                str(METRICS_BINARY), str(instance), str(output), row["best_s"],
                row["best_d"], row["objective"], str(time_limit)
            ]
            log_path = log_directory / f"{row['instance']}.log"
            print("+", " ".join(command), flush=True)
            with log_path.open("w") as log:
                subprocess.run(command, cwd=ROOT, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            completed.add(row["instance"])


def mean(rows: list[dict[str, str]], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows)


def summarize_metrics(input_path: Path, output_path: Path) -> None:
    rows = read_rows(input_path)
    groups: dict[tuple[int, int], list[dict[str, str]]] = {}
    for row in rows:
        key = (int(row["n_committed"]), int(row["mu"]))
        groups.setdefault(key, []).append(row)
    fields = [
        "opt_s", "opt_d", "normalized_s", "normalized_d",
        "expected_twc", "booking_cost", "expected_overtime",
        "overtime_cost", "overtime_probability", "booking_resource_share",
        "booking_total_share", "s0_deviation_pct",
        "committed_deviation_pct", "committed_expected_twc",
        "committed_booking_cost", "committed_expected_overtime",
        "committed_overtime_cost", "delta_twc", "delta_booking_cost",
        "delta_overtime_cost", "metric_seconds"
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n_committed", "mu", "instances", *fields])
        for (committed, overtime_rate), group in sorted(groups.items()):
            writer.writerow([
                committed, overtime_rate, len(group),
                *(f"{mean(group, field):.12g}" for field in fields)
            ])


def write_provenance(path: Path, extra: dict[str, object]) -> None:
    tracked = [
        METRICS_SOURCE, ROOT / "src" / "frontier_labeling_core.inc",
        ROOT / "src" / "instance.hpp", Path(__file__)
    ]
    if GENERATOR_SOURCE.exists():
        tracked.append(GENERATOR_SOURCE)
    data = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "source_sha256": {
            str(item.relative_to(ROOT)): digest(item) for item in tracked
        },
        **extra,
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def metrics_mode(args: argparse.Namespace) -> None:
    compile_tools(include_generator=False)
    run_directories = [
        path if path.is_absolute() else ROOT / path
        for path in map(Path, args.run_dirs)
    ]
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    summary = Path(args.summary)
    if not summary.is_absolute():
        summary = ROOT / summary
    append_metrics(run_directories, output, args.time_limit)
    summarize_metrics(output, summary)
    write_provenance(output.with_suffix(".provenance.json"), {
        "binary_sha256": digest(METRICS_BINARY),
        "input_results_sha256": {
            str(path / "results.csv"): digest(path / "results.csv")
            for path in run_directories
        },
    })
    print(f"Metrics: {output}")
    print(f"Summary: {summary}")


def paired_configuration(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: value for key, value in vars(args).items()
        if key not in {"mode"}
    }


def check_configuration(path: Path, configuration: dict[str, object]) -> None:
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != configuration:
            raise ValueError(
                "The run directory contains a different design; choose a new name")
    else:
        path.write_text(json.dumps(configuration, indent=2, sort_keys=True) + "\n")


def paired_mode(args: argparse.Namespace) -> None:
    compile_tools(include_generator=True)
    run_directory = Path(args.run_dir)
    if not run_directory.is_absolute():
        run_directory = ROOT / run_directory
    run_directory.mkdir(parents=True, exist_ok=True)
    configuration = paired_configuration(args)
    check_configuration(run_directory / "resolved_config.json", configuration)
    instances = run_directory / "instances"
    instances.mkdir(exist_ok=True)
    manifest = instances / "manifest_paired.csv"
    if not manifest.exists():
        command = [
            str(GENERATOR_BINARY), str(instances), "--name", args.name,
            "--target-jobs", str(args.target_jobs), "--scenarios",
            str(args.scenarios), "--processing-max", str(args.processing_max),
            "--weight-max", str(args.weight_max), "--resource-share",
            str(args.resource_share), "--pool-multiplier",
            str(args.pool_multiplier), "--arrival-heterogeneity",
            str(args.arrival_heterogeneity), "--committed-counts",
            ",".join(map(str, args.committed_counts)), "--repetitions",
            str(args.repetitions), "--lambda", str(args.lambda_rate),
            "--mu", str(args.mu_rate), "--seed-base", str(args.seed_base)
        ]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

    manifest_rows = read_rows(manifest)
    verify_paired_instances(manifest_rows, instances, args.pool_multiplier)
    results = run_directory / "results.csv"
    completed = {row["instance"] for row in read_rows(results)}
    logs = run_directory / "logs"
    logs.mkdir(exist_ok=True)
    manifest_rows.sort(key=lambda row: (
        int(row["committed"]), int(row["repetition"])))
    for row in manifest_rows:
        if row["instance"] in completed:
            continue
        instance = instances / row["path"]
        command = [
            str(PROPOSED_BINARY), str(instance), str(results),
            str(args.time_limit), str(args.threads)
        ]
        log_path = logs / f"{row['instance']}__proposed.log"
        print("+", " ".join(command), flush=True)
        with log_path.open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        completed.add(row["instance"])

    subprocess.run([
        sys.executable, str(ROOT / "summarize_results.py"), str(results),
        "--output", str(run_directory / "summary.csv")
    ], cwd=ROOT, check=True)
    metrics = run_directory / "sensitivity_metrics.csv"
    append_metrics([run_directory], metrics, args.time_limit)
    summarize_metrics(metrics, run_directory / "sensitivity_summary.csv")
    instance_hashes = {
        path.name: digest(path) for path in sorted(instances.glob("*.dat"))
    }
    (run_directory / "instance_sha256.json").write_text(
        json.dumps(instance_hashes, indent=2, sort_keys=True) + "\n")
    write_provenance(run_directory / "sensitivity_provenance.json", {
        "binary_sha256": {
            "bin/proposed_solver": digest(PROPOSED_BINARY),
            "bin/sensitivity_metrics": digest(METRICS_BINARY),
            "bin/generate_paired_committed": digest(GENERATOR_BINARY),
        },
        "results_sha256": digest(results),
        "metrics_sha256": digest(metrics),
    })
    print(f"Paired sensitivity results: {run_directory}")


def main() -> int:
    args = parser().parse_args()
    try:
        if args.mode == "metrics":
            metrics_mode(args)
        else:
            paired_mode(args)
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"run_sensitivity: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
