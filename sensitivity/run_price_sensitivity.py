#!/usr/bin/env python3
"""Run the paired resource-price sensitivity experiment."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess


ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "bin" / "price_sensitivity"
SOURCE = ROOT / "sensitivity" / "price_sensitivity.cpp"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-run", default="runs/sensitivity_mu30_n60_k400")
    parser.add_argument(
        "--run-dir", default="runs/sensitivity_price_level")
    parser.add_argument("--rates", nargs="+", type=int,
                        default=[5, 10, 20, 40, 70, 100])
    parser.add_argument("--premium-ratio", type=int, default=3)
    parser.add_argument("--repetitions", nargs="+", type=int,
                        default=list(range(1, 11)))
    parser.add_argument("--time-limit", type=float, default=1800.0)
    parser.add_argument("--jobs", type=int, default=3)
    return parser.parse_args()


def rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def base_instance(directory: Path, repetition: int) -> Path:
    matches = sorted((directory / "instances").glob(f"*_r{repetition}.dat"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one base instance for repetition {repetition}, found "
            f"{len(matches)}")
    return matches[0]


def instance_seed(path: Path) -> int:
    with path.open() as handle:
        for line in handle:
            if line.startswith("seed "):
                return int(line.split()[1])
    raise ValueError(f"Missing seed in {path}")


def summarize(input_path: Path, output_path: Path) -> None:
    data = rows(input_path)
    groups: dict[int, list[dict[str, str]]] = {}
    for row in data:
        groups.setdefault(int(row["lambda"]), []).append(row)
    fields = [
        "opt_s", "opt_d", "objective", "expected_twc",
        "expected_overtime", "overtime_probability",
        "conditional_overtime", "booking_cost", "booking_total_share",
        "expected_processing", "normalized_s", "normalized_d",
        "s0_best_d", "s0_objective", "s0_deviation_pct", "committed_s",
        "committed_d", "committed_objective", "committed_deviation_pct",
        "seconds", "generated_labels", "peak_labels", "retained_vectors",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lambda", "mu", "instances", *fields])
        for booking_rate, group in sorted(groups.items()):
            writer.writerow([
                booking_rate, int(group[0]["mu"]), len(group),
                *(f"{statistics.fmean(float(row[field]) for row in group):.12g}"
                  for field in fields),
            ])


def main() -> int:
    args = arguments()
    if any(rate <= 0 for rate in args.rates):
        raise ValueError("Every booking rate must be positive")
    if args.premium_ratio <= 1:
        raise ValueError("The premium ratio must exceed one")
    subprocess.run(["make", "bin/price_sensitivity"], cwd=ROOT, check=True)

    base_run = Path(args.base_run)
    if not base_run.is_absolute():
        base_run = ROOT / base_run
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / "price_metrics.csv"
    completed = {
        int(row["seed"]) for row in rows(output)
        if int(row["lambda"]) == min(args.rates)
    }
    logs = run_dir / "logs"
    logs.mkdir(exist_ok=True)
    partials = run_dir / "partials"
    partials.mkdir(exist_ok=True)
    instances: list[Path] = []
    pending: list[tuple[int, Path, Path]] = []
    for repetition in args.repetitions:
        instance = base_instance(base_run, repetition)
        instances.append(instance)
        seed = instance_seed(instance)
        if seed in completed:
            continue
        pending.append((repetition, instance, partials / f"r{repetition}.csv"))

    def run_one(task: tuple[int, Path, Path]) -> Path:
        repetition, instance, partial = task
        if partial.exists():
            partial.unlink()
        command = [
            str(BINARY), str(instance), str(partial),
            ",".join(map(str, sorted(set(args.rates)))),
            str(args.premium_ratio), str(args.time_limit), "pricelevel",
        ]
        print("+", " ".join(command), flush=True)
        with (logs / f"r{repetition}.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        return partial

    completed_partials: list[Path] = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = {executor.submit(run_one, task): task[0] for task in pending}
        for future in as_completed(futures):
            completed_partials.append(future.result())
            print(f"Completed repetition {futures[future]}", flush=True)

    combined = rows(output)
    for partial in completed_partials:
        combined.extend(rows(partial))
    unique = {
        (int(row["seed"]), int(row["lambda"])): row for row in combined
    }
    if unique:
        ordered = [unique[key] for key in sorted(unique)]
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ordered[0]))
            writer.writeheader()
            writer.writerows(ordered)

    summary = run_dir / "price_summary.csv"
    summarize(output, summary)

    # The lambda=10 member must reproduce the existing premium-ratio-three run.
    prior = {
        int(row["seed"]): row
        for row in rows(base_run / "results.csv")
        if row["method"] == "proposed" and row["status"] == "optimal"
    }
    for row in rows(output):
        if int(row["lambda"]) != 10:
            continue
        old = prior[int(row["seed"])]
        if (int(row["opt_s"]), int(row["opt_d"])) != (
                int(old["best_s"]), int(old["best_d"])):
            raise ValueError("The lambda=10 optimal contract was not reproduced")
        tolerance = 1e-7 * max(1.0, abs(float(old["objective"])))
        if abs(float(row["objective"]) - float(old["objective"])) > tolerance:
            raise ValueError("The lambda=10 objective was not reproduced")

    provenance = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "rates": sorted(set(args.rates)),
        "premium_ratio": args.premium_ratio,
        "repetitions": args.repetitions,
        "time_limit": args.time_limit,
        "source_sha256": digest(SOURCE),
        "binary_sha256": digest(BINARY),
        "base_instance_sha256": {
            path.name: digest(path) for path in instances
        },
        "output_sha256": digest(output),
    }
    (run_dir / "price_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"Metrics: {output}")
    print(f"Summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
