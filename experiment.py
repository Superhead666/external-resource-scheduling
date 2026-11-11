#!/usr/bin/env python3
"""Configure, generate, and run a reproducible benchmark experiment."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any

from provenance import record_instances, record_run, verify_sources


MAIN_METHODS = ["proposed", "time_indexed", "cp", "integer_lshaped_cp"]
OPTIONAL_METHODS = ["compact", "integer_lshaped_milp"]
COMPONENT_METHODS = [
    "component_dense_direct",
    "component_sparse_direct",
    "component_label_dominance_batched",
    "component_terminal_dominance_batched",
    "component_dominance_direct",
    "component_dominance_batched",
]
METHODS = MAIN_METHODS + OPTIONAL_METHODS + COMPONENT_METHODS
BUILD_TARGETS = {
    "proposed": "bin/proposed_solver",
    "compact": "bin/gurobi_baselines",
    "time_indexed": "bin/gurobi_baselines",
    "cp": "bin/cp_baseline",
    "integer_lshaped_milp": "bin/integer_lshaped",
    "integer_lshaped_cp": "bin/integer_lshaped",
    "component_dense_direct": "bin/component_solver",
    "component_sparse_direct": "bin/component_solver",
    "component_label_dominance_batched": "bin/component_solver",
    "component_terminal_dominance_batched": "bin/component_solver",
    "component_dominance_direct": "bin/component_solver",
    "component_dominance_batched": "bin/component_solver",
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Generate and run a configurable experiment. Values supplied on the "
            "command line override the JSON configuration."
        )
    )
    result.add_argument("--config", default="configs/scalability.json")
    result.add_argument("--name")
    result.add_argument(
        "--output-root",
        help="directory containing the named run (default: runs)")
    result.add_argument("--results-file", help="result CSV filename")
    result.add_argument("--failures-file", help="failure CSV filename")
    result.add_argument("--summary-file", help="summary CSV filename")
    result.add_argument(
        "--instance-results-file",
        help="optional enriched instance-level result CSV filename")
    result.add_argument("--target-jobs", nargs="+", type=int)
    result.add_argument("--scenario-counts", nargs="+", type=int)
    result.add_argument(
        "--scenario-mode", choices=["sampled", "exhaustive"])
    result.add_argument("--processing-max", nargs="+", type=int)
    result.add_argument("--weight-max", nargs="+", type=int)
    result.add_argument("--resource-share", nargs="+", type=float)
    result.add_argument("--pool-multiplier", nargs="+", type=float)
    result.add_argument("--arrival-probability", nargs="+", type=float)
    result.add_argument("--arrival-heterogeneity", nargs="+", type=float)
    result.add_argument(
        "--correlation", nargs="+", choices=["independent", "equal"])
    result.add_argument("--committed-fraction", nargs="+", type=float)
    result.add_argument("--repetitions", type=int)
    result.add_argument("--lambda-rate", type=int)
    result.add_argument("--mu-rate", type=int)
    result.add_argument("--seed-base", type=int)
    result.add_argument("--time-limit", type=float)
    result.add_argument("--threads", type=int)
    result.add_argument("--memory-limit-gb", type=float)
    result.add_argument("--diagram-fallback-width", type=int)
    result.add_argument("--diagram-time-limit", type=float)
    result.add_argument("--methods", nargs="+", choices=METHODS + ["all"])
    result.add_argument(
        "--run-order", choices=["instance-major", "method-major"],
        help="execution order; instance-major preserves matched comparisons")
    result.add_argument("--generate-only", action="store_true")
    result.add_argument("--retry-failures", action="store_true")
    result.add_argument("--no-build", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def load_configuration(root: Path, path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    if not path.is_absolute():
        path = root / path
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("The configuration root must be a JSON object")
    return value


def apply_overrides(configuration: dict[str, Any], args: argparse.Namespace) -> None:
    if args.pool_multiplier is not None and args.arrival_probability is not None:
        raise ValueError(
            "Use either --pool-multiplier or --arrival-probability, not both")
    mapping = {
        "name": args.name,
        "output_root": args.output_root,
        "results_file": args.results_file,
        "failures_file": args.failures_file,
        "summary_file": args.summary_file,
        "instance_results_file": args.instance_results_file,
        "target_jobs": args.target_jobs,
        "scenario_counts": args.scenario_counts,
        "scenario_mode": args.scenario_mode,
        "processing_max": args.processing_max,
        "weight_max": args.weight_max,
        "resource_share": args.resource_share,
        "pool_multiplier": args.pool_multiplier,
        "arrival_heterogeneity": args.arrival_heterogeneity,
        "correlation": args.correlation,
        "committed_fraction": args.committed_fraction,
        "repetitions": args.repetitions,
        "lambda": args.lambda_rate,
        "mu": args.mu_rate,
        "seed_base": args.seed_base,
        "time_limit": args.time_limit,
        "threads": args.threads,
        "memory_limit_gb": args.memory_limit_gb,
        "diagram_fallback_width": args.diagram_fallback_width,
        "diagram_time_limit": args.diagram_time_limit,
        "methods": args.methods,
        "run_order": args.run_order,
    }
    for key, value in mapping.items():
        if value is not None:
            configuration[key] = value
    if args.arrival_probability is not None:
        if any(not (0.0 < value <= 1.0)
               for value in args.arrival_probability):
            raise ValueError(
                "Every arrival probability must lie in (0,1]")
        configuration["pool_multiplier"] = [
            1.0 / value for value in args.arrival_probability]
        if args.arrival_heterogeneity is None:
            configuration["arrival_heterogeneity"] = [0.0]


def as_list(configuration: dict[str, Any], key: str, conversion: type) -> list[Any]:
    value = configuration.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a nonempty JSON array")
    return [conversion(item) for item in value]


def normalize(configuration: dict[str, Any]) -> dict[str, Any]:
    required_scalars = ["name", "repetitions", "time_limit", "threads"]
    missing = [key for key in required_scalars if key not in configuration]
    if missing:
        raise ValueError(f"Missing configuration fields: {', '.join(missing)}")

    result: dict[str, Any] = {
        "name": str(configuration["name"]),
        "output_root": str(configuration.get("output_root", "runs")),
        "results_file": str(configuration.get("results_file", "results.csv")),
        "failures_file": str(configuration.get("failures_file", "failures.csv")),
        "summary_file": str(configuration.get("summary_file", "summary.csv")),
        "target_jobs": as_list(configuration, "target_jobs", int),
        "scenario_counts": as_list(configuration, "scenario_counts", int),
        "scenario_mode": str(configuration.get("scenario_mode", "sampled")),
        "processing_max": as_list(configuration, "processing_max", int),
        "weight_max": as_list(configuration, "weight_max", int),
        "resource_share": as_list(configuration, "resource_share", float),
        "pool_multiplier": as_list(configuration, "pool_multiplier", float),
        "arrival_heterogeneity": as_list(
            configuration, "arrival_heterogeneity", float),
        "correlation": as_list(configuration, "correlation", str),
        "committed_fraction": as_list(
            configuration, "committed_fraction", float),
        "repetitions": int(configuration["repetitions"]),
        "lambda": int(configuration.get("lambda", 10)),
        "mu": int(configuration.get("mu", 30)),
        "seed_base": int(configuration.get("seed_base", 27000000)),
        "time_limit": float(configuration["time_limit"]),
        "threads": int(configuration["threads"]),
        "memory_limit_gb": float(configuration.get("memory_limit_gb", 0.0)),
        "diagram_fallback_width": int(
            configuration.get("diagram_fallback_width", 0)),
        "diagram_time_limit": float(
            configuration.get("diagram_time_limit", configuration["time_limit"])),
        "methods": [str(item) for item in configuration.get("methods", ["proposed"])],
        "run_order": str(configuration.get("run_order", "instance-major")),
    }
    if "instance_results_file" in configuration:
        result["instance_results_file"] = str(
            configuration["instance_results_file"])

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", result["name"]):
        raise ValueError("name may contain only letters, digits, underscores, and hyphens")
    file_keys = ["results_file", "failures_file", "summary_file"]
    if "instance_results_file" in result:
        file_keys.append("instance_results_file")
    for key in file_keys:
        filename = Path(result[key])
        if filename.name != result[key] or filename.suffix.lower() != ".csv":
            raise ValueError(f"{key} must be a CSV filename without a directory")
    output_files = [result[key] for key in file_keys]
    if len(set(output_files)) != len(output_files):
        raise ValueError("Output CSV filenames must differ")
    for key in ["target_jobs", "scenario_counts", "processing_max", "weight_max"]:
        if any(value <= 0 for value in result[key]):
            raise ValueError(f"Every value in {key} must be positive")
    if result["scenario_mode"] not in {"sampled", "exhaustive"}:
        raise ValueError("scenario_mode must be sampled or exhaustive")
    if result["scenario_mode"] == "exhaustive" \
            and len(result["scenario_counts"]) != 1:
        raise ValueError(
            "Exhaustive mode requires one scenario count equal to 2^M")
    for key in ["resource_share", "committed_fraction"]:
        if any(not (0.0 < value <= 1.0) for value in result[key]):
            raise ValueError(f"Every value in {key} must lie in (0,1]")
    if any(value < 1.0 for value in result["pool_multiplier"]):
        raise ValueError("Every pool multiplier must be at least one")
    if any(not (0.0 <= value < 1.0)
           for value in result["arrival_heterogeneity"]):
        raise ValueError("Every arrival heterogeneity must lie in [0,1)")
    unknown_correlations = sorted(
        set(result["correlation"]) - {"independent", "equal"})
    if unknown_correlations:
        raise ValueError(
            f"Unknown correlations: {', '.join(unknown_correlations)}")
    if result["repetitions"] <= 0 or result["time_limit"] <= 0 or result["threads"] <= 0:
        raise ValueError("repetitions, time_limit, and threads must be positive")
    if result["memory_limit_gb"] < 0:
        raise ValueError("memory_limit_gb must be nonnegative")
    if result["diagram_fallback_width"] < 0:
        raise ValueError("diagram_fallback_width must be nonnegative")
    if result["diagram_time_limit"] <= 0:
        raise ValueError("diagram_time_limit must be positive")
    if not (0 < result["lambda"] < result["mu"]):
        raise ValueError("Rates must satisfy 0 < lambda < mu")
    if result["methods"] == ["all"]:
        result["methods"] = MAIN_METHODS.copy()
    unknown = sorted(set(result["methods"]) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}")
    result["methods"] = list(dict.fromkeys(result["methods"]))
    if result["run_order"] not in {"instance-major", "method-major"}:
        raise ValueError("run_order must be instance-major or method-major")
    return result


def csv_values(values: list[Any]) -> str:
    return ",".join(str(value) for value in values)


def execute(command: list[str | Path], *, root: Path, dry_run: bool) -> None:
    rendered = [str(item) for item in command]
    print("+", shlex.join(rendered), flush=True)
    if not dry_run:
        subprocess.run(rendered, cwd=root, check=True)


def expected_instances(configuration: dict[str, Any]) -> int:
    factors = [
        len(configuration["target_jobs"]),
        len(configuration["scenario_counts"]),
        len(configuration["processing_max"]),
        len(configuration["weight_max"]),
        len(configuration["resource_share"]),
        len(configuration["pool_multiplier"]),
        len(configuration["arrival_heterogeneity"]),
        len(configuration["correlation"]),
        len(configuration["committed_fraction"]),
        configuration["repetitions"],
    ]
    return math.prod(factors)


def check_configuration_lock(path: Path, configuration: dict[str, Any],
                             dry_run: bool) -> None:
    if path.exists():
        with path.open() as handle:
            previous = json.load(handle)
        # Resolved configurations created before the explicit scenario-mode
        # option all used sampled scenarios.
        previous.setdefault("scenario_mode", "sampled")
        previous.setdefault("output_root", "runs")
        previous.setdefault("results_file", "results.csv")
        previous.setdefault("failures_file", "failures.csv")
        previous.setdefault("summary_file", "summary.csv")
        previous.setdefault("run_order", "instance-major")
        if previous != configuration:
            differing = sorted(
                key for key in set(previous) | set(configuration)
                if previous.get(key) != configuration.get(key)
            )
            raise ValueError(
                "Experiment name already exists with different settings "
                f"({', '.join(differing)}). Choose a new --name to avoid mixing results."
            )
        return
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            json.dump(configuration, handle, indent=2, sort_keys=True)
            handle.write("\n")


def main() -> int:
    args = parser().parse_args()
    root = Path(__file__).resolve().parent
    try:
        configuration = load_configuration(root, args.config)
        apply_overrides(configuration, args)
        configuration = normalize(configuration)

        source_hashes = verify_sources(root)

        output_root = Path(configuration["output_root"])
        if not output_root.is_absolute():
            output_root = root / output_root
        run_directory = output_root / configuration["name"]
        run_lock = None
        if not args.dry_run:
            run_directory.mkdir(parents=True, exist_ok=True)
            run_lock = (run_directory / ".run.lock").open("a")
            try:
                fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("This experiment is already running") from None
        check_configuration_lock(
            run_directory / "resolved_config.json", configuration, args.dry_run)

        count = expected_instances(configuration)
        total_runs = count * len(configuration["methods"])
        print(f"Experiment: {configuration['name']}")
        print(f"Instances: {count}; method runs: {total_runs}")
        print(f"Output directory: {run_directory}")

        targets = ["bin/generate_instances"]
        targets.extend(BUILD_TARGETS[method] for method in configuration["methods"])
        targets = list(dict.fromkeys(targets))
        if not args.no_build:
            execute(["make", "-j2", *targets], root=root, dry_run=args.dry_run)
        if not args.dry_run:
            record_run(root, run_directory, targets, source_hashes)

        instances = run_directory / "instances"
        generator = [
            root / "bin/generate_instances", instances, "custom",
            "--name", configuration["name"],
            "--target-jobs", csv_values(configuration["target_jobs"]),
            "--scenarios", csv_values(configuration["scenario_counts"]),
            "--scenario-mode", configuration["scenario_mode"],
            "--processing-max", csv_values(configuration["processing_max"]),
            "--weight-max", csv_values(configuration["weight_max"]),
            "--resource-share", csv_values(configuration["resource_share"]),
            "--pool-multiplier", csv_values(configuration["pool_multiplier"]),
            "--arrival-heterogeneity",
            csv_values(configuration["arrival_heterogeneity"]),
            "--correlation", csv_values(configuration["correlation"]),
            "--committed-fraction", csv_values(configuration["committed_fraction"]),
            "--repetitions", str(configuration["repetitions"]),
            "--lambda", str(configuration["lambda"]),
            "--mu", str(configuration["mu"]),
            "--seed-base", str(configuration["seed_base"]),
        ]
        execute(generator, root=root, dry_run=args.dry_run)
        if not args.dry_run:
            record_instances(run_directory)

        if args.generate_only:
            print("Generation complete; solver run skipped.")
            return 0

        results = run_directory / configuration["results_file"]
        failures = run_directory / configuration["failures_file"]
        runner = [
            sys.executable, root / "run_suite.py",
            "--manifest", instances / "manifest_custom.csv",
            "--results", results,
            "--failures", failures,
            "--logs", run_directory / "logs",
            "--methods", *configuration["methods"],
            "--run-order", configuration["run_order"],
            "--time-limit", str(configuration["time_limit"]),
            "--threads", str(configuration["threads"]),
            "--memory-limit-gb", str(configuration["memory_limit_gb"]),
        ]
        if configuration["diagram_fallback_width"] > 0:
            runner.extend([
                "--diagram-fallback-width",
                str(configuration["diagram_fallback_width"]),
                "--diagram-time-limit",
                str(configuration["diagram_time_limit"]),
            ])
        if args.retry_failures:
            runner.append("--retry-failures")
        execute(runner, root=root, dry_run=args.dry_run)

        if args.dry_run:
            return 0
        if results.exists() and results.stat().st_size > 0:
            summary_command = [
                sys.executable, root / "summarize_results.py", results,
                "--output", run_directory / configuration["summary_file"],
            ]
            if "instance_results_file" in configuration:
                summary_command.extend([
                    "--instance-output",
                    run_directory / configuration["instance_results_file"],
                ])
            execute(summary_command, root=root, dry_run=False)
        else:
            print("No method result was recorded; inspect failures.csv and logs.")
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"experiment: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
