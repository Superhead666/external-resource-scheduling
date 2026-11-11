#!/usr/bin/env python3
"""Run benchmark methods sequentially with checkpointing and per-run logs."""

from __future__ import annotations

import argparse
import ctypes
import csv
import os
from pathlib import Path
import subprocess
import sys
import time


METHOD_COMMANDS = {
    "proposed": lambda root, instance, output, limit, threads, log: [
        root / "bin/proposed_solver", instance, output, str(limit), str(threads)
    ],
    "compact": lambda root, instance, output, limit, threads, log: [
        root / "bin/gurobi_baselines", "compact", instance, output,
        str(limit), str(threads), log
    ],
    "time_indexed": lambda root, instance, output, limit, threads, log: [
        root / "bin/gurobi_baselines", "time_indexed", instance, output,
        str(limit), str(threads), log
    ],
    "cp": lambda root, instance, output, limit, threads, log: [
        root / "bin/cp_baseline", instance, output, str(limit), str(threads), log
    ],
    "integer_lshaped": lambda root, instance, output, limit, threads, log: [
        root / "bin/integer_lshaped", instance, output,
        str(limit), str(threads), log, "milp"
    ],
    "integer_lshaped_milp": lambda root, instance, output, limit, threads, log: [
        root / "bin/integer_lshaped", instance, output,
        str(limit), str(threads), log, "milp"
    ],
    "integer_lshaped_cp": lambda root, instance, output, limit, threads, log: [
        root / "bin/integer_lshaped", instance, output,
        str(limit), str(threads), log, "cp"
    ],
    "component_dense_direct": lambda root, instance, output, limit, threads, log: [
        root / "bin/component_solver", "dense_direct", instance, output,
        str(limit), str(threads)
    ],
    "component_sparse_direct": lambda root, instance, output, limit, threads, log: [
        root / "bin/component_solver", "sparse_direct", instance, output,
        str(limit), str(threads)
    ],
    "component_label_dominance_batched": lambda root, instance, output, limit, threads, log: [
        root / "bin/component_solver", "label_dominance_batched", instance, output,
        str(limit), str(threads)
    ],
    "component_terminal_dominance_batched": lambda root, instance, output, limit, threads, log: [
        root / "bin/component_solver", "terminal_dominance_batched", instance, output,
        str(limit), str(threads)
    ],
    "component_dominance_direct": lambda root, instance, output, limit, threads, log: [
        root / "bin/component_solver", "dominance_direct", instance, output,
        str(limit), str(threads)
    ],
    "component_dominance_batched": lambda root, instance, output, limit, threads, log: [
        root / "bin/component_solver", "dominance_batched", instance, output,
        str(limit), str(threads)
    ],
}

RESULT_METHOD_NAMES = {
    "proposed": "proposed",
    "compact": "compact_milp",
    "time_indexed": "time_indexed_milp",
    "cp": "cp",
    "integer_lshaped": "integer_lshaped_milp",
    "integer_lshaped_milp": "integer_lshaped_milp",
    "integer_lshaped_cp": "integer_lshaped_cp",
    "component_dense_direct": "component_dense_direct",
    "component_sparse_direct": "component_sparse_direct",
    "component_label_dominance_batched": "component_label_dominance_batched",
    "component_terminal_dominance_batched": "component_terminal_dominance_batched",
    "component_dominance_direct": "component_dominance_direct",
    "component_dominance_batched": "component_dominance_batched",
}


def completed_pairs(path: Path) -> set[tuple[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="") as handle:
        return {(row["instance"], row["method"]) for row in csv.DictReader(handle)}


def failed_pairs(path: Path) -> set[tuple[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="") as handle:
        return {(row["instance"], row["method"]) for row in csv.DictReader(handle)}


def result_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_failure(path: Path, instance: str, method: str, reason: str,
                   seconds: float, return_code: int | None) -> None:
    add_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if add_header:
            writer.writerow(["instance", "method", "reason", "seconds", "return_code"])
        writer.writerow([instance, method, reason, f"{seconds:.6f}", return_code])


RESULT_FIELDS = [
    "instance", "class", "seed", "method", "status", "objective", "bound",
    "gap", "seconds", "best_s", "best_d", "n_scenarios", "n_committed",
    "n_potential", "min_jobs", "mean_jobs", "max_jobs", "pmax", "smax",
    "lambda", "mu", "time_limit", "threads", "nodes_or_branches",
    "variables", "constraints", "generated_labels", "peak_labels",
    "peak_memory_mb", "retained_vectors", "iterations", "label_seconds",
    "completed_scenarios",
    "contract_seconds", "max_scenario_seconds",
    "max_scenario_generated_labels", "max_scenario_vectors",
]


def append_limit_result(path: Path, row: dict[str, str], method: str,
                        status: str, seconds: float, time_limit: float,
                        threads: int, peak_memory_mb: float = 0.0) -> None:
    add_header = not path.exists() or path.stat().st_size == 0
    values = {
        "instance": row["instance"], "class": row["class"],
        "seed": row["seed"], "method": method, "status": status,
        "objective": "NA", "bound": "NA", "gap": "NA",
        "seconds": f"{seconds:.12g}", "best_s": -1, "best_d": -1,
        "n_scenarios": row["scenarios"], "n_committed": row["committed"],
        "n_potential": row["potential"], "min_jobs": row["min_jobs"],
        "mean_jobs": row["mean_jobs"], "max_jobs": row["max_jobs"],
        "pmax": row["pmax"], "smax": row["smax"],
        "lambda": row["lambda"], "mu": row["mu"],
        "time_limit": time_limit, "threads": threads,
        "peak_memory_mb": f"{peak_memory_mb:.6f}",
    }
    for field in RESULT_FIELDS:
        values.setdefault(field, 0)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if add_header:
            writer.writeheader()
        writer.writerow(values)


class MacProcessTaskInfo(ctypes.Structure):
    _fields_ = [
        ("virtual_size", ctypes.c_uint64), ("resident_size", ctypes.c_uint64),
        ("total_user", ctypes.c_uint64), ("total_system", ctypes.c_uint64),
        ("threads_user", ctypes.c_uint64), ("threads_system", ctypes.c_uint64),
        ("policy", ctypes.c_int32), ("faults", ctypes.c_int32),
        ("pageins", ctypes.c_int32), ("cow_faults", ctypes.c_int32),
        ("messages_sent", ctypes.c_int32),
        ("messages_received", ctypes.c_int32),
        ("syscalls_mach", ctypes.c_int32), ("syscalls_unix", ctypes.c_int32),
        ("csw", ctypes.c_int32), ("threadnum", ctypes.c_int32),
        ("numrunning", ctypes.c_int32), ("priority", ctypes.c_int32),
    ]


MAC_PROC_PIDINFO = None


def resident_bytes(pid: int) -> int | None:
    if sys.platform == "darwin":
        global MAC_PROC_PIDINFO
        if MAC_PROC_PIDINFO is None:
            library = ctypes.CDLL("/usr/lib/libproc.dylib")
            MAC_PROC_PIDINFO = library.proc_pidinfo
            MAC_PROC_PIDINFO.argtypes = [
                ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                ctypes.c_void_p, ctypes.c_int]
            MAC_PROC_PIDINFO.restype = ctypes.c_int
        information = MacProcessTaskInfo()
        received = MAC_PROC_PIDINFO(
            pid, 4, 0, ctypes.byref(information), ctypes.sizeof(information))
        return information.resident_size \
            if received == ctypes.sizeof(information) else None
    status = Path(f"/proc/{pid}/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return None


def run_with_limits(command: list[str], log_path: Path,
                    environment: dict[str, str], timeout: float,
                    memory_limit_gb: float) -> tuple[int | None, str | None, float]:
    memory_limit = int(memory_limit_gb * 1024 ** 3)
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=environment)
        deadline = time.monotonic() + timeout
        peak_resident = 0
        missing_samples = 0
        while True:
            return_code = process.poll()
            if return_code is not None:
                return return_code, None, peak_resident / 1024 ** 2
            if memory_limit > 0:
                resident = resident_bytes(process.pid)
                if resident is None:
                    return_code = process.poll()
                    if return_code is not None:
                        return return_code, None, peak_resident / 1024 ** 2
                    missing_samples += 1
                    if missing_samples >= 10:
                        process.kill()
                        process.wait()
                        raise RuntimeError("Unable to monitor solver memory usage")
                    time.sleep(0.01)
                    continue
                missing_samples = 0
                peak_resident = max(peak_resident, resident)
                if resident > memory_limit:
                    process.kill()
                    process.wait()
                    return (process.returncode, "memory_limit",
                            peak_resident / 1024 ** 2)
            if time.monotonic() >= deadline:
                process.kill()
                process.wait()
                return (process.returncode, "external_timeout",
                        peak_resident / 1024 ** 2)
            time.sleep(0.1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="instances/manifest_pilot.csv")
    parser.add_argument("--results", default="results/pilot_results.csv")
    parser.add_argument("--failures", default="results/pilot_failures.csv")
    parser.add_argument("--logs", default="logs/pilot")
    parser.add_argument("--methods", nargs="+",
                        default=["proposed", "time_indexed", "cp",
                                 "integer_lshaped_cp"])
    parser.add_argument("--run-order", choices=["instance-major", "method-major"],
                        default="instance-major")
    parser.add_argument("--time-limit", type=float, default=1200.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-limit-gb", type=float, default=0.0)
    parser.add_argument("--diagram-fallback-width", type=int, default=0)
    parser.add_argument("--diagram-time-limit", type=float)
    parser.add_argument("--max-instances", type=int)
    parser.add_argument("--classes", nargs="+")
    parser.add_argument("--target-jobs", nargs="+", type=int)
    parser.add_argument("--scenario-counts", nargs="+", type=int)
    parser.add_argument("--repetitions", nargs="+", type=int)
    parser.add_argument("--retry-failures", action="store_true")
    args = parser.parse_args()
    if args.diagram_fallback_width < 0:
        parser.error("--diagram-fallback-width must be nonnegative")
    if args.diagram_time_limit is not None and args.diagram_time_limit <= 0:
        parser.error("--diagram-time-limit must be positive")
    if args.memory_limit_gb < 0:
        parser.error("--memory-limit-gb must be nonnegative")

    root = Path(__file__).resolve().parent
    manifest = (root / args.manifest).resolve()
    results = (root / args.results).resolve()
    failures = (root / args.failures).resolve()
    logs = (root / args.logs).resolve()
    results.parent.mkdir(parents=True, exist_ok=True)
    failures.parent.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    methods = args.methods
    if methods == ["all"]:
        methods = ["proposed", "time_indexed", "cp",
                   "integer_lshaped_cp"]
    unknown = [method for method in methods if method not in METHOD_COMMANDS]
    if unknown:
        parser.error(f"unknown methods: {', '.join(unknown)}")

    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.classes:
        rows = [row for row in rows if row["class"] in args.classes]
    if args.target_jobs:
        rows = [row for row in rows if int(row["target_jobs"]) in args.target_jobs]
    if args.scenario_counts:
        rows = [row for row in rows if int(row["scenarios"]) in args.scenario_counts]
    if args.repetitions:
        rows = [row for row in rows if int(row["rep"]) in args.repetitions]
    if args.max_instances is not None:
        rows = rows[: args.max_instances]

    done = completed_pairs(results)
    failed = failed_pairs(failures)
    jobs: list[tuple[dict[str, str], str]] = []
    pairs = ((row, method) for row in rows for method in methods)
    if args.run_order == "method-major":
        pairs = ((row, method) for method in methods for row in rows)
    for row, method in pairs:
        recorded_method = RESULT_METHOD_NAMES[method]
        pair = (row["instance"], recorded_method)
        if pair in done:
            continue
        if pair in failed and not args.retry_failures:
            continue
        jobs.append((row, method))

    print(f"Scheduled {len(jobs)} runs from {len(rows)} instances", flush=True)
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    environment["LANG"] = "C"
    for number, (row, method) in enumerate(jobs, start=1):
        instance_path = (manifest.parent / row["file"]).resolve()
        log_path = logs / f"{row['instance']}__{method}.log"
        solver_log_path = logs / f"{row['instance']}__{method}.solver.log"
        command = METHOD_COMMANDS[method](
            root, instance_path, results, args.time_limit, args.threads,
            solver_log_path)
        command = [str(item) for item in command]
        print(f"[{number}/{len(jobs)}] {row['instance']} {method}", flush=True)
        started = time.monotonic()
        try:
            if method.startswith("component_") and args.memory_limit_gb > 0:
                return_code, stopped, peak_memory_mb = run_with_limits(
                    command, log_path, environment, args.time_limit + 30.0,
                    args.memory_limit_gb)
            else:
                with log_path.open("w") as log:
                    completed = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT,
                        env=environment, timeout=args.time_limit + 30.0,
                        check=False)
                return_code, stopped, peak_memory_mb = completed.returncode, None, 0.0
            elapsed = time.monotonic() - started
            if stopped == "memory_limit":
                append_limit_result(
                    results, row, RESULT_METHOD_NAMES[method], "memory_limit",
                    elapsed, args.time_limit, args.threads, peak_memory_mb)
                print(
                    f"  stopped at the {args.memory_limit_gb:g} GB memory limit "
                    f"after {elapsed:.2f} s", flush=True)
            elif stopped == "external_timeout":
                append_failure(
                    failures, row["instance"], RESULT_METHOD_NAMES[method],
                    "external_timeout", elapsed, return_code)
                print(
                    f"  killed after external timeout ({elapsed:.2f} s)",
                    flush=True)
            elif return_code != 0:
                append_failure(failures, row["instance"], RESULT_METHOD_NAMES[method],
                               "nonzero_exit", elapsed, return_code)
                print(f"  failed with return code {return_code}", flush=True)
            else:
                print(f"  completed in {elapsed:.2f} s", flush=True)
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - started
            append_failure(failures, row["instance"], RESULT_METHOD_NAMES[method],
                           "external_timeout", elapsed, None)
            print(f"  killed after external timeout ({elapsed:.2f} s)", flush=True)
        except RuntimeError as error:
            elapsed = time.monotonic() - started
            append_failure(failures, row["instance"], RESULT_METHOD_NAMES[method],
                           "memory_monitor_error", elapsed, None)
            print(f"  memory monitor failed: {error}", flush=True)

    if args.diagram_fallback_width > 0 and "proposed" in methods:
        width = args.diagram_fallback_width
        diagram_method = f"diagram_w{width}"
        diagram_limit = (args.diagram_time_limit
                         if args.diagram_time_limit is not None
                         else args.time_limit)
        recorded = result_rows(results)
        proposed_status = {
            row["instance"]: row["status"]
            for row in recorded if row["method"] == "proposed"
        }
        diagram_status = {
            row["instance"]: row["status"]
            for row in recorded if row["method"] == diagram_method
        }
        proposed_failures = {
            instance for instance, method in failed_pairs(failures)
            if method == "proposed"
        }
        fallback_rows = []
        for row in rows:
            status = proposed_status.get(row["instance"])
            exact_failed = status in {"time_limit", "memory_limit"}
            exact_failed = exact_failed or row["instance"] in proposed_failures
            if not exact_failed:
                continue
            prior = diagram_status.get(row["instance"])
            if prior is not None and not (
                    args.retry_failures and prior in {"time_limit", "memory_limit"}):
                continue
            fallback_rows.append(row)

        print(
            f"Scheduled {len(fallback_rows)} width-{width} diagram fallbacks",
            flush=True)
        for number, row in enumerate(fallback_rows, start=1):
            instance_path = (manifest.parent / row["file"]).resolve()
            log_path = logs / f"{row['instance']}__{diagram_method}.log"
            command = [
                str(root / "bin/proposed_solver"), str(instance_path),
                str(results), str(diagram_limit), str(args.threads), str(width),
            ]
            print(
                f"[fallback {number}/{len(fallback_rows)}] "
                f"{row['instance']} {diagram_method}", flush=True)
            started = time.monotonic()
            try:
                with log_path.open("w") as log:
                    completed = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT,
                        env=environment, timeout=diagram_limit + 30.0,
                        check=False)
                elapsed = time.monotonic() - started
                if completed.returncode != 0:
                    append_failure(
                        failures, row["instance"], diagram_method,
                        "nonzero_exit", elapsed, completed.returncode)
                    print(
                        f"  fallback failed with return code "
                        f"{completed.returncode}", flush=True)
                else:
                    print(f"  fallback completed in {elapsed:.2f} s", flush=True)
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - started
                append_failure(
                    failures, row["instance"], diagram_method,
                    "external_timeout", elapsed, None)
                print(
                    f"  fallback killed after external timeout "
                    f"({elapsed:.2f} s)", flush=True)

    print(f"Results: {results}")
    print(f"Failures: {failures}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
