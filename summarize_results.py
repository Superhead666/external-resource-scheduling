#!/usr/bin/env python3
"""Create compact solved-count, time, and gap summaries from benchmark CSV output."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import re


def number(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def target_size(instance: str) -> int:
    matches = re.findall(r"_n(\d+)_", instance)
    return int(matches[-1]) if matches else -1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="results/pilot_results.csv")
    parser.add_argument("--output", default="results/pilot_summary.csv")
    parser.add_argument(
        "--instance-output",
        help="optional CSV containing every run and its deviation from the proposed optimum")
    args = parser.parse_args()
    source = Path(args.results)
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    reference: dict[str, float] = {}
    for row in rows:
        objective = number(row.get("objective", ""))
        if (row.get("method") == "proposed"
                and row.get("status") == "optimal"
                and math.isfinite(objective)):
            reference[row["instance"]] = objective
    for row in rows:
        objective = number(row.get("objective", ""))
        if row.get("status") == "optimal" and math.isfinite(objective):
            reference.setdefault(row["instance"], objective)

    deviations: dict[int, float] = {}
    for index, row in enumerate(rows):
        objective = number(row.get("objective", ""))
        optimum = reference.get(row["instance"], math.nan)
        if math.isfinite(objective) and math.isfinite(optimum):
            deviations[index] = max(0.0, (objective - optimum)
                                    / max(1.0, abs(optimum)))

    if args.instance_output:
        instance_output = Path(args.instance_output)
        instance_output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys()) if rows else []
        fieldnames.extend(["reference_objective", "incumbent_deviation"])
        with instance_output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            order = sorted(range(len(rows)), key=lambda index: (
                rows[index]["instance"], rows[index]["method"]))
            for index in order:
                output_row = dict(rows[index])
                optimum = reference.get(rows[index]["instance"], math.nan)
                output_row["reference_objective"] = (
                    optimum if math.isfinite(optimum) else "NA")
                output_row["incumbent_deviation"] = deviations.get(index, "NA")
                writer.writerow(output_row)

    groups: dict[tuple[str, str, int, int], list[dict[str, str]]] = {}
    for index, row in enumerate(rows):
        row["_row_index"] = str(index)
        key = (row["method"], row["class"], target_size(row["instance"]),
               int(row["n_scenarios"]))
        groups.setdefault(key, []).append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "class", "target_jobs", "scenarios", "instances",
                         "optimal", "feasible", "mean_seconds", "median_seconds",
                         "mean_gap", "max_gap", "mean_incumbent_deviation",
                         "max_incumbent_deviation", "mean_peak_labels",
                         "max_peak_labels", "mean_peak_memory_mb",
                         "max_peak_memory_mb", "mean_generated_labels",
                         "mean_retained_vectors_solved",
                         "mean_label_seconds_solved",
                         "mean_contract_seconds_solved",
                         "mean_contract_grid",
                         "mean_completed_scenarios", "max_scenario_seconds",
                         "time_limit", "memory_limit", "mean_seconds_solved",
                         "max_seconds_solved"])
        for key in sorted(groups):
            values = groups[key]
            times = sorted(number(row["seconds"]) for row in values
                           if math.isfinite(number(row["seconds"])))
            gaps = [number(row["gap"]) for row in values
                    if math.isfinite(number(row["gap"]))]
            optimal = sum(row["status"] == "optimal" for row in values)
            solved_times = [number(row["seconds"]) for row in values
                            if row["status"] == "optimal"
                            and math.isfinite(number(row["seconds"]))]
            feasible = sum(row["objective"] != "NA" for row in values)
            mean_time = sum(times) / len(times) if times else math.nan
            if not times:
                median_time = math.nan
            elif len(times) % 2:
                median_time = times[len(times) // 2]
            else:
                median_time = 0.5 * (times[len(times) // 2 - 1]
                                     + times[len(times) // 2])
            mean_gap = sum(gaps) / len(gaps) if gaps else math.nan
            max_gap = max(gaps) if gaps else math.nan
            incumbent_deviations = [
                deviations[int(row["_row_index"])] for row in values
                if int(row["_row_index"]) in deviations]
            peaks = [number(row.get("peak_labels", "")) for row in values]
            peaks = [value for value in peaks if math.isfinite(value)]
            generated = [number(row.get("generated_labels", "")) for row in values]
            generated = [value for value in generated if math.isfinite(value)]
            memory = [number(row.get("peak_memory_mb", "")) for row in values]
            memory = [value for value in memory if math.isfinite(value)]
            solved_values = [row for row in values if row["status"] == "optimal"]
            retained = [number(row.get("retained_vectors", ""))
                        for row in solved_values]
            retained = [value for value in retained if math.isfinite(value)]
            label_times = [number(row.get("label_seconds", ""))
                           for row in solved_values]
            label_times = [value for value in label_times
                           if math.isfinite(value)]
            contract_times = [number(row.get("contract_seconds", ""))
                              for row in solved_values]
            contract_times = [value for value in contract_times
                              if math.isfinite(value)]
            grids = [(number(row.get("pmax", "")) + 1.0)
                     * (number(row.get("smax", "")) + 1.0)
                     for row in values]
            grids = [value for value in grids if math.isfinite(value)]
            completed = [
                number(row.get("completed_scenarios", row.get("iterations", "")))
                for row in values]
            completed = [value for value in completed if math.isfinite(value)]
            scenario_times = [
                number(row.get("max_scenario_seconds", "")) for row in values]
            scenario_times = [
                value for value in scenario_times if math.isfinite(value)]
            writer.writerow([*key, len(values), optimal, feasible, mean_time,
                             median_time, mean_gap, max_gap,
                             sum(incumbent_deviations) / len(incumbent_deviations)
                             if incumbent_deviations else math.nan,
                             max(incumbent_deviations)
                             if incumbent_deviations else math.nan,
                             sum(peaks) / len(peaks) if peaks else math.nan,
                             max(peaks) if peaks else math.nan,
                             sum(memory) / len(memory) if memory else math.nan,
                             max(memory) if memory else math.nan,
                             sum(generated) / len(generated)
                             if generated else math.nan,
                             sum(retained) / len(retained)
                             if retained else math.nan,
                             sum(label_times) / len(label_times)
                             if label_times else math.nan,
                             sum(contract_times) / len(contract_times)
                             if contract_times else math.nan,
                             sum(grids) / len(grids) if grids else math.nan,
                             sum(completed) / len(completed)
                             if completed else math.nan,
                             max(scenario_times) if scenario_times else math.nan,
                             sum(row["status"].startswith("time_limit")
                                 for row in values),
                             sum(row["status"].startswith("memory_limit")
                                 for row in values),
                             sum(solved_times) / len(solved_times)
                             if solved_times else math.nan,
                             max(solved_times) if solved_times else math.nan])

    by_instance: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row["status"] == "optimal":
            by_instance.setdefault(row["instance"], []).append(row)
    mismatches = []
    for instance, values in by_instance.items():
        objectives = [number(row["objective"]) for row in values]
        if objectives and max(objectives) - min(objectives) > 1e-5 * max(1.0, abs(min(objectives))):
            mismatches.append((instance, [(row["method"], row["objective"])
                                          for row in values]))
    print(f"Read {len(rows)} result rows")
    print(f"Wrote {output}")
    if args.instance_output:
        print(f"Wrote {Path(args.instance_output)}")
    if mismatches:
        print("WARNING: optimal-objective mismatches detected")
        for mismatch in mismatches:
            print(mismatch)
        return 2
    print("All shared optimal objectives agree within tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
