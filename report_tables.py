#!/usr/bin/env python3
"""Reconstruct the manuscript tables from the archived CSVs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from statistics import fmean

DATA = Path(__file__).resolve().parent / "data"


def read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, columns, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def mean(rows, column):
    return fmean(float(row[column]) for row in rows)


def groups(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[key(row)].append(row)
    return sorted(grouped.items())


def load_results(directory):
    manifest = next((directory / "instances").glob("manifest_*.csv"))
    settings = {row["instance"]: row for row in read(manifest)}
    return [settings[row["instance"]] | row for row in read(directory / "results.csv")]


def size_profile(row):
    return int(row["target_jobs"]), int(row["processing_max"]), float(row["resource_share"])


def scalability(output):
    folder = DATA / "scalability"
    bounds = {row["instance"]: float(row["lower_bound"])
              for row in read(folder / "empty_set_lower_bounds.csv")}
    runs = load_results(folder)
    ratios = {row["instance"]: 100 * bounds[row["instance"]] / float(row["objective"])
              for row in runs}
    scenarios = sorted({int(row["n_scenarios"]) for row in runs})
    columns = ["expected_jobs", "p_max", "resource_share"]
    for k in scenarios:
        columns.extend([f"Avg.Time_K{k}", f"Max.Time_K{k}", f"Avg.Labels_K{k}"])
    columns.append("Avg.LB")

    def line(profile, rows):
        values = list(profile)
        for _, batch in groups(rows, lambda row: int(row["n_scenarios"])):
            values.extend([f'{mean(batch, "seconds"):.1f}',
                           f'{max(float(row["seconds"]) for row in batch):.1f}',
                           f'{mean(batch, "peak_labels") / 1000:.1f}'])
        values.append(f'{fmean(ratios[row["instance"]] for row in rows):.2f}')
        return values

    lines = [line(profile, batch) for profile, batch in groups(runs, size_profile)]
    lines.append(line(("Overall", "", ""), runs))
    write(output / "scalability.csv", columns, lines)


def comparison(output):
    folder = DATA / "baselines"
    runs = [row for part in sorted(folder.glob("part_*")) for row in load_results(part)]
    def key(row):
        return int(row["n_scenarios"]), int(row["seed"])

    exact = {key(row): float(row["objective"]) for row in runs
             if row["method"] == "proposed" and row["status"] == "optimal"}
    bounds = {key(row): float(row["lower_bound"])
              for row in read(folder / "empty_set_lower_bounds.csv")}
    lines = []
    for (k, method), rows in groups(
            [row for row in runs if row["method"] != "proposed"],
            lambda row: (int(row["n_scenarios"]), row["method"])):
        unsolved = [row for row in rows if row["status"] != "optimal"]
        deviations = []
        gaps = []
        for row in unsolved:
            incumbent = float(row["objective"])
            optimum = exact[key(row)]
            bound = max(float(row["bound"]), bounds[key(row)])
            deviations.append(100 * (incumbent - optimum) / optimum)
            gaps.append(100 * (incumbent - bound) / incumbent)
        seconds = fmean(float(row["seconds"] if row["status"] == "optimal"
                              else row["time_limit"]) for row in rows)
        lines.append([k, method, len(rows) - len(unsolved),
                      "TL" if len(unsolved) == len(rows) else f"{seconds:.2f}",
                      f"{fmean(deviations):.3f}" if deviations else "--",
                      f"{fmean(gaps):.3f}" if gaps else "--"])
    write(output / "baselines.csv",
          ["scenarios", "method", "No.Opt", "Avg.Time", "Avg.Dev", "Avg.Gap"], lines)


def components(output):
    folder = DATA / "components"
    runs = load_results(folder / "hierarchy") + load_results(folder / "evaluation")
    order = ["component_dense_direct", "component_sparse_direct",
             "component_dominance_direct", "component_dominance_batched"]
    lines = []
    for (n, p_max, method), rows in groups(
            runs, lambda row: (int(row["target_jobs"]), int(row["processing_max"]),
                               order.index(row["method"]))):
        lines.append([n, p_max, order[method], f'{mean(rows, "seconds"):.2f}',
                      f'{mean(rows, "label_seconds"):.2f}',
                      f'{mean(rows, "contract_seconds"):.2f}',
                      f'{mean(rows, "peak_memory_mb"):.1f}',
                      f'{mean(rows, "retained_vectors") / 1000:.1f}'])
    write(output / "components.csv",
          ["expected_jobs", "p_max", "method", "Avg.Time", "Gen.Time", "Eval.Time",
           "Avg.Memory", "Avg.Vectors"], lines)


def sensitivity(output):
    def conditional_overtime(row):
        probability = float(row["overtime_probability"])
        return float(row["expected_overtime"]) / probability if probability else 0.0

    def overtime_percentage(rows):
        probability = sum(Decimal(row["overtime_probability"]) for row in rows)
        percentage = 100 * probability / len(rows)
        return percentage.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

    for family, level, outcome, heading, scale in (
        ("premium", "mu", "booking_total_share", "booking_share_pct", 100),
        ("committed", "n_committed", "committed_deviation_pct", "committed_loss_pct", 1),
    ):
        lines = []
        for level_value, rows in groups(read(DATA / family / "sensitivity_metrics.csv"),
                                         lambda row: int(row[level])):
            lines.append([
                level_value, f'{mean(rows, "opt_s"):.1f}',
                f'{100 * fmean(float(r["opt_d"]) / float(r["expected_processing"]) for r in rows):.1f}',
                overtime_percentage(rows),
                f'{fmean(map(conditional_overtime, rows)):.1f}',
                f'{mean(rows, outcome) * scale:.2f}',
            ])
        write(output / f"{family}.csv",
              [level, "opt_s", "booking_duration_pct", "overtime_probability_pct",
               "conditional_overtime", heading], lines)

    price_rows = []
    for rate, rows in groups(read(DATA / "price/price_metrics.csv"),
                             lambda row: int(row["lambda"])):
        resource_share = fmean(
            (float(row["booking_cost"]) + float(row["mu"]) * float(row["expected_overtime"]))
            / float(row["objective"]) for row in rows
        )
        price_rows.append([rate, f'{mean(rows, "opt_s"):.1f}',
                           f'{100 * mean(rows, "booking_total_share"):.2f}',
                           f'{100 * resource_share:.2f}'])
    write(output / "price.csv",
          ["lambda", "opt_s", "booking_share_pct", "resource_cost_share_pct"], price_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tables"))
    output = parser.parse_args().output_dir
    output.mkdir(parents=True, exist_ok=True)
    scalability(output)
    comparison(output)
    components(output)
    sensitivity(output)


if __name__ == "__main__":
    main()
