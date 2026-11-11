# Scheduling with Advance Booking of an External Resource under Job-Set Uncertainty

Replication files for the computational experiments in the manuscript of the same title. The study compares an exact label-setting algorithm with time-indexed MILP, CP Optimizer, and integer L-shaped baselines. It also tests the algorithm's components and the sensitivity of the contract to resource prices and the fraction of committed jobs.

## Build and quick check

Use Make, a C++20 compiler, and Python 3.10 or later. The Python scripts use the standard library.

```sh
make
python3 experiment.py --name smoke --target-jobs 10 --scenario-counts 2 \
  --processing-max 25 --resource-share 0.25 --repetitions 1
python3 report_tables.py
```

The last command reconstructs the manuscript's experimental tables from the archived results and writes CSVs to `runs/tables/`. The `committed.csv` output also gives the data for the committed-fraction figure.

To build the baselines, use Gurobi 13.0.2 and IBM ILOG CPLEX Optimization Studio 22.1.1 with CP Optimizer and valid local solver licenses.

```sh
make baselines
```

The Makefile defaults match the original macOS ARM installations. If the installations are elsewhere, set `GUROBI_HOME` and `CPLEX_HOME` in the environment. You can also override `GUROBI_VERSION`, `CP_PLATFORM`, and the include and link flags in the Makefile. The exact algorithm, component study, and sensitivity calculations build without either commercial solver. The reported runs used Apple clang 14.0.3 with `-O3`, one solver thread, and a 1,800-second limit per instance. The component study also imposed a 24 GiB resident-memory limit.

## Archived instances and results

| Directory | Contents |
| --- | --- |
| `data/scalability/` | 360 instances, exact solver records, and the empty-set lower bounds |
| `data/baselines/` | 40 matched instances for 1, 2, 5, and 10 scenarios, stored in the three original run batches, with the solver records and separate empty-set lower bounds |
| `data/components/` | 20 four-method comparisons at 10 expected jobs and 10 two-method comparisons at 60 expected jobs |
| `data/premium/` | Five overtime-rate settings on the same ten instances, including the committed-only cost decomposition at the base rate |
| `data/committed/` | Ten paired replications at each of the six committed fractions |
| `data/price/` | The booking-rate sweep at a fixed premium ratio using the inputs in `data/premium/mu30/` |

Each manifest links a run record to the original solver input in its `instances/` directory. The baseline comparison matches records by seed and scenario count. Its reported gaps use the larger of the archived solver bound and the empty-set lower bound, as in the manuscript. Reruns can vary in runtime and in the bounds reached by the time limit.

## Rerun the experiments

For an exact replay of the saved instance data, pass a manifest to `run_suite.py`. For example, the following reruns the first scalability instance. Remove `--max-instances 1` to run the full scalability set.

```sh
python3 run_suite.py --manifest data/scalability/instances/manifest_custom.csv \
  --methods proposed --max-instances 1 --time-limit 1800 --threads 1 \
  --results runs/replay/scalability.csv --failures runs/replay/failures.csv \
  --logs runs/replay/logs
```

To regenerate the designs and run them, use the corresponding JSON configurations. The C++ random-number distributions can differ across standard libraries, so use the archived instances when comparing results on identical inputs.

```sh
python3 experiment.py --config configs/scalability.json
python3 experiment.py --config configs/baselines.json
python3 experiment.py --config configs/component_hierarchy.json
python3 experiment.py --config configs/component_evaluation.json
```

To compute the empty-set lower bound for an individual instance, run `bin/empty_set_lower_bound INSTANCE.dat OUTPUT.csv` after `make`.

The overtime-premium experiment has a separate configuration at the base rate. The following loop uses the same seed and job-generation settings at each rate.

```sh
for mu in 15 20 30 40 60; do
  python3 experiment.py --config configs/premium.json --name "premium_mu${mu}" --mu-rate "$mu"
done
python3 sensitivity/run_sensitivity.py metrics \
  --run-dirs runs/premium_mu15 runs/premium_mu20 runs/premium_mu30 runs/premium_mu40 runs/premium_mu60 \
  --output runs/premium_metrics.csv --summary runs/premium_summary.csv
```

To rerun the paired committed-fraction experiment on the original inputs and the price-level sweep on the base premium inputs, use

```sh
mkdir -p runs/committed
cp -R data/committed/instances runs/committed/
python3 sensitivity/run_sensitivity.py paired --run-dir runs/committed
python3 sensitivity/run_price_sensitivity.py --base-run data/premium/mu30 \
  --run-dir runs/price --jobs 1
```

All commands are run from the repository root. Generated outputs and logs go under `runs/`.
