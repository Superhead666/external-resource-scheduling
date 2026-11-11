#include "instance.hpp"

// Reuse the extensively validated implementation developed during the
// preliminary structural tests. Renaming its command-line entry point keeps the
// benchmark wrapper and the validation program in one translation unit.
#define main frontier_validation_program_main
#include "frontier_labeling_core.inc"
#undef main

#include <atomic>
#include <mutex>

namespace {

struct ScenarioOutput {
    std::vector<erb::i64> table;
    LabelStats statistics;
    std::size_t vectors = 0;
    double table_seconds = 0.0;
    double solve_seconds = 0.0;
};

struct ScenarioBounds {
    ScenarioOutput upper;
    ScenarioOutput lower;
};

std::vector<erb::i64> rectangular_recourse_table(
        const std::vector<Descriptor>& descriptors, int scenario_processing,
        int maximum_arrival, int maximum_duration, int overtime_rate,
        const std::chrono::steady_clock::time_point& deadline) {
    const int duration_count = maximum_duration + 1;
    std::vector<erb::i64> table(
        static_cast<std::size_t>(maximum_arrival + 1) * duration_count, INF);
    for (int s = 0; s <= maximum_arrival; ++s) {
        if (std::chrono::steady_clock::now() >= deadline) {
            throw std::runtime_error("time_limit");
        }
        std::vector<erb::i64> exact_duration(
            static_cast<std::size_t>(scenario_processing + 1), INF);
        for (const Descriptor& descriptor : descriptors) {
            const int idle = std::max(0, s - descriptor.before_first_r);
            const int resource_duration = descriptor.through_last_r + idle - s;
            const erb::i64 completion_cost = descriptor.twc_without_idle
                + static_cast<erb::i64>(descriptor.weight_after_first_r) * idle;
            exact_duration[resource_duration] = std::min(
                exact_duration[resource_duration], completion_cost);
        }
        std::vector<erb::i64> prefix(scenario_processing + 1, INF);
        std::vector<erb::i64> suffix(scenario_processing + 2, INF);
        erb::i64 best = INF;
        for (int duration = 0; duration <= scenario_processing; ++duration) {
            best = std::min(best, exact_duration[duration]);
            prefix[duration] = best;
        }
        best = INF;
        for (int duration = scenario_processing; duration >= 0; --duration) {
            if (exact_duration[duration] < INF / 2) {
                best = std::min(best, exact_duration[duration]
                    + static_cast<erb::i64>(overtime_rate) * duration);
            }
            suffix[duration] = best;
        }
        for (int booked = 0; booked <= maximum_duration; ++booked) {
            const int capped = std::min(booked, scenario_processing);
            erb::i64 value = prefix[capped];
            if (booked < scenario_processing && suffix[booked + 1] < INF / 2) {
                value = std::min(value, suffix[booked + 1]
                    - static_cast<erb::i64>(overtime_rate) * booked);
            }
            table[static_cast<std::size_t>(s) * duration_count + booked] = value;
        }
    }
    return table;
}

ScenarioOutput solve_scenario(const erb::Instance& instance,
                              const erb::Scenario& scenario,
                              int maximum_arrival, int maximum_duration,
                              const std::chrono::steady_clock::time_point& deadline) {
    const auto scenario_start = std::chrono::steady_clock::now();
    std::vector<Job> jobs;
    jobs.reserve(scenario.jobs.size());
    int processing = 0;
    for (int id : scenario.jobs) {
        const erb::Job& source = instance.jobs[id];
        jobs.push_back({source.id, source.processing, source.weight, source.resource});
        processing += source.processing;
    }
    ScenarioOutput output;
    std::vector<Descriptor> descriptors = strongest_exact_descriptors(
        jobs, &output.statistics, &deadline);
    output.vectors = descriptors.size();
    const auto table_start = std::chrono::steady_clock::now();
    output.table = rectangular_recourse_table(
        descriptors, processing, maximum_arrival, maximum_duration,
        instance.mu, deadline);
    output.table_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - table_start).count();
    output.solve_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - scenario_start).count();
    return output;
}

void accumulate_statistics(erb::SolveRecord& record,
                           const ScenarioOutput& output) {
    record.generated_labels += output.statistics.generated_states;
    record.peak_labels = std::max(record.peak_labels,
                                  output.statistics.peak_states);
    record.retained_vectors += output.vectors;
    record.label_seconds += output.statistics.seconds;
    record.contract_seconds += output.table_seconds;
    record.maximum_scenario_seconds = std::max(
        record.maximum_scenario_seconds, output.solve_seconds);
    record.maximum_scenario_generated_labels = std::max(
        record.maximum_scenario_generated_labels,
        output.statistics.generated_states);
    record.maximum_scenario_vectors = std::max(
        record.maximum_scenario_vectors, output.vectors);
}

ScenarioOutput solve_scenario_diagram(
        const erb::Instance& instance, const erb::Scenario& scenario,
        int maximum_arrival, int maximum_duration, DiagramMode mode,
        std::size_t width,
        const std::chrono::steady_clock::time_point& deadline) {
    std::vector<Job> jobs;
    jobs.reserve(scenario.jobs.size());
    int processing = 0;
    for (int id : scenario.jobs) {
        const erb::Job& source = instance.jobs[id];
        jobs.push_back({source.id, source.processing, source.weight, source.resource});
        processing += source.processing;
    }
    ScenarioOutput output;
    // A restricted diagram retains feasible labels and therefore gives an
    // upper bound.  A relaxed diagram replaces discarded labels by a component-
    // wise optimistic label and gives a lower bound.  Both use the exact boundary
    // and WSPT-span results, which do not depend on the diagram width.
    const std::vector<Descriptor> descriptors = labeling_descriptors(
        jobs, &output.statistics, true, mode, width, 1, false,
        true, true, true, &deadline);
    output.vectors = descriptors.size();
    const auto table_start = std::chrono::steady_clock::now();
    output.table = rectangular_recourse_table(
        descriptors, processing, maximum_arrival, maximum_duration,
        instance.mu, deadline);
    output.table_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - table_start).count();
    output.solve_seconds = output.statistics.seconds + output.table_seconds;
    return output;
}

erb::SolveRecord solve(const erb::Instance& instance, double time_limit,
                       int threads) {
    const auto start = std::chrono::steady_clock::now();
    const auto deadline = start + std::chrono::milliseconds(
        static_cast<long long>(1000.0 * time_limit));
    const int maximum_arrival = erb::max_ordinary_processing(instance);
    const int maximum_duration = erb::max_processing(instance);
    const int duration_count = maximum_duration + 1;
    const std::size_t contract_count = static_cast<std::size_t>(maximum_arrival + 1)
        * duration_count;

    const int worker_count = std::max(1, std::min(
        threads, static_cast<int>(instance.scenarios.size())));
    std::vector<std::vector<double>> worker_expectations(
        worker_count, std::vector<double>(contract_count, 0.0));
    std::vector<erb::SolveRecord> worker_records(worker_count);
    std::atomic<std::size_t> next{0};
    std::atomic<bool> failed{false};
    std::exception_ptr exception;
    std::mutex exception_mutex;
    auto worker = [&](int worker_index) {
        try {
            std::vector<double>& expected = worker_expectations[worker_index];
            erb::SolveRecord& worker_record = worker_records[worker_index];
            while (!failed.load()) {
                const std::size_t index = next.fetch_add(1);
                if (index >= instance.scenarios.size()) break;
                ScenarioOutput output = solve_scenario(
                    instance, instance.scenarios[index], maximum_arrival,
                    maximum_duration, deadline);
                const double probability = instance.scenarios[index].probability;
                for (std::size_t contract = 0; contract < contract_count;
                     ++contract) {
                    expected[contract] += probability
                        * static_cast<double>(output.table[contract]);
                }
                accumulate_statistics(worker_record, output);
                ++worker_record.iterations;
                ++worker_record.completed_scenarios;
            }
        } catch (...) {
            failed.store(true);
            std::lock_guard<std::mutex> guard(exception_mutex);
            if (!exception) exception = std::current_exception();
        }
    };
    if (worker_count == 1) {
        worker(0);
    } else {
        std::vector<std::thread> workers;
        for (int k = 0; k < worker_count; ++k) workers.emplace_back(worker, k);
        for (std::thread& thread : workers) thread.join();
    }
    const auto scenarios_done = std::chrono::steady_clock::now();

    erb::SolveRecord record;
    record.method = "proposed";
    for (const erb::SolveRecord& worker_record : worker_records) {
        record.generated_labels += worker_record.generated_labels;
        record.peak_labels = std::max(record.peak_labels,
                                      worker_record.peak_labels);
        record.retained_vectors += worker_record.retained_vectors;
        record.iterations += worker_record.iterations;
        record.completed_scenarios += worker_record.completed_scenarios;
        record.label_seconds += worker_record.label_seconds;
        record.contract_seconds += worker_record.contract_seconds;
        record.maximum_scenario_seconds = std::max(
            record.maximum_scenario_seconds,
            worker_record.maximum_scenario_seconds);
        record.maximum_scenario_generated_labels = std::max(
            record.maximum_scenario_generated_labels,
            worker_record.maximum_scenario_generated_labels);
        record.maximum_scenario_vectors = std::max(
            record.maximum_scenario_vectors,
            worker_record.maximum_scenario_vectors);
    }
    if (exception) {
        try {
            std::rethrow_exception(exception);
        } catch (const std::bad_alloc&) {
            record.status = "memory_limit";
        } catch (const std::runtime_error& error) {
            if (std::string(error.what()) != "time_limit") throw;
            record.status = "time_limit";
        }
        record.seconds = std::chrono::duration<double>(
            scenarios_done - start).count();
        return record;
    }

    std::vector<double> expected = std::move(worker_expectations.front());
    for (int worker_index = 1; worker_index < worker_count; ++worker_index) {
        for (std::size_t index = 0; index < contract_count; ++index) {
            expected[index] += worker_expectations[worker_index][index];
        }
    }
    double optimum = std::numeric_limits<double>::infinity();
    for (int s = 0; s <= maximum_arrival; ++s) {
        for (int d = 0; d <= maximum_duration; ++d) {
            const std::size_t index = static_cast<std::size_t>(s) * duration_count + d;
            const double value = static_cast<double>(instance.lambda) * d
                + expected[index];
            if (value < optimum - 1e-9) {
                optimum = value;
                record.best_s = s;
                record.best_d = d;
            }
        }
    }
    const auto stop = std::chrono::steady_clock::now();
    record.status = "optimal";
    record.objective = optimum;
    record.bound = optimum;
    record.gap = 0.0;
    record.seconds = std::chrono::duration<double>(stop - start).count();
    record.contract_seconds += std::chrono::duration<double>(
        stop - scenarios_done).count();
    return record;
}

erb::SolveRecord solve_with_diagram_bounds(const erb::Instance& instance,
                                           double time_limit, int threads,
                                           std::size_t width) {
    if (width == 0) throw std::runtime_error("Diagram width must be positive");
    const auto start = std::chrono::steady_clock::now();
    const auto deadline = start + std::chrono::milliseconds(
        static_cast<long long>(1000.0 * time_limit));
    const int maximum_arrival = erb::max_ordinary_processing(instance);
    const int maximum_duration = erb::max_processing(instance);
    const int duration_count = maximum_duration + 1;
    const std::size_t contract_count = static_cast<std::size_t>(maximum_arrival + 1)
        * duration_count;

    std::vector<ScenarioBounds> outputs(instance.scenarios.size());
    std::atomic<std::size_t> next{0};
    std::atomic<bool> failed{false};
    std::exception_ptr exception;
    std::mutex exception_mutex;
    auto worker = [&]() {
        try {
            while (!failed.load()) {
                const std::size_t index = next.fetch_add(1);
                if (index >= instance.scenarios.size()) break;
                outputs[index].upper = solve_scenario_diagram(
                    instance, instance.scenarios[index], maximum_arrival,
                    maximum_duration, DiagramMode::Restricted, width, deadline);
                outputs[index].lower = solve_scenario_diagram(
                    instance, instance.scenarios[index], maximum_arrival,
                    maximum_duration, DiagramMode::Relaxed, width, deadline);
            }
        } catch (...) {
            failed.store(true);
            std::lock_guard<std::mutex> guard(exception_mutex);
            if (!exception) exception = std::current_exception();
        }
    };
    const int worker_count = std::max(1, std::min(
        threads, static_cast<int>(instance.scenarios.size())));
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (int k = 0; k < worker_count; ++k) workers.emplace_back(worker);
    for (std::thread& thread : workers) thread.join();
    if (exception) std::rethrow_exception(exception);

    std::vector<double> expected_upper(contract_count, 0.0);
    std::vector<double> expected_lower(contract_count, 0.0);
    erb::SolveRecord record;
    record.method = "diagram_w" + std::to_string(width);
    for (std::size_t omega = 0; omega < outputs.size(); ++omega) {
        const double probability = instance.scenarios[omega].probability;
        for (std::size_t index = 0; index < contract_count; ++index) {
            expected_upper[index] += probability
                * static_cast<double>(outputs[omega].upper.table[index]);
            expected_lower[index] += probability
                * static_cast<double>(outputs[omega].lower.table[index]);
        }
        for (const ScenarioOutput* output :
             {&outputs[omega].upper, &outputs[omega].lower}) {
            accumulate_statistics(record, *output);
        }
    }

    double upper = std::numeric_limits<double>::infinity();
    double lower = std::numeric_limits<double>::infinity();
    for (int s = 0; s <= maximum_arrival; ++s) {
        for (int d = 0; d <= maximum_duration; ++d) {
            const std::size_t index = static_cast<std::size_t>(s) * duration_count + d;
            const double booking = static_cast<double>(instance.lambda) * d;
            const double upper_value = booking + expected_upper[index];
            const double lower_value = booking + expected_lower[index];
            if (upper_value < upper - 1e-9) {
                upper = upper_value;
                record.best_s = s;
                record.best_d = d;
            }
            lower = std::min(lower, lower_value);
        }
    }
    record.objective = upper;
    record.bound = lower;
    record.gap = std::max(0.0, (upper - lower) / std::max(1.0, std::abs(upper)));
    record.status = record.gap <= 1e-12 ? "optimal" : "feasible";
    record.seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    record.iterations = static_cast<int>(2 * instance.scenarios.size());
    record.completed_scenarios = static_cast<int>(instance.scenarios.size());
    return record;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 6) {
        std::cerr << "Usage: proposed_solver INSTANCE OUTPUT_CSV [TIME_LIMIT] "
                     "[THREADS] [DIAGRAM_WIDTH]\n";
        return 1;
    }
    const double time_limit = argc >= 4 ? std::stod(argv[3]) : 1200.0;
    const int threads = argc >= 5 ? std::stoi(argv[4]) : 1;
    const std::size_t diagram_width = argc >= 6
        ? static_cast<std::size_t>(std::stoull(argv[5])) : 0;
    try {
        const erb::Instance instance = erb::read_instance(argv[1]);
        erb::SolveRecord record;
        const auto solve_start = std::chrono::steady_clock::now();
        try {
            record = diagram_width == 0
                ? solve(instance, time_limit, threads)
                : solve_with_diagram_bounds(instance, time_limit, threads,
                                            diagram_width);
        } catch (const std::bad_alloc&) {
            record.method = diagram_width == 0
                ? "proposed" : "diagram_w" + std::to_string(diagram_width);
            record.status = "memory_limit";
        } catch (const std::runtime_error& error) {
            record.method = diagram_width == 0
                ? "proposed" : "diagram_w" + std::to_string(diagram_width);
            if (std::string(error.what()) == "time_limit") {
                record.status = "time_limit";
            } else {
                throw;
            }
        }
        if (record.status != "optimal") {
            record.seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - solve_start).count();
        }
        erb::append_record(argv[2], instance, record, time_limit, threads);
        std::cout << instance.name << " proposed " << record.status
                  << " objective=" << record.objective
                  << " seconds=" << record.seconds << '\n';
    } catch (const std::exception& error) {
        std::cerr << "proposed_solver: " << error.what() << '\n';
        return 2;
    }
    return 0;
}
