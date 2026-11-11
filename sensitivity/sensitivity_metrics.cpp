#include "instance.hpp"

// Use exactly the terminal-vector generator used by the submitted exact method.
#define main sensitivity_frontier_program_main
#include "frontier_labeling_core.inc"
#undef main

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct ContractEvaluation {
    erb::i64 recourse = INF;
    erb::i64 twc = INF;
    int resource_duration = -1;
    int overtime = -1;
};

struct SensitivityRecord {
    double expected_twc = 0.0;
    double expected_overtime = 0.0;
    double overtime_probability = 0.0;
    double expected_processing = 0.0;
    double expected_resource_processing = 0.0;
    double reconstructed_objective = 0.0;
    int s0_duration = -1;
    double s0_objective = std::numeric_limits<double>::infinity();
    int committed_arrival = -1;
    int committed_duration = -1;
    double committed_expected_twc = 0.0;
    double committed_expected_overtime = 0.0;
    double committed_objective = std::numeric_limits<double>::infinity();
    std::size_t generated_labels = 0;
    std::size_t peak_labels = 0;
    std::size_t retained_vectors = 0;
    double seconds = 0.0;
};

std::vector<Job> jobs_for_ids(const erb::Instance& instance,
                              const std::vector<int>& ids) {
    std::vector<Job> jobs;
    jobs.reserve(ids.size());
    for (int id : ids) {
        const erb::Job& source = instance.jobs.at(id);
        jobs.push_back(
            {source.id, source.processing, source.weight, source.resource});
    }
    return jobs;
}

ContractEvaluation evaluate_contract(const std::vector<Descriptor>& descriptors,
                                     int arrival, int duration,
                                     int overtime_rate) {
    ContractEvaluation best;
    for (const Descriptor& descriptor : descriptors) {
        const int idle = std::max(0, arrival - descriptor.before_first_r);
        const int resource_duration =
            descriptor.through_last_r + idle - arrival;
        const int overtime = std::max(0, resource_duration - duration);
        const erb::i64 twc = descriptor.twc_without_idle
            + static_cast<erb::i64>(descriptor.weight_after_first_r) * idle;
        const erb::i64 recourse = twc
            + static_cast<erb::i64>(overtime_rate) * overtime;
        // The tie-breaking rule makes the reported operational quantities
        // reproducible without changing the recourse value.
        if (recourse < best.recourse
            || (recourse == best.recourse && overtime < best.overtime)
            || (recourse == best.recourse && overtime == best.overtime
                && resource_duration < best.resource_duration)
            || (recourse == best.recourse && overtime == best.overtime
                && resource_duration == best.resource_duration && twc < best.twc)) {
            best = {recourse, twc, resource_duration, overtime};
        }
    }
    if (best.recourse >= INF / 2) {
        throw std::runtime_error("No feasible terminal vector");
    }
    return best;
}

std::vector<erb::i64> recourse_row(const std::vector<Descriptor>& descriptors,
                                   int arrival, int maximum_duration,
                                   int overtime_rate) {
    int maximum_resource_duration = 0;
    for (const Descriptor& descriptor : descriptors) {
        const int idle = std::max(0, arrival - descriptor.before_first_r);
        maximum_resource_duration = std::max(
            maximum_resource_duration,
            descriptor.through_last_r + idle - arrival);
    }
    std::vector<erb::i64> exact(
        static_cast<std::size_t>(maximum_resource_duration + 1), INF);
    for (const Descriptor& descriptor : descriptors) {
        const int idle = std::max(0, arrival - descriptor.before_first_r);
        const int resource_duration =
            descriptor.through_last_r + idle - arrival;
        const erb::i64 twc = descriptor.twc_without_idle
            + static_cast<erb::i64>(descriptor.weight_after_first_r) * idle;
        exact[resource_duration] = std::min(exact[resource_duration], twc);
    }

    std::vector<erb::i64> prefix(exact.size(), INF);
    std::vector<erb::i64> suffix(exact.size() + 1, INF);
    erb::i64 best = INF;
    for (int resource_duration = 0;
         resource_duration <= maximum_resource_duration; ++resource_duration) {
        best = std::min(best, exact[resource_duration]);
        prefix[resource_duration] = best;
    }
    best = INF;
    for (int resource_duration = maximum_resource_duration;
         resource_duration >= 0; --resource_duration) {
        if (exact[resource_duration] < INF / 2) {
            best = std::min(best, exact[resource_duration]
                + static_cast<erb::i64>(overtime_rate) * resource_duration);
        }
        suffix[resource_duration] = best;
    }

    std::vector<erb::i64> row(
        static_cast<std::size_t>(maximum_duration + 1), INF);
    for (int booked = 0; booked <= maximum_duration; ++booked) {
        const int capped = std::min(booked, maximum_resource_duration);
        erb::i64 value = prefix[capped];
        if (booked < maximum_resource_duration
            && suffix[booked + 1] < INF / 2) {
            value = std::min(value, suffix[booked + 1]
                - static_cast<erb::i64>(overtime_rate) * booked);
        }
        row[booked] = value;
    }
    return row;
}

std::pair<int, int> committed_only_contract(
        const erb::Instance& instance,
        const std::chrono::steady_clock::time_point& deadline,
        LabelStats& statistics, std::size_t& vector_count) {
    std::vector<int> ids(static_cast<std::size_t>(instance.committed_jobs));
    std::iota(ids.begin(), ids.end(), 0);
    const std::vector<Descriptor> descriptors = strongest_exact_descriptors(
        jobs_for_ids(instance, ids), &statistics, &deadline);
    vector_count += descriptors.size();

    erb::i64 best_value = INF;
    int best_arrival = -1;
    int best_duration = -1;
    for (const Descriptor& descriptor : descriptors) {
        const int arrival = descriptor.before_first_r;
        const int duration =
            descriptor.through_last_r - descriptor.before_first_r;
        const erb::i64 value = descriptor.twc_without_idle
            + static_cast<erb::i64>(instance.lambda) * duration;
        if (value < best_value
            || (value == best_value
                && std::pair(arrival, duration)
                    < std::pair(best_arrival, best_duration))) {
            best_value = value;
            best_arrival = arrival;
            best_duration = duration;
        }
    }
    if (best_arrival < 0) {
        throw std::runtime_error("The committed-only problem is infeasible");
    }
    return {best_arrival, best_duration};
}

SensitivityRecord calculate(const erb::Instance& instance, int optimal_arrival,
                            int optimal_duration, double reference_objective,
                            double time_limit) {
    const auto start = std::chrono::steady_clock::now();
    const auto deadline = start + std::chrono::milliseconds(
        static_cast<long long>(1000.0 * time_limit));
    SensitivityRecord record;

    LabelStats committed_statistics;
    std::size_t committed_vectors = 0;
    std::tie(record.committed_arrival, record.committed_duration)
        = committed_only_contract(instance, deadline, committed_statistics,
                                  committed_vectors);
    record.generated_labels += committed_statistics.generated_states;
    record.peak_labels = std::max(
        record.peak_labels, committed_statistics.peak_states);
    record.retained_vectors += committed_vectors;

    const int maximum_duration = erb::max_processing(instance);
    std::vector<double> expected_s0(
        static_cast<std::size_t>(maximum_duration + 1), 0.0);
    double committed_recourse = 0.0;

    for (const erb::Scenario& scenario : instance.scenarios) {
        if (std::chrono::steady_clock::now() >= deadline) {
            throw std::runtime_error("time_limit");
        }
        LabelStats statistics;
        const std::vector<Descriptor> descriptors = strongest_exact_descriptors(
            jobs_for_ids(instance, scenario.jobs), &statistics, &deadline);
        record.generated_labels += statistics.generated_states;
        record.peak_labels = std::max(record.peak_labels, statistics.peak_states);
        record.retained_vectors += descriptors.size();

        const ContractEvaluation optimum = evaluate_contract(
            descriptors, optimal_arrival, optimal_duration, instance.mu);
        record.expected_twc += scenario.probability * optimum.twc;
        record.expected_overtime += scenario.probability * optimum.overtime;
        record.overtime_probability += scenario.probability
            * static_cast<double>(optimum.overtime > 0);
        record.expected_processing += scenario.probability
            * erb::scenario_processing(instance, scenario);
        record.expected_resource_processing += scenario.probability
            * erb::scenario_resource_processing(instance, scenario);

        const ContractEvaluation committed = evaluate_contract(
            descriptors, record.committed_arrival,
            record.committed_duration, instance.mu);
        record.committed_expected_twc += scenario.probability * committed.twc;
        record.committed_expected_overtime +=
            scenario.probability * committed.overtime;
        committed_recourse += scenario.probability * committed.recourse;

        const std::vector<erb::i64> row = recourse_row(
            descriptors, 0, maximum_duration, instance.mu);
        for (int duration = 0; duration <= maximum_duration; ++duration) {
            expected_s0[duration] += scenario.probability * row[duration];
        }
    }

    record.reconstructed_objective
        = static_cast<double>(instance.lambda) * optimal_duration
        + record.expected_twc
        + static_cast<double>(instance.mu) * record.expected_overtime;
    const double tolerance = 1e-7 * std::max(1.0, std::abs(reference_objective));
    if (std::abs(record.reconstructed_objective - reference_objective)
        > tolerance) {
        throw std::runtime_error(
            "The reconstructed optimum differs from the stored result");
    }

    for (int duration = 0; duration <= maximum_duration; ++duration) {
        const double value = static_cast<double>(instance.lambda) * duration
            + expected_s0[duration];
        if (value < record.s0_objective - 1e-9) {
            record.s0_objective = value;
            record.s0_duration = duration;
        }
    }
    record.committed_objective
        = static_cast<double>(instance.lambda) * record.committed_duration
        + committed_recourse;
    const double decomposed_difference
        = record.committed_expected_twc - record.expected_twc
        + static_cast<double>(instance.lambda)
            * (record.committed_duration - optimal_duration)
        + static_cast<double>(instance.mu)
            * (record.committed_expected_overtime - record.expected_overtime);
    if (std::abs(decomposed_difference
                 - (record.committed_objective - reference_objective))
        > tolerance) {
        throw std::runtime_error(
            "The committed-only cost decomposition is inconsistent");
    }
    if (record.s0_objective + tolerance < reference_objective
        || record.committed_objective + tolerance < reference_objective) {
        throw std::runtime_error(
            "A restricted policy improves on the stored optimum");
    }
    record.seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    return record;
}

void append_record(const std::string& path, const erb::Instance& instance,
                   int optimal_arrival, int optimal_duration,
                   double reference_objective, const SensitivityRecord& record) {
    const bool header = !std::filesystem::exists(path)
        || std::filesystem::file_size(path) == 0;
    std::ofstream output(path, std::ios::app);
    if (!output) throw std::runtime_error("Cannot open " + path);
    if (header) {
        output << "instance,class,seed,n_scenarios,n_committed,n_potential,"
                  "lambda,mu,opt_s,opt_d,objective,reconstructed_objective,"
                  "expected_twc,booking_cost,expected_overtime,overtime_cost,"
                  "overtime_probability,resource_cost,booking_resource_share,"
                  "booking_total_share,expected_processing,"
                  "expected_resource_processing,normalized_s,normalized_d,"
                  "s0_best_d,s0_objective,s0_deviation_pct,committed_s,"
                  "committed_d,committed_expected_twc,"
                  "committed_booking_cost,committed_expected_overtime,"
                  "committed_overtime_cost,committed_objective,"
                  "committed_deviation_pct,delta_twc,delta_booking_cost,"
                  "delta_overtime_cost,"
                  "metric_seconds,generated_labels,peak_labels,retained_vectors\n";
    }
    const double booking = static_cast<double>(instance.lambda) * optimal_duration;
    const double overtime = static_cast<double>(instance.mu)
        * record.expected_overtime;
    const double resource_cost = booking + overtime;
    const auto deviation = [&](double value) {
        return 100.0 * (value - reference_objective)
            / std::max(1.0, std::abs(reference_objective));
    };
    output << std::setprecision(12)
           << instance.name << ',' << instance.instance_class << ','
           << instance.seed << ',' << instance.scenarios.size() << ','
           << instance.committed_jobs << ',' << instance.potential_jobs << ','
           << instance.lambda << ',' << instance.mu << ',' << optimal_arrival
           << ',' << optimal_duration << ',' << reference_objective << ','
           << record.reconstructed_objective << ',' << record.expected_twc
           << ',' << booking << ',' << record.expected_overtime << ','
           << overtime << ',' << record.overtime_probability << ','
           << resource_cost << ','
           << (resource_cost > 0.0 ? booking / resource_cost : 0.0) << ','
           << booking / reference_objective << ',' << record.expected_processing
           << ',' << record.expected_resource_processing << ','
           << optimal_arrival / record.expected_processing << ','
           << optimal_duration / record.expected_processing << ','
           << record.s0_duration << ',' << record.s0_objective << ','
           << deviation(record.s0_objective) << ','
           << record.committed_arrival << ',' << record.committed_duration << ','
           << record.committed_expected_twc << ','
           << static_cast<double>(instance.lambda) * record.committed_duration
           << ',' << record.committed_expected_overtime << ','
           << static_cast<double>(instance.mu)
                * record.committed_expected_overtime << ','
           << record.committed_objective << ','
           << deviation(record.committed_objective) << ','
           << record.committed_expected_twc - record.expected_twc << ','
           << static_cast<double>(instance.lambda)
                * (record.committed_duration - optimal_duration) << ','
           << static_cast<double>(instance.mu)
                * (record.committed_expected_overtime
                   - record.expected_overtime) << ','
           << record.seconds
           << ',' << record.generated_labels << ',' << record.peak_labels << ','
           << record.retained_vectors << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 7 || argc > 7) {
        std::cerr << "Usage: sensitivity_metrics INSTANCE OUTPUT_CSV "
                     "OPT_S OPT_D OPT_OBJECTIVE TIME_LIMIT\n";
        return 1;
    }
    try {
        const erb::Instance instance = erb::read_instance(argv[1]);
        const int optimal_arrival = std::stoi(argv[3]);
        const int optimal_duration = std::stoi(argv[4]);
        const double objective = std::stod(argv[5]);
        const double time_limit = std::stod(argv[6]);
        const SensitivityRecord record = calculate(
            instance, optimal_arrival, optimal_duration, objective, time_limit);
        append_record(argv[2], instance, optimal_arrival, optimal_duration,
                      objective, record);
        std::cout << instance.name << " sensitivity_metrics optimal"
                  << " overtime_probability=" << record.overtime_probability
                  << " committed_objective=" << record.committed_objective
                  << " seconds=" << record.seconds << '\n';
    } catch (const std::exception& error) {
        std::cerr << "sensitivity_metrics: " << error.what() << '\n';
        return 2;
    }
    return 0;
}
