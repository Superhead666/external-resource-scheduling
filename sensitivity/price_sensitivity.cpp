#include "instance.hpp"

// Use the same terminal-vector generator as the submitted exact method.
#define main price_sensitivity_frontier_program_main
#include "frontier_labeling_core.inc"
#undef main

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct ContractEvaluation {
    erb::i64 recourse = INF;
    erb::i64 twc = INF;
    int resource_duration = -1;
    int overtime = -1;
};

struct RateResult {
    int lambda = 0;
    int mu = 0;
    int opt_s = -1;
    int opt_d = -1;
    double objective = std::numeric_limits<double>::infinity();
    int committed_s = -1;
    int committed_d = -1;
    int s0_d = -1;
    double s0_objective = std::numeric_limits<double>::infinity();
    double expected_twc = 0.0;
    double expected_overtime = 0.0;
    double overtime_probability = 0.0;
    double committed_expected_twc = 0.0;
    double committed_expected_overtime = 0.0;
    double committed_objective = 0.0;
    std::vector<double> expected_recourse;
    std::vector<double> expected_s0;
};

std::vector<int> parse_rates(const std::string& text) {
    std::vector<int> rates;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.empty()) throw std::runtime_error("Empty rate in rate list");
        const int value = std::stoi(item);
        if (value <= 0) throw std::runtime_error("Rates must be positive");
        rates.push_back(value);
    }
    if (rates.empty()) throw std::runtime_error("The rate list is empty");
    std::sort(rates.begin(), rates.end());
    rates.erase(std::unique(rates.begin(), rates.end()), rates.end());
    return rates;
}

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

std::vector<erb::i64> recourse_table(
        const std::vector<Descriptor>& descriptors, int scenario_processing,
        int maximum_arrival, int maximum_duration, int overtime_rate,
        const std::chrono::steady_clock::time_point& deadline) {
    const int duration_count = maximum_duration + 1;
    std::vector<erb::i64> table(
        static_cast<std::size_t>(maximum_arrival + 1) * duration_count, INF);
    for (int arrival = 0; arrival <= maximum_arrival; ++arrival) {
        if (std::chrono::steady_clock::now() >= deadline) {
            throw std::runtime_error("time_limit");
        }
        std::vector<erb::i64> exact(
            static_cast<std::size_t>(scenario_processing + 1), INF);
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
             resource_duration <= scenario_processing; ++resource_duration) {
            best = std::min(best, exact[resource_duration]);
            prefix[resource_duration] = best;
        }
        best = INF;
        for (int resource_duration = scenario_processing;
             resource_duration >= 0; --resource_duration) {
            if (exact[resource_duration] < INF / 2) {
                best = std::min(best, exact[resource_duration]
                    + static_cast<erb::i64>(overtime_rate) * resource_duration);
            }
            suffix[resource_duration] = best;
        }
        for (int booked = 0; booked <= maximum_duration; ++booked) {
            const int capped = std::min(booked, scenario_processing);
            erb::i64 value = prefix[capped];
            if (booked < scenario_processing && suffix[booked + 1] < INF / 2) {
                value = std::min(value, suffix[booked + 1]
                    - static_cast<erb::i64>(overtime_rate) * booked);
            }
            table[static_cast<std::size_t>(arrival) * duration_count + booked]
                = value;
        }
    }
    return table;
}

std::pair<int, int> committed_contract(
        const std::vector<Descriptor>& descriptors, int booking_rate) {
    erb::i64 best_value = INF;
    std::pair<int, int> best{-1, -1};
    for (const Descriptor& descriptor : descriptors) {
        const int arrival = descriptor.before_first_r;
        const int duration =
            descriptor.through_last_r - descriptor.before_first_r;
        const erb::i64 value = descriptor.twc_without_idle
            + static_cast<erb::i64>(booking_rate) * duration;
        const std::pair<int, int> contract{arrival, duration};
        if (value < best_value || (value == best_value && contract < best)) {
            best_value = value;
            best = contract;
        }
    }
    if (best.first < 0) {
        throw std::runtime_error("The committed-only problem is infeasible");
    }
    return best;
}

void append_header_if_needed(const std::string& output_path) {
    const bool header = !std::filesystem::exists(output_path)
        || std::filesystem::file_size(output_path) == 0;
    if (!header) return;
    std::ofstream output(output_path, std::ios::app);
    if (!output) throw std::runtime_error("Cannot open " + output_path);
    output << "instance,seed,lambda,mu,n_scenarios,n_committed,"
              "opt_s,opt_d,objective,expected_twc,expected_overtime,"
              "overtime_probability,conditional_overtime,booking_cost,"
              "booking_total_share,expected_processing,normalized_s,"
              "normalized_d,s0_best_d,s0_objective,s0_deviation_pct,"
              "committed_s,committed_d,committed_objective,"
              "committed_deviation_pct,seconds,generated_labels,"
              "peak_labels,retained_vectors\n";
}

void solve(const erb::Instance& instance, const std::vector<int>& booking_rates,
           int premium_ratio, const std::string& output_path,
           double time_limit) {
    const auto start = std::chrono::steady_clock::now();
    const auto deadline = start + std::chrono::milliseconds(
        static_cast<long long>(1000.0 * time_limit));
    const int maximum_arrival = erb::max_ordinary_processing(instance);
    const int maximum_duration = erb::max_processing(instance);
    const int duration_count = maximum_duration + 1;
    const std::size_t contract_count =
        static_cast<std::size_t>(maximum_arrival + 1) * duration_count;

    std::vector<RateResult> results;
    results.reserve(booking_rates.size());
    for (int booking_rate : booking_rates) {
        RateResult result;
        result.lambda = booking_rate;
        result.mu = premium_ratio * booking_rate;
        result.expected_recourse.assign(contract_count, 0.0);
        result.expected_s0.assign(static_cast<std::size_t>(duration_count), 0.0);
        results.push_back(std::move(result));
    }

    std::vector<int> committed_ids(
        static_cast<std::size_t>(instance.committed_jobs));
    std::iota(committed_ids.begin(), committed_ids.end(), 0);
    LabelStats committed_stats;
    const std::vector<Descriptor> committed_descriptors
        = strongest_exact_descriptors(
            jobs_for_ids(instance, committed_ids), &committed_stats, &deadline);
    for (RateResult& result : results) {
        std::tie(result.committed_s, result.committed_d)
            = committed_contract(committed_descriptors, result.lambda);
    }

    std::vector<std::vector<Descriptor>> scenario_descriptors;
    scenario_descriptors.reserve(instance.scenarios.size());
    std::size_t generated_labels = committed_stats.generated_states;
    std::size_t peak_labels = committed_stats.peak_states;
    std::size_t retained_vectors = committed_descriptors.size();
    double expected_processing = 0.0;

    for (const erb::Scenario& scenario : instance.scenarios) {
        if (std::chrono::steady_clock::now() >= deadline) {
            throw std::runtime_error("time_limit");
        }
        LabelStats statistics;
        scenario_descriptors.push_back(strongest_exact_descriptors(
            jobs_for_ids(instance, scenario.jobs), &statistics, &deadline));
        const std::vector<Descriptor>& descriptors = scenario_descriptors.back();
        generated_labels += statistics.generated_states;
        peak_labels = std::max(peak_labels, statistics.peak_states);
        retained_vectors += descriptors.size();
        const int processing = erb::scenario_processing(instance, scenario);
        expected_processing += scenario.probability * processing;
        for (RateResult& result : results) {
            const std::vector<erb::i64> table = recourse_table(
                descriptors, processing, maximum_arrival, maximum_duration,
                result.mu, deadline);
            for (std::size_t index = 0; index < contract_count; ++index) {
                result.expected_recourse[index] += scenario.probability
                    * static_cast<double>(table[index]);
            }
            for (int booked = 0; booked <= maximum_duration; ++booked) {
                result.expected_s0[booked] += scenario.probability
                    * static_cast<double>(table[booked]);
            }
        }
    }

    for (RateResult& result : results) {
        for (int arrival = 0; arrival <= maximum_arrival; ++arrival) {
            for (int booked = 0; booked <= maximum_duration; ++booked) {
                const std::size_t index =
                    static_cast<std::size_t>(arrival) * duration_count + booked;
                const double value = static_cast<double>(result.lambda) * booked
                    + result.expected_recourse[index];
                if (value < result.objective - 1e-9) {
                    result.objective = value;
                    result.opt_s = arrival;
                    result.opt_d = booked;
                }
            }
        }
    }

    for (std::size_t omega = 0; omega < instance.scenarios.size(); ++omega) {
        const erb::Scenario& scenario = instance.scenarios[omega];
        const std::vector<Descriptor>& descriptors = scenario_descriptors[omega];
        for (RateResult& result : results) {
            const ContractEvaluation optimum = evaluate_contract(
                descriptors, result.opt_s, result.opt_d, result.mu);
            result.expected_twc += scenario.probability * optimum.twc;
            result.expected_overtime += scenario.probability * optimum.overtime;
            result.overtime_probability += scenario.probability
                * static_cast<double>(optimum.overtime > 0);

            const ContractEvaluation committed = evaluate_contract(
                descriptors, result.committed_s, result.committed_d, result.mu);
            result.committed_expected_twc += scenario.probability * committed.twc;
            result.committed_expected_overtime +=
                scenario.probability * committed.overtime;

        }
    }

    append_header_if_needed(output_path);
    std::ofstream output(output_path, std::ios::app);
    if (!output) throw std::runtime_error("Cannot open " + output_path);
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    for (RateResult& result : results) {
        for (int booked = 0; booked <= maximum_duration; ++booked) {
            const double value = static_cast<double>(result.lambda) * booked
                + result.expected_s0[booked];
            if (value < result.s0_objective - 1e-9) {
                result.s0_objective = value;
                result.s0_d = booked;
            }
        }
        result.committed_objective
            = static_cast<double>(result.lambda) * result.committed_d
            + result.committed_expected_twc
            + static_cast<double>(result.mu)
                * result.committed_expected_overtime;
        const double reconstructed = static_cast<double>(result.lambda)
                * result.opt_d + result.expected_twc
            + static_cast<double>(result.mu) * result.expected_overtime;
        const double tolerance = 1e-7 * std::max(1.0, result.objective);
        if (std::abs(reconstructed - result.objective) > tolerance) {
            throw std::runtime_error("The reconstructed objective is inconsistent");
        }
        const auto deviation = [&](double value) {
            return 100.0 * (value - result.objective)
                / std::max(1.0, std::abs(result.objective));
        };
        const double conditional_overtime = result.overtime_probability > 0.0
            ? result.expected_overtime / result.overtime_probability : 0.0;
        const double booking_cost =
            static_cast<double>(result.lambda) * result.opt_d;
        output << std::setprecision(12)
               << instance.name << ',' << instance.seed << ',' << result.lambda
               << ',' << result.mu << ',' << instance.scenarios.size() << ','
               << instance.committed_jobs << ',' << result.opt_s << ','
               << result.opt_d << ',' << result.objective << ','
               << result.expected_twc << ',' << result.expected_overtime << ','
               << result.overtime_probability << ',' << conditional_overtime
               << ',' << booking_cost << ','
               << booking_cost / result.objective << ',' << expected_processing
               << ',' << result.opt_s / expected_processing << ','
               << result.opt_d / expected_processing << ',' << result.s0_d
               << ',' << result.s0_objective << ','
               << deviation(result.s0_objective) << ',' << result.committed_s
               << ',' << result.committed_d << ','
               << result.committed_objective << ','
               << deviation(result.committed_objective) << ',' << seconds
               << ',' << generated_labels << ',' << peak_labels << ','
               << retained_vectors << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "Usage: price_sensitivity INSTANCE OUTPUT_CSV "
                     "LAMBDA_LIST PREMIUM_RATIO TIME_LIMIT RUN_TAG\n";
        return 1;
    }
    try {
        erb::Instance instance = erb::read_instance(argv[1]);
        instance.name += "_" + std::string(argv[6]);
        solve(instance, parse_rates(argv[3]), std::stoi(argv[4]), argv[2],
              std::stod(argv[5]));
        std::cout << instance.name << " price_sensitivity optimal\n";
    } catch (const std::exception& error) {
        std::cerr << "price_sensitivity: " << error.what() << '\n';
        return 2;
    }
    return 0;
}
