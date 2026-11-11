#pragma once

#include "instance.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace erb {

struct WeightedSumSchedule {
    i64 weighted_completion = 0;
    int resource_duration = 0;
};

inline bool wspt_less_id(const Instance& instance, int first, int second) {
    const Job& left = instance.jobs.at(first);
    const Job& right = instance.jobs.at(second);
    const i64 lhs = static_cast<i64>(left.processing) * right.weight;
    const i64 rhs = static_cast<i64>(right.processing) * left.weight;
    return lhs != rhs ? lhs < rhs : left.id < right.id;
}

inline std::vector<int> wspt_order(const Instance& instance,
                                  const Scenario& scenario) {
    std::vector<int> order = scenario.jobs;
    std::stable_sort(order.begin(), order.end(), [&](int first, int second) {
        return wspt_less_id(instance, first, second);
    });
    return order;
}

// Briskorn et al.'s O(n log n) construction for
// min sum_j w_j C_j + rate * resource_duration.  The jobs supplied by a
// scenario are first placed in WSPT order.  Ordinary jobs between the first
// and last resource jobs are then moved before or after the resource segment
// according to the two threshold comparisons in their algorithm.
inline WeightedSumSchedule weighted_sum_schedule(
        const Instance& instance, const Scenario& scenario, long double rate) {
    const std::vector<int> jobs = wspt_order(instance, scenario);
    int first_resource = -1;
    int last_resource = -1;
    int resource_processing = 0;
    int resource_weight = 0;
    for (int k = 0; k < static_cast<int>(jobs.size()); ++k) {
        const Job& job = instance.jobs.at(jobs[k]);
        if (!job.resource) continue;
        if (first_resource < 0) first_resource = k;
        last_resource = k;
        resource_processing += job.processing;
        resource_weight += job.weight;
    }
    if (first_resource < 0) {
        throw std::runtime_error("Scenario lower bound requires a resource job");
    }

    std::vector<int> prefix_processing(jobs.size(), 0);
    std::vector<int> prefix_weight(jobs.size(), 0);
    int processing = 0;
    int weight = 0;
    for (int k = first_resource; k <= last_resource; ++k) {
        const Job& job = instance.jobs.at(jobs[k]);
        if (job.resource) {
            processing += job.processing;
            weight += job.weight;
        }
        prefix_processing[k] = processing;
        prefix_weight[k] = weight;
    }

    std::vector<int> suffix_processing(jobs.size(), 0);
    std::vector<int> suffix_weight(jobs.size(), 0);
    processing = 0;
    weight = 0;
    for (int k = last_resource; k >= first_resource; --k) {
        const Job& job = instance.jobs.at(jobs[k]);
        if (job.resource) {
            processing += job.processing;
            weight += job.weight;
        }
        suffix_processing[k] = processing;
        suffix_weight[k] = weight;
    }

    std::vector<char> before(jobs.size(), 0);
    std::vector<char> after(jobs.size(), 0);
    for (int k = first_resource; k <= last_resource; ++k) {
        const Job& job = instance.jobs.at(jobs[k]);
        if (job.resource) continue;
        const i64 job_resource = static_cast<i64>(job.weight)
            * resource_processing;
        const i64 resource_job = static_cast<i64>(resource_weight)
            * job.processing;
        if (job_resource >= resource_job) {
            const long double left = static_cast<long double>(job.weight)
                * prefix_processing[k];
            const long double right = static_cast<long double>(job.processing)
                * (static_cast<long double>(prefix_weight[k]) - rate);
            before[k] = left > right;
        } else {
            const long double left = static_cast<long double>(job.weight)
                * suffix_processing[k];
            const long double right = static_cast<long double>(job.processing)
                * (static_cast<long double>(suffix_weight[k]) + rate);
            after[k] = left < right;
        }
    }

    std::vector<int> sequence;
    sequence.reserve(jobs.size());
    for (int k = 0; k < first_resource; ++k) sequence.push_back(jobs[k]);
    for (int k = first_resource; k <= last_resource; ++k) {
        if (before[k]) sequence.push_back(jobs[k]);
    }
    for (int k = first_resource; k <= last_resource; ++k) {
        if (!before[k] && !after[k]) sequence.push_back(jobs[k]);
    }
    for (int k = first_resource; k <= last_resource; ++k) {
        if (after[k]) sequence.push_back(jobs[k]);
    }
    for (int k = last_resource + 1; k < static_cast<int>(jobs.size()); ++k) {
        sequence.push_back(jobs[k]);
    }

    int time = 0;
    int first_start = -1;
    int last_completion = -1;
    i64 completion = 0;
    for (int id : sequence) {
        const Job& job = instance.jobs.at(id);
        if (job.resource && first_start < 0) first_start = time;
        time += job.processing;
        completion += static_cast<i64>(job.weight) * time;
        if (job.resource) last_completion = time;
    }
    return {completion, last_completion - first_start};
}

inline std::vector<long double> weighted_sum_breakpoints(
        const Instance& instance, const Scenario& scenario) {
    const std::vector<int> jobs = wspt_order(instance, scenario);
    int first_resource = -1;
    int last_resource = -1;
    int resource_processing = 0;
    int resource_weight = 0;
    for (int k = 0; k < static_cast<int>(jobs.size()); ++k) {
        const Job& job = instance.jobs.at(jobs[k]);
        if (!job.resource) continue;
        if (first_resource < 0) first_resource = k;
        last_resource = k;
        resource_processing += job.processing;
        resource_weight += job.weight;
    }
    if (first_resource < 0) {
        throw std::runtime_error("Scenario lower bound requires a resource job");
    }

    std::vector<int> prefix_processing(jobs.size(), 0);
    std::vector<int> prefix_weight(jobs.size(), 0);
    int processing = 0;
    int weight = 0;
    for (int k = first_resource; k <= last_resource; ++k) {
        const Job& job = instance.jobs.at(jobs[k]);
        if (job.resource) {
            processing += job.processing;
            weight += job.weight;
        }
        prefix_processing[k] = processing;
        prefix_weight[k] = weight;
    }

    std::vector<int> suffix_processing(jobs.size(), 0);
    std::vector<int> suffix_weight(jobs.size(), 0);
    processing = 0;
    weight = 0;
    for (int k = last_resource; k >= first_resource; --k) {
        const Job& job = instance.jobs.at(jobs[k]);
        if (job.resource) {
            processing += job.processing;
            weight += job.weight;
        }
        suffix_processing[k] = processing;
        suffix_weight[k] = weight;
    }

    std::vector<long double> values{0.0L,
        static_cast<long double>(instance.mu)};
    for (int k = first_resource; k <= last_resource; ++k) {
        const Job& job = instance.jobs.at(jobs[k]);
        if (job.resource) continue;
        long double threshold = 0.0L;
        if (static_cast<i64>(job.weight) * resource_processing
            >= static_cast<i64>(resource_weight) * job.processing) {
            threshold = prefix_weight[k]
                - static_cast<long double>(job.weight) * prefix_processing[k]
                    / job.processing;
        } else {
            threshold = static_cast<long double>(job.weight)
                    * suffix_processing[k] / job.processing
                - suffix_weight[k];
        }
        if (threshold > 0.0L && threshold < instance.mu) {
            values.push_back(threshold);
        }
    }
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end(),
        [](long double first, long double second) {
            return std::abs(first - second) <= 1e-15L;
        }), values.end());
    return values;
}

// Section 5.4 bound: max_{0 <= xi <= mu}{Lambda(xi) - xi D}.
// The returned values are biased downward by a small numerical tolerance so
// that they remain safe lower bounds when passed to double-precision solvers.
inline std::vector<double> scenario_lower_bounds(
        const Instance& instance, const Scenario& scenario,
        int maximum_duration) {
    const std::vector<long double> breakpoints =
        weighted_sum_breakpoints(instance, scenario);
    std::vector<std::pair<long double, WeightedSumSchedule>> lines;
    lines.reserve(breakpoints.size());
    for (long double rate : breakpoints) {
        lines.emplace_back(rate,
            weighted_sum_schedule(instance, scenario, rate));
    }

    std::vector<double> result(maximum_duration + 1, 0.0);
    for (int duration = 0; duration <= maximum_duration; ++duration) {
        long double best = -std::numeric_limits<long double>::infinity();
        for (const auto& [rate, line] : lines) {
            const long double value = line.weighted_completion
                + rate * (line.resource_duration - duration);
            best = std::max(best, value);
        }
        const long double tolerance = 1e-12L
            * std::max(1.0L, std::abs(best));
        result[duration] = static_cast<double>(best - tolerance);
    }
    return result;
}

}  // namespace erb
