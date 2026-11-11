#include "instance.hpp"

#define main frontier_validation_program_main
#include "frontier_labeling_core.inc"
#undef main

#include <chrono>
#include <iostream>
#include <unordered_map>

namespace {

enum class ComponentVariant {
    DenseDirect,
    SparseDirect,
    LabelDominanceBatched,
    TerminalDominanceBatched,
    DominanceDirect,
    DominanceBatched
};

struct ScenarioOutput {
    std::vector<erb::i64> table;
    LabelStats statistics;
    std::size_t vectors = 0;
    double table_seconds = 0.0;
    double solve_seconds = 0.0;
};

struct OrderedProblem {
    std::vector<Job> jobs;
    std::vector<int> order;
    std::vector<char> middle_eligible;
    int first = -1;
    int last = -1;
    int total_weight = 0;
};

void check_deadline(const std::chrono::steady_clock::time_point& deadline) {
    if (std::chrono::steady_clock::now() >= deadline) {
        throw std::runtime_error("time_limit");
    }
}

OrderedProblem prepare_problem(const std::vector<Job>& jobs) {
    OrderedProblem problem;
    problem.jobs = jobs;
    std::vector<int> wspt(jobs.size());
    std::iota(wspt.begin(), wspt.end(), 0);
    std::stable_sort(wspt.begin(), wspt.end(), [&](int a, int b) {
        return wspt_less(jobs[a], jobs[b]);
    });

    int first_rank = -1;
    int last_rank = -1;
    for (int rank = 0; rank < static_cast<int>(wspt.size()); ++rank) {
        problem.total_weight += jobs[wspt[rank]].w;
        if (jobs[wspt[rank]].resource) {
            if (first_rank < 0) first_rank = rank;
            last_rank = rank;
        }
    }
    if (first_rank < 0) {
        throw std::runtime_error("At least one resource job is required");
    }
    problem.first = wspt[first_rank];
    problem.last = wspt[last_rank];
    for (int rank = 0; rank < static_cast<int>(wspt.size()); ++rank) {
        const int index = wspt[rank];
        if (index == problem.first || index == problem.last) continue;
        problem.order.push_back(index);
        problem.middle_eligible.push_back(
            static_cast<char>(first_rank < rank && rank < last_rank));
    }
    return problem;
}

Descriptor make_descriptor(const OrderedProblem& problem, const Label& state) {
    const Job& f = problem.jobs[problem.first];
    const Job& l = problem.jobs[problem.last];
    const bool single_boundary = problem.first == problem.last;
    const int p_inside = single_boundary
        ? f.p : f.p + state.p_middle + l.p;
    const int w_inside = single_boundary
        ? f.w : f.w + state.w_middle + l.w;

    erb::i64 twc = state.cost;
    if (single_boundary) {
        twc += static_cast<erb::i64>(f.w) * f.p;
    } else {
        twc += static_cast<erb::i64>(state.w_middle) * f.p
            + static_cast<erb::i64>(f.w) * f.p
            + static_cast<erb::i64>(l.w)
                * (f.p + state.p_middle + l.p);
    }
    twc += static_cast<erb::i64>(w_inside) * state.p_before;
    twc += static_cast<erb::i64>(state.w_after)
        * (state.p_before + p_inside);
    return {state.p_before, state.p_before + p_inside,
            w_inside + state.w_after, twc};
}

std::vector<AssignmentArc> admissible_arcs(const Job& job,
                                           bool single_boundary,
                                           bool middle_eligible) {
    if (job.resource) {
        if (single_boundary) {
            throw std::runtime_error(
                "A nonboundary resource job cannot have one boundary job");
        }
        return {AssignmentArc::Middle};
    }
    std::vector<AssignmentArc> arcs{AssignmentArc::Before,
                                    AssignmentArc::After};
    if (!single_boundary && middle_eligible) {
        arcs.push_back(AssignmentArc::Middle);
    }
    return arcs;
}

std::vector<Descriptor> sparse_descriptors_without_dominance(
        const std::vector<Job>& jobs, LabelStats& stats,
        const std::chrono::steady_clock::time_point& deadline) {
    const auto start = std::chrono::steady_clock::now();
    const OrderedProblem problem = prepare_problem(jobs);
    const bool single_boundary = problem.first == problem.last;
    std::vector<Label> current{{0, 0, 0, 0, 0}};
    int processed_p = 0;
    int processed_w = 0;
    stats.peak_states = 1;

    for (std::size_t k = 0; k < problem.order.size(); ++k) {
        check_deadline(deadline);
        const Job& job = problem.jobs[problem.order[k]];
        const std::vector<AssignmentArc> arcs = admissible_arcs(
            job, single_boundary, problem.middle_eligible[k] != 0);
        std::unordered_map<State, erb::i64, StateHash> best;
        best.reserve(current.size() * arcs.size());
        std::size_t generated_in_layer = 0;
        for (const Label& label : current) {
            for (AssignmentArc arc : arcs) {
                if ((++generated_in_layer & ((1u << 20) - 1)) == 0) {
                    check_deadline(deadline);
                }
                const Label candidate = transition_label(
                    label, job, processed_p, processed_w, arc, false);
                const State key{candidate.p_before, candidate.p_middle,
                                candidate.w_middle, candidate.w_after};
                auto [iterator, inserted] = best.emplace(key, candidate.cost);
                if (!inserted && candidate.cost < iterator->second) {
                    iterator->second = candidate.cost;
                }
                ++stats.generated_states;
            }
        }
        current.clear();
        current.reserve(best.size());
        for (const auto& [state, cost] : best) {
            current.push_back({state.p_before, state.p_middle,
                               state.w_middle, state.w_after, cost});
        }
        stats.peak_states = std::max(stats.peak_states, current.size());
        processed_p += job.p;
        processed_w += job.w;
    }

    std::vector<Descriptor> descriptors;
    descriptors.reserve(current.size());
    for (const Label& state : current) {
        descriptors.push_back(make_descriptor(problem, state));
    }
    stats.final_descriptors = descriptors.size();
    stats.seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    return descriptors;
}

std::size_t triangular_count(int total) {
    // The dense DP stores every coordinate pair satisfying the natural sum
    // bound.  Omitting pairs whose two components exceed the processed total
    // avoids charging the dense formulation for provably impossible states.
    const std::size_t first = static_cast<std::size_t>(total) + 1;
    const std::size_t second = static_cast<std::size_t>(total) + 2;
    if (first > std::numeric_limits<std::size_t>::max() / second) {
        throw std::bad_alloc();
    }
    return first * second / 2;
}

std::size_t dense_size(int p, int w) {
    const std::size_t processing_pairs = triangular_count(p);
    const std::size_t weight_pairs = triangular_count(w);
    if (processing_pairs
        > std::numeric_limits<std::size_t>::max() / weight_pairs) {
        throw std::bad_alloc();
    }
    return processing_pairs * weight_pairs;
}

std::size_t triangular_index(int first, int second, int total) {
    return static_cast<std::size_t>(first) * (total + 1)
        - static_cast<std::size_t>(first) * (first - 1) / 2
        + second;
}

std::size_t dense_index(int p_before, int p_middle, int w_middle, int w_after,
                        int p_limit, int w_limit) {
    return triangular_index(p_before, p_middle, p_limit)
        * triangular_count(w_limit)
        + triangular_index(w_middle, w_after, w_limit);
}

std::vector<Descriptor> dense_descriptors(
        const std::vector<Job>& jobs, LabelStats& stats,
        const std::chrono::steady_clock::time_point& deadline) {
    const auto start = std::chrono::steady_clock::now();
    const OrderedProblem problem = prepare_problem(jobs);
    const bool single_boundary = problem.first == problem.last;
    std::vector<erb::i64> current(1, 0);
    int processed_p = 0;
    int processed_w = 0;
    stats.peak_states = 1;

    for (std::size_t k = 0; k < problem.order.size(); ++k) {
        check_deadline(deadline);
        const Job& job = problem.jobs[problem.order[k]];
        const std::vector<AssignmentArc> arcs = admissible_arcs(
            job, single_boundary, problem.middle_eligible[k] != 0);
        const int next_p = processed_p + job.p;
        const int next_w = processed_w + job.w;
        const std::size_t next_size = dense_size(next_p, next_w);
        std::vector<erb::i64> next(next_size, INF);
        stats.peak_states = std::max(stats.peak_states,
                                    current.size() + next.size());

        std::size_t visited = 0;
        for (int p_before = 0; p_before <= processed_p; ++p_before) {
            for (int p_middle = 0;
                 p_middle <= processed_p - p_before; ++p_middle) {
                for (int w_middle = 0; w_middle <= processed_w; ++w_middle) {
                    for (int w_after = 0;
                         w_after <= processed_w - w_middle; ++w_after) {
                        if ((++visited & ((1u << 20) - 1)) == 0) {
                            check_deadline(deadline);
                        }
                        const std::size_t index = dense_index(
                            p_before, p_middle, w_middle, w_after,
                            processed_p, processed_w);
                        if (current[index] >= INF / 2) continue;
                        const Label label{p_before, p_middle, w_middle,
                                          w_after, current[index]};
                        for (AssignmentArc arc : arcs) {
                            const Label candidate = transition_label(
                                label, job, processed_p, processed_w,
                                arc, false);
                            const std::size_t candidate_index = dense_index(
                                candidate.p_before, candidate.p_middle,
                                candidate.w_middle, candidate.w_after,
                                next_p, next_w);
                            next[candidate_index] = std::min(
                                next[candidate_index], candidate.cost);
                            ++stats.generated_states;
                        }
                    }
                }
            }
        }
        current.swap(next);
        processed_p = next_p;
        processed_w = next_w;
    }

    std::vector<Descriptor> descriptors;
    std::size_t visited = 0;
    for (int p_before = 0; p_before <= processed_p; ++p_before) {
        for (int p_middle = 0;
             p_middle <= processed_p - p_before; ++p_middle) {
            for (int w_middle = 0; w_middle <= processed_w; ++w_middle) {
                for (int w_after = 0;
                     w_after <= processed_w - w_middle; ++w_after) {
                    if ((++visited & ((1u << 20) - 1)) == 0) {
                        check_deadline(deadline);
                    }
                    const std::size_t index = dense_index(
                        p_before, p_middle, w_middle, w_after,
                        processed_p, processed_w);
                    if (current[index] >= INF / 2) continue;
                    descriptors.push_back(make_descriptor(
                        problem, {p_before, p_middle, w_middle, w_after,
                                  current[index]}));
                }
            }
        }
    }
    stats.final_descriptors = descriptors.size();
    stats.seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    return descriptors;
}

std::vector<erb::i64> direct_recourse_table(
        const std::vector<Descriptor>& descriptors, int maximum_arrival,
        int maximum_duration, int overtime_rate,
        const std::chrono::steady_clock::time_point& deadline) {
    const int duration_count = maximum_duration + 1;
    std::vector<erb::i64> table(
        static_cast<std::size_t>(maximum_arrival + 1) * duration_count, INF);
    for (int s = 0; s <= maximum_arrival; ++s) {
        check_deadline(deadline);
        for (int d = 0; d <= maximum_duration; ++d) {
            erb::i64 best = INF;
            for (const Descriptor& descriptor : descriptors) {
                const int idle = std::max(0, s - descriptor.before_first_r);
                const int resource_duration =
                    descriptor.through_last_r + idle - s;
                const erb::i64 value = descriptor.twc_without_idle
                    + static_cast<erb::i64>(descriptor.weight_after_first_r) * idle
                    + static_cast<erb::i64>(overtime_rate)
                        * std::max(0, resource_duration - d);
                best = std::min(best, value);
            }
            table[static_cast<std::size_t>(s) * duration_count + d] = best;
        }
    }
    return table;
}

std::vector<erb::i64> batched_recourse_table(
        const std::vector<Descriptor>& descriptors, int scenario_processing,
        int maximum_arrival, int maximum_duration, int overtime_rate,
        const std::chrono::steady_clock::time_point& deadline) {
    const int duration_count = maximum_duration + 1;
    std::vector<erb::i64> table(
        static_cast<std::size_t>(maximum_arrival + 1) * duration_count, INF);
    for (int s = 0; s <= maximum_arrival; ++s) {
        check_deadline(deadline);
        std::vector<erb::i64> exact_duration(
            static_cast<std::size_t>(scenario_processing + 1), INF);
        for (const Descriptor& descriptor : descriptors) {
            const int idle = std::max(0, s - descriptor.before_first_r);
            const int resource_duration =
                descriptor.through_last_r + idle - s;
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

std::string method_name(ComponentVariant variant) {
    if (variant == ComponentVariant::DenseDirect) return "component_dense_direct";
    if (variant == ComponentVariant::SparseDirect) return "component_sparse_direct";
    if (variant == ComponentVariant::LabelDominanceBatched) {
        return "component_label_dominance_batched";
    }
    if (variant == ComponentVariant::TerminalDominanceBatched) {
        return "component_terminal_dominance_batched";
    }
    if (variant == ComponentVariant::DominanceDirect) {
        return "component_dominance_direct";
    }
    return "component_dominance_batched";
}

ComponentVariant parse_variant(const std::string& text) {
    if (text == "dense_direct") return ComponentVariant::DenseDirect;
    if (text == "sparse_direct") return ComponentVariant::SparseDirect;
    if (text == "label_dominance_batched") {
        return ComponentVariant::LabelDominanceBatched;
    }
    if (text == "terminal_dominance_batched") {
        return ComponentVariant::TerminalDominanceBatched;
    }
    if (text == "dominance_direct") return ComponentVariant::DominanceDirect;
    if (text == "dominance_batched") return ComponentVariant::DominanceBatched;
    throw std::runtime_error("Unknown component variant: " + text);
}

ScenarioOutput solve_scenario(
        const erb::Instance& instance, const erb::Scenario& scenario,
        int maximum_arrival, int maximum_duration, ComponentVariant variant,
        const std::chrono::steady_clock::time_point& deadline) {
    const auto start = std::chrono::steady_clock::now();
    std::vector<Job> jobs;
    jobs.reserve(scenario.jobs.size());
    int processing = 0;
    for (int id : scenario.jobs) {
        const erb::Job& source = instance.jobs[id];
        jobs.push_back({source.id, source.processing, source.weight,
                        source.resource});
        processing += source.processing;
    }

    ScenarioOutput output;
    std::vector<Descriptor> descriptors;
    if (variant == ComponentVariant::DenseDirect) {
        descriptors = dense_descriptors(jobs, output.statistics, deadline);
    } else if (variant == ComponentVariant::SparseDirect) {
        descriptors = sparse_descriptors_without_dominance(
            jobs, output.statistics, deadline);
    } else if (variant == ComponentVariant::LabelDominanceBatched) {
        descriptors = labeling_descriptors(
            jobs, &output.statistics,
            true, DiagramMode::Exact, 0, 1, false,
            false, true, true, &deadline, false, false, false);
    } else if (variant == ComponentVariant::TerminalDominanceBatched) {
        descriptors = labeling_descriptors(
            jobs, &output.statistics,
            true, DiagramMode::Exact, 0, 1, false,
            true, true, false, &deadline, false, true, true);
    } else {
        descriptors = strongest_exact_descriptors(
            jobs, &output.statistics, &deadline);
    }
    output.vectors = descriptors.size();
    const auto table_start = std::chrono::steady_clock::now();
    if (variant == ComponentVariant::LabelDominanceBatched
        || variant == ComponentVariant::TerminalDominanceBatched
        || variant == ComponentVariant::DominanceBatched) {
        output.table = batched_recourse_table(
            descriptors, processing, maximum_arrival, maximum_duration,
            instance.mu, deadline);
    } else {
        output.table = direct_recourse_table(
            descriptors, maximum_arrival, maximum_duration,
            instance.mu, deadline);
    }
    output.table_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - table_start).count();
    output.solve_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    return output;
}

void add_statistics(erb::SolveRecord& record, const ScenarioOutput& output) {
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

erb::SolveRecord solve(const erb::Instance& instance, ComponentVariant variant,
                       double time_limit) {
    const auto start = std::chrono::steady_clock::now();
    const auto deadline = start + std::chrono::milliseconds(
        static_cast<long long>(1000.0 * time_limit));
    const int maximum_arrival = erb::max_ordinary_processing(instance);
    const int maximum_duration = erb::max_processing(instance);
    const int duration_count = maximum_duration + 1;
    const std::size_t contract_count =
        static_cast<std::size_t>(maximum_arrival + 1) * duration_count;
    std::vector<double> expected(contract_count, 0.0);
    erb::SolveRecord record;
    record.method = method_name(variant);

    for (const erb::Scenario& scenario : instance.scenarios) {
        const ScenarioOutput output = solve_scenario(
            instance, scenario, maximum_arrival, maximum_duration,
            variant, deadline);
        for (std::size_t index = 0; index < contract_count; ++index) {
            expected[index] += scenario.probability
                * static_cast<double>(output.table[index]);
        }
        add_statistics(record, output);
        ++record.iterations;
        ++record.completed_scenarios;
    }

    const auto scenario_stop = std::chrono::steady_clock::now();
    double optimum = std::numeric_limits<double>::infinity();
    for (int s = 0; s <= maximum_arrival; ++s) {
        for (int d = 0; d <= maximum_duration; ++d) {
            const std::size_t index =
                static_cast<std::size_t>(s) * duration_count + d;
            const double value = static_cast<double>(instance.lambda) * d
                + expected[index];
            if (value < optimum - 1e-9) {
                optimum = value;
                record.best_s = s;
                record.best_d = d;
            }
        }
    }
    record.contract_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - scenario_stop).count();
    record.status = "optimal";
    record.objective = optimum;
    record.bound = optimum;
    record.gap = 0.0;
    record.seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    return record;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5 || argc > 6) {
        std::cerr << "Usage: component_solver VARIANT INSTANCE OUTPUT_CSV "
                     "TIME_LIMIT [THREADS]\n";
        return 1;
    }
    const std::string variant_text = argv[1];
    const double time_limit = std::stod(argv[4]);
    const int threads = argc >= 6 ? std::stoi(argv[5]) : 1;
    if (threads != 1) {
        std::cerr << "component_solver requires one thread for matched timing\n";
        return 1;
    }
    try {
        const ComponentVariant variant = parse_variant(variant_text);
        const erb::Instance instance = erb::read_instance(argv[2]);
        const auto start = std::chrono::steady_clock::now();
        erb::SolveRecord record;
        record.method = method_name(variant);
        try {
            record = solve(instance, variant, time_limit);
        } catch (const std::bad_alloc&) {
            record.status = "memory_limit";
        } catch (const std::runtime_error& error) {
            if (std::string(error.what()) != "time_limit") throw;
            record.status = "time_limit";
        }
        if (record.status != "optimal") {
            record.seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
        }
        erb::append_record(argv[3], instance, record, time_limit, threads);
        std::cout << instance.name << ' ' << record.method << ' '
                  << record.status << " objective=" << record.objective
                  << " seconds=" << record.seconds << '\n';
    } catch (const std::exception& error) {
        std::cerr << "component_solver: " << error.what() << '\n';
        return 2;
    }
    return 0;
}
