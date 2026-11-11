#include "instance.hpp"
#include "scenario_lower_bound.hpp"
#include "gurobi_c++.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <tuple>

namespace {

std::string status_name(int status, int solution_count) {
    if (status == GRB_OPTIMAL) return "optimal";
    if (status == GRB_TIME_LIMIT) return solution_count > 0 ? "time_limit_feasible" : "time_limit";
    if (status == GRB_MEM_LIMIT) return solution_count > 0 ? "memory_limit_feasible" : "memory_limit";
    if (status == GRB_INFEASIBLE) return "infeasible";
    if (status == GRB_UNBOUNDED) return "unbounded";
    if (status == GRB_INTERRUPTED) return solution_count > 0 ? "interrupted_feasible" : "interrupted";
    return "status_" + std::to_string(status);
}

void configure(GRBModel& model, const erb::Instance& instance, double time_limit,
               int threads, const std::string& log_file) {
    model.set(GRB_DoubleParam_TimeLimit, time_limit);
    model.set(GRB_IntParam_Threads, std::max(1, threads));
    model.set(GRB_DoubleParam_MIPGap, 0.0);
    model.set(GRB_IntParam_Seed, instance.seed % 2000000000);
    if (log_file.empty() || log_file == "-") {
        model.set(GRB_IntParam_OutputFlag, 0);
    } else {
        model.set(GRB_IntParam_OutputFlag, 1);
        model.set(GRB_IntParam_LogToConsole, 0);
        model.set(GRB_StringParam_LogFile, log_file);
    }
}

double remaining_time(const std::chrono::steady_clock::time_point& start,
                      double time_limit) {
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    return std::max(0.01, time_limit - elapsed);
}

erb::SolveRecord extract_record(GRBModel& model, const std::string& method,
                                const std::chrono::steady_clock::time_point& start,
                                int best_s, int best_d) {
    erb::SolveRecord record;
    record.method = method;
    const int status = model.get(GRB_IntAttr_Status);
    const int solution_count = model.get(GRB_IntAttr_SolCount);
    record.status = status_name(status, solution_count);
    if (solution_count > 0) record.objective = model.get(GRB_DoubleAttr_ObjVal);
    record.bound = model.get(GRB_DoubleAttr_ObjBound);
    record.gap = erb::relative_gap(record.objective, record.bound);
    record.seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    record.best_s = best_s;
    record.best_d = best_d;
    record.nodes_or_branches = model.get(GRB_DoubleAttr_NodeCount);
    record.variables = model.get(GRB_IntAttr_NumVars);
    record.constraints = static_cast<long long>(model.get(GRB_IntAttr_NumConstrs))
        + model.get(GRB_IntAttr_NumGenConstrs);
    return record;
}

erb::SolveRecord solve_compact(GRBEnv& environment, const erb::Instance& instance,
                               double time_limit, int threads,
                               const std::string& log_file) {
    const auto start = std::chrono::steady_clock::now();
    GRBModel model(environment);
    configure(model, instance, time_limit, threads, log_file);
    const int maximum_arrival = erb::max_ordinary_processing(instance);
    const int maximum_duration = erb::max_processing(instance);
    GRBVar arrival = model.addVar(0.0, maximum_arrival, 0.0, GRB_INTEGER, "s");
    GRBVar duration = model.addVar(0.0, maximum_duration, instance.lambda,
                                   GRB_INTEGER, "D");
    arrival.set(GRB_DoubleAttr_Start, 0.0);
    duration.set(GRB_DoubleAttr_Start, maximum_duration);

    for (const erb::Scenario& scenario : instance.scenarios) {
        const int n = static_cast<int>(scenario.jobs.size());
        const int horizon = maximum_arrival + erb::scenario_processing(instance, scenario);
        std::vector<GRBVar> completion;
        completion.reserve(n);
        for (int position = 0; position < n; ++position) {
            const erb::Job& job = instance.jobs[scenario.jobs[position]];
            completion.push_back(model.addVar(
                job.processing, horizon,
                scenario.probability * job.weight, GRB_CONTINUOUS,
                "C_" + std::to_string(scenario.id) + "_" + std::to_string(job.id)));
        }

        std::vector<int> wspt_order(n);
        std::iota(wspt_order.begin(), wspt_order.end(), 0);
        std::stable_sort(wspt_order.begin(), wspt_order.end(), [&](int a, int b) {
            const erb::Job& left = instance.jobs[scenario.jobs[a]];
            const erb::Job& right = instance.jobs[scenario.jobs[b]];
            const erb::i64 lhs = static_cast<erb::i64>(left.processing) * right.weight;
            const erb::i64 rhs = static_cast<erb::i64>(right.processing) * left.weight;
            return lhs != rhs ? lhs < rhs : left.id < right.id;
        });
        std::vector<int> wspt_rank(n);
        int warm_completion = 0;
        for (int rank = 0; rank < n; ++rank) {
            const int position = wspt_order[rank];
            wspt_rank[position] = rank;
            warm_completion += instance.jobs[scenario.jobs[position]].processing;
            completion[position].set(GRB_DoubleAttr_Start, warm_completion);
        }

        for (int a = 0; a < n; ++a) {
            for (int b = a + 1; b < n; ++b) {
                const erb::Job& job_a = instance.jobs[scenario.jobs[a]];
                const erb::Job& job_b = instance.jobs[scenario.jobs[b]];
                GRBVar order = model.addVar(
                    0.0, 1.0, 0.0, GRB_BINARY,
                    "y_" + std::to_string(scenario.id) + "_"
                        + std::to_string(a) + "_" + std::to_string(b));
                model.addGenConstrIndicator(order, 1, completion[b] - completion[a],
                                            GRB_GREATER_EQUAL, job_b.processing);
                model.addGenConstrIndicator(order, 0, completion[a] - completion[b],
                                            GRB_GREATER_EQUAL, job_a.processing);
                order.set(GRB_DoubleAttr_Start, wspt_rank[a] < wspt_rank[b] ? 1.0 : 0.0);
            }
        }

        GRBVar overtime = model.addVar(
            0.0, horizon, scenario.probability * instance.mu, GRB_CONTINUOUS,
            "ot_" + std::to_string(scenario.id));
        overtime.set(GRB_DoubleAttr_Start, 0.0);
        for (int position = 0; position < n; ++position) {
            const erb::Job& job = instance.jobs[scenario.jobs[position]];
            if (!job.resource) continue;
            model.addConstr(completion[position] >= arrival + job.processing);
            model.addConstr(overtime >= completion[position] - arrival - duration);
        }
    }
    model.set(GRB_IntAttr_ModelSense, GRB_MINIMIZE);
    model.set(GRB_DoubleParam_TimeLimit, remaining_time(start, time_limit));
    model.optimize();
    const int solution_count = model.get(GRB_IntAttr_SolCount);
    const int best_s = solution_count > 0
        ? static_cast<int>(std::llround(arrival.get(GRB_DoubleAttr_X))) : -1;
    const int best_d = solution_count > 0
        ? static_cast<int>(std::llround(duration.get(GRB_DoubleAttr_X))) : -1;
    return extract_record(model, "compact_milp", start, best_s, best_d);
}

struct TimeIndexedJob {
    int job_id = -1;
    std::vector<GRBVar> starts;
    GRBLinExpr start_time = 0.0;
    GRBLinExpr completion_time = 0.0;
};

erb::SolveRecord solve_time_indexed(GRBEnv& environment,
                                    const erb::Instance& instance,
                                    double time_limit, int threads,
                                    const std::string& log_file,
                                    bool relax_integrality = false,
                                    int fixed_arrival = -1,
                                    int fixed_duration = -1) {
    const auto start = std::chrono::steady_clock::now();
    GRBModel model(environment);
    configure(model, instance, time_limit, threads, log_file);
    const int maximum_arrival = erb::max_ordinary_processing(instance);
    const int maximum_duration = erb::max_processing(instance);

    const char choice_type = relax_integrality ? GRB_CONTINUOUS : GRB_BINARY;
    std::vector<GRBVar> arrival_choice;
    for (int s = 0; s <= maximum_arrival; ++s) {
        const double lower = fixed_arrival < 0 ? 0.0 : (s == fixed_arrival);
        const double upper = fixed_arrival < 0 ? 1.0 : (s == fixed_arrival);
        GRBVar variable = model.addVar(lower, upper, 0.0, choice_type,
                                       "a_" + std::to_string(s));
        const int warm_arrival = fixed_arrival < 0 ? 0 : fixed_arrival;
        variable.set(GRB_DoubleAttr_Start, s == warm_arrival ? 1.0 : 0.0);
        arrival_choice.push_back(variable);
    }
    std::vector<GRBVar> duration_choice;
    for (int d = 0; d <= maximum_duration; ++d) {
        const double lower = fixed_duration < 0 ? 0.0 : (d == fixed_duration);
        const double upper = fixed_duration < 0 ? 1.0 : (d == fixed_duration);
        GRBVar variable = model.addVar(
            lower, upper, static_cast<double>(instance.lambda) * d, choice_type,
            "b_" + std::to_string(d));
        const int warm_duration = fixed_duration < 0
            ? maximum_duration : fixed_duration;
        variable.set(GRB_DoubleAttr_Start, d == warm_duration ? 1.0 : 0.0);
        duration_choice.push_back(variable);
    }
    GRBLinExpr select_arrival = 0.0;
    GRBLinExpr select_duration = 0.0;
    for (GRBVar& variable : arrival_choice) select_arrival += variable;
    for (GRBVar& variable : duration_choice) select_duration += variable;
    model.addConstr(select_arrival == 1.0);
    model.addConstr(select_duration == 1.0);
    GRBLinExpr arrival_expression = 0.0;
    GRBLinExpr duration_expression = 0.0;
    for (int s = 0; s <= maximum_arrival; ++s) arrival_expression += s * arrival_choice[s];
    for (int d = 0; d <= maximum_duration; ++d) duration_expression += d * duration_choice[d];
    std::vector<GRBLinExpr> arrived_by(maximum_arrival);
    GRBLinExpr cumulative_arrival = 0.0;
    for (int t = 0; t < maximum_arrival; ++t) {
        cumulative_arrival += arrival_choice[t];
        arrived_by[t] = cumulative_arrival;
    }

    for (const erb::Scenario& scenario : instance.scenarios) {
        const int horizon = maximum_arrival + erb::scenario_processing(instance, scenario);
        GRBLinExpr scenario_cost = 0.0;
        std::vector<TimeIndexedJob> variables;
        variables.reserve(scenario.jobs.size());
        for (int id : scenario.jobs) {
            const erb::Job& job = instance.jobs[id];
            TimeIndexedJob data;
            data.job_id = id;
            data.starts.reserve(horizon - job.processing + 1);
            for (int t = 0; t + job.processing <= horizon; ++t) {
                GRBVar variable = model.addVar(
                    0.0, 1.0,
                    scenario.probability * job.weight * (t + job.processing),
                    relax_integrality ? GRB_CONTINUOUS : GRB_BINARY,
                    "x_" + std::to_string(scenario.id) + "_"
                        + std::to_string(id) + "_" + std::to_string(t));
                variable.set(GRB_DoubleAttr_Start, 0.0);
                scenario_cost += job.weight * (t + job.processing) * variable;
                data.start_time += t * variable;
                data.completion_time += (t + job.processing) * variable;
                data.starts.push_back(variable);
            }
            GRBLinExpr assigned = 0.0;
            for (GRBVar& variable : data.starts) assigned += variable;
            model.addConstr(assigned == 1.0);
            variables.push_back(std::move(data));
        }

        for (int time = 0; time < horizon; ++time) {
            GRBLinExpr active = 0.0;
            for (const TimeIndexedJob& data : variables) {
                const int processing = instance.jobs[data.job_id].processing;
                const int first = std::max(0, time - processing + 1);
                const int last = std::min(time, static_cast<int>(data.starts.size()) - 1);
                for (int t = first; t <= last; ++t) active += data.starts[t];
            }
            model.addConstr(active <= 1.0);
        }

        GRBVar overtime = model.addVar(
            0.0, horizon, scenario.probability * instance.mu, GRB_CONTINUOUS,
            "ot_" + std::to_string(scenario.id));
        overtime.set(GRB_DoubleAttr_Start, 0.0);
        scenario_cost += instance.mu * overtime;
        for (const TimeIndexedJob& data : variables) {
            const erb::Job& job = instance.jobs[data.job_id];
            for (int t = 0; t < static_cast<int>(data.starts.size()); ++t) {
                if (job.resource && t < maximum_arrival) {
                    model.addConstr(data.starts[t] <= arrived_by[t]);
                }
            }
            if (job.resource) {
                model.addConstr(overtime >= data.completion_time
                    - arrival_expression - duration_expression);
            }
        }

        const std::vector<double> lower_bounds = erb::scenario_lower_bounds(
            instance, scenario, maximum_duration);
        GRBLinExpr scenario_bound = 0.0;
        for (int d = 0; d <= maximum_duration; ++d) {
            scenario_bound += lower_bounds[d] * duration_choice[d];
        }
        model.addConstr(scenario_cost >= scenario_bound,
            "scenario_lb_" + std::to_string(scenario.id));

        std::vector<int> order(scenario.jobs.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            const erb::Job& left = instance.jobs[scenario.jobs[a]];
            const erb::Job& right = instance.jobs[scenario.jobs[b]];
            const erb::i64 lhs = static_cast<erb::i64>(left.processing) * right.weight;
            const erb::i64 rhs = static_cast<erb::i64>(right.processing) * left.weight;
            return lhs != rhs ? lhs < rhs : left.id < right.id;
        });
        std::vector<int> wspt_rank(order.size());
        for (int rank = 0; rank < static_cast<int>(order.size()); ++rank) {
            wspt_rank[order[rank]] = rank;
        }

        std::vector<int> resource_positions;
        for (int position : order) {
            if (instance.jobs[scenario.jobs[position]].resource) {
                resource_positions.push_back(position);
            }
        }
        for (int rank = 1; rank < static_cast<int>(resource_positions.size()); ++rank) {
            const int preceding = resource_positions[rank - 1];
            const int following = resource_positions[rank];
            model.addConstr(
                variables[following].start_time
                    >= variables[preceding].completion_time,
                "resource_wspt_" + std::to_string(scenario.id) + "_"
                    + std::to_string(rank));
        }

        if (!relax_integrality && !resource_positions.empty()) {
            const int first_resource = resource_positions.front();
            const int last_resource = resource_positions.back();
            for (int position = 0;
                 position < static_cast<int>(variables.size()); ++position) {
                const erb::Job& job = instance.jobs[variables[position].job_id];
                if (job.resource
                    || (wspt_rank[first_resource] < wspt_rank[position]
                        && wspt_rank[position] < wspt_rank[last_resource])) {
                    continue;
                }
                GRBVar before_segment = model.addVar(
                    0.0, 1.0, 0.0, GRB_BINARY,
                    "outside_middle_" + std::to_string(scenario.id) + "_"
                        + std::to_string(job.id));
                const bool wspt_before = wspt_rank[position] < wspt_rank[first_resource];
                before_segment.set(GRB_DoubleAttr_Start, wspt_before ? 1.0 : 0.0);
                model.addGenConstrIndicator(
                    before_segment, 1,
                    variables[first_resource].start_time
                        - variables[position].completion_time,
                    GRB_GREATER_EQUAL, 0.0);
                model.addGenConstrIndicator(
                    before_segment, 0,
                    variables[position].start_time
                        - variables[last_resource].completion_time,
                    GRB_GREATER_EQUAL, 0.0);
            }
        }
        if (!relax_integrality) {
            int warm_start = 0;
            for (int position : order) {
                variables[position].starts[warm_start].set(GRB_DoubleAttr_Start, 1.0);
                warm_start += instance.jobs[variables[position].job_id].processing;
            }
        }
    }

    model.set(GRB_IntAttr_ModelSense, GRB_MINIMIZE);
    model.set(GRB_DoubleParam_TimeLimit, remaining_time(start, time_limit));
    model.optimize();
    int best_s = -1;
    int best_d = -1;
    if (model.get(GRB_IntAttr_SolCount) > 0) {
        double largest = -1.0;
        for (int s = 0; s <= maximum_arrival; ++s) {
            const double value = arrival_choice[s].get(GRB_DoubleAttr_X);
            if (value > largest) { largest = value; best_s = s; }
        }
        largest = -1.0;
        for (int d = 0; d <= maximum_duration; ++d) {
            const double value = duration_choice[d].get(GRB_DoubleAttr_X);
            if (value > largest) { largest = value; best_d = d; }
        }
    }
    std::string method = "time_indexed_milp";
    if (relax_integrality) {
        method = fixed_arrival >= 0
            ? "time_indexed_fixed_contract_lp" : "time_indexed_lp";
    }
    return extract_record(model, method, start, best_s, best_d);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4 || argc > 9) {
        std::cerr << "Usage: gurobi_baselines METHOD INSTANCE OUTPUT_CSV "
                     "[TIME_LIMIT] [THREADS] [LOG_FILE] [FIXED_S FIXED_D]\n"
                     "METHOD is compact, time_indexed, time_indexed_lp, or "
                     "time_indexed_fixed_lp\n";
        return 1;
    }
    const std::string method = argv[1];
    const double time_limit = argc >= 5 ? std::stod(argv[4]) : 1200.0;
    const int threads = argc >= 6 ? std::stoi(argv[5]) : 1;
    const std::string log_file = argc >= 7 ? argv[6] : "";
    try {
        const erb::Instance instance = erb::read_instance(argv[2]);
        GRBEnv environment(true);
        environment.set(GRB_IntParam_OutputFlag, 0);
        environment.start();
        erb::SolveRecord record;
        if (method == "compact") {
            record = solve_compact(environment, instance, time_limit, threads, log_file);
        } else if (method == "time_indexed") {
            record = solve_time_indexed(environment, instance, time_limit, threads, log_file);
        } else if (method == "time_indexed_lp") {
            record = solve_time_indexed(environment, instance, time_limit, threads,
                                        log_file, true);
        } else if (method == "time_indexed_fixed_lp") {
            if (argc != 9) {
                throw std::runtime_error(
                    "time_indexed_fixed_lp requires FIXED_S and FIXED_D");
            }
            const int fixed_s = std::stoi(argv[7]);
            const int fixed_d = std::stoi(argv[8]);
            if (fixed_s < 0 || fixed_s > erb::max_ordinary_processing(instance)
                || fixed_d < 0 || fixed_d > erb::max_processing(instance)) {
                throw std::runtime_error("Fixed contract is outside the contract domain");
            }
            record = solve_time_indexed(environment, instance, time_limit, threads,
                                        log_file, true, fixed_s, fixed_d);
        } else {
            throw std::runtime_error("Unknown method " + method);
        }
        erb::append_record(argv[3], instance, record, time_limit, threads);
        std::cout << instance.name << ' ' << record.method << ' ' << record.status
                  << " objective=" << record.objective << " bound=" << record.bound
                  << " gap=" << record.gap << " seconds=" << record.seconds << '\n';
    } catch (const GRBException& error) {
        std::cerr << "Gurobi error " << error.getErrorCode() << ": "
                  << error.getMessage() << '\n';
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "gurobi_baselines: " << error.what() << '\n';
        return 3;
    }
    return 0;
}
