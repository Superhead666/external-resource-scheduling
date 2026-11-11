#include "instance.hpp"
#include "cp_structure.hpp"
#include "scenario_lower_bound.hpp"
#include "gurobi_c++.h"
#include <ilcp/cp.h>

#include <chrono>
#include <iostream>
#include <unordered_set>

namespace {

enum class RecourseOracle { Milp, Cp };

struct StartVariable {
    int time = 0;
    GRBVar variable;
};

struct RecourseResult {
    bool feasible = false;
    bool optimal = false;
    double objective = std::numeric_limits<double>::infinity();
    double bound = -std::numeric_limits<double>::infinity();
    double nodes = 0.0;
};

void configure(GRBModel& model, const erb::Instance& instance,
               double time_limit, int threads) {
    model.set(GRB_IntParam_OutputFlag, 0);
    model.set(GRB_DoubleParam_TimeLimit, std::max(0.01, time_limit));
    model.set(GRB_IntParam_Threads, std::max(1, threads));
    model.set(GRB_DoubleParam_MIPGap, 0.0);
    model.set(GRB_IntParam_Seed, instance.seed % 2000000000);
}

double elapsed_seconds(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
}

RecourseResult solve_recourse(GRBEnv& environment, const erb::Instance& instance,
                              const erb::Scenario& scenario, int arrival,
                              int duration, double time_limit, int threads) {
    const auto start = std::chrono::steady_clock::now();
    GRBModel model(environment);
    configure(model, instance, time_limit, threads);
    const int processing_total = erb::scenario_processing(instance, scenario);
    const int horizon = arrival + processing_total;
    std::vector<std::vector<StartVariable>> starts;
    starts.reserve(scenario.jobs.size());
    for (int id : scenario.jobs) {
        const erb::Job& job = instance.jobs[id];
        std::vector<StartVariable> job_starts;
        const int first_time = job.resource ? arrival : 0;
        for (int time = first_time; time + job.processing <= horizon; ++time) {
            GRBVar variable = model.addVar(
                0.0, 1.0, static_cast<double>(job.weight) * (time + job.processing),
                GRB_BINARY);
            variable.set(GRB_DoubleAttr_Start, 0.0);
            job_starts.push_back({time, variable});
        }
        GRBLinExpr assigned = 0.0;
        for (StartVariable& start : job_starts) assigned += start.variable;
        model.addConstr(assigned == 1.0);
        starts.push_back(std::move(job_starts));
    }
    for (int time = 0; time < horizon; ++time) {
        GRBLinExpr active = 0.0;
        for (int position = 0; position < static_cast<int>(scenario.jobs.size()); ++position) {
            const int processing = instance.jobs[scenario.jobs[position]].processing;
            for (const StartVariable& start : starts[position]) {
                if (start.time <= time && time < start.time + processing) {
                    active += start.variable;
                }
            }
        }
        model.addConstr(active <= 1.0);
    }
    GRBVar overtime = model.addVar(0.0, horizon, instance.mu, GRB_CONTINUOUS);
    GRBLinExpr scenario_cost = instance.mu * overtime;
    for (int position = 0; position < static_cast<int>(scenario.jobs.size()); ++position) {
        const erb::Job& job = instance.jobs[scenario.jobs[position]];
        for (const StartVariable& start : starts[position]) {
            scenario_cost += static_cast<double>(job.weight)
                * (start.time + job.processing) * start.variable;
        }
        if (!job.resource) continue;
        GRBLinExpr completion = 0.0;
        for (const StartVariable& start : starts[position]) {
            completion += (start.time + job.processing) * start.variable;
        }
        model.addConstr(overtime >= completion - arrival - duration);
    }
    const std::vector<double> lower_bounds = erb::scenario_lower_bounds(
        instance, scenario, erb::max_processing(instance));
    model.addConstr(scenario_cost >= lower_bounds.at(duration));

    std::vector<int> order(scenario.jobs.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        const erb::Job& left = instance.jobs[scenario.jobs[a]];
        const erb::Job& right = instance.jobs[scenario.jobs[b]];
        const erb::i64 lhs = static_cast<erb::i64>(left.processing) * right.weight;
        const erb::i64 rhs = static_cast<erb::i64>(right.processing) * left.weight;
        return lhs != rhs ? lhs < rhs : left.id < right.id;
    });
    int next_start = 0;
    int last_resource_completion = arrival;
    for (int position : order) {
        const erb::Job& job = instance.jobs[scenario.jobs[position]];
        if (job.resource) next_start = std::max(next_start, arrival);
        bool found = false;
        for (StartVariable& start : starts[position]) {
            if (start.time == next_start) {
                start.variable.set(GRB_DoubleAttr_Start, 1.0);
                found = true;
                break;
            }
        }
        if (!found) throw std::runtime_error("Warm-start time is outside the horizon");
        next_start += job.processing;
        if (job.resource) last_resource_completion = next_start;
    }
    overtime.set(GRB_DoubleAttr_Start,
        std::max(0, last_resource_completion - arrival - duration));

    model.set(GRB_IntAttr_ModelSense, GRB_MINIMIZE);
    model.set(GRB_DoubleParam_TimeLimit,
        std::max(0.01, time_limit - elapsed_seconds(start)));
    model.optimize();
    RecourseResult result;
    const int status = model.get(GRB_IntAttr_Status);
    result.feasible = model.get(GRB_IntAttr_SolCount) > 0;
    result.optimal = status == GRB_OPTIMAL;
    if (result.feasible) result.objective = model.get(GRB_DoubleAttr_ObjVal);
    result.bound = model.get(GRB_DoubleAttr_ObjBound);
    result.nodes = model.get(GRB_DoubleAttr_NodeCount);
    return result;
}

RecourseResult solve_recourse_cp(const erb::Instance& instance,
                                 const erb::Scenario& scenario, int arrival,
                                 int duration, double time_limit, int workers) {
    const auto start = std::chrono::steady_clock::now();
    IloEnv environment;
    RecourseResult result;
    try {
        IloModel model(environment);
        const int horizon = arrival + erb::scenario_processing(instance, scenario);
        IloIntervalVarArray intervals(environment);
        IloIntExprArray resource_ends(environment);
        IloIntExpr objective(environment);
        std::vector<IloIntervalVar> interval_by_position;
        interval_by_position.reserve(scenario.jobs.size());
        for (int id : scenario.jobs) {
            const erb::Job& job = instance.jobs[id];
            IloIntervalVar interval(environment, job.processing);
            interval.setStartMin(job.resource ? arrival : 0);
            interval.setStartMax(horizon - job.processing);
            interval.setEndMax(horizon);
            intervals.add(interval);
            interval_by_position.push_back(interval);
            objective += job.weight * IloEndOf(interval);
            if (job.resource) resource_ends.add(IloEndOf(interval));
        }
        IloIntervalSequenceVar sequence(environment, intervals);
        model.add(IloNoOverlap(environment, sequence));
        const std::vector<int> order = erb::add_cp_sequence_structure(
            environment, model, instance, scenario, interval_by_position);
        IloIntExpr last_resource_completion = IloMax(resource_ends);
        IloIntExpr overtime = IloMax(
            last_resource_completion - arrival - duration, 0);
        objective += instance.mu * overtime;
        const std::vector<double> lower_bounds = erb::scenario_lower_bounds(
            instance, scenario, erb::max_processing(instance));
        model.add(objective >= lower_bounds.at(duration));
        model.add(IloMinimize(environment, objective));

        IloCP cp(model);
        cp.setParameter(IloCP::Workers, std::max(1, workers));
        cp.setParameter(IloCP::RandomSeed, instance.seed + scenario.id);
        cp.setParameter(IloCP::OptimalityTolerance, 0.0);
        cp.setParameter(IloCP::RelativeOptimalityTolerance, 0.0);
        cp.setParameter(IloCP::LogVerbosity, IloCP::Quiet);

        IloSolution warm_start(environment);
        int next_start = 0;
        for (int position : order) {
            const erb::Job& job = instance.jobs[scenario.jobs[position]];
            if (job.resource) next_start = std::max(next_start, arrival);
            warm_start.setPresent(interval_by_position[position]);
            warm_start.setStart(interval_by_position[position], next_start);
            next_start += job.processing;
        }
        cp.setStartingPoint(warm_start);
        cp.setParameter(IloCP::TimeLimit,
            std::max(0.01, time_limit - elapsed_seconds(start)));

        const bool feasible = cp.solve();
        result.feasible = feasible;
        result.optimal = cp.getStatus() == IloAlgorithm::Optimal;
        if (feasible) result.objective = cp.getObjValue();
        result.bound = cp.getObjBound();
        result.nodes = cp.getInfo(IloCP::NumberOfBranches);

        warm_start.end();
        cp.end();
        overtime.end();
        last_resource_completion.end();
        objective.end();
        resource_ends.end();
        intervals.end();
        model.end();
    } catch (...) {
        environment.end();
        throw;
    }
    environment.end();
    return result;
}

erb::SolveRecord solve_integer_lshaped(GRBEnv& environment,
                                       const erb::Instance& instance,
                                       double time_limit, int threads,
                                       const std::string& log_file,
                                       RecourseOracle oracle) {
    const auto start = std::chrono::steady_clock::now();
    const int maximum_arrival = erb::max_ordinary_processing(instance);
    const int maximum_duration = erb::max_processing(instance);
    const int duration_count = maximum_duration + 1;
    const int contract_count = (maximum_arrival + 1) * duration_count;

    GRBModel master(environment);
    configure(master, instance, time_limit, threads);
    if (!log_file.empty() && log_file != "-") {
        master.set(GRB_IntParam_OutputFlag, 1);
        master.set(GRB_IntParam_LogToConsole, 0);
        master.set(GRB_StringParam_LogFile, log_file);
    }
    std::vector<GRBVar> contract;
    contract.reserve(contract_count);
    for (int s = 0; s <= maximum_arrival; ++s) {
        for (int d = 0; d <= maximum_duration; ++d) {
            GRBVar variable = master.addVar(
                0.0, 1.0, static_cast<double>(instance.lambda) * d, GRB_BINARY,
                "y_" + std::to_string(s) + "_" + std::to_string(d));
            variable.set(GRB_DoubleAttr_Start,
                         s == 0 && d == maximum_duration ? 1.0 : 0.0);
            contract.push_back(variable);
        }
    }
    GRBLinExpr one_contract = 0.0;
    for (GRBVar& variable : contract) one_contract += variable;
    master.addConstr(one_contract == 1.0);

    std::vector<GRBVar> theta;
    std::vector<double> global_lower;
    theta.reserve(instance.scenarios.size());
    global_lower.reserve(instance.scenarios.size());
    for (const erb::Scenario& scenario : instance.scenarios) {
        const double wspt = static_cast<double>(erb::wspt_value(instance, scenario));
        const std::vector<double> lower_bounds = erb::scenario_lower_bounds(
            instance, scenario, maximum_duration);
        global_lower.push_back(*std::min_element(
            lower_bounds.begin(), lower_bounds.end()));
        GRBVar recourse = master.addVar(
            wspt, GRB_INFINITY, scenario.probability, GRB_CONTINUOUS,
            "theta_" + std::to_string(scenario.id));
        recourse.set(GRB_DoubleAttr_Start, wspt);
        theta.push_back(recourse);
        GRBLinExpr duration_bound = 0.0;
        for (int s = 0; s <= maximum_arrival; ++s) {
            for (int d = 0; d <= maximum_duration; ++d) {
                const int index = s * duration_count + d;
                duration_bound += lower_bounds[d] * contract[index];
            }
        }
        master.addConstr(recourse >= duration_bound);
    }
    master.set(GRB_IntAttr_ModelSense, GRB_MINIMIZE);

    std::unordered_set<int> evaluated;
    erb::SolveRecord record;
    record.method = oracle == RecourseOracle::Milp
        ? "integer_lshaped_milp" : "integer_lshaped_cp";
    double incumbent = static_cast<double>(instance.lambda) * maximum_duration;
    for (const erb::Scenario& scenario : instance.scenarios) {
        incumbent += scenario.probability
            * static_cast<double>(erb::wspt_value(instance, scenario));
    }
    record.best_s = 0;
    record.best_d = maximum_duration;
    double last_bound = -std::numeric_limits<double>::infinity();
    while (true) {
        const double elapsed = elapsed_seconds(start);
        const double remaining = time_limit - elapsed;
        if (remaining <= 0.01) break;
        master.set(GRB_DoubleParam_TimeLimit, remaining);
        master.optimize();
        record.nodes_or_branches += master.get(GRB_DoubleAttr_NodeCount);
        const double current_bound = master.get(GRB_DoubleAttr_ObjBound);
        if (current_bound > -GRB_INFINITY / 2.0) {
            last_bound = std::max(last_bound, current_bound);
        }
        if (master.get(GRB_IntAttr_SolCount) == 0) break;
        if (std::isfinite(incumbent) && last_bound >= incumbent - 1e-7) {
            record.status = "optimal";
            break;
        }
        int selected = -1;
        double largest = -1.0;
        for (int index = 0; index < contract_count; ++index) {
            const double value = contract[index].get(GRB_DoubleAttr_X);
            if (value > largest) { largest = value; selected = index; }
        }
        if (selected < 0) throw std::runtime_error("Master selected no contract");
        if (evaluated.count(selected)) {
            // With no new cut to add, further progress requires additional
            // master-search time. Optimality is declared only through the
            // valid master bound checked above.
            break;
        }
        const int selected_s = selected / duration_count;
        const int selected_d = selected % duration_count;
        std::vector<RecourseResult> recourse_results;
        recourse_results.reserve(instance.scenarios.size());
        bool all_feasible = true;
        bool all_optimal = true;
        double candidate = static_cast<double>(instance.lambda) * selected_d;
        for (const erb::Scenario& scenario : instance.scenarios) {
            const double sub_elapsed = elapsed_seconds(start);
            const double sub_remaining = time_limit - sub_elapsed;
            if (sub_remaining <= 0.01) {
                all_feasible = false;
                all_optimal = false;
                break;
            }
            RecourseResult result = oracle == RecourseOracle::Milp
                ? solve_recourse(environment, instance, scenario, selected_s,
                                 selected_d, sub_remaining, threads)
                : solve_recourse_cp(instance, scenario, selected_s, selected_d,
                                    sub_remaining, threads);
            record.nodes_or_branches += result.nodes;
            all_feasible = all_feasible && result.feasible;
            all_optimal = all_optimal && result.optimal;
            if (result.feasible) candidate += scenario.probability * result.objective;
            recourse_results.push_back(result);
            if (!result.optimal) break;
        }
        if (all_feasible && recourse_results.size() == instance.scenarios.size()
            && candidate < incumbent) {
            incumbent = candidate;
            record.best_s = selected_s;
            record.best_d = selected_d;
        }
        if (!all_optimal || recourse_results.size() != instance.scenarios.size()) {
            break;
        }
        for (std::size_t omega = 0; omega < instance.scenarios.size(); ++omega) {
            const double q = recourse_results[omega].objective;
            master.addConstr(theta[omega] >= global_lower[omega]
                + (q - global_lower[omega]) * contract[selected]);
        }
        evaluated.insert(selected);
        ++record.iterations;
    }
    record.objective = incumbent;
    record.bound = last_bound;
    record.gap = erb::relative_gap(incumbent, last_bound);
    if (record.status.empty()) {
        record.status = std::isfinite(incumbent)
            ? "time_limit_feasible" : "time_limit";
    }
    record.seconds = elapsed_seconds(start);
    record.variables = master.get(GRB_IntAttr_NumVars);
    record.constraints = master.get(GRB_IntAttr_NumConstrs);
    return record;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 7) {
        std::cerr << "Usage: integer_lshaped INSTANCE OUTPUT_CSV [TIME_LIMIT] "
                     "[THREADS] [LOG_FILE] [ORACLE]\n"
                     "ORACLE is milp (default) or cp\n";
        return 1;
    }
    const double time_limit = argc >= 4 ? std::stod(argv[3]) : 1200.0;
    const int threads = argc >= 5 ? std::stoi(argv[4]) : 1;
    const std::string log_file = argc >= 6 ? argv[5] : "";
    const std::string oracle_name = argc >= 7 ? argv[6] : "milp";
    try {
        const erb::Instance instance = erb::read_instance(argv[1]);
        RecourseOracle oracle;
        if (oracle_name == "milp") oracle = RecourseOracle::Milp;
        else if (oracle_name == "cp") oracle = RecourseOracle::Cp;
        else throw std::runtime_error("ORACLE must be milp or cp");
        GRBEnv environment(true);
        environment.set(GRB_IntParam_OutputFlag, 0);
        environment.start();
        erb::SolveRecord record = solve_integer_lshaped(
            environment, instance, time_limit, threads, log_file, oracle);
        erb::append_record(argv[2], instance, record, time_limit, threads);
        std::cout << instance.name << ' ' << record.method << ' ' << record.status
                  << " objective=" << record.objective << " bound=" << record.bound
                  << " gap=" << record.gap << " seconds=" << record.seconds
                  << " iterations=" << record.iterations << '\n';
    } catch (const GRBException& error) {
        std::cerr << "Gurobi error " << error.getErrorCode() << ": "
                  << error.getMessage() << '\n';
        return 2;
    } catch (const IloException& error) {
        std::cerr << "CP Optimizer error: " << error << '\n';
        return 3;
    } catch (const std::exception& error) {
        std::cerr << "integer_lshaped: " << error.what() << '\n';
        return 4;
    }
    return 0;
}
