#include "instance.hpp"
#include "cp_structure.hpp"
#include "scenario_lower_bound.hpp"

#include <ilcp/cp.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <utility>
#include <vector>

namespace {

struct WarmInterval {
    int job_id = -1;
    IloIntervalVar interval;
};

double remaining_time(const std::chrono::steady_clock::time_point& start,
                      double time_limit) {
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    return std::max(0.01, time_limit - elapsed);
}

erb::SolveRecord solve_cp(const erb::Instance& instance, double time_limit,
                          int workers, const std::string& log_file) {
    const auto start = std::chrono::steady_clock::now();
    const int maximum_arrival = erb::max_ordinary_processing(instance);
    const int maximum_duration = erb::max_processing(instance);
    IloEnv environment;
    erb::SolveRecord record;
    record.method = "cp";
    try {
        IloModel model(environment);
        IloIntVar arrival(environment, 0, maximum_arrival, "s");
        IloIntVar duration(environment, 0, maximum_duration, "D");
        IloNumExpr objective(environment);
        objective += instance.lambda * duration;
        std::vector<std::vector<WarmInterval>> warm_intervals;
        std::vector<std::vector<int>> warm_orders;
        warm_intervals.reserve(instance.scenarios.size());
        warm_orders.reserve(instance.scenarios.size());

        for (const erb::Scenario& scenario : instance.scenarios) {
            const int horizon = maximum_arrival
                + erb::scenario_processing(instance, scenario);
            IloIntervalVarArray intervals(environment);
            IloIntExprArray resource_ends(environment);
            IloIntExpr scenario_cost(environment);
            std::vector<WarmInterval> scenario_intervals;
            std::vector<IloIntervalVar> interval_by_position;
            scenario_intervals.reserve(scenario.jobs.size());
            interval_by_position.reserve(scenario.jobs.size());
            for (int id : scenario.jobs) {
                const erb::Job& job = instance.jobs[id];
                IloIntervalVar interval(environment, job.processing,
                    ("I_" + std::to_string(scenario.id) + "_"
                        + std::to_string(id)).c_str());
                interval.setStartMin(0);
                interval.setStartMax(horizon - job.processing);
                interval.setEndMax(horizon);
                intervals.add(interval);
                scenario_intervals.push_back({id, interval});
                interval_by_position.push_back(interval);
                scenario_cost += job.weight * IloEndOf(interval);
                objective += scenario.probability * job.weight * IloEndOf(interval);
                if (job.resource) {
                    model.add(IloStartOf(interval) >= arrival);
                    resource_ends.add(IloEndOf(interval));
                }
            }
            IloIntervalSequenceVar sequence(environment, intervals,
                ("sequence_" + std::to_string(scenario.id)).c_str());
            model.add(IloNoOverlap(environment, sequence));
            warm_orders.push_back(erb::add_cp_sequence_structure(
                environment, model, instance, scenario, interval_by_position));
            IloIntExpr last_resource_completion = IloMax(resource_ends);
            IloIntExpr overtime = IloMax(
                last_resource_completion - arrival - duration, 0);
            scenario_cost += instance.mu * overtime;
            objective += scenario.probability * instance.mu * overtime;

            const std::vector<double> lower_bounds = erb::scenario_lower_bounds(
                instance, scenario, maximum_duration);
            IloNumArray lower_bound_values(environment, maximum_duration + 1);
            for (int d = 0; d <= maximum_duration; ++d) {
                lower_bound_values[d] = lower_bounds[d];
            }
            model.add(scenario_cost >= IloElement(lower_bound_values, duration));

            warm_intervals.push_back(std::move(scenario_intervals));
            lower_bound_values.end();
            scenario_cost.end();
            overtime.end();
            last_resource_completion.end();
            resource_ends.end();
            intervals.end();
        }
        model.add(IloMinimize(environment, objective));

        IloCP cp(model);
        IloSolution warm_start(environment);
        warm_start.setValue(arrival, 0);
        warm_start.setValue(duration, maximum_duration);
        for (std::size_t omega = 0; omega < instance.scenarios.size(); ++omega) {
            int next_start = 0;
            for (int position : warm_orders[omega]) {
                WarmInterval& item = warm_intervals[omega][position];
                warm_start.setPresent(item.interval);
                warm_start.setStart(item.interval, next_start);
                next_start += instance.jobs[item.job_id].processing;
            }
        }
        cp.setStartingPoint(warm_start);
        cp.setParameter(IloCP::TimeLimit, remaining_time(start, time_limit));
        cp.setParameter(IloCP::Workers, std::max(1, workers));
        cp.setParameter(IloCP::RandomSeed, instance.seed);
        cp.setParameter(IloCP::OptimalityTolerance, 0.0);
        cp.setParameter(IloCP::RelativeOptimalityTolerance, 0.0);
        if (log_file.empty() || log_file == "-") {
            cp.setParameter(IloCP::LogVerbosity, IloCP::Quiet);
        } else {
            cp.setParameter(IloCP::LogVerbosity, IloCP::Normal);
            // CP Optimizer does not accept a log filename directly through the
            // Concert parameter interface. The suite runner redirects stderr
            // and stdout to the requested per-run log file.
        }
        const bool feasible = cp.solve();
        const IloAlgorithm::Status status = cp.getStatus();
        if (status == IloAlgorithm::Optimal) record.status = "optimal";
        else if (feasible) record.status = "time_limit_feasible";
        else record.status = "time_limit";
        if (feasible) {
            record.objective = cp.getObjValue();
            record.best_s = static_cast<int>(cp.getValue(arrival));
            record.best_d = static_cast<int>(cp.getValue(duration));
        }
        record.bound = cp.getObjBound();
        record.gap = erb::relative_gap(record.objective, record.bound);
        record.nodes_or_branches = cp.getInfo(IloCP::NumberOfBranches);
        record.seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        warm_start.end();
        objective.end();
        cp.end();
        model.end();
    } catch (...) {
        environment.end();
        throw;
    }
    environment.end();
    return record;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 6) {
        std::cerr << "Usage: cp_baseline INSTANCE OUTPUT_CSV [TIME_LIMIT] "
                     "[WORKERS] [LOG_FILE]\n";
        return 1;
    }
    const double time_limit = argc >= 4 ? std::stod(argv[3]) : 1200.0;
    const int workers = argc >= 5 ? std::stoi(argv[4]) : 1;
    const std::string log_file = argc >= 6 ? argv[5] : "";
    try {
        const erb::Instance instance = erb::read_instance(argv[1]);
        erb::SolveRecord record = solve_cp(instance, time_limit, workers, log_file);
        erb::append_record(argv[2], instance, record, time_limit, workers);
        std::cout << instance.name << " cp " << record.status
                  << " objective=" << record.objective << " bound=" << record.bound
                  << " gap=" << record.gap << " seconds=" << record.seconds << '\n';
    } catch (const IloException& error) {
        std::cerr << "CP Optimizer error: " << error << '\n';
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "cp_baseline: " << error.what() << '\n';
        return 3;
    }
    return 0;
}
