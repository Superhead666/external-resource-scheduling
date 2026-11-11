#pragma once

#include "instance.hpp"

#include <ilcp/cp.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace erb {

inline std::vector<int> add_cp_sequence_structure(
        IloEnv environment, IloModel& model, const Instance& instance,
        const Scenario& scenario,
        const std::vector<IloIntervalVar>& intervals) {
    std::vector<int> order(scenario.jobs.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int first, int second) {
        const Job& left = instance.jobs.at(scenario.jobs.at(first));
        const Job& right = instance.jobs.at(scenario.jobs.at(second));
        const i64 lhs = static_cast<i64>(left.processing) * right.weight;
        const i64 rhs = static_cast<i64>(right.processing) * left.weight;
        return lhs != rhs ? lhs < rhs : left.id < right.id;
    });

    std::vector<int> wspt_rank(order.size());
    std::vector<int> resource_positions;
    for (int rank = 0; rank < static_cast<int>(order.size()); ++rank) {
        const int position = order[rank];
        wspt_rank[position] = rank;
        if (instance.jobs.at(scenario.jobs.at(position)).resource) {
            resource_positions.push_back(position);
        }
    }
    for (int rank = 1; rank < static_cast<int>(resource_positions.size()); ++rank) {
        model.add(IloEndBeforeStart(
            environment, intervals[resource_positions[rank - 1]],
            intervals[resource_positions[rank]]));
    }

    if (resource_positions.empty()) return order;
    const int first_resource = resource_positions.front();
    const int last_resource = resource_positions.back();
    for (int position = 0; position < static_cast<int>(intervals.size()); ++position) {
        const Job& job = instance.jobs.at(scenario.jobs.at(position));
        if (job.resource
            || (wspt_rank[first_resource] < wspt_rank[position]
                && wspt_rank[position] < wspt_rank[last_resource])) {
            continue;
        }
        IloOr outside_middle(environment);
        outside_middle.add(
            IloEndOf(intervals[position]) <= IloStartOf(intervals[first_resource]));
        outside_middle.add(
            IloEndOf(intervals[last_resource]) <= IloStartOf(intervals[position]));
        model.add(outside_middle);
    }
    return order;
}

}  // namespace erb
