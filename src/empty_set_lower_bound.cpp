#include "instance.hpp"
#include "scenario_lower_bound.hpp"

#include <chrono>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: empty_set_lower_bound INSTANCE OUTPUT_CSV\n";
        return 1;
    }
    try {
        const auto start = std::chrono::steady_clock::now();
        const erb::Instance instance = erb::read_instance(argv[1]);
        const int maximum_duration = erb::max_processing(instance);
        std::vector<double> expected_bound(maximum_duration + 1, 0.0);
        for (const erb::Scenario& scenario : instance.scenarios) {
            const std::vector<double> scenario_bound =
                erb::scenario_lower_bounds(instance, scenario, maximum_duration);
            for (int duration = 0; duration <= maximum_duration; ++duration) {
                expected_bound[duration] += scenario.probability
                    * scenario_bound[duration];
            }
        }

        double lower_bound = std::numeric_limits<double>::infinity();
        int best_duration = -1;
        for (int duration = 0; duration <= maximum_duration; ++duration) {
            const double value = instance.lambda * duration
                + expected_bound[duration];
            if (value < lower_bound) {
                lower_bound = value;
                best_duration = duration;
            }
        }
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();

        std::ofstream output(argv[2]);
        if (!output) throw std::runtime_error(
            std::string("Cannot create output file ") + argv[2]);
        output << "instance,scenarios,lower_bound,best_d,seconds\n";
        output << std::setprecision(12) << instance.name << ','
               << instance.scenarios.size() << ',' << lower_bound << ','
               << best_duration << ',' << seconds << '\n';
    } catch (const std::exception& error) {
        std::cerr << "empty_set_lower_bound: " << error.what() << '\n';
        return 2;
    }
    return 0;
}
