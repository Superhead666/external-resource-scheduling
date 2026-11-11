#include "instance.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct MasterJob {
    int processing = 0;
    int weight = 0;
    bool resource = false;
    double arrival_multiplier = 1.0;
};

std::vector<int> integer_list(const std::string& text) {
    std::vector<int> values;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) values.push_back(std::stoi(item));
    return values;
}

std::string required(const std::map<std::string, std::string>& options,
                     const std::string& key) {
    const auto iterator = options.find(key);
    if (iterator == options.end()) {
        throw std::runtime_error("Missing option " + key);
    }
    return iterator->second;
}

std::string optional(const std::map<std::string, std::string>& options,
                     const std::string& key, const std::string& fallback) {
    const auto iterator = options.find(key);
    return iterator == options.end() ? fallback : iterator->second;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || (argc - 2) % 2 != 0) {
            throw std::runtime_error(
                "Usage: generate_paired_committed OUTPUT_DIR --option value ...");
        }
        const std::filesystem::path output_directory = argv[1];
        std::map<std::string, std::string> options;
        for (int index = 2; index < argc; index += 2) {
            options[argv[index]] = argv[index + 1];
        }
        const std::string name = optional(
            options, "--name", "sensitivity_committed_paired");
        const int target_jobs = std::stoi(required(options, "--target-jobs"));
        const int scenario_count = std::stoi(required(options, "--scenarios"));
        const int processing_maximum = std::stoi(
            required(options, "--processing-max"));
        const int weight_maximum = std::stoi(required(options, "--weight-max"));
        const double resource_share = std::stod(
            required(options, "--resource-share"));
        const int alternatives_per_uncertain_job = static_cast<int>(std::lround(
            std::stod(required(options, "--pool-multiplier"))));
        const double heterogeneity = std::stod(
            required(options, "--arrival-heterogeneity"));
        const std::vector<int> committed_counts = integer_list(
            required(options, "--committed-counts"));
        const int repetitions = std::stoi(required(options, "--repetitions"));
        const int booking_rate = std::stoi(optional(options, "--lambda", "10"));
        const int overtime_rate = std::stoi(optional(options, "--mu", "30"));
        const int seed_base = std::stoi(
            optional(options, "--seed-base", "37000000"));

        if (target_jobs <= 0 || scenario_count <= 0 || processing_maximum <= 0
            || weight_maximum <= 0 || repetitions <= 0
            || alternatives_per_uncertain_job <= 0) {
            throw std::runtime_error("Positive design parameters are required");
        }
        if (!(resource_share > 0.0 && resource_share <= 1.0)
            || !(heterogeneity >= 0.0 && heterogeneity < 1.0)
            || !(booking_rate > 0 && overtime_rate > booking_rate)) {
            throw std::runtime_error("Invalid probability or cost parameter");
        }
        if (committed_counts.empty() || !std::is_sorted(
                committed_counts.begin(), committed_counts.end())
            || committed_counts.front() <= 0
            || committed_counts.back() > target_jobs) {
            throw std::runtime_error(
                "Committed counts must be increasing and lie in [1,n]");
        }

        std::filesystem::create_directories(output_directory);
        std::ofstream manifest(output_directory / "manifest_paired.csv");
        manifest << "instance,path,class,seed,target_jobs,scenarios,repetition,"
                    "processing_max,weight_max,resource_share,pool_multiplier,"
                    "arrival_heterogeneity,committed,potential,"
                    "expected_additional,min_jobs,mean_jobs,max_jobs,pmax,smax,"
                    "lambda,mu\n";

        std::uniform_int_distribution<int> processing(1, processing_maximum);
        std::uniform_int_distribution<int> weight(1, weight_maximum);
        std::bernoulli_distribution resource(resource_share);
        std::uniform_real_distribution<double> multiplier(
            1.0 - heterogeneity, 1.0 + heterogeneity);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);

        const int block_size = alternatives_per_uncertain_job;
        const int master_jobs = target_jobs * block_size;
        for (int repetition = 1; repetition <= repetitions; ++repetition) {
            const int seed = seed_base + repetition;
            std::mt19937 rng(seed);
            std::vector<MasterJob> master(static_cast<std::size_t>(master_jobs));
            for (MasterJob& job : master) {
                job.processing = processing(rng);
                job.weight = weight(rng);
                job.arrival_multiplier = multiplier(rng);
            }

            // The representative job is candidate zero in each block.  Once a
            // block becomes known, that candidate is confirmed and the other
            // candidates in the block cease to be possible realizations.
            bool initial_resource = false;
            while (!initial_resource) {
                initial_resource = false;
                for (MasterJob& job : master) job.resource = resource(rng);
                for (int block = 0; block < committed_counts.front(); ++block) {
                    initial_resource = initial_resource
                        || master[static_cast<std::size_t>(block * block_size)].resource;
                }
            }

            std::vector<double> common_uniforms(
                static_cast<std::size_t>(scenario_count) * master_jobs);
            for (double& value : common_uniforms) value = uniform(rng);

            for (int committed : committed_counts) {
                erb::Instance instance;
                instance.instance_class = name + "_nc" + std::to_string(committed);
                instance.name = instance.instance_class + "_n"
                    + std::to_string(target_jobs) + "_K"
                    + std::to_string(committed == target_jobs ? 1 : scenario_count)
                    + "_r" + std::to_string(repetition);
                instance.seed = seed;
                instance.lambda = booking_rate;
                instance.mu = overtime_rate;
                instance.committed_jobs = committed;
                instance.potential_jobs
                    = (target_jobs - committed) * block_size;
                instance.jobs.reserve(static_cast<std::size_t>(
                    instance.committed_jobs + instance.potential_jobs));

                for (int block = 0; block < committed; ++block) {
                    const MasterJob& source = master[
                        static_cast<std::size_t>(block * block_size)];
                    const int id = static_cast<int>(instance.jobs.size());
                    instance.jobs.push_back({id, source.processing, source.weight,
                                             source.resource, true});
                }
                std::vector<int> master_index;
                master_index.reserve(static_cast<std::size_t>(
                    instance.potential_jobs));
                for (int block = committed; block < target_jobs; ++block) {
                    for (int candidate = 0; candidate < block_size; ++candidate) {
                        const int source_index = block * block_size + candidate;
                        const MasterJob& source = master[
                            static_cast<std::size_t>(source_index)];
                        const int id = static_cast<int>(instance.jobs.size());
                        instance.jobs.push_back({id, source.processing, source.weight,
                                                 source.resource, false});
                        master_index.push_back(source_index);
                    }
                }

                std::vector<double> probabilities(master_index.size(), 0.0);
                if (!master_index.empty()) {
                    double sum = 0.0;
                    for (int source_index : master_index) {
                        sum += master[static_cast<std::size_t>(source_index)]
                            .arrival_multiplier;
                    }
                    const double expected = target_jobs - committed;
                    for (std::size_t j = 0; j < master_index.size(); ++j) {
                        probabilities[j] = expected
                            * master[static_cast<std::size_t>(master_index[j])]
                                .arrival_multiplier / sum;
                        if (probabilities[j] > 1.0) {
                            throw std::runtime_error(
                                "An arrival probability exceeds one");
                        }
                    }
                }

                const int levels_scenarios
                    = committed == target_jobs ? 1 : scenario_count;
                for (int omega = 0; omega < levels_scenarios; ++omega) {
                    erb::Scenario scenario;
                    scenario.id = omega;
                    scenario.probability = 1.0 / levels_scenarios;
                    for (int id = 0; id < committed; ++id) {
                        scenario.jobs.push_back(id);
                    }
                    for (std::size_t j = 0; j < master_index.size(); ++j) {
                        const std::size_t common_index
                            = static_cast<std::size_t>(omega) * master_jobs
                            + static_cast<std::size_t>(master_index[j]);
                        if (common_uniforms[common_index] <= probabilities[j]) {
                            scenario.jobs.push_back(
                                committed + static_cast<int>(j));
                        }
                    }
                    instance.scenarios.push_back(std::move(scenario));
                }
                erb::validate_instance(instance);
                const std::filesystem::path path
                    = output_directory / (instance.name + ".dat");
                erb::write_instance(instance, path.string());
                manifest << std::setprecision(12)
                         << instance.name << ',' << path.filename().string() << ','
                         << instance.instance_class << ',' << seed << ','
                         << target_jobs << ',' << instance.scenarios.size() << ','
                         << repetition << ',' << processing_maximum << ','
                         << weight_maximum << ',' << resource_share << ','
                         << block_size << ',' << heterogeneity << ',' << committed
                         << ',' << instance.potential_jobs << ','
                         << target_jobs - committed << ','
                         << erb::min_scenario_jobs(instance) << ','
                         << erb::mean_scenario_jobs(instance) << ','
                         << erb::max_scenario_jobs(instance) << ','
                         << erb::max_processing(instance) << ','
                         << erb::max_ordinary_processing(instance) << ','
                         << booking_rate << ',' << overtime_rate << '\n';
            }
        }
        std::cout << "Generated "
                  << repetitions * static_cast<int>(committed_counts.size())
                  << " paired sensitivity instances in " << output_directory
                  << '\n';
    } catch (const std::exception& error) {
        std::cerr << "generate_paired_committed: " << error.what() << '\n';
        return 2;
    }
    return 0;
}
