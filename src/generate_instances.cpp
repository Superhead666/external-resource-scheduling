#include "instance.hpp"

#include <bit>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>

namespace {

struct Profile {
    std::string name;
    int processing_max;
    int weight_max;
    double resource_share;
    double pool_multiplier;
    double arrival_heterogeneity;
    std::string correlation;
};

struct GenerationStatistics {
    double expected_additional = 0.0;
    double minimum_probability = 0.0;
    double mean_probability = 0.0;
    double maximum_probability = 0.0;
    int distinct_scenarios = 0;
    double realized_additional_mean = 0.0;
    double realized_additional_variance = 0.0;
    double expected_pairwise_intersection = 0.0;
    double intersection_over_expected_arrivals = 0.0;
    double intersection_over_expected_union = 0.0;
};

std::pair<int, int> draw_job_data(const Profile& profile, std::mt19937& rng) {
    if (profile.correlation == "equal") {
        if (profile.processing_max != profile.weight_max) {
            throw std::runtime_error(
                "The equal correlation class requires processing-max = weight-max");
        }
        std::uniform_int_distribution<int> value(1, profile.processing_max);
        const int draw = value(rng);
        return {draw, draw};
    }
    if (profile.correlation != "independent") {
        throw std::runtime_error("Unknown processing-time/weight correlation class");
    }
    std::uniform_int_distribution<int> processing(1, profile.processing_max);
    std::uniform_int_distribution<int> weight(1, profile.weight_max);
    return {processing(rng), weight(rng)};
}

std::vector<double> arrival_probabilities(int pool_size, double expected_count,
                                          double heterogeneity,
                                          std::mt19937& rng) {
    if (pool_size == 0) return {};
    if (!(expected_count > 0.0 && expected_count <= pool_size)) {
        throw std::runtime_error("Invalid expected number of additional jobs");
    }
    if (!(heterogeneity >= 0.0 && heterogeneity < 1.0)) {
        throw std::runtime_error("Arrival heterogeneity must lie in [0,1)");
    }

    std::uniform_real_distribution<double> multiplier(
        1.0 - heterogeneity, 1.0 + heterogeneity);
    std::vector<double> probabilities(pool_size, 0.0);
    for (int attempt = 0; attempt < 10000; ++attempt) {
        double total = 0.0;
        for (double& probability : probabilities) {
            probability = heterogeneity == 0.0 ? 1.0 : multiplier(rng);
            total += probability;
        }
        const double scale = expected_count / total;
        double maximum = 0.0;
        for (double& probability : probabilities) {
            probability *= scale;
            maximum = std::max(maximum, probability);
        }
        if (maximum <= 1.0) return probabilities;
    }
    throw std::runtime_error(
        "Unable to generate arrival probabilities not exceeding one");
}

erb::Instance generate_instance(int target_jobs, int scenario_count, int repetition,
                                const Profile& profile, int seed,
                                double committed_fraction,
                                int booking_rate, int overtime_rate,
                                const std::string& scenario_mode,
                                GenerationStatistics& statistics) {
    erb::Instance instance;
    instance.instance_class = profile.name;
    instance.seed = seed;
    instance.lambda = booking_rate;
    instance.mu = overtime_rate;
    instance.committed_jobs = std::min(
        target_jobs, std::max(1, static_cast<int>(std::lround(
            committed_fraction * target_jobs))));

    statistics.expected_additional = static_cast<double>(
        target_jobs - instance.committed_jobs);
    instance.potential_jobs = statistics.expected_additional > 0.0
        ? std::max(1, static_cast<int>(std::ceil(
            profile.pool_multiplier * statistics.expected_additional)))
        : 0;
    if (statistics.expected_additional > instance.potential_jobs + 1e-9) {
        throw std::runtime_error(
            "The pool multiplier gives fewer potential jobs than expected arrivals");
    }

    std::mt19937 rng(seed);
    std::bernoulli_distribution requires_resource(profile.resource_share);
    const int job_count = instance.committed_jobs + instance.potential_jobs;
    instance.jobs.reserve(job_count);
    for (int id = 0; id < job_count; ++id) {
        const auto [processing, weight] = draw_job_data(profile, rng);
        const bool committed = id < instance.committed_jobs;
        instance.jobs.push_back(
            {id, processing, weight, false, committed});
    }

    // Condition the committed set on containing a resource job.  Redrawing the
    // resource indicators preserves the requested Bernoulli share, unlike
    // changing one completed draw after observing that none requires the resource.
    bool committed_resource = false;
    for (int attempt = 0; attempt < 100000 && !committed_resource; ++attempt) {
        committed_resource = false;
        for (int id = 0; id < instance.committed_jobs; ++id) {
            instance.jobs[id].resource = requires_resource(rng);
            committed_resource = committed_resource || instance.jobs[id].resource;
        }
    }
    if (!committed_resource) {
        throw std::runtime_error(
            "Unable to generate a committed job requiring the resource");
    }
    for (int id = instance.committed_jobs; id < job_count; ++id) {
        instance.jobs[id].resource = requires_resource(rng);
    }

    const std::vector<double> probabilities = arrival_probabilities(
        instance.potential_jobs, statistics.expected_additional,
        profile.arrival_heterogeneity, rng);
    if (!probabilities.empty()) {
        const auto [minimum, maximum] = std::minmax_element(
            probabilities.begin(), probabilities.end());
        statistics.minimum_probability = *minimum;
        statistics.maximum_probability = *maximum;
        statistics.mean_probability = std::accumulate(
            probabilities.begin(), probabilities.end(), 0.0)
            / probabilities.size();
        statistics.expected_pairwise_intersection = std::inner_product(
            probabilities.begin(), probabilities.end(), probabilities.begin(), 0.0);
        statistics.intersection_over_expected_arrivals =
            statistics.expected_pairwise_intersection
            / statistics.expected_additional;
        const double expected_union = 2.0 * statistics.expected_additional
            - statistics.expected_pairwise_intersection;
        statistics.intersection_over_expected_union =
            statistics.expected_pairwise_intersection / expected_union;
    }

    std::set<std::vector<int>> distinct_arrival_sets;
    if (scenario_mode == "sampled") {
        // Scenarios are independent Monte Carlo observations.  Duplicate arrival
        // sets are retained because removing them changes the sampled distribution.
        // Equal scenario weights therefore form the sample-average approximation.
        for (int omega = 0; omega < scenario_count; ++omega) {
            erb::Scenario scenario;
            scenario.id = omega;
            scenario.probability = 1.0 / scenario_count;
            scenario.jobs.reserve(target_jobs);
            for (int id = 0; id < instance.committed_jobs; ++id) {
                scenario.jobs.push_back(id);
            }
            std::vector<int> arrivals;
            for (int j = 0; j < instance.potential_jobs; ++j) {
                std::bernoulli_distribution arrives(probabilities[j]);
                if (arrives(rng)) {
                    const int id = instance.committed_jobs + j;
                    scenario.jobs.push_back(id);
                    arrivals.push_back(j);
                }
            }
            distinct_arrival_sets.insert(arrivals);
            instance.scenarios.push_back(std::move(scenario));
        }
    } else if (scenario_mode == "exhaustive") {
        if (instance.potential_jobs > 20) {
            throw std::runtime_error(
                "Exhaustive generation is limited to at most 20 potential jobs");
        }
        if (std::any_of(probabilities.begin(), probabilities.end(),
                        [](double probability) {
                            return !(probability > 0.0 && probability < 1.0);
                        })) {
            throw std::runtime_error(
                "Exhaustive generation requires every arrival probability in (0,1)");
        }
        const std::uint64_t exhaustive_count =
            std::uint64_t{1} << instance.potential_jobs;
        if (scenario_count != static_cast<int>(exhaustive_count)) {
            throw std::runtime_error(
                "For exhaustive generation, --scenarios must equal 2^M = "
                + std::to_string(exhaustive_count));
        }
        instance.scenarios.reserve(scenario_count);
        for (std::uint64_t mask = 0; mask < exhaustive_count; ++mask) {
            erb::Scenario scenario;
            scenario.id = static_cast<int>(mask);
            scenario.probability = 1.0;
            scenario.jobs.reserve(
                instance.committed_jobs + static_cast<int>(std::popcount(mask)));
            for (int id = 0; id < instance.committed_jobs; ++id) {
                scenario.jobs.push_back(id);
            }
            for (int j = 0; j < instance.potential_jobs; ++j) {
                const bool present = (mask & (std::uint64_t{1} << j)) != 0;
                scenario.probability *= present
                    ? probabilities[j] : 1.0 - probabilities[j];
                if (present) {
                    scenario.jobs.push_back(instance.committed_jobs + j);
                }
            }
            instance.scenarios.push_back(std::move(scenario));
        }
        statistics.distinct_scenarios = scenario_count;
    } else {
        throw std::runtime_error("Scenario mode must be sampled or exhaustive");
    }

    if (scenario_mode == "sampled") {
        statistics.distinct_scenarios =
            static_cast<int>(distinct_arrival_sets.size());
    }
    for (const erb::Scenario& scenario : instance.scenarios) {
        const double additional = static_cast<double>(
            scenario.jobs.size() - instance.committed_jobs);
        statistics.realized_additional_mean +=
            scenario.probability * additional;
    }
    for (const erb::Scenario& scenario : instance.scenarios) {
        const double additional = static_cast<double>(
            scenario.jobs.size() - instance.committed_jobs);
        const double difference = additional
            - statistics.realized_additional_mean;
        statistics.realized_additional_variance +=
            scenario.probability * difference * difference;
    }
    instance.name = profile.name + "_n" + std::to_string(target_jobs)
        + "_K" + std::to_string(instance.scenarios.size())
        + "_r" + std::to_string(repetition);
    erb::validate_instance(instance);
    return instance;
}

void usage() {
    std::cerr << "Usage: generate_instances OUTPUT_DIRECTORY "
                 "[smoke|overnight|pilot|full|main|custom] [custom options]\n";
}

std::vector<int> integer_list(const std::string& value) {
    std::vector<int> result;
    std::stringstream stream(value);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (!token.empty()) result.push_back(std::stoi(token));
    }
    if (result.empty()) throw std::runtime_error("An integer list is empty");
    return result;
}

std::vector<double> real_list(const std::string& value) {
    std::vector<double> result;
    std::stringstream stream(value);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (!token.empty()) result.push_back(std::stod(token));
    }
    if (result.empty()) throw std::runtime_error("A real-valued list is empty");
    return result;
}

std::vector<std::string> string_list(const std::string& value) {
    std::vector<std::string> result;
    std::stringstream stream(value);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (!token.empty()) result.push_back(token);
    }
    if (result.empty()) throw std::runtime_error("A string list is empty");
    return result;
}

int rate_tag(double value) {
    return static_cast<int>(std::lround(1000.0 * value));
}

std::string correlation_tag(const std::string& value) {
    return value == "independent" ? "ind" : value;
}

std::string multiplier_tag(double value) {
    const double rounded = std::round(value);
    if (std::abs(value - rounded) <= 1e-9) {
        return std::to_string(static_cast<long long>(rounded));
    }
    return std::to_string(rate_tag(value));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            usage();
            return 1;
        }
        const std::filesystem::path output_directory = argv[1];
        const std::string suite = argc >= 3 ? argv[2] : "pilot";
        std::filesystem::create_directories(output_directory);

        std::vector<int> sizes;
        std::vector<int> scenario_counts;
        std::vector<Profile> profiles;
        std::vector<double> committed_fractions{0.50};
        int repetitions = 0;
        int booking_rate = 10;
        int overtime_rate = 30;
        int seed_base = 27000000;
        std::string custom_name = "custom";
        std::string scenario_mode = "sampled";
        if (suite == "smoke") {
            sizes = {6, 8};
            scenario_counts = {3};
            profiles = {{"regular", 10, 10, 0.40, 2.0, 0.0, "independent"}};
            repetitions = 1;
        } else if (suite == "overnight" || suite == "pilot") {
            sizes = {12, 18, 24};
            scenario_counts = {5, 20};
            profiles = {
                {"regular", 10, 10, 0.40, 2.0, 0.0, "independent"},
                {"stress", 50, 10, 0.20, 2.0, 0.0, "independent"}
            };
            repetitions = suite == "overnight" ? 1 : 2;
        } else if (suite == "full") {
            sizes = {12, 18, 24, 30, 36};
            scenario_counts = {5, 10, 20};
            profiles = {
                {"regular", 10, 10, 0.40, 2.0, 0.0, "independent"},
                {"wide", 50, 10, 0.40, 2.0, 0.0, "independent"},
                {"sparse", 10, 10, 0.20, 2.0, 0.0, "independent"},
                {"stress", 50, 10, 0.20, 2.0, 0.0, "independent"}
            };
            repetitions = 5;
        } else if (suite == "main") {
            sizes = {30, 50};
            scenario_counts = {50};
            profiles = {
                {"p20", 20, 10, 0.20, 100.0, 0.50, "independent"},
                {"p25", 25, 10, 0.20, 100.0, 0.50, "independent"},
                {"p50", 50, 10, 0.20, 100.0, 0.50, "independent"}
            };
            repetitions = 3;
        } else if (suite == "custom") {
            if ((argc - 3) % 2 != 0) {
                throw std::runtime_error(
                    "Custom options must be supplied as --option value pairs");
            }
            std::map<std::string, std::string> options;
            for (int index = 3; index < argc; index += 2) {
                const std::string key = argv[index];
                if (!key.starts_with("--")) {
                    throw std::runtime_error("Invalid custom option " + key);
                }
                options[key] = argv[index + 1];
            }
            auto required = [&](const std::string& key) -> const std::string& {
                auto iterator = options.find(key);
                if (iterator == options.end()) {
                    throw std::runtime_error("Missing custom option " + key);
                }
                return iterator->second;
            };
            auto optional = [&](const std::string& key,
                                const std::string& fallback) -> std::string {
                auto iterator = options.find(key);
                return iterator == options.end() ? fallback : iterator->second;
            };

            custom_name = optional("--name", "custom");
            sizes = integer_list(required("--target-jobs"));
            scenario_counts = integer_list(required("--scenarios"));
            const std::vector<int> processing_maxima = integer_list(
                required("--processing-max"));
            const std::vector<int> weight_maxima = integer_list(
                required("--weight-max"));
            const std::vector<double> resource_shares = real_list(
                required("--resource-share"));
            const std::vector<double> pool_multipliers = real_list(
                required("--pool-multiplier"));
            const std::vector<double> arrival_heterogeneities = real_list(
                optional("--arrival-heterogeneity", "0.5"));
            const std::vector<std::string> correlations = string_list(
                optional("--correlation", "independent"));
            committed_fractions = real_list(required("--committed-fraction"));
            repetitions = std::stoi(required("--repetitions"));
            booking_rate = std::stoi(optional("--lambda", "10"));
            overtime_rate = std::stoi(optional("--mu", "30"));
            seed_base = std::stoi(optional("--seed-base", "27000000"));
            scenario_mode = optional("--scenario-mode", "sampled");
            if (scenario_mode != "sampled" && scenario_mode != "exhaustive") {
                throw std::runtime_error(
                    "Scenario mode must be sampled or exhaustive");
            }

            for (int processing_maximum : processing_maxima) {
                for (int weight_maximum : weight_maxima) {
                    for (double resource_share : resource_shares) {
                        for (double pool_multiplier : pool_multipliers) {
                            for (double heterogeneity : arrival_heterogeneities) {
                                for (const std::string& correlation : correlations) {
                                    if (processing_maximum <= 0 || weight_maximum <= 0) {
                                        throw std::runtime_error(
                                            "Processing-time and weight maxima must be positive");
                                    }
                                    if (!(resource_share > 0.0
                                          && resource_share <= 1.0)) {
                                        throw std::runtime_error(
                                            "Resource shares must lie in (0,1]");
                                    }
                                    if (pool_multiplier < 1.0) {
                                        throw std::runtime_error(
                                            "Pool multipliers must be at least one");
                                    }
                                    if (!(heterogeneity >= 0.0
                                          && heterogeneity < 1.0)) {
                                        throw std::runtime_error(
                                            "Arrival heterogeneity must lie in [0,1)");
                                    }
                                    if (correlation != "independent"
                                        && correlation != "equal") {
                                        throw std::runtime_error(
                                            "Correlation must be independent or equal");
                                    }
                                    std::ostringstream class_name;
                                    class_name << custom_name
                                               << "_p" << processing_maximum
                                               << "_w" << weight_maximum
                                               << "_rho" << rate_tag(resource_share)
                                               << "_kap" << multiplier_tag(pool_multiplier)
                                               << "_h" << rate_tag(heterogeneity)
                                               << '_' << correlation_tag(correlation);
                                    profiles.push_back({
                                        class_name.str(), processing_maximum,
                                        weight_maximum, resource_share,
                                        pool_multiplier, heterogeneity, correlation});
                                }
                            }
                        }
                    }
                }
            }
            for (double fraction : committed_fractions) {
                if (!(fraction > 0.0 && fraction <= 1.0)) {
                    throw std::runtime_error(
                        "Committed fractions must lie in (0,1]");
                }
            }
            if (repetitions <= 0 || booking_rate <= 0
                || overtime_rate <= booking_rate) {
                throw std::runtime_error(
                    "Rates must satisfy 0 < lambda < mu and replications must be positive");
            }
        } else {
            throw std::runtime_error("Unknown suite " + suite);
        }

        const std::filesystem::path manifest_path = output_directory
            / ("manifest_" + suite + ".csv");
        std::ofstream manifest(manifest_path);
        if (!manifest) throw std::runtime_error("Cannot create manifest");
        manifest << "instance,file,class,seed,target_jobs,scenarios,rep,"
                    "processing_max,weight_max,correlation,resource_share,"
                    "pool_multiplier,arrival_heterogeneity,scenario_mode,"
                    "committed_fraction,"
                    "committed,potential,expected_additional,q_min,q_mean,q_max,"
                    "distinct_scenarios,realized_additional_mean,"
                    "realized_additional_variance,expected_pairwise_intersection,"
                    "intersection_over_expected_arrivals,"
                    "intersection_over_expected_union,min_jobs,mean_jobs,max_jobs,"
                    "pmax,smax,lambda,mu\n";

        int instance_index = 0;
        for (std::size_t size_index = 0; size_index < sizes.size(); ++size_index) {
            const int target_jobs = sizes[size_index];
            for (int scenario_count : scenario_counts) {
                for (std::size_t profile_index = 0;
                     profile_index < profiles.size(); ++profile_index) {
                    const Profile& profile = profiles[profile_index];
                    for (std::size_t fraction_index = 0;
                         fraction_index < committed_fractions.size();
                         ++fraction_index) {
                        const double committed_fraction =
                            committed_fractions[fraction_index];
                        for (int repetition = 1; repetition <= repetitions; ++repetition) {
                            const int profile_code = profile.name == "regular" ? 1
                                : profile.name == "wide" ? 2
                                : profile.name == "sparse" ? 3
                                : profile.name == "stress" ? 4
                                : profile.name == "p20" ? 5
                                : profile.name == "p25" ? 6 : 7;
                            const int seed = suite == "custom"
                                ? seed_base
                                    + static_cast<int>(size_index) * 1000000
                                    + static_cast<int>(profile_index) * 10000
                                    + static_cast<int>(fraction_index) * 100
                                    + repetition
                                : 27000000 + target_jobs * 10000
                                    + scenario_count * 100 + profile_code * 10
                                    + repetition;
                            Profile named_profile = profile;
                            if (suite == "custom") {
                                named_profile.name += "_a" + std::to_string(
                                    rate_tag(committed_fraction));
                            }
                            GenerationStatistics statistics;
                            erb::Instance instance = generate_instance(
                                target_jobs, scenario_count, repetition,
                                named_profile, seed, committed_fraction,
                                booking_rate, overtime_rate, scenario_mode,
                                statistics);
                            const std::filesystem::path path = output_directory
                                / (instance.name + ".dat");
                            erb::write_instance(instance, path.string());
                            manifest << std::setprecision(12)
                                     << instance.name << ','
                                     << path.filename().string() << ','
                                     << instance.instance_class << ',' << seed << ','
                                     << target_jobs << ','
                                     << instance.scenarios.size() << ','
                                     << repetition << ',' << profile.processing_max << ','
                                     << profile.weight_max << ',' << profile.correlation << ','
                                     << profile.resource_share << ','
                                     << profile.pool_multiplier << ','
                                     << profile.arrival_heterogeneity << ','
                                     << scenario_mode << ','
                                     << committed_fraction << ','
                                     << instance.committed_jobs << ','
                                     << instance.potential_jobs << ','
                                     << statistics.expected_additional << ','
                                     << statistics.minimum_probability << ','
                                     << statistics.mean_probability << ','
                                     << statistics.maximum_probability << ','
                                     << statistics.distinct_scenarios << ','
                                     << statistics.realized_additional_mean << ','
                                     << statistics.realized_additional_variance << ','
                                     << statistics.expected_pairwise_intersection << ','
                                     << statistics.intersection_over_expected_arrivals << ','
                                     << statistics.intersection_over_expected_union << ','
                                     << erb::min_scenario_jobs(instance) << ','
                                     << erb::mean_scenario_jobs(instance) << ','
                                     << erb::max_scenario_jobs(instance) << ','
                                     << erb::max_processing(instance) << ','
                                     << erb::max_ordinary_processing(instance) << ','
                                     << instance.lambda << ',' << instance.mu << '\n';
                            ++instance_index;
                        }
                    }
                }
            }
        }
        std::cout << "Generated " << instance_index << ' ' << suite
                  << " instances in " << output_directory << '\n';
        std::cout << "Manifest: " << manifest_path << '\n';
    } catch (const std::exception& error) {
        std::cerr << "generate_instances: " << error.what() << '\n';
        return 2;
    }
    return 0;
}
