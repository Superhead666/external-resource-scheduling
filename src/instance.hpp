#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

namespace erb {

using i64 = std::int64_t;

struct Job {
    int id = -1;
    int processing = 0;
    int weight = 0;
    bool resource = false;
    bool committed = false;
};

struct Scenario {
    int id = -1;
    double probability = 0.0;
    std::vector<int> jobs;
};

struct Instance {
    std::string name;
    std::string instance_class;
    int seed = 0;
    int lambda = 10;
    int mu = 30;
    int committed_jobs = 0;
    int potential_jobs = 0;
    std::vector<Job> jobs;
    std::vector<Scenario> scenarios;
};

inline int scenario_processing(const Instance& instance, const Scenario& scenario) {
    int value = 0;
    for (int id : scenario.jobs) value += instance.jobs.at(id).processing;
    return value;
}

inline int scenario_weight(const Instance& instance, const Scenario& scenario) {
    int value = 0;
    for (int id : scenario.jobs) value += instance.jobs.at(id).weight;
    return value;
}

inline int scenario_resource_processing(const Instance& instance,
                                        const Scenario& scenario) {
    int value = 0;
    for (int id : scenario.jobs) {
        if (instance.jobs.at(id).resource) value += instance.jobs.at(id).processing;
    }
    return value;
}

inline int scenario_ordinary_processing(const Instance& instance,
                                        const Scenario& scenario) {
    int value = 0;
    for (int id : scenario.jobs) {
        if (!instance.jobs.at(id).resource) value += instance.jobs.at(id).processing;
    }
    return value;
}

inline int max_processing(const Instance& instance) {
    int value = 0;
    for (const Scenario& scenario : instance.scenarios) {
        value = std::max(value, scenario_processing(instance, scenario));
    }
    return value;
}

inline int max_ordinary_processing(const Instance& instance) {
    int value = 0;
    for (const Scenario& scenario : instance.scenarios) {
        value = std::max(value, scenario_ordinary_processing(instance, scenario));
    }
    return value;
}

inline double mean_scenario_jobs(const Instance& instance) {
    double value = 0.0;
    for (const Scenario& scenario : instance.scenarios) {
        value += scenario.probability * static_cast<double>(scenario.jobs.size());
    }
    return value;
}

inline int max_scenario_jobs(const Instance& instance) {
    int value = 0;
    for (const Scenario& scenario : instance.scenarios) {
        value = std::max(value, static_cast<int>(scenario.jobs.size()));
    }
    return value;
}

inline int min_scenario_jobs(const Instance& instance) {
    int value = instance.scenarios.empty()
        ? 0 : static_cast<int>(instance.scenarios.front().jobs.size());
    for (const Scenario& scenario : instance.scenarios) {
        value = std::min(value, static_cast<int>(scenario.jobs.size()));
    }
    return value;
}

inline void validate_instance(const Instance& instance) {
    if (instance.name.empty()) throw std::runtime_error("Instance name is empty");
    if (instance.jobs.empty()) throw std::runtime_error("Instance has no jobs");
    if (instance.scenarios.empty()) throw std::runtime_error("Instance has no scenarios");
    if (instance.lambda <= 0 || instance.mu <= instance.lambda) {
        throw std::runtime_error("Rates must satisfy 0 < lambda < mu");
    }
    if (instance.committed_jobs + instance.potential_jobs
        != static_cast<int>(instance.jobs.size())) {
        throw std::runtime_error("Committed and potential job counts are inconsistent");
    }
    for (int id = 0; id < static_cast<int>(instance.jobs.size()); ++id) {
        const Job& job = instance.jobs[id];
        if (job.id != id || job.processing <= 0 || job.weight <= 0) {
            throw std::runtime_error("Invalid job data");
        }
        if (id < instance.committed_jobs && !job.committed) {
            throw std::runtime_error("Committed job flag is inconsistent");
        }
    }
    double probability_sum = 0.0;
    for (const Scenario& scenario : instance.scenarios) {
        if (scenario.probability <= 0.0 || scenario.jobs.empty()) {
            throw std::runtime_error("Invalid scenario data");
        }
        probability_sum += scenario.probability;
        bool has_resource = false;
        std::vector<char> present(instance.jobs.size(), 0);
        for (int id : scenario.jobs) {
            if (id < 0 || id >= static_cast<int>(instance.jobs.size()) || present[id]) {
                throw std::runtime_error("Invalid or repeated scenario job identifier");
            }
            present[id] = 1;
            has_resource = has_resource || instance.jobs[id].resource;
        }
        for (int id = 0; id < instance.committed_jobs; ++id) {
            if (!present[id]) throw std::runtime_error("A committed job is absent");
        }
        if (!has_resource) throw std::runtime_error("A scenario has no resource job");
    }
    if (std::abs(probability_sum - 1.0) > 1e-9) {
        throw std::runtime_error("Scenario probabilities do not sum to one");
    }
}

inline void write_instance(const Instance& instance, const std::string& path) {
    validate_instance(instance);
    std::ofstream output(path);
    if (!output) throw std::runtime_error("Cannot create instance file " + path);
    output << "ERB_INSTANCE_V1\n";
    output << "name " << instance.name << '\n';
    output << "class " << instance.instance_class << '\n';
    output << "seed " << instance.seed << '\n';
    output << "lambda " << instance.lambda << '\n';
    output << "mu " << instance.mu << '\n';
    output << "committed " << instance.committed_jobs << '\n';
    output << "potential " << instance.potential_jobs << '\n';
    output << "jobs " << instance.jobs.size() << '\n';
    for (const Job& job : instance.jobs) {
        output << job.id << ' ' << job.processing << ' ' << job.weight << ' '
               << static_cast<int>(job.resource) << ' '
               << static_cast<int>(job.committed) << '\n';
    }
    output << "scenarios " << instance.scenarios.size() << '\n';
    output << std::setprecision(17);
    for (const Scenario& scenario : instance.scenarios) {
        output << scenario.id << ' ' << scenario.probability << ' '
               << scenario.jobs.size();
        for (int id : scenario.jobs) output << ' ' << id;
        output << '\n';
    }
    output << "end\n";
}

inline Instance read_instance(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open instance file " + path);
    std::string token;
    input >> token;
    if (token != "ERB_INSTANCE_V1") throw std::runtime_error("Unsupported instance format");
    Instance instance;
    auto require = [&](const std::string& expected) {
        input >> token;
        if (token != expected) {
            throw std::runtime_error("Expected token " + expected + " in " + path);
        }
    };
    require("name"); input >> instance.name;
    require("class"); input >> instance.instance_class;
    require("seed"); input >> instance.seed;
    require("lambda"); input >> instance.lambda;
    require("mu"); input >> instance.mu;
    require("committed"); input >> instance.committed_jobs;
    require("potential"); input >> instance.potential_jobs;
    require("jobs");
    int job_count = 0;
    input >> job_count;
    instance.jobs.resize(job_count);
    for (int k = 0; k < job_count; ++k) {
        int resource = 0;
        int committed = 0;
        Job job;
        input >> job.id >> job.processing >> job.weight >> resource >> committed;
        job.resource = resource != 0;
        job.committed = committed != 0;
        if (job.id < 0 || job.id >= job_count) throw std::runtime_error("Invalid job id");
        instance.jobs[job.id] = job;
    }
    require("scenarios");
    int scenario_count = 0;
    input >> scenario_count;
    instance.scenarios.resize(scenario_count);
    for (int k = 0; k < scenario_count; ++k) {
        Scenario scenario;
        int count = 0;
        input >> scenario.id >> scenario.probability >> count;
        scenario.jobs.resize(count);
        for (int& id : scenario.jobs) input >> id;
        if (scenario.id < 0 || scenario.id >= scenario_count) {
            throw std::runtime_error("Invalid scenario id");
        }
        instance.scenarios[scenario.id] = std::move(scenario);
    }
    require("end");
    validate_instance(instance);
    return instance;
}

struct SolveRecord {
    std::string instance;
    std::string method;
    std::string status;
    double objective = std::numeric_limits<double>::infinity();
    double bound = -std::numeric_limits<double>::infinity();
    double gap = std::numeric_limits<double>::infinity();
    double seconds = 0.0;
    int best_s = -1;
    int best_d = -1;
    double nodes_or_branches = 0.0;
    long long variables = 0;
    long long constraints = 0;
    std::size_t generated_labels = 0;
    std::size_t peak_labels = 0;
    double peak_memory_mb = 0.0;
    std::size_t retained_vectors = 0;
    int iterations = 0;
    int completed_scenarios = 0;
    double label_seconds = 0.0;
    double contract_seconds = 0.0;
    double maximum_scenario_seconds = 0.0;
    std::size_t maximum_scenario_generated_labels = 0;
    std::size_t maximum_scenario_vectors = 0;
};

inline double relative_gap(double objective, double bound) {
    if (!std::isfinite(objective) || !std::isfinite(bound)) {
        return std::numeric_limits<double>::infinity();
    }
    return std::max(0.0, objective - bound) / std::max(1.0, std::abs(objective));
}

inline double peak_resident_memory_mb() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) return 0.0;
#if defined(__APPLE__)
    return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);
#else
    return static_cast<double>(usage.ru_maxrss) / 1024.0;
#endif
}

inline void append_record(const std::string& output_path, const Instance& instance,
                          const SolveRecord& record, double time_limit, int threads) {
    const bool add_header = !std::filesystem::exists(output_path)
        || std::filesystem::file_size(output_path) == 0;
    std::ofstream output(output_path, std::ios::app);
    if (!output) throw std::runtime_error("Cannot open result file " + output_path);
    if (add_header) {
        output << "instance,class,seed,method,status,objective,bound,gap,seconds,"
                  "best_s,best_d,n_scenarios,n_committed,n_potential,min_jobs,"
                  "mean_jobs,max_jobs,pmax,smax,lambda,mu,time_limit,threads,"
                  "nodes_or_branches,variables,constraints,generated_labels,"
                  "peak_labels,peak_memory_mb,retained_vectors,iterations,label_seconds,"
                  "completed_scenarios,contract_seconds,max_scenario_seconds,"
                  "max_scenario_generated_labels,max_scenario_vectors\n";
    }
    output << std::setprecision(12)
           << instance.name << ',' << instance.instance_class << ',' << instance.seed
           << ',' << record.method << ',' << record.status << ',';
    auto write_number = [&](double value) {
        if (std::isfinite(value)) output << value;
        else output << "NA";
    };
    write_number(record.objective); output << ',';
    write_number(record.bound); output << ',';
    write_number(record.gap); output << ',';
    write_number(record.seconds); output << ','
           << record.best_s << ',' << record.best_d << ','
           << instance.scenarios.size() << ',' << instance.committed_jobs << ','
           << instance.potential_jobs << ',' << min_scenario_jobs(instance) << ','
           << mean_scenario_jobs(instance) << ',' << max_scenario_jobs(instance) << ','
           << max_processing(instance) << ',' << max_ordinary_processing(instance) << ','
           << instance.lambda << ',' << instance.mu << ',' << time_limit << ','
           << threads << ',' << record.nodes_or_branches << ',' << record.variables
           << ',' << record.constraints << ',' << record.generated_labels << ','
           << record.peak_labels << ','
           << std::max(record.peak_memory_mb, peak_resident_memory_mb()) << ','
           << record.retained_vectors << ','
           << record.iterations << ',' << record.label_seconds << ','
           << record.completed_scenarios << ',' << record.contract_seconds << ','
           << record.maximum_scenario_seconds
           << ',' << record.maximum_scenario_generated_labels << ','
           << record.maximum_scenario_vectors << '\n';
}

inline i64 wspt_value(const Instance& instance, const Scenario& scenario) {
    std::vector<int> order = scenario.jobs;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        const Job& left_job = instance.jobs[a];
        const Job& right_job = instance.jobs[b];
        i64 left = static_cast<i64>(left_job.processing) * right_job.weight;
        i64 right = static_cast<i64>(right_job.processing) * left_job.weight;
        return left != right ? left < right : a < b;
    });
    int completion = 0;
    i64 value = 0;
    for (int id : order) {
        completion += instance.jobs[id].processing;
        value += static_cast<i64>(instance.jobs[id].weight) * completion;
    }
    return value;
}

}  // namespace erb
