#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

using i64 = std::int64_t;

namespace {

constexpr i64 INF = std::numeric_limits<i64>::max() / 8;

struct Job {
    int id;
    int p;
    int w;
    bool resource;
};

struct Descriptor {
    int before_first_r;
    int through_last_r;
    int weight_after_first_r;
    i64 twc_without_idle;
};

struct DescriptorKey {
    int before;
    int through;
    int weight_after;

    bool operator==(const DescriptorKey& other) const {
        return before == other.before && through == other.through
            && weight_after == other.weight_after;
    }
};

struct DescriptorKeyHash {
    std::size_t operator()(const DescriptorKey& x) const {
        std::size_t h = static_cast<std::size_t>(x.before);
        h = h * 1000003u + static_cast<std::size_t>(x.through);
        h = h * 1000003u + static_cast<std::size_t>(x.weight_after);
        return h;
    }
};

struct State {
    int p_before;
    int p_middle;
    int w_middle;
    int w_after;

    bool operator==(const State& other) const {
        return p_before == other.p_before && p_middle == other.p_middle
            && w_middle == other.w_middle && w_after == other.w_after;
    }
};

struct StateHash {
    std::size_t operator()(const State& x) const {
        std::size_t h = static_cast<std::size_t>(x.p_before);
        h = h * 1000003u + static_cast<std::size_t>(x.p_middle);
        h = h * 1000003u + static_cast<std::size_t>(x.w_middle);
        h = h * 1000003u + static_cast<std::size_t>(x.w_after);
        return h;
    }
};

struct Label {
    int p_before;
    int p_middle;
    int w_middle;
    int w_after;
    i64 cost;
};

struct LabelStats {
    std::size_t peak_states = 0;
    std::size_t generated_states = 0;
    std::size_t final_descriptors = 0;
    double seconds = 0.0;
};

enum class DiagramMode {
    Exact,
    Restricted,
    Relaxed
};

bool wspt_less(const Job& a, const Job& b) {
    i64 left = static_cast<i64>(a.p) * b.w;
    i64 right = static_cast<i64>(b.p) * a.w;
    if (left != right) return left < right;
    return a.id < b.id;
}

std::vector<Descriptor> merge_identical_descriptors(
        std::vector<Descriptor> input) {
    std::unordered_map<DescriptorKey, i64, DescriptorKeyHash> best;
    best.reserve(input.size());
    for (const Descriptor& d : input) {
        DescriptorKey key{d.before_first_r, d.through_last_r,
                          d.weight_after_first_r};
        auto [it, inserted] = best.emplace(key, d.twc_without_idle);
        if (!inserted && d.twc_without_idle < it->second) it->second = d.twc_without_idle;
    }
    std::vector<Descriptor>().swap(input);
    std::vector<Descriptor> candidates;
    candidates.reserve(best.size());
    for (const auto& [key, value] : best) {
        candidates.push_back({key.before, key.through, key.weight_after, value});
    }
    std::sort(candidates.begin(), candidates.end(), [](const Descriptor& a,
                                                       const Descriptor& b) {
        return std::tie(a.before_first_r, a.through_last_r,
                        a.weight_after_first_r, a.twc_without_idle)
            < std::tie(b.before_first_r, b.through_last_r,
                       b.weight_after_first_r, b.twc_without_idle);
    });
    return candidates;
}

std::vector<Descriptor> compress_descriptors(std::vector<Descriptor> input) {
    std::vector<Descriptor> candidates =
        merge_identical_descriptors(std::move(input));
    std::vector<Descriptor> result;
    result.reserve(candidates.size());
    std::size_t begin = 0;
    while (begin < candidates.size()) {
        std::size_t end = begin + 1;
        while (end < candidates.size()
               && candidates[end].before_first_r == candidates[begin].before_first_r
               && candidates[end].through_last_r == candidates[begin].through_last_r) {
            ++end;
        }
        i64 best_cost = INF;
        for (std::size_t i = begin; i < end; ++i) {
            if (candidates[i].twc_without_idle < best_cost) {
                result.push_back(candidates[i]);
                best_cost = candidates[i].twc_without_idle;
            }
        }
        begin = end;
    }
    return result;
}

std::vector<Descriptor> permutation_descriptors(const std::vector<Job>& jobs) {
    std::vector<int> permutation(jobs.size());
    for (int i = 0; i < static_cast<int>(jobs.size()); ++i) permutation[i] = i;
    std::vector<Descriptor> result;
    do {
        int first = -1;
        int last = -1;
        for (int k = 0; k < static_cast<int>(permutation.size()); ++k) {
            if (jobs[permutation[k]].resource) {
                if (first < 0) first = k;
                last = k;
            }
        }
        if (first < 0) continue;
        int time = 0;
        int before = 0;
        int through = 0;
        int weight_after = 0;
        i64 twc = 0;
        for (int k = 0; k < static_cast<int>(permutation.size()); ++k) {
            const Job& job = jobs[permutation[k]];
            if (k < first) before += job.p;
            time += job.p;
            twc += static_cast<i64>(job.w) * time;
            if (k == last) through = time;
            if (k >= first) weight_after += job.w;
        }
        result.push_back({before, through, weight_after, twc});
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    return compress_descriptors(std::move(result));
}

bool label_state_less(const Label& a, const Label& b) {
    if (a.p_before != b.p_before) return a.p_before < b.p_before;
    if (a.p_middle != b.p_middle) return a.p_middle < b.p_middle;
    if (a.w_middle != b.w_middle) return a.w_middle < b.w_middle;
    if (a.w_after != b.w_after) return a.w_after < b.w_after;
    return a.cost < b.cost;
}

class FenwickMinimum {
public:
    explicit FenwickMinimum(int size)
        : values_(size + 1, INF), stamps_(size + 1, 0) {}

    void next_group() { ++stamp_; }

    void update(int index, i64 value) {
        for (int i = index + 1; i < static_cast<int>(values_.size()); i += i & -i) {
            if (stamps_[i] != stamp_) {
                stamps_[i] = stamp_;
                values_[i] = value;
            } else {
                values_[i] = std::min(values_[i], value);
            }
        }
    }

    i64 prefix_min(int index) const {
        i64 result = INF;
        for (int i = index + 1; i > 0; i -= i & -i) {
            if (stamps_[i] == stamp_) result = std::min(result, values_[i]);
        }
        return result;
    }

private:
    std::vector<i64> values_;
    std::vector<int> stamps_;
    int stamp_ = 1;
};

class Fenwick2DMinimum {
public:
    explicit Fenwick2DMinimum(int size)
        : side_(size + 2), values_(static_cast<std::size_t>(side_) * side_, INF),
          stamps_(static_cast<std::size_t>(side_) * side_, 0) {}

    void next_group() { ++stamp_; }

    void update(int first, int second, i64 value) {
        for (int i = first + 1; i < side_; i += i & -i) {
            for (int j = second + 1; j < side_; j += j & -j) {
                std::size_t index = static_cast<std::size_t>(i) * side_ + j;
                if (stamps_[index] != stamp_) {
                    stamps_[index] = stamp_;
                    values_[index] = value;
                } else {
                    values_[index] = std::min(values_[index], value);
                }
            }
        }
    }

    i64 prefix_min(int first, int second) const {
        i64 result = INF;
        for (int i = first + 1; i > 0; i -= i & -i) {
            for (int j = second + 1; j > 0; j -= j & -j) {
                std::size_t index = static_cast<std::size_t>(i) * side_ + j;
                if (stamps_[index] == stamp_) {
                    result = std::min(result, values_[index]);
                }
            }
        }
        return result;
    }

private:
    int side_;
    std::vector<i64> values_;
    std::vector<int> stamps_;
    int stamp_ = 1;
};

class Fenwick2DCompressedMinimum {
public:
    explicit Fenwick2DCompressedMinimum(
            const std::vector<std::pair<int, int>>& points) {
        first_values_.reserve(points.size());
        for (const auto& [first, second] : points) {
            (void)second;
            first_values_.push_back(first);
        }
        std::sort(first_values_.begin(), first_values_.end());
        first_values_.erase(std::unique(first_values_.begin(), first_values_.end()),
                            first_values_.end());
        second_values_.resize(first_values_.size() + 1);
        for (const auto& [first, second] : points) {
            int index = static_cast<int>(std::lower_bound(
                first_values_.begin(), first_values_.end(), first)
                - first_values_.begin()) + 1;
            for (int i = index; i < static_cast<int>(second_values_.size());
                 i += i & -i) {
                second_values_[i].push_back(second);
            }
        }
        values_.resize(second_values_.size());
        for (std::size_t i = 1; i < second_values_.size(); ++i) {
            auto& coordinates = second_values_[i];
            std::sort(coordinates.begin(), coordinates.end());
            coordinates.erase(std::unique(coordinates.begin(), coordinates.end()),
                              coordinates.end());
            values_[i].assign(coordinates.size() + 1, INF);
        }
    }

    void update(int first, int second, i64 value) {
        int index = static_cast<int>(std::lower_bound(
            first_values_.begin(), first_values_.end(), first)
            - first_values_.begin()) + 1;
        for (int i = index; i < static_cast<int>(second_values_.size());
             i += i & -i) {
            int second_index = static_cast<int>(std::lower_bound(
                second_values_[i].begin(), second_values_[i].end(), second)
                - second_values_[i].begin()) + 1;
            for (int j = second_index; j < static_cast<int>(values_[i].size());
                 j += j & -j) {
                values_[i][j] = std::min(values_[i][j], value);
            }
        }
    }

    i64 prefix_min(int first, int second) const {
        i64 result = INF;
        int index = static_cast<int>(std::upper_bound(
            first_values_.begin(), first_values_.end(), first)
            - first_values_.begin());
        for (int i = index; i > 0; i -= i & -i) {
            int second_index = static_cast<int>(std::upper_bound(
                second_values_[i].begin(), second_values_[i].end(), second)
                - second_values_[i].begin());
            for (int j = second_index; j > 0; j -= j & -j) {
                result = std::min(result, values_[i][j]);
            }
        }
        return result;
    }

private:
    std::vector<int> first_values_;
    std::vector<std::vector<int>> second_values_;
    std::vector<std::vector<i64>> values_;
};

std::vector<Descriptor> prune_descriptor_envelopes(
        const std::vector<Descriptor>& descriptors) {
    if (descriptors.empty()) return {};
    std::vector<std::size_t> order(descriptors.size());
    std::iota(order.begin(), order.end(), 0);
    std::vector<char> dominated(descriptors.size(), 0);

    // First valid order: larger b and smaller e, W+, and F.
    std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
        const Descriptor& a = descriptors[left];
        const Descriptor& b = descriptors[right];
        if (a.before_first_r != b.before_first_r) {
            return a.before_first_r > b.before_first_r;
        }
        return std::tie(a.through_last_r, a.weight_after_first_r,
                        a.twc_without_idle)
            < std::tie(b.through_last_r, b.weight_after_first_r,
                       b.twc_without_idle);
    });
    {
        std::vector<std::pair<int, int>> first_points;
        first_points.reserve(descriptors.size());
        for (const Descriptor& descriptor : descriptors) {
            first_points.emplace_back(descriptor.through_last_r,
                                      descriptor.weight_after_first_r);
        }
        Fenwick2DCompressedMinimum first_sweep(first_points);
        for (std::size_t index : order) {
            const Descriptor& descriptor = descriptors[index];
            if (first_sweep.prefix_min(descriptor.through_last_r,
                                       descriptor.weight_after_first_r)
                <= descriptor.twc_without_idle) {
                dominated[index] = 1;
            }
            first_sweep.update(descriptor.through_last_r,
                               descriptor.weight_after_first_r,
                               descriptor.twc_without_idle);
        }
    }

    // Second valid order: smaller b, gamma=e-b, W+, and H=F-W+b.
    std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
        const Descriptor& a = descriptors[left];
        const Descriptor& b = descriptors[right];
        int gamma_a = a.through_last_r - a.before_first_r;
        int gamma_b = b.through_last_r - b.before_first_r;
        i64 h_a = a.twc_without_idle
            - static_cast<i64>(a.weight_after_first_r) * a.before_first_r;
        i64 h_b = b.twc_without_idle
            - static_cast<i64>(b.weight_after_first_r) * b.before_first_r;
        return std::tie(a.before_first_r, gamma_a,
                        a.weight_after_first_r, h_a)
            < std::tie(b.before_first_r, gamma_b,
                       b.weight_after_first_r, h_b);
    });
    {
        std::vector<std::pair<int, int>> second_points;
        second_points.reserve(descriptors.size());
        for (const Descriptor& descriptor : descriptors) {
            second_points.emplace_back(
                descriptor.through_last_r - descriptor.before_first_r,
                descriptor.weight_after_first_r);
        }
        Fenwick2DCompressedMinimum second_sweep(second_points);
        for (std::size_t index : order) {
            const Descriptor& descriptor = descriptors[index];
            int gamma = descriptor.through_last_r - descriptor.before_first_r;
            i64 h = descriptor.twc_without_idle
                - static_cast<i64>(descriptor.weight_after_first_r)
                    * descriptor.before_first_r;
            if (second_sweep.prefix_min(gamma, descriptor.weight_after_first_r)
                <= h) {
                dominated[index] = 1;
            }
            second_sweep.update(gamma, descriptor.weight_after_first_r, h);
        }
    }

    std::vector<Descriptor> result;
    result.reserve(descriptors.size());
    for (std::size_t i = 0; i < descriptors.size(); ++i) {
        if (!dominated[i]) result.push_back(descriptors[i]);
    }
    return result;
}

// Deliberately ignores the domain s >= b of each affine piece.  This routine is
// used only to test whether an ordinary (W+, H) hull is a safe global pruning
// rule; a counterexample demonstrates why the pieces must instead be treated as
// rays with descriptor-specific starting points.
std::vector<Descriptor> prune_by_naive_gamma_hull(
        const std::vector<Descriptor>& descriptors, int ptotal) {
    std::vector<char> active(descriptors.size(), 0);
    std::unordered_map<int, std::vector<std::size_t>> classes;
    for (std::size_t i = 0; i < descriptors.size(); ++i) {
        classes[descriptors[i].through_last_r
                - descriptors[i].before_first_r].push_back(i);
    }
    for (const auto& [gamma, indices] : classes) {
        (void)gamma;
        for (int s = 0; s <= ptotal; ++s) {
            i64 best = INF;
            for (std::size_t index : indices) {
                const Descriptor& descriptor = descriptors[index];
                i64 h = descriptor.twc_without_idle
                    - static_cast<i64>(descriptor.weight_after_first_r)
                        * descriptor.before_first_r;
                best = std::min(best, h
                    + static_cast<i64>(descriptor.weight_after_first_r) * s);
            }
            for (std::size_t index : indices) {
                const Descriptor& descriptor = descriptors[index];
                i64 h = descriptor.twc_without_idle
                    - static_cast<i64>(descriptor.weight_after_first_r)
                        * descriptor.before_first_r;
                if (h + static_cast<i64>(descriptor.weight_after_first_r) * s
                    == best) {
                    active[index] = 1;
                }
            }
        }
    }
    std::vector<Descriptor> result;
    for (std::size_t i = 0; i < descriptors.size(); ++i) {
        if (active[i]) result.push_back(descriptors[i]);
    }
    return result;
}

void prune_across_middle_processing(std::vector<Label>& labels,
                                    Fenwick2DMinimum& fenwick,
                                    int remaining_weight) {
    std::size_t write = 0;
    std::size_t begin = 0;
    while (begin < labels.size()) {
        std::size_t end = begin + 1;
        while (end < labels.size()
               && labels[end].p_before == labels[begin].p_before) {
            ++end;
        }
        fenwick.next_group();
        for (std::size_t i = begin; i < end; ++i) {
            const Label& label = labels[i];
            i64 adjusted_cost = label.cost
                - static_cast<i64>(label.p_middle) * remaining_weight;
            if (fenwick.prefix_min(label.w_middle, label.w_after)
                <= adjusted_cost) {
                continue;
            }
            fenwick.update(label.w_middle, label.w_after, adjusted_cost);
            labels[write++] = label;
        }
        begin = end;
    }
    labels.resize(write);
}

void prune_across_before_processing_equal_middle_weight(
        std::vector<Label>& labels,
        FenwickMinimum& fenwick,
        int remaining_weight,
        bool use_counting_sort = false,
        int total_processing = 0,
        int total_weight = 0) {
    if (use_counting_sort) {
        const std::size_t weight_range = static_cast<std::size_t>(total_weight) + 1;
        const std::size_t key_count =
            (static_cast<std::size_t>(total_processing) + 1) * weight_range;
        std::vector<std::size_t> offsets(key_count + 1, 0);
        for (const Label& label : labels) {
            std::size_t key = static_cast<std::size_t>(label.p_middle)
                    * weight_range + label.w_middle;
            ++offsets[key + 1];
        }
        for (std::size_t key = 1; key < offsets.size(); ++key) {
            offsets[key] += offsets[key - 1];
        }
        std::vector<Label> reordered(labels.size());
        for (const Label& label : labels) {
            std::size_t key = static_cast<std::size_t>(label.p_middle)
                    * weight_range + label.w_middle;
            reordered[offsets[key]++] = label;
        }
        labels.swap(reordered);
    } else {
        std::sort(labels.begin(), labels.end(), [](const Label& a, const Label& b) {
            if (a.p_middle != b.p_middle) return a.p_middle < b.p_middle;
            if (a.w_middle != b.w_middle) return a.w_middle < b.w_middle;
            if (a.p_before != b.p_before) return a.p_before < b.p_before;
            if (a.w_after != b.w_after) return a.w_after < b.w_after;
            return a.cost < b.cost;
        });
    }

    std::size_t write = 0;
    std::size_t begin = 0;
    while (begin < labels.size()) {
        std::size_t end = begin + 1;
        while (end < labels.size()
               && labels[end].p_middle == labels[begin].p_middle
               && labels[end].w_middle == labels[begin].w_middle) {
            ++end;
        }
        fenwick.next_group();
        for (std::size_t i = begin; i < end; ++i) {
            const Label& label = labels[i];
            i64 adjusted_cost = label.cost
                - static_cast<i64>(label.p_before) * remaining_weight;
            if (fenwick.prefix_min(label.w_after) <= adjusted_cost) continue;
            fenwick.update(label.w_after, adjusted_cost);
            labels[write++] = label;
        }
        begin = end;
    }
    labels.resize(write);
    std::sort(labels.begin(), labels.end(), label_state_less);
}

void prune_across_before_processing_indexed(
        std::vector<Label>& labels,
        FenwickMinimum& fenwick,
        int remaining_weight,
        int total_processing,
        int total_weight) {
    const std::size_t weight_range = static_cast<std::size_t>(total_weight) + 1;
    const std::size_t key_count =
        (static_cast<std::size_t>(total_processing) + 1) * weight_range;
    std::vector<std::size_t> offsets(key_count + 1, 0);
    for (const Label& label : labels) {
        std::size_t key = static_cast<std::size_t>(label.p_middle)
                * weight_range + label.w_middle;
        ++offsets[key + 1];
    }
    for (std::size_t key = 1; key < offsets.size(); ++key) {
        offsets[key] += offsets[key - 1];
    }
    std::vector<std::size_t> order(labels.size());
    for (std::size_t i = 0; i < labels.size(); ++i) {
        std::size_t key = static_cast<std::size_t>(labels[i].p_middle)
                * weight_range + labels[i].w_middle;
        order[offsets[key]++] = i;
    }

    std::vector<char> keep(labels.size(), 1);
    std::size_t begin = 0;
    for (std::size_t key = 0; key < key_count; ++key) {
        std::size_t end = offsets[key];
        if (begin != end) {
            fenwick.next_group();
            for (std::size_t position = begin; position < end; ++position) {
                std::size_t index = order[position];
                const Label& label = labels[index];
                i64 adjusted_cost = label.cost
                    - static_cast<i64>(label.p_before) * remaining_weight;
                if (fenwick.prefix_min(label.w_after) <= adjusted_cost) {
                    keep[index] = 0;
                } else {
                    fenwick.update(label.w_after, adjusted_cost);
                }
            }
        }
        begin = end;
    }
    std::size_t write = 0;
    for (std::size_t i = 0; i < labels.size(); ++i) {
        if (keep[i]) labels[write++] = labels[i];
    }
    labels.resize(write);
}

void prune_across_before_and_middle_processing_naive(
        std::vector<Label>& labels,
        int remaining_weight) {
    std::vector<char> dominated(labels.size(), 0);
    for (std::size_t i = 0; i < labels.size(); ++i) {
        i64 adjusted_i = labels[i].cost
            - static_cast<i64>(labels[i].p_before + labels[i].p_middle)
                * remaining_weight;
        for (std::size_t q = 0; q < labels.size(); ++q) {
            if (q == i) continue;
            i64 adjusted_q = labels[q].cost
                - static_cast<i64>(labels[q].p_before + labels[q].p_middle)
                    * remaining_weight;
            if (labels[q].p_before <= labels[i].p_before
                && labels[q].p_middle <= labels[i].p_middle
                && labels[q].w_middle <= labels[i].w_middle
                && labels[q].w_after <= labels[i].w_after
                && adjusted_q <= adjusted_i) {
                dominated[i] = 1;
                break;
            }
        }
    }
    std::size_t write = 0;
    for (std::size_t i = 0; i < labels.size(); ++i) {
        if (!dominated[i]) labels[write++] = labels[i];
    }
    labels.resize(write);
}

std::vector<std::size_t> select_group_labels(const std::vector<Label>& labels,
                                             std::size_t begin,
                                             std::size_t end,
                                             std::size_t count,
                                             int total_processing) {
    std::vector<std::size_t> selected;
    if (count == 0 || begin == end) return selected;
    count = std::min(count, end - begin);
    std::vector<char> used(end - begin, 0);
    auto add_best = [&](double alpha, double beta) {
        std::size_t best = begin;
        double best_score = std::numeric_limits<double>::infinity();
        for (std::size_t i = begin; i < end; ++i) {
            double score = static_cast<double>(labels[i].cost)
                + alpha * labels[i].w_middle + beta * labels[i].w_after;
            if (score < best_score) {
                best_score = score;
                best = i;
            }
        }
        if (!used[best - begin]) {
            used[best - begin] = 1;
            selected.push_back(best);
        }
    };
    int grid = std::max<int>(2, static_cast<int>(count));
    for (int k = 0; k < grid && selected.size() < count; ++k) {
        double fraction = grid == 1 ? 0.0 : static_cast<double>(k) / (grid - 1);
        add_best(fraction * total_processing, total_processing);
        if (selected.size() < count) add_best(0.0, fraction * total_processing);
    }
    if (selected.size() < count) {
        std::vector<std::size_t> order;
        order.reserve(end - begin);
        for (std::size_t i = begin; i < end; ++i) order.push_back(i);
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            double score_a = labels[a].cost
                + 0.5 * total_processing * labels[a].w_middle
                + total_processing * labels[a].w_after;
            double score_b = labels[b].cost
                + 0.5 * total_processing * labels[b].w_middle
                + total_processing * labels[b].w_after;
            return score_a < score_b;
        });
        for (std::size_t i : order) {
            if (selected.size() == count) break;
            if (!used[i - begin]) {
                used[i - begin] = 1;
                selected.push_back(i);
            }
        }
    }
    std::sort(selected.begin(), selected.end());
    return selected;
}

std::vector<Label> merge_and_prune_labels(std::vector<Label>& generated,
                                          FenwickMinimum& fenwick,
                                          DiagramMode mode,
                                          std::size_t width,
                                          int total_processing,
                                          bool already_sorted = false) {
    if (!already_sorted) {
        std::sort(generated.begin(), generated.end(), label_state_less);
    }
    std::size_t write = 0;
    std::size_t begin = 0;
    while (begin < generated.size()) {
        std::size_t end = begin + 1;
        while (end < generated.size()
               && generated[end].p_before == generated[begin].p_before
               && generated[end].p_middle == generated[begin].p_middle) {
            ++end;
        }
        fenwick.next_group();
        std::size_t pareto_begin = write;
        std::size_t i = begin;
        while (i < end) {
            const Label& label = generated[i];
            // For equal processing totals, lower middle/after weights and lower
            // internal cost dominate for every completion of the partial assignment.
            if (fenwick.prefix_min(label.w_after) > label.cost) {
                generated[write++] = label;
                fenwick.update(label.w_after, label.cost);
            }
            std::size_t next = i + 1;
            while (next < end
                   && generated[next].w_middle == label.w_middle
                   && generated[next].w_after == label.w_after) {
                ++next;
            }
            i = next;
        }
        std::size_t pareto_end = write;
        if (mode != DiagramMode::Exact && pareto_end - pareto_begin > width) {
            std::size_t actual_count = mode == DiagramMode::Relaxed
                ? (width > 1 ? width - 1 : 0) : width;
            std::vector<std::size_t> selected = select_group_labels(
                generated, pareto_begin, pareto_end, actual_count, total_processing);
            std::vector<char> keep(pareto_end - pareto_begin, 0);
            for (std::size_t index : selected) keep[index - pareto_begin] = 1;
            Label merged{generated[pareto_begin].p_before,
                         generated[pareto_begin].p_middle,
                         std::numeric_limits<int>::max(),
                         std::numeric_limits<int>::max(), INF};
            bool has_merged = false;
            for (std::size_t i = pareto_begin; i < pareto_end; ++i) {
                if (keep[i - pareto_begin]) continue;
                has_merged = true;
                merged.w_middle = std::min(merged.w_middle, generated[i].w_middle);
                merged.w_after = std::min(merged.w_after, generated[i].w_after);
                merged.cost = std::min(merged.cost, generated[i].cost);
            }
            std::vector<Label> replacement;
            replacement.reserve(width);
            for (std::size_t index : selected) replacement.push_back(generated[index]);
            if (mode == DiagramMode::Relaxed && has_merged) replacement.push_back(merged);
            write = pareto_begin;
            for (const Label& label : replacement) generated[write++] = label;
        }
        begin = end;
    }
    generated.resize(write);
    if (mode != DiagramMode::Exact) {
        std::sort(generated.begin(), generated.end(), label_state_less);
    }
    return std::move(generated);
}

std::vector<Label> merge_identical_labels(std::vector<Label>& generated,
                                          bool already_sorted = false) {
    if (!already_sorted) {
        std::sort(generated.begin(), generated.end(), label_state_less);
    }
    std::size_t write = 0;
    std::size_t begin = 0;
    while (begin < generated.size()) {
        // label_state_less places the least-cost copy first when all four
        // state components agree.
        generated[write++] = generated[begin];
        std::size_t end = begin + 1;
        while (end < generated.size()
               && generated[end].p_before == generated[begin].p_before
               && generated[end].p_middle == generated[begin].p_middle
               && generated[end].w_middle == generated[begin].w_middle
               && generated[end].w_after == generated[begin].w_after) {
            ++end;
        }
        begin = end;
    }
    generated.resize(write);
    return std::move(generated);
}

// For labels with equal (p_before,p_middle), every future assignment adds the
// same processing totals and the same internal-cost increments.  At completion,
// the only label-dependent terms are
//
//   u = w_middle + w_after,
//   z(gamma) = cost + w_middle (p_before+p_f)
//                    + w_after (p_before+gamma).
//
// Since z is affine in the final resource duration gamma, dominance over every
// feasible completion is certified by its values at the smallest and largest
// still-achievable gamma.  The three-dimensional Pareto test below is performed
// in O(q log q) time within a group of q labels.
struct ObjectivePoint {
    int combined_weight;
    i64 value_at_minimum;
    i64 value_at_maximum;
    std::size_t label_index;
};

struct ObjectiveDominanceWorkspace {
    std::vector<ObjectivePoint> points;
    std::vector<i64> coordinates;
    std::vector<i64> fenwick;
    std::vector<char> keep;
};

void mark_objective_aligned_group(
        const std::vector<Label>& labels, std::size_t begin, std::size_t end,
        int first_processing, int last_processing,
        int remaining_resource_processing,
        int remaining_middle_eligible_processing,
        ObjectiveDominanceWorkspace& workspace) {
    workspace.keep.assign(end - begin, 1);
    if (end - begin < 2) return;
    const int p_before = labels[begin].p_before;
    const int gamma_minimum = first_processing
        + labels[begin].p_middle + remaining_resource_processing
        + last_processing;
    const int gamma_maximum = gamma_minimum
        + remaining_middle_eligible_processing;
    workspace.points.clear();
    workspace.coordinates.clear();
    workspace.points.reserve(end - begin);
    workspace.coordinates.reserve(end - begin);
    for (std::size_t i = begin; i < end; ++i) {
        const Label& label = labels[i];
        const i64 common = label.cost
            + static_cast<i64>(label.w_middle)
                * (p_before + first_processing);
        const i64 at_minimum = common
            + static_cast<i64>(label.w_after)
                * (p_before + gamma_minimum);
        const i64 at_maximum = common
            + static_cast<i64>(label.w_after)
                * (p_before + gamma_maximum);
        workspace.points.push_back({label.w_middle + label.w_after,
                                    at_minimum, at_maximum, i});
        workspace.coordinates.push_back(at_minimum);
    }
    std::sort(workspace.points.begin(), workspace.points.end(),
              [](const ObjectivePoint& a, const ObjectivePoint& b) {
        return std::tie(a.combined_weight, a.value_at_minimum,
                        a.value_at_maximum, a.label_index)
            < std::tie(b.combined_weight, b.value_at_minimum,
                       b.value_at_maximum, b.label_index);
    });
    std::sort(workspace.coordinates.begin(), workspace.coordinates.end());
    workspace.coordinates.erase(
        std::unique(workspace.coordinates.begin(), workspace.coordinates.end()),
        workspace.coordinates.end());
    workspace.fenwick.assign(workspace.coordinates.size() + 1, INF);
    auto query = [&](std::size_t index) {
        i64 result = INF;
        for (; index > 0; index -= index & -index) {
            result = std::min(result, workspace.fenwick[index]);
        }
        return result;
    };
    auto update = [&](std::size_t index, i64 value) {
        for (; index < workspace.fenwick.size(); index += index & -index) {
            workspace.fenwick[index] = std::min(workspace.fenwick[index], value);
        }
    };
    for (const ObjectivePoint& point : workspace.points) {
        const std::size_t coordinate = static_cast<std::size_t>(
            std::lower_bound(workspace.coordinates.begin(),
                             workspace.coordinates.end(),
                             point.value_at_minimum)
            - workspace.coordinates.begin()) + 1;
        if (query(coordinate) <= point.value_at_maximum) {
            workspace.keep[point.label_index - begin] = 0;
        } else {
            update(coordinate, point.value_at_maximum);
        }
    }
}

void prune_objective_aligned_equal_totals(
        std::vector<Label>& labels, int first_processing, int last_processing,
        int remaining_resource_processing,
        int remaining_middle_eligible_processing) {
    if (labels.size() < 2) return;

    ObjectiveDominanceWorkspace workspace;
    std::size_t write = 0;
    std::size_t begin = 0;
    while (begin < labels.size()) {
        std::size_t end = begin + 1;
        while (end < labels.size()
               && labels[end].p_before == labels[begin].p_before
               && labels[end].p_middle == labels[begin].p_middle) {
            ++end;
        }
        mark_objective_aligned_group(
            labels, begin, end, first_processing, last_processing,
            remaining_resource_processing,
            remaining_middle_eligible_processing, workspace);
        for (std::size_t i = begin; i < end; ++i) {
            if (workspace.keep[i - begin]) labels[write++] = labels[i];
        }
        begin = end;
    }
    labels.resize(write);
}

std::vector<Label> truncate_label_groups(std::vector<Label>& labels,
                                         DiagramMode mode,
                                         std::size_t width,
                                         int total_processing) {
    if (mode == DiagramMode::Exact) return std::move(labels);
    std::vector<Label> result;
    result.reserve(labels.size());
    std::size_t begin = 0;
    while (begin < labels.size()) {
        std::size_t end = begin + 1;
        while (end < labels.size()
               && labels[end].p_before == labels[begin].p_before
               && labels[end].p_middle == labels[begin].p_middle) {
            ++end;
        }
        if (end - begin <= width) {
            result.insert(result.end(), labels.begin() + begin,
                          labels.begin() + end);
        } else {
            const std::size_t actual_count = mode == DiagramMode::Relaxed
                ? (width > 1 ? width - 1 : 0) : width;
            const std::vector<std::size_t> selected = select_group_labels(
                labels, begin, end, actual_count, total_processing);
            std::vector<char> keep(end - begin, 0);
            for (std::size_t index : selected) keep[index - begin] = 1;
            for (std::size_t index : selected) result.push_back(labels[index]);
            if (mode == DiagramMode::Relaxed) {
                Label merged{labels[begin].p_before, labels[begin].p_middle,
                             std::numeric_limits<int>::max(),
                             std::numeric_limits<int>::max(), INF};
                bool has_merged = false;
                for (std::size_t i = begin; i < end; ++i) {
                    if (keep[i - begin]) continue;
                    has_merged = true;
                    merged.w_middle = std::min(merged.w_middle,
                                               labels[i].w_middle);
                    merged.w_after = std::min(merged.w_after,
                                              labels[i].w_after);
                    merged.cost = std::min(merged.cost, labels[i].cost);
                }
                if (has_merged) result.push_back(merged);
            }
        }
        begin = end;
    }
    std::sort(result.begin(), result.end(), label_state_less);
    return result;
}

enum class AssignmentArc {
    Before,
    Middle,
    After
};

Label transition_label(const Label& label, const Job& job, int processed_p,
                       int processed_weight, AssignmentArc arc,
                       bool reverse_order) {
    if (reverse_order) {
        int w_before = processed_weight - label.w_middle - label.w_after;
        if (arc == AssignmentArc::Before) {
            return {label.p_before + job.p, label.p_middle,
                    label.w_middle, label.w_after,
                    label.cost + static_cast<i64>(job.p) * (w_before + job.w)};
        }
        if (arc == AssignmentArc::Middle) {
            return {label.p_before, label.p_middle + job.p,
                    label.w_middle + job.w, label.w_after,
                    label.cost + static_cast<i64>(job.p)
                        * (label.w_middle + job.w)};
        }
        return {label.p_before, label.p_middle,
                label.w_middle, label.w_after + job.w,
                label.cost + static_cast<i64>(job.p) * (label.w_after + job.w)};
    }
    int p_after = processed_p - label.p_before - label.p_middle;
    if (arc == AssignmentArc::Before) {
        return {label.p_before + job.p, label.p_middle,
                label.w_middle, label.w_after,
                label.cost + static_cast<i64>(job.w) * (label.p_before + job.p)};
    }
    if (arc == AssignmentArc::Middle) {
        return {label.p_before, label.p_middle + job.p,
                label.w_middle + job.w, label.w_after,
                label.cost + static_cast<i64>(job.w) * (label.p_middle + job.p)};
    }
    return {label.p_before, label.p_middle,
            label.w_middle, label.w_after + job.w,
            label.cost + static_cast<i64>(job.w) * (p_after + job.p)};
}

std::vector<Label> generate_sorted_successors(const std::vector<Label>& current,
                                              const Job& job,
                                              int processed_p,
                                              int processed_weight,
                                              bool single_boundary,
                                              bool reverse_order,
                                              bool allow_middle = true) {
    std::array<AssignmentArc, 3> arcs{};
    std::size_t arc_count = 0;
    if (!job.resource) {
        arcs[arc_count++] = AssignmentArc::Before;
        arcs[arc_count++] = AssignmentArc::After;
    }
    if (!single_boundary && (job.resource || allow_middle)) {
        arcs[arc_count++] = AssignmentArc::Middle;
    } else if (job.resource) {
        throw std::runtime_error("Unexpected resource job with a single boundary");
    }

    struct Stream {
        AssignmentArc arc;
        std::size_t index;
        Label head;
    };
    std::array<Stream, 3> streams{};
    std::size_t stream_count = 0;
    for (std::size_t k = 0; k < arc_count; ++k) {
        AssignmentArc arc = arcs[k];
        if (!current.empty()) {
            streams[stream_count++] = {
                arc, 0, transition_label(current.front(), job, processed_p,
                                         processed_weight, arc, reverse_order)};
        }
    }

    std::vector<Label> generated;
    generated.reserve(current.size() * arc_count);
    while (stream_count > 0) {
        std::size_t best = 0;
        for (std::size_t k = 1; k < stream_count; ++k) {
            if (label_state_less(streams[k].head, streams[best].head)) best = k;
        }
        generated.push_back(streams[best].head);
        ++streams[best].index;
        if (streams[best].index == current.size()) {
            streams[best] = streams[stream_count - 1];
            --stream_count;
        } else {
            streams[best].head = transition_label(
                current[streams[best].index], job, processed_p, processed_weight,
                streams[best].arc, reverse_order);
        }
    }
    return generated;
}

void generate_and_prune_objective_successors(
        const std::vector<Label>& current, const Job& job, int processed_p,
        int processed_weight, bool single_boundary, bool allow_middle,
        int first_processing, int last_processing,
        int remaining_resource_processing,
        int remaining_middle_eligible_processing,
        std::size_t& generated_count, std::vector<Label>& next,
        std::vector<Label>& group,
        ObjectiveDominanceWorkspace& workspace) {
    std::array<AssignmentArc, 3> arcs{};
    std::size_t arc_count = 0;
    if (!job.resource) {
        arcs[arc_count++] = AssignmentArc::Before;
        arcs[arc_count++] = AssignmentArc::After;
    }
    if (!single_boundary && (job.resource || allow_middle)) {
        arcs[arc_count++] = AssignmentArc::Middle;
    } else if (job.resource) {
        throw std::runtime_error("Unexpected resource job with a single boundary");
    }

    struct Stream {
        AssignmentArc arc;
        std::size_t index;
        Label head;
    };
    std::array<Stream, 3> streams{};
    std::size_t stream_count = 0;
    for (std::size_t k = 0; k < arc_count; ++k) {
        if (!current.empty()) {
            streams[stream_count++] = {
                arcs[k], 0,
                transition_label(current.front(), job, processed_p,
                                 processed_weight, arcs[k], false)};
        }
    }

    next.clear();
    if (next.capacity() < current.size()) next.reserve(current.size());
    group.clear();
    auto flush_group = [&]() {
        if (group.empty()) return;
        mark_objective_aligned_group(
            group, 0, group.size(), first_processing, last_processing,
            remaining_resource_processing,
            remaining_middle_eligible_processing, workspace);
        for (std::size_t i = 0; i < group.size(); ++i) {
            if (workspace.keep[i]) next.push_back(group[i]);
        }
        group.clear();
    };

    while (stream_count > 0) {
        std::size_t best = 0;
        for (std::size_t k = 1; k < stream_count; ++k) {
            if (label_state_less(streams[k].head, streams[best].head)) best = k;
        }
        const Label candidate = streams[best].head;
        if (!group.empty()
            && (candidate.p_before != group.front().p_before
                || candidate.p_middle != group.front().p_middle)) {
            flush_group();
        }
        group.push_back(candidate);
        ++generated_count;
        ++streams[best].index;
        if (streams[best].index == current.size()) {
            streams[best] = streams[stream_count - 1];
            --stream_count;
        } else {
            streams[best].head = transition_label(
                current[streams[best].index], job, processed_p,
                processed_weight, streams[best].arc, false);
        }
    }
    flush_group();
}

std::vector<Descriptor> labeling_descriptors(const std::vector<Job>& jobs,
                                             LabelStats* stats = nullptr,
                                             bool canonical_resource_order = false,
                                             DiagramMode mode = DiagramMode::Exact,
                                             std::size_t width = 0,
                                             int boundary_threads = 1,
                                             bool reverse_order = false,
                                             bool use_cross_middle_dominance = false,
                                             bool use_descriptor_dominance = false,
                                             bool restrict_middle_to_resource_span = false,
                                             bool use_cross_before_dominance = false,
                                             bool use_full_processing_dominance = false,
                                             bool use_ordinary_remaining_weight = false,
                                             bool use_counting_cross_before = false,
                                             bool use_indexed_cross_before = false,
                                             bool use_objective_aligned_dominance = false,
                                             const std::chrono::steady_clock::time_point*
                                                 deadline = nullptr,
                                             bool use_basic_equal_total_dominance = true,
                                             bool merge_identical_states = true,
                                             bool use_basic_descriptor_dominance = true) {
    auto start_time = std::chrono::steady_clock::now();
    std::vector<int> resource_jobs;
    for (int j = 0; j < static_cast<int>(jobs.size()); ++j) {
        if (jobs[j].resource) resource_jobs.push_back(j);
    }
    if (resource_jobs.empty()) throw std::runtime_error("At least one resource job is required");

    std::vector<std::pair<int, int>> boundaries;
    if (resource_jobs.size() == 1) {
        boundaries.emplace_back(resource_jobs[0], resource_jobs[0]);
    } else if (canonical_resource_order) {
        std::stable_sort(resource_jobs.begin(), resource_jobs.end(), [&](int a, int b) {
            return wspt_less(jobs[a], jobs[b]);
        });
        boundaries.emplace_back(resource_jobs.front(), resource_jobs.back());
    } else {
        for (int first : resource_jobs) {
            for (int last : resource_jobs) {
                if (first != last) boundaries.emplace_back(first, last);
            }
        }
    }

    int total_weight = 0;
    int total_processing = 0;
    for (const Job& job : jobs) total_weight += job.w;
    for (const Job& job : jobs) total_processing += job.p;

    std::vector<int> wspt_order(jobs.size());
    for (int j = 0; j < static_cast<int>(jobs.size()); ++j) wspt_order[j] = j;
    std::stable_sort(wspt_order.begin(), wspt_order.end(), [&](int a, int b) {
        return wspt_less(jobs[a], jobs[b]);
    });
    std::vector<int> wspt_rank(jobs.size());
    for (int k = 0; k < static_cast<int>(wspt_order.size()); ++k) {
        wspt_rank[wspt_order[k]] = k;
    }

    struct BoundaryResult {
        std::vector<Descriptor> descriptors;
        std::size_t peak = 0;
        std::size_t generated = 0;
    };
    auto solve_boundary = [&](int first, int last) {
        if (deadline && std::chrono::steady_clock::now() >= *deadline) {
            throw std::runtime_error("time_limit");
        }
        BoundaryResult boundary_result;
        std::vector<int> ordered;
        ordered.reserve(jobs.size() - (first == last ? 1 : 2));
        if (reverse_order) {
            for (auto iterator = wspt_order.rbegin(); iterator != wspt_order.rend();
                 ++iterator) {
                if (*iterator != first && *iterator != last) ordered.push_back(*iterator);
            }
        } else {
            for (int j : wspt_order) {
                if (j != first && j != last) ordered.push_back(j);
            }
        }

        std::vector<Label> current{{0, 0, 0, 0, 0}};
        std::vector<Label> next_buffer;
        std::vector<Label> objective_group;
        ObjectiveDominanceWorkspace objective_workspace;
        FenwickMinimum fenwick(total_weight);
        FenwickMinimum before_fenwick(total_weight);
        std::optional<Fenwick2DMinimum> cross_fenwick;
        if (use_cross_middle_dominance && !reverse_order
            && mode == DiagramMode::Exact) {
            cross_fenwick.emplace(total_weight);
        }
        int processed_p = 0;
        int processed_weight = 0;
        int ordered_weight = total_weight - jobs[first].w;
        if (last != first) ordered_weight -= jobs[last].w;
        int remaining_ordinary_weight = 0;
        int remaining_resource_processing = 0;
        int remaining_middle_eligible_processing = 0;
        for (int index : ordered) {
            if (jobs[index].resource) {
                remaining_resource_processing += jobs[index].p;
            } else {
                remaining_ordinary_weight += jobs[index].w;
                const bool may_enter_middle = !restrict_middle_to_resource_span
                    || !canonical_resource_order
                    || (wspt_rank[first] <= wspt_rank[index]
                        && wspt_rank[index] <= wspt_rank[last]);
                if (may_enter_middle) {
                    remaining_middle_eligible_processing += jobs[index].p;
                }
            }
        }

        for (int index : ordered) {
            if (deadline && std::chrono::steady_clock::now() >= *deadline) {
                throw std::runtime_error("time_limit");
            }
            const Job& job = jobs[index];
            bool allow_middle = !restrict_middle_to_resource_span
                || !canonical_resource_order || job.resource
                || (wspt_rank[first] <= wspt_rank[index]
                    && wspt_rank[index] <= wspt_rank[last]);
            const int next_remaining_resource_processing
                = remaining_resource_processing - (job.resource ? job.p : 0);
            const int next_remaining_ordinary_weight
                = remaining_ordinary_weight - (job.resource ? 0 : job.w);
            const int next_remaining_middle_eligible_processing
                = remaining_middle_eligible_processing
                  - (!job.resource && allow_middle ? job.p : 0);
            const bool stream_objective_groups
                = use_objective_aligned_dominance && !reverse_order
                  && mode == DiagramMode::Exact
                  && !use_basic_equal_total_dominance
                  && !merge_identical_states;
            std::vector<Label> generated_labels;
            std::size_t layer_generated = 0;
            if (stream_objective_groups) {
                generate_and_prune_objective_successors(
                    current, job, processed_p, processed_weight, first == last,
                    allow_middle, jobs[first].p,
                    first == last ? 0 : jobs[last].p,
                    next_remaining_resource_processing,
                    next_remaining_middle_eligible_processing,
                    layer_generated, next_buffer, objective_group,
                    objective_workspace);
                current.swap(next_buffer);
            } else {
                generated_labels = generate_sorted_successors(
                    current, job, processed_p, processed_weight, first == last,
                    reverse_order, allow_middle);
                layer_generated = generated_labels.size();
            }
            processed_p += job.p;
            processed_weight += job.w;
            remaining_resource_processing = next_remaining_resource_processing;
            remaining_ordinary_weight = next_remaining_ordinary_weight;
            remaining_middle_eligible_processing
                = next_remaining_middle_eligible_processing;
            boundary_result.generated += layer_generated;
            const DiagramMode merge_mode = use_objective_aligned_dominance
                ? DiagramMode::Exact : mode;
            if (stream_objective_groups) {
                // The successor stream has already been pruned group by group.
            } else if (use_basic_equal_total_dominance) {
                current = merge_and_prune_labels(
                    generated_labels, fenwick, merge_mode, width,
                    total_processing, true);
            } else if (merge_identical_states) {
                current = merge_identical_labels(generated_labels, true);
            } else {
                current = std::move(generated_labels);
            }
            if (use_objective_aligned_dominance && !reverse_order
                && !stream_objective_groups) {
                prune_objective_aligned_equal_totals(
                    current, jobs[first].p, first == last ? 0 : jobs[last].p,
                    remaining_resource_processing,
                    remaining_middle_eligible_processing);
            }
            if (mode != DiagramMode::Exact
                && use_objective_aligned_dominance) {
                current = truncate_label_groups(current, mode, width,
                                                total_processing);
            }
            int middle_compensation_weight = use_ordinary_remaining_weight
                ? remaining_ordinary_weight
                : ordered_weight - processed_weight;
            int before_compensation_weight = use_ordinary_remaining_weight
                ? remaining_ordinary_weight : ordered_weight - processed_weight;
            int total_remaining_weight = ordered_weight - processed_weight;
            if (cross_fenwick) {
                prune_across_middle_processing(
                    current, *cross_fenwick, middle_compensation_weight);
            }
            if (use_cross_before_dominance && !reverse_order
                && mode == DiagramMode::Exact) {
                if (use_indexed_cross_before) {
                    prune_across_before_processing_indexed(
                        current, before_fenwick, before_compensation_weight,
                        total_processing, total_weight);
                } else {
                    prune_across_before_processing_equal_middle_weight(
                        current, before_fenwick, before_compensation_weight,
                        use_counting_cross_before, total_processing, total_weight);
                }
            }
            if (use_full_processing_dominance && !reverse_order
                && mode == DiagramMode::Exact) {
                prune_across_before_and_middle_processing_naive(
                    current, total_remaining_weight);
            }
            boundary_result.peak = std::max(boundary_result.peak, current.size());
        }

        if (deadline && std::chrono::steady_clock::now() >= *deadline) {
            throw std::runtime_error("time_limit");
        }

        const Job& f = jobs[first];
        const Job& l = jobs[last];
        boundary_result.descriptors.reserve(current.size());
        for (const Label& state : current) {
            int p_inside = 0;
            int w_inside = 0;
            if (first == last) {
                p_inside = f.p;
                w_inside = f.w;
            } else {
                p_inside = f.p + state.p_middle + l.p;
                w_inside = f.w + state.w_middle + l.w;
            }

            // internal_cost also contains the B and A internal completion costs.
            // For first != last it additionally contains the middle-block cost.
            i64 twc = state.cost;
            if (first != last) {
                // Replace the middle-only component already included in internal_cost
                // by its shift behind the first boundary job, then add both boundaries.
                twc += static_cast<i64>(state.w_middle) * f.p
                    + static_cast<i64>(f.w) * f.p
                    + static_cast<i64>(l.w) * (f.p + state.p_middle + l.p);
            } else {
                twc += static_cast<i64>(f.w) * f.p;
            }
            twc += static_cast<i64>(w_inside) * state.p_before;
            twc += static_cast<i64>(state.w_after) * (state.p_before + p_inside);

            int through = state.p_before + p_inside;
            int weight_after = total_weight;
            // Jobs assigned before the first resource job are the only jobs excluded.
            int w_before = total_weight - w_inside - state.w_after;
            weight_after -= w_before;
            boundary_result.descriptors.push_back(
                {state.p_before, through, weight_after, twc});
        }
        return boundary_result;
    };

    struct WorkerResult {
        std::vector<Descriptor> descriptors;
        std::size_t peak = 0;
        std::size_t generated = 0;
        std::exception_ptr exception;
    };
    int workers = std::max(1, boundary_threads);
    workers = std::min(workers, static_cast<int>(boundaries.size()));
    std::vector<WorkerResult> worker_results(workers);
    std::atomic<std::size_t> next_boundary{0};
    auto worker_function = [&](int worker_index) {
        try {
            WorkerResult& worker = worker_results[worker_index];
            while (true) {
                std::size_t index = next_boundary.fetch_add(1);
                if (index >= boundaries.size()) break;
                const auto [first, last] = boundaries[index];
                BoundaryResult result = solve_boundary(first, last);
                worker.peak = std::max(worker.peak, result.peak);
                worker.generated += result.generated;
                worker.descriptors.insert(worker.descriptors.end(),
                                          result.descriptors.begin(),
                                          result.descriptors.end());
            }
        } catch (...) {
            worker_results[worker_index].exception = std::current_exception();
        }
    };

    if (workers == 1) {
        worker_function(0);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(workers);
        for (int worker = 0; worker < workers; ++worker) {
            threads.emplace_back(worker_function, worker);
        }
        for (std::thread& thread : threads) thread.join();
    }

    std::size_t peak = 0;
    std::size_t generated = 0;
    for (WorkerResult& worker : worker_results) {
        if (worker.exception) std::rethrow_exception(worker.exception);
        peak = std::max(peak, worker.peak);
        generated += worker.generated;
    }

    std::vector<Descriptor> all;
    if (workers == 1) {
        all = std::move(worker_results.front().descriptors);
    } else {
        std::size_t total_descriptors = 0;
        for (WorkerResult& worker : worker_results) {
            worker.descriptors = compress_descriptors(std::move(worker.descriptors));
            total_descriptors += worker.descriptors.size();
        }
        all.reserve(total_descriptors);
        for (WorkerResult& worker : worker_results) {
            all.insert(all.end(),
                       std::make_move_iterator(worker.descriptors.begin()),
                       std::make_move_iterator(worker.descriptors.end()));
            std::vector<Descriptor>().swap(worker.descriptors);
        }
    }

    std::vector<Descriptor> result = use_basic_descriptor_dominance
        ? compress_descriptors(std::move(all))
        : merge_identical_descriptors(std::move(all));
    if (use_descriptor_dominance) {
        result = prune_descriptor_envelopes(result);
    }
    if (stats) {
        auto stop_time = std::chrono::steady_clock::now();
        stats->peak_states = peak;
        stats->generated_states = generated;
        stats->final_descriptors = result.size();
        stats->seconds = std::chrono::duration<double>(stop_time - start_time).count();
    }
    return result;
}

std::vector<Descriptor> strongest_exact_descriptors(
        const std::vector<Job>& jobs, LabelStats* stats = nullptr,
        const std::chrono::steady_clock::time_point* deadline = nullptr) {
    // This wrapper is the single production definition of the exact algorithm.
    // It keeps the solver aligned with the method stated in the manuscript.
    return labeling_descriptors(
        jobs, stats,
        true,                // WSPT-extreme resource boundary jobs
        DiagramMode::Exact,
        0,
        1,
        false,               // forward WSPT recursion
        false,               // omit the slower cross-middle-processing pass
        true,                // terminal-vector dominance
        true,                // restrict M to the WSPT span of the boundaries
        false,               // omit the slower cross-before-processing pass
        false,               // omit the quadratic diagnostic dominance pass
        true,                // retain the exact remaining-job calculations
        false,               // comparison-sort implementation is not selected
        false,               // indexed processing-total pass is not selected
        true,                // objective-aligned equal-total dominance
        deadline,
        false,               // subsumed equal-total dominance pass
        false);              // prune successor groups directly while merging
}

// When there is a single resource job, no ordinary job can be processed inside
// the resource interval.  The four-dimensional label therefore reduces to
// (p_before,w_after), with the minimum internal completion cost stored for each
// pair.  This dense recursion is pseudo-polynomial in the total processing time
// and total weight of the ordinary jobs.
std::vector<Descriptor> single_resource_descriptors(
        const std::vector<Job>& jobs, LabelStats* stats = nullptr,
        bool use_descriptor_dominance = true,
        const std::chrono::steady_clock::time_point* deadline = nullptr) {
    const auto start_time = std::chrono::steady_clock::now();
    int resource_index = -1;
    int resource_count = 0;
    int ordinary_processing = 0;
    int ordinary_weight = 0;
    for (int j = 0; j < static_cast<int>(jobs.size()); ++j) {
        if (jobs[j].resource) {
            resource_index = j;
            ++resource_count;
        } else {
            ordinary_processing += jobs[j].p;
            ordinary_weight += jobs[j].w;
        }
    }
    if (resource_count != 1) {
        throw std::runtime_error(
            "single_resource_descriptors requires exactly one resource job");
    }

    std::vector<int> order;
    order.reserve(jobs.size() - 1);
    for (int j = 0; j < static_cast<int>(jobs.size()); ++j) {
        if (j != resource_index) order.push_back(j);
    }
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return wspt_less(jobs[a], jobs[b]);
    });

    const std::size_t stride = static_cast<std::size_t>(ordinary_weight + 1);
    const std::size_t state_count = static_cast<std::size_t>(ordinary_processing + 1)
        * stride;
    std::vector<i64> current(state_count, INF);
    std::vector<i64> next(state_count, INF);
    current[0] = 0;
    int processed_processing = 0;
    int processed_weight = 0;
    std::size_t peak = 1;
    std::size_t generated = 0;
    for (int index : order) {
        if (deadline && std::chrono::steady_clock::now() >= *deadline) {
            throw std::runtime_error("time_limit");
        }
        std::fill(next.begin(), next.end(), INF);
        std::size_t retained = 0;
        const Job& job = jobs[index];
        for (int p_before = 0; p_before <= processed_processing; ++p_before) {
            if (deadline && (p_before & 31) == 0
                && std::chrono::steady_clock::now() >= *deadline) {
                throw std::runtime_error("time_limit");
            }
            const int p_after = processed_processing - p_before;
            const std::size_t row = static_cast<std::size_t>(p_before) * stride;
            for (int w_after = 0; w_after <= processed_weight; ++w_after) {
                const i64 value = current[row + w_after];
                if (value >= INF / 2) continue;
                generated += 2;
                const std::size_t before_index =
                    static_cast<std::size_t>(p_before + job.p) * stride + w_after;
                const i64 before_value = value
                    + static_cast<i64>(job.w) * (p_before + job.p);
                if (before_value < next[before_index]) {
                    if (next[before_index] >= INF / 2) ++retained;
                    next[before_index] = before_value;
                }
                const std::size_t after_index = row + w_after + job.w;
                const i64 after_value = value
                    + static_cast<i64>(job.w) * (p_after + job.p);
                if (after_value < next[after_index]) {
                    if (next[after_index] >= INF / 2) ++retained;
                    next[after_index] = after_value;
                }
            }
        }
        current.swap(next);
        processed_processing += job.p;
        processed_weight += job.w;
        peak = std::max(peak, retained);
    }

    const Job& resource = jobs[resource_index];
    std::vector<Descriptor> descriptors;
    descriptors.reserve(peak);
    for (int p_before = 0; p_before <= ordinary_processing; ++p_before) {
        const std::size_t row = static_cast<std::size_t>(p_before) * stride;
        for (int w_after = 0; w_after <= ordinary_weight; ++w_after) {
            const i64 value = current[row + w_after];
            if (value >= INF / 2) continue;
            const i64 twc = value
                + static_cast<i64>(resource.w) * (p_before + resource.p)
                + static_cast<i64>(w_after) * (p_before + resource.p);
            descriptors.push_back({p_before, p_before + resource.p,
                                   resource.w + w_after, twc});
        }
    }
    descriptors = compress_descriptors(std::move(descriptors));
    if (use_descriptor_dominance) {
        descriptors = prune_descriptor_envelopes(descriptors);
    }
    if (stats) {
        stats->peak_states = peak;
        stats->generated_states = generated;
        stats->final_descriptors = descriptors.size();
        stats->seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_time).count();
    }
    return descriptors;
}

std::vector<Descriptor> labeling_descriptors_shared_last(
        const std::vector<Job>& jobs, LabelStats* stats = nullptr) {
    auto start_time = std::chrono::steady_clock::now();
    std::vector<int> resource_jobs;
    int total_weight = 0;
    int total_processing = 0;
    for (int j = 0; j < static_cast<int>(jobs.size()); ++j) {
        total_weight += jobs[j].w;
        total_processing += jobs[j].p;
        if (jobs[j].resource) resource_jobs.push_back(j);
    }
    if (resource_jobs.empty()) {
        throw std::runtime_error("At least one resource job is required");
    }
    if (resource_jobs.size() == 1) {
        return labeling_descriptors(jobs, stats);
    }

    std::vector<int> wspt_order(jobs.size());
    for (int j = 0; j < static_cast<int>(jobs.size()); ++j) wspt_order[j] = j;
    std::stable_sort(wspt_order.begin(), wspt_order.end(), [&](int a, int b) {
        return wspt_less(jobs[a], jobs[b]);
    });

    std::vector<Descriptor> all;
    std::size_t peak = 0;
    std::size_t generated = 0;
    for (int first : resource_jobs) {
        std::vector<Label> unselected{{0, 0, 0, 0, 0}};
        std::vector<std::vector<Label>> selected(jobs.size());
        FenwickMinimum fenwick(total_weight);
        int processed_all = 0;

        auto advance = [&](std::vector<Label>& labels, const Job& job,
                           int effective_processed) {
            if (labels.empty()) return;
            std::vector<Label> successors = generate_sorted_successors(
                labels, job, effective_processed, 0, false, false);
            generated += successors.size();
            labels = merge_and_prune_labels(successors, fenwick,
                                            DiagramMode::Exact, 0,
                                            total_processing, true);
        };

        for (int index : wspt_order) {
            if (index == first) continue;
            const Job& job = jobs[index];
            if (job.resource) {
                // Selecting the current resource job as the last boundary skips
                // it from the WSPT-ordered middle block.
                std::vector<Label> choose_as_last = unselected;
                generated += choose_as_last.size();
                advance(unselected, job, processed_all);
                for (int last : resource_jobs) {
                    if (last == first || last == index || selected[last].empty()) continue;
                    advance(selected[last], job, processed_all - jobs[last].p);
                }
                selected[index] = std::move(choose_as_last);
            } else {
                advance(unselected, job, processed_all);
                for (int last : resource_jobs) {
                    if (last == first || selected[last].empty()) continue;
                    advance(selected[last], job, processed_all - jobs[last].p);
                }
            }
            processed_all += job.p;
            std::size_t layer_states = unselected.size();
            for (int last : resource_jobs) layer_states += selected[last].size();
            peak = std::max(peak, layer_states);
        }

        const Job& f = jobs[first];
        for (int last : resource_jobs) {
            if (last == first) continue;
            const Job& l = jobs[last];
            for (const Label& state : selected[last]) {
                int p_inside = f.p + state.p_middle + l.p;
                int w_inside = f.w + state.w_middle + l.w;
                i64 twc = state.cost
                    + static_cast<i64>(state.w_middle) * f.p
                    + static_cast<i64>(f.w) * f.p
                    + static_cast<i64>(l.w) * (f.p + state.p_middle + l.p)
                    + static_cast<i64>(w_inside) * state.p_before
                    + static_cast<i64>(state.w_after)
                        * (state.p_before + p_inside);
                int through = state.p_before + p_inside;
                int w_before = total_weight - w_inside - state.w_after;
                int weight_after = total_weight - w_before;
                all.push_back({state.p_before, through, weight_after, twc});
            }
        }
    }

    std::vector<Descriptor> result = compress_descriptors(std::move(all));
    if (stats) {
        auto stop_time = std::chrono::steady_clock::now();
        stats->peak_states = peak;
        stats->generated_states = generated;
        stats->final_descriptors = result.size();
        stats->seconds = std::chrono::duration<double>(stop_time - start_time).count();
    }
    return result;
}

std::vector<i64> frontier_for_s(const std::vector<Descriptor>& descriptors,
                                int s, int ptotal) {
    std::vector<i64> phi(ptotal + 1, INF);
    for (const Descriptor& d : descriptors) {
        int idle = std::max(0, s - d.before_first_r);
        int exposure = d.through_last_r + idle - s;
        if (exposure < 0 || exposure > ptotal) {
            throw std::runtime_error("Exposure outside the expected range");
        }
        i64 twc = d.twc_without_idle
            + static_cast<i64>(d.weight_after_first_r) * idle;
        phi[exposure] = std::min(phi[exposure], twc);
    }
    return phi;
}

bool compare_frontiers(const std::vector<Job>& jobs,
                       const std::vector<Descriptor>& exact,
                       const std::vector<Descriptor>& structured,
                       std::string* message) {
    int ptotal = 0;
    for (const Job& job : jobs) ptotal += job.p;
    for (int s = 0; s <= ptotal; ++s) {
        std::vector<i64> a = frontier_for_s(exact, s, ptotal);
        std::vector<i64> b = frontier_for_s(structured, s, ptotal);
        for (int x = 0; x <= ptotal; ++x) {
            if (a[x] != b[x]) {
                if (message) {
                    *message = "Mismatch at s=" + std::to_string(s)
                        + " x=" + std::to_string(x)
                        + " exact=" + std::to_string(a[x])
                        + " structured=" + std::to_string(b[x]);
                }
                return false;
            }
        }
    }
    return true;
}

bool compare_cumulative_frontiers(const std::vector<Job>& jobs,
                                  const std::vector<Descriptor>& exact,
                                  const std::vector<Descriptor>& candidate,
                                  std::string* message) {
    int ptotal = 0;
    for (const Job& job : jobs) ptotal += job.p;
    for (int s = 0; s <= ptotal; ++s) {
        std::vector<i64> a = frontier_for_s(exact, s, ptotal);
        std::vector<i64> b = frontier_for_s(candidate, s, ptotal);
        i64 cumulative_a = INF;
        i64 cumulative_b = INF;
        for (int x = 0; x <= ptotal; ++x) {
            cumulative_a = std::min(cumulative_a, a[x]);
            cumulative_b = std::min(cumulative_b, b[x]);
            if (cumulative_a != cumulative_b) {
                if (message) {
                    *message = "Cumulative mismatch at s=" + std::to_string(s)
                        + " x=" + std::to_string(x)
                        + " exact=" + std::to_string(cumulative_a)
                        + " candidate=" + std::to_string(cumulative_b);
                }
                return false;
            }
        }
    }
    return true;
}

void print_instance(const std::vector<Job>& jobs) {
    for (const Job& job : jobs) {
        std::cerr << "  id=" << job.id << " p=" << job.p << " w=" << job.w
                  << " r=" << job.resource << '\n';
    }
}

std::vector<Job> random_instance(int n, int seed, int scale, double resource_share) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> processing(1, scale);
    std::uniform_int_distribution<int> weight(1, scale);
    std::bernoulli_distribution resource(resource_share);
    std::vector<Job> jobs;
    bool any_resource = false;
    for (int j = 0; j < n; ++j) {
        Job job{j, processing(rng), weight(rng), resource(rng)};
        any_resource = any_resource || job.resource;
        jobs.push_back(job);
    }
    if (!any_resource) jobs[0].resource = true;
    return jobs;
}

void validate() {
    int tested = 0;
    int exhaustive = 0;
    constexpr int exhaustive_n = 5;
    int data_patterns = 1;
    for (int j = 0; j < exhaustive_n; ++j) data_patterns *= 4;
    for (int pattern = 0; pattern < data_patterns; ++pattern) {
        std::vector<Job> jobs;
        int value = pattern;
        for (int j = 0; j < exhaustive_n; ++j) {
            int type = value % 4;
            value /= 4;
            jobs.push_back({j, (type & 1) ? 4 : 1, (type & 2) ? 4 : 1, false});
        }
        for (int mask = 1; mask < (1 << exhaustive_n); ++mask) {
            for (int j = 0; j < exhaustive_n; ++j) jobs[j].resource = mask & (1 << j);
            std::vector<Descriptor> exact = permutation_descriptors(jobs);
            std::vector<Descriptor> structured = labeling_descriptors(jobs);
            std::string message;
            if (!compare_frontiers(jobs, exact, structured, &message)) {
                std::cerr << "Exhaustive structural validation failed: " << message << '\n';
                print_instance(jobs);
                std::exit(2);
            }
            std::vector<Descriptor> canonical = labeling_descriptors(
                jobs, nullptr, true);
            if (!compare_cumulative_frontiers(jobs, exact, canonical, &message)) {
                std::cerr << "Exhaustive canonical-boundary validation failed: "
                          << message << '\n';
                print_instance(jobs);
                std::exit(2);
            }
            std::vector<Descriptor> cross = labeling_descriptors(
                jobs, nullptr, false, DiagramMode::Exact, 0, 1, false, true);
            if (!compare_cumulative_frontiers(jobs, exact, cross, &message)) {
                std::cerr << "Exhaustive cross-middle validation failed: "
                          << message << '\n';
                print_instance(jobs);
                std::exit(2);
            }
            std::vector<Descriptor> envelope =
                prune_descriptor_envelopes(structured);
            if (!compare_cumulative_frontiers(jobs, exact, envelope, &message)) {
                std::cerr << "Exhaustive descriptor-dominance validation failed: "
                          << message << '\n';
                print_instance(jobs);
                std::exit(2);
            }
            std::vector<Descriptor> combined = labeling_descriptors(
                jobs, nullptr, true, DiagramMode::Exact, 0, 1, false, true, true);
            if (!compare_cumulative_frontiers(jobs, exact, combined, &message)) {
                std::cerr << "Exhaustive combined-method validation failed: "
                          << message << '\n';
                print_instance(jobs);
                std::exit(2);
            }
            ++exhaustive;
        }
    }
    std::cout << "exhaustive_validation n=5 instances=" << exhaustive << '\n';
    for (int n : {5, 6, 7, 8, 9}) {
        int repetitions = n <= 7 ? 100 : (n == 8 ? 60 : 20);
        for (int rep = 0; rep < repetitions; ++rep) {
            int scale = rep % 3 == 0 ? 5 : (rep % 3 == 1 ? 10 : 50);
            double share = rep % 4 == 0 ? 0.20 : (rep % 4 == 1 ? 0.40 : 0.70);
            std::vector<Job> jobs = random_instance(n, 810000 + n * 1000 + rep,
                                                    scale, share);
            std::vector<Descriptor> exact = permutation_descriptors(jobs);
            LabelStats stats;
            std::vector<Descriptor> structured = labeling_descriptors(jobs, &stats);
            std::string message;
            if (!compare_frontiers(jobs, exact, structured, &message)) {
                std::cerr << "Structural validation failed for n=" << n
                          << " rep=" << rep << ": " << message << '\n';
                print_instance(jobs);
                std::exit(2);
            }
            std::vector<Descriptor> parallel = labeling_descriptors(
                jobs, nullptr, false, DiagramMode::Exact, 0, 4);
            if (!compare_frontiers(jobs, structured, parallel, &message)) {
                std::cerr << "Parallel validation failed for n=" << n
                          << " rep=" << rep << ": " << message << '\n';
                print_instance(jobs);
                std::exit(2);
            }
            std::vector<Descriptor> shared = labeling_descriptors_shared_last(jobs);
            if (!compare_frontiers(jobs, structured, shared, &message)) {
                std::cerr << "Shared-last validation failed for n=" << n
                          << " rep=" << rep << ": " << message << '\n';
                print_instance(jobs);
                std::exit(2);
            }
            std::vector<Descriptor> reverse = labeling_descriptors(
                jobs, nullptr, false, DiagramMode::Exact, 0, 1, true);
            if (!compare_frontiers(jobs, structured, reverse, &message)) {
                std::cerr << "Reverse-order validation failed for n=" << n
                          << " rep=" << rep << ": " << message << '\n';
                print_instance(jobs);
                std::exit(2);
            }
            std::vector<Descriptor> canonical = labeling_descriptors(
                jobs, nullptr, true);
            if (!compare_cumulative_frontiers(jobs, structured, canonical, &message)) {
                std::cerr << "Canonical-boundary validation failed for n=" << n
                          << " rep=" << rep << ": " << message << '\n';
                print_instance(jobs);
                std::exit(2);
            }
            std::vector<Descriptor> cross = labeling_descriptors(
                jobs, nullptr, false, DiagramMode::Exact, 0, 1, false, true);
            if (!compare_cumulative_frontiers(jobs, structured, cross, &message)) {
                std::cerr << "Cross-middle validation failed for n=" << n
                          << " rep=" << rep << ": " << message << '\n';
                print_instance(jobs);
                std::exit(2);
            }
            std::vector<Descriptor> envelope =
                prune_descriptor_envelopes(structured);
            if (!compare_cumulative_frontiers(jobs, structured, envelope, &message)) {
                std::cerr << "Descriptor-dominance validation failed for n=" << n
                          << " rep=" << rep << ": " << message << '\n';
                print_instance(jobs);
                std::exit(2);
            }
            std::vector<Descriptor> combined = labeling_descriptors(
                jobs, nullptr, true, DiagramMode::Exact, 0, 1, false, true, true);
            if (!compare_cumulative_frontiers(jobs, structured, combined, &message)) {
                std::cerr << "Combined-method validation failed for n=" << n
                          << " rep=" << rep << ": " << message << '\n';
                print_instance(jobs);
                std::exit(2);
            }
            ++tested;
        }
        std::cout << "validated n=" << n << " instances=" << repetitions << '\n';
    }
    std::cout << "structural_validation_passed,total_instances="
              << tested + exhaustive << '\n';
}

void find_naive_hull_counterexample() {
    constexpr int n = 5;
    int data_patterns = 1;
    for (int j = 0; j < n; ++j) data_patterns *= 4;
    int tested = 0;
    for (int pattern = 0; pattern < data_patterns; ++pattern) {
        std::vector<Job> jobs;
        int value = pattern;
        int ptotal = 0;
        for (int j = 0; j < n; ++j) {
            int type = value % 4;
            value /= 4;
            jobs.push_back({j, (type & 1) ? 4 : 1,
                            (type & 2) ? 4 : 1, false});
            ptotal += jobs.back().p;
        }
        for (int mask = 1; mask < (1 << n); ++mask) {
            for (int j = 0; j < n; ++j) jobs[j].resource = mask & (1 << j);
            std::vector<Descriptor> exact = permutation_descriptors(jobs);
            std::vector<Descriptor> hull =
                prune_by_naive_gamma_hull(exact, ptotal);
            std::string message;
            ++tested;
            if (!compare_cumulative_frontiers(jobs, exact, hull, &message)) {
                std::cout << "naive_hull_counterexample tested=" << tested
                          << " descriptors=" << exact.size()
                          << " retained=" << hull.size()
                          << " " << message << '\n';
                print_instance(jobs);
                return;
            }
        }
    }
    std::cout << "naive_hull_no_counterexample tested=" << tested << '\n';
}

void benchmark(const std::string& output) {
    std::ofstream out(output);
    out << "n,rep,scale,resource_share,ptotal,resource_jobs,seconds,peak_states,"
           "generated_states,descriptors\n";
    for (int n : {10, 12, 15, 18, 20, 25, 30}) {
        for (int rep = 0; rep < 10; ++rep) {
            int scale = rep < 5 ? 10 : 50;
            double share = rep % 3 == 0 ? 0.20 : (rep % 3 == 1 ? 0.40 : 0.70);
            std::vector<Job> jobs = random_instance(n, 910000 + n * 100 + rep,
                                                    scale, share);
            int ptotal = 0;
            int nr = 0;
            for (const Job& job : jobs) {
                ptotal += job.p;
                nr += job.resource;
            }
            LabelStats stats;
            std::vector<Descriptor> descriptors = labeling_descriptors(jobs, &stats);
            out << n << ',' << rep << ',' << scale << ',' << share << ','
                << ptotal << ',' << nr << ',' << std::fixed << std::setprecision(6)
                << stats.seconds << ',' << stats.peak_states << ','
                << stats.generated_states << ',' << descriptors.size() << '\n';
            out.flush();
            std::cout << "n=" << n << " rep=" << rep << " P=" << ptotal
                      << " nr=" << nr << " sec=" << stats.seconds
                      << " peak=" << stats.peak_states
                      << " desc=" << descriptors.size() << '\n';
        }
    }
}

i64 exact_recourse(const std::vector<Descriptor>& descriptors,
                   int s, int d, int mu) {
    i64 best = INF;
    for (const Descriptor& descriptor : descriptors) {
        int idle = std::max(0, s - descriptor.before_first_r);
        int exposure = descriptor.through_last_r + idle - s;
        i64 twc = descriptor.twc_without_idle
            + static_cast<i64>(descriptor.weight_after_first_r) * idle;
        best = std::min(best, twc + static_cast<i64>(mu) * std::max(0, exposure - d));
    }
    return best;
}

void bound_experiment(const std::string& output) {
    std::ofstream out(output);
    out << "n,rep,ptotal,resource_jobs,observations,old_ratio,eta_ratio,"
           "old_ratio_large_d,eta_ratio_large_d\n";
    constexpr int mu = 30;
    for (int n : {7, 8, 9}) {
        int repetitions = n == 9 ? 30 : 60;
        for (int rep = 0; rep < repetitions; ++rep) {
            std::vector<Job> jobs = random_instance(n, 1010000 + n * 1000 + rep,
                                                    rep % 2 == 0 ? 10 : 50,
                                                    rep % 3 == 0 ? 0.25 : 0.50);
            std::vector<Descriptor> descriptors = permutation_descriptors(jobs);
            int ptotal = 0;
            int resource_p = 0;
            int resource_jobs = 0;
            for (const Job& job : jobs) {
                ptotal += job.p;
                if (job.resource) {
                    resource_p += job.p;
                    ++resource_jobs;
                }
            }
            i64 wspt = INF;
            for (const Descriptor& descriptor : descriptors) {
                wspt = std::min(wspt, descriptor.twc_without_idle);
            }
            std::vector<double> lambda_eta(2 * mu + 1,
                                           std::numeric_limits<double>::infinity());
            for (int k = 0; k <= 2 * mu; ++k) {
                double eta = 0.5 * k;
                for (const Descriptor& descriptor : descriptors) {
                    int er = descriptor.through_last_r - descriptor.before_first_r;
                    lambda_eta[k] = std::min(
                        lambda_eta[k], descriptor.twc_without_idle + eta * er);
                }
            }

            double old_sum = 0.0;
            double eta_sum = 0.0;
            double old_large = 0.0;
            double eta_large = 0.0;
            i64 observations = 0;
            i64 large_observations = 0;
            for (int s = 0; s <= ptotal; ++s) {
                std::vector<i64> phi = frontier_for_s(descriptors, s, ptotal);
                std::vector<i64> prefix(ptotal + 1, INF);
                std::vector<i64> suffix(ptotal + 2, INF);
                i64 current = INF;
                for (int x = 0; x <= ptotal; ++x) {
                    current = std::min(current, phi[x]);
                    prefix[x] = current;
                }
                current = INF;
                for (int x = ptotal; x >= 0; --x) {
                    if (phi[x] < INF / 2) {
                        current = std::min(current, phi[x] + static_cast<i64>(mu) * x);
                    }
                    suffix[x] = current;
                }
                for (int d = 0; d <= ptotal; ++d) {
                    i64 exact_value = prefix[d];
                    if (d + 1 <= ptotal && suffix[d + 1] < INF / 2) {
                        exact_value = std::min(
                            exact_value, suffix[d + 1] - static_cast<i64>(mu) * d);
                    }
                    double exact = static_cast<double>(exact_value);
                    double span = static_cast<double>(wspt)
                        + mu * std::max(0, resource_p - d);
                    double old_composite = lambda_eta.back() - mu * d;
                    double old_bound = std::max(span, old_composite);
                    double eta_bound = span;
                    for (int k = 0; k <= 2 * mu; ++k) {
                        double eta = 0.5 * k;
                        eta_bound = std::max(eta_bound, lambda_eta[k] - eta * d);
                    }
                    if (old_bound > exact + 1e-8 || eta_bound > exact + 1e-8) {
                        throw std::runtime_error("A lower bound exceeded exact recourse");
                    }
                    old_sum += old_bound / exact;
                    eta_sum += eta_bound / exact;
                    ++observations;
                    if (d >= ptotal / 2) {
                        old_large += old_bound / exact;
                        eta_large += eta_bound / exact;
                        ++large_observations;
                    }
                }
            }
            out << n << ',' << rep << ',' << ptotal << ',' << resource_jobs << ','
                << observations << ',' << std::fixed << std::setprecision(8)
                << old_sum / observations << ',' << eta_sum / observations << ','
                << old_large / large_observations << ','
                << eta_large / large_observations << '\n';
            out.flush();
        }
        std::cout << "bound_experiment_completed n=" << n
                  << " repetitions=" << repetitions << '\n';
    }
}

struct ScenarioResult {
    std::vector<Descriptor> descriptors;
    LabelStats stats;
    int ptotal = 0;
};

std::vector<i64> recourse_table(const std::vector<Descriptor>& descriptors,
                                int ptotal, int pmax, int mu) {
    int width = pmax + 1;
    std::vector<i64> table(width * width, INF);
    for (int s = 0; s <= pmax; ++s) {
        std::vector<i64> phi = frontier_for_s(descriptors, s, ptotal);
        std::vector<i64> prefix(ptotal + 1, INF);
        std::vector<i64> suffix(ptotal + 2, INF);
        i64 current = INF;
        for (int x = 0; x <= ptotal; ++x) {
            current = std::min(current, phi[x]);
            prefix[x] = current;
        }
        current = INF;
        for (int x = ptotal; x >= 0; --x) {
            if (phi[x] < INF / 2) {
                current = std::min(current, phi[x] + static_cast<i64>(mu) * x);
            }
            suffix[x] = current;
        }
        for (int d = 0; d <= pmax; ++d) {
            int capped_d = std::min(d, ptotal);
            i64 value = prefix[capped_d];
            if (d < ptotal && suffix[d + 1] < INF / 2) {
                value = std::min(value, suffix[d + 1] - static_cast<i64>(mu) * d);
            }
            table[s * width + d] = value;
        }
    }
    return table;
}

void stochastic_experiment(const std::string& output,
                           bool canonical_boundaries = false,
                           bool use_cross_middle_dominance = false,
                           bool use_descriptor_dominance = false) {
    std::ofstream out(output);
    out << "rep,scenarios,committed,potential,resource_share,pmax,opt_s,opt_d,objective,"
           "frontier_seconds,table_seconds,total_seconds,peak_states,max_descriptors,"
           "mean_scenario_jobs,max_scenario_jobs\n";
    constexpr int committed = 10;
    constexpr int potential = 10;
    constexpr int lambda = 10;
    constexpr int mu = 30;
    for (double resource_share : {0.20, 0.40, 0.70}) {
        for (int scenarios : {5, 10, 20}) {
            for (int rep = 0; rep < 5; ++rep) {
                auto whole_start = std::chrono::steady_clock::now();
                int seed = 1210000 + static_cast<int>(resource_share * 1000)
                    + scenarios * 100 + rep;
                std::vector<Job> universe = random_instance(
                    committed + potential, seed, 10, resource_share);
                bool committed_resource = false;
                for (int j = 0; j < committed; ++j) {
                    committed_resource = committed_resource || universe[j].resource;
                }
                if (!committed_resource) universe[0].resource = true;

                std::mt19937 rng(seed + 17);
                std::bernoulli_distribution arrival(0.50);
                std::vector<std::vector<Job>> scenario_jobs;
                int pmax = 0;
                int total_n = 0;
                int max_n = 0;
                for (int omega = 0; omega < scenarios; ++omega) {
                    std::vector<Job> jobs;
                    for (int j = 0; j < committed; ++j) jobs.push_back(universe[j]);
                    for (int j = 0; j < potential; ++j) {
                        if (arrival(rng)) jobs.push_back(universe[committed + j]);
                    }
                    int ptotal = 0;
                    for (const Job& job : jobs) ptotal += job.p;
                    pmax = std::max(pmax, ptotal);
                    total_n += static_cast<int>(jobs.size());
                    max_n = std::max(max_n, static_cast<int>(jobs.size()));
                    scenario_jobs.push_back(std::move(jobs));
                }

                auto frontier_start = std::chrono::steady_clock::now();
                std::vector<std::future<ScenarioResult>> futures;
                futures.reserve(scenario_jobs.size());
                for (const std::vector<Job>& jobs : scenario_jobs) {
                    futures.push_back(std::async(std::launch::async,
                                                 [jobs, canonical_boundaries,
                                                  use_cross_middle_dominance,
                                                  use_descriptor_dominance]() {
                        ScenarioResult result;
                        for (const Job& job : jobs) result.ptotal += job.p;
                        result.descriptors = labeling_descriptors(
                            jobs, &result.stats, canonical_boundaries,
                            DiagramMode::Exact, 0, 1, false,
                            use_cross_middle_dominance,
                            use_descriptor_dominance);
                        return result;
                    }));
                }
                std::vector<ScenarioResult> results;
                results.reserve(futures.size());
                std::size_t peak_states = 0;
                std::size_t max_descriptors = 0;
                for (auto& future : futures) {
                    results.push_back(future.get());
                    peak_states = std::max(peak_states, results.back().stats.peak_states);
                    max_descriptors = std::max(max_descriptors,
                                               results.back().descriptors.size());
                }
                auto frontier_stop = std::chrono::steady_clock::now();

                auto table_start = std::chrono::steady_clock::now();
                std::vector<std::vector<i64>> tables;
                tables.reserve(results.size());
                for (const ScenarioResult& result : results) {
                    tables.push_back(recourse_table(result.descriptors, result.ptotal,
                                                    pmax, mu));
                }
                int width = pmax + 1;
                double optimum = std::numeric_limits<double>::infinity();
                int best_s = -1;
                int best_d = -1;
                for (int s = 0; s <= pmax; ++s) {
                    for (int d = 0; d <= pmax; ++d) {
                        double value = static_cast<double>(lambda) * d;
                        for (const auto& table : tables) {
                            value += static_cast<double>(table[s * width + d]) / scenarios;
                        }
                        if (value < optimum) {
                            optimum = value;
                            best_s = s;
                            best_d = d;
                        }
                    }
                }
                auto table_stop = std::chrono::steady_clock::now();
                auto whole_stop = std::chrono::steady_clock::now();
                double frontier_seconds = std::chrono::duration<double>(
                    frontier_stop - frontier_start).count();
                double table_seconds = std::chrono::duration<double>(
                    table_stop - table_start).count();
                double total_seconds = std::chrono::duration<double>(
                    whole_stop - whole_start).count();
                out << rep << ',' << scenarios << ',' << committed << ',' << potential
                    << ',' << resource_share << ',' << pmax << ',' << best_s << ','
                    << best_d << ',' << std::fixed << std::setprecision(6) << optimum
                    << ',' << frontier_seconds << ',' << table_seconds << ','
                    << total_seconds << ',' << peak_states << ',' << max_descriptors
                    << ',' << static_cast<double>(total_n) / scenarios << ',' << max_n
                    << '\n';
                out.flush();
                std::cout << "stochastic share=" << resource_share << " K=" << scenarios
                          << " rep=" << rep << " max_n=" << max_n
                          << " P=" << pmax << " frontier_s=" << frontier_seconds
                          << " table_s=" << table_seconds << '\n';
            }
        }
    }
}

void formulation_comparison_data(const std::string& output) {
    std::ofstream out(output);
    out << "n,rep,objective,seconds,peak_states,descriptors,s,d,ptotal\n";
    constexpr int mu = 30;
    for (int n : {8, 10, 12, 15, 20}) {
        for (int rep = 0; rep < 5; ++rep) {
            std::vector<Job> jobs = random_instance(n, 4701 + 100 * n + rep,
                                                    10, 0.40);
            int ptotal = 0;
            for (const Job& job : jobs) ptotal += job.p;
            int s = ptotal / 4;
            int d = std::max(1, ptotal / 3);
            LabelStats stats;
            std::vector<Descriptor> descriptors = labeling_descriptors(jobs, &stats);
            i64 objective = exact_recourse(descriptors, s, d, mu);
            out << n << ',' << rep << ',' << objective << ',' << std::fixed
                << std::setprecision(8) << stats.seconds << ',' << stats.peak_states
                << ',' << descriptors.size() << ',' << s << ',' << d << ','
                << ptotal << '\n';
            out.flush();
            std::cout << "comparison n=" << n << " rep=" << rep
                      << " sec=" << stats.seconds << " objective=" << objective << '\n';
        }
    }
}

int automatic_boundary_threads(const std::vector<Job>& jobs) {
    int resource_jobs = 0;
    for (const Job& job : jobs) resource_jobs += job.resource;
    if (resource_jobs <= 1) return 1;
    int ordinary_jobs = static_cast<int>(jobs.size()) - resource_jobs;
    int available = static_cast<int>(std::thread::hardware_concurrency());
    if (available <= 0) available = 1;
    int workers = std::min(8, available);
    // Large ordinary-job sets can create several multi-million-label boundary
    // subproblems simultaneously. Four workers preserve a safer memory margin
    // on the 32 GB reference machine.
    if (ordinary_jobs >= 35) workers = std::min(workers, 4);
    return std::min(workers, resource_jobs * (resource_jobs - 1));
}

void single_stress_case(int n, int seed, int scale, double resource_share,
                        int boundary_threads) {
    std::vector<Job> jobs = random_instance(n, seed, scale, resource_share);
    if (boundary_threads <= 0) boundary_threads = automatic_boundary_threads(jobs);
    int ptotal = 0;
    int nr = 0;
    for (const Job& job : jobs) {
        ptotal += job.p;
        nr += job.resource;
    }
    LabelStats stats;
    std::vector<Descriptor> descriptors = labeling_descriptors(
        jobs, &stats, false, DiagramMode::Exact, 0, boundary_threads);
    std::cout << "stress n=" << n << " seed=" << seed << " scale=" << scale
              << " share=" << resource_share << " P=" << ptotal << " nr=" << nr
              << " threads=" << boundary_threads
              << " seconds=" << std::fixed << std::setprecision(8) << stats.seconds
              << " peak_states=" << stats.peak_states
              << " generated_states=" << stats.generated_states
              << " descriptors=" << descriptors.size() << '\n';
}

void shared_last_stress_case(int n, int seed, int scale, double resource_share) {
    std::vector<Job> jobs = random_instance(n, seed, scale, resource_share);
    int ptotal = 0;
    int nr = 0;
    for (const Job& job : jobs) {
        ptotal += job.p;
        nr += job.resource;
    }
    LabelStats pair_stats;
    LabelStats shared_stats;
    std::vector<Descriptor> pair = labeling_descriptors(jobs, &pair_stats);
    std::vector<Descriptor> shared = labeling_descriptors_shared_last(jobs, &shared_stats);
    std::string message;
    if (!compare_frontiers(jobs, pair, shared, &message)) {
        throw std::runtime_error("Shared-last frontier mismatch: " + message);
    }
    std::cout << "shared n=" << n << " seed=" << seed << " scale=" << scale
              << " share=" << resource_share << " P=" << ptotal << " nr=" << nr
              << " pair_seconds=" << std::fixed << std::setprecision(8)
              << pair_stats.seconds << " shared_seconds=" << shared_stats.seconds
              << " pair_generated=" << pair_stats.generated_states
              << " shared_generated=" << shared_stats.generated_states
              << " pair_peak=" << pair_stats.peak_states
              << " shared_peak=" << shared_stats.peak_states
              << " descriptors=" << shared.size() << '\n';
}

void order_stress_case(int n, int seed, int scale, double resource_share) {
    std::vector<Job> jobs = random_instance(n, seed, scale, resource_share);
    int ptotal = 0;
    int nr = 0;
    for (const Job& job : jobs) {
        ptotal += job.p;
        nr += job.resource;
    }
    LabelStats forward_stats;
    LabelStats reverse_stats;
    std::vector<Descriptor> forward = labeling_descriptors(jobs, &forward_stats);
    std::vector<Descriptor> reverse = labeling_descriptors(
        jobs, &reverse_stats, false, DiagramMode::Exact, 0, 1, true);
    std::string message;
    if (!compare_frontiers(jobs, forward, reverse, &message)) {
        throw std::runtime_error("Reverse-order frontier mismatch: " + message);
    }
    std::cout << "order n=" << n << " seed=" << seed << " scale=" << scale
              << " share=" << resource_share << " P=" << ptotal << " nr=" << nr
              << " forward_seconds=" << std::fixed << std::setprecision(8)
              << forward_stats.seconds << " reverse_seconds=" << reverse_stats.seconds
              << " forward_generated=" << forward_stats.generated_states
              << " reverse_generated=" << reverse_stats.generated_states
              << " forward_peak=" << forward_stats.peak_states
              << " reverse_peak=" << reverse_stats.peak_states
              << " descriptors=" << reverse.size() << '\n';
}

void canonical_stress_case(int n, int seed, int scale, double resource_share,
                           bool compare_full, bool use_cross_middle_dominance,
                           bool use_descriptor_dominance) {
    std::vector<Job> jobs = random_instance(n, seed, scale, resource_share);
    int ptotal = 0;
    int nr = 0;
    for (const Job& job : jobs) {
        ptotal += job.p;
        nr += job.resource;
    }
    LabelStats canonical_stats;
    std::vector<Descriptor> canonical = labeling_descriptors(
        jobs, &canonical_stats, true, DiagramMode::Exact, 0, 1, false,
        use_cross_middle_dominance, use_descriptor_dominance);
    double full_seconds = -1.0;
    std::size_t full_generated = 0;
    if (compare_full) {
        LabelStats full_stats;
        int threads = automatic_boundary_threads(jobs);
        std::vector<Descriptor> full = labeling_descriptors(
            jobs, &full_stats, false, DiagramMode::Exact, 0, threads);
        std::string message;
        if (!compare_cumulative_frontiers(jobs, full, canonical, &message)) {
            throw std::runtime_error("Canonical-boundary mismatch: " + message);
        }
        full_seconds = full_stats.seconds;
        full_generated = full_stats.generated_states;
    }
    std::cout << "canonical n=" << n << " seed=" << seed << " scale=" << scale
              << " share=" << resource_share << " P=" << ptotal << " nr=" << nr
              << " seconds=" << std::fixed << std::setprecision(8)
              << canonical_stats.seconds
              << " generated=" << canonical_stats.generated_states
              << " peak=" << canonical_stats.peak_states
              << " descriptors=" << canonical.size()
              << " compared=" << compare_full
              << " cross=" << use_cross_middle_dominance
              << " envelope=" << use_descriptor_dominance
              << " full_seconds=" << full_seconds
              << " full_generated=" << full_generated << '\n';
}

void colleague_refinement_case(int n, int seed, int scale,
                               double resource_share) {
    std::vector<Job> jobs = random_instance(n, seed, scale, resource_share);
    int ptotal = 0;
    int nr = 0;
    for (const Job& job : jobs) {
        ptotal += job.p;
        nr += job.resource;
    }

    auto solve = [&](bool restrict_middle, bool cross_before, LabelStats& stats) {
        return labeling_descriptors(jobs, &stats, true, DiagramMode::Exact, 0, 1,
                                    false, false, false, restrict_middle,
                                    cross_before);
    };
    LabelStats base_stats;
    LabelStats span_stats;
    LabelStats before_stats;
    LabelStats combined_stats;
    std::vector<Descriptor> base = solve(false, false, base_stats);
    std::vector<Descriptor> span = solve(true, false, span_stats);
    std::vector<Descriptor> before = solve(false, true, before_stats);
    std::vector<Descriptor> combined = solve(true, true, combined_stats);

    std::string message;
    for (const auto& candidate : {std::cref(span), std::cref(before),
                                  std::cref(combined)}) {
        if (!compare_cumulative_frontiers(jobs, base, candidate.get(), &message)) {
            throw std::runtime_error("Colleague-refinement mismatch: " + message);
        }
    }

    auto print_stats = [](const char* name, const LabelStats& stats) {
        std::cout << ' ' << name << "_seconds=" << std::fixed
                  << std::setprecision(8) << stats.seconds
                  << ' ' << name << "_generated=" << stats.generated_states
                  << ' ' << name << "_peak=" << stats.peak_states;
    };
    std::cout << "colleague n=" << n << " seed=" << seed << " scale=" << scale
              << " share=" << resource_share << " P=" << ptotal << " nr=" << nr;
    print_stats("base", base_stats);
    print_stats("span", span_stats);
    print_stats("before", before_stats);
    print_stats("combined", combined_stats);
    std::cout << '\n';
}

void cross_before_followup_case(int n, int seed, int scale,
                                double resource_share, int variant) {
    std::vector<Job> jobs = random_instance(n, seed, scale, resource_share);
    int ptotal = 0;
    int wtotal = 0;
    int nr = 0;
    for (const Job& job : jobs) {
        ptotal += job.p;
        wtotal += job.w;
        nr += job.resource;
    }
    const bool use_cross_before = variant != 0;
    const bool use_ordinary_weight = variant >= 2;
    const bool use_counting_sort = variant == 3;
    const bool use_indexed_sort = variant >= 4;
    const bool use_middle_rule = variant >= 5;
    const bool use_objective_aligned = variant >= 6;
    LabelStats stats;
    std::vector<Descriptor> result = labeling_descriptors(
        jobs, &stats, true, DiagramMode::Exact, 0, 1, false, use_middle_rule, false,
        true, use_cross_before, false, use_ordinary_weight, use_counting_sort,
        use_indexed_sort, use_objective_aligned);
    std::cout << "crosspb_followup n=" << n << " seed=" << seed
              << " scale=" << scale << " share=" << resource_share
              << " variant=" << variant << " P=" << ptotal << " W=" << wtotal
              << " nr=" << nr << " seconds=" << std::fixed
              << std::setprecision(8) << stats.seconds
              << " generated=" << stats.generated_states
              << " peak=" << stats.peak_states
              << " descriptors=" << result.size() << '\n';
}

void validate_colleague_refinements() {
    constexpr int n = 5;
    int data_patterns = 1;
    for (int j = 0; j < n; ++j) data_patterns *= 4;
    int tested = 0;
    for (int pattern = 0; pattern < data_patterns; ++pattern) {
        std::vector<Job> jobs;
        int value = pattern;
        for (int j = 0; j < n; ++j) {
            int type = value % 4;
            value /= 4;
            jobs.push_back({j, (type & 1) ? 4 : 1,
                            (type & 2) ? 4 : 1, false});
        }
        for (int mask = 1; mask < (1 << n); ++mask) {
            for (int j = 0; j < n; ++j) jobs[j].resource = mask & (1 << j);
            std::vector<Descriptor> exact = permutation_descriptors(jobs);
            std::vector<Descriptor> span = labeling_descriptors(
                jobs, nullptr, true, DiagramMode::Exact, 0, 1, false, false,
                false, true, false);
            std::vector<Descriptor> before = labeling_descriptors(
                jobs, nullptr, true, DiagramMode::Exact, 0, 1, false, false,
                false, false, true);
            std::vector<Descriptor> combined = labeling_descriptors(
                jobs, nullptr, true, DiagramMode::Exact, 0, 1, false, false,
                false, true, true);
            std::vector<Descriptor> full_rule = labeling_descriptors(
                jobs, nullptr, true, DiagramMode::Exact, 0, 1, false, false,
                false, true, false, true);
            std::vector<Descriptor> tight_before = labeling_descriptors(
                jobs, nullptr, true, DiagramMode::Exact, 0, 1, false, false,
                false, true, true, false, true, false);
            std::vector<Descriptor> counting_before = labeling_descriptors(
                jobs, nullptr, true, DiagramMode::Exact, 0, 1, false, false,
                false, true, true, false, true, true);
            std::vector<Descriptor> indexed_before = labeling_descriptors(
                jobs, nullptr, true, DiagramMode::Exact, 0, 1, false, false,
                false, true, true, false, true, false, true);
            std::vector<Descriptor> strongest = labeling_descriptors(
                jobs, nullptr, true, DiagramMode::Exact, 0, 1, false, true,
                false, true, true, false, true, false, true);
            std::vector<Descriptor> objective_aligned = labeling_descriptors(
                jobs, nullptr, true, DiagramMode::Exact, 0, 1, false, true,
                false, true, true, false, true, false, true, true);
            std::vector<Descriptor> objective_only = labeling_descriptors(
                jobs, nullptr, true, DiagramMode::Exact, 0, 1, false, false,
                false, true, false, false, true, false, false, true, nullptr,
                false);
            std::vector<Descriptor> direct_objective = labeling_descriptors(
                jobs, nullptr, true, DiagramMode::Exact, 0, 1, false, false,
                false, true, false, false, true, false, false, true, nullptr,
                false, false);
            std::vector<Descriptor> production = strongest_exact_descriptors(jobs);
            std::string message;
            for (const auto& candidate : {std::cref(span), std::cref(before),
                                          std::cref(combined),
                                          std::cref(full_rule),
                                          std::cref(tight_before),
                                          std::cref(counting_before),
                                          std::cref(indexed_before),
                                          std::cref(strongest),
                                          std::cref(objective_aligned),
                                          std::cref(objective_only),
                                          std::cref(direct_objective),
                                          std::cref(production)}) {
                if (!compare_cumulative_frontiers(jobs, exact, candidate.get(),
                                                  &message)) {
                    std::cerr << "Colleague-refinement validation failed: "
                              << message << '\n';
                    print_instance(jobs);
                    std::exit(2);
                }
            }
            ++tested;
        }
    }
    std::cout << "colleague_refinement_validation_passed instances=" << tested
              << '\n';
}

std::vector<i64> complete_recourse_table(const std::vector<Descriptor>& descriptors,
                                         int ptotal, int mu) {
    return recourse_table(descriptors, ptotal, ptotal, mu);
}

void diagram_refinement_experiment(const std::string& output) {
    std::ofstream out(output);
    out << "n,rep,width,ptotal,resource_jobs,restricted_seconds,relaxed_seconds,"
           "restricted_peak,relaxed_peak,mean_gap,max_gap,certified,valid\n";
    constexpr int mu = 30;
    for (int n : {18, 20}) {
        for (int rep = 0; rep < 5; ++rep) {
            std::vector<Job> jobs = random_instance(n, 1310000 + n * 100 + rep,
                                                    rep % 2 == 0 ? 10 : 50,
                                                    rep < 3 ? 0.20 : 0.40);
            int ptotal = 0;
            int nr = 0;
            for (const Job& job : jobs) {
                ptotal += job.p;
                nr += job.resource;
            }
            LabelStats exact_stats;
            std::vector<Descriptor> exact = labeling_descriptors(
                jobs, &exact_stats, true, DiagramMode::Exact, 0, 1, false,
                true, true, true, true, false, true, false, true, true);
            std::vector<i64> exact_table = complete_recourse_table(exact, ptotal, mu);
            for (std::size_t width : {1u, 2u, 4u, 8u, 16u, 32u, 64u}) {
                LabelStats restricted_stats;
                LabelStats relaxed_stats;
                std::vector<Descriptor> restricted = labeling_descriptors(
                    jobs, &restricted_stats, true, DiagramMode::Restricted, width,
                    1, false, false, true, true, false, false, false, false,
                    false, true);
                std::vector<Descriptor> relaxed = labeling_descriptors(
                    jobs, &relaxed_stats, true, DiagramMode::Relaxed, width,
                    1, false, false, true, true, false, false, false, false,
                    false, true);
                std::vector<i64> upper = complete_recourse_table(restricted, ptotal, mu);
                std::vector<i64> lower = complete_recourse_table(relaxed, ptotal, mu);
                double gap_sum = 0.0;
                double max_gap = 0.0;
                bool certified = true;
                bool valid = true;
                for (std::size_t i = 0; i < exact_table.size(); ++i) {
                    if (lower[i] > exact_table[i] || upper[i] < exact_table[i]) valid = false;
                    if (lower[i] != upper[i]) certified = false;
                    double gap = static_cast<double>(upper[i] - lower[i]) / upper[i];
                    gap_sum += gap;
                    max_gap = std::max(max_gap, gap);
                }
                if (!valid) throw std::runtime_error("Invalid relaxed or restricted diagram");
                out << n << ',' << rep << ',' << width << ',' << ptotal << ',' << nr
                    << ',' << std::fixed << std::setprecision(8)
                    << restricted_stats.seconds << ',' << relaxed_stats.seconds << ','
                    << restricted_stats.peak_states << ',' << relaxed_stats.peak_states
                    << ',' << gap_sum / exact_table.size() << ',' << max_gap << ','
                    << certified << ',' << valid << '\n';
                out.flush();
            }
            std::cout << "diagram_refinement n=" << n << " rep=" << rep
                      << " exact_s=" << exact_stats.seconds << " P=" << ptotal
                      << " nr=" << nr << '\n';
        }
    }
}

void validate_single_resource_recursion() {
    constexpr int n = 6;
    int data_patterns = 1;
    for (int j = 0; j < n; ++j) data_patterns *= 4;
    int tested = 0;
    for (int pattern = 0; pattern < data_patterns; ++pattern) {
        std::vector<Job> jobs;
        int value = pattern;
        for (int j = 0; j < n; ++j) {
            const int type = value % 4;
            value /= 4;
            jobs.push_back({j, (type & 1) ? 4 : 1,
                            (type & 2) ? 4 : 1, j == pattern % n});
        }
        const std::vector<Descriptor> exact = permutation_descriptors(jobs);
        const std::vector<Descriptor> specialized =
            single_resource_descriptors(jobs);
        std::string message;
        if (!compare_cumulative_frontiers(jobs, exact, specialized, &message)) {
            std::cerr << "Single-resource validation failed: " << message << '\n';
            print_instance(jobs);
            std::exit(2);
        }
        ++tested;
    }
    std::cout << "single_resource_validation_passed instances=" << tested << '\n';
}

void single_resource_comparison(int n, int seed, int scale) {
    std::vector<Job> jobs = random_instance(n, seed, scale, 0.0);
    for (Job& job : jobs) job.resource = false;
    jobs[static_cast<std::size_t>(seed) % jobs.size()].resource = true;
    LabelStats generic_stats;
    LabelStats specialized_stats;
    const std::vector<Descriptor> generic = labeling_descriptors(
        jobs, &generic_stats, true, DiagramMode::Exact, 0, 1, false,
        true, true, true, true, false, true, false, true, true);
    const std::vector<Descriptor> specialized =
        single_resource_descriptors(jobs, &specialized_stats);
    std::string message;
    if (!compare_cumulative_frontiers(jobs, generic, specialized, &message)) {
        throw std::runtime_error("Single-resource comparison mismatch: " + message);
    }
    std::cout << "single_resource_compare n=" << n << " seed=" << seed
              << " scale=" << scale
              << " generic_s=" << generic_stats.seconds
              << " specialized_s=" << specialized_stats.seconds
              << " generic_generated=" << generic_stats.generated_states
              << " specialized_generated=" << specialized_stats.generated_states
              << " generic_peak=" << generic_stats.peak_states
              << " specialized_peak=" << specialized_stats.peak_states
              << " generic_descriptors=" << generic.size()
              << " specialized_descriptors=" << specialized.size() << '\n';
}

void correlation_stress_case(int n, int seed, int scale, double resource_share,
                             int correlation) {
    std::mt19937 processing_rng(seed);
    std::mt19937 weight_rng(seed + 100003);
    std::mt19937 resource_rng(seed + 200003);
    std::uniform_int_distribution<int> processing(1, scale);
    std::uniform_int_distribution<int> independent_weight(1, scale);
    const int radius = std::max(1, scale / 5);
    std::uniform_int_distribution<int> noise(-radius, radius);
    std::bernoulli_distribution resource(resource_share);
    std::vector<Job> jobs;
    jobs.reserve(n);
    int resource_count = 0;
    for (int j = 0; j < n; ++j) {
        const int p = processing(processing_rng);
        int w = independent_weight(weight_rng);
        if (correlation == 1) {
            w = std::clamp(p + noise(weight_rng), 1, scale);
        } else if (correlation == 2) {
            w = p;
        }
        const bool requires_resource = resource(resource_rng);
        resource_count += requires_resource;
        jobs.push_back({j, p, w, requires_resource});
    }
    if (resource_count == 0) {
        jobs[0].resource = true;
        resource_count = 1;
    }
    LabelStats stats;
    const std::vector<Descriptor> descriptors = labeling_descriptors(
        jobs, &stats, true, DiagramMode::Exact, 0, 1, false,
        true, true, true, true, false, true, false, true, true);
    std::cout << "correlation_stress n=" << n << " seed=" << seed
              << " scale=" << scale << " share=" << resource_share
              << " correlation=" << correlation
              << " resource_jobs=" << resource_count
              << " seconds=" << stats.seconds
              << " generated=" << stats.generated_states
              << " peak=" << stats.peak_states
              << " descriptors=" << descriptors.size() << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string mode = argc >= 2 ? argv[1] : "validate";
        if (mode == "validate") {
            validate();
        } else if (mode == "hull_counterexample") {
            find_naive_hull_counterexample();
        } else if (mode == "benchmark") {
            std::string output = argc >= 3 ? argv[2] : "frontier_results.csv";
            benchmark(output);
        } else if (mode == "bounds") {
            std::string output = argc >= 3 ? argv[2] : "frontier_bound_results.csv";
            bound_experiment(output);
        } else if (mode == "stochastic") {
            std::string output = argc >= 3 ? argv[2] : "frontier_stochastic_results.csv";
            stochastic_experiment(output);
        } else if (mode == "stochastic_canonical") {
            std::string output = argc >= 3
                ? argv[2] : "frontier_stochastic_canonical_results.csv";
            stochastic_experiment(output, true);
        } else if (mode == "stochastic_best") {
            std::string output = argc >= 3
                ? argv[2] : "frontier_stochastic_best_results.csv";
            stochastic_experiment(output, true, true, true);
        } else if (mode == "compare") {
            std::string output = argc >= 3 ? argv[2] : "frontier_comparison_results.csv";
            formulation_comparison_data(output);
        } else if (mode == "diagrams") {
            std::string output = argc >= 3 ? argv[2] : "frontier_diagram_results.csv";
            diagram_refinement_experiment(output);
        } else if (mode == "validate_single_resource") {
            validate_single_resource_recursion();
        } else if (mode == "single_resource_compare") {
            if (argc != 5) {
                throw std::runtime_error(
                    "single_resource_compare requires: n seed processing_scale");
            }
            single_resource_comparison(std::stoi(argv[2]), std::stoi(argv[3]),
                                       std::stoi(argv[4]));
        } else if (mode == "correlation_stress") {
            if (argc != 7) {
                throw std::runtime_error(
                    "correlation_stress requires: n seed processing_scale "
                    "resource_share correlation");
            }
            correlation_stress_case(
                std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]),
                std::stod(argv[5]), std::stoi(argv[6]));
        } else if (mode == "single") {
            if (argc != 6 && argc != 7) {
                throw std::runtime_error(
                    "single mode requires: n seed processing_scale resource_share "
                    "[boundary_threads]");
            }
            single_stress_case(std::stoi(argv[2]), std::stoi(argv[3]),
                               std::stoi(argv[4]), std::stod(argv[5]),
                               argc == 7 ? std::stoi(argv[6]) : 0);
        } else if (mode == "shared") {
            if (argc != 6) {
                throw std::runtime_error(
                    "shared mode requires: n seed processing_scale resource_share");
            }
            shared_last_stress_case(std::stoi(argv[2]), std::stoi(argv[3]),
                                    std::stoi(argv[4]), std::stod(argv[5]));
        } else if (mode == "order") {
            if (argc != 6) {
                throw std::runtime_error(
                    "order mode requires: n seed processing_scale resource_share");
            }
            order_stress_case(std::stoi(argv[2]), std::stoi(argv[3]),
                              std::stoi(argv[4]), std::stod(argv[5]));
        } else if (mode == "canonical") {
            if (argc < 6 || argc > 9) {
                throw std::runtime_error(
                    "canonical mode requires: n seed processing_scale resource_share "
                    "[compare_full] [cross_middle] [descriptor_dominance]");
            }
            canonical_stress_case(std::stoi(argv[2]), std::stoi(argv[3]),
                                  std::stoi(argv[4]), std::stod(argv[5]),
                                  argc >= 7 && std::stoi(argv[6]) != 0,
                                  argc >= 8 && std::stoi(argv[7]) != 0,
                                  argc >= 9 && std::stoi(argv[8]) != 0);
        } else if (mode == "validate_colleague") {
            validate_colleague_refinements();
        } else if (mode == "colleague") {
            if (argc != 6) {
                throw std::runtime_error(
                    "colleague mode requires: n seed processing_scale resource_share");
            }
            colleague_refinement_case(std::stoi(argv[2]), std::stoi(argv[3]),
                                      std::stoi(argv[4]), std::stod(argv[5]));
        } else if (mode == "crosspb_followup") {
            if (argc != 7) {
                throw std::runtime_error(
                    "crosspb_followup mode requires: n seed processing_scale "
                    "resource_share variant");
            }
            cross_before_followup_case(
                std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]),
                std::stod(argv[5]), std::stoi(argv[6]));
        } else {
            std::cerr << "Usage: frontier_labeling "
                         "[validate|validate_colleague|validate_single_resource|single_resource_compare|correlation_stress|colleague|crosspb_followup|hull_counterexample|benchmark|bounds|stochastic|stochastic_canonical|stochastic_best|compare|diagrams|single|shared|order|canonical] "
                         "[output.csv]\n";
            return 1;
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
