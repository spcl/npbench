#include <algorithm>
#include <cassert>
#include <chrono>
#include <functional>
#include <limits>
#include <vector>

template<bool HighResIsSteady = std::chrono::high_resolution_clock::is_steady>
struct SteadyClock {
    using type = std::chrono::high_resolution_clock;
};

template<>
struct SteadyClock<false> {
    using type = std::chrono::steady_clock;
};

inline SteadyClock<>::type::time_point benchmark_now() {
    return SteadyClock<>::type::now();
}

inline double benchmark_duration_seconds(
    SteadyClock<>::type::time_point start,
    SteadyClock<>::type::time_point end) {
    return std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
}

inline double benchmark(uint64_t iterations, const std::function<void()> &op) {
    op();

    std::vector<double> runtimes;
    for (uint64_t i = 0; i < iterations; i++) {
        auto start = benchmark_now();
        op();
        auto end = benchmark_now();
        double elapsed_seconds = benchmark_duration_seconds(start, end);
        runtimes.push_back(elapsed_seconds);
    }

    std::sort(runtimes.begin(), runtimes.end());

    double median;
    if (runtimes.size() % 2 == 1) {
        size_t med = runtimes.size() / 2;
        median = runtimes.at(med);
    } else {
        size_t med = runtimes.size() / 2;
        median = (runtimes.at(med) + runtimes.at(med + 1)) / 2.0;
    }

    return median;
}