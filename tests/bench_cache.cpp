// bench_cache.cpp — Phase 0.4 driver: correctness + a first look at the cache payoff.
//
// *** LOCALLY UNVERIFIED *** (links nauty). Run this first to validate the cache layer.
//
// (1) Correctness: on small graphs the cached result MUST equal the cache-free result.
//     (The cache-free engine is independently validated in tests/validate.cpp.)
// (2) Illustration: on a larger graph, compare branch count and wall-clock with vs
//     without the cache, and report the cache hit rate.
#include "chromatic.hpp"
#include "nauty_cache.hpp"
#include <iostream>
#include <random>
#include <chrono>

using namespace crl;

static Graph erdosRenyi(int n, double p, std::mt19937& rng) {
    Graph g(n);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (u(rng) < p) g.addEdge(i, j);
    return g;
}

int main() {
    // (1) cached == cache-free on small graphs
    std::mt19937 rng(2024);
    int mismatches = 0;
    for (int t = 0; t < 30; ++t) {
        int n = 5 + static_cast<int>(rng() % 4);   // 5..8
        Graph g = erdosRenyi(n, 0.5, rng);
        Poly noCache = computeChromatic(g, selectMinDegreeSum);
        NautyCache cache; Stats st;
        Poly cached = computeChromatic(g, selectMinDegreeSum, cache, st);
        if (!(noCache == cached)) ++mismatches;
    }
    std::cout << "(1) cache vs cache-free agreement on 30 small graphs: "
              << (mismatches == 0 ? "OK" : "MISMATCH x" + std::to_string(mismatches)) << "\n";

    // (2) payoff on a moderate graph. Keep n modest: the cache-free leg blows up fast.
    std::mt19937 r2(7);
    const int n = 13;
    const double p = 0.5;
    Graph g = erdosRenyi(n, p, r2);

    {
        NullCache c; Stats st;
        auto t0 = std::chrono::steady_clock::now();
        Poly poly = computeChromaticT(g, selectMinDegreeSum, c, st);
        auto t1 = std::chrono::steady_clock::now();
        std::cout << "(2) no cache   : branches=" << st.branches
                  << "  time=" << std::chrono::duration<double>(t1 - t0).count() << "s\n";
    }
    {
        NautyCache cache; Stats st;
        auto t0 = std::chrono::steady_clock::now();
        Poly poly = computeChromatic(g, selectMinDegreeSum, cache, st);
        auto t1 = std::chrono::steady_clock::now();
        std::cout << "    nauty cache: branches=" << st.branches
                  << "  unique=" << cache.size()
                  << "  hits=" << cache.hits << "  misses=" << cache.misses
                  << "  hit_rate=" << cache.hitRate()
                  << "  time=" << std::chrono::duration<double>(t1 - t0).count() << "s\n";
        std::cout << "    P(G,k) = " << poly.str() << "\n";
    }
    return 0;
}
