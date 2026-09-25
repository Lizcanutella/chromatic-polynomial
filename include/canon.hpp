// canon.hpp - Phase 0.4 (part 1): exact canonical form via nauty.
//
// *** LOCALLY UNVERIFIED ***  This file uses the nauty C API but was not compiled or run
// in the environment where it was written (nauty was unavailable). It is written to match
// nauty's documented dense interface and reviewed carefully, but TREAT THE FIRST COMPILE
// AS A TEST. Specific things to check on your machine (see README "Verifying the cache"):
//   1. Header path: <nauty/nauty.h> (Debian/Ubuntu) vs "nauty.h" (source build).
//   2. Link library name: -lnauty.
//   3. That ADDELEMENT / GRAPHROW / EMPTYGRAPH / SETWORDSNEEDED / densenauty resolve.
//   4. WORDSIZE consistency (nauty_check below will abort loudly on mismatch).
//
// Correctness rationale: nauty's canonical labelling makes two graphs isomorphic iff their
// canonical forms are bit-identical. So the serialized canonical adjacency IS an exact key.
//  No separate isomorphism re-check is needed; equal keys imply isomorphic graphs and the
// chromatic polynomial is an isomorphism invariant. (We prefix the vertex count to keep
// keys of different-order graphs trivially distinct.)
//
// Single-threaded only: nauty uses internal global state (and this function keeps a static
// options block), so canonicalKey() must never run concurrently. Calling it from two threads
// at once silently corrupts nauty's canonical labelling and yields wrong cache keys.
// The NautyReentryGuard below converts that silent corruption into a LOUD std::runtime_error. 
// For parallelism, run independent processes (see chromatic_rl.parallel) or use_cache=False.

#pragma once
#include "graph.hpp"
#include <string>
#include <vector>
#include <cstring>
#include <atomic>
#include <stdexcept>

#include <nauty/nauty.h>   // if your install lacks the nauty/ prefix, change to "nauty.h"
// #include "nauty.h"


namespace crl {

namespace detail {
// Process-wide flag marking that a canonicalKey() call is in flight. nauty's global state
// admits no nesting and no concurrency, so exactly one call may be active at a time.
inline std::atomic<bool>& nautyInUse() {
    static std::atomic<bool> in_use{false};
    return in_use;
}
// RAII: claims the flag on construction, releases on destruction. If the flag is already
// held (another thread is inside canonicalKey), the constructor throws WITHOUT claiming it,
// so the legitimate holder's destructor still releases correctly.
struct NautyReentryGuard {
    NautyReentryGuard() {
        bool expected = false;
        if (!nautyInUse().compare_exchange_strong(expected, true,
                                                  std::memory_order_acq_rel)) {
            throw std::runtime_error(
                "crl::canonicalKey: nauty's global state is not concurrency-safe and was "
                "entered concurrently. Run independent processes (see chromatic_rl.parallel) "
                "or disable the cache (use_cache=False).");
        }
    }
    ~NautyReentryGuard() { nautyInUse().store(false, std::memory_order_release); }
    NautyReentryGuard(const NautyReentryGuard&) = delete;
    NautyReentryGuard& operator=(const NautyReentryGuard&) = delete;
};
} // namespace detail

inline std::string canonicalKey(const Graph& G) {
    const int n = G.numVertices();
    if (n == 0) return std::string(sizeof(int), '\0');  // degenerate; never reached in engine

    detail::NautyReentryGuard nauty_guard;  // loud failure on concurrent/nested entry

    const int m = SETWORDSNEEDED(n);
    nauty_check(WORDSIZE, m, n, NAUTYVERSIONID);   // aborts if ABI/WORDSIZE mismatch

    std::vector<graph> g (static_cast<size_t>(m) * n, 0);
    std::vector<graph> cg(static_cast<size_t>(m) * n, 0);
    std::vector<int> lab(n), ptn(n), orbits(n);

    EMPTYGRAPH(g.data(), m, n);
    for (const auto& e : G.edges()) {              // undirected: set both directions
        ADDELEMENT(GRAPHROW(g.data(), e.first,  m), e.second);
        ADDELEMENT(GRAPHROW(g.data(), e.second, m), e.first);
    }

    static DEFAULTOPTIONS_GRAPH(options);
    options.getcanon = TRUE;
    statsblk stats;

    densenauty(g.data(), lab.data(), ptn.data(), orbits.data(),
               &options, &stats, m, n, cg.data());

    // Serialize: [n][canonical adjacency words]. Width-agnostic via raw bytes.
    std::string key;
    key.resize(sizeof(int) + cg.size() * sizeof(graph));
    std::memcpy(&key[0], &n, sizeof(int));
    if (!cg.empty())
        std::memcpy(&key[sizeof(int)], cg.data(), cg.size() * sizeof(graph));
    return key;
}

} // namespace crl
