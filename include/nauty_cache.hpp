// nauty_cache.hpp — Phase 0.4 (part 2): isomorphism cache for the engine.
//
// *** LOCALLY UNVERIFIED *** (depends on canon.hpp / nauty). See canon.hpp header.
//
// Implements the cache interface the templated engine expects:
//     const Poly* lookup(const Graph&);   // nullptr on miss
//     void        store (const Graph&, const Poly&);
//
// Keyed on the exact nauty canonical form, so a key match is a true isomorphism match.
// Pointer returned by lookup() points into the map; std::unordered_map guarantees that
// references/pointers to elements stay valid across later insertions (only iterators may
// be invalidated by rehashing), so it is safe to hold across the recursive store() calls.
//
// NOTE (perf, intentionally deferred): a miss canonicalizes once in lookup() and again in
// store(), i.e. two nauty calls per unique graph. The clean fix is to thread the key from
// lookup() to store() through the engine; left for a later increment to keep the engine's
// cache interface trivial. Canonicalization still dominates only when subtrees are small.
#pragma once
#include "graph.hpp"
#include "polynomial.hpp"
#include "canon.hpp"
#include <unordered_map>
#include <string>

namespace crl {

class NautyCache {
public:
    const Poly* lookup(const Graph& g) {
        auto it = table_.find(canonicalKey(g));
        if (it != table_.end()) { ++hits; return &it->second; }
        ++misses;
        return nullptr;
    }

    void store(const Graph& g, const Poly& p) {
        table_.emplace(canonicalKey(g), p);
        words += polyWords(p);          // monotone: peak == final
    }

    long long hits = 0;
    long long misses = 0;
    long long words = 0;                // total coefficient-limbs held in the cache
    size_t size() const { return table_.size(); }
    double hitRate() const {
        long long t = hits + misses;
        return t ? static_cast<double>(hits) / static_cast<double>(t) : 0.0;
    }

private:
    std::unordered_map<std::string, Poly> table_;
};

} // namespace crl
