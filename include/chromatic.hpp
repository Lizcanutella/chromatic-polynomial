// chromatic.hpp - Phase 0.2: deletion-contraction engine + 0.6 selector interface.
//
// computeChromatic(G, select) returns the chromatic polynomial P(G, k).
//
// The edge-selection POLICY is a swappable std::function - this is the architectural
// keystone of the whole project. Hand heuristics implement it now; the learned policy
// implements the same signature later, with no change to the engine.

#pragma once
#include "graph.hpp"
#include "polynomial.hpp"
#include "trace.hpp"
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <vector>
#include <queue>
#include <cstdint>
#include <cmath>
#include <random>
#include <cassert>
#include <stdexcept>

namespace crl {

// Thrown by computeChromaticT when Stats.budget >= 0 and the live branch count
// exceeds it. Lets a caller cap an exponential cache-OFF rollout; unwinds the
// recursion cleanly. A truncated computation yields NO polynomial by design.
struct BudgetExceeded : std::runtime_error {
    BudgetExceeded() : std::runtime_error("branch budget exceeded") {}
};

// A selector receives the current graph and returns one edge (u, v) to branch on.
// It is only called when G is connected, has >= 1 edge, and is not a recognised
// base case (tree / cycle / complete), so a valid non-bridge-agnostic edge exists.
using EdgeSelector = std::function<std::pair<int,int>(const Graph&)>;

// ---- closed-form base cases ---------------------------------------------------------

// Empty graph on n vertices (no edges): k^n.
inline Poly chromaticEmpty(int n) { return Poly::monomial(n); }

// Tree on n vertices: k (k-1)^{n-1}.
inline Poly chromaticTree(int n) {
    Poly p = Poly::monomial(1);                 // k
    for (int i = 0; i < n - 1; ++i) p = p * Poly::linear(1);  // *(k-1)
    return p;
}

// Complete graph K_n: falling factorial k(k-1)...(k-n+1).
inline Poly chromaticComplete(int n) {
    Poly p = Poly::one();
    for (int i = 0; i < n; ++i) p = p * Poly::linear(i);
    return p;
}

// Cycle C_n (n>=3): (k-1)^n + (-1)^n (k-1).
inline Poly chromaticCycle(int n) {
    Poly km1 = Poly::linear(1);                 // (k-1)
    Poly p = Poly::one();
    for (int i = 0; i < n; ++i) p = p * km1;            // (k-1)^n
    // term = (-1)^n (k-1):  + (k-1) when n even, - (k-1) when n odd.
    return (n % 2 == 0) ? (p + km1) : (p - km1);
}

// Wheel W_n (n>=4): hub joined to every vertex of a rim cycle C_{n-1}.
// chi(W_n, t) = t[(t-2)^{n-1} + (-1)^{n-1}(t-2)]   (paper Theorem 4)
inline Poly chromaticWheel(int n) {
    assert(n >= 4);
    Poly km2 = Poly::linear(2);                 // (k-2)
    Poly p = Poly::one();
    for (int i = 0; i < n - 1; ++i) p = p * km2;     // (k-2)^{n-1}
    // bracket = (k-2)^{n-1} + (-1)^{n-1}(k-2)
    Poly bracket = ((n - 1) % 2 == 0) ? (p + km2) : (p - km2);
    return Poly::monomial(1) * bracket;         // multiply by k
}

// ---- structural helpers -------------------------------------------------------------

// Connected components as vertex-index lists (BFS over the current labelling).
inline std::vector<std::vector<int>> components(const Graph& g) {
    int n = g.numVertices();
    std::vector<int> comp(n, -1);
    std::vector<std::vector<int>> out;
    for (int s = 0; s < n; ++s) {
        if (comp[s] != -1) continue;
        std::vector<int> cur;
        std::queue<int> q; q.push(s); comp[s] = static_cast<int>(out.size());
        while (!q.empty()) {
            int x = q.front(); q.pop(); cur.push_back(x);
            for (int y : g.neighbours(x))
                if (comp[y] == -1) { comp[y] = comp[s]; q.push(y); }
        }
        out.push_back(std::move(cur));
    }
    return out;
}

// Connected components of the COMPLEMENT graph Gbar, in g's ORIGINAL labels. A vertex's
// Gbar-neighbours are its G-non-neighbours. size() == 1 means Gbar is connected (G is not a
// join); size() > 1 means G is the join of the induced subgraphs on these parts. O(n^2).
inline std::vector<std::vector<int>> complementComponents(const Graph& g) {
    const int n = g.numVertices();
    std::vector<int> comp(n, -1);
    std::vector<std::vector<int>> out;
    for (int s = 0; s < n; ++s) {
        if (comp[s] != -1) continue;
        std::vector<int> cur;
        std::queue<int> q; q.push(s); comp[s] = static_cast<int>(out.size());
        while (!q.empty()) {
            int x = q.front(); q.pop(); cur.push_back(x);
            for (int y = 0; y < n; ++y)
                if (comp[y] == -1 && y != x && !g.hasEdge(x, y)) {
                    comp[y] = comp[s]; q.push(y);
                }
        }
        out.push_back(std::move(cur));
    }
    return out;
}

// Build the complement graph Ḡ: same vertices 0..n-1; edge (u,v) iff !g.hasEdge(u,v). O(n^2).
inline Graph complementGraph(const Graph& g) {
    const int n = g.numVertices();
    Graph h(n);
    for (int u = 0; u < n; ++u)
        for (int v = u + 1; v < n; ++v)
            if (!g.hasEdge(u, v)) h.addEdge(u, v);
    return h;
}

// Enumerate all set-partitions of `items` (Bell number many). {} -> one empty partition.
// Each partition is a vector of blocks; each block is a vector<int>.
inline std::vector<std::vector<std::vector<int>>> setPartitions(const std::vector<int>& items) {
    std::vector<std::vector<std::vector<int>>> out;
    if (items.empty()) { out.push_back({}); return out; }
    std::vector<int> rest(items.begin() + 1, items.end());
    int first = items.front();
    for (auto& parts : setPartitions(rest)) {
        for (std::size_t i = 0; i < parts.size(); ++i) {   // add `first` to each existing block
            auto np = parts;
            np[i].insert(np[i].begin(), first);
            out.push_back(std::move(np));
        }
        auto np = parts;                                    // or place `first` in its own new block
        np.insert(np.begin(), std::vector<int>{first});
        out.push_back(std::move(np));
    }
    return out;
}

// Build the piece graph for the complement clique-separator decomposition.
// Layout: nodes 0..q-1 = q beta-reps (one per tau block), nodes q..q+|comp|-1 = comp vertices.
// Edges: beta-reps form a clique; comp-internal edges copied from g;
// beta-rep t <-> comp-vertex u iff (some vertex of tau[t] is g-adjacent to u) OR (t not in allowedBlocks).
inline Graph pieceGraph(const Graph& g, const std::vector<int>& comp,
                        const std::vector<std::vector<int>>& tau,
                        const std::set<int>& allowedBlocks) {
    const int q = static_cast<int>(tau.size());
    const int nc = static_cast<int>(comp.size());
    Graph h(q + nc);
    for (int a = 0; a < q; ++a)                       // beta-clique
        for (int b = a + 1; b < q; ++b) h.addEdge(a, b);
    for (int i = 0; i < nc; ++i)                       // comp-internal edges
        for (int j = i + 1; j < nc; ++j)
            if (g.hasEdge(comp[i], comp[j])) h.addEdge(q + i, q + j);
    for (int t = 0; t < q; ++t) {                      // beta-rep t <-> comp vertices
        bool forbidden = (allowedBlocks.count(t) == 0);
        for (int i = 0; i < nc; ++i) {
            bool natural = false;
            for (int s : tau[t]) if (g.hasEdge(s, comp[i])) { natural = true; break; }
            if (natural || forbidden) h.addEdge(t, q + i);
        }
    }
    return h;
}

namespace detail {
// δ_strip(j, D): inner exactly-claimed inclusion-exclusion over subsets D' ⊆ D, with the q
// forced β-blocks stripped from each piece a-vector. D holds block indices (0..q-1).
// Sign: (-1)^(|D| - |D'|). Strip: drop the first q entries of A (the β-block slots).
//
// pieceCache: per-tau memo keyed by (j, allowedQMask). allowedQMask is a q-bit mask over
// tau-block indices {0..q-1} identifying which blocks are in the allowed set. Each distinct
// (j, allowed-set) piece graph is computed exactly once per tau and reused across all assign
// and D' terms, as specified in the "General l" section of the design spec. The caller must
// reset the cache between tau iterations (by creating it inside the tau loop).
inline std::vector<mpz_class> deltaStrip(
        const Graph& g, const std::vector<int>& comp, int j,
        const std::vector<std::vector<int>>& tau, const std::vector<int>& D,
        const std::function<Poly(const Graph&)>& chi,
        std::map<std::pair<int,int>, std::vector<mpz_class>>& pieceCache) {
    const int q = static_cast<int>(tau.size());
    const int d = static_cast<int>(D.size());
    std::vector<mpz_class> res(1, mpz_class(0));
    for (int mask = 0; mask < (1 << d); ++mask) {
        std::set<int> allowed;
        int allowedQMask = 0;
        int bits = 0;
        for (int i = 0; i < d; ++i) if (mask & (1 << i)) {
            allowed.insert(D[i]);
            allowedQMask |= (1 << D[i]);
            ++bits;
        }
        auto key = std::make_pair(j, allowedQMask);
        auto it = pieceCache.find(key);
        std::vector<mpz_class> strip;
        if (it != pieceCache.end()) {
            strip = it->second;                             // reuse previously computed result
        } else {
            Graph pg = pieceGraph(g, comp, tau, allowed);
            std::vector<mpz_class> A = chi(pg).toFallingFactorial();
            if (static_cast<int>(A.size()) > q) strip.assign(A.begin() + q, A.end());
            else strip.assign(1, mpz_class(0));
            pieceCache.emplace(key, strip);                 // store for reuse
        }
        mpz_class sign = ((d - bits) & 1) ? mpz_class(-1) : mpz_class(1);
        avecAddInPlace(res, strip, sign);
    }
    return res;
}
} // namespace detail

// Verified general-l complement clique-separator decomposition (dual-CRT-1).
// Returns χ(G) in the power basis when the complement Ḡ has a clique separator S of size l,
// where comps = components of Ḡ - S (both given as vertex labels in g).
//
// Formula (falling-factorial a-vector):
//   a(G) = Σ_{τ∈Part(S)} Σ_{assign: blocks(τ)→{0..p}} shift_q( ⊛_j δ_j )
//   q = |τ|; D_j = τ-blocks assigned to component j (assign=0 ↔ unclaimed/pure-S).
//   δ_j = deltaStrip(comp_j, τ, D_j) = strip_q( Σ_{D'⊆D_j} (-1)^{|D_j|-|D'|} A_le(j,D') )
//   shift_q = prepend q zeros (re-add the q shared β slots).
//
// Correctness: verified 348/348 random instances vs brute-force oracle in the reference
// (docs/superpowers/references/2026-06-30-complement-clique-decomp-reference.py).
inline Poly complementCliqueDecompose(
        const Graph& g, const std::vector<int>& S,
        const std::vector<std::vector<int>>& comps,
        const std::function<Poly(const Graph&)>& chi) {
    const int p = static_cast<int>(comps.size());
    std::vector<mpz_class> total(1, mpz_class(0));
    for (auto& tau : setPartitions(S)) {
        const int q = static_cast<int>(tau.size());
        // Per-tau piece memo: key=(j, allowedQMask), value=stripped a-vector.
        // allowedQMask is a q-bit mask over tau-block indices {0..q-1}. Each distinct
        // (j, allowed-set) piece graph is computed exactly once and reused across all
        // assign and D' terms, implementing the spec's "computed once and reused" promise.
        // Scoped inside the tau loop so it is automatically reset per tau (different
        // contraction → different graph; sharing across tau would be incorrect).
        std::map<std::pair<int,int>, std::vector<mpz_class>> pieceCache;
        // assign[t] ∈ {0..p}: 0 = pure-S (unclaimed), j = component j-1 claims block t.
        std::vector<int> assign(q, 0);
        while (true) {
            std::vector<std::vector<int>> Ds(p);
            for (int t = 0; t < q; ++t) if (assign[t] >= 1) Ds[assign[t] - 1].push_back(t);
            std::vector<mpz_class> comb(1, mpz_class(1));    // umbral identity
            for (int j = 0; j < p; ++j)
                comb = avecConvolve(comb, detail::deltaStrip(g, comps[j], j, tau, Ds[j], chi, pieceCache));
            avecAddInPlace(total, avecShiftUp(comb, q), mpz_class(1));
            // increment mixed-radix assign in base (p+1)
            int t = 0;
            for (; t < q; ++t) { if (++assign[t] <= p) break; assign[t] = 0; }
            if (t == q) break;
        }
    }
    return Poly::fromFallingFactorial(total);
}

// Build the induced subgraph on a vertex subset, relabelled to 0..|subset|-1.
inline Graph inducedSubgraph(const Graph& g, const std::vector<int>& verts) {
    std::vector<int> idx(g.numVertices(), -1);
    for (int i = 0; i < (int)verts.size(); ++i) idx[verts[i]] = i;
    Graph h((int)verts.size());
    for (int v : verts)
        for (int w : g.neighbours(v))
            if (idx[w] != -1 && v < w) h.addEdge(idx[v], idx[w]);
    return h;
}

// Biconnected components (blocks) as vertex-index lists in ORIGINAL labels, via
// Hopcroft-Tarjan DFS (disc/low + an edge stack). A connected graph with no cut vertex
// returns a single block; an isolated vertex is returned as its own trivial block.
// O(n+m). Retained as a tested helper; the engine's CRT-1 reduction now uses cliqueSeparator for all l>=1.
inline std::vector<std::vector<int>> biconnectedComponents(const Graph& g) {
    const int n = g.numVertices();
    std::vector<int> disc(n, -1), low(n, 0);
    std::vector<std::pair<int,int>> estack;
    std::vector<std::vector<int>> blocks;
    int timer = 0;

    std::function<void(int,int)> dfs = [&](int u, int parent) {
        disc[u] = low[u] = ++timer;
        for (int w : g.neighbours(u)) {
            if (disc[w] == -1) {                       // tree edge
                estack.emplace_back(u, w);
                dfs(w, u);
                low[u] = std::min(low[u], low[w]);
                if (low[w] >= disc[u]) {                // u is a cut vertex (or root): pop a block
                    std::set<int> verts;
                    while (true) {
                        auto e = estack.back(); estack.pop_back();
                        verts.insert(e.first); verts.insert(e.second);
                        if (e.first == u && e.second == w) break;
                    }
                    blocks.emplace_back(verts.begin(), verts.end());
                }
            } else if (w != parent && disc[w] < disc[u]) {   // back edge (pushed once)
                estack.emplace_back(u, w);
                low[u] = std::min(low[u], disc[w]);
            }
        }
    };

    for (int s = 0; s < n; ++s) {
        if (disc[s] != -1) continue;
        dfs(s, -1);
        if (g.degree(s) == 0) blocks.push_back({s});   // isolated vertex: trivial block
    }
    return blocks;
}

// True iff removing the vertex set S (in g's labels) disconnects g into >1 component.
// S itself is excluded; we test connectivity of the induced subgraph on V \ S.
inline bool separatesInto(const Graph& g, const std::vector<int>& S) {
    std::vector<char> inS(g.numVertices(), 0);
    for (int v : S) inS[v] = 1;
    std::vector<int> rest;
    for (int v = 0; v < g.numVertices(); ++v) if (!inS[v]) rest.push_back(v);
    if (rest.empty()) return false;
    Graph h = inducedSubgraph(g, rest);
    return components(h).size() > 1;
}

// For each component C of g - S, the piece G[C ∪ S], in g's ORIGINAL labels.
// Used by the CRT-1 clique-separator reduction.
inline std::vector<std::vector<int>> cliqueSeparatorPieces(const Graph& g,
                                                           const std::vector<int>& S) {
    std::vector<char> inS(g.numVertices(), 0);
    for (int v : S) inS[v] = 1;
    std::vector<int> rest;
    for (int v = 0; v < g.numVertices(); ++v) if (!inS[v]) rest.push_back(v);
    Graph h = inducedSubgraph(g, rest);                 // labels 0..rest.size()-1 index into rest
    std::vector<std::vector<int>> pieces;
    for (auto& comp : components(h)) {
        std::vector<int> piece = S;                     // glue the shared clique onto every piece
        for (int local : comp) piece.push_back(rest[local]);
        pieces.push_back(std::move(piece));
    }
    return pieces;
}

// MCS-M minimal triangulation (Berry, Blair, Heggernes, Peyton 2004).
// Fills H = G + fill-edges to a chordal graph and numbers vertices alpha[v] in 1..n
// (vertex chosen at outer step i gets alpha = i; higher numbers were chosen earlier).
// The "madj" sets {H-neighbours with higher alpha} are clique-separator candidates:
// in the chordal H they are cliques, so they are the right places to look for a
// clique separator of the original G. O(n*m).
//
// Reachability rule (defines the fill): when numbering u, an unnumbered vertex v gets
// edge (u,v) in H iff there is a path u-x1-...-v in G whose interior vertices xj are
// unnumbered with weight w(xj) < w(v). Implemented with weight buckets.
inline void mcsmTriangulation(const Graph& g, Graph& H, std::vector<int>& alpha) {
    const int n = g.numVertices();
    H = g;                                   // start from G; only add fill edges
    alpha.assign(n, 0);
    std::vector<int> w(n, 0);                // weights
    std::vector<char> numbered(n, 0);

    for (int i = n; i >= 1; --i) {
        int u = -1;                          // unnumbered vertex of maximum weight
        for (int v = 0; v < n; ++v)
            if (!numbered[v] && (u == -1 || w[v] > w[u])) u = v;
        alpha[u] = i;
        numbered[u] = 1;

        std::vector<char> reached(n, 0);
        std::vector<std::vector<int>> bucket(n);   // bucket[wt] = frontier at path-weight wt
        std::vector<int> R;                        // vertices that get an edge to u in H
        reached[u] = 1;
        for (int y : g.neighbours(u)) {            // direct neighbours always reach u
            if (numbered[y] || reached[y]) continue;
            reached[y] = 1;
            bucket[w[y]].push_back(y);
            R.push_back(y);
        }
        for (int wt = 0; wt < n; ++wt) {
            for (std::size_t idx = 0; idx < bucket[wt].size(); ++idx) {
                int y = bucket[wt][idx];
                for (int z : g.neighbours(y)) {
                    if (numbered[z] || reached[z]) continue;   // skip numbered or already-reached (incl. u)
                    reached[z] = 1;
                    if (w[z] > wt) {               // reached via lower-weight path -> connect
                        R.push_back(z);
                        bucket[w[z]].push_back(z);
                    } else {                        // conduit: extend the path at this level
                        bucket[wt].push_back(z);
                    }
                }
            }
        }
        for (int v : R) { H.addEdge(u, v); w[v] += 1; }   // add fill (no-op if already edge)
    }
}

// True iff every pair of vertices in S is adjacent in g (S induces a complete subgraph).
// A set of size 0 or 1 is trivially a clique.
inline bool isCliqueInG(const Graph& g, const std::vector<int>& S) {
    for (std::size_t i = 0; i < S.size(); ++i)
        for (std::size_t j = i + 1; j < S.size(); ++j)
            if (!g.hasEdge(S[i], S[j])) return false;
    return true;
}

// --- Clique-separator family ---
// The functions below enumerate and select clique separators via MCS-M triangulation.
// Each candidate is a clique in the ORIGINAL g whose removal disconnects g; the guard
// makes the result safe regardless of MCS-M subtleties (false candidates are skipped,
// never yielding a wrong decomposition). Subsumes the former l=1 block step and the
// l=2/3 brute force.  O(n*m) triangulation, O(n*(n+m)) total including separation tests.
// Collect ALL distinct valid MCS-M-surfaced clique separators of g (each a clique in the
// ORIGINAL g whose removal disconnects g). Inner vectors are sorted ascending (deterministic
// piece-label order); the outer list is de-duplicated, preserving first-occurrence in v=0..n-1
// order. Empty iff g is an atom (no clique separator).
inline std::vector<std::vector<int>> cliqueSeparatorCandidates(const Graph& g) {
    const int n = g.numVertices();
    Graph H; std::vector<int> alpha;
    mcsmTriangulation(g, H, alpha);

    std::vector<std::vector<int>> cands;
    for (int v = 0; v < n; ++v) {
        std::vector<int> S;                          // madj_H(v): H-neighbours numbered higher
        for (int wv : H.neighbours(v))
            if (alpha[wv] > alpha[v]) S.push_back(wv);
        if (S.empty()) continue;
        if (!isCliqueInG(g, S)) continue;            // clique in the original graph
        if (!separatesInto(g, S)) continue;          // actually disconnects g
        std::sort(S.begin(), S.end());               // deterministic label order
        bool dup = false;                            // de-duplicate (different v can surface same S)
        for (auto& c : cands) if (c == S) { dup = true; break; }
        if (!dup) cands.push_back(std::move(S));
    }
    return cands;
}

// Stable argmin by separator size (the historical tie-break: smallest size, and among equal
// sizes the first in surfacing order). Empty input -> empty result.
inline std::vector<int> pickSmallestFirst(const std::vector<std::vector<int>>& cands) {
    const std::vector<int>* best = nullptr;
    for (const auto& c : cands)
        if (best == nullptr || c.size() < best->size()) best = &c;
    return best ? *best : std::vector<int>{};
}

// Smallest clique minimal separator, or empty if g is an atom. Fast short-circuiting search
// (prunes candidates that cannot beat the current best BEFORE the clique/separation tests, and
// stops at the first size-1 separator). This is the DEFAULT production path; the full candidate
// enumeration (cliqueSeparatorCandidates) is paid only when a CliqueSeparatorSelector is installed.
inline std::vector<int> cliqueSeparator(const Graph& g) {
    const int n = g.numVertices();
    Graph H; std::vector<int> alpha;
    mcsmTriangulation(g, H, alpha);
    std::vector<int> best;
    for (int v = 0; v < n; ++v) {
        std::vector<int> S;
        for (int wv : H.neighbours(v))
            if (alpha[wv] > alpha[v]) S.push_back(wv);
        if (S.empty()) continue;
        if (!best.empty() && S.size() >= best.size()) continue;
        if (!isCliqueInG(g, S)) continue;
        if (!separatesInto(g, S)) continue;
        best = std::move(S);
        if (best.size() == 1) break;
    }
    std::sort(best.begin(), best.end());
    return best;
}

inline bool isCycle(const Graph& g) {
    int n = g.numVertices();
    if (n < 3 || g.numEdges() != n) return false;
    for (int v = 0; v < n; ++v) if (g.degree(v) != 2) return false;
    return true; // caller must ensure connectivity; then n edges + all degree 2 => single cycle
}

// True iff g is a wheel W_n (n>=4): a single hub adjacent to all other vertices whose
// removal leaves a cycle. On success, `hub` is set to the hub's index. O(n+m).
// (K_4 is also W_4 but is caught by the complete-graph base case before this runs.)
inline bool isWheel(const Graph& g, int& hub) {
    const int n = g.numVertices();
    if (n < 4) return false;
    if (g.numEdges() != 2LL * (n - 1)) return false;       // n-1 spokes + n-1 rim edges
    int h = -1;
    for (int v = 0; v < n; ++v)
        if (g.degree(v) == n - 1) { h = v; break; }         // hub touches all others
    if (h < 0) return false;
    std::vector<int> rest;
    for (int v = 0; v < n; ++v) if (v != h) rest.push_back(v);
    Graph rim = inducedSubgraph(g, rest);
    if (components(rim).size() != 1 || !isCycle(rim)) return false;  // rim must be ONE cycle
    hub = h;
    return true;
}

// ---- caching + stats infrastructure -------------------------------------------------

// nodes = computation-tree nodes entered; branches = delete-contract expansions.
// cur_live_words / peak_live_words = running and high-water sum of coefficient-limbs
// held simultaneously across the live recursion stack (a cheap peak-memory proxy).
struct Stats {
    long long nodes = 0;
    long long branches = 0;
    long long budget = -1;        // branch-axis cap; -1 = unlimited
    long long node_budget = -1;   // node-axis cap; -1 = unlimited. The PPO reward axis:
                                  // a branch cap does NOT bound nodes (reductions add
                                  // nodes without branching), so a controller that
                                  // defers reductions can run away inside a branch
                                  // budget. Measured 2026-08-16: erdos_renyi p=0.3
                                  // n=18 reached 3,269,961 nodes under a 200k branch cap.
    long long cur_live_words = 0;
    long long peak_live_words = 0;
    long long complement_enum_terms = 0;   // deterministic decompose-cost proxy (Touchard terms)
};

namespace detail {
// RAII: while alive, `words` coefficient-limbs are counted as held; the running
// counter's high-water mark is tracked. Used around recursive child calls so an
// accumulator/child polynomial held across a deeper recursion is counted.
struct LiveWordsScope {
    Stats& st;
    long long w;
    LiveWordsScope(Stats& s, long long words) : st(s), w(words) {
        st.cur_live_words += w;
        if (st.cur_live_words > st.peak_live_words) st.peak_live_words = st.cur_live_words;
    }
    ~LiveWordsScope() { st.cur_live_words -= w; }
    LiveWordsScope(const LiveWordsScope&) = delete;
    LiveWordsScope& operator=(const LiveWordsScope&) = delete;
};
} // namespace detail;

// A cache that stores nothing: the zero-overhead default (no virtual dispatch, inlines away).
struct NullCache {
    const Poly* lookup(const Graph&) { return nullptr; }
    void store(const Graph&, const Poly&) {}
};

constexpr int kComplementSepMaxL = 3;

// Per-reduction control for the decomposition-timing diagnostic. Global-static, matching
// complementSepEnabled(); single-threaded on the cache path. Indices are ReductionKind.
namespace detail {
enum ReductionKind { RK_JOIN = 0, RK_CLIQUE = 1, RK_COMPLEMENT = 2 };
struct ReductionConfig {
    bool   enabled[3] = {true, true, true};
    double defer_p[3] = {1.0, 1.0, 1.0};
    std::mt19937_64 rng{0};
    // Call ONLY when the reduction is applicable (a separator/decomposition exists),
    // so the RNG advances once per genuine deferral opportunity.
    bool shouldFire(int k) {
        if (!enabled[k]) return false;
        if (defer_p[k] >= 1.0) return true;     // eager: no RNG draw (default path)
        if (defer_p[k] <= 0.0) return false;    // never
        std::uniform_real_distribution<double> u(0.0, 1.0);
        return u(rng) < defer_p[k];
    }
    void reseed(std::uint64_t s) { rng.seed(s); }
};
} // namespace detail

inline detail::ReductionConfig& reductionConfig() {
    static detail::ReductionConfig c;
    return c;
}

// Runtime toggle for the complement clique-separator reduction.
// Default: true (reduction active). Set to false for A/B benchmarking.
// Thread-local in theory but the engine is single-threaded; using a plain static bool
// avoids ODR issues in header-only code. Do NOT set in production. Only for benchmarks.
inline bool& complementSepEnabled() { static bool e = true; return e; }

// Optional separator-selection seam: choose which detected clique separator to decompose on.
// Consulted at the clique-separator firing site; must return one of the provided candidates.
// Default: empty (the engine uses pickSmallestFirst, the historical smallest-first choice).
using CliqueSeparatorSelector =
    std::function<std::vector<int>(const Graph&, const std::vector<std::vector<int>>&)>;
inline CliqueSeparatorSelector& cliqueSeparatorSelector() {
    static CliqueSeparatorSelector s; return s;
}

// Per-node context for the "when to apply complement_sep" decision. `graph` is the current node's
// graph (borrowed reference, valid only during the decider call). The scalar fields are the features.
// NOTE: n and m are stored as int (cast from int n / long long m at the call site); safe for the
// graph sizes this engine handles: would truncate only at absurd vertex/edge counts.
struct ComplementSepContext {
    const Graph& graph;
    int    l;             // separator size |S|
    int    p;             // piece count (components of Ḡ - S)
    double touchard;      // touchardTerms(l, p) - analytic local decompose cost
    int    n;             // node vertex count (int; see NOTE above)
    int    m;             // node edge count (int; see NOTE above)
    double comp_density;  // fraction of complement edges: (C(n,2) - m) / C(n,2)
};
// Optional decider: return true to apply complementCliqueDecompose, false to branch instead. When
// empty (default), the engine uses reductionConfig().shouldFire(RK_COMPLEMENT) - byte-for-byte the
// current path. Single-threaded on the cache path, like reductionConfig()/complementSepEnabled().
using ComplementSepDecider = std::function<bool(const ComplementSepContext&)>;
inline ComplementSepDecider& complementSepDecider() { static ComplementSepDecider d; return d; }

// ---------------------------------------------------------------------------
// Unified solver-control seam. When installed, ONE policy chooses at every node
// between the available decompositions and branching (target AND direction).
// When empty (the default) the engine takes its historical path unchanged, so
// the expensive candidate enumeration is paid only when a policy is present;
// the same arrangement CliqueSeparatorSelector already uses.
// ---------------------------------------------------------------------------
enum class ActionKind { FireJoin, FireClique, FireComplement, BranchDown, BranchUp };

// A complement clique separator together with the components of Gbar - S, so the
// firing site never has to recompute them.
struct ComplementSepCandidate {
    std::vector<int> sep;
    std::vector<std::vector<int>> comps;
};

struct Action {
    ActionKind kind = ActionKind::BranchDown;
    std::vector<int> sep;              // FireClique / FireComplement only
    std::pair<int,int> edge{-1, -1};   // BranchDown / BranchUp only
};

struct ActionSet {
    bool join_available = false;
    std::vector<std::vector<int>> join_parts;             // complementComponents(g) when available
    std::vector<std::vector<int>> clique_seps;
    std::vector<ComplementSepCandidate> complement_seps;
    std::vector<std::pair<int,int>> down_edges;
    std::vector<std::pair<int,int>> up_edges;             // non-empty only when 2m >= C(n,2)
};

using SolverPolicy = std::function<Action(const Graph&, const ActionSet&)>;
inline SolverPolicy& solverPolicy() { static SolverPolicy p; return p; }

// Detect a complement clique separator of size 1..kComplementSepMaxL in g.
// If Gbar = complementGraph(g) has a clique separator S of size l in [1, kComplementSepMaxL],
// returns true, fills S (vertex labels in g), and fills comps with the components of Gbar-S
// (vertex labels in g). Returns false if no such separator exists or Gbar is an atom.
// Precondition: g is connected and complement(g) is connected (join reduction already handled
// the disconnected-complement case). Pieces are strictly smaller than g, so recursion terminates.
inline bool complementCliqueSeparatorPieces(const Graph& g, std::vector<int>& S,
                                            std::vector<std::vector<int>>& comps) {
    Graph gbar = complementGraph(g);
    S = cliqueSeparator(gbar);                       // clique in Gbar = independent set in g
    if (S.empty() || static_cast<int>(S.size()) > kComplementSepMaxL) return false;
    std::vector<char> inS(g.numVertices(), 0);
    for (int v : S) inS[v] = 1;
    std::vector<int> rest;
    for (int v = 0; v < g.numVertices(); ++v) if (!inS[v]) rest.push_back(v);
    Graph h = inducedSubgraph(gbar, rest);           // Gbar - S, relabelled 0..rest.size()-1
    comps.clear();
    for (auto& comp : components(h)) {               // map local labels back to g labels
        std::vector<int> c;
        for (int local : comp) c.push_back(rest[local]);
        comps.push_back(std::move(c));
    }
    return comps.size() > 1;                          // a separator must split into ≥2 pieces
}

// All complement clique separators of size 1..kComplementSepMaxL, each paired with the
// components of Gbar - S (vertex labels in g). Mirrors complementCliqueSeparatorPieces
// exactly (same relabel-and-map-back order, no extra sorting) but enumerates every
// candidate instead of only the smallest. Paid ONLY on the SolverPolicy path.
inline std::vector<ComplementSepCandidate>
complementCliqueSeparatorCandidates(const Graph& g) {
    std::vector<ComplementSepCandidate> out;
    Graph gbar = complementGraph(g);
    for (auto& S : cliqueSeparatorCandidates(gbar)) {
        if (S.empty() || static_cast<int>(S.size()) > kComplementSepMaxL) continue;
        std::vector<char> inS(g.numVertices(), 0);
        for (int v : S) inS[v] = 1;
        std::vector<int> rest;
        for (int v = 0; v < g.numVertices(); ++v) if (!inS[v]) rest.push_back(v);
        Graph h = inducedSubgraph(gbar, rest);       // Gbar - S, relabelled 0..rest.size()-1
        ComplementSepCandidate cand;
        cand.sep = S;
        for (auto& comp : components(h)) {           // map local labels back to g labels
            std::vector<int> c;
            for (int local : comp) c.push_back(rest[local]);
            cand.comps.push_back(std::move(c));
        }
        if (cand.comps.size() < 2) continue;         // not a separator of Gbar after all
        out.push_back(std::move(cand));
    }
    return out;
}

// Assemble every legal action at this node. Called ONLY when a SolverPolicy is installed.
// Preconditions (guaranteed by the caller): m > 0, g connected, not a base-case shape.
inline ActionSet buildActionSet(const Graph& g, bool bidir) {
    ActionSet as;
    const int n = g.numVertices();
    const long long m = g.numEdges();
    const long long maxE = (long long)n * (n - 1) / 2;

    auto cc = complementComponents(g);
    if (cc.size() > 1) { as.join_available = true; as.join_parts = std::move(cc); }

    as.clique_seps = cliqueSeparatorCandidates(g);
    as.complement_seps = complementCliqueSeparatorCandidates(g);

    // Phi-guard. With BOTH directions available the recursion must stay a DAG ordered by
    // Phi = min(m, C(n,2) - m); otherwise G -> G+f -> (G+f)-f = G cycles and the engine
    // never terminates. So in bidir mode down is legal only on the sparse side and up only
    // on the dense side (both legal at the 2m == maxE boundary). In down-only mode there is
    // no up move, hence no cycle, and every edge is a legal down-branch.
    // These masks ARE the guard: a policy cannot express an illegal move.
    if (!bidir || 2 * m <= maxE)
        for (int u = 0; u < n; ++u)
            for (int v = u + 1; v < n; ++v)
                if (g.hasEdge(u, v)) as.down_edges.push_back({u, v});

    if (bidir && 2 * m >= maxE)
        for (int u = 0; u < n; ++u)
            for (int v = u + 1; v < n; ++v)
                if (!g.hasEdge(u, v)) as.up_edges.push_back({u, v});

    return as;
}

// Stirling numbers of the second kind S(n,k): partitions of n labels into k blocks.
inline double stirling2(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    if (k == 0) return n == 0 ? 1.0 : 0.0;
    std::vector<double> prev(k + 1, 0.0), cur(k + 1, 0.0);
    prev[0] = 1.0;
    for (int i = 1; i <= n; ++i) {
        cur.assign(k + 1, 0.0);
        for (int j = 1; j <= std::min(i, k); ++j)
            cur[j] = j * prev[j] + prev[j - 1];
        std::swap(prev, cur);
    }
    return prev[k];
}

// Closed-form local cost of complementCliqueDecompose on a separator of size l with p pieces:
// the inner loop runs Σ_{τ∈Part(S)} (p+1)^|τ| = Σ_{q=1..l} S(l,q)·(p+1)^q umbral-convolution terms
// (the Touchard polynomial B_l(p+1)). Deterministic in (l, p); this is the wall_s cost driver the
// decomposition-timing diagnostic identified. Used both as the proxy counter and as a feature.
inline double touchardTerms(int l, int p) {
    double total = 0.0, pw = 1.0;
    for (int q = 1; q <= l; ++q) { pw *= (p + 1); total += stirling2(l, q) * pw; }
    return total;
}

// ---- the engine ---------------------------------------------------------------------
//
// Templated on the cache type so NullCache reproduces the original cache-free engine with
// no overhead, while a nauty-backed cache (nauty_cache.hpp) plugs in without touching this
// file or pulling nauty into the core.
//
// Cost ordering is deliberate: the cheap base-case reductions (O(n+m)) run BEFORE we pay
// for canonicalization, so the (expensive) cache is consulted only on graphs we would
// otherwise branch on. That is exactly the tradeoff this project studies.
template <class Cache, class Recorder>
Poly computeChromaticT(const Graph& g, const EdgeSelector& select, Cache& cache,
                       Stats& st, Recorder& rec, int recCtx, bool bidir = false) {
    ++st.nodes;
    // Checked before rec.enter() so the node that busts the cap leaves no half-open
    // record behind; every already-entered node still has its span closed by
    // NodeSpanScope's destructor as this exception unwinds the stack.
    if (st.node_budget >= 0 && st.nodes > st.node_budget) throw BudgetExceeded();
    const int n = g.numVertices();
    const long long m = g.numEdges();

    const int myId = rec.enter(recCtx, g);

    // Writes subtree_nodes on EVERY return path (including the early base cases), so the
    // tree-MDP identity C(v) = 1 + sum C(children) holds by construction. `start` is
    // st.nodes - 1 because ++st.nodes already ran for this node, so st.nodes - start counts
    // this node plus everything visited beneath it. This is the PPO return for `nodes`.
    struct NodeSpanScope {
        Stats& st; Recorder& rec; int id; long long start;
        NodeSpanScope(Stats& s, Recorder& r, int i)
            : st(s), rec(r), id(i), start(s.nodes - 1) {}
        ~NodeSpanScope() { rec.endNodes(id, st.nodes - start); }
    } _span(st, rec, myId);

    if (m == 0) { rec.classify(myId, NodeKind::BaseEmpty); return chromaticEmpty(n); }

    auto comps = components(g);
    if (comps.size() > 1) {                        // disconnected -> product
        rec.classify(myId, NodeKind::Disconnected);
        Poly prod = Poly::one();
        for (auto& c : comps) {
            detail::LiveWordsScope hold(st, polyWords(prod));
            prod = prod * computeChromaticT(inducedSubgraph(g, c), select, cache,
                                            st, rec, rec.spawnPiece(myId, c), bidir);
        }
        return prod;
    }

    if (m == n - 1) { rec.classify(myId, NodeKind::BaseTree); return chromaticTree(n); }
    if (m == (long long)n * (n - 1) / 2) {
        rec.classify(myId, NodeKind::BaseComplete); return chromaticComplete(n);
    }
    if (isCycle(g)) { rec.classify(myId, NodeKind::BaseCycle); return chromaticCycle(n); }
    { int hub; if (isWheel(g, hub)) { rec.classify(myId, NodeKind::BaseWheel); return chromaticWheel(n); } }

    // ---- Unified SolverPolicy seam -------------------------------------------------
    // When installed, ONE policy chooses among all legal actions: fire a join /
    // clique-separator / complement-clique-separator decomposition (and WHICH separator),
    // or branch (target AND direction). Falls through to the historical path when empty,
    // so the default engine (including its cheap short-circuiting cliqueSeparator search)
    // is untouched. Every action strictly decreases a well-founded measure, so termination
    // holds for ANY policy; every action is an exact identity, so the polynomial cannot change.
    if (SolverPolicy& pol = solverPolicy()) {
        ActionSet as = buildActionSet(g, bidir);
        Action act = pol(g, as);

        // An edge is an UNORDERED pair everywhere else in the engine (deleteEdge and
        // insertEdge are symmetric in their endpoints, and contractEdge normalises with
        // min/max internally) while buildActionSet emits every candidate as (u,v) with
        // u<v. Normalise the policy's pair to that convention so legality never depends
        // on orientation. Without this, the prior-art rules that pick a vertex and THEN
        // one of its neighbours (mindeg, mindeg_dt, monagan) raise "not an edge of g"
        // through the seam on any node where the neighbour's index is the lower one,
        // despite working fine on the plain selector= path. The membership checks below
        // are unchanged, so a genuinely illegal action still raises.
        if ((act.kind == ActionKind::BranchDown || act.kind == ActionKind::BranchUp) &&
            act.edge.first > act.edge.second)
            act.edge = {act.edge.second, act.edge.first};

        switch (act.kind) {
        case ActionKind::FireJoin:       rec.recordAction(myId, "FireJoin"); break;
        case ActionKind::FireClique:     rec.recordAction(myId, "FireClique"); break;
        case ActionKind::FireComplement: rec.recordAction(myId, "FireComplement"); break;
        case ActionKind::BranchDown:     rec.recordAction(myId, "BranchDown"); break;
        case ActionKind::BranchUp:       rec.recordAction(myId, "BranchUp"); break;
        }

        auto contains = [](const std::vector<std::pair<int,int>>& v,
                           const std::pair<int,int>& e) {
            for (auto& x : v) if (x == e) return true;
            return false;
        };

        switch (act.kind) {
        case ActionKind::FireJoin: {
            if (!as.join_available)
                throw std::runtime_error("SolverPolicy: join not available at this node");
            rec.classify(myId, NodeKind::Join);
            Poly prod = Poly::one();
            for (auto& part : as.join_parts) {
                detail::LiveWordsScope hold(st, polyWords(prod));
                prod = umbralProduct(prod,
                         computeChromaticT(inducedSubgraph(g, part), select, cache,
                                           st, rec, rec.spawnPiece(myId, part), bidir));
            }
            return prod;
        }
        case ActionKind::FireClique: {
            bool ok = false;
            for (auto& c : as.clique_seps) if (c == act.sep) { ok = true; break; }
            if (!ok) throw std::runtime_error("SolverPolicy: clique separator not a candidate");
            const int l = static_cast<int>(act.sep.size());
            auto pieces = cliqueSeparatorPieces(g, act.sep);
            rec.classifySep(myId, NodeKind::CliqueSep, act.sep, {});
            Poly prod = Poly::one();
            for (auto& verts : pieces) {
                detail::LiveWordsScope hold(st, polyWords(prod));
                prod = prod * computeChromaticT(inducedSubgraph(g, verts), select, cache,
                                                st, rec, rec.spawnPiece(myId, verts), bidir);
            }
            const Poly div = chromaticComplete(l);
            for (int i = 1; i < static_cast<int>(pieces.size()); ++i) prod = prod.divExact(div);
            return prod;
        }
        case ActionKind::FireComplement: {
            const ComplementSepCandidate* pick = nullptr;
            for (auto& c : as.complement_seps) if (c.sep == act.sep) { pick = &c; break; }
            if (!pick)
                throw std::runtime_error("SolverPolicy: complement separator not a candidate");
            rec.classifySep(myId, NodeKind::ComplementSep, pick->sep, pick->comps);
            st.complement_enum_terms += std::llround(
                touchardTerms(static_cast<int>(pick->sep.size()),
                              static_cast<int>(pick->comps.size())));
            auto chi = [&](const Graph& h) -> Poly {
                return computeChromaticT(h, select, cache, st, rec, rec.spawnIE(myId), bidir);
            };
            return complementCliqueDecompose(g, pick->sep, pick->comps, chi);
        }
        case ActionKind::BranchDown: {
            if (!contains(as.down_edges, act.edge))
                throw std::runtime_error("SolverPolicy: BranchDown edge is not an edge of g");
            break;                                  // fall through to the shared branch code
        }
        case ActionKind::BranchUp: {
            if (!contains(as.up_edges, act.edge))
                throw std::runtime_error("SolverPolicy: BranchUp target is not a legal non-edge");
            break;
        }
        }

        // Chose to branch: consult the cache first, exactly as the default path does.
        if (const Poly* hit = cache.lookup(g)) {
            rec.classify(myId, NodeKind::CacheHit);
            return *hit;
        }
        const long long bStartP = st.branches;
        ++st.branches;
        if (st.budget >= 0 && st.branches > st.budget) throw BudgetExceeded();
        const int pu = act.edge.first, pv = act.edge.second;
        const int myBranchP = rec.branch(myId, g, {pu, pv});
        Poly resP;
        if (act.kind == ActionKind::BranchUp) {
            Poly a = computeChromaticT(g.insertEdge(pu, pv), select, cache, st, rec,
                                       rec.spawnEdge(myBranchP, ViaKind::Insert, pu, pv), bidir);
            detail::LiveWordsScope hold(st, polyWords(a));
            Poly b = computeChromaticT(g.contractEdge(pu, pv), select, cache, st, rec,
                                       rec.spawnEdge(myBranchP, ViaKind::Contract, pu, pv), bidir);
            resP = a + b;
        } else {
            Poly a = computeChromaticT(g.deleteEdge(pu, pv), select, cache, st, rec,
                                       rec.spawnEdge(myBranchP, ViaKind::Delete, pu, pv), bidir);
            detail::LiveWordsScope hold(st, polyWords(a));
            Poly b = computeChromaticT(g.contractEdge(pu, pv), select, cache, st, rec,
                                       rec.spawnEdge(myBranchP, ViaKind::Contract, pu, pv), bidir);
            resP = a - b;
        }
        rec.end(myBranchP, st.branches - bStartP);
        cache.store(g, resP);
        return resP;
    }
    // ---- end SolverPolicy seam -----------------------------------------------------

    // Join reduction (complement-side dual of disconnected -> product): if Gbar is
    // disconnected, G is the join of the induced subgraphs on Gbar's components, and
    // chi(G) = umbral product (convolution in the falling-factorial basis) of the pieces.
    // Selector- and direction-independent; fires on BOTH paths before the cache/branch.
    // Each piece is a strictly smaller induced subgraph, so recursion terminates. Together
    // with the disconnected->product step above this realizes the full recursive
    // series-parallel (cograph) decomposition. (K_n is caught by the complete-graph base
    // case above, so this never fires on a complete graph.)
    {
        auto cc = complementComponents(g);
        if (cc.size() > 1 && reductionConfig().shouldFire(detail::RK_JOIN)) {
            rec.classify(myId, NodeKind::Join);
            Poly prod = Poly::one();                 // umbral identity (a-vector [1])
            for (auto& part : cc) {
                detail::LiveWordsScope hold(st, polyWords(prod));
                prod = umbralProduct(prod,
                         computeChromaticT(inducedSubgraph(g, part), select, cache,
                                           st, rec, rec.spawnPiece(myId, part), bidir));
            }
            return prod;
        }
    }

    // Clique-separator decomposition (CRT-1, any l >= 1): chi(G) = prod chi(piece_i) / chi(K_l)^{p-1}.
    // MCS-M-based detection; subsumes the former l=1 block step. Selector-independent and
    // not a branch. Each piece is strictly smaller and does not contain S as a separator,
    // so recursion terminates.
    // Default path: fast smallest-first search (unchanged production cost). Only when a
    // CliqueSeparatorSelector is installed do we pay the full candidate enumeration.
    std::vector<int> sep;
    if (CliqueSeparatorSelector& sel = cliqueSeparatorSelector()) {
        auto candidates = cliqueSeparatorCandidates(g);
        if (!candidates.empty()) {
            sep = sel(g, candidates);
            bool ok = false;
            for (auto& c : candidates) if (c == sep) { ok = true; break; }
            if (!ok) throw std::runtime_error("cliqueSeparatorSelector returned a non-candidate");
        }
    } else {
        sep = cliqueSeparator(g);
    }
    if (!sep.empty() && reductionConfig().shouldFire(detail::RK_CLIQUE)) {
        const int l = static_cast<int>(sep.size());
        auto pieces = cliqueSeparatorPieces(g, sep);
        rec.classifySep(myId, NodeKind::CliqueSep, sep, {});
        Poly prod = Poly::one();
        for (auto& verts : pieces) {
            detail::LiveWordsScope hold(st, polyWords(prod));
            prod = prod * computeChromaticT(inducedSubgraph(g, verts), select, cache,
                                            st, rec, rec.spawnPiece(myId, verts), bidir);
        }
        const Poly div = chromaticComplete(l);          // chi(K_l), monic
        for (int i = 1; i < static_cast<int>(pieces.size()); ++i)
            prod = prod.divExact(div);                  // divide by chi(K_l)^{p-1}
        return prod;
    }

    // Complement clique-separator decomposition (P3+ dual-CRT-1, l=1..L_max): if Gbar has a clique
    // separator S (independent set in G), decompose χ(G) via the verified umbral inclusion-exclusion
    // formula. Selector- and direction-independent; fires on BOTH paths before cache/branch. The join
    // reduction above already handled l=0 (Gbar disconnected), so Gbar is connected here. Pieces are
    // strictly smaller, so recursion terminates.
    {
        std::vector<int> cS;
        std::vector<std::vector<int>> cComps;
        if (complementSepEnabled() &&
            complementCliqueSeparatorPieces(g, cS, cComps)) {
            const int cl = static_cast<int>(cS.size());
            const int cp = static_cast<int>(cComps.size());
            const double tw = touchardTerms(cl, cp);
            bool fire;
            if (ComplementSepDecider& dec = complementSepDecider()) {
                const double denom = static_cast<double>(n) * (n - 1) / 2.0;
                ComplementSepContext ctx{
                    g, cl, cp, tw, static_cast<int>(n), static_cast<int>(m),
                    denom > 0 ? (denom - static_cast<double>(m)) / denom : 0.0};
                fire = dec(ctx);
            } else {
                fire = reductionConfig().shouldFire(detail::RK_COMPLEMENT);
            }
            if (fire) {
                rec.classifySep(myId, NodeKind::ComplementSep, cS, cComps);
                st.complement_enum_terms += std::llround(tw);
                auto chi = [&](const Graph& h) -> Poly {
                    return computeChromaticT(h, select, cache, st, rec,
                                             rec.spawnIE(myId), bidir);
                };
                return complementCliqueDecompose(g, cS, cComps, chi);
            }
        }
    }

    if (const Poly* hit = cache.lookup(g)) { rec.classify(myId, NodeKind::CacheHit); return *hit; }  // isomorphic subgraph already solved

    const long long bStart = st.branches;
    ++st.branches;
    if (st.budget >= 0 && st.branches > st.budget) throw BudgetExceeded();
    auto [u, v] = select(g);                       // policy decision
    const int myBranch = rec.branch(myId, g, {u, v});

    Poly res;
    if (bidir && !g.hasEdge(u, v)) {
        // UP-branch on a non-edge f = uv: χ(G) = χ(G+f) + χ(G/f). Legal only on the dense
        // side (Φ(G) = C(n,2) - m), where both children strictly decrease Φ.
        const long long maxEdges = (long long)n * (n - 1) / 2;
        assert(2 * m >= maxEdges && "Phi-guard: up-branch must be on the dense side");
        Poly a = computeChromaticT(g.insertEdge(u, v), select, cache, st, rec,
                                   rec.spawnEdge(myBranch, ViaKind::Insert, u, v), bidir);
        {
            detail::LiveWordsScope hold(st, polyWords(a));
            Poly b = computeChromaticT(g.contractEdge(u, v), select, cache, st, rec,
                                       rec.spawnEdge(myBranch, ViaKind::Contract, u, v), bidir);
            res = a + b;
        }
    } else {
        // DOWN-branch on edge e = uv: χ(G) = χ(G-e) - χ(G/e). The default recurrence; in
        // bidir mode it is legal only on the sparse side (Φ(G) = m).
        if (bidir) {
            const long long maxEdges = (long long)n * (n - 1) / 2;
            assert(2 * m <= maxEdges && "Phi-guard: down-branch must be on the sparse side");
        }
        Poly a = computeChromaticT(g.deleteEdge(u, v), select, cache, st, rec,
                                   rec.spawnEdge(myBranch, ViaKind::Delete, u, v), bidir);
        {
            detail::LiveWordsScope hold(st, polyWords(a));
            Poly b = computeChromaticT(g.contractEdge(u, v), select, cache, st, rec,
                                       rec.spawnEdge(myBranch, ViaKind::Contract, u, v), bidir);
            res = a - b;
        }
    }
    rec.end(myBranch, st.branches - bStart);
    cache.store(g, res);
    return res;
}

// Behaviour-preserving overload for callers that don't record (NullRecorder, zero cost).
template <class Cache>
Poly computeChromaticT(const Graph& g, const EdgeSelector& select, Cache& cache, Stats& st,
                       bool bidir = false) {
    NullRecorder rec;
    return computeChromaticT(g, select, cache, st, rec, -1, bidir);
}

// Convenience overloads ---------------------------------------------------------------

// Cache-free (original behaviour); used by the validation suite.
inline Poly computeChromatic(const Graph& g, const EdgeSelector& select) {
    NullCache cache; Stats st;
    return computeChromaticT(g, select, cache, st);
}

// Cache-free, reporting stats.
inline Poly computeChromatic(const Graph& g, const EdgeSelector& select, Stats& st) {
    NullCache cache;
    return computeChromaticT(g, select, cache, st);
}

// With an explicit cache (e.g. NautyCache) and stats.
template <class Cache>
Poly computeChromatic(const Graph& g, const EdgeSelector& select, Cache& cache, Stats& st) {
    return computeChromaticT(g, select, cache, st);
}

// ---- bidirectional (opt-in) convenience overloads ----------------------------------
// Same engine, bidir=true: the selector's returned pair is dispatched on adjacency
// (adjacent -> down, non-adjacent -> up), guarded by Φ. Pair with a direction-aware
// selector such as selectBidirectionalStructural; an existing edge-only selector would trip
// the Φ-guard on a dense graph, which is exactly what the assertion is there to catch.

inline Poly computeChromaticBidir(const Graph& g, const EdgeSelector& select) {
    NullCache cache; Stats st;
    return computeChromaticT(g, select, cache, st, /*bidir=*/true);
}

inline Poly computeChromaticBidir(const Graph& g, const EdgeSelector& select, Stats& st) {
    NullCache cache;
    return computeChromaticT(g, select, cache, st, /*bidir=*/true);
}

template <class Cache>
Poly computeChromaticBidir(const Graph& g, const EdgeSelector& select, Cache& cache, Stats& st) {
    return computeChromaticT(g, select, cache, st, /*bidir=*/true);
}

// ---- baseline heuristics (Phase 0.6) ------------------------------------------------

// Pick the edge whose endpoints have the smallest combined degree.
inline std::pair<int,int> selectMinDegreeSum(const Graph& g) {
    std::pair<int,int> best{-1,-1};
    int bestScore = INT32_MAX;
    for (auto& e : g.edges()) {
        int s = g.degree(e.first) + g.degree(e.second);
        if (s < bestScore) { bestScore = s; best = e; }
    }
    return best;
}

// Uniform-random edge. NOT a heuristic; the control that calibrates how much of a
// selector's advantage over the incumbent is "structure" rather than "any rule at all"
// Carries its own seeded RNG so a given seed is reproducible; the RNG is
// advanced per call, so branch counts are a function of (seed, instance) only.
inline EdgeSelector makeRandomSelector(uint64_t seed) {
    auto rng = std::make_shared<std::mt19937_64>(seed);
    return [rng](const Graph& g) -> std::pair<int,int> {
        const std::vector<std::pair<int,int>> es = g.edges();   // edges() returns by value
        if (es.empty()) return {-1, -1};
        std::uniform_int_distribution<size_t> pick(0, es.size() - 1);
        return es[pick(*rng)];
    };
}

// Pick the edge whose endpoints have the largest combined degree.
inline std::pair<int,int> selectMaxDegreeSum(const Graph& g) {
    std::pair<int,int> best{-1,-1};
    int bestScore = -1;
    for (auto& e : g.edges()) {
        int s = g.degree(e.first) + g.degree(e.second);
        if (s > bestScore) { bestScore = s; best = e; }
    }
    return best;
}

// Structural bidirectional direction rule (P1, no learning). Auto-routes by density to
// the side where both children strictly decrease Φ = min(m, C(n,2)-m):
//   * sparse side (2m <= C(n,2)): branch DOWN on the min-degree-sum edge.
//   * dense  side (2m >  C(n,2)): branch UP   on the min-degree-sum NON-adjacent pair.
// The returned pair is always Φ-legal for the engine's bidir dispatch (it is an edge iff
// the sparse-side test holds). Reached only at a real branch node (post-reductions), so on
// the dense side G is not complete and a non-edge is guaranteed to exist.
inline std::pair<int,int> selectBidirectionalStructural(const Graph& g) {
    const int n = g.numVertices();
    const long long m = g.numEdges();
    const long long maxEdges = (long long)n * (n - 1) / 2;
    if (2 * m <= maxEdges) return selectMinDegreeSum(g);   // sparse/tie -> down on an edge

    std::pair<int,int> best{-1,-1};                        // dense -> up on a non-edge
    int bestScore = INT32_MAX;
    for (int u = 0; u < n; ++u)
        for (int v = u + 1; v < n; ++v)
            if (!g.hasEdge(u, v)) {
                int s = g.degree(u) + g.degree(v);
                if (s < bestScore) { bestScore = s; best = {u, v}; }
            }
    return best;
}

} // namespace crl
