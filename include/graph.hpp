// graph.hpp — Phase 0.1: simple-graph representation for chromatic deletion-contraction.
//
// Design notes:
//  * Vertices are labelled 0..n-1 and kept contiguous. Contraction rebuilds a compact
//    graph with n-1 vertices.
//  * Adjacency is stored as sorted neighbour sets, so parallel edges collapse for free
//    (this is the chromatic-specific simplification: proper colourings only see
//    adjacency, so multi-edges are irrelevant). Loops never arise because we never add
//    an edge from a vertex to itself.
//  * Operations return NEW graphs (immutable style). This is chosen for correctness and
//    clarity in the recursion; it is deliberately not the fast path. A later increment
//    will swap in an in-place representation with undo / copy-on-write behind the same
//    interface. Correctness first, speed second (see build order).
#pragma once
#include <vector>
#include <set>
#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <utility>

namespace crl {

class Graph {
public:
    explicit Graph(int n = 0) : adj_(n) {}

    // Convenience constructor: Graph(n, {{u1,v1},{u2,v2},...})
    Graph(int n, std::initializer_list<std::pair<int,int>> edges) : adj_(n) {
        for (auto& e : edges) addEdge(e.first, e.second);
    }

    int numVertices() const { return static_cast<int>(adj_.size()); }

    long long numEdges() const {
        long long deg = 0;
        for (const auto& s : adj_) deg += static_cast<long long>(s.size());
        return deg / 2;
    }

    const std::set<int>& neighbours(int v) const { return adj_[v]; }
    int degree(int v) const { return static_cast<int>(adj_[v].size()); }

    void addEdge(int u, int v) {
        if (u == v) return;            // never create loops
        adj_[u].insert(v);             // set semantics drop parallels
        adj_[v].insert(u);
    }

    bool hasEdge(int u, int v) const { return adj_[u].count(v) > 0; }

    // All edges as (u, v) with u < v.
    std::vector<std::pair<int,int>> edges() const {
        std::vector<std::pair<int,int>> es;
        for (int u = 0; u < numVertices(); ++u)
            for (int v : adj_[u])
                if (u < v) es.emplace_back(u, v);
        return es;
    }

    // G - e : delete the edge, vertex count unchanged.
    Graph deleteEdge(int u, int v) const {
        Graph g = *this;
        g.adj_[u].erase(v);
        g.adj_[v].erase(u);
        return g;
    }

    // G + f : add the (possibly absent) edge, vertex count unchanged. The dual of
    // deleteEdge — the up-direction primitive. Idempotent if the edge already exists
    // (set semantics); a u==v request is a no-op (no loops). Immutable style.
    Graph insertEdge(int u, int v) const {
        Graph g = *this;
        g.addEdge(u, v);
        return g;
    }

    // G / e : contract the edge. v is merged into u, labels compacted to 0..n-2.
    // Parallel edges and the contracted edge itself disappear via set semantics.
    Graph contractEdge(int a, int b) const {
        int u = std::min(a, b), v = std::max(a, b);   // keep u, drop v (u < v)
        const int n = numVertices();
        Graph g(n - 1);
        auto remap = [v](int x) { return x < v ? x : x - 1; };  // collapse label v
        for (int x = 0; x < n; ++x) {
            if (x == v) continue;
            int rx = remap(x);
            for (int y : adj_[x]) {
                if (y == x) continue;
                int target = (y == v) ? u : y;   // v's edges redirect onto u
                if (target == x) continue;       // would be a self-loop -> skip
                g.addEdge(rx, remap(target));
            }
        }
        return g;
    }

    bool operator==(const Graph& o) const { return adj_ == o.adj_; }

private:
    std::vector<std::set<int>> adj_;
};

} // namespace crl
