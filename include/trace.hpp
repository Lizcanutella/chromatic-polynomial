// trace.hpp — decision-trace recording for the deletion-contraction engine.
//
// Recorder concept (threaded through computeChromaticT like a Cache):
//   NullRecorder      — zero-overhead default; every method inlines to nothing.
//   TraceRecorder     — legacy branch-only trace (one record per BRANCH node).
//                       enter/spawn* are pass-throughs of the nearest branch-ancestor
//                       id, so its output is byte-identical to the historical begin/end
//                       recorder. RL training consumes this format; do not change it.
//   FullTraceRecorder — one record per engine node (branches, base cases, reductions,
//                       cache hits), each carrying the `via` operation that derived its
//                       graph from its parent's. Vertex identity and edge lists are
//                       REPLAYED downstream (python/chromatic_rl/viz/replay.py) from
//                       via + the root graph; only synthetic ie_piece graphs (built by
//                       complementCliqueDecompose's pieceGraph) store explicit edges.
//
// subtree_branches is cache-aware: a child that hits the cache contributes ~0, so the
// recorded cost is the real cached cost — path-dependent by design.
#pragma once
#include "graph.hpp"
#include <cstdint>
#include <string>
#include <vector>
#include <utility>

namespace crl {

enum class NodeKind : std::uint8_t { Unset, Branch, BaseEmpty, BaseTree, BaseComplete,
    BaseCycle, BaseWheel, Disconnected, Join, CliqueSep, ComplementSep, CacheHit };
enum class ViaKind : std::uint8_t { Root, Delete, Contract, Insert, Piece, IEPiece };

struct TraceRecord {
    int id = -1;
    int parent_id = -1;
    int n = 0;
    std::vector<std::pair<int,int>> edges;   // candidate action space at this node
    std::pair<int,int> chosen{-1,-1};
    long long subtree_branches = 0;          // branch-nodes in this subtree, incl. self
};

struct FullTraceRecord {
    int id = -1;
    int parent_id = -1;
    ViaKind via = ViaKind::Root;
    int via_u = -1, via_v = -1;              // Delete / Contract / Insert
    std::vector<int> via_verts;              // Piece: parent-label vertex list
    NodeKind kind = NodeKind::Unset;
    int n = 0;
    long long m = 0;
    std::vector<std::pair<int,int>> edges;   // ONLY for via == IEPiece (not replayable)
    std::pair<int,int> chosen{-1,-1};        // Branch only
    long long subtree_branches = 0;          // Branch only (cache-aware, as before)
    std::vector<int> sep;                    // CliqueSep / ComplementSep separator
    std::vector<std::vector<int>> pieces;    // ComplementSep components (g-labels)
    std::string action_kind;                 // SolverPolicy choice; empty when no policy installed
    long long subtree_nodes = 0;             // 1 + sum over children; the tree-MDP return for nodes
};

// Zero-overhead default: methods inline away to nothing.
struct NullRecorder {
    int  enter(int, const Graph&) { return -1; }
    void classify(int, NodeKind) {}
    void classifySep(int, NodeKind, const std::vector<int>&,
                     const std::vector<std::vector<int>>&) {}
    int  spawnEdge(int, ViaKind, int, int) { return -1; }
    int  spawnPiece(int, const std::vector<int>&) { return -1; }
    int  spawnIE(int) { return -1; }
    int  branch(int, const Graph&, std::pair<int,int>) { return -1; }
    void end(int, long long) {}
    void recordAction(int, const char*) {}
    void endNodes(int, long long) {}
};

// Legacy branch-only recorder. enter/spawn* pass the nearest branch-ancestor id
// through unchanged; branch() materialises a record exactly like the old begin().
struct TraceRecorder {
    std::vector<TraceRecord> records;

    int  enter(int recCtx, const Graph&) { return recCtx; }
    void classify(int, NodeKind) {}
    void classifySep(int, NodeKind, const std::vector<int>&,
                     const std::vector<std::vector<int>>&) {}
    int  spawnEdge(int parent, ViaKind, int, int) { return parent; }
    int  spawnPiece(int parent, const std::vector<int>&) { return parent; }
    int  spawnIE(int parent) { return parent; }
    int  branch(int recCtx, const Graph& g, std::pair<int,int> chosen) {
        int id = static_cast<int>(records.size());
        TraceRecord r;
        r.id = id;
        r.parent_id = recCtx;
        r.n = g.numVertices();
        r.edges = g.edges();
        r.chosen = chosen;
        records.push_back(std::move(r));
        return id;
    }
    void end(int id, long long subtree) { records[id].subtree_branches = subtree; }
    // The legacy branch-only trace stays byte-identical: RL consumers depend on its shape,
    // so the new SolverPolicy instrumentation is deliberately a no-op here.
    void recordAction(int, const char*) {}
    void endNodes(int, long long) {}
};

// Enriched recorder: one record per engine node. Parents spawn children's records
// (assigning DFS-ordered ids) right before recursing; enter() fills the node's
// n/m (and edges for synthetic ie_piece graphs).
struct FullTraceRecorder {
    std::vector<FullTraceRecord> records;

    int make(int parent, ViaKind k) {
        int id = static_cast<int>(records.size());
        FullTraceRecord r;
        r.id = id;
        r.parent_id = parent;
        r.via = k;
        records.push_back(std::move(r));
        return id;
    }
    int enter(int recCtx, const Graph& g) {
        if (recCtx == -1) recCtx = make(-1, ViaKind::Root);
        records[recCtx].n = g.numVertices();
        records[recCtx].m = g.numEdges();
        if (records[recCtx].via == ViaKind::IEPiece) records[recCtx].edges = g.edges();
        return recCtx;
    }
    void classify(int id, NodeKind k) { records[id].kind = k; }
    void classifySep(int id, NodeKind k, const std::vector<int>& S,
                     const std::vector<std::vector<int>>& pieces) {
        records[id].kind = k;
        records[id].sep = S;
        records[id].pieces = pieces;
    }
    int spawnEdge(int parent, ViaKind k, int u, int v) {
        int id = make(parent, k);
        records[id].via_u = u;
        records[id].via_v = v;
        return id;
    }
    int spawnPiece(int parent, const std::vector<int>& verts) {
        int id = make(parent, ViaKind::Piece);
        records[id].via_verts = verts;
        return id;
    }
    int spawnIE(int parent) { return make(parent, ViaKind::IEPiece); }
    int branch(int id, const Graph&, std::pair<int,int> chosen) {
        records[id].kind = NodeKind::Branch;
        records[id].chosen = chosen;
        return id;
    }
    void end(int id, long long subtree) { records[id].subtree_branches = subtree; }
    void recordAction(int id, const char* kind) { records[id].action_kind = kind; }
    void endNodes(int id, long long nodes) { records[id].subtree_nodes = nodes; }
};

} // namespace crl
