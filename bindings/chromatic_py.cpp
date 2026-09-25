// pybind11 module exposing the chromatic engine to Python.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <memory>
#include <vector>
#include <utility>

#include "graph.hpp"
#include "polynomial.hpp"
#include "chromatic.hpp"
#include "nauty_cache.hpp"
#include "trace.hpp"
#include <gmpxx.h>
#include <string>

namespace py = pybind11;
using crl::Graph;
using crl::Poly;
using crl::Stats;
using crl::NullCache;
using crl::NautyCache;
using crl::NullRecorder;
using crl::TraceRecorder;
using crl::FullTraceRecorder;
using crl::EdgeSelector;

// crl::NodeKind, crl::ViaKind -> Python-facing string names (used by FullTraceRecord).
static const char* kindName(crl::NodeKind k) {
    switch (k) {
        case crl::NodeKind::Branch:        return "branch";
        case crl::NodeKind::BaseEmpty:     return "base_empty";
        case crl::NodeKind::BaseTree:      return "base_tree";
        case crl::NodeKind::BaseComplete:  return "base_complete";
        case crl::NodeKind::BaseCycle:     return "base_cycle";
        case crl::NodeKind::BaseWheel:     return "base_wheel";
        case crl::NodeKind::Disconnected:  return "disconnected";
        case crl::NodeKind::Join:          return "join";
        case crl::NodeKind::CliqueSep:     return "clique_sep";
        case crl::NodeKind::ComplementSep: return "complement_sep";
        case crl::NodeKind::CacheHit:      return "cache_hit";
        default:                           return "unset";
    }
}
static const char* viaName(crl::ViaKind k) {
    switch (k) {
        case crl::ViaKind::Delete:   return "delete";
        case crl::ViaKind::Contract: return "contract";
        case crl::ViaKind::Insert:   return "insert";
        case crl::ViaKind::Piece:    return "piece";
        case crl::ViaKind::IEPiece:  return "ie_piece";
        default:                     return "root";
    }
}

// Exact GMP integer -> arbitrary-precision Python int (via decimal string).
static py::int_ mpz_to_pyint(const mpz_class& z) {
    return py::reinterpret_steal<py::int_>(
        PyLong_FromString(const_cast<char*>(z.get_str().c_str()), nullptr, 10));
}

// Aggregated, Python-facing run statistics.
struct RunStats {
    long long nodes = 0, branches = 0, hits = 0, misses = 0;
    double hit_rate = 0.0;
    long long peak_live_words = 0, cache_words = 0;
    long long complement_enum_terms = 0;
};

struct Result {
    Poly poly;
    RunStats stats;
    py::object trace;   // py::none() unless record=True (legacy TraceRecord list) or record="full" (FullTraceRecord list)
    bool truncated = false;   // the node budget bit; `poly` is None and makes no claim
};

// Resolve the EdgeSelector from a Python argument: None / "min_degree_sum" /
// "max_degree_sum" use the fast C++ heuristics, or pass any Python callable
// (Graph) -> (u, v) , this is how a custom edge-selection policy plugs in.
static EdgeSelector resolve_selector(const py::object& selector) {
    if (selector.is_none()) return crl::selectMinDegreeSum;
    if (py::isinstance<py::str>(selector)) {
        std::string s = selector.cast<std::string>();
        if (s == "min_degree_sum") return crl::selectMinDegreeSum;
        if (s == "max_degree_sum") return crl::selectMaxDegreeSum;
        throw py::value_error("unknown built-in selector: " + s);
    }
    if (py::isinstance<py::function>(selector) || py::hasattr(selector, "__call__")) {
        // Wrap the Python callable. The GIL is held throughout compute() (single
        // threaded), so we call it directly. The graph is passed by reference (no copy).
        py::object fn = selector;
        return [fn](const Graph& g) -> std::pair<int,int> {
            py::object r = fn(py::cast(&g, py::return_value_policy::reference));
            std::pair<int,int> e = r.cast<std::pair<int,int>>();
            const int n = g.numVertices();
            const bool in_range = e.first >= 0 && e.first < n &&
                                  e.second >= 0 && e.second < n;
            if (!in_range || !g.hasEdge(e.first, e.second))
                throw py::value_error(
                    "policy selector returned a non-edge (" +
                    std::to_string(e.first) + "," + std::to_string(e.second) + ")");
            return e;
        };
    }
    throw py::type_error("selector must be None, a built-in selector name, or a callable");
}

// Set reductionConfig() from a Python dict, restoring defaults on scope exit so the
// global never leaks between calls. Keys: "join" / "clique_sep" / "complement_sep".
struct ReductionConfigGuard {
    ReductionConfigGuard(const py::object& spec, unsigned long long seed) {
        auto& cfg = crl::reductionConfig();
        cfg = crl::detail::ReductionConfig{};            // defaults: all on, p=1
        cfg.reseed(seed);
        if (!spec.is_none()) {
            py::dict d = spec.cast<py::dict>();
            auto set = [&](const char* key, int idx) {
                if (d.contains(key)) {
                    py::dict e = d[key].cast<py::dict>();
                    if (e.contains("enabled")) cfg.enabled[idx] = e["enabled"].cast<bool>();
                    if (e.contains("p"))       cfg.defer_p[idx] = e["p"].cast<double>();
                }
            };
            set("join", crl::detail::RK_JOIN);
            set("clique_sep", crl::detail::RK_CLIQUE);
            set("complement_sep", crl::detail::RK_COMPLEMENT);
        }
    }
    ~ReductionConfigGuard() { crl::reductionConfig() = crl::detail::ReductionConfig{}; }
};

static Result run_compute(const Graph& g, const py::object& selector,
                          bool use_cache, int record_mode, long long branch_budget,
                          long long node_budget,
                          const py::object& reductions, unsigned long long defer_seed) {
    ReductionConfigGuard _guard(reductions, defer_seed);
    EdgeSelector sel = resolve_selector(selector);
    Result out;
    Stats st;
    st.budget = branch_budget;
    st.node_budget = node_budget;

    // The recorders are hoisted out of the branch so their contents survive a
    // BudgetExceeded unwind: every span is already closed on the way out by
    // NodeSpanScope's destructor, so the partial trace is complete and usable.
    FullTraceRecorder fr;
    TraceRecorder tr;
    try {
        if (record_mode == 2) {
            if (use_cache) {
                NautyCache cache;
                out.poly = crl::computeChromaticT(g, sel, cache, st, fr, -1);
                out.stats.hits = cache.hits;
                out.stats.misses = cache.misses;
                out.stats.hit_rate = cache.hitRate();
                out.stats.cache_words = cache.words;
            } else {
                NullCache cache;
                out.poly = crl::computeChromaticT(g, sel, cache, st, fr, -1);
            }
        } else if (record_mode == 1) {
            if (use_cache) {
                NautyCache cache;
                out.poly = crl::computeChromaticT(g, sel, cache, st, tr, -1);
                out.stats.hits = cache.hits;
                out.stats.misses = cache.misses;
                out.stats.hit_rate = cache.hitRate();
                out.stats.cache_words = cache.words;
            } else {
                NullCache cache;
                out.poly = crl::computeChromaticT(g, sel, cache, st, tr, -1);
            }
        } else {
            if (use_cache) {
                NautyCache cache;
                out.poly = crl::computeChromatic(g, sel, cache, st);
                out.stats.hits = cache.hits;
                out.stats.misses = cache.misses;
                out.stats.hit_rate = cache.hitRate();
                out.stats.cache_words = cache.words;
            } else {
                NullCache cache;
                out.poly = crl::computeChromatic(g, sel, cache, st);
            }
        }
    } catch (const crl::BudgetExceeded&) {
        if (node_budget < 0) throw;
        out.truncated = true;
    }

    if (record_mode == 2) {
        py::list records;
        for (auto& r : fr.records) records.append(r);
        out.trace = records;
    } else if (record_mode == 1) {
        py::list records;
        for (auto& r : tr.records) records.append(r);
        out.trace = records;
    } else {
        out.trace = py::none();
    }
    out.stats.nodes = st.nodes;
    out.stats.branches = st.branches;
    out.stats.peak_live_words = st.peak_live_words;
    out.stats.complement_enum_terms = st.complement_enum_terms;
    return out;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "chromatic_rl native core (deletion-contraction engine bindings)";
    m.attr("__version__") = "0.1.0";

    py::class_<Graph>(m, "Graph")
        .def(py::init([](int n, const std::vector<std::pair<int,int>>& edges) {
                 auto g = std::make_unique<Graph>(n);
                 for (const auto& e : edges) g->addEdge(e.first, e.second);
                 return g;
             }),
             py::arg("n"), py::arg("edges"))
        .def("num_vertices", &Graph::numVertices)
        .def("num_edges", [](const Graph& g) { return g.numEdges(); })
        .def("degree", &Graph::degree, py::arg("v"))
        .def("neighbours", [](const Graph& g, int v) {
                 const auto& s = g.neighbours(v);
                 return std::vector<int>(s.begin(), s.end());
             }, py::arg("v"))
        .def("edges", &Graph::edges)
        .def("delete_edge", &Graph::deleteEdge, py::arg("u"), py::arg("v"))
        .def("contract_edge", &Graph::contractEdge, py::arg("u"), py::arg("v"))
        .def("to_arrays", [](const Graph& g) {
                 auto es = g.edges();
                 const py::ssize_t mm = static_cast<py::ssize_t>(es.size());
                 py::array_t<int64_t> arr({py::ssize_t(2), mm});
                 auto r = arr.mutable_unchecked<2>();
                 for (py::ssize_t i = 0; i < mm; ++i) {
                     r(0, i) = es[i].first;
                     r(1, i) = es[i].second;
                 }
                 return py::make_tuple(g.numVertices(), arr);
             });

    py::class_<Poly>(m, "Poly")
        .def("coeffs", [](const Poly& p) {
                 py::list out;
                 for (const auto& z : p.coeffs()) out.append(mpz_to_pyint(z));
                 return out;
             })
        .def("degree", &Poly::degree)
        .def("evaluate", [](const Poly& p, long k) { return mpz_to_pyint(p.eval(k)); },
             py::arg("k"))
        .def("__str__", &Poly::str);

    py::class_<RunStats>(m, "Stats")
        .def_readonly("nodes", &RunStats::nodes)
        .def_readonly("branches", &RunStats::branches)
        .def_readonly("hits", &RunStats::hits)
        .def_readonly("misses", &RunStats::misses)
        .def_readonly("hit_rate", &RunStats::hit_rate)
        .def_readonly("peak_live_words", &RunStats::peak_live_words)
        .def_readonly("cache_words", &RunStats::cache_words)
        .def_readonly("complement_enum_terms", &RunStats::complement_enum_terms);

    py::class_<Result>(m, "Result")
        // None when the node budget bit: a truncated run makes no polynomial claim,
        // which is what keeps "any policy yields the identical polynomial" absolute
        // rather than weakened into "unless it ran out of budget".
        .def_property_readonly("poly", [](const Result& r) -> py::object {
            if (r.truncated) return py::none();
            return py::cast(r.poly);
        })
        .def_readonly("stats", &Result::stats)
        .def_readonly("trace", &Result::trace)
        .def_readonly("truncated", &Result::truncated);

    py::class_<crl::TraceRecord>(m, "TraceRecord")
        .def_readonly("id", &crl::TraceRecord::id)
        .def_readonly("parent_id", &crl::TraceRecord::parent_id)
        .def_readonly("n", &crl::TraceRecord::n)
        .def_readonly("edges", &crl::TraceRecord::edges)
        .def_readonly("chosen", &crl::TraceRecord::chosen)
        .def_readonly("subtree_branches", &crl::TraceRecord::subtree_branches);

    py::class_<crl::FullTraceRecord>(m, "FullTraceRecord")
        .def_readonly("id", &crl::FullTraceRecord::id)
        .def_readonly("parent_id", &crl::FullTraceRecord::parent_id)
        .def_property_readonly("via",
            [](const crl::FullTraceRecord& r) { return viaName(r.via); })
        .def_readonly("via_u", &crl::FullTraceRecord::via_u)
        .def_readonly("via_v", &crl::FullTraceRecord::via_v)
        .def_readonly("via_verts", &crl::FullTraceRecord::via_verts)
        .def_property_readonly("kind",
            [](const crl::FullTraceRecord& r) { return kindName(r.kind); })
        .def_readonly("n", &crl::FullTraceRecord::n)
        .def_readonly("m", &crl::FullTraceRecord::m)
        .def_readonly("edges", &crl::FullTraceRecord::edges)
        .def_readonly("chosen", &crl::FullTraceRecord::chosen)
        .def_readonly("subtree_branches", &crl::FullTraceRecord::subtree_branches)
        .def_readonly("sep", &crl::FullTraceRecord::sep)
        .def_readonly("pieces", &crl::FullTraceRecord::pieces);

    // Uniform-random edge selector at a fixed seed: a simple control baseline that
    // calibrates how much of a policy's advantage is structure rather than "any rule
    // at all". Deterministic given (seed, instance); run over several seeds for a
    // distribution.
    m.def("random_selector", [](uint64_t seed) {
        auto sel = crl::makeRandomSelector(seed);
        return py::cpp_function([sel](const Graph& g) { return sel(g); });
    }, py::arg("seed") = 0);

    // Smallest clique minimal separator (original labels), empty if g is an atom.
    // Its length is the separator size l. Wraps the verified MCS-M routine.
    m.def("clique_separator", [](const Graph& g) {
        return crl::cliqueSeparator(g);
    }, py::arg("graph"));

    // All distinct valid MCS-M-surfaced clique separators (original labels).
    // Empty if g is an atom.
    m.def("clique_separator_candidates", [](const Graph& g) {
        return crl::cliqueSeparatorCandidates(g);
    }, py::arg("graph"));

    // Pieces G[C_i ∪ S] for a given clique separator S (one per component of G-S), each with
    // the shared clique glued on. Returns fresh crl.Graph objects (relabelled 0..k-1).
    m.def("clique_separator_pieces", [](const Graph& g, const std::vector<int>& S) {
        std::vector<Graph> out;
        for (auto& verts : crl::cliqueSeparatorPieces(g, S))
            out.push_back(crl::inducedSubgraph(g, verts));
        return out;
    }, py::arg("graph"), py::arg("S"));

    // Expose a built-in C++ selector as a standalone Python callable
    // (Graph) -> (u, v).
    m.def("builtin_selector", [](const std::string& name) {
        EdgeSelector sel;
        if (name == "min_degree_sum")      sel = crl::selectMinDegreeSum;
        else if (name == "max_degree_sum") sel = crl::selectMaxDegreeSum;
        else throw py::value_error("unknown built-in selector: " + name);
        return py::cpp_function([sel](const Graph& g) { return sel(g); });
    }, py::arg("name"));

    py::register_exception<crl::BudgetExceeded>(m, "BudgetExceeded");

    m.def("compute",
          [](const Graph& g, const py::object& selector, bool use_cache,
             const py::object& record, const py::object& branch_budget,
             const py::object& node_budget,
             const py::object& reductions, unsigned long long defer_seed) {
              long long b = branch_budget.is_none()
                                ? -1
                                : branch_budget.cast<long long>();
              int record_mode = 0;
              if (py::isinstance<py::str>(record)) {
                  if (record.cast<std::string>() != "full")
                      throw py::value_error("record must be False, True, or 'full'");
                  record_mode = 2;
              } else if (record.cast<bool>()) {
                  record_mode = 1;
              }
              long long nb = node_budget.is_none()
                                 ? -1
                                 : node_budget.cast<long long>();
              return run_compute(g, selector, use_cache, record_mode, b, nb, reductions,
                                 defer_seed);
          },
          py::arg("graph"), py::arg("selector") = py::none(),
          py::arg("use_cache") = true, py::arg("record") = false,
          py::arg("branch_budget") = py::none(),
          py::arg("node_budget") = py::none(),
          py::arg("reductions") = py::none(), py::arg("defer_seed") = 0);
}
