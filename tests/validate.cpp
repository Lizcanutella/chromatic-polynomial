// validate.cpp — Phase 0.7: correctness validation for the engine.
//
// Two independent checks:
//  (A) Closed forms: K_n, C_n, path/tree, empty graph against known formulae.
//  (B) Brute force: for random small graphs, count proper k-colourings directly for
//      k = 0..n and compare to the engine's polynomial evaluated at those points.
//      Two degree-<=n polynomials agreeing at n+1 distinct points are identical, so
//      this is a rigorous equality check, not a spot check.
//
// The engine result must also be invariant to the selector: if two heuristics disagree
// on the polynomial, that is a bug, not a performance difference. We assert this too.
#include "chromatic.hpp"
#include <algorithm>
#include <iostream>
#include <random>
#include <functional>
#include <set>

using namespace crl;

static int g_fail = 0;
static void check(bool ok, const std::string& name) {
    std::cout << (ok ? "  [PASS] " : "  [FAIL] ") << name << "\n";
    if (!ok) ++g_fail;
}

// Independent brute-force count of proper k-colourings.
static mpz_class bruteCount(const Graph& g, int k) {
    int n = g.numVertices();
    std::vector<int> col(n, -1);
    std::function<mpz_class(int)> rec = [&](int v) -> mpz_class {
        if (v == n) return 1;
        mpz_class total = 0;
        for (int c = 0; c < k; ++c) {
            bool ok = true;
            for (int w : g.neighbours(v))
                if (w < v && col[w] == c) { ok = false; break; }
            if (!ok) continue;
            col[v] = c;
            total += rec(v + 1);
            col[v] = -1;
        }
        return total;
    };
    return rec(0);
}

static Graph makePath(int n) { Graph g(n); for (int i=0;i+1<n;++i) g.addEdge(i,i+1); return g; }
static Graph makeCycle(int n){ Graph g(n); for (int i=0;i<n;++i) g.addEdge(i,(i+1)%n); return g; }
static Graph makeComplete(int n){ Graph g(n); for(int i=0;i<n;++i)for(int j=i+1;j<n;++j) g.addEdge(i,j); return g; }
static Graph makeEmpty(int n){ return Graph(n); }

// K_n minus a perfect matching (n even): the cocktail-party graph, a DENSE instance
// (2m > C(n,2)) that forces the bidirectional engine onto the up-direction.
static Graph makeKnMinusMatching(int n) {
    Graph g = makeComplete(n);
    for (int i = 0; i + 1 < n; i += 2) g = g.deleteEdge(i, i + 1);
    return g;
}

// Complete tripartite K_{1,2,3}: parts {0} | {1,2} | {3,4,5}, all cross edges present.
// A complete-multipartite graph: its complement is K_1 + K_2 + K_3 (disconnected), so the
// join reduction fully decomposes it.
static Graph makeK123() {
    Graph g(6);
    std::vector<std::vector<int>> parts = {{0}, {1, 2}, {3, 4, 5}};
    for (std::size_t a = 0; a < parts.size(); ++a)
        for (std::size_t b = a + 1; b < parts.size(); ++b)
            for (int u : parts[a]) for (int v : parts[b]) g.addEdge(u, v);
    return g;
}

// Wheel W_n: hub = vertex n-1, rim cycle on 0..n-2, hub joined to all rim vertices.
static Graph makeWheel(int n){
    Graph g(n); int r = n - 1, h = n - 1;
    for (int i = 0; i < r; ++i) { g.addEdge(i, (i + 1) % r); g.addEdge(h, i); }
    return g;
}

// Lollipop: K_4 on {0,1,2,3} plus a triangle {0,4,5} sharing the cut vertex 0.
static Graph makeLollipop(){
    Graph g(6);
    for (int i = 0; i < 4; ++i) for (int j = i+1; j < 4; ++j) g.addEdge(i, j);
    g.addEdge(0,4); g.addEdge(4,5); g.addEdge(5,0);
    return g;
}

// Bowtie: two triangles {0,1,2} and {0,3,4} sharing the cut vertex 0.
static Graph makeBowtie(){
    Graph g(5);
    g.addEdge(0,1); g.addEdge(1,2); g.addEdge(2,0);
    g.addEdge(0,3); g.addEdge(3,4); g.addEdge(4,0);
    return g;
}

// Triple bowtie: three triangles {0,1,2}, {0,3,4}, {0,5,6} all sharing cut vertex 0.
// 3 blocks, ONE cut vertex (shared by all three) -> divisor exponent r-1 = 2, not 1.
static Graph makeTripleBowtie(){
    Graph g(7);
    g.addEdge(0,1); g.addEdge(1,2); g.addEdge(2,0);
    g.addEdge(0,3); g.addEdge(3,4); g.addEdge(4,0);
    g.addEdge(0,5); g.addEdge(5,6); g.addEdge(6,0);
    return g;
}

// Diamond = K_4 minus edge {2,3}: two triangles sharing edge {0,1}. l=2 clique separator.
static Graph makeDiamond(){
    Graph g(4);
    g.addEdge(0,1); g.addEdge(0,2); g.addEdge(1,2);     // triangle {0,1,2}
    g.addEdge(0,3); g.addEdge(1,3);                     // triangle {0,1,3}
    return g;
}

// Two K_4s sharing triangle {0,1,2}; apex 3 and apex 4 (3,4 not adjacent). l=3 separator.
static Graph makeTwoK4SharingTriangle(){
    Graph g(5);
    g.addEdge(0,1); g.addEdge(0,2); g.addEdge(1,2);     // shared triangle
    g.addEdge(3,0); g.addEdge(3,1); g.addEdge(3,2);     // K_4 #1 apex
    g.addEdge(4,0); g.addEdge(4,1); g.addEdge(4,2);     // K_4 #2 apex
    return g;
}

// Book B_3 = three triangles sharing edge {0,1}; pages 2,3,4. l=2 separator with p=3.
static Graph makeBook3(){
    Graph g(5);
    g.addEdge(0,1);
    g.addEdge(0,2); g.addEdge(1,2);
    g.addEdge(0,3); g.addEdge(1,3);
    g.addEdge(0,4); g.addEdge(1,4);
    return g;
}

// K_{2,3}: parts {0,1} and {2,3,4}. 2-connected, triangle-free; its only separator {0,1}
// is NOT a clique, so it has no clique separator and the engine must branch on it.
static Graph makeK23(){
    Graph g(5);
    for (int a = 0; a < 2; ++a)
        for (int b = 2; b < 5; ++b) g.addEdge(a, b);
    return g;
}

// Two K_5s sharing the K_4 on {0,1,2,3}; apex 4 and apex 5 (4,5 not adjacent).
// l=4 clique separator {0,1,2,3}. chi = k(k-1)(k-2)(k-3)(k-4)^2.
static Graph makeTwoK5SharingK4(){
    Graph g(6);
    for (int a = 0; a < 4; ++a)                       // shared K_4
        for (int b = a + 1; b < 4; ++b) g.addEdge(a, b);
    for (int b = 0; b < 4; ++b) { g.addEdge(4, b); g.addEdge(5, b); }   // two apices
    return g;
}

static Graph makeErdosRenyi(int n, double p, std::mt19937& rng) {
    Graph g(n);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (u(rng) < p) g.addEdge(i, j);
    return g;
}

static void test_complement_setpartitions() {
    using namespace crl;
    // complementGraph: G has only edge (0,2); Gbar should be path 0-1-2
    Graph g(3); g.addEdge(0, 2);
    Graph gb = complementGraph(g);
    check(gb.hasEdge(0, 1) && gb.hasEdge(1, 2) && !gb.hasEdge(0, 2),
          "complementGraph: Gbar of K_3 minus one edge is a path");

    // setPartitions: Bell(2) = 2
    auto ps = setPartitions(std::vector<int>{7, 8});
    check(ps.size() == 2, "setPartitions({7,8}) has 2 partitions (Bell(2)=2)");

    // setPartitions: Bell(3) = 5
    auto p3 = setPartitions(std::vector<int>{1, 2, 3});
    check(p3.size() == 5, "setPartitions({1,2,3}) has 5 partitions (Bell(3)=5)");

    std::cout << "test_complement_setpartitions passed\n";
}

static void test_piece_graph() {
    using namespace crl;
    // g = complement(P3): edge (0,2); vertices {0,1,2}. Gbar = path 0-1-2, S={1} (cut vertex).
    // Take S={1}, tau = {{1}} (q=1), comp = {0}.  g-adjacency 1-0? g has only edge (0,2), so no.
    Graph g(3); g.addEdge(0,2);
    std::vector<std::vector<int>> tau{{1}};
    // allowed = {0}: block 0 may merge -> rep NOT force-joined to comp; natural adj 1~0 is false.
    Graph pa = pieceGraph(g, std::vector<int>{0}, tau, std::set<int>{0});
    check(pa.numVertices() == 2, "pieceGraph: allowed={0} has 2 vertices");
    check(!pa.hasEdge(0,1), "pieceGraph: allowed={0}, rep not adjacent to comp-vertex 0");
    // allowed = {} : block 0 forbidden -> rep force-joined to comp vertex 0.
    Graph pf = pieceGraph(g, std::vector<int>{0}, tau, std::set<int>{});
    check(pf.hasEdge(0,1), "pieceGraph: allowed={}, rep force-joined to comp vertex 0");
    std::cout << "test_piece_graph passed\n";
}

static std::vector<mpz_class> ff_of(const crl::Poly& p) { return p.toFallingFactorial(); }

static void test_complement_reduction_oracle() {
    using namespace crl;
    // n=6, 10 edges.  Verified properties (independent Python oracle):
    //   (a) g is connected                 -> disconnected-components reduction cannot preempt
    //   (b) complement(g) is connected     -> join reduction cannot preempt
    //   (c) g has NO clique separator      -> down-CRT-1 cannot preempt
    //   (d) complement(g) HAS a clique sep -> complementCliqueSeparatorPieces fires
    //   chi(g)[k=3] == 6; falling-factorial a-vector == [0,0,0,1,6,5,1].
    Graph g(6);
    g.addEdge(0,1); g.addEdge(0,3); g.addEdge(0,4); g.addEdge(0,5);
    g.addEdge(1,3); g.addEdge(1,4);
    g.addEdge(2,3); g.addEdge(2,4); g.addEdge(2,5);
    g.addEdge(3,5);

    // 1. Prove down-CRT-1 cannot preempt.
    check(cliqueSeparator(g).empty(),
          "complement clique-sep: g has no clique separator (down-CRT-1 cannot preempt)");

    // 2. Prove join reduction cannot preempt (complement must be connected).
    check(complementComponents(g).size() == 1,
          "complement clique-sep: complement connected (join cannot preempt)");

    // 3. Prove the new reduction fires: complement has a clique separator.
    std::vector<int> S;
    std::vector<std::vector<int>> comps;
    check(complementCliqueSeparatorPieces(g, S, comps),
          "complement clique-sep: complement clique-separator detector fires");

    // 4. Independent correctness gate: engine result must match the brute-force oracle.
    Stats st;
    Poly r = computeChromatic(g, selectMinDegreeSum, st);
    bool agree = true;
    for (int k = 0; k <= g.numVertices(); ++k)
        if (r.eval(k) != bruteCount(g, k)) { agree = false; break; }
    check(agree, "complement clique-sep: n=6 instance matches brute force");

    // Bonus: falling-factorial a-vector matches independently verified reference.
    check(ff_of(r) == (std::vector<mpz_class>{0,0,0,1,6,5,1}),
          "complement clique-sep: a-vector matches verified reference [0,0,0,1,6,5,1]");
}

static void test_complement_decompose() {
    using namespace crl;
    auto chi = [](const Graph& h){ return computeChromatic(h, selectMinDegreeSum); };

    // PIN l=1, p=2: g = complement(P3): edge (1,2). S={0}, comps={{1},{2}}. a=[0,0,2,1].
    { Graph g(3); g.addEdge(1,2);
      Poly r = complementCliqueDecompose(g, {0}, {{1},{2}}, chi);
      check(ff_of(r) == (std::vector<mpz_class>{0,0,2,1}),
            "complementCliqueDecompose: l=1 p=2 complement(P3) a=[0,0,2,1]"); }

    // PIN l=2, p=2: n=4, E={(1,2),(1,3),(2,3)}. S={0,1}, comps={{2},{3}}. a=[0,0,0,3,1].
    { Graph g(4); g.addEdge(1,2); g.addEdge(1,3); g.addEdge(2,3);
      Poly r = complementCliqueDecompose(g, {0,1}, {{2},{3}}, chi);
      check(ff_of(r) == (std::vector<mpz_class>{0,0,0,3,1}),
            "complementCliqueDecompose: l=2 p=2 triangle+isolated a=[0,0,0,3,1]"); }

    // PIN l=3, p=2: n=5, E={(1,3),(1,4),(2,4),(3,4)}. S={0,1,2}, comps={{3},{4}}. a=[0,0,0,6,6,1].
    { Graph g(5); g.addEdge(1,3); g.addEdge(1,4); g.addEdge(2,4); g.addEdge(3,4);
      Poly r = complementCliqueDecompose(g, {0,1,2}, {{3},{4}}, chi);
      check(ff_of(r) == (std::vector<mpz_class>{0,0,0,6,6,1}),
            "complementCliqueDecompose: l=3 p=2 a=[0,0,0,6,6,1]"); }

    // PIN l=5, p=2: g7 = one edge (5,6) + isolated {0,1,2,3,4}. complement(g7) = K7 - e has a
    // size-5 clique separator S={0,1,2,3,4}; comps of Gbar-S = {5},{6}. chi(g7) = k^6(k-1), whose
    // falling-factorial a-vector is [0,0,32,211,285,125,20,1]. This exercises the general-l umbral
    // inclusion-exclusion at l=5 (52 set-partitions of |S|=5) — the largest l the formula supports
    // in practice. (The engine caps *detection* at kComplementSepMaxL=3, so l=5 is reached only by
    // a direct call; the C2 investigation, 2026-07-07, found firing it via a raised cap is a
    // measured NO-GO. This PIN guards the math's correctness independently of that decision.)
    { Graph g(7); g.addEdge(5,6);
      Poly r = complementCliqueDecompose(g, {0,1,2,3,4}, {{5},{6}}, chi);
      check(ff_of(r) == (std::vector<mpz_class>{0,0,32,211,285,125,20,1}),
            "complementCliqueDecompose: l=5 p=2 a=[0,0,32,211,285,125,20,1]"); }

    std::cout << "test_complement_decompose passed\n";
}

static void test_avec_helpers() {
    using namespace crl;
    // convolution: [0,1,1] (idx1,idx2) ⊛ [0,1] (idx1) = [0,0,1,1] (idx2,idx3)
    std::vector<mpz_class> a{0,1,1}, b{0,1};
    auto c = avecConvolve(a, b);
    check((c == std::vector<mpz_class>{0,0,1,1}), "avecConvolve([0,1,1],[0,1]) == [0,0,1,1]");
    // identity: [1] ⊛ a == a
    check((avecConvolve(std::vector<mpz_class>{1}, a) == a), "avecConvolve: {1} is the identity");
    // add with scale: acc[0,0,2,2] += -1 * [0,0,0,1] -> [0,0,2,1]
    std::vector<mpz_class> acc{0,0,2,2};
    avecAddInPlace(acc, std::vector<mpz_class>{0,0,0,1}, mpz_class(-1));
    check((acc == std::vector<mpz_class>{0,0,2,1}), "avecAddInPlace: scale=-1 decrements acc[3]");
    // shift up: prepend one zero to [1,2,1] -> [0,1,2,1]
    check((avecShiftUp(std::vector<mpz_class>{1,2,1}, 1) == std::vector<mpz_class>{0,1,2,1}),
          "avecShiftUp: prepends 1 zero");
    std::cout << "test_avec_helpers passed\n";
}

int main() {
    auto sel = selectMinDegreeSum;   // default selector for closed-form tests

    std::cout << "== (A) closed-form checks ==\n";
    for (int n = 1; n <= 8; ++n)
        check(computeChromatic(makeEmpty(n), sel) == chromaticEmpty(n),
              "empty K_bar n=" + std::to_string(n));
    for (int n = 2; n <= 8; ++n)
        check(computeChromatic(makePath(n), sel) == chromaticTree(n),
              "path (tree) n=" + std::to_string(n));
    for (int n = 3; n <= 8; ++n)
        check(computeChromatic(makeCycle(n), sel) == chromaticCycle(n),
              "cycle C_n n=" + std::to_string(n));
    for (int n = 1; n <= 8; ++n)
        check(computeChromatic(makeComplete(n), sel) == chromaticComplete(n),
              "complete K_n n=" + std::to_string(n));

    // Spot-check a known value: P(C_4, k) = (k-1)^4 + (k-1) ; at k=3 -> 16+2 = 18.
    check(chromaticCycle(4).eval(3) == 18, "C_4 eval at k=3 equals 18");
    // P(K_4, k) at k=4 = 4*3*2*1 = 24.
    check(chromaticComplete(4).eval(4) == 24, "K_4 eval at k=4 equals 24");

    // divByKPow: exact division by k^p. K_3 = t(t-1)(t-2) = t^3-3t^2+2t; /k = (t-1)(t-2).
    // Invariant: (P / k^p) * k^p == P  when P is divisible by k^p.
    {
        Poly k3 = chromaticComplete(3);                 // = (k-1)(k-2)*k, divisible by k^1
        check(k3.divByKPow(1) * Poly::monomial(1) == k3, "divByKPow(1) round-trips on K_3");
        check(k3.divByKPow(1).eval(3) == k3.eval(3) / 3, "divByKPow(1) of K_3 eval matches /3 at k=3");
        check(k3.divByKPow(1).coeffs()[0] == 2,          "divByKPow(1) of K_3 has constant term 2");
    }

    // divExact: exact long division by a monic divisor (chi(K_l) is monic).
    // K_4 = k(k-1)(k-2)(k-3); K_4 / K_2 = (k-2)(k-3); at k=5 -> 3*2 = 6.
    {
        Poly k4 = chromaticComplete(4);
        Poly k2 = chromaticComplete(2);          // k(k-1), monic
        Poly q  = k4.divExact(k2);               // expect (k-2)(k-3)
        check(q * k2 == k4,        "divExact: K_4 / K_2 round-trips");
        check(q.eval(5) == 6,      "divExact: (K_4 / K_2) eval at k=5 equals 6");
        // dividing K_4 by K_3 leaves (k-3); at k=5 -> 2.
        check(k4.divExact(chromaticComplete(3)).eval(5) == 2,
                                   "divExact: (K_4 / K_3) eval at k=5 equals 2");
    }

    // == (E) falling-factorial basis + umbral product (join reduction machinery) ==
    std::cout << "== (E) falling-factorial basis + umbral product ==\n";
    {
        // a-vector of chi(C_5) = sum a_j (k)_j. Verified: [0,0,0,5,5,1].
        std::vector<mpz_class> aC5 = chromaticCycle(5).toFallingFactorial();
        bool aOk = (aC5 == std::vector<mpz_class>{0,0,0,5,5,1});
        check(aOk, "toFallingFactorial(C_5) == [0,0,0,5,5,1]");
        check(Poly::fromFallingFactorial(aC5) == chromaticCycle(5),
              "fromFallingFactorial round-trips chi(C_5)");
        // Ebar_2 has chi = k^2 = (k)_1 + (k)_2  -> a = [0,1,1].
        check(Poly::fromFallingFactorial(std::vector<mpz_class>{0,1,1}) == chromaticEmpty(2),
              "fromFallingFactorial([0,1,1]) == k^2");
        // Umbral identity: Poly::one() has a-vector [1]; umbralProduct(one, p) == p.
        check(umbralProduct(Poly::one(), chromaticCycle(5)) == chromaticCycle(5),
              "umbralProduct identity: one (x) C_5 == C_5");
        // Join K_{2,3} = Ebar_2 join Ebar_3. Verified power form: k^5-6k^4+15k^3-17k^2+7k.
        Poly k23expected(std::vector<mpz_class>{0, 7, -17, 15, -6, 1});
        check(umbralProduct(chromaticEmpty(2), chromaticEmpty(3)) == k23expected,
              "umbralProduct(Ebar_2, Ebar_3) == chi(K_{2,3})");
    }
    {
        // Complement-component detection. makeKnMinusMatching(6) is the octahedron
        // K_{2,2,2}; its complement is 3 disjoint edges -> 3 components.
        check(complementComponents(makeComplete(5)).size() == 5,
              "complementComponents(K_5) == 5 singletons (Gbar edgeless)");
        check(complementComponents(makeKnMinusMatching(6)).size() == 3,
              "complementComponents(octahedron K_{2,2,2}) == 3");
        check(complementComponents(makeCycle(5)).size() == 1,
              "complementComponents(C_5) == 1 (self-complementary, connected)");
    }
    {
        // The join reduction must produce the CORRECT polynomial (brute-force oracle) and
        // must fully decompose complete-multipartite graphs into closed-form pieces in 0
        // branches (Gbar disconnected -> each piece is an empty graph / base case).
        Graph oct = makeKnMinusMatching(6);          // octahedron K_{2,2,2}
        Stats so; Poly poct = computeChromatic(oct, selectMinDegreeSum, so);
        bool octAgree = true;
        for (int k = 0; k <= 8; ++k)
            if (poct.eval(k) != bruteCount(oct, k)) { octAgree = false; break; }
        check(octAgree, "join: octahedron K_{2,2,2} matches brute force");
        check(so.branches == 0, "join: octahedron solved in 0 branches (join reduction)");

        Graph k123 = makeK123();
        Stats sk; Poly pk = computeChromatic(k123, selectMinDegreeSum, sk);
        bool k123Agree = true;
        for (int k = 0; k <= 8; ++k)
            if (pk.eval(k) != bruteCount(k123, k)) { k123Agree = false; break; }
        check(k123Agree, "join: complete tripartite K_{1,2,3} matches brute force");
        check(sk.branches == 0, "join: K_{1,2,3} solved in 0 branches (join reduction)");
    }

    // Wheel (Theorem 4): W_4 == K_4 (independent closed form), and a spot value.
    // W_5 = hub + C_4: chi(W_5,t)=t[(t-2)^4+(t-2)]; at t=3 -> 3*(1+1)=6; at t=4 -> 4*(16+2)=72.
    check(chromaticWheel(4) == chromaticComplete(4), "chromaticWheel(4) equals K_4");
    check(chromaticWheel(5).eval(3) == 6,  "chromaticWheel(5) eval at k=3 equals 6");
    check(chromaticWheel(5).eval(4) == 72, "chromaticWheel(5) eval at k=4 equals 72");

    // isWheel detection: true on a real wheel (hub identified), false on cycle / K_5.
    {
        int hub = -1;
        check(isWheel(makeWheel(5), hub) && hub == 4, "isWheel detects W_5 with hub=4");
        check(!isWheel(makeCycle(5), hub),            "isWheel rejects C_5");
        check(!isWheel(makeComplete(5), hub),         "isWheel rejects K_5 (not a wheel)");
        // False-positive guard: hub joined to two disjoint triangles is NOT a wheel,
        // even though hub has degree n-1 and the rim is 2-regular with n-1 edges.
        {
            Graph imposter(7);                       // hub=6 + triangles {0,1,2},{3,4,5}
            imposter.addEdge(0,1); imposter.addEdge(1,2); imposter.addEdge(2,0);
            imposter.addEdge(3,4); imposter.addEdge(4,5); imposter.addEdge(5,3);
            for (int v = 0; v < 6; ++v) imposter.addEdge(6, v);
            check(!isWheel(imposter, hub), "isWheel rejects hub + two disjoint triangles");
        }
        // Hub need not be the last vertex: W_5 with hub at vertex 0.
        {
            Graph w(5);                              // hub=0, rim cycle 1-2-3-4-1
            for (int v = 1; v <= 4; ++v) w.addEdge(0, v);
            w.addEdge(1,2); w.addEdge(2,3); w.addEdge(3,4); w.addEdge(4,1);
            int h = -1;
            check(isWheel(w, h) && h == 0, "isWheel detects W_5 with hub at vertex 0");
        }
    }

    // biconnectedComponents: block counts on known structures.
    check(biconnectedComponents(makeBowtie()).size()   == 2, "bowtie has 2 blocks");
    check(biconnectedComponents(makeCycle(5)).size()   == 1, "C_5 is a single block");
    check(biconnectedComponents(makeComplete(4)).size()== 1, "K_4 is a single block");
    check(biconnectedComponents(makePath(4)).size()    == 3, "P_4 has 3 edge-blocks");

    // Vertex-membership (not just counts): a wrong assignment would silently corrupt
    // the engine's per-block product. The cut vertex must appear in every block it joins.
    {
        auto has = [](const std::vector<int>& b, int x){ return std::find(b.begin(), b.end(), x) != b.end(); };
        auto bs = biconnectedComponents(makeBowtie());
        bool v0inBoth = bs.size() == 2 && has(bs[0], 0) && has(bs[1], 0);
        check(v0inBoth, "bowtie cut vertex 0 appears in both blocks");
        // each bowtie block is one triangle: 3 vertices, and {1,2} resp {3,4} are NOT shared
        bool triangleSizes = bs.size() == 2 && bs[0].size() == 3 && bs[1].size() == 3;
        check(triangleSizes, "bowtie blocks each have 3 vertices");

        auto c5 = biconnectedComponents(makeCycle(5));
        check(c5.size() == 1 && c5[0].size() == 5, "C_5 single block contains all 5 vertices");

        // P_4 blocks are the three edges {0,1},{1,2},{2,3}; union covers all 4 vertices.
        auto p4 = biconnectedComponents(makePath(4));
        std::set<int> covered;
        for (auto& b : p4) for (int v : b) covered.insert(v);
        bool allPairs = p4.size() == 3;
        for (auto& b : p4) if (b.size() != 2) allPairs = false;
        check(allPairs, "P_4 blocks are all 2-vertex (edge) blocks");
        check(covered == std::set<int>({0,1,2,3}), "P_4 blocks cover all 4 vertices");
    }

    // mcsmTriangulation: H ⊇ G, alpha is a permutation of 1..n, and fill counts
    // match the known minimal triangulation of cycles (C_n needs n-3 chords).
    {
        auto fillCount = [](const Graph& g) {
            Graph H; std::vector<int> alpha;
            mcsmTriangulation(g, H, alpha);
            // alpha is a permutation of 1..n
            std::vector<int> seen(g.numVertices() + 1, 0);
            for (int v = 0; v < g.numVertices(); ++v) seen[alpha[v]] = 1;
            bool perm = true;
            for (int i = 1; i <= g.numVertices(); ++i) perm &= (seen[i] == 1);
            return std::pair<long long,bool>{H.numEdges() - g.numEdges(), perm};
        };
        check(fillCount(makeComplete(4)).first == 0,  "MCS-M: K_4 already chordal (0 fill)");
        check(fillCount(makeComplete(4)).second,       "MCS-M: alpha is a permutation (K_4)");
        check(fillCount(makeCycle(4)).first == 1,      "MCS-M: C_4 needs 1 chord");
        check(fillCount(makeCycle(4)).second,          "MCS-M: alpha is a permutation (C_4, with fill)");
        check(fillCount(makeCycle(5)).first == 2,      "MCS-M: C_5 needs 2 chords");
        check(fillCount(makeDiamond()).first == 0,     "MCS-M: diamond already chordal (0 fill)");
    }

    // isCliqueInG: a set is a clique iff every pair is adjacent in G.
    check(isCliqueInG(makeComplete(4), {0,1,2}),   "isCliqueInG: triangle of K_4 is a clique");
    check(isCliqueInG(makeDiamond(),   {0,1}),     "isCliqueInG: shared edge {0,1} is a clique");
    check(!isCliqueInG(makeK23(),      {0,1}),     "isCliqueInG: K_{2,3} part {0,1} is NOT a clique");
    check(isCliqueInG(makeK23(),       {0}),       "isCliqueInG: single vertex is trivially a clique");

    // cliqueSeparator: l=2 (separating edge), l=3 (separating triangle), and negatives.
    check(cliqueSeparator(makeDiamond()).size() == 2,
          "diamond has a size-2 clique separator");
    check(cliqueSeparator(makeTwoK4SharingTriangle()).size() == 3,
          "two K_4 sharing a triangle: size-3 clique separator");
    check(cliqueSeparator(makeBook3()).size() == 2,
          "book B_3 has a size-2 clique separator");
    check(cliqueSeparator(makeCycle(5)).empty(),    "C_5 has no clique separator");
    check(cliqueSeparator(makeComplete(5)).empty(), "K_5 has no clique separator");
    check(cliqueSeparator(makeK23()).empty(),
          "K_{2,3} has no clique separator (its separator is not a clique)");
    check(cliqueSeparator(makeBowtie()).size() == 1,
          "bowtie has a size-1 clique separator (cut vertex)");
    check(cliqueSeparator(makeTwoK5SharingK4()).size() == 4,
          "two K_5 sharing a K_4: size-4 clique separator");

    std::cout << "== (B) brute-force cross-check on random Erdos-Renyi graphs ==\n";
    std::mt19937 rng(12345);
    int trials = 40;
    int bruteFails = 0, selectorFails = 0;
    for (int t = 0; t < trials; ++t) {
        int n = 4 + (rng() % 5);                 // n in [4,8]
        double p = 0.3 + (rng() % 40) / 100.0;   // p in [0.30,0.69]
        Graph g = makeErdosRenyi(n, p, rng);

        Poly poly = computeChromatic(g, selectMinDegreeSum);

        // (B1) agree with brute force at k = 0..n  => identical polynomials.
        bool agree = true;
        for (int k = 0; k <= n; ++k)
            if (poly.eval(k) != bruteCount(g, k)) { agree = false; break; }
        if (!agree) ++bruteFails;

        // (B2) selector-invariance: every heuristic must give the same polynomial.
        if (!(poly == computeChromatic(g, selectMaxDegreeSum))) ++selectorFails;
    }
    check(bruteFails == 0,
          "engine matches brute force on " + std::to_string(trials) + " random graphs");
    check(selectorFails == 0,
          "result invariant to selector on " + std::to_string(trials) + " random graphs");

    // (B') Random clique-sums sharing a K_l (l=4,5): two sides each attached to all of a
    // shared K_l (plus random intra-side edges) so the size-l clique is guaranteed to
    // separate them. This exercises the general MCS-M clique-separator path on l>=4 cuts
    // — coverage the n<=8 ER section above does not reliably produce — against the oracle.
    std::cout << "== (B') brute-force cross-check on random K_l clique-sums (l=4,5) ==\n";
    {
        const int csTrials = 30;
        int csFails = 0, csSelFails = 0, csExercised = 0;
        for (int t = 0; t < csTrials; ++t) {
            int l = 4 + (int)(rng() % 2);            // shared clique size 4 or 5
            int a = 1 + (int)(rng() % 2);            // side A extra vertices
            int b = 1 + (int)(rng() % 2);            // side B extra vertices
            if (l == 5) { a = 1; b = 1; }            // cap n at 8 (brute force stays cheap)
            int n = l + a + b;
            Graph g(n);
            for (int i = 0; i < l; ++i)              // shared clique S = {0..l-1}
                for (int j = i + 1; j < l; ++j) g.addEdge(i, j);
            auto wireSide = [&](int start, int count) {
                for (int v = start; v < start + count; ++v) {
                    for (int s = 0; s < l; ++s) g.addEdge(v, s);     // attach to all of S
                    for (int u = start; u < v; ++u)                 // random intra-side edges
                        if (rng() % 2) g.addEdge(u, v);
                }
            };
            wireSide(l, a);
            wireSide(l + a, b);

            if (!cliqueSeparator(g).empty()) ++csExercised;          // the new path must fire
            Poly poly = computeChromatic(g, selectMinDegreeSum);
            bool agree = true;
            for (int k = 0; k <= n; ++k)
                if (poly.eval(k) != bruteCount(g, k)) { agree = false; break; }
            if (!agree) ++csFails;
            if (!(poly == computeChromatic(g, selectMaxDegreeSum))) ++csSelFails;
        }
        check(csFails == 0,
              "engine matches brute force on " + std::to_string(csTrials) + " random K_l clique-sums");
        check(csSelFails == 0,
              "result invariant to selector on " + std::to_string(csTrials) + " random K_l clique-sums");
        check(csExercised == csTrials,
              "every random clique-sum exercised the clique-separator path");
    }

    std::cout << "== (C) cut-vertex + wheel graphs vs brute force ==\n";
    {
        auto crossCheck = [&](const Graph& g, const std::string& name) {
            Poly poly = computeChromatic(g, selectMinDegreeSum);
            int n = g.numVertices();
            bool agree = true;
            for (int k = 0; k <= n; ++k)
                if (poly.eval(k) != bruteCount(g, k)) { agree = false; break; }
            check(agree, name + " matches brute force");
            // selector-invariance through the new reduction paths too
            check(poly == computeChromatic(g, selectMaxDegreeSum), name + " selector-invariant");
        };
        crossCheck(makeBowtie(),   "bowtie (2x K_3)");
        crossCheck(makeLollipop(), "lollipop (K_4 + K_3)");
        crossCheck(makeWheel(5),   "wheel W_5");
        crossCheck(makeWheel(6),         "wheel W_6");
        crossCheck(makeTripleBowtie(), "triple bowtie (3x K_3, 1 cut vertex)");

        // Bowtie closed form: t(t-1)^2(t-2)^2; at k=3 -> 3*4*1 = 12.
        check(computeChromatic(makeBowtie(), selectMinDegreeSum).eval(3) == 12,
              "bowtie eval at k=3 equals 12");
        // Engine wheel result equals the closed form (detection path is exercised).
        check(computeChromatic(makeWheel(6), selectMinDegreeSum) == chromaticWheel(6),
              "engine W_6 equals chromaticWheel(6)");

        // 3 blocks share ONE cut vertex: chi = [k(k-1)(k-2)]^3 / k^2 = k(k-1)^3(k-2)^3.
        // At k=3: 3 * 2^3 * 1^3 = 24. This pins the divisor exponent at r-1=2 (not #cut vertices=1).
        check(computeChromatic(makeTripleBowtie(), selectMinDegreeSum).eval(3) == 24,
              "triple bowtie eval at k=3 equals 24");
        { Stats s; computeChromatic(makeTripleBowtie(), selectMinDegreeSum, s);
          check(s.branches == 0, "triple bowtie solved with 0 branches (l=1 clique separator, p=3)"); }

        // The reductions must REDUCE work: W_6 hits the closed form in 0 branches,
        // and the bowtie decomposes into triangles (each a base case) -> 0 branches.
        { Stats s; computeChromatic(makeWheel(6), selectMinDegreeSum, s);
          check(s.branches == 0, "W_6 solved with 0 branches (wheel base case)"); }
        { Stats s; computeChromatic(makeBowtie(), selectMinDegreeSum, s);
          check(s.branches == 0, "bowtie solved with 0 branches (l=1 clique separator)"); }

        // --- CRT-1 clique-separator decomposition (any l>=1, MCS-M) ---
        crossCheck(makeDiamond(),               "diamond (2x K_3 sharing an edge)");
        crossCheck(makeTwoK4SharingTriangle(),  "two K_4 sharing a triangle");
        crossCheck(makeBook3(),                 "book B_3 (3x K_3 sharing an edge)");

        // Diamond closed form: k(k-1)(k-2)^2; at k=3 -> 3*2*1 = 6.
        check(computeChromatic(makeDiamond(), selectMinDegreeSum).eval(3) == 6,
              "diamond eval at k=3 equals 6");
        // Two K_4 sharing a triangle: k(k-1)(k-2)(k-3)^2; at k=4 -> 4*3*2*1 = 24.
        check(computeChromatic(makeTwoK4SharingTriangle(), selectMinDegreeSum).eval(4) == 24,
              "two-K4-triangle eval at k=4 equals 24");
        // Book B_3: k(k-1)(k-2)^3; at k=3 -> 3*2*1 = 6.
        check(computeChromatic(makeBook3(), selectMinDegreeSum).eval(3) == 6,
              "book B_3 eval at k=3 equals 6");

        // 0 branches: every clique-separator piece is a complete graph (base case).
        { Stats s; computeChromatic(makeDiamond(), selectMinDegreeSum, s);
          check(s.branches == 0, "diamond solved with 0 branches (l=2 clique separator)"); }
        { Stats s; computeChromatic(makeTwoK4SharingTriangle(), selectMinDegreeSum, s);
          check(s.branches == 0, "two-K4-triangle solved with 0 branches (l=3 clique separator)"); }
        { Stats s; computeChromatic(makeBook3(), selectMinDegreeSum, s);
          check(s.branches == 0, "book B_3 solved with 0 branches (l=2, p=3 clique separator)"); }

        // l=4: two K_5 sharing a K_4. chi = k(k-1)(k-2)(k-3)(k-4)^2; at k=5 -> 120.
        crossCheck(makeTwoK5SharingK4(), "two K_5 sharing a K_4 (l=4)");
        check(computeChromatic(makeTwoK5SharingK4(), selectMinDegreeSum).eval(5) == 120,
              "two-K5-K4 eval at k=5 equals 120");
        { Stats s; computeChromatic(makeTwoK5SharingK4(), selectMinDegreeSum, s);
          check(s.branches == 0, "two-K5-K4 solved with 0 branches (l=4 clique separator)"); }
    }

    std::cout << "== (D) bidirectional primitives ==\n";
    {
        // G+f: adding the missing chord to a path P3 yields a triangle (3 edges).
        Graph p3 = makePath(3);                 // edges {0-1, 1-2}
        Graph tri = p3.insertEdge(0, 2);
        check(tri.hasEdge(0, 2) && tri.numEdges() == 3 && tri.numVertices() == 3,
              "insertEdge adds the non-edge (P3 + chord = triangle)");

        // G+f is idempotent on an existing edge (set semantics, no parallels).
        Graph p3b = p3.insertEdge(0, 1);
        check(p3b.numEdges() == 2 && p3b.hasEdge(0, 1),
              "insertEdge on an existing edge is a no-op");

        // G+f does not mutate the receiver (immutable style, like deleteEdge).
        check(p3.numEdges() == 2 && !p3.hasEdge(0, 2),
              "insertEdge leaves the original graph unchanged");
    }

    {
        // Sparse/tie side: P4 (4 vertices, 3 edges, C(4,2)=6, 2m=6 == tie -> down): an EDGE.
        Graph p4 = makePath(4);
        auto sp = selectBidirectionalStructural(p4);
        check(p4.hasEdge(sp.first, sp.second),
              "structural selector returns an edge on the sparse/tie side");

        // Dense side: K5 minus one edge (n=5, C(5,2)=10, m=9, 2m=18 > 10 -> up):
        // returns the single NON-adjacent pair.
        Graph k5me = makeComplete(5).deleteEdge(0, 1);
        auto dn = selectBidirectionalStructural(k5me);
        check(!k5me.hasEdge(dn.first, dn.second),
              "structural selector returns a non-edge on the dense side");
    }

    std::cout << "== (D') bidirectional engine: regression + dense correctness ==\n";
    {
        // (D'1) Regression equality on random instances (including dense): the bidirectional
        // engine (structural selector) must equal the validated down-only result exactly.
        std::mt19937 rng2(98765);
        int n_eq = 40, mism = 0;
        for (int t = 0; t < n_eq; ++t) {
            int n = 4 + (rng2() % 5);                 // n in [4,8]
            double p = 0.3 + (rng2() % 50) / 100.0;   // p in [0.30,0.79] (some dense)
            Graph g = makeErdosRenyi(n, p, rng2);
            Poly down = computeChromatic(g, selectMinDegreeSum);
            Poly bi   = computeChromaticBidir(g, selectBidirectionalStructural);
            if (!(down == bi)) ++mism;
        }
        check(mism == 0,
              "bidirectional == down-only on " + std::to_string(n_eq) + " random graphs");

        // (D'2) Dense case vs the independent brute-force oracle, AND the up-direction is
        // actually exercised (a non-adjacent pair returned at least once). The Φ-guard
        // asserts run live here (asserts are enabled: -O2, no -DNDEBUG).
        //
        // Note: makeKnMinusMatching(6) (octahedron K_{2,2,2}) is now decomposed by the join
        // reduction (its complement is 3 disjoint edges = disconnected), so the bidirectional
        // selector is never invoked on it. Brute-force correctness still holds (checked here),
        // but the up-direction exercise uses K_6 minus one edge instead: its complement is the
        // single edge {0,1}, which is connected, so the join reduction does not fire and the
        // bidirectional selector is actually invoked on the dense side.
        {
            Graph octahedron = makeKnMinusMatching(6);
            Poly poct = computeChromaticBidir(octahedron, selectBidirectionalStructural);
            bool agree = true;
            for (int k = 0; k <= 6; ++k)
                if (poct.eval(k) != bruteCount(octahedron, k)) { agree = false; break; }
            check(agree, "bidirectional matches brute force on K6 minus a perfect matching");
        }
        int upMoves = 0;
        EdgeSelector counting = [&](const Graph& gg) {
            auto pr = selectBidirectionalStructural(gg);
            if (!gg.hasEdge(pr.first, pr.second)) ++upMoves;
            return pr;
        };
        // g = complement(C_6): n=6, m=9, 2m=18 > C(6,2)=15 (dense). complement(g) = C_6, a
        // chordless 2-connected cycle with NO clique separator of any size, so neither the join
        // reduction nor the complement clique-separator reduction fires. g itself also has no
        // CRT-1 clique separator. The structural selector is therefore invoked on the dense side
        // and returns a non-edge -> upMoves > 0.
        Graph dense = complementGraph(makeCycle(6));
        Poly poly = computeChromaticBidir(dense, counting);
        bool agree = true;
        for (int k = 0; k <= 6; ++k)
            if (poly.eval(k) != bruteCount(dense, k)) { agree = false; break; }
        check(agree, "bidirectional matches brute force on complement(C_6)");
        check(upMoves > 0, "bidirectional exercised the up-direction on the dense instance");

        // (D'3) Up-direction selector-invariance: a DIFFERENT legal selector (max-degree-sum
        // edge on the sparse side, max-degree-sum non-edge on the dense side) gives the same
        // polynomial. Both only ever return legal pairs, so the Φ-guard never trips.
        EdgeSelector altLegal = [](const Graph& gg) -> std::pair<int,int> {
            const int n = gg.numVertices();
            const long long m = gg.numEdges();
            const long long maxE = (long long)n * (n - 1) / 2;
            if (2 * m <= maxE) return selectMaxDegreeSum(gg);    // legal down (sparse)
            std::pair<int,int> best{-1,-1}; int bestScore = -1;  // legal up (dense): max-deg-sum non-edge
            for (int u = 0; u < n; ++u)
                for (int v = u + 1; v < n; ++v)
                    if (!gg.hasEdge(u, v)) {
                        int s = gg.degree(u) + gg.degree(v);
                        if (s > bestScore) { bestScore = s; best = {u, v}; }
                    }
            return best;
        };
        check(computeChromaticBidir(dense, altLegal) == poly,
              "bidirectional result invariant to the (legal) up-direction selector on complement(C_6)");
    }

    std::cout << "== (F) a-vector arithmetic helpers ==\n";
    test_avec_helpers();

    std::cout << "== (G) complementGraph + setPartitions ==\n";
    test_complement_setpartitions();

    std::cout << "== (H) pieceGraph builder ==\n";
    test_piece_graph();

    std::cout << "== (I) complementCliqueDecompose (verified general-l formula) ==\n";
    test_complement_decompose();

    std::cout << "== (J) complement clique-separator reduction (engine integration) ==\n";
    test_complement_reduction_oracle();

    std::cout << "\n== example output ==\n";
    {
        std::mt19937 r2(7);
        Graph g = makeErdosRenyi(6, 0.5, r2);
        std::cout << "  random G(6,0.5): P(G,k) = "
                  << computeChromatic(g, selectMinDegreeSum).str() << "\n";
        std::cout << "  Petersen-like check K_4: P = "
                  << computeChromatic(makeComplete(4), sel).str() << "\n";
    }

    std::cout << "\n" << (g_fail == 0 ? "ALL CHECKS PASSED" : "FAILURES: " + std::to_string(g_fail)) << "\n";
    return g_fail == 0 ? 0 : 1;
}
