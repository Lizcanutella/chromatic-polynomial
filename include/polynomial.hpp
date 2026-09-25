// polynomial.hpp — Phase 0.3: exact integer polynomials in the power basis.
//
// coeffs_[i] is the coefficient of k^i. Coefficients are exact GMP integers
// (chromatic-polynomial coefficients are Stirling-number-sized and overflow 64-bit
// quickly, e.g. already past n ~ 20 for K_n). The operations needed by the engine are
// modest: subtraction (the deletion-contraction step), multiplication (component
// products and base-case formulae), and integer evaluation (validation cross-check).
#pragma once
#include <cassert>
#include <vector>
#include <string>
#include <gmpxx.h>

namespace crl {

class Poly {
public:
    Poly() = default;
    explicit Poly(std::vector<mpz_class> c) : c_(std::move(c)) { trim(); }

    static Poly zero() { return Poly(); }
    static Poly one() { return Poly(std::vector<mpz_class>{mpz_class(1)}); }

    // k^p
    static Poly monomial(int p) {
        std::vector<mpz_class> c(p + 1, mpz_class(0));
        c[p] = 1;
        return Poly(std::move(c));
    }

    // (k - a)
    static Poly linear(long a) {
        return Poly(std::vector<mpz_class>{mpz_class(-a), mpz_class(1)});
    }

    int degree() const { return c_.empty() ? -1 : static_cast<int>(c_.size()) - 1; }
    bool isZero() const { return c_.empty(); }
    const std::vector<mpz_class>& coeffs() const { return c_; }

    Poly operator+(const Poly& o) const {
        std::vector<mpz_class> r(std::max(c_.size(), o.c_.size()), mpz_class(0));
        for (size_t i = 0; i < c_.size(); ++i)   r[i] += c_[i];
        for (size_t i = 0; i < o.c_.size(); ++i) r[i] += o.c_[i];
        return Poly(std::move(r));
    }

    Poly operator-(const Poly& o) const {
        std::vector<mpz_class> r(std::max(c_.size(), o.c_.size()), mpz_class(0));
        for (size_t i = 0; i < c_.size(); ++i)   r[i] += c_[i];
        for (size_t i = 0; i < o.c_.size(); ++i) r[i] -= o.c_[i];
        return Poly(std::move(r));
    }

    Poly operator*(const Poly& o) const {
        if (isZero() || o.isZero()) return Poly();
        std::vector<mpz_class> r(c_.size() + o.c_.size() - 1, mpz_class(0));
        for (size_t i = 0; i < c_.size(); ++i)
            for (size_t j = 0; j < o.c_.size(); ++j)
                r[i + j] += c_[i] * o.c_[j];
        return Poly(std::move(r));
    }

    // Exact division by k^p: drops the p lowest-degree coefficients (each guaranteed
    // zero by the caller, by divisibility). p<=0 is identity. A tested algebra primitive;
    // the engine's CRT-1 reduction now divides via the general divExact(chromaticComplete(l)).
    Poly divByKPow(int p) const {
        if (p <= 0) return *this;
        for (int i = 0; i < p && i < static_cast<int>(c_.size()); ++i)
            assert(c_[i] == 0 && "divByKPow: dividend not divisible by k^p");
        if (static_cast<size_t>(p) >= c_.size()) return Poly();   // everything shifted away -> zero
        return Poly(std::vector<mpz_class>(c_.begin() + p, c_.end()));
    }

    // Exact division by a divisor whose leading coefficient is +1 or -1 ("monic up to
    // sign"), via polynomial long division. The caller guarantees exact divisibility
    // (CRT-1 theorem); we assert a zero remainder. chi(K_l) is monic, so the quotient
    // stays entirely in the integers (no rationals).
    Poly divExact(const Poly& d) const {
        assert(!d.isZero() && "divExact: division by zero polynomial");
        const mpz_class& lead = d.c_.back();
        assert((lead == 1 || lead == -1) && "divExact: divisor must be monic up to sign");
        if (isZero()) return Poly();
        const int dd = d.degree();
        if (degree() < dd) {                       // nonzero & lower-degree => not divisible
            for (const auto& a : c_) assert(a == 0 && "divExact: not divisible");
            return Poly();
        }
        std::vector<mpz_class> rem = c_;
        std::vector<mpz_class> q(degree() - dd + 1, mpz_class(0));
        for (int i = degree(); i >= dd; --i) {
            if (rem[i] == 0) continue;
            mpz_class factor = rem[i] / lead;       // exact: lead = +/-1
            q[i - dd] = factor;
            for (int j = 0; j <= dd; ++j)
                rem[i - dd + j] -= factor * d.c_[j];
        }
        for (const auto& r : rem)
            assert(r == 0 && "divExact: nonzero remainder (dividend not divisible)");
        return Poly(std::move(q));
    }

    // Falling-factorial ("umbral") coefficients a_j with p(k) = sum_j a_j (k)_j,
    // where (k)_j = k(k-1)...(k-j+1). Newton forward differences:
    //   a_j = (1/j!) sum_{i=0}^{j} (-1)^{j-i} C(j,i) p(i).
    // For chromatic polynomials the a_j are non-negative integers (partition counts);
    // exact divisibility by j! is asserted.
    std::vector<mpz_class> toFallingFactorial() const {
        const int d = degree();
        if (d < 0) return {};                       // zero polynomial -> empty vector
        std::vector<mpz_class> a(d + 1);
        for (int j = 0; j <= d; ++j) {
            mpz_class s = 0, binom = 1;             // binom tracks C(j, i)
            for (int i = 0; i <= j; ++i) {
                mpz_class term = binom * eval(i);
                if ((j - i) & 1) s -= term; else s += term;
                binom = binom * (j - i) / (i + 1);  // C(j,i) -> C(j,i+1), exact integer
            }
            mpz_class fact = 1;
            for (int t = 2; t <= j; ++t) fact *= t; // j!
            assert(s % fact == 0 && "toFallingFactorial: non-integer umbral coefficient");
            a[j] = s / fact;
        }
        return a;
    }

    // Inverse of toFallingFactorial: rebuild p(k) = sum_t a[t] (k)_t in the power basis.
    static Poly fromFallingFactorial(const std::vector<mpz_class>& a) {
        Poly out = Poly::zero();
        Poly ff = Poly::one();                      // (k)_0 = 1
        for (std::size_t t = 0; t < a.size(); ++t) {
            if (a[t] != 0) out = out + Poly(std::vector<mpz_class>{a[t]}) * ff;
            ff = ff * Poly::linear(static_cast<long>(t));   // (k)_{t+1} = (k)_t * (k - t)
        }
        return out;
    }

    // Evaluate at integer k (used to cross-check against brute-force colouring counts).
    mpz_class eval(long k) const {
        mpz_class acc = 0, kp = 1;
        for (size_t i = 0; i < c_.size(); ++i) { acc += c_[i] * kp; kp *= k; }
        return acc;
    }

    bool operator==(const Poly& o) const { return c_ == o.c_; }

    std::string str() const {
        if (isZero()) return "0";
        std::string s;
        for (int i = degree(); i >= 0; --i) {
            if (c_[i] == 0) continue;
            if (!s.empty()) s += " + ";
            s += c_[i].get_str();
            if (i >= 1) s += "*k^" + std::to_string(i);
        }
        return s.empty() ? "0" : s;
    }

private:
    void trim() { while (!c_.empty() && c_.back() == 0) c_.pop_back(); }
    std::vector<mpz_class> c_;
};

// chi of the join G1 v G2: convolution of the two falling-factorial coefficient vectors.
// (In a join every colour class lies in one part and the parts use distinct colours, so
// the falling-factorial coefficients convolve.) Associative and commutative; Poly::one()
// (a-vector [1]) is the identity, so it folds cleanly over many parts.
inline Poly umbralProduct(const Poly& x, const Poly& y) {
    std::vector<mpz_class> a = x.toFallingFactorial();
    std::vector<mpz_class> b = y.toFallingFactorial();
    if (a.empty() || b.empty()) return Poly::zero();
    std::vector<mpz_class> r(a.size() + b.size() - 1, mpz_class(0));
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j)
            r[i + j] += a[i] * b[j];
    return Poly::fromFallingFactorial(r);
}

// ---------------------------------------------------------------------------
// Raw a-vector (falling-factorial coefficient vector) arithmetic helpers.
// These operate directly on std::vector<mpz_class> without boxing in Poly.
// ---------------------------------------------------------------------------

/// Convolve two a-vectors: r[i+j] += a[i]*b[j].
/// Identity element is {1}; empty input returns {0}.
inline std::vector<mpz_class> avecConvolve(const std::vector<mpz_class>& a,
                                           const std::vector<mpz_class>& b) {
    if (a.empty() || b.empty()) return {mpz_class(0)};
    std::vector<mpz_class> r(a.size() + b.size() - 1, mpz_class(0));
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j)
            r[i + j] += a[i] * b[j];
    return r;
}

/// In-place scaled accumulation: acc[i] += scale * v[i], growing acc as needed.
inline void avecAddInPlace(std::vector<mpz_class>& acc,
                           const std::vector<mpz_class>& v, const mpz_class& scale) {
    if (v.size() > acc.size()) acc.resize(v.size(), mpz_class(0));
    for (std::size_t i = 0; i < v.size(); ++i) acc[i] += scale * v[i];
}

/// Shift up: prepend q zeros (multiply by (k)_q in the falling-factorial basis).
inline std::vector<mpz_class> avecShiftUp(const std::vector<mpz_class>& v, int q) {
    std::vector<mpz_class> r(q, mpz_class(0));
    r.insert(r.end(), v.begin(), v.end());
    return r;
}

// Coefficient-word count: total GMP limbs across all coefficients. A cheap, exact
// proxy for the memory footprint of a polynomial (used by the engine memory proxies).
inline long long polyWords(const Poly& p) {
    long long w = 0;
    for (const auto& c : p.coeffs()) w += static_cast<long long>(mpz_size(c.get_mpz_t()));
    return w;
}

} // namespace crl
