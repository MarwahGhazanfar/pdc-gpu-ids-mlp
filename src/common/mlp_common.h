// Shared code for the serial, OpenMP and CUDA backends: data loading, RNG, parameter init,
// forward/backward pass (templated so gradient checks can run in double), metrics.
#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

constexpr int H1 = 64;  // hidden layer 1 size (frozen architecture)
constexpr int H2 = 32;  // hidden layer 2 size

inline double now_sec() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

// ---------- data ----------
struct Dataset {
    int F = 0, N = 0;
    std::vector<float> X;      // N x F, row-major
    std::vector<int32_t> y;    // N
};

template <typename V>
bool read_bin(const std::string& path, std::vector<V>& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize(sz / sizeof(V));
    size_t n = std::fread(out.data(), sizeof(V), out.size(), f);
    std::fclose(f);
    return n == out.size();
}

inline bool load_split(const std::string& dir, const std::string& name, int F, Dataset& d) {
    d.F = F;
    if (!read_bin(dir + "/X_" + name + ".bin", d.X)) return false;
    if (!read_bin(dir + "/y_" + name + ".bin", d.y)) return false;
    if (d.X.size() % F != 0) return false;
    d.N = (int)(d.X.size() / F);
    return d.y.size() == (size_t)d.N;
}

// ---------- deterministic RNG (splitmix64: identical on every platform) ----------
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    double uniform() { return ((next() >> 11) + 0.5) * (1.0 / 9007199254740992.0); }  // (0,1)
    double normal() {
        double u1 = uniform(), u2 = uniform();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }
    uint64_t below(uint64_t n) { return next() % n; }
};

// ---------- parameters ----------
template <typename T>
struct Params {
    std::vector<T> W1, b1, W2, b2, W3, b3;  // W_l is (in x out), row-major
    std::array<std::vector<T>*, 6> list() { return {&W1, &b1, &W2, &b2, &W3, &b3}; }
    std::array<const std::vector<T>*, 6> list() const { return {&W1, &b1, &W2, &b2, &W3, &b3}; }
};

template <typename T>
Params<T> zero_params(int F, int K) {
    Params<T> p;
    p.W1.assign((size_t)F * H1, 0); p.b1.assign(H1, 0);
    p.W2.assign((size_t)H1 * H2, 0); p.b2.assign(H2, 0);
    p.W3.assign((size_t)H2 * K, 0); p.b3.assign(K, 0);
    return p;
}

// He initialisation (for ReLU), zero biases. Same seed -> same weights in every backend.
inline Params<float> init_params(int F, int K, uint64_t seed) {
    Params<float> p = zero_params<float>(F, K);
    Rng r(seed);
    auto fill = [&](std::vector<float>& W, int fan_in) {
        double sd = std::sqrt(2.0 / fan_in);
        for (auto& w : W) w = (float)(r.normal() * sd);
    };
    fill(p.W1, F);
    fill(p.W2, H1);
    fill(p.W3, H2);
    return p;
}

template <typename To, typename From>
Params<To> convert_params(const Params<From>& s) {
    Params<To> d;
    auto sl = s.list();
    auto dl = d.list();
    for (int t = 0; t < 6; t++) dl[t]->assign(sl[t]->begin(), sl[t]->end());
    return d;
}

inline bool save_params(const std::string& path, const Params<float>& p) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    for (auto* v : p.list()) std::fwrite(v->data(), sizeof(float), v->size(), f);
    std::fclose(f);
    return true;
}

// ---------- forward / backward ----------
template <typename T>
struct Acts {
    std::vector<T> Z1, A1, Z2, A2, Z3, D3, D2, D1;
    void resize(int B, int K) {
        Z1.resize((size_t)B * H1); A1.resize((size_t)B * H1); D1.resize((size_t)B * H1);
        Z2.resize((size_t)B * H2); A2.resize((size_t)B * H2); D2.resize((size_t)B * H2);
        Z3.resize((size_t)B * K);  D3.resize((size_t)B * K);
    }
};

// Z = A*W + b   (A: M x In, W: In x Out, Z: M x Out)
template <typename T>
void dense(const T* A, const T* W, const T* b, T* Z, int M, int In, int Out) {
    for (int i = 0; i < M; i++) {
        T* z = Z + (size_t)i * Out;
        for (int j = 0; j < Out; j++) z[j] = b[j];
        const T* a = A + (size_t)i * In;
        for (int k = 0; k < In; k++) {
            T av = a[k];
            const T* w = W + (size_t)k * Out;
            for (int j = 0; j < Out; j++) z[j] += av * w[j];
        }
    }
}

template <typename T>
void relu(const T* Z, T* A, size_t n) {
    for (size_t i = 0; i < n; i++) A[i] = Z[i] > 0 ? Z[i] : 0;
}

template <typename T>
void forward(const Params<T>& p, const T* X, int B, int F, int K, Acts<T>& a) {
    dense(X, p.W1.data(), p.b1.data(), a.Z1.data(), B, F, H1);
    relu(a.Z1.data(), a.A1.data(), (size_t)B * H1);
    dense(a.A1.data(), p.W2.data(), p.b2.data(), a.Z2.data(), B, H1, H2);
    relu(a.Z2.data(), a.A2.data(), (size_t)B * H2);
    dense(a.A2.data(), p.W3.data(), p.b3.data(), a.Z3.data(), B, H2, K);
}

// gW += A^T * D ; gb += colsum(D)   (A: M x In, D: M x Out)
template <typename T>
void grad_dense(const T* A, const T* D, T* gW, T* gb, int M, int In, int Out) {
    for (int i = 0; i < M; i++) {
        const T* d = D + (size_t)i * Out;
        for (int j = 0; j < Out; j++) gb[j] += d[j];
        const T* a = A + (size_t)i * In;
        for (int k = 0; k < In; k++) {
            T av = a[k];
            T* g = gW + (size_t)k * Out;
            for (int j = 0; j < Out; j++) g[j] += av * d[j];
        }
    }
}

// Weighted softmax cross-entropy, averaged over the batch. Fills g with the batch-averaged gradients.
// cw may be nullptr (all weights 1). Returns the loss.
template <typename T>
double loss_and_grad(const Params<T>& p, const T* X, const int32_t* y, int B, int F, int K,
                     const T* cw, Acts<T>& a, Params<T>& g) {
    forward(p, X, B, F, K, a);
    double loss = 0;
    for (int i = 0; i < B; i++) {
        const T* z = a.Z3.data() + (size_t)i * K;
        T m = z[0];
        for (int j = 1; j < K; j++) m = std::max(m, z[j]);
        T sum = 0;
        for (int j = 0; j < K; j++) sum += std::exp(z[j] - m);
        T lse = std::log(sum) + m;
        T w = cw ? cw[y[i]] : (T)1;
        loss += (double)(w * (lse - z[y[i]]));
        T* d = a.D3.data() + (size_t)i * K;
        for (int j = 0; j < K; j++) d[j] = w * (std::exp(z[j] - lse) - (j == y[i] ? (T)1 : (T)0)) / (T)B;
    }
    loss /= B;

    for (auto* v : g.list()) std::fill(v->begin(), v->end(), (T)0);
    grad_dense(a.A2.data(), a.D3.data(), g.W3.data(), g.b3.data(), B, H2, K);
    for (int i = 0; i < B; i++)
        for (int k = 0; k < H2; k++) {
            T s = 0;
            for (int j = 0; j < K; j++) s += a.D3[(size_t)i * K + j] * p.W3[(size_t)k * K + j];
            a.D2[(size_t)i * H2 + k] = a.Z2[(size_t)i * H2 + k] > 0 ? s : (T)0;
        }
    grad_dense(a.A1.data(), a.D2.data(), g.W2.data(), g.b2.data(), B, H1, H2);
    for (int i = 0; i < B; i++)
        for (int k = 0; k < H1; k++) {
            T s = 0;
            for (int j = 0; j < H2; j++) s += a.D2[(size_t)i * H2 + j] * p.W2[(size_t)k * H2 + j];
            a.D1[(size_t)i * H1 + k] = a.Z1[(size_t)i * H1 + k] > 0 ? s : (T)0;
        }
    grad_dense(X, a.D1.data(), g.W1.data(), g.b1.data(), B, F, H1);
    return loss;
}

// ---------- metrics ----------
struct Metrics {
    double acc = 0, macro_p = 0, macro_r = 0, macro_f1 = 0, loss = 0;
    std::vector<double> prec, rec, f1;
    std::vector<long> conf;  // K x K, row = true class, col = predicted class
};

inline Metrics evaluate(const Params<float>& p, const Dataset& d, int K) {
    Metrics m;
    m.conf.assign((size_t)K * K, 0);
    const int CH = 8192;
    Acts<float> a;
    a.resize(CH, K);
    double loss = 0;
    for (int s = 0; s < d.N; s += CH) {
        int b = std::min(CH, d.N - s);
        forward(p, d.X.data() + (size_t)s * d.F, b, d.F, K, a);
        for (int i = 0; i < b; i++) {
            const float* z = a.Z3.data() + (size_t)i * K;
            int arg = 0;
            float mx = z[0];
            for (int j = 1; j < K; j++) if (z[j] > mx) { mx = z[j]; arg = j; }
            double sum = 0;
            for (int j = 0; j < K; j++) sum += std::exp((double)z[j] - mx);
            int t = d.y[s + i];
            loss += std::log(sum) + mx - z[t];
            m.conf[(size_t)t * K + arg]++;
        }
    }
    m.loss = loss / d.N;
    m.prec.assign(K, 0); m.rec.assign(K, 0); m.f1.assign(K, 0);
    long correct = 0;
    for (int k = 0; k < K; k++) {
        long tp = m.conf[(size_t)k * K + k], rs = 0, cs = 0;
        for (int j = 0; j < K; j++) { rs += m.conf[(size_t)k * K + j]; cs += m.conf[(size_t)j * K + k]; }
        correct += tp;
        m.prec[k] = cs ? (double)tp / cs : 0;
        m.rec[k] = rs ? (double)tp / rs : 0;
        m.f1[k] = (m.prec[k] + m.rec[k]) > 0 ? 2 * m.prec[k] * m.rec[k] / (m.prec[k] + m.rec[k]) : 0;
        m.macro_p += m.prec[k] / K; m.macro_r += m.rec[k] / K; m.macro_f1 += m.f1[k] / K;
    }
    m.acc = (double)correct / d.N;
    return m;
}

inline void print_metrics(const char* title, const Metrics& m, int K) {
    static const char* names[] = {"Benign", "DoS/DDoS", "PortScan", "Brute-force", "Web Attack", "Other"};
    std::printf("\n--- %s ---\n", title);
    std::printf("loss %.4f | accuracy %.4f | macro-P %.4f | macro-R %.4f | macro-F1 %.4f\n",
                m.loss, m.acc, m.macro_p, m.macro_r, m.macro_f1);
    std::printf("%-12s %9s %9s %9s\n", "class", "precision", "recall", "F1");
    for (int k = 0; k < K; k++)
        std::printf("%-12s %9.4f %9.4f %9.4f\n", k < 6 ? names[k] : "?", m.prec[k], m.rec[k], m.f1[k]);
    std::printf("confusion matrix (rows = true, cols = predicted):\n");
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) std::printf("%9ld", m.conf[(size_t)i * K + j]);
        std::printf("\n");
    }
}
