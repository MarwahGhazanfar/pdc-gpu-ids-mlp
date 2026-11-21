// Compare two saved weight files: parameter differences and prediction agreement on a data split.
// Build: g++ -O3 -march=native -std=c++17 src/common/compare_weights.cpp -o build/compare_weights
// Usage: ./build/compare_weights A.bin B.bin [data_dir] [split]
#include "mlp_common.h"

static bool load_params(const std::string& path, int F, int K, Params<float>& p) {
    p = zero_params<float>(F, K);
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    for (auto* v : p.list())
        if (std::fread(v->data(), sizeof(float), v->size(), f) != v->size()) { std::fclose(f); return false; }
    std::fclose(f);
    return true;
}

static std::vector<int> predict(const Params<float>& p, const Dataset& d, int K) {
    std::vector<int> out(d.N);
    const int CH = 8192;
    Acts<float> a;
    a.resize(CH, K);
    for (int s = 0; s < d.N; s += CH) {
        int b = std::min(CH, d.N - s);
        forward(p, d.X.data() + (size_t)s * d.F, b, d.F, K, a);
        for (int i = 0; i < b; i++) {
            const float* z = a.Z3.data() + (size_t)i * K;
            int arg = 0;
            for (int j = 1; j < K; j++) if (z[j] > z[arg]) arg = j;
            out[s + i] = arg;
        }
    }
    return out;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s A.bin B.bin [data_dir] [split]\n", argv[0]); return 1; }
    const int F = 78, K = 6;
    std::string dir = argc > 3 ? argv[3] : "data/processed", split = argc > 4 ? argv[4] : "val";
    Params<float> A, B;
    if (!load_params(argv[1], F, K, A) || !load_params(argv[2], F, K, B)) { std::fprintf(stderr, "could not read weight files\n"); return 1; }

    double maxabs = 0, sd = 0, sw = 0;
    size_t n = 0;
    auto al = A.list(); auto bl = B.list();
    for (int t = 0; t < 6; t++)
        for (size_t i = 0; i < al[t]->size(); i++) {
            double x = (*al[t])[i], y = (*bl[t])[i], d = x - y;
            maxabs = std::max(maxabs, std::fabs(d));
            sd += d * d; sw += x * x; n++;
        }
    std::printf("parameters compared: %zu\n", n);
    std::printf("max abs difference : %.3e\n", maxabs);
    std::printf("RMS difference     : %.3e (relative to RMS weight %.3e: %.3e)\n",
                std::sqrt(sd / n), std::sqrt(sw / n), std::sqrt(sd / n) / std::sqrt(sw / n));
    if (maxabs == 0) std::printf("RESULT: weights are bit-for-bit IDENTICAL\n");

    Dataset d;
    if (!load_split(dir, split, F, d)) { std::fprintf(stderr, "could not load %s split from %s\n", split.c_str(), dir.c_str()); return 1; }
    auto pa = predict(A, d, K), pb = predict(B, d, K);
    long agree = 0;
    for (int i = 0; i < d.N; i++) agree += pa[i] == pb[i];
    Metrics ma = evaluate(A, d, K), mb = evaluate(B, d, K);
    std::printf("\n%s split (%d rows)\n", split.c_str(), d.N);
    std::printf("prediction agreement: %.4f %% (%ld of %d)\n", 100.0 * agree / d.N, agree, d.N);
    std::printf("A: loss %.5f acc %.5f macro-F1 %.5f\n", ma.loss, ma.acc, ma.macro_f1);
    std::printf("B: loss %.5f acc %.5f macro-F1 %.5f\n", mb.loss, mb.acc, mb.macro_f1);
    return 0;
}
