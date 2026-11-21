// OpenMP implementation: same model, data, initialisation, shuffling and hyperparameters as the serial version.
// Each mini-batch is split by rows across threads; every thread computes gradients into its own private
// buffer, then the buffers are summed in fixed thread order and one SGD-momentum update is applied.
// Build: g++ -O3 -march=native -fopenmp -std=c++17 -Wall -Wextra src/openmp/mlp_omp.cpp -o build/mlp_omp
#include "../common/mlp_common.h"
#include <cstring>
#include <omp.h>

struct Args {
    std::string data_dir = "data/processed", log = "results/omp_log.csv", save = "", cw = "sqrt";
    int F = 78, K = 6, batch = 512, epochs = 30, threads = 0;
    double lr = 0.01, momentum = 0.9;
    uint64_t seed = 42;
    int max_train = 0;
    bool eval_test = false;
};

static Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        auto nxt = [&]() { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", s.c_str()); std::exit(1); } return std::string(argv[++i]); };
        if (s == "--data-dir") a.data_dir = nxt();
        else if (s == "--log") a.log = nxt();
        else if (s == "--save") a.save = nxt();
        else if (s == "--class-weights") a.cw = nxt();
        else if (s == "--features") a.F = std::stoi(nxt());
        else if (s == "--classes") a.K = std::stoi(nxt());
        else if (s == "--batch") a.batch = std::stoi(nxt());
        else if (s == "--epochs") a.epochs = std::stoi(nxt());
        else if (s == "--threads") a.threads = std::stoi(nxt());
        else if (s == "--lr") a.lr = std::stod(nxt());
        else if (s == "--momentum") a.momentum = std::stod(nxt());
        else if (s == "--seed") a.seed = std::stoull(nxt());
        else if (s == "--max-train") a.max_train = std::stoi(nxt());
        else if (s == "--eval-test") a.eval_test = true;
        else { std::fprintf(stderr, "unknown option %s\n", s.c_str()); std::exit(1); }
    }
    return a;
}

int main(int argc, char** argv) {
    Args a = parse(argc, argv);
    const int F = a.F, K = a.K;
    const int nt = a.threads > 0 ? a.threads : omp_get_max_threads();
    if (nt > 64) { std::fprintf(stderr, "at most 64 threads supported\n"); return 1; }
    omp_set_dynamic(0);

    Dataset tr, va, te;
    if (!load_split(a.data_dir, "train", F, tr) || !load_split(a.data_dir, "val", F, va)) {
        std::fprintf(stderr, "Could not load data from %s (run preprocess first)\n", a.data_dir.c_str());
        return 1;
    }
    int N = tr.N;
    if (a.max_train > 0 && a.max_train < N) N = a.max_train;
    const char* bind = std::getenv("OMP_PROC_BIND");
    std::printf("OpenMP threads %d (omp max %d, OMP_PROC_BIND=%s)\n", nt, omp_get_max_threads(), bind ? bind : "unset");
    std::printf("train rows %d (using %d) | val rows %d | F=%d K=%d | batch %d | epochs %d | lr %.4g | momentum %.2f | seed %llu\n",
                tr.N, N, va.N, F, K, a.batch, a.epochs, a.lr, a.momentum, (unsigned long long)a.seed);

    std::vector<float> cw(K, 1.0f);
    if (a.cw != "none") {
        std::vector<double> cnt(K, 0);
        for (int i = 0; i < N; i++) cnt[tr.y[i]]++;
        for (int k = 0; k < K; k++) {
            double w = (double)N / (K * std::max(cnt[k], 1.0));
            cw[k] = (float)(a.cw == "inv" ? w : std::sqrt(w));
        }
    }
    std::printf("class weights (%s):", a.cw.c_str());
    for (int k = 0; k < K; k++) std::printf(" %.3f", cw[k]);
    std::printf("\n");

    Params<float> P = init_params(F, K, a.seed);
    Params<float> V = zero_params<float>(F, K);

    // thread-private buffers (allocated once)
    const int maxrows = (a.batch + nt - 1) / nt + 1;
    std::vector<Params<float>> Gt(nt, zero_params<float>(F, K));
    std::vector<Acts<float>> At(nt);
    std::vector<std::vector<float>> Xb(nt, std::vector<float>((size_t)maxrows * F));
    std::vector<std::vector<int32_t>> yb(nt, std::vector<int32_t>(maxrows));
    std::vector<std::array<const float*, 6>> gp(nt);
    for (int t = 0; t < nt; t++) {
        At[t].resize(maxrows, K);
        auto l = Gt[t].list();
        for (int i = 0; i < 6; i++) gp[t][i] = l[i]->data();
    }
    std::vector<int> rows(nt, 0);
    std::vector<double> lsum(nt, 0.0);
    std::vector<int> perm(N);

    FILE* lg = std::fopen(a.log.c_str(), "w");
    if (lg) std::fprintf(lg, "epoch,train_loss,val_loss,val_acc,val_macro_f1,epoch_seconds\n");

    double train_total = 0;
    const float lr = (float)a.lr, mu = (float)a.momentum;
    for (int ep = 1; ep <= a.epochs; ep++) {
        for (int i = 0; i < N; i++) perm[i] = i;
        Rng sh(a.seed ^ (0x9E3779B97F4A7C15ULL * (uint64_t)ep));
        for (int i = N - 1; i > 0; i--) std::swap(perm[i], perm[sh.below((uint64_t)i + 1)]);

        double t0 = now_sec(), loss_sum = 0;
        for (int s = 0; s < N; s += a.batch) {
            const int b = std::min(a.batch, N - s);
            #pragma omp parallel num_threads(nt)
            {
                const int tid = omp_get_thread_num();
                const int r0 = (int)((long long)b * tid / nt), r1 = (int)((long long)b * (tid + 1) / nt);
                const int nr = r1 - r0;
                rows[tid] = nr;
                if (nr > 0) {
                    for (int i = 0; i < nr; i++) {
                        int r = perm[s + r0 + i];
                        std::memcpy(&Xb[tid][(size_t)i * F], &tr.X[(size_t)r * F], F * sizeof(float));
                        yb[tid][i] = tr.y[r];
                    }
                    lsum[tid] = loss_and_grad(P, Xb[tid].data(), yb[tid].data(), nr, F, K, cw.data(), At[tid], Gt[tid]) * nr;
                } else {
                    lsum[tid] = 0;
                }
                #pragma omp barrier   // all gradients done; weights are only modified after this point
                float sc[64];
                for (int th = 0; th < nt; th++) sc[th] = (float)rows[th] / (float)b;
                auto pl = P.list();
                auto vl = V.list();
                for (int t = 0; t < 6; t++) {
                    float* p = pl[t]->data();
                    float* v = vl[t]->data();
                    const long n = (long)pl[t]->size();
                    #pragma omp for schedule(static) nowait
                    for (long i = 0; i < n; i++) {
                        float g = 0.0f;
                        for (int th = 0; th < nt; th++)       // fixed order -> deterministic reduction
                            if (rows[th] > 0) g += sc[th] * gp[th][t][i];
                        v[i] = mu * v[i] - lr * g;
                        p[i] += v[i];
                    }
                }
            }
            for (int th = 0; th < nt; th++) loss_sum += lsum[th];
        }
        double et = now_sec() - t0;
        train_total += et;
        Metrics m = evaluate(P, va, K);
        std::printf("epoch %3d | train loss %.4f | val loss %.4f | val acc %.4f | val macro-F1 %.4f | %.2f s\n",
                    ep, loss_sum / N, m.loss, m.acc, m.macro_f1, et);
        if (lg) { std::fprintf(lg, "%d,%.6f,%.6f,%.6f,%.6f,%.4f\n", ep, loss_sum / N, m.loss, m.acc, m.macro_f1, et); std::fflush(lg); }
    }
    if (lg) std::fclose(lg);
    std::printf("\nTRAINING TIME (epochs only, excludes validation): %.2f s total, %.2f s per epoch\n",
                train_total, train_total / a.epochs);

    print_metrics("final validation metrics", evaluate(P, va, K), K);
    if (a.eval_test) {
        if (!load_split(a.data_dir, "test", F, te)) { std::fprintf(stderr, "could not load test split\n"); return 1; }
        print_metrics("TEST metrics (final evaluation)", evaluate(P, te, K), K);
    }
    if (!a.save.empty() && save_params(a.save, P)) std::printf("saved weights to %s\n", a.save.c_str());
    return 0;
}
