// Serial reference implementation: 78-64-32-6 MLP, ReLU, softmax cross-entropy, SGD with momentum.
// Build: g++ -O3 -march=native -std=c++17 -Wall -Wextra src/serial/mlp_serial.cpp -o build/mlp_serial
#include "../common/mlp_common.h"
#include <cstring>

struct Args {
    std::string data_dir = "data/processed", log = "results/serial_log.csv", save = "", cw = "sqrt";
    int F = 78, K = 6, batch = 512, epochs = 30;
    double lr = 0.01, momentum = 0.9;
    uint64_t seed = 42;
    int max_train = 0;
    bool eval_test = false, gradcheck = false;
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
        else if (s == "--lr") a.lr = std::stod(nxt());
        else if (s == "--momentum") a.momentum = std::stod(nxt());
        else if (s == "--seed") a.seed = std::stoull(nxt());
        else if (s == "--max-train") a.max_train = std::stoi(nxt());
        else if (s == "--eval-test") a.eval_test = true;
        else if (s == "--gradcheck") a.gradcheck = true;
        else { std::fprintf(stderr, "unknown option %s\n", s.c_str()); std::exit(1); }
    }
    return a;
}

int main(int argc, char** argv) {
    Args a = parse(argc, argv);
    const int F = a.F, K = a.K;

    Dataset tr, va, te;
    if (!load_split(a.data_dir, "train", F, tr) || !load_split(a.data_dir, "val", F, va)) {
        std::fprintf(stderr, "Could not load data from %s (run preprocess first)\n", a.data_dir.c_str());
        return 1;
    }
    int N = tr.N;
    if (a.max_train > 0 && a.max_train < N) N = a.max_train;
    std::printf("train rows %d (using %d) | val rows %d | F=%d K=%d | batch %d | epochs %d | lr %.4g | momentum %.2f | seed %llu\n",
                tr.N, N, va.N, F, K, a.batch, a.epochs, a.lr, a.momentum, (unsigned long long)a.seed);

    // class weights from the training rows in use: none | sqrt | inv
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

    // ---------- gradient check (double precision, central differences) ----------
    if (a.gradcheck) {
        const int B = 16;
        Params<double> pd = convert_params<double>(P), g = zero_params<double>(F, K), gs = zero_params<double>(F, K);
        std::vector<double> Xd(tr.X.begin(), tr.X.begin() + (size_t)B * F), cwd(cw.begin(), cw.end());
        std::vector<int32_t> yb(tr.y.begin(), tr.y.begin() + B);
        Acts<double> acts;
        acts.resize(B, K);
        loss_and_grad(pd, Xd.data(), yb.data(), B, F, K, cwd.data(), acts, g);
        Rng rr(7);
        const char* names[] = {"W1", "b1", "W2", "b2", "W3", "b3"};
        double worst = 0;
        auto pl = pd.list();
        auto gl = g.list();
        for (int t = 0; t < 6; t++) {
            double tmax = 0;
            for (int s = 0; s < 25; s++) {
                size_t i = rr.below(pl[t]->size());
                double orig = (*pl[t])[i], eps = 1e-5;
                (*pl[t])[i] = orig + eps;
                double lp = loss_and_grad(pd, Xd.data(), yb.data(), B, F, K, cwd.data(), acts, gs);
                (*pl[t])[i] = orig - eps;
                double lm = loss_and_grad(pd, Xd.data(), yb.data(), B, F, K, cwd.data(), acts, gs);
                (*pl[t])[i] = orig;
                double num = (lp - lm) / (2 * eps), ana = (*gl[t])[i];
                double rel = std::fabs(num - ana) / std::max(1e-8, std::fabs(num) + std::fabs(ana));
                tmax = std::max(tmax, rel);
            }
            std::printf("gradcheck %-2s max relative error %.3e\n", names[t], tmax);
            worst = std::max(worst, tmax);
        }
        std::printf("GRADCHECK %s (worst %.3e, threshold 1e-5)\n", worst < 1e-5 ? "PASSED" : "FAILED", worst);
        return worst < 1e-5 ? 0 : 2;
    }

    // ---------- training ----------
    Params<float> V = zero_params<float>(F, K), G = zero_params<float>(F, K);
    Acts<float> acts;
    acts.resize(a.batch, K);
    std::vector<float> Xb((size_t)a.batch * F);
    std::vector<int32_t> yb(a.batch);
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
            int b = std::min(a.batch, N - s);
            for (int i = 0; i < b; i++) {
                int r = perm[s + i];
                std::memcpy(&Xb[(size_t)i * F], &tr.X[(size_t)r * F], F * sizeof(float));
                yb[i] = tr.y[r];
            }
            loss_sum += loss_and_grad(P, Xb.data(), yb.data(), b, F, K, cw.data(), acts, G) * b;
            auto pl = P.list(); auto vl = V.list(); auto gl = G.list();
            for (int t = 0; t < 6; t++) {
                float* p = pl[t]->data(); float* v = vl[t]->data(); const float* g = gl[t]->data();
                size_t n = pl[t]->size();
                for (size_t i = 0; i < n; i++) { v[i] = mu * v[i] - lr * g[i]; p[i] += v[i]; }
            }
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
