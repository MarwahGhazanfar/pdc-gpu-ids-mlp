// CUDA implementation: same model, data, initialisation, shuffling and hyperparameters as the serial version.
// The whole training set lives on the GPU (uploaded once); each mini-batch is gathered, trained and updated on the device.
// Build (Colab T4): nvcc -O3 -arch=sm_75 -std=c++17 src/cuda/mlp_cuda.cu -o build/mlp_cuda
#include "../common/mlp_common.h"
#include <cstring>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                      \
    do {                                                                                      \
        cudaError_t e_ = (call);                                                              \
        if (e_ != cudaSuccess) {                                                              \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(1);                                                                     \
        }                                                                                     \
    } while (0)

constexpr int TILE = 16;

// C = op(A) * op(B) (+ bias[col]) (masked by mask>0). Logical shapes: C is MxN, reduction length Kd.
// TA=false: A stored MxKd.  TA=true: A stored KdxM (use A^T).
// TB=false: B stored KdxN.  TB=true: B stored NxKd (use B^T).
template <bool TA, bool TB>
__global__ void gemm_kernel(const float* A, const float* B, float* C, int M, int N, int Kd,
                            const float* bias, const float* mask) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.f;
    for (int t = 0; t < Kd; t += TILE) {
        const int ak = t + threadIdx.x;
        const int bk = t + threadIdx.y;
        float av = 0.f, bv = 0.f;
        if (row < M && ak < Kd) av = TA ? A[(size_t)ak * M + row] : A[(size_t)row * Kd + ak];
        if (bk < Kd && col < N) bv = TB ? B[(size_t)col * Kd + bk] : B[(size_t)bk * N + col];
        As[threadIdx.y][threadIdx.x] = av;
        Bs[threadIdx.y][threadIdx.x] = bv;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; k++) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) {
        float v = acc;
        if (bias) v += bias[col];
        if (mask) v = mask[(size_t)row * N + col] > 0.f ? v : 0.f;
        C[(size_t)row * N + col] = v;
    }
}

template <bool TA, bool TB>
static void gemm(const float* A, const float* B, float* C, int M, int N, int Kd,
                 const float* bias = nullptr, const float* mask = nullptr) {
    dim3 blk(TILE, TILE), grd((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemm_kernel<TA, TB><<<grd, blk>>>(A, B, C, M, N, Kd, bias, mask);
}

__global__ void relu_kernel(const float* Z, float* A, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) A[i] = Z[i] > 0.f ? Z[i] : 0.f;
}

__global__ void gather_kernel(const float* X, const int* y, const int* perm, int b, int F, float* Xb, int* yb) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= b * F) return;
    int i = idx / F, c = idx - i * F;
    int r = perm[i];
    Xb[idx] = X[(size_t)r * F + c];
    if (c == 0) yb[i] = y[r];
}

// one thread per row: weighted softmax cross-entropy; writes dL/dZ3 (already divided by batch size)
__global__ void softmax_loss_kernel(const float* Z3, const int* y, const float* cw, float* D3, double* loss, int B, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
    const float* z = Z3 + (size_t)i * K;
    float m = z[0];
    for (int j = 1; j < K; j++) m = fmaxf(m, z[j]);
    float sum = 0.f;
    for (int j = 0; j < K; j++) sum += expf(z[j] - m);
    float lse = logf(sum) + m;
    int t = y[i];
    float w = cw[t];
    atomicAdd(loss, (double)(w * (lse - z[t])));
    float* d = D3 + (size_t)i * K;
    for (int j = 0; j < K; j++) d[j] = w * (expf(z[j] - lse) - (j == t ? 1.f : 0.f)) / (float)B;
}

// out[j] = sum_i D[i*N + j]; one block per column
__global__ void colsum_kernel(const float* D, float* out, int M, int N) {
    int j = blockIdx.x;
    float s = 0.f;
    for (int i = threadIdx.x; i < M; i += blockDim.x) s += D[(size_t)i * N + j];
    __shared__ float sh[128];
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int o = 64; o > 0; o >>= 1) {
        if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[j] = sh[0];
}

__global__ void sgd_kernel(float* P, float* V, const float* G, int n, float lr, float mu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = mu * V[i] - lr * G[i];
        V[i] = v;
        P[i] += v;
    }
}

struct Args {
    std::string data_dir = "data/processed", log = "results/cuda_log.csv", save = "", cw = "sqrt", ckpt = "", resume = "";
    int F = 78, K = 6, batch = 512, epochs = 30, ckpt_every = 5;
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
        else if (s == "--ckpt") a.ckpt = nxt();
        else if (s == "--resume") a.resume = nxt();
        else if (s == "--ckpt-every") a.ckpt_every = std::stoi(nxt());
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
        else { std::fprintf(stderr, "unknown option %s\n", s.c_str()); std::exit(1); }
    }
    return a;
}

static std::vector<float> flatten(const Params<float>& p) {
    std::vector<float> f;
    for (auto* v : p.list()) f.insert(f.end(), v->begin(), v->end());
    return f;
}
static void unflatten(const std::vector<float>& f, Params<float>& p) {
    size_t off = 0;
    for (auto* v : p.list()) { std::copy(f.begin() + off, f.begin() + off + v->size(), v->begin()); off += v->size(); }
}

int main(int argc, char** argv) {
    Args a = parse(argc, argv);
    const int F = a.F, K = a.K, B = a.batch;

    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("GPU: %s (%d SMs, %.1f GB)\n", prop.name, prop.multiProcessorCount, prop.totalGlobalMem / 1073741824.0);

    Dataset tr, va, te;
    if (!load_split(a.data_dir, "train", F, tr) || !load_split(a.data_dir, "val", F, va)) {
        std::fprintf(stderr, "Could not load data from %s\n", a.data_dir.c_str());
        return 1;
    }
    int N = tr.N;
    if (a.max_train > 0 && a.max_train < N) N = a.max_train;
    std::printf("train rows %d (using %d) | val rows %d | F=%d K=%d | batch %d | epochs %d | lr %.4g | momentum %.2f | seed %llu\n",
                tr.N, N, va.N, F, K, B, a.epochs, a.lr, a.momentum, (unsigned long long)a.seed);

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

    // ---- parameters on host, flattened for the device ----
    Params<float> Ph = init_params(F, K, a.seed);
    std::vector<float> Pf = flatten(Ph);
    const int NP = (int)Pf.size();
    std::vector<float> Vf(NP, 0.f);
    int start_epoch = 1;
    if (!a.resume.empty()) {
        FILE* f = std::fopen(a.resume.c_str(), "rb");
        int hdr[3];
        if (!f || std::fread(hdr, sizeof(int), 3, f) != 3 || hdr[0] != 0x4D4C5043 || hdr[2] != NP ||
            std::fread(Pf.data(), sizeof(float), NP, f) != (size_t)NP || std::fread(Vf.data(), sizeof(float), NP, f) != (size_t)NP) {
            std::fprintf(stderr, "could not resume from %s\n", a.resume.c_str());
            return 1;
        }
        std::fclose(f);
        start_epoch = hdr[1] + 1;
        std::printf("resumed from %s: continuing at epoch %d\n", a.resume.c_str(), start_epoch);
    }
    const size_t oW1 = 0, ob1 = (size_t)F * H1, oW2 = ob1 + H1, ob2 = oW2 + (size_t)H1 * H2,
                 oW3 = ob2 + H2, ob3 = oW3 + (size_t)H2 * K;

    // ---- device buffers ----
    float *d_X, *d_P, *d_V, *d_G, *d_cw, *d_Xb, *d_Z1, *d_A1, *d_D1, *d_Z2, *d_A2, *d_D2, *d_Z3, *d_D3;
    int *d_y, *d_perm, *d_yb;
    double* d_loss;
    double t_up0 = now_sec();
    CUDA_CHECK(cudaMalloc(&d_X, (size_t)N * F * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, (size_t)N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_perm, (size_t)N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_P, NP * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, NP * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_G, NP * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cw, K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Xb, (size_t)B * F * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_yb, (size_t)B * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_Z1, (size_t)B * H1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A1, (size_t)B * H1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_D1, (size_t)B * H1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Z2, (size_t)B * H2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A2, (size_t)B * H2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_D2, (size_t)B * H2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Z3, (size_t)B * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_D3, (size_t)B * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_X, tr.X.data(), (size_t)N * F * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, tr.y.data(), (size_t)N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_P, Pf.data(), NP * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, Vf.data(), NP * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cw, cw.data(), K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    const double t_upload = now_sec() - t_up0;
    std::printf("H2D upload of dataset + parameters: %.3f s\n", t_upload);

    std::vector<int> perm(N);
    FILE* lg = std::fopen(a.log.c_str(), "w");
    if (lg) std::fprintf(lg, "epoch,train_loss,val_loss,val_acc,val_macro_f1,epoch_seconds,kernel_seconds\n");
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    double train_total = 0, kernel_total = 0;
    const float lr = (float)a.lr, mu = (float)a.momentum;
    const int TB = 256;
    int epochs_run = 0;
    for (int ep = start_epoch; ep <= a.epochs; ep++) {
        for (int i = 0; i < N; i++) perm[i] = i;
        Rng sh(a.seed ^ (0x9E3779B97F4A7C15ULL * (uint64_t)ep));
        for (int i = N - 1; i > 0; i--) std::swap(perm[i], perm[sh.below((uint64_t)i + 1)]);

        CUDA_CHECK(cudaDeviceSynchronize());
        double t0 = now_sec();
        CUDA_CHECK(cudaMemcpy(d_perm, perm.data(), (size_t)N * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(double)));
        CUDA_CHECK(cudaEventRecord(ev0));
        for (int s = 0; s < N; s += B) {
            const int b = std::min(B, N - s);
            gather_kernel<<<(b * F + TB - 1) / TB, TB>>>(d_X, d_y, d_perm + s, b, F, d_Xb, d_yb);
            // forward
            gemm<false, false>(d_Xb, d_P + oW1, d_Z1, b, H1, F, d_P + ob1);
            relu_kernel<<<(b * H1 + TB - 1) / TB, TB>>>(d_Z1, d_A1, b * H1);
            gemm<false, false>(d_A1, d_P + oW2, d_Z2, b, H2, H1, d_P + ob2);
            relu_kernel<<<(b * H2 + TB - 1) / TB, TB>>>(d_Z2, d_A2, b * H2);
            gemm<false, false>(d_A2, d_P + oW3, d_Z3, b, K, H2, d_P + ob3);
            softmax_loss_kernel<<<(b + 127) / 128, 128>>>(d_Z3, d_yb, d_cw, d_D3, d_loss, b, K);
            // backward
            gemm<true, false>(d_A2, d_D3, d_G + oW3, H2, K, b);
            colsum_kernel<<<K, 128>>>(d_D3, d_G + ob3, b, K);
            gemm<false, true>(d_D3, d_P + oW3, d_D2, b, H2, K, nullptr, d_Z2);
            gemm<true, false>(d_A1, d_D2, d_G + oW2, H1, H2, b);
            colsum_kernel<<<H2, 128>>>(d_D2, d_G + ob2, b, H2);
            gemm<false, true>(d_D2, d_P + oW2, d_D1, b, H1, H2, nullptr, d_Z1);
            gemm<true, false>(d_Xb, d_D1, d_G + oW1, F, H1, b);
            colsum_kernel<<<H1, 128>>>(d_D1, d_G + ob1, b, H1);
            sgd_kernel<<<(NP + TB - 1) / TB, TB>>>(d_P, d_V, d_G, NP, lr, mu);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        const double et = now_sec() - t0;
        float kms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&kms, ev0, ev1));
        train_total += et;
        kernel_total += kms / 1000.0;
        epochs_run++;

        double loss_sum = 0;
        CUDA_CHECK(cudaMemcpy(&loss_sum, d_loss, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(Pf.data(), d_P, NP * sizeof(float), cudaMemcpyDeviceToHost));
        unflatten(Pf, Ph);
        Metrics m = evaluate(Ph, va, K);
        std::printf("epoch %3d | train loss %.4f | val loss %.4f | val acc %.4f | val macro-F1 %.4f | %.2f s\n",
                    ep, loss_sum / N, m.loss, m.acc, m.macro_f1, et);
        if (lg) { std::fprintf(lg, "%d,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f\n", ep, loss_sum / N, m.loss, m.acc, m.macro_f1, et, kms / 1000.0); std::fflush(lg); }

        if (!a.ckpt.empty() && (ep % a.ckpt_every == 0 || ep == a.epochs)) {
            CUDA_CHECK(cudaMemcpy(Vf.data(), d_V, NP * sizeof(float), cudaMemcpyDeviceToHost));
            FILE* f = std::fopen(a.ckpt.c_str(), "wb");
            if (f) {
                int hdr[3] = {0x4D4C5043, ep, NP};
                std::fwrite(hdr, sizeof(int), 3, f);
                std::fwrite(Pf.data(), sizeof(float), NP, f);
                std::fwrite(Vf.data(), sizeof(float), NP, f);
                std::fclose(f);
                std::printf("  checkpoint saved at epoch %d -> %s\n", ep, a.ckpt.c_str());
            }
        }
    }
    if (lg) std::fclose(lg);
    const int ne = std::max(epochs_run, 1);
    std::printf("\nTRAINING TIME (epochs only, excludes validation): %.2f s total, %.2f s per epoch\n", train_total, train_total / ne);
    std::printf("KERNEL TIME (CUDA events, device work only): %.2f s total, %.2f s per epoch\n", kernel_total, kernel_total / ne);
    std::printf("H2D UPLOAD (once): %.3f s | END-TO-END (upload + epochs): %.2f s\n", t_upload, t_upload + train_total);

    print_metrics("final validation metrics", evaluate(Ph, va, K), K);
    if (a.eval_test) {
        if (!load_split(a.data_dir, "test", F, te)) { std::fprintf(stderr, "could not load test split\n"); return 1; }
        print_metrics("TEST metrics (final evaluation)", evaluate(Ph, te, K), K);
    }
    if (!a.save.empty() && save_params(a.save, Ph)) std::printf("saved weights to %s\n", a.save.c_str());
    return 0;
}
