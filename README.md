# GPU-Accelerated Intrusion-Detection Classifier

**A reproducible parallel MLP training framework using serial C++, OpenMP and CUDA**

Course: Parallel and Distributed Computing (Theory) — BSCYS-F-24-A, Air University (NCSA)
Instructor: Muhammad Ahmad Nawaz

## Team

| Name | Roll No. | Role |
|---|---|---|
| Marwah Ghazanfar | 241557 | **Team Lead** — serial MLP core, gradient checks, CUDA prototype, repo owner |
| Fiza Iman | 241555 | Dataset preparation, evaluation scripts, OpenMP parallelization, benchmarking |

## Summary

We train a compact multi-layer perceptron (78-64-32-K, ReLU, softmax, SGD with momentum) to classify
network flows from the public **CICIDS2017** dataset into benign and attack classes. The same model, data
split, initialization and hyperparameters are implemented in three backends:

1. **Serial C++** — correctness and timing reference
2. **OpenMP** — primary parallel deliverable (shared-memory CPU)
3. **CUDA** — secondary path, run on a Google Colab NVIDIA T4

The goal is **not** a new detection algorithm. It is to measure how parallel execution changes training
time (speedup, efficiency, scaling) while keeping results numerically consistent and predictive quality
comparable.

## Problem and computational intensity

Training repeats forward propagation, backpropagation and an update for every mini-batch of every epoch,
dominated by dense matrix multiplication. Rough estimate (to be confirmed by measurement): ~7,300
parameters, ~43,000 FLOP per sample, ~80 GFLOP per epoch on ~1.9M training samples, ~2.4 TFLOP per
30-epoch run. See `docs/Phase0_Scope_Pack.docx` for details.

## Repository layout

```
data/          git-ignored; download script and checksums only
preprocess/    Python: cleaning, duplicate removal, split, scaling, binary export
src/serial/    serial C++ reference MLP
src/openmp/    OpenMP parallel MLP
src/cuda/      CUDA MLP (Colab T4)
bench/         benchmark scripts and run manifests
results/       CSV timing logs, metrics, plots
docs/          scope pack, proposal, final report
```

## Build and run (planned — filled in as each phase is completed)

```bash
# Serial
g++ -O3 -march=native -std=c++17 src/serial/*.cpp -o build/mlp_serial

# OpenMP
g++ -O3 -march=native -fopenmp -std=c++17 src/openmp/*.cpp -o build/mlp_omp
OMP_NUM_THREADS=4 ./build/mlp_omp --config bench/config.json

# CUDA (Colab T4)
nvcc -O3 -arch=sm_75 src/cuda/*.cu -o build/mlp_cuda
```

## Dataset

CICIDS2017 (`MachineLearningCSV` files) from the Canadian Institute for Cybersecurity:
https://www.unb.ca/cic/datasets/ids-2017.html

Data files are **not committed**. See `data/README.md` for the download steps and checksum verification.

## Experiment plan (summary)

- Split: stratified 70/15/15 (train/val/test); scaling fitted on training data only
- Threads: 1, 2, 4, 8 (bounded by the VM's cores); batch sizes 128, 512, 2048; ~30 epochs
- 1 warm-up + at least 5 measured repetitions per configuration
- Metrics: training time, per-epoch time, speedup S(p)=T(1)/T(p), efficiency E(p)=S(p)/p;
  accuracy, macro-F1, per-class recall, confusion matrix; GPU kernel-only vs end-to-end time

## Reproducibility

Every run stores a manifest: dataset hash, split id, feature list, label mapping, compiler flags,
CPU/GPU model, thread count, batch size, learning rate and seed. Backends are compared within documented
floating-point tolerances (not bitwise identity).

## Project status

| Phase | Status | Tag |
|---|---|---|
| Phase 0 – Scope & strategy | Done | `v0.0-scope` |
| Preprocessing pipeline | Planned | |
| Serial baseline | Planned | `v0.1-serial` |
| OpenMP version | Planned | `v0.2-openmp` |
| CUDA version | Planned | `v0.3-cuda` |
| Final report | Planned | `v1.0-final` |

## Branching and tagging

See [`docs/BRANCHING.md`](docs/BRANCHING.md).

## Ethics and originality

- Offline training/evaluation prototype only; no scanning, attacking or testing of third-party systems.
- Only the public CICIDS2017 benchmark is used; any extra traffic is generated in a team-owned isolated lab.
- Core MLP code is written by the team without deep-learning frameworks. External sources are cited.
- AI assistance was used in a limited supporting role (wording, concept clarification, diagram formatting);
  the team is responsible for all design, code and results.

## Reference

[1] I. Sharafaldin, A. H. Lashkari and A. A. Ghorbani, "Toward Generating a New Intrusion Detection
Dataset and Intrusion Traffic Characterization," ICISSP, 2018, pp. 108-116.
