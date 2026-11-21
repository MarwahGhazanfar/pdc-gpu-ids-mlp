#!/usr/bin/env python3
"""Summarise the thread-scaling benchmark: mean/std time, speedup, efficiency, Amdahl estimate, plots."""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "results/bench_threads.csv"
prefix = sys.argv[2] if len(sys.argv) > 2 else "results/bench"
df = pd.read_csv(src)
df = df[df.rep >= 1]                       # drop warm-up round
g = df.groupby(["backend", "threads"]).agg(
    runs=("train_seconds", "count"), mean_s=("train_seconds", "mean"), std_s=("train_seconds", "std"),
    min_s=("train_seconds", "min"), max_s=("train_seconds", "max"),
    sec_per_epoch=("sec_per_epoch", "mean"), val_macro_f1=("val_macro_f1", "mean"),
    f1_min=("val_macro_f1", "min"), f1_max=("val_macro_f1", "max")).reset_index()

Ts = g[g.backend == "serial"].mean_s.iloc[0]
omp = g[g.backend == "omp"].sort_values("threads").copy()
T1 = omp[omp.threads == 1].mean_s.iloc[0]
omp["speedup_vs_serial"] = Ts / omp.mean_s
omp["speedup_S(p)"] = T1 / omp.mean_s            # S(p) = T(1)/T(p), T(1) = OpenMP on 1 thread
omp["efficiency_E(p)"] = omp["speedup_S(p)"] / omp.threads
omp["amdahl_serial_frac"] = [((1 / s - 1 / p) / (1 - 1 / p)) if p > 1 else np.nan
                             for s, p in zip(omp["speedup_S(p)"], omp.threads)]
omp["cv_%"] = 100 * omp.std_s / omp.mean_s

pd.set_option("display.width", 200)
print(f"Serial baseline: {Ts:.2f} s (mean of {g[g.backend=='serial'].runs.iloc[0]} runs, "
      f"std {g[g.backend=='serial'].std_s.iloc[0]:.2f})")
cols = ["threads", "runs", "mean_s", "std_s", "cv_%", "sec_per_epoch", "speedup_vs_serial",
        "speedup_S(p)", "efficiency_E(p)", "amdahl_serial_frac", "val_macro_f1"]
print(omp[cols].round(3).to_string(index=False))
out = omp[cols].copy()
out.insert(0, "serial_mean_s", Ts)
out.to_csv(prefix + "_summary.csv", index=False)

p = omp.threads.to_numpy()
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(p, p, "k--", label="ideal")
ax[0].errorbar(p, omp["speedup_S(p)"], yerr=T1 * omp.std_s / omp.mean_s ** 2, marker="o", capsize=3, label="OpenMP S(p)")
ax[0].set_xlabel("threads"); ax[0].set_ylabel("speedup  T(1)/T(p)"); ax[0].set_title("Speedup"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[0].set_xticks(p)
ax[1].plot(p, omp["efficiency_E(p)"], marker="o")
ax[1].axhline(1, color="k", ls="--"); ax[1].set_ylim(0, 1.1)
ax[1].set_xlabel("threads"); ax[1].set_ylabel("efficiency  S(p)/p"); ax[1].set_title("Parallel efficiency"); ax[1].grid(alpha=.3)
ax[1].set_xticks(p)
plt.tight_layout(); plt.savefig(prefix + "_speedup_efficiency.png", dpi=150); plt.close()

labels = ["serial"] + [f"OMP {t}" for t in p]
means = [Ts] + omp.mean_s.tolist()
stds = [g[g.backend == "serial"].std_s.iloc[0]] + omp.std_s.tolist()
plt.figure(figsize=(6, 4))
plt.bar(labels, means, yerr=stds, capsize=3)
plt.ylabel("training time (s)"); plt.title("Training time (mean of runs, error bar = std)")
plt.tight_layout(); plt.savefig(prefix + "_time.png", dpi=150); plt.close()
print(f"\nSaved {prefix}_summary.csv, {prefix}_speedup_efficiency.png, {prefix}_time.png")
