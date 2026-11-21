#!/usr/bin/env python3
"""Preprocess CICIDS2017 MachineLearningCSV files for the serial / OpenMP / CUDA MLP.

Steps: read -> map labels to 6 classes -> drop invalid rows (NaN/inf) -> drop exact duplicates
       -> stratified 70/15/15 split -> standardize with TRAIN statistics only -> export float32 binaries.
"""
import argparse, glob, hashlib, json, os, sys, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CLASSES = ["Benign", "DoS/DDoS", "PortScan", "Brute-force", "Web Attack", "Other"]
K = len(CLASSES)


def map_label(raw):
    s = raw.strip()
    if s == "BENIGN":
        return 0
    if s.startswith("DoS") or s == "DDoS":
        return 1
    if s == "PortScan":
        return 2
    if s in ("FTP-Patator", "SSH-Patator"):
        return 3
    if s.startswith("Web Attack"):
        return 4
    return 5  # Bot, Infiltration, Heartbleed, or anything unseen


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/MachineLearningCVE")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--manifest", default="results/preprocess_manifest.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clip", type=float, default=10.0, help="clip standardized values to [-clip, clip]; 0 = off")
    args = ap.parse_args()
    t0 = time.time()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.csv")))
    if not files:
        sys.exit("No CSV files found in " + args.data_dir)

    feature_names, X_parts, y_parts, file_info = None, [], [], {}
    for f in files:
        df = pd.read_csv(f, encoding="latin-1", low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        names = list(df.columns[:-1])
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            sys.exit("Column mismatch in " + f)
        y = df.iloc[:, -1].astype(str).map(map_label).to_numpy(dtype=np.int8)
        feats = df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce")
        with np.errstate(over="ignore"):
            X = feats.to_numpy(dtype=np.float32)
        X_parts.append(X)
        y_parts.append(y)
        file_info[os.path.basename(f)] = {"rows": int(len(y)), "sha256": sha256_of(f)}
        print(f"read {os.path.basename(f)}: {len(y)} rows")
        del df, feats
    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    del X_parts, y_parts
    n_raw = int(len(y))
    raw_by_class = np.bincount(y, minlength=K)

    # 1) invalid rows (NaN or +/-inf in any feature)
    bad = ~np.isfinite(X).all(axis=1)
    n_invalid = int(bad.sum())
    invalid_by_class = np.bincount(y[bad], minlength=K)
    X, y = X[~bad], y[~bad]

    # 2) exact duplicates (features + class), keep first occurrence
    tmp = pd.DataFrame(X)
    tmp["label"] = y
    dup = tmp.duplicated(keep="first").to_numpy()
    del tmp
    n_dup = int(dup.sum())
    dup_by_class = np.bincount(y[dup], minlength=K)
    X, y = X[~dup], y[~dup]
    n_clean = int(len(y))

    # 3) stratified 70/15/15 split
    idx = np.arange(n_clean)
    tr, rest = train_test_split(idx, test_size=0.30, stratify=y, random_state=args.seed)
    va, te = train_test_split(rest, test_size=0.50, stratify=y[rest], random_state=args.seed)

    # 4) standardization fitted on TRAIN only
    Xtr = X[tr]
    mean = Xtr.mean(axis=0, dtype=np.float64)
    std = Xtr.std(axis=0, dtype=np.float64)
    del Xtr
    constant = [feature_names[i] for i in np.where(std < 1e-12)[0]]
    std[std < 1e-12] = 1.0

    def prep(ix):
        Z = (X[ix].astype(np.float64) - mean) / std
        if args.clip > 0:
            np.clip(Z, -args.clip, args.clip, out=Z)
        return Z.astype(np.float32)

    # 5) export
    os.makedirs(args.out_dir, exist_ok=True)
    split_info = {}
    for name, ix in (("train", tr), ("val", va), ("test", te)):
        Z = prep(ix)
        yy = y[ix].astype(np.int32)
        Z.tofile(os.path.join(args.out_dir, f"X_{name}.bin"))
        yy.tofile(os.path.join(args.out_dir, f"y_{name}.bin"))
        split_info[name] = {"rows": int(len(ix)), "per_class": np.bincount(yy, minlength=K).tolist()}
        print(f"{name}: X {Z.shape} float32, y {yy.shape} int32")
        del Z

    ntr = np.array(split_info["train"]["per_class"], dtype=np.float64)
    class_weights = (ntr.sum() / (K * np.maximum(ntr, 1))).tolist()

    manifest = {
        "seed": args.seed, "clip": args.clip, "classes": CLASSES,
        "n_features": len(feature_names), "feature_names": feature_names,
        "raw_rows": n_raw, "raw_per_class": raw_by_class.tolist(),
        "invalid_rows_removed": n_invalid, "invalid_per_class": invalid_by_class.tolist(),
        "duplicate_rows_removed": n_dup, "duplicate_per_class": dup_by_class.tolist(),
        "clean_rows": n_clean, "splits": split_info,
        "constant_features": constant, "class_weights_train": class_weights,
        "feature_mean": mean.tolist(), "feature_std": std.tolist(),
        "binary_format": "X_*.bin float32 row-major (rows x n_features); y_*.bin int32 class ids",
        "source_files": file_info,
    }
    os.makedirs(os.path.dirname(args.manifest) or ".", exist_ok=True)
    for path in (os.path.join(args.out_dir, "manifest.json"), args.manifest):
        with open(path, "w") as fh:
            json.dump(manifest, fh, indent=2)

    print("\n=== SUMMARY ===")
    print(f"raw rows: {n_raw} | invalid removed: {n_invalid} | duplicates removed: {n_dup} | clean rows: {n_clean}")
    print(f"features: {len(feature_names)} | constant features: {len(constant)}")
    print(f"{'class':<12}{'raw':>10}{'train':>10}{'val':>9}{'test':>9}")
    for k in range(K):
        print(f"{CLASSES[k]:<12}{raw_by_class[k]:>10}{split_info['train']['per_class'][k]:>10}"
              f"{split_info['val']['per_class'][k]:>9}{split_info['test']['per_class'][k]:>9}")
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
