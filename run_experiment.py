#!/usr/bin/env python3
"""
Run DMD identification and verification experiment.

Computes a score matrix between query and gallery templates (.pkl), then derives:
  - Identification metrics: CMC curve, Rank-1
  - Verification metrics:   ROC curve, TAR@FAR, EER

Identity is determined by (dataset_alias, subject_id), where:
  - dataset_alias = first token of filename before '_' (e.g. sd258_000_... -> "sd258")
  - subject_id    = level-1 subdirectory name (e.g. 000, 001, ...)

Usage:
  python run_experiment.py \\
    --queries-dir   /data/sd258/templates/dmd  --query-regex  ".*-00-.*" \\
    --gallery-dir   /data/sd258/templates/dmd  --gallery-regex ".*-01-.*" \\
    --distractors   /data/sd27/templates/dmd \\
    --output-dir    results/sd258_exp1 \\
    --device cuda --batch-size 256
"""

import argparse
import csv
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_curve
from tqdm import tqdm

import dmd

# ------------------------------------------------------------------
# Template collection
# ------------------------------------------------------------------

def collect_templates(root_dir, filter_regex=None, filter_list=None):
    """Return sorted list of .pkl paths under root_dir, with optional filters."""
    root = Path(root_dir)
    files = sorted(f for f in root.rglob("*.pkl"))

    if filter_regex:
        pattern = re.compile(filter_regex)
        files = [f for f in files if pattern.search(str(f.relative_to(root)))]

    if filter_list:
        with open(filter_list) as fh:
            allowed = set(line.strip() for line in fh if line.strip())
        files = [
            f for f in files
            if str(f.relative_to(root)) in allowed or f.name in allowed
        ]

    return files


def identity_key(pkl_path, root_dir):
    """Return (dataset_alias, subject_id) for a template path.

    dataset_alias = first '_'-delimited token of the filename stem
    subject_id    = name of the level-1 subdirectory relative to root_dir
    """
    rel = pkl_path.relative_to(root_dir)
    subject_id = rel.parts[0]
    dataset_alias = pkl_path.stem.split("_")[0]
    return (dataset_alias, subject_id)


# ------------------------------------------------------------------
# Template reconstruction
# ------------------------------------------------------------------

def pkl_to_dmd_template(data):
    """Reconstruct a DMD-compatible template dict from the standard .pkl format.

    .pkl stores:
        embeddings (N, 768)  -- already flat, stored as-is
        mask       (N, 768)  -- expanded via repeat×12; recover with [:, ::12]
        minutiae   (N, 4)    -- [x, y, angle_ccw, quality] in .min convention

    DMD matcher expects:
        feature    (N, 768)  -- direct
        mask       (N, 64)   -- original foreground mask
        mnt        (1, N, 3) -- [x, y, angle_cw] (clockwise, for lsar_score_torchB)
    """
    feature = torch.from_numpy(data["embeddings"]).float()          # (N, 768)
    mask    = torch.from_numpy(data["mask"][:, ::12]).float()        # (N, 64)

    minutiae   = data["minutiae"].astype(np.float32)                  # (N, 4)
    angle_cw   = (360.0 - minutiae[:, 2]) % 360.0                    # CCW → CW
    mnt_arr    = np.column_stack([minutiae[:, 0], minutiae[:, 1], angle_cw])
    mnt        = torch.from_numpy(mnt_arr).float().unsqueeze(0)      # (1, N, 3)

    return {"feature": feature, "mask": mask, "mnt": mnt}


# ------------------------------------------------------------------
# Genuine mask
# ------------------------------------------------------------------

def build_genuine_mask(query_keys, gallery_keys, query_paths, gallery_paths):
    """Return bool array (Q, G). True where identity matches and paths differ."""
    Q, G = len(query_keys), len(gallery_keys)
    mask = np.zeros((Q, G), dtype=bool)
    for i, (qk, qp) in enumerate(zip(query_keys, query_paths)):
        for j, (gk, gp) in enumerate(zip(gallery_keys, gallery_paths)):
            mask[i, j] = (qk == gk) and (qp != gp)
    return mask


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def compute_cmc(scores, genuine_mask, max_rank=20):
    """Compute CMC curve.

    For each query, finds the best rank (lowest 0-indexed position)
    among all genuine gallery items. Multiple genuines: use best rank.
    Queries with no genuine are excluded from the denominator.

    Returns:
        cmc           : np.ndarray of shape (max_rank,), values in [0, 1]
        n_valid       : int, number of queries that had at least one genuine
    """
    Q, G = scores.shape
    max_rank = min(max_rank, G)

    # rank_matrix[i, j] = 0-based rank of gallery item j for query i
    rank_matrix = np.argsort(np.argsort(-scores, axis=1), axis=1)

    # For each query, best rank among genuines (G = worst sentinel if no genuine)
    rank_of_genuines = np.where(genuine_mask, rank_matrix, G)
    best_ranks = rank_of_genuines.min(axis=1)   # (Q,)

    valid = np.any(genuine_mask, axis=1)        # queries that have at least one genuine
    n_valid = int(valid.sum())

    if n_valid == 0:
        return np.zeros(max_rank), 0

    cmc = np.array([
        np.sum((best_ranks <= k) & valid) / n_valid
        for k in range(max_rank)
    ])
    return cmc, n_valid


def compute_verification(scores, genuine_mask):
    """Compute ROC curve, EER and TAR@FAR from the score matrix.

    Returns:
        fpr, tpr, thresholds : arrays from sklearn.metrics.roc_curve
        eer                  : float
        tar_at_far           : dict {0.001: float, 0.01: float}
    """
    genuine_scores  = scores[genuine_mask]
    impostor_scores = scores[~genuine_mask]

    y_true  = np.concatenate([np.ones(len(genuine_scores)),
                               np.zeros(len(impostor_scores))])
    y_score = np.concatenate([genuine_scores, impostor_scores])

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr

    # EER: point where |FPR - FNR| is smallest
    eer_idx = int(np.argmin(np.abs(fpr - fnr)))
    eer     = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)

    def _tar_at_far(target):
        valid = fpr <= target
        return float(tpr[valid][-1]) if valid.any() else 0.0

    tar_at_far = {0.001: _tar_at_far(0.001), 0.01: _tar_at_far(0.01)}

    return fpr, tpr, thresholds, eer, tar_at_far


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="DMD identification / verification experiment."
    )
    p.add_argument("--queries-dir",    required=True,
                   help="Root directory of query templates (.pkl)")
    p.add_argument("--gallery-dir",    required=True,
                   help="Root directory of gallery templates (.pkl)")
    p.add_argument("--query-regex",    default=None,
                   help="Regex filter applied to query relative paths")
    p.add_argument("--query-list",     default=None,
                   help="Text file with query relative paths (one per line)")
    p.add_argument("--gallery-regex",  default=None,
                   help="Regex filter applied to gallery relative paths")
    p.add_argument("--gallery-list",   default=None,
                   help="Text file with gallery relative paths (one per line)")
    p.add_argument("--distractors",    nargs="*", default=[], metavar="DIR",
                   help="Extra gallery directories (no filter; repetible)")
    p.add_argument("--output-dir",     required=True,
                   help="Directory where results are saved")
    p.add_argument("--device",         default="cuda",
                   help="Torch device for matching (default: cuda)")
    p.add_argument("--batch-size",     type=int, default=256,
                   help="Batch size for identify() (default: 256)")
    p.add_argument("--skip-matching",  action="store_true",
                   help="Skip score computation; load scores.npy from output-dir")
    return p.parse_args()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args   = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    out    = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Collect files -----------------------------------------------
    query_root   = Path(args.queries_dir)
    gallery_root = Path(args.gallery_dir)

    query_files   = collect_templates(query_root,   args.query_regex,   args.query_list)
    gallery_files = collect_templates(gallery_root, args.gallery_regex, args.gallery_list)

    # Distractors: (path, their root_dir)
    distractor_items = []
    for dist_dir in args.distractors:
        dist_root = Path(dist_dir)
        for f in collect_templates(dist_root):
            distractor_items.append((f, dist_root))

    print(f"Queries:     {len(query_files)}")
    print(f"Gallery:     {len(gallery_files)}")
    print(f"Distractors: {len(distractor_items)}")

    if not query_files:
        print("Error: no query templates found.")
        sys.exit(1)
    if not gallery_files and not distractor_items:
        print("Error: no gallery templates found.")
        sys.exit(1)

    # ---- Build full gallery (primary + distractors) ------------------
    all_gallery_paths = gallery_files + [f for f, _ in distractor_items]
    all_gallery_roots = (
        [gallery_root] * len(gallery_files)
        + [root for _, root in distractor_items]
    )

    # ---- Validate: no symmetric overlap ------------------------------
    query_set   = set(query_files)
    gallery_set = set(gallery_files)
    overlap = query_set & gallery_set
    if overlap:
        print(f"Error: {len(overlap)} file(s) appear in both queries and gallery:")
        for p in sorted(overlap):
            print(f"  {p}")
        sys.exit(1)

    # ---- Identity keys -----------------------------------------------
    query_keys   = [identity_key(f, query_root) for f in query_files]
    gallery_keys = [
        identity_key(f, root)
        for f, root in zip(all_gallery_paths, all_gallery_roots)
    ]

    # ---- Genuine mask ------------------------------------------------
    print("Building genuine mask...")
    genuine_mask = build_genuine_mask(
        query_keys, gallery_keys, query_files, all_gallery_paths
    )

    # ---- Warn: multiple genuines per query ---------------------------
    multi = [
        (query_files[i], [all_gallery_paths[j] for j in np.where(genuine_mask[i])[0]])
        for i in range(len(query_files))
        if genuine_mask[i].sum() > 1
    ]
    if multi:
        print(f"Warning: {len(multi)} query(ies) have multiple genuine gallery matches:")
        for q, gs in multi:
            rel_q = q.relative_to(query_root)
            print(f"  Query: {rel_q}")
            for g in gs:
                print(f"    → {g}")

    no_genuine = int((~np.any(genuine_mask, axis=1)).sum())
    if no_genuine:
        print(f"Warning: {no_genuine} query(ies) have no genuine match (excluded from CMC/EER).")

    Q = len(query_files)
    G = len(all_gallery_paths)

    # ---- Score matrix ------------------------------------------------
    scores_path = out / "scores.npy"

    if args.skip_matching:
        if not scores_path.exists():
            print(f"Error: --skip-matching requested but {scores_path} not found.")
            sys.exit(1)
        print(f"Loading pre-computed scores from {scores_path}")
        scores = np.load(scores_path)
        if scores.shape != (Q, G):
            print(f"Error: loaded scores shape {scores.shape} != expected ({Q}, {G}).")
            sys.exit(1)
    else:
        print("Loading templates...")
        query_templates = [
            pkl_to_dmd_template(pickle.load(open(f, "rb")))
            for f in tqdm(query_files, desc="  Queries")
        ]
        gallery_templates = [
            pkl_to_dmd_template(pickle.load(open(f, "rb")))
            for f in tqdm(all_gallery_paths, desc="  Gallery")
        ]

        print("Computing score matrix...")
        matcher = dmd.DmdMatcher()
        scores  = matcher.identify(
            query_templates, gallery_templates,
            device=device, batch_size=args.batch_size
        )  # (Q, G) numpy float64

        np.save(scores_path, scores.astype(np.float32))
        print(f"  Saved → {scores_path}")

    # ---- Save index CSVs and genuine mask ----------------------------
    with open(out / "query_index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "dataset_alias", "subject_id", "path"])
        for i, (key, path) in enumerate(zip(query_keys, query_files)):
            w.writerow([i, key[0], key[1], str(path)])

    with open(out / "gallery_index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["j", "dataset_alias", "subject_id", "path"])
        for j, (key, path) in enumerate(zip(gallery_keys, all_gallery_paths)):
            w.writerow([j, key[0], key[1], str(path)])

    np.save(out / "genuine_mask.npy", genuine_mask)

    # ---- Metrics: identification (CMC) -------------------------------
    print("Computing identification metrics (CMC)...")
    cmc, n_valid = compute_cmc(scores, genuine_mask, max_rank=20)

    with open(out / "cmc.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "cmc"])
        for k, v in enumerate(cmc, 1):
            w.writerow([k, round(float(v), 6)])

    # ---- Metrics: verification (ROC) ---------------------------------
    print("Computing verification metrics (ROC)...")
    fpr, tpr, thresholds, eer, tar_at_far = compute_verification(scores, genuine_mask)

    # sklearn roc_curve: len(thresholds) == len(fpr) - 1; first point is (0, 0)
    with open(out / "roc.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fpr", "tpr", "threshold"])
        w.writerow([0.0, 0.0, ""])
        for fp, tp, th in zip(fpr[1:], tpr[1:], thresholds):
            w.writerow([round(float(fp), 8), round(float(tp), 8), round(float(th), 8)])

    # ---- Summary metrics.json ----------------------------------------
    metrics = {
        "n_queries":          Q,
        "n_gallery":          G,
        "n_gallery_primary":  len(gallery_files),
        "n_distractors":      len(distractor_items),
        "n_valid_queries":    n_valid,
        "rank1":              round(float(cmc[0]),    4),
        "rank5":              round(float(cmc[4]),    4) if len(cmc) >= 5  else None,
        "rank10":             round(float(cmc[9]),    4) if len(cmc) >= 10 else None,
        "eer":                round(eer,              4),
        "tar_at_far_0.1pct":  round(tar_at_far[0.001], 4),
        "tar_at_far_1.0pct":  round(tar_at_far[0.01],  4),
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- Print summary -----------------------------------------------
    print()
    print("=" * 40)
    print(f"Queries      : {Q}")
    print(f"Gallery      : {G}  ({len(gallery_files)} primary + {len(distractor_items)} distractors)")
    print(f"Valid queries: {n_valid}")
    print("-" * 40)
    print(f"Rank-1       : {cmc[0]:.4f}")
    if len(cmc) >= 5:
        print(f"Rank-5       : {cmc[4]:.4f}")
    if len(cmc) >= 10:
        print(f"Rank-10      : {cmc[9]:.4f}")
    print(f"EER          : {eer:.4f}")
    print(f"TAR@FAR=0.1% : {tar_at_far[0.001]:.4f}")
    print(f"TAR@FAR=1.0% : {tar_at_far[0.01]:.4f}")
    print("=" * 40)
    print(f"Results → {out}")


if __name__ == "__main__":
    main()
