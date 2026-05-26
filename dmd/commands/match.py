"""
dmd match — Run DMD identification and verification experiment.

Computes a score matrix between query and gallery templates (.pkl), then derives:
  - Identification metrics: CMC curve, Rank-1
  - Verification metrics:   ROC curve, TAR@FAR, EER

Identity is determined by (dataset_alias, subject_id), where:
  - dataset_alias = first '_'-delimited token of the filename stem
  - subject_id    = level-1 subdirectory name relative to the root dir
"""

import argparse
import csv
import json
import pickle
import re
import sys
from pathlib import Path

from collections import Counter, defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_curve
from tqdm import tqdm

import dmd


# ---------------------------------------------------------------------------
# Template collection
# ---------------------------------------------------------------------------

def _collect_templates(root_dir, filter_regex=None, filter_list=None):
    """Return sorted list of .pkl paths under root_dir, with optional filters."""
    root  = Path(root_dir)
    # os.walk(followlinks=True) handles symlinked subtrees (e.g. FVCL400 layout).
    import os
    found = []
    for r, dirs, fnames in os.walk(root, followlinks=True):
        for fn in fnames:
            if fn.endswith(".pkl"):
                found.append(Path(r) / fn)
    files = sorted(found)

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


def _identity_key(pkl_path, root_dir, identity_regex=None):
    """Return (dataset_alias, subject_id) for a template path.

    dataset_alias = first '_'-delimited token of the filename stem
    subject_id    = level-1 subdirectory name relative to root_dir, OR
                    capture group 1 of `identity_regex` matched against the
                    relative path (when set). The regex form is required for
                    datasets like SD302 where the pasta hierarchy does not
                    preserve finger identity (multiple subject_id folders may
                    point to the same physical finger; identity is encoded in
                    the filename instead).
    """
    rel = pkl_path.relative_to(root_dir)
    dataset_alias = pkl_path.stem.split("_")[0]
    if identity_regex is not None:
        m = identity_regex.search(str(rel))
        if m is None or not m.groups():
            raise ValueError(
                f"identity_regex {identity_regex.pattern!r} did not match {rel!r} "
                "(expected group 1 = identity)"
            )
        subject_id = m.group(1)
    else:
        subject_id = rel.parts[0]
    return (dataset_alias, subject_id)


# ---------------------------------------------------------------------------
# Template reconstruction
# ---------------------------------------------------------------------------

def _pkl_to_dmd_template(data, min_quality=0):
    """Reconstruct a DMD-compatible template dict from the standard .pkl format.

    .pkl stores:
        embeddings (N, 768) — flat features
        mask       (N, 64)  — raw foreground mask (current format), OR
                   (N, 768) — repeat×12 of the (N, 64) mask (legacy format)
        minutiae   (N, 4)   — [x, y, angle_ccw, quality] in .min convention

    Args:
        data:        loaded .pkl dict
        min_quality: drop minutiae (and matching embeddings/mask rows) with quality < N

    DMD matcher expects:
        feature (N, 768)  — direct
        mask    (N, 64)   — original foreground mask
        mnt     (1, N, 3) — [x, y, angle_cw] (clockwise, for lsar_score_torchB)
    """
    embeddings = data["embeddings"]
    mask_arr   = data["mask"]
    minutiae   = data["minutiae"]

    if min_quality > 0 and len(minutiae) > 0:
        keep = minutiae[:, 3] >= min_quality
        embeddings = embeddings[keep]
        mask_arr   = mask_arr[keep]
        minutiae   = minutiae[keep]

    # Recover the raw (N, 64) mask. Legacy .pkl files persist it expanded to
    # (N, 768) via np.repeat(...×12) — take every 12th column. New files
    # already store the (N, 64) form.
    if mask_arr.ndim == 2 and mask_arr.shape[1] == 768:
        mask_arr = mask_arr[:, ::12]

    feature = torch.from_numpy(embeddings).float()              # (N, 768)
    mask    = torch.from_numpy(mask_arr).float()                # (N, 64)

    minutiae  = minutiae.astype(np.float32)                     # (N, 4)
    angle_cw  = (360.0 - minutiae[:, 2]) % 360.0                # CCW → CW
    mnt_arr   = np.column_stack([minutiae[:, 0], minutiae[:, 1], angle_cw])
    mnt       = torch.from_numpy(mnt_arr).float().unsqueeze(0)  # (1, N, 3)

    return {"feature": feature, "mask": mask, "mnt": mnt}


# ---------------------------------------------------------------------------
# Genuine mask
# ---------------------------------------------------------------------------

def _build_genuine_mask(query_keys, gallery_keys, query_paths, gallery_paths):
    """Return bool array (Q, G). True where identities match and paths differ."""
    Q, G = len(query_keys), len(gallery_keys)
    mask = np.zeros((Q, G), dtype=bool)

    gal_by_key = defaultdict(list)
    for j, (gk, gp) in enumerate(zip(gallery_keys, gallery_paths)):
        gal_by_key[gk].append((j, gp))

    for i, (qk, qp) in enumerate(zip(query_keys, query_paths)):
        for j, gp in gal_by_key.get(qk, []):
            if qp != gp:
                mask[i, j] = True

    return mask


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_cmc(scores, genuine_mask, max_rank=20):
    """Compute CMC curve.

    For each query, finds the best rank among all genuine gallery items.
    Queries with no genuine match are excluded from the denominator.

    Returns:
        cmc     : np.ndarray (max_rank,), values in [0, 1]
        n_valid : int, number of queries with at least one genuine match
    """
    Q, G     = scores.shape
    max_rank = min(max_rank, G)

    rank_matrix      = np.argsort(np.argsort(-scores, axis=1), axis=1)
    rank_of_genuines = np.where(genuine_mask, rank_matrix, G)
    best_ranks       = rank_of_genuines.min(axis=1)

    valid   = np.any(genuine_mask, axis=1)
    n_valid = int(valid.sum())

    if n_valid == 0:
        return np.zeros(max_rank), 0

    cmc = np.array([
        np.sum((best_ranks <= k) & valid) / n_valid
        for k in range(max_rank)
    ])
    return cmc, n_valid


def _compute_verification(scores, genuine_mask, threshold_targets=(0.0001, 0.001, 0.01)):
    """Compute ROC curve, EER, TAR@FAR, and operating thresholds.

    Returns:
        fpr, tpr, thresholds : from sklearn.metrics.roc_curve
        eer                  : float
        threshold_eer        : float — threshold at the EER operating point
        tar_at_far           : dict {far: tar}
        threshold_at_far     : dict {far: threshold} — threshold producing FAR ≤ target
    """
    genuine_scores  = scores[genuine_mask]
    impostor_scores = scores[~genuine_mask]

    y_true  = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    y_score = np.concatenate([genuine_scores, impostor_scores])

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr

    eer_idx       = int(np.argmin(np.abs(fpr - fnr)))
    eer           = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    threshold_eer = float(thresholds[eer_idx])

    def _tar_at_far(target):
        valid = fpr <= target
        if not valid.any():
            return 0.0, float("nan")
        idx = int(np.where(valid)[0][-1])
        return float(tpr[idx]), float(thresholds[idx])

    tar_at_far       = {}
    threshold_at_far = {}
    for tgt in threshold_targets:
        tar, thr = _tar_at_far(tgt)
        tar_at_far[tgt]       = tar
        threshold_at_far[tgt] = thr

    return fpr, tpr, thresholds, eer, threshold_eer, tar_at_far, threshold_at_far


def _score_stats(scores):
    """Return mean/std/min/max/p50/p95 for an array of scores."""
    if len(scores) == 0:
        return {k: float("nan") for k in ("mean", "std", "min", "max", "p50", "p95")}
    return {
        "mean": float(np.mean(scores)),
        "std":  float(np.std(scores)),
        "min":  float(np.min(scores)),
        "max":  float(np.max(scores)),
        "p50":  float(np.percentile(scores, 50)),
        "p95":  float(np.percentile(scores, 95)),
    }


def _load_custom_genuine_csv(csv_path, query_paths, gallery_paths, query_root, gallery_root):
    """Build genuine mask from a CSV with columns (query_rel_path, gallery_rel_path, is_genuine).

    Paths in the CSV are interpreted relative to the queries/gallery roots.
    Pairs not listed default to False.
    """
    Q, G = len(query_paths), len(gallery_paths)
    mask = np.zeros((Q, G), dtype=bool)

    q_index = {str(p.relative_to(query_root)): i for i, p in enumerate(query_paths)}
    g_index = {str(p.relative_to(gallery_root)): j for j, p in enumerate(gallery_paths)}

    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            q_key, g_key, is_gen = row[0].strip(), row[1].strip(), row[2].strip()
            if q_key not in q_index or g_key not in g_index:
                continue
            if is_gen.lower() in ("1", "true", "yes", "y", "t"):
                mask[q_index[q_key], g_index[g_key]] = True

    return mask


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv):
    p = argparse.ArgumentParser(
        prog="dmd match",
        description="Run DMD identification / verification experiment, or score "
                    "an explicit pair list (--pairs-list).",
    )
    p.add_argument("--queries-dir",   default=None,
                   help="Root directory of query templates (.pkl). Required unless --pairs-list is set.")
    p.add_argument("--gallery-dir",   default=None,
                   help="Root directory of gallery templates (.pkl). Required unless --pairs-list is set.")
    p.add_argument("--pairs-list",    default=None,
                   help="CSV with explicit pairs to score (columns: query_path, gallery_path "
                        "[, label]). Paths are resolved against --templates-dir (or --queries-dir "
                        "if --templates-dir is omitted). When set, only those pairs are scored; "
                        "no Q×G matrix, no CMC/ROC are computed.")
    p.add_argument("--templates-dir", default=None,
                   help="Single root for resolving paths in --pairs-list. Defaults to --queries-dir.")
    p.add_argument("--save-pair-details", action="store_true",
                   help="In --pairs-list mode, also dump LSAR minutia correspondences per pair "
                        "to pair_details.pkl alongside pair_scores.csv.")
    p.add_argument("--match-files-dir", default=None,
                   help="In --pairs-list mode, write one self-contained .match text file per "
                        "pair under this directory (mirrors query path hierarchy, .pkl→.match). "
                        "Format: #MATCH SCORE/LABEL header lines, then "
                        "'#MMIN SCORE X1 Y1 THETA1 Q1 X2 Y2 THETA2 Q2' column row, then "
                        "one row per LSAR-selected minutia correspondence with the resolved "
                        "minutia values from both templates (CCW angles, .min convention).")
    p.add_argument("--query-regex",   default=None,
                   help="Regex filter applied to query relative paths")
    p.add_argument("--query-list",    default=None,
                   help="Text file with query relative paths (one per line)")
    p.add_argument("--gallery-regex", default=None,
                   help="Regex filter applied to gallery relative paths")
    p.add_argument("--gallery-list",  default=None,
                   help="Text file with gallery relative paths (one per line)")
    p.add_argument("--distractors",   nargs="*", default=[], metavar="DIR",
                   help="Extra gallery directories (no filter; repeatable)")
    p.add_argument("--max-distractors", type=int, default=None, metavar="N",
                   help="Randomly sample at most N distractors (seed=42 for reproducibility)")
    p.add_argument("--output-dir",    required=True,
                   help="Directory where results are saved")
    p.add_argument("--device",        default="cuda",
                   help="Torch device for matching (default: cuda)")
    p.add_argument("--batch-size",    type=int, default=256,
                   help="Batch size for identify() (default: 256)")
    p.add_argument("--skip-matching", action="store_true",
                   help="Skip score computation; load scores.npy from --output-dir")
    p.add_argument("--min-quality",   type=int, default=0,
                   help="Drop minutiae with quality < N when loading templates (default: 0)")
    p.add_argument("--genuine-mask-mode", choices=["auto", "fvc_loo", "custom_csv"], default="auto",
                   help="Genuine mask construction: auto (from filename), fvc_loo (silence multi-genuine warning), custom_csv (read --genuine-mask-csv)")
    p.add_argument("--genuine-mask-csv", default=None,
                   help="CSV with columns (query_rel_path, gallery_rel_path, is_genuine); used when --genuine-mask-mode=custom_csv")
    p.add_argument("--threshold-targets", default="0.0001,0.001,0.01",
                   help="Comma-separated FAR targets for threshold reporting (default: 0.0001,0.001,0.01)")
    p.add_argument("--identity-regex", default=None,
                   help="Regex with group 1 = identity, applied to relative path to derive subject_id "
                        "(replaces default of using level-1 directory). Useful for SD302 where pasta does not "
                        r"preserve finger identity (e.g. 'sd302_(\\d+_\\d+)[-_]')")
    return p.parse_args(argv)


def _load_template(pkl_path, min_quality=0):
    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh)
    return _pkl_to_dmd_template(data, min_quality=min_quality)


class MatchError(Exception):
    """Raised by match_dataset() for caller-facing errors (no templates, overlap, etc.)."""


def match_dataset(
    queries_dir,
    gallery_dir,
    output_dir,
    *,
    query_regex=None,
    query_list=None,
    gallery_regex=None,
    gallery_list=None,
    distractors=(),
    max_distractors=None,
    device="cuda",
    batch_size=256,
    skip_matching=False,
    min_quality=0,
    genuine_mask_mode="auto",
    genuine_mask_csv=None,
    threshold_targets=(0.0001, 0.001, 0.01),
    identity_regex=None,
):
    """Run an identification/verification experiment programmatically.

    Raises MatchError on caller-facing failures. Returns the metrics dict.
    """
    device = device if torch.cuda.is_available() else "cpu"
    out    = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    query_root   = Path(queries_dir)
    gallery_root = Path(gallery_dir)

    query_files   = _collect_templates(query_root,   query_regex,   query_list)
    gallery_files = _collect_templates(gallery_root, gallery_regex, gallery_list)

    distractor_items = []
    for dist_dir in distractors:
        dist_root = Path(dist_dir)
        for f in _collect_templates(dist_root):
            distractor_items.append((f, dist_root))

    n_distractors_total = len(distractor_items)
    if max_distractors is not None and n_distractors_total > max_distractors:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(n_distractors_total, size=max_distractors, replace=False)
        idx.sort()
        distractor_items = [distractor_items[i] for i in idx]
        print(f"Sampled {len(distractor_items)} / {n_distractors_total} distractors (seed=42)")
    elif max_distractors is not None:
        print(f"max_distractors {max_distractors} >= total {n_distractors_total}, using all.")

    print(f"Queries:     {len(query_files)}  ({query_root})")
    print(f"Gallery:     {len(gallery_files)}  ({gallery_root})")
    if distractor_items:
        dist_counts = Counter(str(root) for _, root in distractor_items)
        print(f"Distractors: {len(distractor_items)}")
        for d, c in sorted(dist_counts.items()):
            print(f"  {c:>6}  {d}")
    else:
        print(f"Distractors: 0")
    print(f"Total gallery: {len(gallery_files) + len(distractor_items)}")

    if not query_files:
        raise MatchError(f"no query templates found under {query_root}")
    if not gallery_files and not distractor_items:
        raise MatchError(f"no gallery templates found under {gallery_root}")

    all_gallery_paths = gallery_files + [f for f, _ in distractor_items]
    all_gallery_roots = (
        [gallery_root] * len(gallery_files)
        + [root for _, root in distractor_items]
    )

    overlap = set(query_files) & set(all_gallery_paths)
    if overlap:
        sample = sorted(overlap)[:10]
        more = f" ... and {len(overlap) - 10} more" if len(overlap) > 10 else ""
        raise MatchError(f"{len(overlap)} file(s) appear in both queries and gallery: {sample}{more}")

    id_re = re.compile(identity_regex) if identity_regex else None
    query_keys   = [_identity_key(f, query_root, id_re) for f in query_files]
    gallery_keys = [
        _identity_key(f, root, id_re)
        for f, root in zip(all_gallery_paths, all_gallery_roots)
    ]

    print(f"Building genuine mask (mode={genuine_mask_mode})...")
    if genuine_mask_mode == "custom_csv":
        if not genuine_mask_csv:
            raise MatchError("genuine_mask_mode=custom_csv requires genuine_mask_csv")
        if distractor_items:
            raise MatchError("genuine_mask_mode=custom_csv is not compatible with distractors")
        genuine_mask = _load_custom_genuine_csv(
            genuine_mask_csv, query_files, all_gallery_paths,
            query_root, gallery_root,
        )
    else:
        genuine_mask = _build_genuine_mask(query_keys, gallery_keys, query_files, all_gallery_paths)

    multi = [
        (query_files[i], [all_gallery_paths[j] for j in np.where(genuine_mask[i])[0]])
        for i in range(len(query_files))
        if genuine_mask[i].sum() > 1
    ]
    if multi and genuine_mask_mode != "fvc_loo":
        print(f"Warning: {len(multi)} query(ies) have multiple genuine gallery matches:")
        for q, gs in multi:
            print(f"  Query: {q.relative_to(query_root)}")
            for g in gs:
                print(f"    → {g}")
    elif multi:
        avg = float(np.mean(genuine_mask.sum(axis=1)))
        print(f"FVC-LOO mode: avg genuine matches per query = {avg:.2f}")

    no_genuine = int((~np.any(genuine_mask, axis=1)).sum())
    if no_genuine:
        print(f"Warning: {no_genuine} query(ies) have no genuine match (excluded from CMC/EER).")

    Q = len(query_files)
    G = len(all_gallery_paths)
    scores_path = out / "scores.npy"

    if skip_matching:
        if not scores_path.exists():
            raise MatchError(f"skip_matching requested but {scores_path} not found")
        print(f"Loading pre-computed scores from {scores_path}")
        scores = np.load(scores_path)
        if scores.shape != (Q, G):
            raise MatchError(f"loaded scores shape {scores.shape} != expected ({Q}, {G})")
    else:
        print(f"Loading templates (min_quality={min_quality})...")
        # When queries and gallery point to the same files (FVC LOO), load once.
        same_set = query_files == all_gallery_paths
        query_templates = [
            _load_template(f, min_quality=min_quality)
            for f in tqdm(query_files, desc="  Queries")
        ]
        if same_set:
            gallery_templates = query_templates
        else:
            gallery_templates = [
                _load_template(f, min_quality=min_quality)
                for f in tqdm(all_gallery_paths, desc="  Gallery")
            ]

        print("Computing score matrix...")
        matcher = dmd.DmdMatcher()
        scores  = matcher.identify(
            query_templates, gallery_templates,
            device=device, batch_size=batch_size,
        )

        np.save(scores_path, scores.astype(np.float32))
        print(f"  Saved → {scores_path}")

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

    print("Computing identification metrics (CMC)...")
    cmc, n_valid = _compute_cmc(scores, genuine_mask, max_rank=20)

    with open(out / "cmc.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "cmc"])
        for k, v in enumerate(cmc, 1):
            w.writerow([k, round(float(v), 6)])

    print("Computing verification metrics (ROC)...")
    threshold_targets = tuple(threshold_targets)
    (fpr, tpr, thresholds, eer, threshold_eer,
     tar_at_far, threshold_at_far) = _compute_verification(scores, genuine_mask, threshold_targets)

    with open(out / "roc.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fpr", "tpr", "threshold"])
        w.writerow([0.0, 0.0, ""])
        for fp, tp, th in zip(fpr[1:], tpr[1:], thresholds):
            w.writerow([round(float(fp), 8), round(float(tp), 8), round(float(th), 8)])

    genuine_scores  = scores[genuine_mask]
    impostor_scores = scores[~genuine_mask]
    np.save(out / "genuine_scores.npy",  genuine_scores.astype(np.float32))
    np.save(out / "impostor_scores.npy", impostor_scores.astype(np.float32))

    metrics = {
        "n_queries":         Q,
        "n_gallery":         G,
        "n_gallery_primary": len(gallery_files),
        "n_distractors":     len(distractor_items),
        "n_valid_queries":   n_valid,
        "n_genuine_pairs":   int(genuine_mask.sum()),
        "n_impostor_pairs":  int((~genuine_mask).sum()),
        "min_quality":       min_quality,
        "genuine_mask_mode": genuine_mask_mode,
        "rank1":             round(float(cmc[0]), 4),
        "rank5":             round(float(cmc[4]), 4) if len(cmc) >= 5  else None,
        "rank10":            round(float(cmc[9]), 4) if len(cmc) >= 10 else None,
        "eer":               round(eer, 6),
        "threshold_eer":     round(threshold_eer, 6),
        "tar_at_far":        {f"{k:g}": round(v, 6) for k, v in tar_at_far.items()},
        "threshold_at_far":  {f"{k:g}": round(v, 6) for k, v in threshold_at_far.items()},
        "score_stats": {
            "genuine":  _score_stats(genuine_scores),
            "impostor": _score_stats(impostor_scores),
        },
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print()
    print("=" * 40)
    print(f"Queries      : {Q}")
    print(f"Gallery      : {G}  ({len(gallery_files)} primary + {len(distractor_items)} distractors)")
    print(f"Valid queries: {n_valid}")
    print(f"Genuine pairs: {metrics['n_genuine_pairs']}  Impostor pairs: {metrics['n_impostor_pairs']}")
    print("-" * 40)
    print(f"Rank-1        : {cmc[0]:.4f}")
    if len(cmc) >= 5:
        print(f"Rank-5        : {cmc[4]:.4f}")
    if len(cmc) >= 10:
        print(f"Rank-10       : {cmc[9]:.4f}")
    print(f"EER           : {eer:.4f}  (threshold={threshold_eer:.4f})")
    for tgt in threshold_targets:
        print(f"TAR@FAR={tgt:<7g}: {tar_at_far[tgt]:.4f}  (threshold={threshold_at_far[tgt]:.4f})")
    print("=" * 40)
    print(f"Results → {out}")
    return metrics


def _write_match_file(path, *, score, label, q_rel, g_rel,
                      lambdas, q_mnt_rows, g_mnt_rows):
    """Write one self-contained .match file (text, .min-style header convention).

    Columns: SCORE X1 Y1 THETA1 Q1 X2 Y2 THETA2 Q2
        - SCORE: per-correspondence relaxed score (lambda)
        - X*, Y*: pixel coords (origin top-left)
        - THETA*: minutia angle in degrees, CCW from +x, integer [0, 360)
        - Q*: minutia quality (integer 0-100)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(f"#MATCH SCORE {score:.6f}\n")
        fh.write(f"#MATCH LABEL {label if label is not None else ''}\n")
        fh.write(f"#MATCH Q {q_rel}\n")
        fh.write(f"#MATCH G {g_rel}\n")
        fh.write("#MMIN SCORE X1 Y1 THETA1 Q1 X2 Y2 THETA2 Q2\n")
        for lam, qm, gm in zip(lambdas, q_mnt_rows, g_mnt_rows):
            fh.write(
                f"{float(lam):.6f} "
                f"{int(qm[0])} {int(qm[1])} {int(qm[2])} {int(qm[3])} "
                f"{int(gm[0])} {int(gm[1])} {int(gm[2])} {int(gm[3])}\n"
            )


def match_pair_list(
    pairs_csv,
    output_dir,
    *,
    templates_dir,
    device="cuda",
    batch_size=64,
    min_quality=0,
    save_pair_details=False,
    match_files_dir=None,
):
    """Score an explicit list of (query, gallery) pairs from a CSV.

    Input CSV format (header required):
        query_path,gallery_path[,label]
    Paths are interpreted relative to ``templates_dir``. The optional ``label``
    column is passed through to the output unchanged (typical use: 0/1 for
    impostor/genuine, or a finger-id).

    Outputs (in ``output_dir``):
        pair_scores.csv : query_path, gallery_path, score, label (if present)
        pair_details.pkl : list of {pairs, lambda, n_pair} aligned with the CSV,
                           only when ``save_pair_details=True``.
    """
    device = device if torch.cuda.is_available() else "cpu"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(templates_dir)

    print(f"Loading pair list from {pairs_csv}")
    rows = []
    with open(pairs_csv) as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            raise MatchError(f"pairs CSV {pairs_csv} is empty")
        has_label = len(header) >= 3
        for row in reader:
            if not row or len(row) < 2:
                continue
            q, g = row[0].strip(), row[1].strip()
            label = row[2].strip() if has_label and len(row) >= 3 else None
            rows.append((q, g, label))
    if not rows:
        raise MatchError(f"pairs CSV {pairs_csv} has no data rows")
    print(f"  {len(rows)} pairs")

    need_details = save_pair_details or (match_files_dir is not None)
    need_raw_mnt = match_files_dir is not None

    print(f"Loading templates from {root} (min_quality={min_quality})...")
    cache = {}
    raw_mnt_cache = {} if need_raw_mnt else None
    def _get(rel_path):
        if rel_path not in cache:
            full = root / rel_path
            if not full.exists():
                raise MatchError(f"template not found: {full}")
            cache[rel_path] = _load_template(full, min_quality=min_quality)
            if need_raw_mnt:
                # Raw minutiae (N,4) [x, y, angle_ccw, quality] in .min convention,
                # exactly as stored by extract.py — used to render the .match files.
                with open(full, "rb") as fh:
                    data = pickle.load(fh)
                mnt = np.asarray(data["minutiae"])
                if min_quality > 0 and len(mnt) > 0:
                    mnt = mnt[mnt[:, 3] >= min_quality]
                raw_mnt_cache[rel_path] = mnt
        return cache[rel_path]

    # Pre-load every distinct path so the matching loop is purely GPU-bound.
    unique_paths = sorted({p for r in rows for p in (r[0], r[1])})
    for rel in tqdm(unique_paths, desc="  templates"):
        _get(rel)
    print(f"  loaded {len(unique_paths)} unique templates "
          f"({len(rows) * 2 - len(unique_paths)} reuses)")

    queries  = [_get(q) for q, _, _ in rows]
    gallery  = [_get(g) for _, g, _ in rows]

    print(f"Scoring {len(rows)} pairs on {device} (batch_size={batch_size})...")
    matcher = dmd.DmdMatcher()
    if need_details:
        scores, details = matcher.match_pairs(
            queries, gallery, device=device, batch_size=batch_size,
            return_details=True, progress=True,
        )
    else:
        scores = matcher.match_pairs(
            queries, gallery, device=device, batch_size=batch_size,
            return_details=False, progress=True,
        )
        details = None

    scores_path = out / "pair_scores.csv"
    with open(scores_path, "w", newline="") as fh:
        w = csv.writer(fh)
        if has_label:
            w.writerow(["query_path", "gallery_path", "score", "label"])
        else:
            w.writerow(["query_path", "gallery_path", "score"])
        for (q, g, label), s in zip(rows, scores):
            score_str = "" if np.isnan(s) else f"{float(s):.6f}"
            if has_label:
                w.writerow([q, g, score_str, label if label is not None else ""])
            else:
                w.writerow([q, g, score_str])
    print(f"  scores -> {scores_path}")

    if save_pair_details:
        details_path = out / "pair_details.pkl"
        with open(details_path, "wb") as fh:
            pickle.dump({
                "rows":    [{"query_path": q, "gallery_path": g, "label": label}
                            for q, g, label in rows],
                "details": details,
            }, fh)
        print(f"  details -> {details_path}")

    if match_files_dir is not None:
        match_root = Path(match_files_dir)
        match_root.mkdir(parents=True, exist_ok=True)
        n_written = 0
        for (q_rel, g_rel, label), s, det in tqdm(
            zip(rows, scores, details), total=len(rows), desc="  match files",
        ):
            if np.isnan(s):
                continue
            q_mnt = raw_mnt_cache[q_rel]
            g_mnt = raw_mnt_cache[g_rel]
            pairs_arr = det["pairs"]
            if len(pairs_arr) == 0:
                # Still write an empty file so consumers can rely on its existence.
                q_rows = np.empty((0, 4), dtype=np.int64)
                g_rows = np.empty((0, 4), dtype=np.int64)
            else:
                q_rows = q_mnt[pairs_arr[:, 0]]
                g_rows = g_mnt[pairs_arr[:, 1]]
            # .match path mirrors the query path under match_files_dir (.pkl→.match).
            out_path = match_root / q_rel
            out_path = out_path.with_suffix(".match")
            _write_match_file(
                out_path,
                score=float(s),
                label=label,
                q_rel=q_rel,
                g_rel=g_rel,
                lambdas=det["lambda"],
                q_mnt_rows=q_rows,
                g_mnt_rows=g_rows,
            )
            n_written += 1
        print(f"  match files -> {match_root}  ({n_written} files)")

    n_valid = int(np.sum(~np.isnan(scores)))
    print(f"Done: {n_valid}/{len(rows)} pairs scored "
          f"(mean={np.nanmean(scores):.4f}, std={np.nanstd(scores):.4f})")
    return {"n_pairs": len(rows), "n_valid": n_valid}


def run(argv):
    args = _parse_args(argv)

    # Pair-list mode short-circuits the full identification pipeline.
    if args.pairs_list:
        templates_dir = args.templates_dir or args.queries_dir
        if not templates_dir:
            print("Error: --pairs-list requires --templates-dir (or --queries-dir as fallback)")
            sys.exit(1)
        try:
            match_pair_list(
                args.pairs_list, args.output_dir,
                templates_dir=templates_dir,
                device=args.device, batch_size=args.batch_size,
                min_quality=args.min_quality,
                save_pair_details=args.save_pair_details,
                match_files_dir=args.match_files_dir,
            )
        except MatchError as e:
            print(f"Error: {e}")
            sys.exit(1)
        return

    if not args.queries_dir or not args.gallery_dir:
        print("Error: --queries-dir and --gallery-dir are required for identification mode")
        sys.exit(1)

    threshold_targets = tuple(float(x) for x in args.threshold_targets.split(",") if x.strip())
    try:
        match_dataset(
            args.queries_dir, args.gallery_dir, args.output_dir,
            query_regex=args.query_regex, query_list=args.query_list,
            gallery_regex=args.gallery_regex, gallery_list=args.gallery_list,
            distractors=args.distractors, max_distractors=args.max_distractors,
            device=args.device, batch_size=args.batch_size,
            skip_matching=args.skip_matching, min_quality=args.min_quality,
            genuine_mask_mode=args.genuine_mask_mode,
            genuine_mask_csv=args.genuine_mask_csv,
            threshold_targets=threshold_targets,
            identity_regex=args.identity_regex,
        )
    except MatchError as e:
        print(f"Error: {e}")
        sys.exit(1)
