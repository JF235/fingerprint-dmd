"""Build a genuine pair list for BN48k templates.

BN48k layout (per subject):
    <subject_id>/00/bn48k_<subject>_11-00_l.pkl
    <subject_id>/00/bn48k_<subject>_11-00_r.pkl

The `_l` and `_r` files are two captures of the same finger, so the
(l, r) pair is genuine. This script walks --templates-dir and writes a CSV
of those pairs, suitable for ``dmd match --pairs-list``.

Usage:
    python build_bn48k_genuine_pairs.py \
        --templates-dir /path/to/BN48k/templates/dmd/fingernet \
        --output pairs.csv
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


# Filename → finger variant. Captures the suffix after the last underscore.
_VARIANT_RE = re.compile(r"_([lr])\.pkl$", re.IGNORECASE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates-dir", required=True,
                    help="Root of BN48k DMD templates (subject/capture/*.pkl)")
    ap.add_argument("--output",        required=True,
                    help="Output CSV path")
    ap.add_argument("--label",         default="1",
                    help='Value written in the label column (default "1" for genuine)')
    args = ap.parse_args()

    root = Path(args.templates_dir)
    if not root.exists():
        raise SystemExit(f"templates-dir does not exist: {root}")

    # Group .pkl files by (subject, capture) and split by l/r.
    groups = defaultdict(dict)  # key -> {'l': rel_path, 'r': rel_path}
    skipped = 0
    for pkl in root.rglob("*.pkl"):
        rel = pkl.relative_to(root)
        m = _VARIANT_RE.search(pkl.name)
        if not m:
            skipped += 1
            continue
        variant = m.group(1).lower()
        # Key: (subject_dir, capture_dir) — the level-1 and level-2 dirs.
        parts = rel.parts
        if len(parts) < 3:
            skipped += 1
            continue
        key = (parts[0], parts[1])
        groups[key][variant] = str(rel)

    pairs = []
    incomplete = 0
    for key, variants in sorted(groups.items()):
        if "l" not in variants or "r" not in variants:
            incomplete += 1
            continue
        pairs.append((variants["l"], variants["r"]))

    print(f"Subjects scanned: {len(groups)}")
    print(f"Complete (l+r) pairs: {len(pairs)}")
    print(f"Incomplete groups (missing l or r): {incomplete}")
    if skipped:
        print(f"Skipped files (no _l/_r suffix or shallow path): {skipped}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["query_path", "gallery_path", "label"])
        for l_rel, r_rel in pairs:
            w.writerow([l_rel, r_rel, args.label])

    print(f"Wrote {len(pairs)} pairs -> {out_path}")


if __name__ == "__main__":
    main()
