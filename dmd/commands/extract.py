"""
dmd extract — Extract DMD++ templates from a fingerprint image dataset.

Uses pre-extracted .min minutiae files and DMD++ for descriptor extraction.
Preserves the directory hierarchy from --input-dir into --output-dir.

Output .pkl format per image:
    - minutiae:   np.ndarray (N, 4) int32   [x, y, angle_ccw_degrees, quality]
    - embeddings: np.ndarray (N, 768) float32
    - mask:       np.ndarray (N, 64) float32  (raw foreground mask; the matcher
                                                expands it ×12 internally to (N, 768))

Note: older .pkl files in the wild use (N, 768) for the mask (repeat×12 of the
raw 64-cell mask). Readers in this repo accept both — see
``_pkl_to_dmd_template`` in match.py.
"""

import argparse
import pickle
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

import dmd

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

# DMD++ architecture constants
_NDIM_FEAT    = 6
_N_CHANNELS   = _NDIM_FEAT * 2                      # 12 (texture + minutiae)
_SPATIAL_SIZE = 8                                    # 128 / (2^4 strides)
_D_EMBEDDING  = _N_CHANNELS * _SPATIAL_SIZE ** 2    # 768
_MASK_DIM     = _SPATIAL_SIZE ** 2                   # 64 (one mask cell per spatial position)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_images(input_dir, filter_regex=None, filter_list=None):
    input_path = Path(input_dir)
    # Use os.walk(followlinks=True) so dataset trees with symlinks (e.g. FVCL400
    # mirroring FVC) are traversed correctly. Path.rglob doesn't follow symlinks
    # by default until Python 3.13.
    import os
    found = []
    for root, dirs, fnames in os.walk(input_path, followlinks=True):
        for fn in fnames:
            p = Path(root) / fn
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                found.append(p)
    files = sorted(found)

    if filter_regex:
        pattern = re.compile(filter_regex)
        files = [f for f in files if pattern.search(str(f.relative_to(input_path)))]

    if filter_list:
        with open(filter_list) as fh:
            allowed = set(line.strip() for line in fh if line.strip())
        files = [
            f for f in files
            if str(f.relative_to(input_path)) in allowed or f.name in allowed
        ]

    return files


def load_min_file(min_path, min_quality=0):
    """Load a .min minutiae file.

    .min format:
        Header line starting with #
        Each row: X Y ANGLE QUALITY [TYPE EXTRA...]
        - X, Y: pixel coordinates (origin top-left, x right, y down)
        - ANGLE: degrees, counterclockwise from +x axis, integer [0, 360)
        - QUALITY: integer 0-100
        - TYPE, EXTRA: optional, ignored

    Args:
        min_path:    path to .min file
        min_quality: drop minutiae with quality < min_quality (default 0 = keep all)

    Returns:
        mnt_for_dmd:  (N, 3) float32 [x, y, angle_cw_degrees]  — for extractor.extract()
        mnt_original: (N, 4) int32   [x, y, angle_ccw_degrees, quality]  — preserved in .pkl
    """
    rows = []
    with open(min_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            x, y, angle, quality = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            if quality < min_quality:
                continue
            rows.append((x, y, angle, quality))

    if not rows:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 4), dtype=np.int32)

    raw = np.array(rows, dtype=np.float32)

    # .min angles are CCW from +x (0°=right, 90°=up).
    # DMD's extract_patches passes the angle to cv2.getRotationMatrix2D (rotates CCW).
    # To normalize a minutia at θ_ccw to 0°: rotation = -θ_ccw = (360 - θ_ccw) % 360.
    angle_for_dmd = (360.0 - raw[:, 2]) % 360.0

    mnt_for_dmd  = np.column_stack([raw[:, 0], raw[:, 1], angle_for_dmd]).astype(np.float32)
    mnt_original = raw.astype(np.int32)

    return mnt_for_dmd, mnt_original


def _convert_template(dmd_template, mnt_original):
    """Convert raw DMD output into the standard .pkl format.

    Stores the raw (N, 64) foreground mask — one cell per 8×8 spatial
    position. Earlier versions of this code persisted the mask pre-expanded
    to (N, 768) (repeat×12 across feature channels) to save a multiplication
    inside the matcher; that wasted ~190 KB per template on disk. The matcher
    now expands the mask itself when it loads a template (see
    ``_pkl_to_dmd_template`` in match.py), so we keep the compact form here
    and the readers stay backwards-compatible with the old layout.
    """
    feature  = dmd_template["feature"].cpu().numpy()   # (N, 768)
    mask_raw = dmd_template["mask"].cpu().numpy()       # (N, 64)
    N = feature.shape[0]

    if N == 0:
        return {
            "minutiae":   np.empty((0, 4),            dtype=np.int32),
            "embeddings": np.empty((0, _D_EMBEDDING), dtype=np.float32),
            "mask":       np.empty((0, _MASK_DIM),    dtype=np.float32),
        }

    return {
        "minutiae":   mnt_original[:N].copy(),
        "embeddings": feature.astype(np.float32),
        "mask":       mask_raw.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="dmd extract",
        description="Extract DMD++ templates from a fingerprint image dataset.",
    )
    parser.add_argument("--input-dir",    required=True,
                        help="Root directory of fingerprint images (searched recursively)")
    parser.add_argument("--minutiae-dir", required=True,
                        help="Root directory of .min minutiae files (same hierarchy as --input-dir)")
    parser.add_argument("--output-dir",   required=True,
                        help="Output directory for .pkl templates (hierarchy preserved)")
    parser.add_argument("--filter-regex", default=None,
                        help='Regex applied to relative image path (e.g. ".*-00-.*")')
    parser.add_argument("--filter-list",  default=None,
                        help="Text file with one relative path per line to include")
    parser.add_argument("--device",       default="cuda",
                        help="Torch device for inference (default: cuda)")
    parser.add_argument("--batch-size",   type=int, default=1,
                        help="Number of images to process per batch (default: 1)")
    parser.add_argument("--overwrite",    action="store_true",
                        help="Re-extract even if .pkl already exists (default: skip)")
    parser.add_argument("--min-quality",  type=int, default=0,
                        help="Drop minutiae with quality < N before extraction (default: 0 = keep all)")
    return parser.parse_args(argv)


def extract_dataset(
    extractor,
    input_dir,
    minutiae_dir,
    output_dir,
    *,
    images=None,
    filter_regex=None,
    filter_list=None,
    min_quality=0,
    overwrite=False,
    batch_size=1,
    progress_desc="Extracting",
):
    """Extract DMD templates over a dataset using a pre-loaded extractor.

    If `images` is provided (list of Paths), the filesystem walk is skipped.
    `batch_size > 1` enables ``extractor.extract_batch`` for throughput; on
    failure the batch is retried image-by-image so a single bad image does
    not poison the whole batch. Returns dict with counts:
    ``{processed, skipped, errors, total}``.
    """
    if images is None:
        images = _collect_images(input_dir, filter_regex, filter_list)
    if not images:
        print(f"No images found under {input_dir}")
        return {"processed": 0, "skipped": 0, "errors": 0, "total": 0}

    input_path    = Path(input_dir)
    minutiae_path = Path(minutiae_dir)
    output_path   = Path(output_dir)

    skipped = errors = 0

    # Filter images: skip already-extracted, verify .min present.
    work_items = []  # (img_path, rel_path, out_path, min_path)
    for img_path in images:
        rel_path = img_path.relative_to(input_path)
        out_path = output_path / rel_path.with_suffix(".pkl")

        if not overwrite and out_path.exists():
            skipped += 1
            continue

        min_path = minutiae_path / rel_path.with_suffix(".min")
        if not min_path.exists():
            tqdm.write(f"Warning: no .min file for {rel_path} (expected {min_path})")
            errors += 1
            continue

        work_items.append((img_path, rel_path, out_path, min_path))

    pbar = tqdm(total=len(work_items), desc=progress_desc)
    meta = {"min_quality": min_quality}

    for batch_start in range(0, len(work_items), batch_size):
        batch = work_items[batch_start:batch_start + batch_size]
        batch_imgs      = []
        batch_mnts_dmd  = []
        batch_mnts_orig = []
        batch_meta      = []  # list of (rel_path, out_path) parallel to batch_imgs

        for img_path, rel_path, out_path, min_path in batch:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                tqdm.write(f"Warning: could not load {img_path}")
                errors += 1
                pbar.update(1)
                continue

            mnt_for_dmd, mnt_original = load_min_file(min_path, min_quality=min_quality)
            batch_imgs.append(img)
            batch_mnts_dmd.append(mnt_for_dmd)
            batch_mnts_orig.append(mnt_original)
            batch_meta.append((rel_path, out_path))

        if not batch_imgs:
            continue

        try:
            if len(batch_imgs) == 1:
                templates = [extractor.extract(batch_imgs[0], batch_mnts_dmd[0])]
            else:
                templates = extractor.extract_batch(
                    batch_imgs, batch_mnts_dmd, max_batch_size=64,
                )
        except Exception as e:
            tqdm.write(f"Batch failed ({e}); retrying images individually")
            templates = []
            for (rel_path, _out), img, mnt in zip(batch_meta, batch_imgs, batch_mnts_dmd):
                try:
                    templates.append(extractor.extract(img, mnt))
                except Exception as e2:
                    tqdm.write(f"Error on {rel_path}: {e2}")
                    templates.append(None)

        for tmpl, mnt_orig, (rel_path, out_path) in zip(
            templates, batch_mnts_orig, batch_meta,
        ):
            if tmpl is None:
                errors += 1
                pbar.update(1)
                continue
            result = _convert_template(tmpl, mnt_orig)
            result["meta"] = meta
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                pickle.dump(result, f)
            pbar.update(1)

    pbar.close()
    processed = len(images) - skipped - errors
    return {"processed": processed, "skipped": skipped, "errors": errors, "total": len(images)}


def run(argv):
    args   = _parse_args(argv)
    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"Loading DMD++ model (device={device})...")
    extractor = dmd.DmdExtractor(model_path=dmd.get_model_path("dmd++"), device=device)

    print(f"Scanning {args.input_dir}...")
    counts = extract_dataset(
        extractor,
        args.input_dir,
        args.minutiae_dir,
        args.output_dir,
        filter_regex=args.filter_regex,
        filter_list=args.filter_list,
        min_quality=args.min_quality,
        overwrite=args.overwrite,
        batch_size=args.batch_size,
    )

    if counts["total"] == 0:
        sys.exit(0)
    print(f"Done! Processed: {counts['processed']}, Skipped: {counts['skipped']}, Errors: {counts['errors']}")
