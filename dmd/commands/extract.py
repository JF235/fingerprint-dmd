"""
dmd extract — Extract DMD++ templates from a fingerprint image dataset.

Uses pre-extracted .min minutiae files and DMD++ for descriptor extraction.
Preserves the directory hierarchy from --input-dir into --output-dir.

Output .pkl format per image:
    - minutiae:   np.ndarray (N, 4) int32   [x, y, angle_ccw_degrees, quality]
    - embeddings: np.ndarray (N, 768) float32
    - mask:       np.ndarray (N, 768) float32  (foreground mask broadcast over 12 channels)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_images(input_dir, filter_regex=None, filter_list=None):
    input_path = Path(input_dir)
    files = sorted(f for f in input_path.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS)

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


def load_min_file(min_path):
    """Load a .min minutiae file.

    .min format:
        Header line starting with #
        Each row: X Y ANGLE QUALITY [TYPE EXTRA...]
        - X, Y: pixel coordinates (origin top-left, x right, y down)
        - ANGLE: degrees, counterclockwise from +x axis, integer [0, 360)
        - QUALITY: integer 0-100
        - TYPE, EXTRA: optional, ignored

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

    Expands the foreground mask from (N, 64) to (N, 768) by repeating over
    the 12 feature channels, matching what calculate_score_torchB expects.
    """
    feature  = dmd_template["feature"].cpu().numpy()   # (N, 768)
    mask_raw = dmd_template["mask"].cpu().numpy()       # (N, 64)
    N = feature.shape[0]

    if N == 0:
        return {
            "minutiae":   np.empty((0, 4),            dtype=np.int32),
            "embeddings": np.empty((0, _D_EMBEDDING), dtype=np.float32),
            "mask":       np.empty((0, _D_EMBEDDING), dtype=np.float32),
        }

    return {
        "minutiae":   mnt_original[:N].copy(),
        "embeddings": feature.astype(np.float32),
        "mask":       np.repeat(mask_raw, _N_CHANNELS, axis=1).astype(np.float32),
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
    parser.add_argument("--overwrite",    action="store_true",
                        help="Re-extract even if .pkl already exists (default: skip)")
    return parser.parse_args(argv)


def run(argv):
    args   = _parse_args(argv)
    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"Loading DMD++ model (device={device})...")
    extractor = dmd.DmdExtractor(model_path=dmd.get_model_path("dmd++"), device=device)

    images = _collect_images(args.input_dir, args.filter_regex, args.filter_list)
    print(f"Found {len(images)} images to process")

    if not images:
        print("No images found. Exiting.")
        sys.exit(0)

    input_path    = Path(args.input_dir)
    minutiae_path = Path(args.minutiae_dir)
    output_path   = Path(args.output_dir)

    skipped = errors = 0

    for img_path in tqdm(images, desc="Extracting"):
        rel_path = img_path.relative_to(input_path)
        out_path = output_path / rel_path.with_suffix(".pkl")

        if not args.overwrite and out_path.exists():
            skipped += 1
            continue

        min_path = minutiae_path / rel_path.with_suffix(".min")
        if not min_path.exists():
            tqdm.write(f"Warning: no .min file for {rel_path} (expected {min_path})")
            errors += 1
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            tqdm.write(f"Warning: could not load {img_path}")
            errors += 1
            continue

        try:
            mnt_for_dmd, mnt_original = load_min_file(min_path)
            template = extractor.extract(img, mnt_for_dmd)
            result   = _convert_template(template, mnt_original)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                pickle.dump(result, f)

        except Exception as e:
            tqdm.write(f"Error on {rel_path}: {e}")
            errors += 1

    processed = len(images) - skipped - errors
    print(f"Done! Processed: {processed}, Skipped: {skipped}, Errors: {errors}")
