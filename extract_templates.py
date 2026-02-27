#!/usr/bin/env python3
"""
Extract DMD templates from a fingerprint image dataset.

Uses pre-extracted .min minutiae files and DMD++ for descriptor extraction.
Preserves the directory hierarchy from input-dir into output-dir.

Output .pkl format per image:
    - minutiae: np.ndarray (N, 4) int32 [x, y, angle_degrees_ccw, quality_0_100]
      (angles follow .min convention: CCW from +x, 0°=right, 90°=up, [0,360))
    - embeddings: np.ndarray (N, D) float32, D = 12*8*8 = 768
    - mask: np.ndarray (N, D) float32, foreground mask broadcast over 12 channels
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

# DMD++ architecture constants (from DMD++.yaml and model architecture)
NDIM_FEAT = 6  # per branch
N_CHANNELS = NDIM_FEAT * 2  # 12 (texture + minutiae branches)
SPATIAL_SIZE = 8  # 128 / (2^4 strides) = 8
D_EMBEDDING = N_CHANNELS * SPATIAL_SIZE * SPATIAL_SIZE  # 768


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract DMD templates from a fingerprint image dataset."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Root directory containing fingerprint images (searched recursively)",
    )
    parser.add_argument(
        "--minutiae-dir",
        required=True,
        help="Root directory containing .min minutiae files (same hierarchy as input-dir)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for .pkl templates (preserves relative hierarchy)",
    )
    parser.add_argument(
        "--filter-regex",
        default=None,
        help='Regex applied to relative path to filter images (e.g. ".*-00-.*")',
    )
    parser.add_argument(
        "--filter-list",
        default=None,
        help="Text file with one filename/relative-path per line to include",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for inference (default: cuda)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .pkl files (default: skip)",
    )
    return parser.parse_args()


def collect_images(input_dir, filter_regex=None, filter_list=None):
    """Collect image paths from input_dir, applying optional filters."""
    input_path = Path(input_dir)
    all_files = sorted(
        f for f in input_path.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS
    )

    if filter_regex:
        pattern = re.compile(filter_regex)
        all_files = [
            f for f in all_files if pattern.search(str(f.relative_to(input_path)))
        ]

    if filter_list:
        with open(filter_list) as fh:
            allowed = set(line.strip() for line in fh if line.strip())
        all_files = [
            f
            for f in all_files
            if str(f.relative_to(input_path)) in allowed or f.name in allowed
        ]

    return all_files


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
        mnt_for_dmd: (N, 3) float32 [x, y, angle_cw_degrees] for DMD's extract_patches
        mnt_original: (N, 4) int32 [x, y, angle_ccw_degrees, quality] preserving .min convention
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
    # DMD's extract_patches passes the angle directly to cv2.getRotationMatrix2D,
    # which rotates the image CCW by that value. To normalize a minutia at θ_ccw
    # to 0° we need: θ_ccw + rotation = 0  →  rotation = -θ_ccw = (360 - θ_ccw) % 360.
    # This is equivalent to the "clockwise" angle, matching what the grids pipeline does
    # via ExtractMntStitchMinutiae(clockwise=True).
    angle_for_dmd = (360.0 - raw[:, 2]) % 360.0

    mnt_for_dmd = np.column_stack([raw[:, 0], raw[:, 1], angle_for_dmd]).astype(np.float32)
    mnt_original = raw.astype(np.int32)  # preserve original .min values [x, y, angle_ccw, quality]

    return mnt_for_dmd, mnt_original


def convert_template(dmd_template, mnt_original):
    """Convert raw DMD output + minutiae into the standard .pkl format.

    DMD's get_embedding already flattens the spatial dims:
        feature: (N, 768)  = (N, 12*8*8)  -- already flat
        mask:    (N, 64)   = (N, 1*8*8)   -- already flat

    We expand the mask to (N, 768) by repeating over the 12 channels,
    matching what calculate_score_torchB does: mask.repeat(1, 1, ndim_feat*2).

    Args:
        dmd_template: dict with 'feature' (N, 768) and 'mask' (N, 64)
        mnt_original: (N, 4) int32 [x, y, angle_ccw, quality] in .min convention
    """
    feature = dmd_template["feature"].cpu().numpy()  # (N, 768)
    mask_raw = dmd_template["mask"].cpu().numpy()  # (N, 64)
    N = feature.shape[0]

    if N == 0:
        return {
            "minutiae": np.empty((0, 4), dtype=np.int32),
            "embeddings": np.empty((0, D_EMBEDDING), dtype=np.float32),
            "mask": np.empty((0, D_EMBEDDING), dtype=np.float32),
        }

    embeddings = feature.astype(np.float32)

    # Expand mask: (N, 64) -> repeat 12x -> (N, 768)
    mask_expanded = np.repeat(mask_raw, N_CHANNELS, axis=1).astype(np.float32)

    # Minutiae: preserve original .min convention (CCW angles)
    minutiae = mnt_original[:N].copy()

    return {
        "minutiae": minutiae,
        "embeddings": embeddings,
        "mask": mask_expanded,
    }


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    # Load DMD model
    print(f"Loading DMD++ model (device={device})...")
    dmd_extractor = dmd.DmdExtractor(
        model_path=dmd.get_model_path("dmd++"), device=device
    )

    # Collect images
    images = collect_images(args.input_dir, args.filter_regex, args.filter_list)
    print(f"Found {len(images)} images to process")

    if not images:
        print("No images found. Exiting.")
        sys.exit(0)

    input_path = Path(args.input_dir)
    minutiae_path = Path(args.minutiae_dir)
    output_path = Path(args.output_dir)

    skipped = 0
    errors = 0

    for img_path in tqdm(images, desc="Extracting"):
        rel_path = img_path.relative_to(input_path)
        out_path = output_path / rel_path.with_suffix(".pkl")

        # Skip existing
        if not args.overwrite and out_path.exists():
            skipped += 1
            continue

        # Find corresponding .min file (same relative path, .min extension)
        min_path = minutiae_path / rel_path.with_suffix(".min")
        if not min_path.exists():
            tqdm.write(f"Warning: no .min file for {rel_path} (expected {min_path})")
            errors += 1
            continue

        # Load image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            tqdm.write(f"Warning: could not load {img_path}")
            errors += 1
            continue

        try:
            # 1. Load minutiae from .min file
            mnt_for_dmd, mnt_original = load_min_file(min_path)

            # 2. Extract DMD template
            template = dmd_extractor.extract(img, mnt_for_dmd)

            # 3. Convert to standard format
            result = convert_template(template, mnt_original)

            # 4. Save
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                pickle.dump(result, f)

        except Exception as e:
            tqdm.write(f"Error on {rel_path}: {e}")
            errors += 1
            continue

    processed = len(images) - skipped - errors
    print(f"Done! Processed: {processed}, Skipped: {skipped}, Errors: {errors}")


if __name__ == "__main__":
    main()
