# DMD — Usage Guide

This guide covers the Python API and the `dmd` CLI for template extraction, matching, and visualization.

---

## Table of Contents

- [DMD — Usage Guide](#dmd--usage-guide)
  - [Table of Contents](#table-of-contents)
  - [Setup](#setup)
  - [Python API](#python-api)
    - [Initialization](#initialization)
    - [Template Extraction](#template-extraction)
    - [1:1 Verification](#11-verification)
    - [1:N Identification](#1n-identification)
    - [Template Structure](#template-structure)
    - [Working with .min Files](#working-with-min-files)
  - [CLI](#cli)
    - [dmd extract](#dmd-extract)
    - [dmd match](#dmd-match)
    - [dmd plot](#dmd-plot)
      - [dmd plot fingerprint](#dmd-plot-fingerprint)
      - [dmd plot roc](#dmd-plot-roc)
      - [dmd plot cmc](#dmd-plot-cmc)
      - [dmd plot scores](#dmd-plot-scores)
  - [Use Cases](#use-cases)
    - [1. Single-pair verification from scratch](#1-single-pair-verification-from-scratch)
    - [2. Batch extraction then identification](#2-batch-extraction-then-identification)
    - [3. Save and reuse templates](#3-save-and-reuse-templates)
    - [4. Run a full experiment from the CLI and visualize](#4-run-a-full-experiment-from-the-cli-and-visualize)
    - [5. Inspect a match in detail](#5-inspect-a-match-in-detail)
    - [6. Large-scale gallery with distractors](#6-large-scale-gallery-with-distractors)
    - [7. Re-run metrics without re-matching](#7-re-run-metrics-without-re-matching)

---

## Setup

Install the `dmd` package from the `dmd/` directory:

```bash
cd dmd/
pip install -e .
```

This installs the `dmd` Python module and the `dmd` CLI command.

---

## Python API

### Initialization

```python
import dmd

# Resolve path to pretrained weights ("dmd" or "dmd++")
model_path = dmd.get_model_path("dmd++")

# Initialize extractor (loads model onto device)
extractor = dmd.DmdExtractor(model_path=model_path, device="cuda")

# Initialize matcher (stateless, no model required)
matcher = dmd.DmdMatcher()
```

Available devices: `"cuda"` (recommended), `"cpu"`.
`DmdExtractor` performs a warmup inference on initialization to initialize CUDA kernels.

---

### Template Extraction

**Single image:**

```python
import cv2
import numpy as np

img = cv2.imread("fingerprint.png", cv2.IMREAD_GRAYSCALE)  # (H, W) uint8

# Minutiae: list or array of [x, y, angle_cw_degrees]
# angle_cw_degrees: clockwise from +x axis, range [0, 360)
mnt = np.array([
    [120, 200, 45.0],
    [180, 310, 130.0],
    [250, 150, 270.0],
], dtype=np.float32)

template = extractor.extract(img, mnt)
```

**Batch extraction (multiple images):**

```python
templates = extractor.extract_batch(
    images=[img1, img2, img3],       # list of (H, W) uint8 arrays
    mnts=[mnt1, mnt2, mnt3],         # list of (N_i, 3) float32 arrays
    use_gpu_patches=True,             # GPU-accelerated patch cropping (default: True)
    max_batch_size=64,                # model inference batch size (default: 64)
)
# Returns: list of template dicts, one per image
```

---

### 1:1 Verification

Compare a query fingerprint against a single reference:

```python
score = matcher.match(template_query, template_ref)
# score: numpy scalar, higher = more similar

print(f"Score: {score:.4f}")
```

With detailed output (matched pairs, per-pair scores, relaxed scores):

```python
result = matcher.match(template_query, template_ref, details=True)

print(f"Score:        {result['score']:.4f}")
print(f"Pairs used:   {result['n_pair']}")
print(f"Pair indices: {result['pairs']}")         # (n_pair, 2) — query/gallery minutia indices
print(f"Raw scores:   {result['scores']}")        # (n_pair,) — pre-relaxation
print(f"Final scores: {result['relaxed_scores']}") # (n_pair,) — post-relaxation
```

---

### 1:N Identification

Compare a set of queries against a full gallery:

```python
scores_matrix = matcher.identify(
    queries=query_templates,    # list of query templates
    gallery=gallery_templates,  # list of gallery templates
    device="cuda",              # device for batched computation
    batch_size=256,             # queries processed per batch
)
# scores_matrix: (Q, G) numpy array — scores_matrix[i, j] = score(query_i, gallery_j)

# Best gallery match per query
best_match_idx = scores_matrix.argmax(axis=1)   # (Q,)
best_score     = scores_matrix.max(axis=1)       # (Q,)
```

---

### Template Structure

Each template is a `dict` with three tensors:

```python
template = {
    "feature": torch.Tensor,  # (N, 768)  — flattened descriptor (12 channels × 8×8 spatial)
    "mask":    torch.Tensor,  # (N, 64)   — foreground mask (8×8 spatial cells)
    "mnt":     torch.Tensor,  # (1, N, 3) — minutiae [x, y, angle_cw_degrees]
}
# N = number of minutiae in the fingerprint
```

To convert to numpy for serialization:

```python
import pickle

data = {
    "feature": template["feature"].cpu().numpy(),  # (N, 768) float32
    "mask":    template["mask"].cpu().numpy(),      # (N, 64)  float32
    "mnt":     template["mnt"].cpu().numpy(),       # (1, N, 3)
}
with open("template.pkl", "wb") as f:
    pickle.dump(data, f)
```

---

### Working with .min Files

The `.min` file format stores minutiae as `X Y ANGLE_CCW QUALITY` (CCW = counterclockwise).
DMD expects clockwise angles. Use the loader bundled in the module:

```python
from dmd.commands.extract import load_min_file

# Returns:
#   mnt_for_dmd:  (N, 3) float32 [x, y, angle_cw]  — ready for extractor.extract()
#   mnt_original: (N, 4) int32   [x, y, angle_ccw, quality] — original .min values
mnt_for_dmd, mnt_original = load_min_file("fingerprint.min")

template = extractor.extract(img, mnt_for_dmd)
```

---

## CLI

After `pip install -e .` the `dmd` command is available in your environment.

### dmd extract

Extract DMD++ templates from a directory of images and minutiae files. Preserves the directory hierarchy from `--input-dir` into `--output-dir`, writing one `.pkl` per image.

```
dmd extract --input-dir DIR --minutiae-dir DIR --output-dir DIR [options]
```

**Arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input-dir` | ✓ | — | Root directory of fingerprint images (searched recursively) |
| `--minutiae-dir` | ✓ | — | Root directory of `.min` minutiae files (same hierarchy as `--input-dir`) |
| `--output-dir` | ✓ | — | Output directory for `.pkl` template files |
| `--device` | | `cuda` | Torch device (`cuda` or `cpu`) |
| `--filter-regex` | | — | Regex applied to relative image path to filter files |
| `--filter-list` | | — | Text file with one relative path per line to include |
| `--overwrite` | | skip | Re-extract even if `.pkl` already exists |

**Examples:**

```bash
# Basic extraction
dmd extract \
  --input-dir  /data/sd258/images \
  --minutiae-dir /data/sd258/minutiae \
  --output-dir /data/sd258/templates/dmd

# Extract only query images (filename contains "-latent-")
dmd extract \
  --input-dir  /data/sd27/images \
  --minutiae-dir /data/sd27/minutiae \
  --output-dir /data/sd27/templates/dmd \
  --filter-regex ".*-latent-.*"

# Force re-extraction on CPU
dmd extract \
  --input-dir  /data/test/images \
  --minutiae-dir /data/test/minutiae \
  --output-dir /data/test/templates/dmd \
  --device cpu --overwrite
```

**Output `.pkl` format per file:**

```python
{
    "minutiae":   np.ndarray (N, 4) int32   # [x, y, angle_ccw_degrees, quality]
    "embeddings": np.ndarray (N, 768) float32
    "mask":       np.ndarray (N, 768) float32
}
```

---

### dmd match

Run an identification / verification experiment over template directories. Produces score matrix, CMC, ROC, and summary metrics.

```
dmd match --queries-dir DIR --gallery-dir DIR --output-dir DIR [options]
```

**Arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--queries-dir` | ✓ | — | Root directory of query `.pkl` templates |
| `--gallery-dir` | ✓ | — | Root directory of gallery `.pkl` templates |
| `--output-dir` | ✓ | — | Directory to write results |
| `--query-regex` | | — | Regex filter on query relative paths |
| `--query-list` | | — | Text file with query relative paths |
| `--gallery-regex` | | — | Regex filter on gallery relative paths |
| `--gallery-list` | | — | Text file with gallery relative paths |
| `--distractors` | | — | Additional gallery directories (repeatable) |
| `--device` | | `cuda` | Torch device for score computation |
| `--batch-size` | | `256` | Batch size for `identify()` |
| `--skip-matching` | | — | Skip score computation, load existing `scores.npy` |

**Examples:**

```bash
# Basic latent-to-reference experiment
dmd match \
  --queries-dir  /data/sd258/templates/dmd \
  --gallery-dir  /data/sd258/templates/dmd \
  --query-regex  ".*-latent-.*" \
  --gallery-regex ".*-rolled-.*" \
  --output-dir   results/sd258

# With distractor gallery to increase gallery size
dmd match \
  --queries-dir  /data/sd27/templates/dmd \
  --gallery-dir  /data/sd27/templates/dmd \
  --distractors  /data/sd14/templates/dmd \
  --output-dir   results/sd27_with_distractors \
  --batch-size 512

# Re-compute metrics from pre-existing scores (no GPU needed)
dmd match \
  --queries-dir  /data/sd258/templates/dmd \
  --gallery-dir  /data/sd258/templates/dmd \
  --output-dir   results/sd258 \
  --skip-matching
```

**Output files:**

| File | Description |
|------|-------------|
| `scores.npy` | `(Q, G)` float32 score matrix |
| `query_index.csv` | Query file paths and identity keys |
| `gallery_index.csv` | Gallery file paths and identity keys |
| `genuine_mask.npy` | `(Q, G)` boolean genuine-pair mask |
| `cmc.csv` | CMC curve: `rank, cmc` |
| `roc.csv` | ROC curve: `fpr, tpr, threshold` |
| `metrics.json` | Summary: Rank-1, EER, TAR@FAR 0.1% and 1% |

---

### dmd plot

Visualize extraction and matching results. All plots use matplotlib only.

#### dmd plot fingerprint

Display a fingerprint image with its minutiae overlaid.

```bash
dmd plot fingerprint --image fingerprint.png --minutiae fingerprint.min
dmd plot fingerprint --image fingerprint.png --minutiae fingerprint.min --output vis.png
```

#### dmd plot roc

Plot ROC curve from a results directory. Annotates the Equal Error Rate (EER).

```bash
dmd plot roc --results-dir results/sd258
dmd plot roc --results-dir results/sd258 --output roc.png
```

#### dmd plot cmc

Plot CMC (Cumulative Match Characteristic) curve.

```bash
dmd plot cmc --results-dir results/sd258
dmd plot cmc --results-dir results/sd258 --max-rank 10 --output cmc.png
```

#### dmd plot scores

Plot overlapping genuine / impostor score distributions as histograms.

```bash
dmd plot scores --results-dir results/sd258
dmd plot scores --results-dir results/sd258 --output scores.png
```

---

## Use Cases

### 1. Single-pair verification from scratch

```python
import cv2
import dmd
from dmd.commands.extract import load_min_file

extractor = dmd.DmdExtractor(dmd.get_model_path("dmd++"), device="cuda")
matcher   = dmd.DmdMatcher()

def make_template(img_path, min_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mnt, _ = load_min_file(min_path)
    return extractor.extract(img, mnt)

t_latent = make_template("latent.png", "latent.min")
t_ref    = make_template("reference.png", "reference.min")

score = matcher.match(t_latent, t_ref)
print(f"Score: {score:.4f}  →  {'MATCH' if score > 0.5 else 'NON-MATCH'}")
```

---

### 2. Batch extraction then identification

```python
import cv2, glob, dmd
from pathlib import Path
from dmd.commands.extract import load_min_file

extractor = dmd.DmdExtractor(dmd.get_model_path("dmd++"), device="cuda")
matcher   = dmd.DmdMatcher()

def load_dataset(img_dir, min_dir):
    images, mnts, names = [], [], []
    for img_path in sorted(Path(img_dir).rglob("*.png")):
        min_path = Path(min_dir) / img_path.relative_to(img_dir).with_suffix(".min")
        if not min_path.exists():
            continue
        mnt, _ = load_min_file(min_path)
        images.append(cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE))
        mnts.append(mnt)
        names.append(img_path.stem)
    return images, mnts, names

query_imgs, query_mnts, query_names = load_dataset("data/queries/images", "data/queries/minutiae")
gallery_imgs, gallery_mnts, gallery_names = load_dataset("data/gallery/images", "data/gallery/minutiae")

query_templates   = extractor.extract_batch(query_imgs,   query_mnts)
gallery_templates = extractor.extract_batch(gallery_imgs, gallery_mnts)

scores = matcher.identify(query_templates, gallery_templates, device="cuda", batch_size=256)

for i, name in enumerate(query_names):
    best_j = scores[i].argmax()
    print(f"{name}  →  {gallery_names[best_j]}  (score={scores[i, best_j]:.4f})")
```

---

### 3. Save and reuse templates

Extract once, match many times:

```python
import pickle, dmd, cv2
from dmd.commands.extract import load_min_file

extractor = dmd.DmdExtractor(dmd.get_model_path("dmd++"), device="cuda")

# --- Extraction (done once) ---
img = cv2.imread("fp.png", cv2.IMREAD_GRAYSCALE)
mnt, _ = load_min_file("fp.min")
template = extractor.extract(img, mnt)

with open("fp_template.pkl", "wb") as f:
    pickle.dump({k: v.cpu().numpy() for k, v in template.items()}, f)

# --- Reload and match (GPU not needed) ---
import torch

def load_template(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return {k: torch.from_numpy(v) for k, v in data.items()}

t1 = load_template("fp_template.pkl")
t2 = load_template("other_template.pkl")

matcher = dmd.DmdMatcher()
score = matcher.match(t1, t2)
```

---

### 4. Run a full experiment from the CLI and visualize

```bash
# Step 1 — Extract templates for queries and gallery
dmd extract \
  --input-dir   /data/nist27/images/latent \
  --minutiae-dir /data/nist27/minutiae/latent \
  --output-dir  /data/nist27/templates/latent

dmd extract \
  --input-dir   /data/nist27/images/reference \
  --minutiae-dir /data/nist27/minutiae/reference \
  --output-dir  /data/nist27/templates/reference

# Step 2 — Run matching experiment
dmd match \
  --queries-dir  /data/nist27/templates/latent \
  --gallery-dir  /data/nist27/templates/reference \
  --output-dir   results/nist27 \
  --device cuda

# Step 3 — Visualize results
dmd plot roc    --results-dir results/nist27 --output results/nist27/roc.png
dmd plot cmc    --results-dir results/nist27 --output results/nist27/cmc.png
dmd plot scores --results-dir results/nist27 --output results/nist27/scores.png
```

---

### 5. Inspect a match in detail

```python
import dmd

matcher = dmd.DmdMatcher()
result  = matcher.match(t_latent, t_ref, details=True)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import numpy as np

# Visualize matched minutiae pairs on both images
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, img_path, label in zip(axes, ["latent.png", "reference.png"], ["Query", "Gallery"]):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    ax.imshow(img, cmap="gray")
    ax.set_title(label)
    ax.axis("off")

mnt_q = t_latent["mnt"][0].cpu().numpy()   # (N_q, 3)
mnt_g = t_ref["mnt"][0].cpu().numpy()      # (N_g, 3)

for q_idx, g_idx in result["pairs"]:
    x_q, y_q = mnt_q[q_idx, :2]
    x_g, y_g = mnt_g[g_idx, :2]
    axes[0].plot(x_q, y_q, "ro", markersize=6)
    axes[1].plot(x_g, y_g, "ro", markersize=6)

plt.suptitle(f"Score: {result['score']:.4f}  |  Pairs: {result['n_pair']}")
plt.tight_layout()
plt.savefig("matched_pairs.png", dpi=150)
```

---

### 6. Large-scale gallery with distractors

Use `--distractors` to inflate the gallery with non-genuine entries, simulating realistic identification conditions:

```bash
dmd match \
  --queries-dir   /data/sd258/templates/latent \
  --gallery-dir   /data/sd258/templates/reference \
  --distractors   /data/sd14/templates/reference \
  --output-dir    results/sd258_large_gallery \
  --batch-size 512 --device cuda
```

---

### 7. Re-run metrics without re-matching

If `scores.npy` already exists, skip the GPU-intensive matching step:

```bash
dmd match \
  --queries-dir  /data/sd258/templates/dmd \
  --gallery-dir  /data/sd258/templates/dmd \
  --output-dir   results/sd258 \
  --skip-matching
```

Useful for testing different query/gallery splits without recomputing the full score matrix.
