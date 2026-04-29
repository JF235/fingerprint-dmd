"""dmd experiment — orchestrate multi-extractor / multi-combo experiments.

Reads a YAML config (see dmd/examples/experiments_config.yaml) and dispatches
to the existing extract/match pipelines.

Usage:
    dmd experiment extract --config CONFIG.yaml [--datasets D1,D2] [--extractors E1,E2]
    dmd experiment match   --config CONFIG.yaml [--combos C1,C2]   [--extractors E1,E2]
    dmd experiment full    --config CONFIG.yaml

Outputs (per combo × extractor):
    {output_root}/{combo}/{extractor}/
        scores.npy, genuine_mask.npy
        genuine_scores.npy, impostor_scores.npy
        query_index.csv, gallery_index.csv
        cmc.csv, roc.csv, metrics.json
        run_config.yaml      — snapshot of resolved config
        manifest.json        — git_sha, hostname, timestamp, paths
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
import yaml

import dmd
from dmd.commands import extract as extract_cmd
from dmd.commands import match as match_cmd
from dmd.commands._experiment_config import parse_config, to_dict


STATUS_OK              = "ok"
STATUS_SKIPPED         = "skipped"
STATUS_MISSING_QUERIES = "missing-queries"
STATUS_MISSING_GALLERY = "missing-gallery"
STATUS_ERROR           = "error"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_sha(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def _resolve_subset(requested, defined, kind):
    """Pick a subset by comma-separated names from a list of known names; default to all."""
    if not requested:
        return list(defined)
    chosen = [x.strip() for x in requested.split(",") if x.strip()]
    unknown = [x for x in chosen if x not in defined]
    if unknown:
        raise ValueError(f"Unknown {kind}(s): {unknown}; defined: {list(defined)}")
    return chosen


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------

def _do_extract(cfg, datasets, extractors):
    device = cfg.extraction.device if torch.cuda.is_available() else "cpu"
    print(f"Loading DMD++ model (device={device})...")
    extractor = dmd.DmdExtractor(model_path=dmd.get_model_path("dmd++"), device=device)

    storage = Path(cfg.storage_root)
    summary = []

    for ds_name in datasets:
        ds = cfg.datasets[ds_name]
        input_dir = ds.abs_images(storage)
        if not input_dir.exists():
            print(f"[skip] {ds_name}: images_dir not found ({input_dir})")
            continue

        # Walk the image tree once per dataset, reuse across extractors.
        images = extract_cmd._collect_images(input_dir)
        if not images:
            print(f"[skip] {ds_name}: no images found under {input_dir}")
            continue

        for ext_name in extractors:
            minutiae_dir = ds.abs_minutiae(storage) / ext_name
            output_dir   = ds.abs_templates(storage) / ext_name

            if not minutiae_dir.exists():
                print(f"[skip] {ds_name}/{ext_name}: minutiae_dir not found ({minutiae_dir})")
                continue

            print(f"\n=== Extract {ds_name} / {ext_name} ===")
            print(f"  images:    {input_dir}  ({len(images)} files)")
            print(f"  minutiae:  {minutiae_dir}")
            print(f"  templates: {output_dir}")

            t0 = time.time()
            counts = extract_cmd.extract_dataset(
                extractor,
                input_dir, minutiae_dir, output_dir,
                images=images,
                min_quality=cfg.extraction.min_quality,
                overwrite=cfg.extraction.overwrite,
                progress_desc=f"{ds_name}/{ext_name}",
            )
            elapsed = time.time() - t0
            counts["elapsed_s"] = round(elapsed, 1)
            counts["dataset"]   = ds_name
            counts["extractor"] = ext_name
            summary.append(counts)
            print(f"  Done. processed={counts['processed']} skipped={counts['skipped']} errors={counts['errors']} ({elapsed:.0f}s)")

    print("\n=== Extraction summary ===")
    for s in summary:
        print(f"  {s['dataset']:<12s} {s['extractor']:<14s}  proc={s['processed']:<6d} skip={s['skipped']:<6d} err={s['errors']:<4d} ({s['elapsed_s']}s)")
    return summary


# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------

def _build_manifest(cfg, combo, extractor, status, elapsed_s, error=None):
    manifest = {
        "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname":          socket.gethostname(),
        "user":              os.environ.get("USER", "?"),
        "python_version":    sys.version.split()[0],
        "torch_version":     torch.__version__,
        "cuda_available":    torch.cuda.is_available(),
        "git_sha": {
            "fingerprint-dmd": _git_sha(Path(__file__).resolve().parents[2]),
        },
        "combo":             combo.name,
        "extractor":         extractor,
        "queries_dataset":   combo.queries.dataset,
        "gallery_dataset":   combo.gallery.dataset,
        "distractors":       combo.distractors,
        "genuine_mask_mode": combo.genuine_mask_mode,
        "match_min_quality": cfg.matching.min_quality,
        "status":            status,
        "elapsed_s":         elapsed_s,
    }
    if error is not None:
        manifest["error"] = error
    return manifest


def _write_run_config(out_dir, cfg):
    with open(out_dir / "run_config.yaml", "w") as f:
        yaml.safe_dump(to_dict(cfg), f, default_flow_style=False, sort_keys=False)


def _do_match(cfg, combos, extractors, force=False):
    output_root = Path(cfg.output_root)
    storage     = Path(cfg.storage_root)
    summary     = []

    for combo in combos:
        for ext_name in extractors:
            out_dir = output_root / combo.name / ext_name
            metrics_path = out_dir / "metrics.json"

            if metrics_path.exists() and not force:
                print(f"[skip] {combo.name}/{ext_name}: metrics.json exists (use --force to recompute)")
                summary.append({"combo": combo.name, "extractor": ext_name, "status": STATUS_SKIPPED})
                continue

            qroot = cfg.datasets[combo.queries.dataset].abs_templates(storage) / ext_name
            groot = cfg.datasets[combo.gallery.dataset].abs_templates(storage) / ext_name
            if not qroot.exists():
                print(f"[skip] {combo.name}/{ext_name}: queries templates not found ({qroot})")
                summary.append({"combo": combo.name, "extractor": ext_name, "status": STATUS_MISSING_QUERIES})
                continue
            if not groot.exists():
                print(f"[skip] {combo.name}/{ext_name}: gallery templates not found ({groot})")
                summary.append({"combo": combo.name, "extractor": ext_name, "status": STATUS_MISSING_GALLERY})
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            _write_run_config(out_dir, cfg)

            print(f"\n=== Match {combo.name} / {ext_name} ===")
            t0 = time.time()
            distractor_dirs = [
                cfg.datasets[d].abs_templates(storage) / ext_name
                for d in combo.distractors
            ]
            try:
                match_cmd.match_dataset(
                    qroot, groot, out_dir,
                    query_regex=combo.queries.regex,
                    query_list=combo.queries.list,
                    gallery_regex=combo.gallery.regex,
                    gallery_list=combo.gallery.list,
                    distractors=distractor_dirs,
                    device=cfg.matching.device,
                    batch_size=cfg.matching.batch_size,
                    min_quality=cfg.matching.min_quality,
                    genuine_mask_mode=combo.genuine_mask_mode,
                    genuine_mask_csv=combo.genuine_mask_csv,
                    threshold_targets=tuple(cfg.matching.threshold_targets),
                )
                status = STATUS_OK
                error  = None
            except match_cmd.MatchError as e:
                print(f"[error] {combo.name}/{ext_name}: {e}")
                status, error = STATUS_ERROR, str(e)
            except Exception as e:
                traceback.print_exc()
                with open(out_dir / "error.log", "w") as fh:
                    traceback.print_exc(file=fh)
                status, error = STATUS_ERROR, f"{type(e).__name__}: {e}"

            elapsed = round(time.time() - t0, 1)
            manifest = _build_manifest(cfg, combo, ext_name, status, elapsed, error)
            with open(out_dir / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

            summary.append({
                "combo": combo.name, "extractor": ext_name,
                "status": status, "elapsed_s": elapsed,
            })

    print("\n=== Match summary ===")
    for s in summary:
        ext_name = s.get('extractor', '')
        elapsed  = f"{s['elapsed_s']}s" if 'elapsed_s' in s else "-"
        print(f"  {s['combo']:<28s} {ext_name:<14s}  {s['status']:<18s}  {elapsed}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv):
    p = argparse.ArgumentParser(
        prog="dmd experiment",
        description="Run multi-extractor / multi-combo experiments from a YAML config.",
    )
    p.add_argument("mode", choices=["extract", "match", "full"],
                   help="extract templates, run matching combos, or both")
    p.add_argument("--config", required=True,
                   help="Path to experiment YAML config")
    p.add_argument("--datasets", default=None,
                   help="Comma-separated dataset names (extract mode); default: all defined")
    p.add_argument("--extractors", default=None,
                   help="Comma-separated extractor names; default: all defined")
    p.add_argument("--combos", default=None,
                   help="Comma-separated combo names (match mode); default: all defined")
    p.add_argument("--force", action="store_true",
                   help="Re-run matching even if metrics.json already exists")
    return p.parse_args(argv)


def run(argv):
    args = _parse_args(argv)

    with open(args.config) as f:
        cfg = parse_config(yaml.safe_load(f))

    extractors = _resolve_subset(args.extractors, cfg.extractors, "extractor")

    if args.mode in ("extract", "full"):
        datasets = _resolve_subset(args.datasets, list(cfg.datasets), "dataset")
        _do_extract(cfg, datasets, extractors)

    if args.mode in ("match", "full"):
        combo_names = [c.name for c in cfg.matching.combos]
        chosen      = _resolve_subset(args.combos, combo_names, "combo")
        combos      = [c for c in cfg.matching.combos if c.name in chosen]
        _do_match(cfg, combos, extractors, force=args.force)
