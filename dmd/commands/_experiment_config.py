"""Schema and parser for the YAML config consumed by `dmd experiment`.

Layout (see dmd/examples/experiments_config.yaml):

    storage_root: /storage/jcontreras/data
    output_root:  /storage/jcontreras/data/experiments/ijcb2026/dmd-match
    datasets:
      sd258:
        images_dir:    datasets/fingerprints/SD258/images
        minutiae_root: datasets/fingerprints/SD258/minutiae
        templates_dir: datasets/fingerprints/SD258/templates/dmd
    extractors: [pyfing, fingernet, ...]
    extraction:
      min_quality: 0
      device: cuda
      overwrite: false
    matching:
      min_quality: 0
      device: cuda
      batch_size: 256
      threshold_targets: [0.0001, 0.001, 0.01]
      combos:
        - name: sd258-vs-sd258
          queries: {dataset: sd258, regex: '.*-00-.*'}
          gallery: {dataset: sd258, regex: '.*-01-.*'}
          genuine_mask_mode: auto      # auto | fvc_loo | custom_csv
          # genuine_mask_csv: path/to/file.csv  # custom_csv only
          # distractors: [ts1k]                  # optional pool refs
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class DatasetCfg:
    images_dir:    str
    minutiae_root: str
    templates_dir: str

    def abs_images(self, root):    return Path(root) / self.images_dir
    def abs_minutiae(self, root):  return Path(root) / self.minutiae_root
    def abs_templates(self, root): return Path(root) / self.templates_dir


@dataclass
class ExtractionCfg:
    min_quality: int = 0
    device:      str = "cuda"
    overwrite:   bool = False


@dataclass
class TemplateSelector:
    """Selects a subset of a dataset's templates for queries / gallery."""
    dataset: str
    regex:   Optional[str] = None
    list:    Optional[str] = None  # path to .txt file with one rel path per line


@dataclass
class ComboCfg:
    name:              str
    queries:           TemplateSelector
    gallery:           TemplateSelector
    genuine_mask_mode: str = "auto"
    genuine_mask_csv:  Optional[str] = None
    distractors:       List[str] = field(default_factory=list)


@dataclass
class MatchingCfg:
    min_quality:       int = 0
    device:            str = "cuda"
    batch_size:        int = 256
    threshold_targets: List[float] = field(default_factory=lambda: [0.0001, 0.001, 0.01])
    combos:            List[ComboCfg] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    storage_root: str
    output_root:  str
    datasets:     Dict[str, DatasetCfg]
    extractors:   List[str]
    extraction:   ExtractionCfg
    matching:     MatchingCfg


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_selector(d):
    if not isinstance(d, dict) or "dataset" not in d:
        raise ValueError(f"Selector must be a dict with 'dataset' key: got {d!r}")
    return TemplateSelector(
        dataset=d["dataset"],
        regex=d.get("regex"),
        list=d.get("list"),
    )


def _parse_combo(d):
    required = ("name", "queries", "gallery")
    for k in required:
        if k not in d:
            raise ValueError(f"Combo missing required key {k!r}: {d!r}")
    return ComboCfg(
        name=d["name"],
        queries=_parse_selector(d["queries"]),
        gallery=_parse_selector(d["gallery"]),
        genuine_mask_mode=d.get("genuine_mask_mode", "auto"),
        genuine_mask_csv=d.get("genuine_mask_csv"),
        distractors=list(d.get("distractors", [])),
    )


def parse_config(d):
    """Parse a dict (typically from yaml.safe_load) into an ExperimentConfig."""
    for k in ("storage_root", "output_root", "datasets", "extractors"):
        if k not in d:
            raise ValueError(f"Missing required top-level key: {k!r}")

    datasets = {
        name: DatasetCfg(
            images_dir=cfg["images_dir"],
            minutiae_root=cfg["minutiae_root"],
            templates_dir=cfg["templates_dir"],
        )
        for name, cfg in d["datasets"].items()
    }

    ex = d.get("extraction", {}) or {}
    extraction = ExtractionCfg(
        min_quality=int(ex.get("min_quality", 0)),
        device=str(ex.get("device", "cuda")),
        overwrite=bool(ex.get("overwrite", False)),
    )

    mt = d.get("matching", {}) or {}
    matching = MatchingCfg(
        min_quality=int(mt.get("min_quality", 0)),
        device=str(mt.get("device", "cuda")),
        batch_size=int(mt.get("batch_size", 256)),
        threshold_targets=[float(x) for x in mt.get("threshold_targets", [0.0001, 0.001, 0.01])],
        combos=[_parse_combo(c) for c in mt.get("combos", [])],
    )

    cfg = ExperimentConfig(
        storage_root=str(d["storage_root"]),
        output_root=str(d["output_root"]),
        datasets=datasets,
        extractors=list(d["extractors"]),
        extraction=extraction,
        matching=matching,
    )

    # Validate cross-references
    _validate(cfg)
    return cfg


def _validate(cfg):
    known = set(cfg.datasets)
    for combo in cfg.matching.combos:
        for sel in (combo.queries, combo.gallery):
            if sel.dataset not in known:
                raise ValueError(
                    f"Combo {combo.name!r} references unknown dataset {sel.dataset!r}"
                )
        for d in combo.distractors:
            if d not in known:
                raise ValueError(
                    f"Combo {combo.name!r} references unknown distractor dataset {d!r}"
                )
        if combo.genuine_mask_mode not in ("auto", "fvc_loo", "custom_csv"):
            raise ValueError(
                f"Combo {combo.name!r}: invalid genuine_mask_mode {combo.genuine_mask_mode!r}"
            )
        if combo.genuine_mask_mode == "custom_csv" and not combo.genuine_mask_csv:
            raise ValueError(
                f"Combo {combo.name!r}: genuine_mask_mode=custom_csv requires genuine_mask_csv"
            )


def to_dict(cfg):
    """Serialize back to a plain dict (for snapshotting in run_config.yaml)."""
    return asdict(cfg)
