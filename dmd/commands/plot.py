import argparse
import sys


def _plot_fingerprint(args):
    import cv2
    import matplotlib.pyplot as plt

    from dmd.commands.extract import load_min_file
    import dmd

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: cannot read image: {args.image}")
        sys.exit(1)

    mnt_for_dmd, _ = load_min_file(args.minutiae)
    dmd.plot_mnt(img, mnt_for_dmd)

    if args.output:
        plt.savefig(args.output, bbox_inches="tight", dpi=150)
        print(f"Saved → {args.output}")
    else:
        plt.show()


def _plot_roc(args):
    import csv

    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    roc_file = Path(args.results_dir) / "roc.csv"
    if not roc_file.exists():
        print(f"Error: {roc_file} not found")
        sys.exit(1)

    fpr, tpr = [], []
    with open(roc_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fpr.append(float(row["fpr"]))
            tpr.append(float(row["tpr"]))

    fpr = np.array(fpr)
    tpr = np.array(tpr)
    fnr = 1.0 - tpr
    eer_idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax.scatter([fpr[eer_idx]], [tpr[eer_idx]], color="red", zorder=5,
               label=f"EER = {eer:.3f}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, bbox_inches="tight", dpi=150)
        print(f"Saved → {args.output}")
    else:
        plt.show()


def _plot_cmc(args):
    import csv

    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    cmc_file = Path(args.results_dir) / "cmc.csv"
    if not cmc_file.exists():
        print(f"Error: {cmc_file} not found")
        sys.exit(1)

    ranks, values = [], []
    with open(cmc_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ranks.append(int(row["rank"]))
            values.append(float(row["cmc"]))

    max_rank = args.max_rank if args.max_rank else len(ranks)
    ranks = ranks[:max_rank]
    values = values[:max_rank]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ranks, values, marker="o", markersize=4, linewidth=2)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Identification Rate")
    ax.set_title("CMC Curve")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(ranks)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, bbox_inches="tight", dpi=150)
        print(f"Saved → {args.output}")
    else:
        plt.show()


def _plot_scores(args):
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    out_dir = Path(args.results_dir)
    scores_file = out_dir / "scores.npy"
    mask_file = out_dir / "genuine_mask.npy"

    for f in (scores_file, mask_file):
        if not f.exists():
            print(f"Error: {f} not found")
            sys.exit(1)

    scores = np.load(scores_file).astype(float)
    genuine_mask = np.load(mask_file)

    genuine_scores = scores[genuine_mask]
    impostor_scores = scores[~genuine_mask]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = 60
    ax.hist(impostor_scores, bins=bins, density=True, alpha=0.6, label="Impostor", color="steelblue")
    ax.hist(genuine_scores, bins=bins, density=True, alpha=0.6, label="Genuine", color="tomato")
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distributions")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, bbox_inches="tight", dpi=150)
        print(f"Saved → {args.output}")
    else:
        plt.show()


def run(argv):
    parser = argparse.ArgumentParser(prog="dmd plot")
    sub = parser.add_subparsers(dest="plot_command", required=True)

    # dmd plot fingerprint
    fp = sub.add_parser("fingerprint", help="Visualize fingerprint with minutiae overlay")
    fp.add_argument("--image", required=True, help="Path to fingerprint image")
    fp.add_argument("--minutiae", required=True, help="Path to .min minutiae file")
    fp.add_argument("--output", default=None, help="Save plot to file instead of displaying")

    # dmd plot roc
    rp = sub.add_parser("roc", help="Plot ROC curve from results directory")
    rp.add_argument("--results-dir", required=True, help="Directory containing roc.csv")
    rp.add_argument("--output", default=None, help="Save plot to file instead of displaying")

    # dmd plot cmc
    cp = sub.add_parser("cmc", help="Plot CMC curve from results directory")
    cp.add_argument("--results-dir", required=True, help="Directory containing cmc.csv")
    cp.add_argument("--output", default=None, help="Save plot to file instead of displaying")
    cp.add_argument("--max-rank", type=int, default=None, help="Maximum rank to show")

    # dmd plot scores
    sp = sub.add_parser("scores", help="Plot genuine/impostor score distributions")
    sp.add_argument("--results-dir", required=True, help="Directory containing scores.npy and genuine_mask.npy")
    sp.add_argument("--output", default=None, help="Save plot to file instead of displaying")

    args = parser.parse_args(argv)

    if args.plot_command == "fingerprint":
        _plot_fingerprint(args)
    elif args.plot_command == "roc":
        _plot_roc(args)
    elif args.plot_command == "cmc":
        _plot_cmc(args)
    elif args.plot_command == "scores":
        _plot_scores(args)
