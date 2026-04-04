#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(path: Path):
    train = {}
    val = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            epoch = row.get("epoch")
            if not epoch:
                continue
            epoch = int(epoch)
            if row.get("train_loss"):
                train[epoch] = {
                    "loss": float(row["train_loss"]),
                    "mIoU": float(row["train_mIoU"]),
                    "F1": float(row["train_F1"]),
                    "OA": float(row["train_OA"]),
                }
            if row.get("val_loss"):
                val[epoch] = {
                    "loss": float(row["val_loss"]),
                    "mIoU": float(row["val_mIoU"]),
                    "F1": float(row["val_F1"]),
                    "OA": float(row["val_OA"]),
                }
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, type=Path, help="metrics.csv path")
    ap.add_argument("--out", required=True, type=Path, help="output png path")
    ap.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="optional summary txt path (default: best_epoch_summary.txt next to --out)",
    )
    args = ap.parse_args()

    train, val = load_metrics(args.metrics)

    def series(data, key):
        epochs = sorted(data.keys())
        return epochs, [data[e][key] for e in epochs]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for ax, key, title in [
        (axes[0], "loss", "Loss"),
        (axes[1], "mIoU", "mIoU"),
        (axes[2], "F1", "F1"),
        (axes[3], "OA", "OA"),
    ]:
        if train:
            e, y = series(train, key)
            ax.plot(e, y, label="train")
        if val:
            e, y = series(val, key)
            ax.plot(e, y, label="val")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)

    if val:
        best_epoch = max(val.items(), key=lambda kv: kv[1]["F1"])[0]
        best = val[best_epoch]
        print("best_epoch", best_epoch)
        print("val_loss", best["loss"])
        print("val_mIoU", best["mIoU"])
        print("val_F1", best["F1"])
        print("val_OA", best["OA"])

        summary_path = args.summary or (args.out.parent / "best_epoch_summary.txt")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w") as f:
            f.write("best_epoch: %d\n" % best_epoch)
            f.write("val_loss: %.6f\n" % best["loss"])
            f.write("val_mIoU: %.6f\n" % best["mIoU"])
            f.write("val_F1: %.6f\n" % best["F1"])
            f.write("val_OA: %.6f\n" % best["OA"])


if __name__ == "__main__":
    main()
