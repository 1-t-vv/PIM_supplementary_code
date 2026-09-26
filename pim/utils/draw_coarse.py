import re
import argparse
import os
from pathlib import Path

_MPLCONFIGDIR = Path(__file__).resolve().parents[1] / "out" / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Allow metric keys with numbers, such as l1/edge/cen.
KV_RE = re.compile(r"([A-Za-z0-9_]+)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
EPOCH_RE = re.compile(r"Epoch\s+(\d+)\s*:?", re.IGNORECASE)
METRIC_LABELS = {
    "l1": "L1 Error",
    "edge": "Edge Loss",
    "cen": "Center Loss",
}

# Backward-compatible eval patterns for older logs.
EVAL_TRIGGER_RE = re.compile(r"Epoch\s+(\d+):\s+Running evaluation on test set", re.IGNORECASE)
EVAL_AVG_RE = re.compile(r"\[Eval\]\s+Test Set Avg:\s*(.+?)\s*\(samples=\d+\)", re.IGNORECASE)
EVAL_OLD_L1_RE = re.compile(r"\[Eval\]\s+Test Set Average L1:\s*([-+0-9.eE]+)", re.IGNORECASE)


def parse_kvs(s: str):
    d = {}
    for k, v in KV_RE.findall(s):
        d[k.lower()] = float(v)
    if "edge" not in d and "rad" in d:
        d["edge"] = d["rad"]
    return d


def parse_log(log_path: Path, metrics=("loss", "l1", "edge", "cen")):
    train = {m: ([], []) for m in metrics}  # m -> (epochs, values)
    evalm = {m: ([], []) for m in metrics}

    pending_eval_epoch = None
    last_train_epoch = None

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            me = EPOCH_RE.search(line)
            lower = line.lower()
            if me and "train" in lower and ("loss=" in lower or "avg_loss=" in lower):
                e = int(me.group(1))
                kv = parse_kvs(line)
                if "loss" not in kv and "avg_loss" in kv:
                    kv["loss"] = kv["avg_loss"]
                last_train_epoch = e
                for m in metrics:
                    if m in kv:
                        train[m][0].append(e)
                        train[m][1].append(kv[m])
                continue

            if me and ("eval" in lower or "validate" in lower or "val" in lower) and (
                "error=" in lower or "loss=" in lower or "l1=" in lower
            ):
                e = int(me.group(1))
                kv = parse_kvs(line)
                for m in metrics:
                    if m in kv:
                        evalm[m][0].append(e)
                        evalm[m][1].append(kv[m])
                continue

            mt = EVAL_TRIGGER_RE.search(line)
            if mt:
                pending_eval_epoch = int(mt.group(1))
                continue

            mt = EVAL_AVG_RE.search(line)
            if mt:
                kv = parse_kvs(mt.group(1))
                e = pending_eval_epoch if pending_eval_epoch is not None else last_train_epoch
                if e is not None:
                    for m in metrics:
                        if m in kv:
                            evalm[m][0].append(e)
                            evalm[m][1].append(kv[m])
                pending_eval_epoch = None
                continue

            mt = EVAL_OLD_L1_RE.search(line)
            if mt:
                e = pending_eval_epoch if pending_eval_epoch is not None else last_train_epoch
                if e is not None:
                    evalm["l1"][0].append(e)
                    evalm["l1"][1].append(float(mt.group(1)))
                pending_eval_epoch = None
                continue

    return train, evalm


def apply_skip(xs, ys, skip_epoch):
    if not xs:
        return [], []
    fxs, fys = [], []
    for x, y in zip(xs, ys):
        if x > skip_epoch:
            fxs.append(x)
            fys.append(y)
    return fxs, fys


def plot_metric(ax, name, train_xy, eval_xy):
    tx, ty = train_xy
    ex, ey = eval_xy
    label = METRIC_LABELS.get(name, name)

    any_line = False
    if tx:
        ax.plot(tx, ty, label=f"Training {label}", linewidth=2.0)
        any_line = True
    if ex:
        ax.plot(ex, ey, label=f"Validation {label}", linewidth=1.6, linestyle="--", marker="o", markersize=3)
        any_line = True

    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
    if any_line:
        ax.legend(loc="best", fontsize=9)


def save_plot(log_path: str | Path, out_path: str | Path, skip: int = 0):
    log_path = Path(log_path)
    out_path = Path(out_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    metrics = ("l1", "edge", "cen")
    train, evalm = parse_log(log_path, metrics=metrics)

    for m in metrics:
        train[m] = apply_skip(train[m][0], train[m][1], skip)
        evalm[m] = apply_skip(evalm[m][0], evalm[m][1], skip)

    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 9), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, m in zip(axes, metrics):
        plot_metric(ax, m, train[m], evalm[m])

    axes[-1].set_xlabel("Epoch")
    fig.suptitle(f"Coarse Training and Validation Metrics (skip <= {skip})")

    all_x = []
    for m in metrics:
        all_x += train[m][0]
        all_x += evalm[m][0]
    if all_x:
        axes[-1].set_xlim(left=min(all_x))

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default="out/multi_sphere_flat/logs/coarse_train.log")
    parser.add_argument("--out", type=str, default="out/multi_sphere_flat/plots/coarse_training_metrics.png")
    parser.add_argument("--skip", type=int, default=0, help="skip epochs <= this value")
    args = parser.parse_args()

    out_path = save_plot(args.log, args.out, skip=args.skip)
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()
