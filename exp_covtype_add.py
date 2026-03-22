#!/usr/bin/env python3
"""
Step 1: CoverType Adding experiments only (high + low value).
Produces: results/exp_covtype_add.pdf + .png

Run:
    python exp_covtype_add.py
"""
import os
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_covtype

warnings.filterwarnings("ignore")

from DistShap import DistShap
from shap_utils import portion_performance

# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------
STYLE = {
    "dsvarm":  {"ls": "dashdot", "c": "purple", "lw": 5, "label": "D-SVARM (Ours)"},
    "dist":    {"ls": "solid",   "c": "blue",   "lw": 5, "label": "D-Shapley"},
    "tmc":     {"ls": "dashed",  "c": "green",   "lw": 5, "label": "TMC-Shapley"},
    "loo":     {"ls": "dashed",  "c": "gray",    "lw": 3, "label": "LOO"},
    "random":  {"ls": "dotted", "c": "red",     "lw": 3, "label": "Random"},
}

COL_TITLES = [
    "(a) Adding high value data",
    "(b) Adding low value data",
]


def load_covertype_example_style(train_size=200, test_size=1000, num_test=1000, seed=0):
    """Strictly Example.ipynb Cell 4."""
    np.random.seed(seed)
    data, target = fetch_covtype(return_X_y=True)
    idxs = np.random.permutation(len(data))
    data, target = data[idxs], target[idxs] - 1

    X_train, y_train = data[:train_size], target[:train_size]
    X_dist = data[train_size : -(test_size + num_test)]
    X_test_block = data[-(test_size + num_test) :]
    y_tail = target[-(test_size + num_test) :]

    normalizer_fn = lambda X: (X - np.mean(X_dist, 0)) / np.clip(np.std(X_dist, 0), 1e-12, None)
    X_train = normalizer_fn(X_train)
    X_dist_n = normalizer_fn(X_dist)
    X_test_block = normalizer_fn(X_test_block)

    X_util = X_test_block[-test_size:]
    y_util = y_tail[-test_size:]
    X_heldout = X_test_block[:-test_size]
    y_heldout = y_tail[:-test_size]

    y_train = y_train.astype(int)
    y_util = y_util.astype(int)
    y_heldout = y_heldout.astype(int)

    X_combined = np.vstack([X_heldout, X_util])
    y_combined = np.concatenate([y_heldout, y_util])
    y_dist = target[train_size : -(test_size + num_test)].astype(int)

    return X_train, y_train, X_combined, y_combined, X_dist_n, y_dist


def run_valuations(X_train, y_train, X_combined, y_combined, X_tot, y_tot, directory, seed=42):
    os.makedirs(directory, exist_ok=True)
    m = len(X_train)
    dshap = DistShap(
        X=X_train,
        y=y_train,
        X_test=X_combined,
        y_test=y_combined,
        num_test=1000,
        X_tot=X_tot,
        y_tot=y_tot,
        model_family="logistic",
        metric="accuracy",
        seed=seed,
        directory=directory,
        overwrite=False,
    )
    print(f"  DistShap running ...")
    t0 = time.time()
    dshap.run(
        dsvarm_run=True,
        dist_run=True,
        tmc_run=True,
        loo_run=True,
        save_every=10,
        err=0.05,
        truncation=m,
        alpha=None,
        max_iters=120,
    )
    print(f"  DistShap finished in {time.time() - t0:.1f}s")
    dshap.load_results(verbose=False)

    def mean_mem(key):
        a = dshap.results.get(key)
        if a is None or len(a) == 0:
            return np.zeros(m)
        return np.mean(a, axis=0)

    vals = {
        "dsvarm": mean_mem("mem_dsvarm"),
        "dist": mean_mem("mem_dist"),
        "tmc": mean_mem("mem_tmc"),
        "loo": np.asarray(dshap.vals_loo),
    }
    return dshap, vals


def addition_curves(
    dshap, vals, X_new, y_new, X_init, y_init, X_heldout, y_heldout,
    high_first, n_rnd=5, seed=0
):
    """
    Dense x: 0, 2, 4, ...  (every 2 points)
    y: absolute accuracy on heldout  (%)
    """
    n_new = len(X_new)
    performance_points = np.arange(0, n_new, 2)   # 0,2,4,...,98
    x_add = np.concatenate([[0.0], performance_points[performance_points > 0].astype(float)])

    def perf(order):
        return portion_performance(
            dshap, order, performance_points,
            X_new, y_new, X_init, y_init, X_heldout, y_heldout,
        )

    rng = np.random.RandomState(seed)
    sort = (lambda v: np.argsort(-v)) if high_first else (lambda v: np.argsort(v))

    curves = {}
    for key in ("dsvarm", "dist", "tmc", "loo"):
        v = vals[key][100:]
        curves[key] = perf(sort(v)) * 100.0

    rnd_stack = [perf(rng.permutation(n_new)) * 100.0 for _ in range(n_rnd)]
    curves["random"] = np.mean(rnd_stack, axis=0)

    # align lengths
    min_len = min(len(x_add), min(len(curves[k]) for k in curves))
    x_add = x_add[:min_len]
    for k in curves:
        curves[k] = curves[k][:min_len]
    return x_add, curves


def plot_curves(ax, x, curves, show_legend=False):
    """Draw all method lines on one axes."""
    pairs = [
        ("dsvarm",  "dsvarm"),
        ("dist",    "dist"),
        ("tmc",     "tmc"),
        ("loo",     "loo"),
        ("random",  "random"),
    ]
    for style_key, data_key in pairs:
        if data_key not in curves:
            continue
        y = curves[data_key]
        m = min(len(x), len(y))
        s = STYLE[style_key]
        ax.plot(x[:m], y[:m],
                linestyle=s["ls"], color=s["c"], linewidth=s["lw"],
                label=s["label"])


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    base_temp = os.path.join(base_dir, "temp", "exp_covtype_add")
    os.makedirs(base_temp, exist_ok=True)

    plt.rcParams["font.size"] = 25

    print("=== Loading CoverType ===")
    X_train, y_train, X_combined, y_combined, X_dist_n, y_dist = load_covertype_example_style()
    print(f"  X_train shape: {X_train.shape}, classes: {np.unique(y_train)}")
    print(f"  X_heldout shape: {X_combined[:1000].shape}")

    # Split: first 100 = X_init, last 100 = X_new
    X_init, y_init = X_train[:100], y_train[:100]
    X_new,  y_new  = X_train[100:], y_train[100:]
    X_heldout = X_combined[:1000]
    y_heldout = y_combined[:1000]
    print(f"  X_init: {X_init.shape},  X_new: {X_new.shape}")

    # ------------------------------------------------------------------
    # Run valuations (only once — cached in base_temp)
    # ------------------------------------------------------------------
    dir_ds = os.path.join(base_temp, "valuations")
    dshap, vals = run_valuations(
        X_train, y_train, X_combined, y_combined,
        X_dist_n, y_dist, dir_ds, seed=42,
    )

    # Quick sanity check
    for k, v in vals.items():
        if v is not None:
            print(f"  vals[{k}]: shape={v.shape}, min={v.min():.4f}, max={v.max():.4f}")

    # ------------------------------------------------------------------
    # Compute addition curves
    # ------------------------------------------------------------------
    print("  Computing addition curves ...")
    x_hi, hi_curves = addition_curves(
        dshap, vals, X_new, y_new, X_init, y_init,
        X_heldout, y_heldout, high_first=True, seed=0,
    )
    x_lo, lo_curves = addition_curves(
        dshap, vals, X_new, y_new, X_init, y_init,
        X_heldout, y_heldout, high_first=False, seed=7,
    )

    # ------------------------------------------------------------------
    # Plot 1x2 figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    ax = axes[0]
    plot_curves(ax, x_hi, hi_curves, show_legend=True)
    ax.set_title(COL_TITLES[0], fontsize=35)
    ax.set_xlabel("Number of added training points", fontsize=35)
    ax.set_ylabel("Prediction accuracy (%)", fontsize=35)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([None, 100.5])

    ax = axes[1]
    plot_curves(ax, x_lo, lo_curves)
    ax.set_title(COL_TITLES[1], fontsize=35)
    ax.set_xlabel("Number of added training points", fontsize=35)
    ax.set_ylabel("Prediction accuracy (%)", fontsize=35)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([None, 100.5])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="upper center", ncol=3, fontsize=25,
               bbox_to_anchor=(0.5, 1.04))

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_pdf = os.path.join(results_dir, "exp_covtype_add.pdf")
    out_png = os.path.join(results_dir, "exp_covtype_add.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close()
    print(f"\nSaved:\n  {out_pdf}\n  {out_png}")


if __name__ == "__main__":
    main()
