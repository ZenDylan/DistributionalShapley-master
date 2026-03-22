#!/usr/bin/env python3
"""
exp_point_addition.py
Point removal + point addition experiments on CoverType and Adult (2x4 figure).
Six methods: D-SVARM, D-Shapley, TMC-Shapley, LOO, Data Banzhaf (MSR), Random.

Run from repo folder:
    python exp_point_addition.py
"""
import os
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_covtype, fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")

from DistShap import DistShap
from shap_utils import portion_performance, return_model

# -----------------------------------------------------------------------------
# Style (Example.ipynb + prompt)
# -----------------------------------------------------------------------------
STYLE = {
    "dsvarm": {"ls": "dashdot", "c": "purple", "lw": 5, "label": "D-SVARM (Ours)"},
    "dist": {"ls": "solid", "c": "blue", "lw": 5, "label": "D-Shapley"},
    "tmc": {"ls": "dashed", "c": "green", "lw": 5, "label": "TMC-Shapley"},
    "banzhaf": {"ls": "solid", "c": "orange", "lw": 3, "label": "Data Banzhaf (MSR)"},
    "loo": {"ls": "dashed", "c": "gray", "lw": 3, "label": "LOO"},
    "random": {"ls": "dotted", "c": "red", "lw": 3, "label": "Random"},
}

COL_TITLES = [
    "(a) Removing high value data",
    "(b) Removing low value data",
    "(c) Adding high value data",
    "(d) Adding low value data",
]

ROW_LABELS = ["Cover Type", "Adult"]


def load_covertype_example_style(train_size=200, test_size=1000, num_test=1000, seed=0):
    """Example.ipynb Cell 4: fetch_covtype, permute, target-1, split, normalizer from X_dist."""
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

    # DistShap: utility = last num_test rows of passed X_test (see _save_dataset)
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


def load_adult_example_style(train_size=200, test_size=1000, num_test=1000, seed=42):
    """Adult: OrdinalEncoder + binary; same tail split as run_all_experiments."""
    np.random.seed(seed)
    adult = fetch_openml("adult", version=2, as_frame=False)
    X, y_raw = adult.data, adult.target
    if y_raw.dtype.kind in ("U", "S", "O") or not np.issubdtype(y_raw.dtype, np.number):
        y = (y_raw == ">50K").astype(int)
    else:
        y = y_raw.astype(int)

    is_cat = []
    if hasattr(adult, "feature_types") and adult.feature_types is not None:
        is_cat = [
            str(dt).startswith("categorical") or str(dt).startswith("object")
            for dt in adult.feature_types
        ]
    elif hasattr(adult, "categories") and adult.categories is not None:
        cat_names = set(adult.categories.keys())
        is_cat = [adult.feature_names[i] in cat_names for i in range(len(adult.feature_names))]
    else:
        import pandas as pd
        try:
            df = pd.DataFrame(X)
            is_cat = [str(df[i].dtype) == "object" or "categorical" in str(df[i].dtype) for i in df.columns]
        except Exception:
            KNOWN_CAT = {
                "workclass", "education", "marital-status", "occupation",
                "relationship", "race", "sex", "native-country",
                "workclass", "education-num", "marital-status", "occupation",
                "relationship", "race", "sex", "native-country",
            }
            fn = getattr(adult, "feature_names", None)
            is_cat = [str(f).lower() in KNOWN_CAT for f in (fn or [])]
    cat_cols = [i for i, v in enumerate(is_cat) if v]
    if cat_cols:
        enc = OrdinalEncoder(encoded_missing_value=-1)
        X_cat = enc.fit_transform(X[:, cat_cols])
        X_num = X[:, [i for i in range(X.shape[1]) if i not in cat_cols]].astype(float)
        X = np.concatenate([X_num, X_cat], axis=1)

    idxs = np.random.permutation(len(X))
    X, y = X[idxs], y[idxs]

    n = len(X)
    heldout_size = num_test
    pool_end = min(n - (test_size + heldout_size), train_size + 20000)

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_pool = X[train_size:pool_end]
    y_pool = y[train_size:pool_end]
    X_util = X[-(test_size + heldout_size) : -heldout_size]
    y_util = y[-(test_size + heldout_size) : -heldout_size]
    X_heldout = X[-heldout_size:]
    y_heldout = y[-heldout_size:]

    scaler = StandardScaler()
    scaler.fit(X_pool)
    X_train = scaler.transform(X_train)
    X_pool = scaler.transform(X_pool)
    X_util = scaler.transform(X_util)
    X_heldout = scaler.transform(X_heldout)

    X_combined = np.vstack([X_heldout, X_util])
    y_combined = np.concatenate([y_heldout, y_util])

    return X_train, y_train, X_combined, y_combined, X_pool, y_pool


def compute_banzhaf(X_train, y_train, X_util, y_util, max_updates=100, seed=42):
    try:
        from pydvl.valuation.samplers import MSRSampler
        from pydvl.valuation.utility import ModelUtility
        from pydvl.valuation.scorers import SupervisedScorer
        from pydvl.valuation.methods import BanzhafValuation
        from pydvl.valuation.stopping import MinUpdates
        from pydvl.utils import Dataset
        from joblib import parallel_config
    except ImportError:
        print("  WARNING: pyDVL not installed. Banzhaf skipped.")
        return None

    model = LogisticRegression(solver="liblinear", max_iter=500, random_state=666)
    test_ds = Dataset(X_util, y_util)
    scorer = SupervisedScorer("accuracy", test_data=test_ds, default=0.0, range=(0.0, 1.0))
    utility = ModelUtility(model=model, scorer=scorer)
    dataset = Dataset(X_train, y_train)
    valuation = BanzhafValuation(
        utility=utility,
        sampler=MSRSampler(seed=seed),
        stopping=MinUpdates(max_updates),
        progress=False,
    )
    with parallel_config(n_jobs=1):
        result = valuation.fit(dataset)
    return result.values


def point_removal_frac(X_train, y_train, X_heldout, y_heldout, vals, fracs, high_first=True):
    """Removal curve; absolute accuracy on heldout. fracs in [0, 0.5]."""
    n = len(X_train)
    order = np.argsort(vals)[::-1] if high_first else np.argsort(vals)
    accs = []
    for frac in fracs:
        n_remove = int(round(n * frac))
        keep_idx = order[n_remove:]
        if len(keep_idx) == 0:
            accs.append(np.nan)
            continue
        try:
            m = return_model("logistic")
            m.fit(X_train[keep_idx], y_train[keep_idx])
            accs.append(m.score(X_heldout, y_heldout))
        except Exception:
            accs.append(np.nan)
    return np.array(accs)


def run_valuations(X_train, y_train, X_combined, y_combined, X_tot, y_tot, directory, seed=42):
    """Run DistShap (4 methods) + Banzhaf; return dshap and valuation dict (length 200 each)."""
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
        overwrite=True,
    )
    print(f"  DistShap running (dir={directory}) ...")
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

    out = {
        "dsvarm": mean_mem("mem_dsvarm"),
        "dist": mean_mem("mem_dist"),
        "tmc": mean_mem("mem_tmc"),
        "loo": np.asarray(dshap.vals_loo),
    }
    X_util = X_combined[-1000:]
    y_util = y_combined[-1000:]
    out["banzhaf"] = compute_banzhaf(X_train, y_train, X_util, y_util, max_updates=100, seed=seed)
    return dshap, out


def removal_all_curves(X_train, y_train, X_heldout, y_heldout, vals_by_method, fracs, n_rnd=5, seed=0):
    """Returns dict with keys dsvarm_hi, dsvarm_lo, ... and x_rem as fracs*100."""
    rng = np.random.RandomState(seed)
    curves = {}
    for key in ("dsvarm", "dist", "tmc", "loo"):
        v = vals_by_method[key]
        curves[key + "_hi"] = point_removal_frac(
            X_train, y_train, X_heldout, y_heldout, v, fracs, high_first=True
        )
        curves[key + "_lo"] = point_removal_frac(
            X_train, y_train, X_heldout, y_heldout, v, fracs, high_first=False
        )
    bz = vals_by_method.get("banzhaf")
    if bz is not None:
        curves["banzhaf_hi"] = point_removal_frac(
            X_train, y_train, X_heldout, y_heldout, bz, fracs, high_first=True
        )
        curves["banzhaf_lo"] = point_removal_frac(
            X_train, y_train, X_heldout, y_heldout, bz, fracs, high_first=False
        )
    rnd_hi, rnd_lo = [], []
    for _ in range(n_rnd):
        noise = rng.randn(len(X_train))
        rnd_hi.append(
            point_removal_frac(
                X_train, y_train, X_heldout, y_heldout, noise, fracs, high_first=True
            )
        )
        rnd_lo.append(
            point_removal_frac(
                X_train, y_train, X_heldout, y_heldout, noise, fracs, high_first=False
            )
        )
    curves["random_hi"] = np.nanmean(rnd_hi, axis=0)
    curves["random_lo"] = np.nanmean(rnd_lo, axis=0)
    for k in list(curves.keys()):
        curves[k] = np.asarray(curves[k], dtype=float) * 100.0
    return curves


def addition_curve_pack(
    dshap, vals_by_method, X_new, y_new, X_init, y_init, X_heldout, y_heldout, high_first, n_rnd=5, seed=0
):
    """
    Dense x: 0, 2, 4, ...; y = absolute accuracy %.
    Valuations for new points: columns vals[*][100:].
    """
    n_new = len(X_new)
    performance_points = np.arange(0, n_new, 2)
    x_add = np.concatenate([[0.0], performance_points[performance_points > 0].astype(float)])

    def perf(order):
        return portion_performance(
            dshap,
            order,
            performance_points,
            X_new,
            y_new,
            X_init,
            y_init,
            X_heldout,
            y_heldout,
        )

    rng = np.random.RandomState(seed)
    sort = (lambda v: np.argsort(-v)) if high_first else (lambda v: np.argsort(v))

    curves = {}
    for key in ("dsvarm", "dist", "tmc"):
        v = vals_by_method[key][100:]
        curves[key] = perf(sort(v)) * 100.0

    curves["loo"] = perf(sort(vals_by_method["loo"][100:])) * 100.0

    bz = vals_by_method.get("banzhaf")
    if bz is not None:
        curves["banzhaf"] = perf(sort(bz[100:])) * 100.0

    rnd_stack = [perf(rng.permutation(n_new)) * 100.0 for _ in range(n_rnd)]
    curves["random"] = np.mean(rnd_stack, axis=0)

    # ---- Spearman rank correlation diagnostic ----
    from scipy.stats import spearmanr
    base_vals = vals_by_method["dsvarm"][100:]
    print(f"  [addition] Spearman(dsvarm vs others)  high_first={high_first}")
    for key in ("dist", "tmc", "loo"):
        rho, p = spearmanr(base_vals, vals_by_method[key][100:])
        print(f"    dsvarm vs {key:6s}: rho={rho:.4f}  p={p:.4e}")

    min_len = min(len(x_add), *(len(curves[k]) for k in curves))
    x_add = x_add[:min_len]
    for k in curves:
        curves[k] = curves[k][:min_len]
    return x_add, curves


def plot_method_lines(ax, x, curve_dict, suffix):
    """suffix '_hi' / '_lo' for removal; None for addition (keys dsvarm, dist, ...)."""
    if suffix:
        pairs = [
            ("dsvarm", "dsvarm" + suffix),
            ("dist", "dist" + suffix),
            ("tmc", "tmc" + suffix),
            ("banzhaf", "banzhaf" + suffix),
            ("loo", "loo" + suffix),
            ("random", "random" + suffix),
        ]
    else:
        pairs = [
            ("dsvarm", "dsvarm"),
            ("dist", "dist"),
            ("tmc", "tmc"),
            ("banzhaf", "banzhaf"),
            ("loo", "loo"),
            ("random", "random"),
        ]
    for style_key, data_key in pairs:
        if data_key not in curve_dict:
            continue
        y = curve_dict[data_key]
        m = min(len(x), len(y))
        s = STYLE[style_key]
        ax.plot(
            x[:m],
            y[:m],
            linestyle=s["ls"],
            color=s["c"],
            linewidth=s["lw"],
            label=s["label"],
        )

    # Tighten y-axis for addition sub-plots so curves fill the vertical space
    if suffix is None:
        all_vals = [curve_dict[k] for k in curve_dict if k in dict(pairs)]
        if all_vals:
            y_min = min(v.min() for v in all_vals if len(v) > 0)
            y_max = max(v.max() for v in all_vals if len(v) > 0)
            margin = (y_max - y_min) * 0.08
            ax.set_ylim(y_min - margin, y_max + margin)


def process_dataset(name, pack, base_temp, seed_offset=0):
    X_train, y_train, X_combined, y_combined, X_tot, y_tot = pack
    X_init, y_init = X_train[:100], y_train[:100]
    X_new, y_new = X_train[100:], y_train[100:]
    X_heldout = X_combined[:1000]
    y_heldout = y_combined[:1000]

    dir_ds = os.path.join(base_temp, f"exp_point_{name}_{seed_offset}")
    dshap, vals = run_valuations(
        X_train, y_train, X_combined, y_combined, X_tot, y_tot, dir_ds, seed=42 + seed_offset
    )

    fracs = np.linspace(0, 0.5, 50)
    x_rem = fracs * 100.0
    rem = removal_all_curves(
        X_train, y_train, X_heldout, y_heldout, vals, fracs, n_rnd=5, seed=seed_offset
    )

    x_add_hi, add_hi = addition_curve_pack(
        dshap, vals, X_new, y_new, X_init, y_init, X_heldout, y_heldout, high_first=True, seed=seed_offset
    )
    x_add_lo, add_lo = addition_curve_pack(
        dshap, vals, X_new, y_new, X_init, y_init, X_heldout, y_heldout, high_first=False, seed=seed_offset + 7
    )

    return {
        "x_rem": x_rem,
        "rem": rem,
        "x_add_hi": x_add_hi,
        "add_hi": add_hi,
        "x_add_lo": x_add_lo,
        "add_lo": add_lo,
    }


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    base_temp = os.path.join(base_dir, "temp", "exp_point_addition")
    os.makedirs(base_temp, exist_ok=True)

    plt.rcParams["font.size"] = 25

    print("=== CoverType (Example.ipynb style) ===")
    ct_full = load_covertype_example_style()
    print("=== Adult ===")
    ad_full = load_adult_example_style()

    row_data = [
        process_dataset("CoverType", ct_full, base_temp, 0),
        process_dataset("Adult", ad_full, base_temp, 1),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))

    for r, rd in enumerate(row_data):
        x_rem = rd["x_rem"]
        rem = rd["rem"]

        ax = axes[r, 0]
        plot_method_lines(ax, x_rem, rem, "_hi")
        ax.set_ylabel("Prediction accuracy (%)", fontsize=35)
        ax.set_xlabel("Fraction of train data removed (%)", fontsize=35)
        if r == 0:
            ax.set_title(COL_TITLES[0], fontsize=35)
        ax.grid(True, alpha=0.3)
        ax.text(
            -0.28,
            0.5,
            ROW_LABELS[r],
            transform=ax.transAxes,
            fontsize=35,
            va="center",
            ha="center",
            rotation=90,
        )

        ax = axes[r, 1]
        plot_method_lines(ax, x_rem, rem, "_lo")
        ax.set_xlabel("Fraction of train data removed (%)", fontsize=35)
        if r == 0:
            ax.set_title(COL_TITLES[1], fontsize=35)
        ax.grid(True, alpha=0.3)

        ax = axes[r, 2]
        plot_method_lines(ax, rd["x_add_hi"], rd["add_hi"], None)
        ax.set_xlabel("Number of added training points", fontsize=35)
        if r == 0:
            ax.set_title(COL_TITLES[2], fontsize=35)
        ax.grid(True, alpha=0.3)

        ax = axes[r, 3]
        plot_method_lines(ax, rd["x_add_lo"], rd["add_lo"], None)
        ax.set_xlabel("Number of added training points", fontsize=35)
        if r == 0:
            ax.set_title(COL_TITLES[3], fontsize=35)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        fontsize=25,
        bbox_to_anchor=(0.5, 1.02),
    )
    plt.tight_layout(rect=[0.04, 0, 1, 0.94])
    out_pdf = os.path.join(results_dir, "exp_point_addition_2x4.pdf")
    out_png = os.path.join(results_dir, "exp_point_addition_2x4.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close()
    print(f"Saved:\n  {out_pdf}\n  {out_png}")


if __name__ == "__main__":
    main()
