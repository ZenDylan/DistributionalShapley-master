"""
run_all_experiments.py
Comprehensive evaluation of 6 data valuation methods across 4 datasets and 6 experiments.

Methods: D-SVARM (Ours), D-Shapley, TMC-Shapley, LOO, Data Banzhaf (MSR), Random
Datasets: CoverType, Adult, MNIST, Phoneme
Experiments:
  1. Adding high-value points
  2. Adding low-value points
  3. Removing high-value points
  4. Removing low-value points
  5. Noisy label detection
  6. AUC vs iteration count
"""

import os
import sys
import time
import warnings
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# Project modules
from DistShap import DistShap
from shap_utils import portion_performance, return_model


# =============================================================================
# Dataset Loaders
# =============================================================================

def load_covtype(train_size=200, num_test=1000, random_state=42):
    """Load CoverType dataset, binary classification (class 1 vs rest)."""
    from sklearn.datasets import fetch_covtype

    np.random.seed(random_state)
    data, target = fetch_covtype(return_X_y=True)
    idxs = np.random.permutation(len(data))
    data, target = data[idxs], target[idxs] - 1  # 0-6

    # Binary: class 1 vs rest -> 1, 0
    y_bin = (target == 1).astype(int)
    idxs2 = np.random.permutation(len(data))
    data, y_bin = data[idxs2], y_bin[idxs2]

    # Normalize before splitting
    scaler = StandardScaler()
    scaler.fit(data[train_size:])  # use non-train as reference
    data = scaler.transform(data)

    X_train = data[:train_size]
    y_train = y_bin[:train_size]
    X_pool = data[train_size:]
    y_pool = y_bin[train_size:]
    X_test_full = X_pool[-num_test:]  # reuse tail of pool as test
    y_test_full = y_pool[-num_test:]

    X_test = X_test_full[:num_test]
    y_test = y_test_full[:num_test]
    X_heldout = X_pool[:len(X_pool) - num_test]
    y_heldout = y_pool[:len(y_pool) - num_test]

    print(f"  CoverType: {len(X_train)} train, {len(X_pool)} pool, {len(X_test)} test, "
          f"{len(X_heldout)} heldout, classes={sorted(set(y_train))}")
    return X_train, y_train, X_test, y_test, X_pool, y_pool, X_heldout, y_heldout


def load_adult(train_size=200, num_test=1000, random_state=42):
    """Load Adult dataset with OrdinalEncoder for categorical features."""
    from sklearn.datasets import fetch_openml

    np.random.seed(random_state)
    adult = fetch_openml('adult', version=2, as_frame=False)
    X, y_raw = adult.data, adult.target
    # Convert string target '>50K' / '<=50K' to binary int 1/0
    if y_raw.dtype.kind in ('U', 'S', 'O') or not np.issubdtype(y_raw.dtype, np.number):
        y = (y_raw == '>50K').astype(int)
    else:
        y = y_raw.astype(int)

    # Identify categorical columns
    is_cat = [str(dt).startswith('categorical') or str(dt).startswith('object')
              for dt in adult.feature_types]
    cat_cols = [i for i, v in enumerate(is_cat) if v]

    # OrdinalEncoder for categoricals, passthrough for numericals
    if cat_cols:
        encoder = OrdinalEncoder(encoded_missing_value=-1)
        X_cat = encoder.fit_transform(X[:, cat_cols])
        X_num = X[:, [i for i in range(X.shape[1]) if i not in cat_cols]].astype(float)
        X = np.concatenate([X_num, X_cat], axis=1)

    # Binary target: 0/1 already
    idxs = np.random.permutation(len(X))
    X, y = X[idxs], y[idxs]

    n = len(X)
    train_end = train_size
    test_end = train_size + num_test
    pool_end = min(n - test_end, train_size + 20000)

    X_train = X[:train_end]
    y_train = y[:train_end]
    X_pool = X[train_end:pool_end]
    y_pool = y[train_end:pool_end]
    X_test_full = X[-test_end:]
    y_test_full = y[-test_end:]

    # Normalize using pool statistics
    scaler = StandardScaler()
    scaler.fit(X_pool)
    X_train = scaler.transform(X_train)
    X_pool = scaler.transform(X_pool)
    X_test_full = scaler.transform(X_test_full)

    X_test = X_test_full[:num_test]
    y_test = y_test_full[:num_test]
    X_heldout = X_test_full[num_test:]
    y_heldout = y_test_full[num_test:]

    print(f"  Adult: {len(X_train)} train, {len(X_pool)} pool, {len(X_test)} test, "
          f"{len(X_heldout)} heldout, classes={sorted(set(y_train))}")
    return X_train, y_train, X_test, y_test, X_pool, y_pool, X_heldout, y_heldout


def load_mnist(train_size=200, num_test=1000, random_state=42):
    """Load MNIST, apply PCA to 32 dims, binary (0-4 vs 5-9)."""
    from sklearn.datasets import fetch_openml

    np.random.seed(random_state)
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)

    idxs = np.random.permutation(len(X))
    X, y = X[idxs], y[idxs]

    # Binary: 0-4 vs 5-9
    y_bin = (y >= 5).astype(int)
    idxs2 = np.random.permutation(len(X))
    X, y_bin = X[idxs2], y_bin[idxs2]

    # PCA to 32 dimensions, fit on pool (non-train) for consistency
    scaler = StandardScaler()
    scaler.fit(X[train_size:])
    X_scaled = scaler.transform(X)
    pca = PCA(n_components=32, random_state=42)
    pca.fit(X_scaled[train_size:])
    X_pca = pca.transform(X_scaled)

    n = len(X_pca)
    train_end = train_size
    test_end = train_size + num_test
    pool_end = min(n - test_end, train_size + 20000)

    X_train = X_pca[:train_end]
    y_train = y_bin[:train_end]
    X_pool = X_pca[train_end:pool_end]
    y_pool = y_bin[train_end:pool_end]
    X_test_full = X_pca[-test_end:]
    y_test_full = y_bin[-test_end:]

    X_test = X_test_full[:num_test]
    y_test = y_test_full[:num_test]
    X_heldout = X_test_full[num_test:]
    y_heldout = y_test_full[num_test:]

    print(f"  MNIST: {len(X_train)} train, {len(X_pool)} pool, {len(X_test)} test, "
          f"{len(X_heldout)} heldout, classes={sorted(set(y_train))}")
    return X_train, y_train, X_test, y_test, X_pool, y_pool, X_heldout, y_heldout


def load_phoneme(train_size=200, num_test=1000, random_state=42):
    """Load Phoneme dataset."""
    from sklearn.datasets import fetch_openml

    np.random.seed(random_state)
    phoneme = fetch_openml('phoneme', version=1, as_frame=False)
    X, y = phoneme.data.astype(float), phoneme.target.astype(int)

    idxs = np.random.permutation(len(X))
    X, y = X[idxs], y[idxs]

    n = len(X)
    train_end = train_size
    test_end = train_size + num_test
    pool_end = min(n - test_end, train_size + 20000)

    X_train = X[:train_end]
    y_train = y[:train_end]
    X_pool = X[train_end:pool_end]
    y_pool = y[train_end:pool_end]
    X_test_full = X[-test_end:]
    y_test_full = y[-test_end:]

    # Normalize using pool statistics
    scaler = StandardScaler()
    scaler.fit(X_pool)
    X_train = scaler.transform(X_train)
    X_pool = scaler.transform(X_pool)
    X_test_full = scaler.transform(X_test_full)

    X_test = X_test_full[:num_test]
    y_test = y_test_full[:num_test]
    X_heldout = X_test_full[num_test:]
    y_heldout = y_test_full[num_test:]

    print(f"  Phoneme: {len(X_train)} train, {len(X_pool)} pool, {len(X_test)} test, "
          f"{len(X_heldout)} heldout, classes={sorted(set(y_train))}")
    return X_train, y_train, X_test, y_test, X_pool, y_pool, X_heldout, y_heldout


DATASETS = {
    'CoverType': load_covtype,
    'Adult': load_adult,
    'MNIST': load_mnist,
    'Phoneme': load_phoneme,
}


# =============================================================================
# Banzhaf (Data Banzhaf via pyDVL MSR)
# =============================================================================

def compute_banzhaf(X_train, y_train, X_test, y_test, n_jobs=1, max_updates=100, seed=42):
    """Compute Data Banzhaf values using pyDVL MSR sampler."""
    try:
        from pydvl.valuation.samplers import MSRSampler
        from pydvl.valuation.utility import ModelUtility
        from pydvl.valuation.scorers import SupervisedScorer
        from pydvl.valuation.methods import BanzhafValuation
        from pydvl.valuation.stopping import MinUpdates
        from pydvl.utils import Dataset
        from joblib import parallel_config
    except ImportError:
        print("  WARNING: pyDVL not installed. Skipping Banzhaf.")
        return None

    # Build utility: LogisticRegression model with accuracy scorer
    model = LogisticRegression(solver='liblinear', max_iter=500, random_state=666)
    scorer = SupervisedScorer(
        "accuracy",
        test_data=Dataset(x=X_test, y=y_test),
        default=0.0,
        range=(0.0, 1.0)
    )
    utility = ModelUtility(model=model, scorer=scorer)

    dataset = Dataset(x=X_train, y=y_train)

    valuation = BanzhafValuation(
        utility=utility,
        sampler=MSRSampler(seed=seed),
        stopping=MinUpdates(max_updates),
        progress=False,
    )

    with parallel_config(n_jobs=n_jobs):
        result = valuation.fit(dataset)

    vals = result.values
    print(f"  Banzhaf: computed {len(vals)} values (updates={result.counts[0] if len(result.counts) else max_updates})")
    return vals


# =============================================================================
# Point Removal Experiment (for Exp 3 & 4)
# =============================================================================

def point_removal(X_train, y_train, X_test, y_test, vals, fracs, model_family='logistic'):
    """
    Remove points from X_train in order of vals, evaluate accuracy on X_test at each fraction.

    Args:
        X_train, y_train: Full training set (n points)
        X_test, y_test: Test set
        vals: Values for all n training points
        fracs: Fractions to remove, e.g. [0.05, 0.10, ..., 0.50]
        model_family: Model type

    Returns:
        accs: List of accuracies after removing each fraction
    """
    n = len(X_train)
    order = np.argsort(vals)[::-1]  # high-value first
    accs = []
    for frac in fracs:
        n_remove = int(n * frac)
        keep_idx = order[n_remove:]
        if len(keep_idx) == 0:
            accs.append(np.nan)
            continue
        model = return_model(model_family)
        model.fit(X_train[keep_idx], y_train[keep_idx])
        accs.append(model.score(X_test, y_test))
    return np.array(accs)


def point_removal_low(X_train, y_train, X_test, y_test, vals, fracs, model_family='logistic'):
    """Remove low-value points first (ascending order)."""
    n = len(X_train)
    order = np.argsort(vals)  # low-value first
    accs = []
    for frac in fracs:
        n_remove = int(n * frac)
        keep_idx = order[n_remove:]
        if len(keep_idx) == 0:
            accs.append(np.nan)
            continue
        model = return_model(model_family)
        model.fit(X_train[keep_idx], y_train[keep_idx])
        accs.append(model.score(X_test, y_test))
    return np.array(accs)


# =============================================================================
# Noisy Label Detection (Exp 5) - Valuation on Noisy Data
# =============================================================================

def compute_noisy_vals(X_train, y_train_orig, flip_idx, X_test, y_test,
                       X_tot, y_tot, methods, directory, model_family='logistic',
                       seed=42, max_iters=50):
    """
    Flip labels at flip_idx, run DistShap methods on noisy data, return valuations.

    Args:
        X_train, y_train_orig: Original training data
        flip_idx: Indices of flipped labels
        methods: list of method flags, e.g. ['dsvarm', 'dist', 'tmc', 'loo']
        max_iters: iterations for each method
    Returns:
        vals_dict: {method_name: values array}
    """
    y_train = y_train_orig.copy()
    y_train[flip_idx] = 1 - y_train[flip_idx]

    flags = {m: True for m in methods}
    dshap = DistShap(
        X=X_train, y=y_train,
        X_test=X_test, y_test=y_test,
        num_test=len(X_test),
        X_tot=X_tot, y_tot=y_tot,
        sources=None,
        model_family=model_family,
        metric='accuracy',
        seed=seed,
        directory=directory,
        overwrite=True,
    )

    dshap.run(
        dsvarm_run=flags.get('dsvarm', False),
        dist_run=flags.get('dist', False),
        tmc_run=flags.get('tmc', False),
        loo_run=flags.get('loo', False),
        save_every=10,
        err=0.05,
        max_iters=max_iters,
    )
    dshap.load_results(verbose=False)

    vals = {}
    if 'dsvarm' in methods:
        vals['dsvarm'] = np.mean(dshap.results['mem_dsvarm'], 0)
    if 'dist' in methods:
        vals['dist'] = np.mean(dshap.results['mem_dist'], 0)
    if 'tmc' in methods:
        vals['tmc'] = np.mean(dshap.results['mem_tmc'], 0)
    if 'loo' in methods:
        vals['loo'] = dshap.vals_loo

    return vals, dshap


def noisy_detection_curves(noisy_vals_dict, flip_idx, n):
    """
    Compute cumulative noise discovery curves from noisy valuations.
    flip_idx: ground truth noisy indices (fixed externally)
    Returns: {method: curve}
    """
    n_flip = len(flip_idx)
    curves = {}
    for key, vals in noisy_vals_dict.items():
        if vals is None:
            continue
        sorted_idx = np.argsort(vals)
        noise_found = []
        n_found = 0
        for i in range(1, n + 1):
            if sorted_idx[i - 1] in flip_idx:
                n_found += 1
            noise_found.append(n_found / n_flip if n_flip > 0 else 0)
        curves[key] = np.array(noise_found)
    return curves


# =============================================================================
# AUC Computation
# =============================================================================

def compute_auc(x, y):
    """Compute normalized AUC using trapezoid rule."""
    if len(x) < 2 or len(y) < 2:
        return 0.0
    # Normalize x to [0, 1]
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-12)
    # Sort by x
    order = np.argsort(x_norm)
    x_s, y_s = x_norm[order], y[order]
    auc = np.trapz(y_s, x_s)
    return auc


def performance_auc(perf_values, fractions):
    """Compute AUC from performance curve."""
    if len(perf_values) < 2:
        return 0.0
    return compute_auc(fractions, perf_values)


# =============================================================================
# Run All Valuation Methods for One Dataset
# =============================================================================

def run_all_methods(X_train, y_train, X_test, y_test, X_tot, y_tot,
                    directory, model_family='logistic', seed=42):
    """
    Run all 6 valuation methods on the training set.

    Returns dict with keys: dsvarm, dist, tmc, loo, banzhaf
    Each entry is a numpy array of shape (n_train,).
    """
    results = {}

    # Create DistShap instance
    dshap = DistShap(
        X=X_train, y=y_train,
        X_test=X_test, y_test=y_test,
        num_test=len(X_test),
        X_tot=X_tot, y_tot=y_tot,
        sources=None,
        sample_weight=None,
        model_family=model_family,
        metric='accuracy',
        seed=seed,
        directory=directory,
        overwrite=False,
    )

    print(f"\n  Running D-SVARM, D-Shapley, TMC-Shapley, LOO...")
    start = time.time()
    dshap.run(
        dsvarm_run=True,
        dist_run=True,
        tmc_run=True,
        loo_run=True,
        save_every=10,
        err=0.05,
    )
    print(f"  DistShap methods done in {time.time() - start:.1f}s")

    # Load results
    dshap.load_results(verbose=False)
    vals_tmc = np.mean(dshap.results['mem_tmc'], 0)
    vals_dist = np.mean(dshap.results['mem_dist'], 0)
    vals_dsvarm = np.mean(dshap.results['mem_dsvarm'], 0)
    vals_loo = dshap.vals_loo

    results['dsvarm'] = vals_dsvarm
    results['dist'] = vals_dist
    results['tmc'] = vals_tmc
    results['loo'] = vals_loo
    results['dshap'] = dshap  # keep reference for portion_performance

    # Data Banzhaf via pyDVL (on full training set)
    print(f"  Computing Data Banzhaf (MSR)...")
    start = time.time()
    vals_banzhaf = compute_banzhaf(X_train, y_train, X_test, y_test, n_jobs=1, max_updates=100, seed=seed)
    print(f"  Banzhaf done in {time.time() - start:.1f}s")
    results['banzhaf'] = vals_banzhaf

    return results


# =============================================================================
# Experiment 1: Adding High-Value Points
# =============================================================================

def exp1_add_high(dshap, vals_dict, X_new, y_new, X_init, y_init,
                  X_heldout, y_heldout):
    """
    Addition experiment: start with X_init, add X_new in high-value order.
    Returns dict of performance curves for each method.
    """
    n_new = len(X_new)
    performance_points = np.arange(0, n_new // 2, n_new // 40)
    if len(performance_points) < 2:
        performance_points = np.arange(0, n_new // 2 + 1)

    perf_curves = {}

    # Use portion_performance for DistShap methods
    def perf_func(order):
        return portion_performance(
            dshap, order, performance_points,
            X_new, y_new, X_init, y_init, X_heldout, y_heldout
        )

    for key in ['dsvarm', 'dist', 'tmc']:
        vals_new = vals_dict[key][100:]  # only X_new (last 100 pts)
        order = np.argsort(-vals_new)
        curve = perf_func(order)
        perf_curves[key] = curve

    # LOO: values for all 200, take last 100
    vals_loo_new = vals_dict['loo'][100:]
    order_loo = np.argsort(-vals_loo_new)
    perf_curves['loo'] = perf_func(order_loo)

    # Banzhaf
    vals_bz_new = vals_dict.get('banzhaf')
    if vals_bz_new is not None:
        order_bz = np.argsort(-vals_bz_new[100:])
        perf_curves['banzhaf'] = perf_func(order_bz)

    # Random
    rnd_curves = []
    for _ in range(5):
        order_rnd = np.random.permutation(n_new)
        rnd_curves.append(perf_func(order_rnd))
    perf_curves['random'] = np.mean(rnd_curves, axis=0)

    x_vals = performance_points / n_new * 100
    return perf_curves, x_vals


# =============================================================================
# Experiment 2: Adding Low-Value Points
# =============================================================================

def exp2_add_low(dshap, vals_dict, X_new, y_new, X_init, y_init,
                 X_heldout, y_heldout):
    """Addition experiment: start with X_init, add X_new in low-value order."""
    n_new = len(X_new)
    performance_points = np.arange(0, n_new // 2, n_new // 40)
    if len(performance_points) < 2:
        performance_points = np.arange(0, n_new // 2 + 1)

    perf_curves = {}

    def perf_func(order):
        return portion_performance(
            dshap, order, performance_points,
            X_new, y_new, X_init, y_init, X_heldout, y_heldout
        )

    for key in ['dsvarm', 'dist', 'tmc']:
        vals_new = vals_dict[key][100:]
        order = np.argsort(vals_new)
        curve = perf_func(order)
        perf_curves[key] = curve

    vals_loo_new = vals_dict['loo'][100:]
    order_loo = np.argsort(vals_loo_new)
    perf_curves['loo'] = perf_func(order_loo)

    vals_bz_new = vals_dict.get('banzhaf')
    if vals_bz_new is not None:
        order_bz = np.argsort(vals_bz_new[100:])
        perf_curves['banzhaf'] = perf_func(order_bz)

    rnd_curves = []
    for _ in range(5):
        order_rnd = np.random.permutation(n_new)
        rnd_curves.append(perf_func(order_rnd))
    perf_curves['random'] = np.mean(rnd_curves, axis=0)

    x_vals = performance_points / n_new * 100
    return perf_curves, x_vals


# =============================================================================
# Experiment 3: Removing High-Value Points
# =============================================================================

def exp3_remove_high(X_train, y_train, X_test, y_test, vals_dict, model_family='logistic'):
    """
    Removal experiment: start with full set, remove high-value points first.
    fracs: remove 5% to 50%.
    """
    fracs = np.arange(0.0, 0.55, 0.05)
    perf_curves = {}

    for key in ['dsvarm', 'dist', 'tmc', 'loo']:
        vals = vals_dict[key]
        curve = point_removal(X_train, y_train, X_test, y_test, vals, fracs, model_family)
        perf_curves[key] = curve

    vals_bz = vals_dict.get('banzhaf')
    if vals_bz is not None:
        curve_bz = point_removal(X_train, y_train, X_test, y_test, vals_bz, fracs, model_family)
        perf_curves['banzhaf'] = curve_bz

    # Random removal
    rnd_curves = []
    for _ in range(5):
        vals_rnd = np.random.randn(len(X_train))
        rnd_curves.append(point_removal(X_train, y_train, X_test, y_test, vals_rnd, fracs, model_family))
    perf_curves['random'] = np.mean(rnd_curves, axis=0)

    return perf_curves, fracs * 100


# =============================================================================
# Experiment 4: Removing Low-Value Points
# =============================================================================

def exp4_remove_low(X_train, y_train, X_test, y_test, vals_dict, model_family='logistic'):
    """Removal experiment: start with full set, remove low-value points first."""
    fracs = np.arange(0.0, 0.55, 0.05)
    perf_curves = {}

    for key in ['dsvarm', 'dist', 'tmc', 'loo']:
        vals = vals_dict[key]
        curve = point_removal_low(X_train, y_train, X_test, y_test, vals, fracs, model_family)
        perf_curves[key] = curve

    vals_bz = vals_dict.get('banzhaf')
    if vals_bz is not None:
        curve_bz = point_removal_low(X_train, y_train, X_test, y_test, vals_bz, fracs, model_family)
        perf_curves['banzhaf'] = curve_bz

    rnd_curves = []
    for _ in range(5):
        vals_rnd = np.random.randn(len(X_train))
        rnd_curves.append(point_removal_low(X_train, y_train, X_test, y_test, vals_rnd, fracs, model_family))
    perf_curves['random'] = np.mean(rnd_curves, axis=0)

    return perf_curves, fracs * 100


# =============================================================================
# Experiment 5: Noisy Label Detection
# =============================================================================

def exp5_noisy_detection(X_train, y_train, X_test, y_test, X_tot, y_tot,
                          flip_frac=0.10, seed=42):
    """
    Flip flip_frac of labels, re-run valuations on noisy data, compute noise discovery curves.
    """
    np.random.seed(seed)
    n = len(y_train)
    n_flip = int(n * flip_frac)
    flip_idx = np.random.permutation(n)[:n_flip]

    print(f"  Flipping {n_flip} labels...")
    dir_noisy = f"./temp/noisy_{seed}"
    os.makedirs(dir_noisy, exist_ok=True)

    # Run all methods on noisy data
    noisy_vals, _ = compute_noisy_vals(
        X_train, y_train, flip_idx, X_test, y_test,
        X_tot, y_tot,
        methods=['dsvarm', 'dist', 'tmc', 'loo'],
        directory=dir_noisy,
        seed=seed,
        max_iters=50,
    )

    # Data Banzhaf on noisy data
    # Flip labels for Banzhaf as well
    y_train_flipped = y_train.copy()
    y_train_flipped[flip_idx] = 1 - y_train_flipped[flip_idx]
    vals_bz = compute_banzhaf(X_train, y_train_flipped, X_test, y_test,
                               n_jobs=1, max_updates=50, seed=seed)
    noisy_vals['banzhaf'] = vals_bz

    # Compute noise discovery curves
    detection_curves = noisy_detection_curves(noisy_vals, set(flip_idx), n)

    # Random ordering
    noise_rnd = []
    for _ in range(5):
        order = np.random.permutation(n)
        n_found = 0
        curve = []
        for i in range(1, n + 1):
            if order[i - 1] in flip_idx:
                n_found += 1
            curve.append(n_found / n_flip if n_flip > 0 else 0)
        noise_rnd.append(curve)
    detection_curves['random'] = np.mean(noise_rnd, axis=0)

    import shutil
    try:
        shutil.rmtree(dir_noisy)
    except:
        pass

    fractions = np.arange(1, n + 1) / n * 100
    return detection_curves, fractions


# =============================================================================
# Experiment 6: AUC vs Iteration Count
# =============================================================================

def exp6_auc_vs_iterations(X_train, y_train, X_test, y_test, X_tot, y_tot,
                            X_init, y_init, X_new, y_new,
                            X_heldout, y_heldout,
                            T_values, model_family='logistic', seed=42):
    """
    Run each method for T iterations and measure AUC of addition performance.
    T_values: [10, 20, 50, 100, 200]
    Returns dict of {method: [auc at T=10, auc at T=20, ...]}
    """
    results_by_T = {}
    n_new = len(X_new)
    # performance_points for AUC computation (consistent across all T)
    frac_pts = np.arange(0, n_new // 2, max(1, n_new // 40)) / n_new * 100

    # Run LOO once (doesn't depend on iteration count, always n+1 evaluations)
    dir_loo = f"./temp/loo_{seed}"
    os.makedirs(dir_loo, exist_ok=True)
    dshap_loo = DistShap(
        X=X_train, y=y_train,
        X_test=X_test, y_test=y_test,
        num_test=len(X_test),
        X_tot=X_tot, y_tot=y_tot,
        sources=None,
        model_family=model_family,
        metric='accuracy',
        seed=seed,
        directory=dir_loo,
        overwrite=True,
    )
    dshap_loo.run(loo_run=True)
    vals_loo_full = dshap_loo.vals_loo
    vals_loo_new = vals_loo_full[100:]
    # LOO AUC for exp6 at each T is the same (it doesn't depend on iteration count)
    # We compute it once using vals_loo_new as the ranking
    # and repeat for each T in the plot
    import shutil as _shutil
    try:
        _shutil.rmtree(dir_loo)
    except:
        pass

    for T in T_values:
        print(f"    T={T}...")
        dir_T = f"./temp/iter_T{T}_{seed}"
        os.makedirs(dir_T, exist_ok=True)

        dshap = DistShap(
            X=X_train, y=y_train,
            X_test=X_test, y_test=y_test,
            num_test=len(X_test),
            X_tot=X_tot, y_tot=y_tot,
            sources=None,
            model_family=model_family,
            metric='accuracy',
            seed=seed,
            directory=dir_T,
            overwrite=True,
        )

        # Run only the methods we need for this experiment
        dshap.run(
            dsvarm_run=True,
            dist_run=True,
            tmc_run=True,
            save_every=10,
            err=0.05,
            max_iters=T,
        )
        dshap.load_results(verbose=False)

        vals_tmc = np.mean(dshap.results['mem_tmc'], 0)
        vals_dist = np.mean(dshap.results['mem_dist'], 0)
        vals_dsvarm = np.mean(dshap.results['mem_dsvarm'], 0)

        aucs = {}
        for name, vals_all in [('dsvarm', vals_dsvarm), ('dist', vals_dist), ('tmc', vals_tmc)]:
            vals_new = vals_all[100:]
            order = np.argsort(-vals_new)
            curve = portion_performance(
                dshap, order, np.arange(0, n_new // 2, max(1, n_new // 40)),
                X_new, y_new, X_init, y_init, X_heldout, y_heldout
            )
            aucs[name] = performance_auc(curve, frac_pts / 100)

        # Random
        rnd_aucs = []
        for _ in range(3):
            order = np.random.permutation(n_new)
            curve = portion_performance(
                dshap, order, np.arange(0, n_new // 2, max(1, n_new // 40)),
                X_new, y_new, X_init, y_init, X_heldout, y_heldout
            )
            rnd_aucs.append(performance_auc(curve, frac_pts / 100))
        aucs['random'] = np.mean(rnd_aucs)

        # LOO: computed once, same for all T
        curve_loo = portion_performance(
            dshap, np.argsort(-vals_loo_new), np.arange(0, n_new // 2, max(1, n_new // 40)),
            X_new, y_new, X_init, y_init, X_heldout, y_heldout
        )
        aucs['loo'] = performance_auc(curve_loo, frac_pts / 100)

        # Banzhaf at different iteration levels
        try:
            vals_bz_full = compute_banzhaf(X_train, y_train, X_test, y_test,
                                           n_jobs=1, max_updates=T, seed=seed)
            if vals_bz_full is not None:
                vals_new_bz = vals_bz_full[100:]
                order_bz = np.argsort(-vals_new_bz)
                curve_bz = portion_performance(
                    dshap, order_bz, np.arange(0, n_new // 2, max(1, n_new // 40)),
                    X_new, y_new, X_init, y_init, X_heldout, y_heldout
                )
                aucs['banzhaf'] = performance_auc(curve_bz, frac_pts / 100)
        except Exception:
            aucs['banzhaf'] = 0.0

        for k, v in aucs.items():
            if k not in results_by_T:
                results_by_T[k] = []
            results_by_T[k].append(v)

    return results_by_T, T_values


# =============================================================================
# Plotting
# =============================================================================

STYLE = {
    'dsvarm':  {'ls': 'dashdot', 'c': 'purple',  'lw': 5, 'label': 'D-SVARM (Ours)'},
    'dist':    {'ls': 'solid',   'c': 'blue',    'lw': 5, 'label': 'D-Shapley'},
    'tmc':     {'ls': 'dashed',  'c': 'green',   'lw': 5, 'label': 'TMC-Shapley'},
    'banzhaf': {'ls': 'solid',   'c': 'orange',  'lw': 3, 'label': 'Data Banzhaf (MSR)'},
    'loo':     {'ls': 'dashed',  'c': 'gray',    'lw': 3, 'label': 'LOO'},
    'random':  {'ls': 'dotted',  'c': 'red',     'lw': 3, 'label': 'Random'},
}


def set_style():
    plt.rcParams['font.size'] = 25
    plt.rcParams['axes.labelsize'] = 35
    plt.rcParams['axes.titlesize'] = 35
    plt.rcParams['legend.fontsize'] = 25


def plot_curves(x, curves, xlabel, ylabel, title, out_pdf, out_png, ylabel2=None):
    """Plot all method curves with consistent style."""
    set_style()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    for key, y in curves.items():
        if key not in STYLE:
            continue
        s = STYLE[key]
        ax.plot(x, y, linestyle=s['ls'], color=s['c'], linewidth=s['lw'], label=s['label'])

    ax.set_xlabel(xlabel, fontsize=35)
    ax.set_ylabel(ylabel, fontsize=35)
    ax.set_title(title, fontsize=35)
    ax.legend(fontsize=25)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_pdf, format='pdf', bbox_inches='tight')
    fig.savefig(out_png, format='png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_pdf} and {out_png}")


def plot_exp6(T_values, aucs_by_method, out_pdf, out_png):
    """Plot AUC vs iteration count for Exp 6."""
    set_style()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    for key in STYLE:
        if key not in aucs_by_method:
            continue
        y = aucs_by_method[key]
        s = STYLE[key]
        ax.plot(T_values, y, linestyle=s['ls'], color=s['c'], linewidth=s['lw'], marker='o', label=s['label'])

    ax.set_xlabel('Number of iterations T', fontsize=35)
    ax.set_ylabel('AUC of addition performance', fontsize=35)
    ax.set_title('Performance vs Iterations', fontsize=35)
    ax.legend(fontsize=25)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_pdf, format='pdf', bbox_inches='tight')
    fig.savefig(out_png, format='png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_pdf} and {out_png}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("Data Valuation: Comprehensive Evaluation")
    print("Methods: D-SVARM, D-Shapley, TMC-Shapley, LOO, Data Banzhaf (MSR), Random")
    print("=" * 70)

    train_size = 200
    num_test = 1000
    model_family = 'logistic'
    seed = 42

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    for ds_name, loader_fn in DATASETS.items():
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds_name}")
        print(f"{'=' * 60}")

        ds_results_dir = os.path.join(results_dir, ds_name)
        os.makedirs(ds_results_dir, exist_ok=True)

        np.random.seed(seed)

        # Load dataset
        print("  Loading data...")
        X_train, y_train, X_test, y_test, X_pool, y_pool, X_heldout, y_heldout = \
            loader_fn(train_size=train_size, num_test=num_test, random_state=seed)

        # Combine pool with train for X_tot (D-Shapley background)
        X_tot = np.vstack([X_pool, X_train])
        y_tot = np.concatenate([y_pool, y_train])

        # Split train into init (first 100) and new (last 100)
        X_init, y_init = X_train[:100], y_train[:100]
        X_new, y_new = X_train[100:], y_train[100:]

        # Directory for DistShap results
        ds_dir = f"./temp/{ds_name}_full_{seed}"
        os.makedirs(ds_dir, exist_ok=True)

        # Run all valuation methods
        print(f"\n  Computing valuations for {ds_name}...")
        vals_dict = run_all_methods(
            X_train, y_train, X_test, y_test, X_tot, y_tot,
            directory=ds_dir, model_family=model_family, seed=seed
        )

        # Also load DistShap instance for portion_performance
        dshap = vals_dict['dshap']

        # ---- Experiment 1: Adding High-Value Points ----
        print(f"\n  Running Exp 1: Adding high-value points...")
        perf1, x1 = exp1_add_high(
            dshap, vals_dict, X_new, y_new, X_init, y_init, X_heldout, y_heldout
        )
        # Normalize relative to initial
        for k in perf1:
            if perf1[k][0] > 0:
                perf1[k] = perf1[k] / perf1[k][0] * 100
        plot_curves(
            x1, perf1,
            xlabel='Fraction of data points added (%)',
            ylabel='Relative prediction accuracy (%)',
            title='Adding high-value points',
            out_pdf=os.path.join(ds_results_dir, 'exp1_add_high.pdf'),
            out_png=os.path.join(ds_results_dir, 'exp1_add_high.png'),
        )

        # ---- Experiment 2: Adding Low-Value Points ----
        print(f"\n  Running Exp 2: Adding low-value points...")
        perf2, x2 = exp2_add_low(
            dshap, vals_dict, X_new, y_new, X_init, y_init, X_heldout, y_heldout
        )
        for k in perf2:
            if perf2[k][0] > 0:
                perf2[k] = perf2[k] / perf2[k][0] * 100
        plot_curves(
            x2, perf2,
            xlabel='Fraction of data points added (%)',
            ylabel='Relative prediction accuracy (%)',
            title='Adding low-value points',
            out_pdf=os.path.join(ds_results_dir, 'exp2_add_low.pdf'),
            out_png=os.path.join(ds_results_dir, 'exp2_add_low.png'),
        )

        # ---- Experiment 3: Removing High-Value Points ----
        print(f"\n  Running Exp 3: Removing high-value points...")
        perf3, x3 = exp3_remove_high(
            X_train, y_train, X_heldout, y_heldout, vals_dict, model_family
        )
        # Convert to % of initial accuracy (first element = initial accuracy)
        for k in perf3:
            if perf3[k][0] > 0 and not np.isnan(perf3[k][0]):
                perf3[k] = perf3[k] / perf3[k][0] * 100
        plot_curves(
            x3, perf3,
            xlabel='Fraction of points removed (%)',
            ylabel='Relative prediction accuracy (%)',
            title='Removing high-value points',
            out_pdf=os.path.join(ds_results_dir, 'exp3_remove_high.pdf'),
            out_png=os.path.join(ds_results_dir, 'exp3_remove_high.png'),
        )

        # ---- Experiment 4: Removing Low-Value Points ----
        print(f"\n  Running Exp 4: Removing low-value points...")
        perf4, x4 = exp4_remove_low(
            X_train, y_train, X_heldout, y_heldout, vals_dict, model_family
        )
        for k in perf4:
            if perf4[k][0] > 0 and not np.isnan(perf4[k][0]):
                perf4[k] = perf4[k] / perf4[k][0] * 100
        plot_curves(
            x4, perf4,
            xlabel='Fraction of points removed (%)',
            ylabel='Relative prediction accuracy (%)',
            title='Removing low-value points',
            out_pdf=os.path.join(ds_results_dir, 'exp4_remove_low.pdf'),
            out_png=os.path.join(ds_results_dir, 'exp4_remove_low.png'),
        )

        # ---- Experiment 5: Noisy Label Detection ----
        print(f"\n  Running Exp 5: Noisy label detection...")
        det5, x5 = exp5_noisy_detection(
            X_train, y_train, X_test, y_test, X_tot, y_tot,
            flip_frac=0.10, seed=seed
        )
        plot_curves(
            x5, det5,
            xlabel='Fraction of data checked (%)',
            ylabel='Cumulative noisy labels discovered (%)',
            title='Noisy label detection',
            out_pdf=os.path.join(ds_results_dir, 'exp5_noise_detection.pdf'),
            out_png=os.path.join(ds_results_dir, 'exp5_noise_detection.png'),
        )

        # ---- Experiment 6: AUC vs Iteration Count ----
        print(f"\n  Running Exp 6: AUC vs iterations...")
        T_values = [10, 20, 50, 100, 200]
        aucs6, Tv = exp6_auc_vs_iterations(
            X_train, y_train, X_test, y_test, X_tot, y_tot,
            X_init, y_init, X_new, y_new,
            X_heldout, y_heldout,
            T_values=T_values, model_family=model_family, seed=seed
        )
        plot_exp6(
            Tv, aucs6,
            out_pdf=os.path.join(ds_results_dir, 'exp6_iterations.pdf'),
            out_png=os.path.join(ds_results_dir, 'exp6_iterations.png'),
        )

        print(f"\n  {ds_name} complete. Results in {ds_results_dir}")
        # Clean up temp directory
        if os.path.exists(ds_dir):
            import shutil
            try:
                shutil.rmtree(ds_dir)
            except:
                pass

    print(f"\n{'=' * 60}")
    print("All experiments complete!")
    print(f"Results saved to: {results_dir}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
