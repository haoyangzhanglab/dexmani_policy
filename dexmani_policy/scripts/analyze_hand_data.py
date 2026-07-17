#!/usr/bin/env python3
"""
Comprehensive analysis of hand action data for VQ-VAE codebook design.

Analyzes hand joint data (12 DoF) from pour and multi_grasp tasks:
1. Per-dimension distribution stats (min/max/mean/std), outlier detection
2. PCA analysis: effective degrees of freedom
3. K-means clustering: natural modes in the data
4. Cross-task distribution comparison
5. Temporal delta analysis: frame-to-frame changes
6. Codebook utilization analysis under data augmentation
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import zarr
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy import stats as scipy_stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Silence sklearn warnings
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# Data Loading
# ==============================================================================
def load_hand_data(zarr_path):
    """Load hand joint data (12 DoF) from a Zarr dataset."""
    z = zarr.open(zarr_path, 'r')
    action_ee = z['data/action_ee'][:]  # (N, 21)
    # Hand joints are dimensions 9-20 (12 DoF)
    hand_data = action_ee[:, 9:21].astype(np.float64)  # (N, 12)
    return hand_data

# ==============================================================================
# 1. Per-dimension Distribution Analysis
# ==============================================================================
def analyze_distributions(data, task_name):
    """Compute min/max/mean/std per dimension, check for outliers using IQR."""
    n_dims = data.shape[1]
    stats = {}
    outlier_info = {}

    for d in range(n_dims):
        dim_data = data[:, d]
        stats[f'dim_{d}'] = {
            'min': float(np.min(dim_data)),
            'max': float(np.max(dim_data)),
            'mean': float(np.mean(dim_data)),
            'std': float(np.std(dim_data)),
            'range': float(np.max(dim_data) - np.min(dim_data)),
            'skewness': float(scipy_stats.skew(dim_data)),
            'kurtosis': float(scipy_stats.kurtosis(dim_data)),
        }

        # Outlier detection using IQR
        q1, q3 = np.percentile(dim_data, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers_low = int(np.sum(dim_data < lower))
        n_outliers_high = int(np.sum(dim_data > upper))
        outlier_pct = (n_outliers_low + n_outliers_high) / len(dim_data) * 100

        outlier_info[f'dim_{d}'] = {
            'iqr': float(iqr),
            'q1': float(q1), 'q3': float(q3),
            'outliers_low': n_outliers_low,
            'outliers_high': n_outliers_high,
            'outlier_pct': round(outlier_pct, 2),
        }

    # Also look at near-zero dimensions (constant or near-constant)
    zero_like_dims = []
    for d in range(n_dims):
        if stats[f'dim_{d}']['std'] < 1e-6:
            zero_like_dims.append(d)

    # Check correlation between dimensions
    corr_matrix = np.corrcoef(data.T)

    overall = {
        'task': task_name,
        'n_samples': int(data.shape[0]),
        'n_dims': n_dims,
        'zero_like_dims': zero_like_dims,
        'global_range': [float(np.min(data)), float(np.max(data))],
        'global_mean_abs': float(np.mean(np.abs(data))),
        'global_std': float(np.std(data)),
        'dim_stats': stats,
        'outlier_info': outlier_info,
    }

    return overall, corr_matrix

# ==============================================================================
# 2. PCA Analysis
# ==============================================================================
def analyze_pca(data, task_name):
    """PCA to find effective degrees of freedom."""
    n_samples, n_dims = data.shape

    # Standardize for PCA
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    pca = PCA()
    pca.fit(data_scaled)

    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    # Find number of components for different variance thresholds
    thresholds = [0.50, 0.80, 0.90, 0.95, 0.99]
    n_for_threshold = {}
    for t in thresholds:
        n = int(np.argmax(cumulative_var >= t)) + 1
        n_for_threshold[f'{int(t*100)}%'] = n

    # Effective rank: number of eigenvalues > 1e-3 of max
    effective_rank = int(np.sum(pca.explained_variance_ > pca.explained_variance_.max() * 1e-3))

    # Participation ratio (effective dimensionality)
    eigenvalues = pca.explained_variance_
    participation_ratio = float(np.sum(eigenvalues)**2 / np.sum(eigenvalues**2))

    result = {
        'task': task_name,
        'n_samples': n_samples,
        'explained_variance_ratio': [float(v) for v in explained_var],
        'cumulative_variance': [float(v) for v in cumulative_var],
        'components_for_threshold': n_for_threshold,
        'effective_rank_1e3': effective_rank,
        'participation_ratio': participation_ratio,
    }

    return result, pca, scaler

# ==============================================================================
# 3. K-Means Clustering Analysis
# ==============================================================================
def analyze_clustering(data, task_name, max_k=16):
    """K-means clustering to find natural modes."""
    n_samples = data.shape[0]

    # Standardize
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Subsample if too large to speed up
    max_samples = 5000
    if n_samples > max_samples:
        rng = np.random.RandomState(42)
        indices = rng.choice(n_samples, max_samples, replace=False)
        data_sub = data_scaled[indices]
    else:
        data_sub = data_scaled

    results = {}
    for k in range(2, max_k + 1, 2):  # Step by 2 for speed (2,4,6,...,16)
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=500)
        labels = km.fit_predict(data_sub)
        inertia = float(km.inertia_)
        sil = float(silhouette_score(data_sub, labels)) if k > 1 else 0.0

        # Cluster sizes
        cluster_sizes = [(labels == i).sum() for i in range(k)]
        min_size, max_size = min(cluster_sizes), max(cluster_sizes)
        size_imbalance = (max_size - min_size) / len(labels)

        # Intra-cluster variance
        intra_var = []
        for i in range(k):
            cluster_data = data_sub[labels == i]
            if len(cluster_data) > 1:
                intra_var.append(float(np.var(cluster_data)))
            else:
                intra_var.append(0.0)

        # Inter-cluster distance (min distance between any two centroids)
        centroids = km.cluster_centers_
        inter_dists = []
        for i in range(k):
            for j in range(i+1, k):
                inter_dists.append(float(np.linalg.norm(centroids[i] - centroids[j])))
        min_inter_dist = float(np.min(inter_dists)) if inter_dists else 0.0

        # Elbow: ratio of inertia reduction
        if len(results) > 0:
            prev_inertia = results[f'k_{list(results.keys())[-1].split("_")[-1]}']['inertia']
            inertia_drop = (prev_inertia - inertia) / prev_inertia if prev_inertia > 0 else 0
        else:
            inertia_drop = 0.0

        results[f'k_{k}'] = {
            'k': k,
            'inertia': inertia,
            'inertia_drop_pct': round(inertia_drop * 100, 2),
            'silhouette': round(sil, 4),
            'cluster_sizes': [int(s) for s in cluster_sizes],
            'size_imbalance': round(size_imbalance, 4),
            'mean_intra_var': float(np.mean(intra_var)),
            'min_inter_dist': min_inter_dist,
            'variance_explained_by_clusters': round(1.0 - inertia / float(np.var(data_sub)), 4),
        }

    return results

# ==============================================================================
# 4. Cross-task Distribution Comparison
# ==============================================================================
def compare_tasks(pour_data, multi_data):
    """Compare hand action distributions between pour and multi_grasp."""
    n_dims = pour_data.shape[1]

    comparisons = {}
    for d in range(n_dims):
        pour_dim = pour_data[:, d]
        multi_dim = multi_data[:, d]

        # KS test for distribution difference
        ks_stat, ks_pval = scipy_stats.ks_2samp(pour_dim, multi_dim)

        # Wasserstein distance (Earth Mover's)
        # Sort both and compute L1 distance between sorted values
        n1, n2 = len(pour_dim), len(multi_dim)
        # Simple: use histograms
        wasserstein_dist = scipy_stats.wasserstein_distance(pour_dim, multi_dim)

        comparisons[f'dim_{d}'] = {
            'pour_mean': float(np.mean(pour_dim)),
            'multi_mean': float(np.mean(multi_dim)),
            'pour_std': float(np.std(pour_dim)),
            'multi_std': float(np.std(multi_dim)),
            'mean_diff': float(np.mean(pour_dim) - np.mean(multi_dim)),
            'std_diff': float(np.std(pour_dim) - np.std(multi_dim)),
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pval),
            'wasserstein': float(wasserstein_dist),
            'significantly_different': bool(ks_pval < 0.01),
        }

    # Joint distribution comparison
    # PCA of combined data to see task separation
    combined = np.vstack([pour_data, multi_data])
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined)

    pca = PCA(n_components=2)
    combined_pca = pca.fit_transform(combined_scaled)
    pour_pca = combined_pca[:len(pour_data)]
    multi_pca = combined_pca[len(pour_data):]

    # Overlap in PCA space
    pour_center = np.mean(pour_pca, axis=0)
    multi_center = np.mean(multi_pca, axis=0)
    center_distance = float(np.linalg.norm(pour_center - multi_center))

    # Bhattacharyya-like overlap estimate from PCA
    pour_std_pca = np.std(pour_pca, axis=0)
    multi_std_pca = np.std(multi_pca, axis=0)

    return comparisons, {
        'center_distance_pca2d': center_distance,
        'pca_explained_var': [float(v) for v in pca.explained_variance_ratio_],
        'pour_pca_mean': [float(v) for v in pour_center],
        'multi_pca_mean': [float(v) for v in multi_center],
        'overlap_metric': float(center_distance / (np.mean(pour_std_pca) + np.mean(multi_std_pca) + 1e-8)),
    }

# ==============================================================================
# 5. Temporal Delta Analysis
# ==============================================================================
def analyze_temporal_deltas(data, task_name):
    """Analyze frame-to-frame hand action changes."""
    deltas = np.diff(data, axis=0)  # (N-1, 12)

    # Per-dimension delta stats
    dim_delta_stats = {}
    for d in range(deltas.shape[1]):
        dim_deltas = deltas[:, d]
        dim_delta_stats[f'dim_{d}'] = {
            'mean_abs': float(np.mean(np.abs(dim_deltas))),
            'std': float(np.std(dim_deltas)),
            'max_abs': float(np.max(np.abs(dim_deltas))),
            'pct_zero': float(np.mean(np.abs(dim_deltas) < 1e-6) * 100),
            'pct_small': float(np.mean(np.abs(dim_deltas) < 0.01) * 100),
        }

    # L2 norm of delta vector
    delta_norms = np.linalg.norm(deltas, axis=1)
    delta_norm_stats = {
        'mean': float(np.mean(delta_norms)),
        'std': float(np.std(delta_norms)),
        'median': float(np.median(delta_norms)),
        'p90': float(np.percentile(delta_norms, 90)),
        'p95': float(np.percentile(delta_norms, 95)),
        'p99': float(np.percentile(delta_norms, 99)),
        'max': float(np.max(delta_norms)),
        'pct_zero': float(np.mean(delta_norms < 1e-6) * 100),
    }

    # Autocorrelation per dimension (lag 1)
    autocorr_lag1 = {}
    for d in range(data.shape[1]):
        series = data[:, d]
        autocorr_lag1[f'dim_{d}'] = float(np.corrcoef(series[:-1], series[1:])[0, 1])

    result = {
        'task': task_name,
        'dim_delta_stats': dim_delta_stats,
        'delta_norm_stats': delta_norm_stats,
        'autocorr_lag1': autocorr_lag1,
    }

    return result, deltas

# ==============================================================================
# 6. Augmentation Impact on Data Distribution
# ==============================================================================
def analyze_augmentation_impact(data, task_name):
    """
    Simulate typical data augmentations and measure their impact on:
    - Per-dimension distribution shift
    - Nearest-neighbor structure change
    - Effective dimensionality change
    """
    n_samples = min(5000, data.shape[0])
    rng = np.random.RandomState(42)
    indices = rng.choice(data.shape[0], n_samples, replace=False)
    data_sub = data[indices].copy()

    results = {}

    # Baseline PCA
    scaler = StandardScaler()
    pca_baseline = PCA().fit(scaler.fit_transform(data_sub))

    # 1. Gaussian noise augmentation
    for noise_std in [0.001, 0.005, 0.01, 0.02, 0.05]:
        noise = rng.randn(*data_sub.shape) * noise_std
        data_aug = data_sub + noise
        pca_aug = PCA().fit(scaler.fit_transform(data_aug))
        participation_ratio = float(
            np.sum(pca_aug.explained_variance_)**2 / np.sum(pca_aug.explained_variance_**2)
        )
        n95 = int(np.argmax(np.cumsum(pca_aug.explained_variance_ratio_) >= 0.95)) + 1

        results[f'gaussian_noise_{noise_std}'] = {
            'aug_type': 'gaussian_noise',
            'strength': noise_std,
            'participation_ratio': participation_ratio,
            'pr_change_pct': round((participation_ratio - float(
                np.sum(pca_baseline.explained_variance_)**2 / np.sum(pca_baseline.explained_variance_**2)
            )) / float(
                np.sum(pca_baseline.explained_variance_)**2 / np.sum(pca_baseline.explained_variance_**2)
            ) * 100, 2),
            'n95_components': n95,
            'std_increase_pct': round((np.std(data_aug) / np.std(data_sub) - 1) * 100, 2),
        }

    # 2. Scaling augmentation
    for scale_range in [(0.95, 1.05), (0.9, 1.1), (0.8, 1.2)]:
        scale_factors = rng.uniform(scale_range[0], scale_range[1], size=data_sub.shape)
        data_aug = data_sub * scale_factors
        pca_aug = PCA().fit(scaler.fit_transform(data_aug))
        participation_ratio = float(
            np.sum(pca_aug.explained_variance_)**2 / np.sum(pca_aug.explained_variance_**2)
        )
        n95 = int(np.argmax(np.cumsum(pca_aug.explained_variance_ratio_) >= 0.95)) + 1

        results[f'scale_{scale_range[0]}_{scale_range[1]}'] = {
            'aug_type': 'scale',
            'strength': f'{scale_range[0]}-{scale_range[1]}',
            'participation_ratio': participation_ratio,
            'n95_components': n95,
            'std_change_pct': round((np.std(data_aug) / np.std(data_sub) - 1) * 100, 2),
        }

    # 3. Quantization sensitivity: measure how many unique "bins" the data spans
    # This indicates how many VQ codes would be needed if we quantize uniformly
    for n_bits in [6, 7, 8, 9, 10]:
        n_bins = 2**n_bits
        # For each dimension, count how many bins are occupied
        occupied_bins_per_dim = []
        for d in range(data_sub.shape[1]):
            dim_data = data_sub[:, d]
            d_min, d_max = dim_data.min(), dim_data.max()
            if d_max - d_min < 1e-8:
                occupied_bins_per_dim.append(0)
            else:
                bins = np.linspace(d_min, d_max, n_bins + 1)
                digitized = np.digitize(dim_data, bins) - 1
                digitized = np.clip(digitized, 0, n_bins - 1)
                occupied_bins_per_dim.append(len(np.unique(digitized)))

        # Joint occupancy: product of per-dim occupancies (upper bound)
        results[f'uniform_quantize_{n_bits}bit'] = {
            'aug_type': 'uniform_quantize',
            'strength': f'{n_bits}bit ({n_bins} bins)',
            'n_bins': n_bins,
            'occupied_bins_per_dim': occupied_bins_per_dim,
            'mean_occupied': float(np.mean(occupied_bins_per_dim)),
            'min_occupied': int(np.min(occupied_bins_per_dim)),
            'max_occupied': int(np.max(occupied_bins_per_dim)),
            'joint_occupancy_upper': int(np.prod(occupied_bins_per_dim)),
            'effective_occupancy': int(np.prod([max(1, o) for o in occupied_bins_per_dim])),
        }

    return results


# ==============================================================================
# Main
# ==============================================================================
def main():
    print("=" * 80)
    print("HAND ACTION DATA ANALYSIS FOR VQ-VAE CODEBOOK DESIGN")
    print("=" * 80)

    base = Path(__file__).resolve().parent.parent.parent / 'robot_data'
    pour_path = base / 'pour.zarr'
    multi_path = base / 'multi_grasp.zarr'

    # Load data
    print("\n[1/6] Loading data...")
    pour_data = load_hand_data(str(pour_path))
    multi_data = load_hand_data(str(multi_path))
    print(f"  pour:         {pour_data.shape[0]} samples, {pour_data.shape[1]} dims, range [{pour_data.min():.4f}, {pour_data.max():.4f}]")
    print(f"  multi_grasp:  {multi_data.shape[0]} samples, {multi_data.shape[1]} dims, range [{multi_data.min():.4f}, {multi_data.max():.4f}]")

    # ==========================================================================
    # 1. Distribution Analysis
    # ==========================================================================
    print("\n[2/6] Per-dimension distribution analysis...")
    pour_stats, pour_corr = analyze_distributions(pour_data, "pour")
    multi_stats, multi_corr = analyze_distributions(multi_data, "multi_grasp")

    print(f"\n  --- pour dim stats ---")
    for d in range(pour_data.shape[1]):
        s = pour_stats['dim_stats'][f'dim_{d}']
        o = pour_stats['outlier_info'][f'dim_{d}']
        is_zero = " [CONSTANT]" if s['std'] < 1e-6 else ""
        is_low = " [LOW-VAR]" if 0 < s['std'] < 1e-4 else ""
        print(f"  dim_{d}: mean={s['mean']:>10.6f}  std={s['std']:>10.6f}  "
              f"range=[{s['min']:>10.6f}, {s['max']:>10.6f}]  "
              f"outliers={o['outlier_pct']:>5.1f}%  "
              f"skew={s['skewness']:>7.3f}{is_zero}{is_low}")

    print(f"\n  --- multi_grasp dim stats ---")
    for d in range(multi_data.shape[1]):
        s = multi_stats['dim_stats'][f'dim_{d}']
        o = multi_stats['outlier_info'][f'dim_{d}']
        is_zero = " [CONSTANT]" if s['std'] < 1e-6 else ""
        is_low = " [LOW-VAR]" if 0 < s['std'] < 1e-4 else ""
        print(f"  dim_{d}: mean={s['mean']:>10.6f}  std={s['std']:>10.6f}  "
              f"range=[{s['min']:>10.6f}, {s['max']:>10.6f}]  "
              f"outliers={o['outlier_pct']:>5.1f}%  "
              f"skew={s['skewness']:>7.3f}{is_zero}{is_low}")

    # Correlation matrix summary
    pour_off_diag = pour_corr[~np.eye(pour_corr.shape[0], dtype=bool)]
    multi_off_diag = multi_corr[~np.eye(multi_corr.shape[0], dtype=bool)]
    print(f"\n  --- Correlation matrix summary ---")
    print(f"  pour:         mean(|r|)={np.mean(np.abs(pour_off_diag)):.3f}, "
          f"max(|r|)={np.max(np.abs(pour_off_diag)):.3f}, "
          f"strongly_correlated(|r|>0.7)={np.sum(np.abs(pour_off_diag)>0.7)}/{len(pour_off_diag)}")
    print(f"  multi_grasp:  mean(|r|)={np.mean(np.abs(multi_off_diag)):.3f}, "
          f"max(|r|)={np.max(np.abs(multi_off_diag)):.3f}, "
          f"strongly_correlated(|r|>0.7)={np.sum(np.abs(multi_off_diag)>0.7)}/{len(multi_off_diag)}")

    # ==========================================================================
    # 2. PCA Analysis
    # ==========================================================================
    print("\n[3/6] PCA analysis...")
    pour_pca_result, pour_pca_model, pour_scaler = analyze_pca(pour_data, "pour")
    multi_pca_result, multi_pca_model, multi_scaler = analyze_pca(multi_data, "multi_grasp")

    for res in [pour_pca_result, multi_pca_result]:
        print(f"\n  --- {res['task']} PCA ---")
        print(f"  Participation ratio: {res['participation_ratio']:.2f} (out of {res['n_samples']} dims)")
        print(f"  Effective rank (1e-3): {res['effective_rank_1e3']}")
        for t, n in res['components_for_threshold'].items():
            print(f"  Components for {t} variance: {n}")
        # Show top eigenvalues
        ev = res['explained_variance_ratio']
        print(f"  Top eigenvalues: PC1={ev[0]:.3f}, PC2={ev[1]:.3f}, PC3={ev[2]:.3f}, "
              f"PC4={ev[3]:.3f}, PC5={ev[4]:.3f}, PC6={ev[5]:.3f}")

    # ==========================================================================
    # 4. Cross-task Comparison
    # ==========================================================================
    print("\n[4/6] Cross-task comparison...")
    comparisons, joint_pca = compare_tasks(pour_data, multi_data)
    print(f"\n  PCA2D center distance: {joint_pca['center_distance_pca2d']:.4f}")
    print(f"  Overlap metric: {joint_pca['overlap_metric']:.4f} (<1 = overlapping, >1 = separated)")
    print(f"\n  Per-dimension KS test:")
    n_sig = 0
    for d in range(12):
        c = comparisons[f'dim_{d}']
        sig_marker = " ***" if c['significantly_different'] else ""
        if c['significantly_different']:
            n_sig += 1
        print(f"  dim_{d}: mean_diff={c['mean_diff']:>10.6f}  "
              f"KS={c['ks_statistic']:.4f}  wasserstein={c['wasserstein']:.4f}{sig_marker}")
    print(f"  {n_sig}/12 dimensions significantly different between tasks (p<0.01)")

    # ==========================================================================
    # 5. Temporal Analysis
    # ==========================================================================
    print("\n[5/6] Temporal delta analysis...")
    pour_temporal, pour_deltas = analyze_temporal_deltas(pour_data, "pour")
    multi_temporal, multi_deltas = analyze_temporal_deltas(multi_data, "multi_grasp")

    for temp in [pour_temporal, multi_temporal]:
        dns = temp['delta_norm_stats']
        print(f"\n  --- {temp['task']} temporal ---")
        print(f"  Delta norm: mean={dns['mean']:.6f}, median={dns['median']:.6f}, "
              f"P90={dns['p90']:.6f}, P95={dns['p95']:.6f}, P99={dns['p99']:.6f}, max={dns['max']:.6f}")
        print(f"  Zero delta pct: {dns['pct_zero']:.2f}%")
        print(f"  Per-dim mean abs delta: ", end="")
        for d in range(12):
            print(f"{temp['dim_delta_stats'][f'dim_{d}']['mean_abs']:.6f}", end=" ")
        print()
        print(f"  Autocorr lag1: ", end="")
        for d in range(12):
            print(f"{temp['autocorr_lag1'][f'dim_{d}']:.3f}", end=" ")
        print()

    # ==========================================================================
    # 3. Clustering Analysis (run after PCA since it's slower)
    # ==========================================================================
    print("\n[...] K-Means clustering analysis...")
    pour_clusters = analyze_clustering(pour_data, "pour")
    multi_clusters = analyze_clustering(multi_data, "multi_grasp")

    for name, clusters in [("pour", pour_clusters), ("multi_grasp", multi_clusters)]:
        print(f"\n  --- {name} clustering ---")
        for k_name, info in clusters.items():
            print(f"  k={info['k']:>2d}: inertia={info['inertia']:.2f}  "
                  f"drop={info['inertia_drop_pct']:>6.2f}%  "
                  f"silhouette={info['silhouette']:.4f}  "
                  f"imbalance={info['size_imbalance']:.4f}  "
                  f"min_inter={info['min_inter_dist']:.4f}  "
                  f"cluster_r2={info['variance_explained_by_clusters']:.4f}")
            # Show cluster sizes for the best k
            if info['silhouette'] >= max(c['silhouette'] for c in clusters.values()) - 0.01:
                best_info = info

        if 'best_info' in dir():
            print(f"  Best k={best_info['k']} (silhouette={best_info['silhouette']:.4f}), "
                  f"cluster sizes: {best_info['cluster_sizes']}")

    # ==========================================================================
    # 6. Augmentation Impact
    # ==========================================================================
    print("\n[6/6] Augmentation impact analysis...")
    pour_aug = analyze_augmentation_impact(pour_data, "pour")
    multi_aug = analyze_augmentation_impact(multi_data, "multi_grasp")

    for name, aug_results in [("pour", pour_aug), ("multi_grasp", multi_aug)]:
        print(f"\n  --- {name} augmentation ---")
        for key, info in aug_results.items():
            if 'pr_change_pct' in info:
                print(f"  {key}: PR={info['participation_ratio']:.2f} "
                      f"(change={info['pr_change_pct']:+.1f}%), "
                      f"N95={info['n95_components']}")
            elif 'occupied_bins_per_dim' in info:
                print(f"  {key}: occupied_bins_per_dim={info['occupied_bins_per_dim']}, "
                      f"mean_occupied={info['mean_occupied']:.1f}")

    # ==========================================================================
    # Final Summary & Recommendations
    # ==========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY & VQ-VAE CODEBOOK RECOMMENDATIONS")
    print("=" * 80)

    # Calculate key VQ-relevant metrics
    pour_eff_dim = pour_pca_result['participation_ratio']
    multi_eff_dim = multi_pca_result['participation_ratio']
    pour_n95 = pour_pca_result['components_for_threshold']['95%']
    multi_n95 = multi_pca_result['components_for_threshold']['95%']

    # Non-constant dimensions
    pour_active_dims = sum(1 for d in range(12) if pour_stats['dim_stats'][f'dim_{d}']['std'] > 1e-4)
    multi_active_dims = sum(1 for d in range(12) if multi_stats['dim_stats'][f'dim_{d}']['std'] > 1e-4)

    # Data volume
    total_samples = pour_data.shape[0] + multi_data.shape[0]

    # Temporal smoothness: how much the data changes frame to frame
    pour_delta_norm_mean = pour_temporal['delta_norm_stats']['mean']
    multi_delta_norm_mean = multi_temporal['delta_norm_stats']['mean']
    pour_data_norm_mean = float(np.mean(np.linalg.norm(pour_data, axis=1)))
    pour_smoothness = pour_delta_norm_mean / (pour_data_norm_mean + 1e-8)

    # Joint data volume for combined codebook
    combined_data = np.vstack([pour_data, multi_data])

    print(f"""
DATA CHARACTERISTICS:
  Total samples:          {total_samples}  (pour: {pour_data.shape[0]}, multi_grasp: {multi_data.shape[0]})
  Active dimensions:      pour={pour_active_dims}/12, multi={multi_active_dims}/12 (std > 1e-4)
  Effective dim (PR):     pour={pour_eff_dim:.1f}, multi={multi_eff_dim:.1f}
  PCA components @95%:    pour={pour_n95}, multi={multi_n95}
  Temporal smoothness:    delta_norm/mean_norm = {pour_smoothness:.4f} (pour)
  Cross-task similarity:  {n_sig}/12 dims significantly different (KS test)
  Data range:             global min={combined_data.min():.4f}, max={combined_data.max():.4f}

KEY FINDINGS FOR VQ-VAE:
  1. The hand action space has only {pour_eff_dim:.1f}-{multi_eff_dim:.1f} effective dimensions
     (PR metric). This means most of the 12-dim space is empty — the data lies on a
     low-dimensional manifold. A VQ codebook can be much smaller than the naive
     uniform quantization would suggest.

  2. {pour_active_dims}/12 dimensions are active (non-constant). The near-constant
     dimensions consume codebook capacity without providing useful information.
     Consider removing or down-weighting constant dimensions.

  3. The temporal smoothness is {pour_smoothness:.4f}, meaning consecutive frames
     are very similar. For a VQ-VAE, this means the encoder should be able to
     compress temporal sequences efficiently.

  4. Cross-task distribution differences exist ({n_sig}/12 dims). A shared codebook
     needs enough capacity to cover both task distributions.

CODEBOOK SIZE RECOMMENDATIONS:
""")

    # Heuristic: for each effective dimension, need at least 2-4 codes
    # to resolve variability. Plus redundancy for noise/quantization error.
    base_codes_per_dim = 4
    recommended_size_pour = base_codes_per_dim ** max(2, int(np.ceil(pour_eff_dim)))
    recommended_size_multi = base_codes_per_dim ** max(2, int(np.ceil(multi_eff_dim)))
    combined_eff_dim = max(pour_eff_dim, multi_eff_dim)
    recommended_size_combined = base_codes_per_dim ** max(2, int(np.ceil(combined_eff_dim)))

    # But also need enough codes to cover data volume
    # Rule of thumb: codes should be 1-5% of data_points for meaningful clustering
    code_to_data_ratio = 0.02
    recommended_from_volume = int(total_samples * code_to_data_ratio)

    # Practical recommendation
    # VQ codes are typically powers of 2
    powers_of_2 = [2**i for i in range(4, 13)]  # 16 to 4096
    suggestions = []
    for p in powers_of_2:
        if p >= 64:  # minimum reasonable
            suggestions.append(p)

    print(f"  Theoretical minimum (from effective dimensionality):")
    print(f"    pour only:      >= {recommended_size_pour}")
    print(f"    multi only:     >= {recommended_size_multi}")
    print(f"    combined:       >= {recommended_size_combined}")
    print(f"  Volume-based (2% of data): >= {recommended_from_volume}")
    print(f"")
    print(f"  Recommended codebook sizes to try: {suggestions}")
    print(f"  Start recommendation: K=256 or K=512 (good balance of capacity vs. utilization)")
    print(f"")
    print(f"  STRUCTURE RECOMMENDATIONS:")
    print(f"  1. Use per-dimension normalization before VQ to handle the wide range")
    print(f"     difference across dimensions (0 to 1.4 range vs ~0 constant).")
    print(f"  2. Consider dimension reduction (PCA) before VQ — project 12D to")
    print(f"     {pour_n95}D (95%) or {pour_eff_dim:.0f}D (participation ratio) to reduce codebook burden.")
    print(f"  3. Use EMA codebook update (not just commitment loss) to ensure codebook")
    print(f"     adapts to the actual data distribution.")
    print(f"  4. Initialize codebook with k-means centroids on the training data")
    print(f"     for faster convergence and better utilization.")
    print(f"  5. Monitor perplexity (exp(-entropy of code usage)) as a health metric.")
    print(f"  6. If codebook utilization is low (< 50%), try:")
    print(f"     - Smaller codebook (K=128 or 256)")
    print(f"     - Lower commitment loss weight")
    print(f"     - Add entropy bonus to loss")
    print(f"     - Apply slight Gaussian noise to encoder output before quantization")

    # Save results
    output = {
        'pour_stats': pour_stats,
        'multi_stats': multi_stats,
        'pour_pca': pour_pca_result,
        'multi_pca': multi_pca_result,
        'cross_task': {f'dim_{d}': comparisons[f'dim_{d}'] for d in range(12)},
        'joint_pca': joint_pca,
        'pour_temporal': pour_temporal,
        'multi_temporal': multi_temporal,
        'pour_clusters': pour_clusters,
        'multi_clusters': multi_clusters,
        'pour_augmentation': pour_aug,
        'multi_augmentation': multi_aug,
        'recommendations': {
            'effective_dim_pour': pour_eff_dim,
            'effective_dim_multi': multi_eff_dim,
            'pca_n95_pour': pour_n95,
            'pca_n95_multi': multi_n95,
            'active_dims': {'pour': pour_active_dims, 'multi': multi_active_dims},
            'suggested_codebook_sizes': suggestions,
            'start_recommendation': 256,
        }
    }

    output_path = str(Path(__file__).resolve().parent / 'hand_data_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nFull results saved to: {output_path}")

if __name__ == '__main__':
    main()
