#!/usr/bin/env python3
"""Print alpha x n_clusters grids of Silhouette, seed ARI, and four R scores.

Imports the compute-only wrapper in cluster_analysis.py. Reads reduced JSONs
and clustering.yaml. Save one complete reference-seed report per parameter
pair and stage under cluster_grid_search_results, then print the grids and
save one alpha line-and-marker PDF per selected stage in that same root.
No extra KMeans fits are needed.
Use both --no-save-results and --no-save-alpha-curves for a print-only run.
The normal cluster_analysis.py CLI and pipeline.sh remain separate.
"""

import argparse
import json
import shutil
import sys
from copy import deepcopy
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from cluster_analysis import (
    build_features,
    prepare_clustering,
    run_clustering,
    save_clustering_run,
    validate_alpha,
    validate_n_clusters,
    validate_random_seed,
)


def validated_list(values, name, validator):
    """Preserve caller order; reject empty/duplicate values and scalar strings."""
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a nonempty list, not a string")
    try:
        values = list(values)
    except TypeError:
        raise ValueError(f"{name} must be a nonempty iterable") from None
    if not values:
        raise ValueError(f"{name} must not be empty")
    normalized = [validator(value) for value in values]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicate values")
    return normalized


def pairwise_seed_ari(labels_by_seed):
    """Compare all unordered, distinct seed pairs on the SAME language rows.

    This function never fits a model. Cluster-number permutations do not
    change ARI. With one seed there are no pairs, so both summaries are NaN.
    """
    items = [(seed, np.asarray(labels)) for seed, labels in labels_by_seed.items()]
    if not items:
        raise ValueError("At least one seed's label vector is required")
    shape = items[0][1].shape
    if len(shape) != 1 or shape[0] == 0:
        raise ValueError("Seed labels must be nonempty one-dimensional vectors")
    if any(labels.shape != shape for _, labels in items):
        raise ValueError("All seed label vectors must align with the same language rows")
    details = []
    for (seed_a, labels_a), (seed_b, labels_b) in combinations(items, 2):
        value = float(adjusted_rand_score(labels_a, labels_b))
        if not np.isfinite(value):
            raise ValueError(f"Non-finite ARI for seed pair ({seed_a}, {seed_b})")
        details.append({"seed_a": seed_a, "seed_b": seed_b, "ARI": value})
    values = [item["ARI"] for item in details]
    mean = float(np.mean(values)) if values else float("nan")
    minimum = float(np.min(values)) if values else float("nan")
    return mean, minimum, details


def format_pairwise_ari(mean_matrix, min_matrix):
    """Format a mean/min cell without discarding the numeric result matrices."""
    if (not mean_matrix.index.equals(min_matrix.index)
            or not mean_matrix.columns.equals(min_matrix.columns)):
        raise ValueError("ARI mean and minimum matrices must have matching axes")
    rows = []
    for mean_row, min_row in zip(mean_matrix.to_numpy(), min_matrix.to_numpy(), strict=True):
        rows.append([
            f"{mean:.6f} / {minimum:.6f}"
            if np.isfinite(mean) and np.isfinite(minimum) else "N/A / N/A"
            for mean, minimum in zip(mean_row, min_row, strict=True)
        ])
    return pd.DataFrame(rows, index=mean_matrix.index, columns=mean_matrix.columns)


def build_variance_reference_blocks(frame):
    """Fixed, unweighted evaluation blocks on the fitted-language population.

    Q is standardized log10(quantity); S is sqrt(p / 2) for the normalized
    source proportions, after the existing fallback policy. Neither block
    contains alpha. In particular, alpha=0 or 1 never erases an evaluation
    block. Use the same blocks for every alpha, k, and seed within a stage.
    """
    values = frame.to_numpy(dtype=float)
    if values.ndim != 2 or len(values) == 0 or values.shape[1] < 2:
        raise ValueError("Reference blocks require nonempty quantity and source columns")
    if not np.isfinite(values).all():
        raise ValueError("Reference block values must be finite")
    proportions = values[:, 1:]
    if ((proportions < 0) | (proportions > 1)).any():
        raise ValueError("Reference source proportions must lie between 0 and 1")
    if not np.allclose(proportions.sum(axis=1), 1):
        raise ValueError("Reference source proportions must sum to one")
    quantity = StandardScaler().fit_transform(values[:, :1])
    sources = np.sqrt(proportions / 2.0)
    return quantity, sources


def block_explained_variance(block, labels):
    """Compute R=1-W/T for a fixed block and an existing partition, without fit.

    T = sum_i ||x_i - mean(x)||^2.
    W = sum_c sum_{i in c} ||x_i - mean(x in c)||^2.
    Means are computed IN THIS BLOCK (also for sqrt-transformed sources),
    not by transforming raw source means or reusing weighted KMeans centers.
    Languages have equal weight. A zero-variance block yields NaN, not 0/1.
    """
    values = np.asarray(block, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if (values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0
            or not np.isfinite(values).all()):
        raise ValueError("Variance block must be a nonempty finite 1D/2D array")
    labels = np.asarray(labels)
    if (labels.shape != (len(values),)
            or not np.issubdtype(labels.dtype, np.integer) or (labels < 0).any()):
        raise ValueError("Variance labels must be aligned nonnegative integers; no Cluster -1")

    # Translation improves numerical stability and makes exactly constant
    # blocks exactly zero, even if a floating-point mean would round slightly.
    shifted = values - values[0]
    centered = shifted - shifted.mean(axis=0)
    total = float(np.square(centered).sum())
    if not np.isfinite(total):
        raise ValueError("Non-finite total sum of squares")
    if total == 0:
        return float("nan")
    within = 0.0
    for cluster in np.unique(labels):
        members = centered[labels == cluster]
        within += float(np.square(members - members.mean(axis=0)).sum())
    if not np.isfinite(within):
        raise ValueError("Non-finite within-cluster sum of squares")
    value = 1.0 - within / total
    # Only remove tiny roundoff outside the theoretical [0, 1] range.
    if not np.isfinite(value) or value < -1e-10 or value > 1.0 + 1e-10:
        raise ValueError(f"Invalid block explained-variance ratio: {value}")
    return float(np.clip(value, 0.0, 1.0))


def block_total_sum_of_squares(block):
    """Fixed total variation before alpha weighting, on the fitted languages."""
    values = np.asarray(block, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if (values.ndim != 2 or 0 in values.shape
            or not np.isfinite(values).all()):
        raise ValueError("Variance block must be a nonempty finite 1D/2D array")
    shifted = values - values[0]
    total = float(np.square(shifted - shifted.mean(axis=0)).sum())
    if not np.isfinite(total):
        raise ValueError("Non-finite total sum of squares")
    return total


def combined_explained_variance(r_q, r_s, total_q, total_s, alpha):
    """Return balanced and current-alpha-space R for an existing partition.

    balanced = (R_Q + R_S) / 2, undefined if either block has zero variance.
    joint = (alpha*T_Q*R_Q + (1-alpha)*T_S*R_S)
            / (alpha*T_Q + (1-alpha)*T_S).
    Zero-weight/zero-variance terms contribute nothing to joint, even when
    their individual R is undefined. A zero joint denominator yields NaN.
    This evaluates fixed labels and never changes the clustering or centers.
    """
    alpha = validate_alpha(alpha)
    totals = [float(total_q), float(total_s)]
    ratios = [float(r_q), float(r_s)]
    for index, (total, ratio) in enumerate(zip(totals, ratios, strict=True)):
        if not np.isfinite(total) or total < 0:
            raise ValueError("Block total sums of squares must be finite and nonnegative")
        if np.isinf(ratio):
            raise ValueError("Explained-variance ratios cannot be infinite")
        if total == 0:
            ratios[index] = float("nan")
        elif not np.isfinite(ratio) or not -1e-10 <= ratio <= 1.0 + 1e-10:
            raise ValueError("A positive-variance block requires a finite R in [0,1]")
        else:
            ratios[index] = float(np.clip(ratio, 0.0, 1.0))
    balanced = (0.5 * ratios[0] + 0.5 * ratios[1]
                if all(np.isfinite(ratios)) else float("nan"))
    weights = [alpha * totals[0], (1.0 - alpha) * totals[1]]
    denominator = sum(weights)
    if not np.isfinite(denominator):
        raise ValueError("Non-finite weighted total sum of squares")
    # Exclude zero contributions explicitly: 0 * NaN would contaminate the sum.
    joint = (sum((weight / denominator) * ratio
                 for weight, ratio in zip(weights, ratios, strict=True) if weight > 0)
             if denominator > 0 else float("nan"))
    return balanced, joint


def format_resource_explained_variance(r_q_matrix, r_s_matrix,
                                      r_balanced_matrix, r_joint_matrix):
    """Display R_Q / R_S / R_balanced / R_joint to THREE decimal places.

    Round only the display strings; all numeric matrices retain full precision.
    Each undefined component is displayed as N/A independently of the others.
    """
    matrices = (r_q_matrix, r_s_matrix, r_balanced_matrix, r_joint_matrix)
    if any(not r_q_matrix.index.equals(matrix.index)
           or not r_q_matrix.columns.equals(matrix.columns) for matrix in matrices[1:]):
        raise ValueError("All four explained-variance matrices must have matching axes")
    def number(value):
        return f"{value:.3f}" if np.isfinite(value) else "N/A"
    rows = [
        [" / ".join(number(value) for value in cell)
         for cell in zip(*matrix_rows, strict=True)]
        for matrix_rows in zip(*(matrix.to_numpy() for matrix in matrices), strict=True)
    ]
    return pd.DataFrame(rows, index=r_q_matrix.index, columns=r_q_matrix.columns)



def save_alpha_curves_pdf(stage, matrix, output_path):
    """One PDF page per stage, one panel per K, four seed-mean lines with markers.

    Reuse full-precision R matrices: never refit or round plotting values.
    All panels share a y scale; NaN means undefined and breaks lines, not zero.
    The four labels remain in the legend even if a whole series is undefined.
    """
    if stage not in ("pretrain", "posttrain"):
        raise ValueError("stage must be 'pretrain' or 'posttrain'")
    output_path = Path(output_path)
    if output_path.suffix.casefold() != ".pdf":
        raise ValueError("Alpha curves must be saved to a .pdf path")
    if not output_path.parent.is_dir():
        raise NotADirectoryError(output_path.parent)
    specs = (
        ("R_Q_mean", r"$R_Q$", "o"),
        ("R_S_mean", r"$R_S$", "s"),
        ("R_balanced_mean", r"$R_{\mathrm{balanced}}$", "^"),
        ("R_joint_mean", r"$R_{\mathrm{joint}}$", "D"),
    )
    # Validate before opening the output file or allocating a figure.
    alphas = validated_list(matrix.index, "plot alphas", validate_alpha)
    clusters = validated_list(matrix.columns, "plot n_clusters", validate_n_clusters)
    order = np.argsort(np.asarray(alphas), kind="stable")
    x = np.asarray(alphas, dtype=float)[order]
    series = []
    for key, label, marker in specs:
        values = matrix.attrs.get(key)
        if not isinstance(values, pd.DataFrame):
            raise ValueError(f"Missing numeric alpha-curve matrix: {key}")
        if (not values.index.equals(matrix.index)
                or not values.columns.equals(matrix.columns)):
            raise ValueError(f"Alpha-curve axes do not match: {key}")
        numbers = values.to_numpy(dtype=float)
        finite = np.isfinite(numbers)
        if (np.isinf(numbers).any() or (numbers[finite] < -1e-10).any()
                or (numbers[finite] > 1.0 + 1e-10).any()):
            raise ValueError(f"{key} must contain ratios in [0,1] or NaN")
        series.append((numbers[order], label, marker))

    # Lazy import: the new plotting path is never entered when disabled.
    import matplotlib.pyplot as plt

    n_panels = len(clusters)
    ncols = min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols
    height = 3.5 * nrows + 0.9
    fig = None
    temporary = None
    try:
        # Local rc context keeps ordinary cluster_analysis.py styling unchanged.
        with plt.rc_context({"pdf.fonttype": 42}):
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(4.6 * ncols, height),
                squeeze=False, sharex=True, sharey=True,
            )
            axes_flat = axes.ravel()
            margin = max(float(x[-1] - x[0]) * 0.06, 0.02)
            x_limits = (max(-0.02, float(x[0]) - margin),
                        min(1.02, float(x[-1]) + margin))
            # Slice before strict zip: e.g. five panels occupy a six-axis grid.
            for column, (ax, k) in enumerate(zip(axes_flat[:n_panels], clusters, strict=True)):
                any_defined = False
                for numbers, label, marker in series:
                    y = numbers[:, column]
                    valid = np.isfinite(y)
                    any_defined = any_defined or bool(valid.any())
                    # Keep NaNs so a line never bridges an undefined alpha value.
                    # Markers retain the scatter points; all four artists keep
                    # the default color cycle and global legend aligned.
                    ax.plot(x, y, linestyle="-", marker=marker, markersize=6,
                            linewidth=1.2, alpha=0.85, label=label)
                ax.set_title(f"K = {k}")
                ax.set_xlabel(r"$\alpha$")
                ax.set_ylabel("Explained variance (seed mean)")
                ax.set_xlim(*x_limits)
                ax.set_ylim(-0.02, 1.02)
                ax.tick_params(labelbottom=True, labelleft=True)
                ax.grid(True, linestyle="--", alpha=0.3)
                ax.set_axisbelow(True)
                if not any_defined:
                    ax.text(0.5, 0.5, "No defined values", ha="center",
                            va="center", transform=ax.transAxes)
            for ax in axes_flat[n_panels:]:
                fig.delaxes(ax)
            handles, labels = axes_flat[0].get_legend_handles_labels()
            title = "Pre-training" if stage == "pretrain" else "Post-training"
            fig.suptitle(f"{title}: alpha sensitivity (seed means)",
                         y=1.0 - 0.08 / height, fontsize=13)
            fig.legend(handles, labels, loc="upper center", ncol=4,
                       bbox_to_anchor=(0.5, 1.0 - 0.37 / height), frameon=False)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0 - 0.80 / height))
            # Replace only after the PDF is complete; a failed save keeps the old file.
            with NamedTemporaryFile(dir=output_path.parent,
                                   prefix=f".{output_path.stem}-",
                                   suffix=".pdf", delete=False) as stream:
                temporary = Path(stream.name)
            fig.savefig(temporary, format="pdf", bbox_inches="tight",
                        metadata={"Title": f"{title} alpha sensitivity",
                                  "Subject": "Seed-mean R_Q, R_S, R_balanced and R_joint by K"})
            temporary.replace(output_path)
        return output_path
    finally:
        if fig is not None:
            plt.close(fig)
        if temporary is not None and temporary.exists():
            temporary.unlink()


GRID_REPORT_FILES = (
    "cluster_results.csv", "clustering_matrix.csv", "cluster_summary.csv",
    "PCA_summary.csv", "cluster_metrics.json", "cluster_profiles.png", "cluster_pca.png",
)


def grid_result_folder_name(alpha, n_clusters):
    """Round-trip float spelling: never merge nearby alpha values by rounding."""
    alpha = validate_alpha(alpha)
    k = validate_n_clusters(n_clusters)
    if alpha == 0:
        alpha = 0.0  # Avoid different directories for -0.0 and 0.0.
    return f"result_{alpha}_{k}"


def json_finite_or_none(value):
    """Represent undefined R/ARI as JSON null; do not emit nonstandard NaN."""
    if isinstance(value, dict):
        return {key: json_finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_finite_or_none(item) for item in value]
    if isinstance(value, (float, np.floating)):
        if np.isinf(value):
            raise ValueError("Infinite diagnostic value cannot be saved")
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def save_grid_cell_reports(results_root, stage, alpha, k, reference_run,
                           seed_runs, pairs, ari_mean, ari_min, resource_scores,
                           total_q, total_s):
    """Publish all seven standard reports for the FIRST supplied seed.

    Keep ordinary summary/PCA/label semantics: never average cluster IDs,
    coordinates, or cluster centers across seeds. Record all seed-wise scores,
    their means and pairwise ARIs in cluster_metrics.json['grid_search'].
    KMeans is NOT rerun. Successful cells survive a later grid interruption.
    Render into a temporary directory first, preserving an old complete report
    if rendering fails. Preserve unrelated files on repeated execution.
    """
    if stage not in ("pretrain", "posttrain"):
        raise ValueError("Invalid report stage")
    alpha = validate_alpha(alpha)
    k = validate_n_clusters(k)
    seeds = [validate_random_seed(item['seed']) for item in seed_runs]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Report seeds must be nonempty and unique")
    if (reference_run.prepared.data_type != stage
            or reference_run.config['alpha'] != alpha
            or reference_run.config['n_clusters'] != k
            or reference_run.config['random_state'] != seeds[0]):
        raise ValueError("Reference run must match this cell and the first supplied seed")
    if [item['seed'] for item in resource_scores] != seeds:
        raise ValueError("Resource scores must match report seeds in order")
    if len(pairs) != len(seeds) * (len(seeds) - 1) // 2:
        raise ValueError("Report must include every distinct seed pair")

    metrics = deepcopy(reference_run.metrics)
    metrics.update(
        stability_seeds=seeds,
        pairwise_ARI_mean=ari_mean,
        pairwise_ARI_min=ari_min,
        iterations_by_seed=[int(item['iterations']) for item in seed_runs],
    )
    metrics['grid_search'] = {
        'alpha': alpha, 'n_clusters': k, 'reference_seed': seeds[0],
        'reference_policy': 'first supplied seed, not best seed; no averaging of labels or cluster centers',
        'report_scope': 'standard language table, cluster summary and PCA describe the reference seed; grid means are separate',
        'random_seeds': seeds,
        'seed_mean_silhouette': float(np.mean([item['silhouette'] for item in seed_runs])),
        'seed_mean_R': {name: float(np.mean([item[name] for item in resource_scores]))
                        for name in ('R_Q', 'R_S', 'R_balanced', 'R_joint')},
        'R_block_totals': {'T_Q': total_q, 'T_S': total_s},
        'per_seed_scores': seed_runs,
        'per_seed_R_scores': resource_scores,
        'per_seed_pair_ARI': pairs,
        'pairwise_ARI_pairs': len(pairs),
        'fit_calls': len(seeds),
        'n_init': reference_run.config['n_init'],
    }
    metrics = json_finite_or_none(metrics)
    # Validate JSON before allocating directories. Copying leaves the input run intact.
    json.dumps(metrics, ensure_ascii=False, allow_nan=False)
    report_run = replace(reference_run, metrics=metrics)
    results_root = Path(results_root)
    pair_dir = results_root / grid_result_folder_name(alpha, k)
    final = pair_dir / f"meta_data_{stage}_clustered"
    if any(path.is_symlink() for path in (results_root, pair_dir, final)):
        raise ValueError("Grid result directories must not be symlinks")
    if final.exists() and not final.is_dir():
        raise NotADirectoryError(final)
    pair_dir.mkdir(parents=True, exist_ok=True)
    # Temporary sibling directories ensure publication stays on the same filesystem.
    temp_root = Path(mkdtemp(prefix=f".{stage}-reports-", dir=pair_dir))
    staged = temp_root / 'new'
    previous = temp_root / 'previous'
    preserve_backup = False
    try:
        if final.is_dir():
            shutil.copytree(final, staged, symlinks=True)
        else:
            staged.mkdir()
        for filename in GRID_REPORT_FILES + ('cluster_centers.csv', 'cluster_center.csv'):
            if (staged / filename).is_symlink():
                raise ValueError(f"Refusing a symlink for generated report: {filename}")
        save_clustering_run(report_run, staged, print_results=False)
        for filename in GRID_REPORT_FILES:
            path = staged / filename
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Incomplete grid report: {filename}")
        had_previous = final.exists()
        if had_previous:
            final.rename(previous)
        try:
            staged.rename(final)
        except BaseException:
            if had_previous:
                try:
                    previous.rename(final)
                except BaseException as error:
                    # Never delete the old complete report if rollback itself fails.
                    preserve_backup = True
                    raise OSError(
                        f"Report publication and rollback failed; previous report retained at {previous}"
                    ) from error
            raise
    finally:
        if not preserve_backup:
            shutil.rmtree(temp_root)
    return final


def run_grid_search(alphas, n_clusters_list, random_seeds, *, root=None,
                    data_type="all", print_results=True, save_alpha_curves=True,
                    save_results=True):
    """Run the full Cartesian product, returning {stage: mean-score DataFrame}.

    Rows follow alphas, columns follow n_clusters_list. Each cell averages the
    overall (language-equal) full-space Silhouette over ALL supplied seeds.
    Every seed runs exactly one KMeans.fit with YAML n_init, not the normal
    report workflow's nested ten-seed stability check. Cluster -1 is excluded.

    Each cell also compares every unordered seed pair on the same language
    rows, reporting mean/minimum ARI. It reuses labels, without additional fits.
    One seed gives an undefined ARI (NaN in numeric matrices; N/A when printed).
    Return values remain the Silhouette DataFrames for backward compatibility.
    Their attrs additionally contain pairwise_ARI_mean and pairwise_ARI_min
    (numeric DataFrames), per_seed_pair_ARI, and pairwise_ARI_pairs_per_cell.
    Numerical matrices remain in memory; per-cell reports and alpha PDFs are optional outputs.
    Changing alpha changes the distance
    metric: a maximum across alpha rows is not by itself proof of the most
    substantively appropriate weighting. Seed ARI tests initialization
    stability, not agreement with ground truth or cross-stage agreement.

    The third grid reports seed-mean R_Q / R_S / R_balanced / R_joint.
    R_Q and R_S use fixed, unweighted blocks; R_balanced=(R_Q+R_S)/2.
    R_joint weights R_Q and R_S by alpha*T_Q and (1-alpha)*T_S, respectively.
    It describes the CURRENT alpha-weighted space, not a common cross-alpha
    evaluation metric. All four reuse the SAME fitted labels; no extra fits.
    Source centroids are means of sqrt(p/2), not sqrt of mean p. Cluster -1
    is excluded. Each score averages ALL supplied seeds, without nanmean.
    Only the third grid is rounded to three decimals for display.
    attrs includes four numeric R_*_mean matrices, R_block_totals, and
    per_seed_R_scores with all four values. The ordinary cluster_analysis.py
    entry point is unchanged and never runs this code.

    By default save {stage}_alpha_curves.pdf in cluster_grid_search_results/
    under the project root, and all seven standard reports per cell to
    cluster_grid_search_results/result_{alpha}_{k}/
    meta_data_{stage}_clustered/. Report labels/plots use the first supplied
    seed, while grid means and all supplied seeds' stability diagnostics are
    also preserved in cluster_metrics.json. Save each cell after its seeds
    finish, without additional fits. print_results=False only suppresses stdout.
    Set BOTH save_results=False and save_alpha_curves=False for no file writes.
    Figure panels follow caller K order; x uses actual alpha values sorted for
    plotting only. Matrices and their precision are unchanged.
    """
    if not isinstance(save_results, bool):
        raise ValueError("save_results must be a boolean")
    if not isinstance(save_alpha_curves, bool):
        raise ValueError("save_alpha_curves must be a boolean")
    alphas = validated_list(alphas, "alphas", validate_alpha)
    n_clusters_list = validated_list(n_clusters_list, "n_clusters_list", validate_n_clusters)
    random_seeds = validated_list(random_seeds, "random_seeds", validate_random_seed)
    if data_type not in ("all", "pretrain", "posttrain"):
        raise ValueError("data_type must be 'all', 'pretrain', or 'posttrain'")
    project_root = Path(__file__).resolve().parent if root is None else Path(root).expanduser().resolve()
    stages = ("pretrain", "posttrain") if data_type == "all" else (data_type,)

    # Read each stage only once. Alpha changes require reconstructing weighted
    # features, never reusing a matrix saved under a different alpha.
    inputs = {stage: prepare_clustering(data_type=stage, root=project_root) for stage in stages}
    reference_blocks = {}
    for stage, prepared in inputs.items():
        selected = prepared.frame.loc[prepared.usable]
        reference_blocks[stage] = build_variance_reference_blocks(selected)
        n = len(selected)
        for k in n_clusters_list:
            if k >= n:
                raise ValueError(f"{stage}: n_clusters={k} must be less than {n} usable languages")
        for alpha in alphas:
            features = build_features(selected, alpha)
            distinct = len(np.unique(features.to_numpy(), axis=0))
            if max(n_clusters_list) > distinct:
                raise ValueError(
                    f"{stage}, alpha={alpha}: only {distinct} distinct feature vectors; "
                    f"cannot fit n_clusters={max(n_clusters_list)}"
                )

    results_root = project_root / "cluster_grid_search_results"
    if save_results or save_alpha_curves:
        if results_root.is_symlink():
            raise ValueError("Grid results root must not be a symlink")
        if results_root.exists() and not results_root.is_dir():
            raise NotADirectoryError(results_root)
    if save_results:
        for alpha in alphas:
            for k in n_clusters_list:
                pair_dir = results_root / grid_result_folder_name(alpha, k)
                for path in [pair_dir] + [pair_dir / f"meta_data_{stage}_clustered" for stage in stages]:
                    if path.is_symlink():
                        raise ValueError(f"Grid report directory must not be a symlink: {path}")
                    if path.exists() and not path.is_dir():
                        raise NotADirectoryError(path)

    matrices = {}
    for stage, prepared in inputs.items():
        matrix = pd.DataFrame(index=alphas, columns=n_clusters_list, dtype=float)
        matrix.index.name = "alpha"
        matrix.columns.name = "n_clusters"
        ari_mean = matrix.copy()
        ari_min = matrix.copy()
        r_q_mean = matrix.copy()
        r_s_mean = matrix.copy()
        r_balanced_mean = matrix.copy()
        r_joint_mean = matrix.copy()
        quantity_block, source_block = reference_blocks[stage]
        total_q = block_total_sum_of_squares(quantity_block)
        total_s = block_total_sum_of_squares(source_block)
        expected_index = prepared.frame.loc[prepared.usable].index
        details = []
        pairwise_details = []
        resource_details = []
        saved_reports = []
        for alpha in alphas:
            for k in n_clusters_list:
                scores = []
                r_q_scores, r_s_scores = [], []
                balanced_scores, joint_scores = [], []
                labels_by_seed = {}
                reference_run = None
                cell_seed_runs = []
                cell_resource_scores = []
                for seed in random_seeds:
                    try:
                        run = run_clustering(
                            alpha, k, data_type=stage, prepared=prepared,
                            random_seed=seed, evaluate_stability=False,
                        )
                        score = float(run.metrics["silhouette_score"])
                        if not np.isfinite(score):
                            raise ValueError("non-finite Silhouette")
                        if not run.selected.index.equals(expected_index):
                            raise ValueError("Language rows/order changed between seed runs")
                        labels = np.asarray(run.model.labels_)
                        if labels.shape != (len(expected_index),):
                            raise ValueError("Cluster labels do not align with language rows")
                        if (not np.issubdtype(labels.dtype, np.integer)
                                or (labels < 0).any() or len(np.unique(labels)) != k):
                            raise ValueError("Expected exactly k fitted clusters and no Cluster -1")
                        r_q = block_explained_variance(quantity_block, labels)
                        r_s = block_explained_variance(source_block, labels)
                        r_balanced, r_joint = combined_explained_variance(
                            r_q, r_s, total_q, total_s, alpha,
                        )
                    except (ValueError, TypeError, FloatingPointError) as error:
                        # Never silently drop failed seeds and average a partial cell.
                        raise ValueError(
                            f"{stage}, alpha={alpha}, n_clusters={k}, seed={seed}: {error}"
                        ) from error
                    if save_results:
                        if reference_run is None:
                            reference_run = run
                        cell_seed_runs.append({
                            "seed": seed, "silhouette": score,
                            "inertia": float(run.metrics["inertia"]),
                            "iterations": int(run.model.n_iter_),
                            "cluster_sizes": dict(run.metrics["cluster_sizes"]),
                        })
                    scores.append(score)
                    r_q_scores.append(r_q)
                    r_s_scores.append(r_s)
                    balanced_scores.append(r_balanced)
                    joint_scores.append(r_joint)
                    resource_details.append({"alpha": alpha, "n_clusters": k,
                                             "seed": seed, "R_Q": r_q, "R_S": r_s,
                                             "R_balanced": r_balanced, "R_joint": r_joint})
                    if save_results:
                        cell_resource_scores.append(dict(resource_details[-1]))
                    labels_by_seed[seed] = labels.copy()
                    details.append({"alpha": alpha, "n_clusters": k,
                                    "seed": seed, "silhouette": score})
                matrix.loc[alpha, k] = float(np.mean(scores))
                # Average every seed, never choose the best seed or use nanmean.
                # Constant reference blocks yield NaN for every seed in a cell.
                r_q_mean.loc[alpha, k] = float(np.mean(r_q_scores))
                r_s_mean.loc[alpha, k] = float(np.mean(r_s_scores))
                r_balanced_mean.loc[alpha, k] = float(np.mean(balanced_scores))
                r_joint_mean.loc[alpha, k] = float(np.mean(joint_scores))
                try:
                    mean, minimum, pairs = pairwise_seed_ari(labels_by_seed)
                except (ValueError, TypeError, FloatingPointError) as error:
                    raise ValueError(
                        f"{stage}, alpha={alpha}, n_clusters={k}, pairwise ARI: {error}"
                    ) from error
                ari_mean.loc[alpha, k] = mean
                ari_min.loc[alpha, k] = minimum
                pairwise_details.extend(
                    {"alpha": alpha, "n_clusters": k, **pair} for pair in pairs
                )
                if save_results:
                    try:
                        path = save_grid_cell_reports(
                            results_root, stage, alpha, k, reference_run,
                            cell_seed_runs, pairs, mean, minimum, cell_resource_scores,
                            total_q, total_s,
                        )
                    except (OSError, ValueError, TypeError, KeyError) as error:
                        raise ValueError(
                            f"{stage}, alpha={alpha}, n_clusters={k}: failed to save reports: {error}"
                        ) from error
                    saved_reports.append({"alpha": alpha, "n_clusters": k,
                                          "reference_seed": random_seeds[0], "path": str(path)})
        matrix.attrs.update(
            data_type=stage,
            usable_languages=int(prepared.usable.sum()),
            random_seeds=list(random_seeds),
            n_init=prepared.config["n_init"],
            fit_calls=len(alphas) * len(n_clusters_list) * len(random_seeds),
            per_seed_scores=details,
            saved_reports=saved_reports,
            reference_seed=random_seeds[0],
            pairwise_ARI_mean=ari_mean,
            pairwise_ARI_min=ari_min,
            per_seed_pair_ARI=pairwise_details,
            R_Q_mean=r_q_mean,
            R_S_mean=r_s_mean,
            R_balanced_mean=r_balanced_mean,
            R_joint_mean=r_joint_mean,
            R_block_totals={"T_Q": total_q, "T_S": total_s},
            per_seed_R_scores=resource_details,
            R_scope={
                "definition": "R=1-W/T; within-cluster and total centered sums of squares",
                "quantity_block": "standardized log10 quantity, without alpha weighting",
                "source_block": "sqrt(normalized source proportions / 2), without alpha weighting",
                "population": "same fitted languages across all alpha/k/seeds; Cluster -1 excluded; upstream source defaults retained",
                "centroids": "recompute language-equal means in each fixed evaluation block for the existing labels",
                "balanced": "(R_Q+R_S)/2; equal weight per feature block; same evaluation convention across alpha",
                "joint": "(alpha*T_Q*R_Q+(1-alpha)*T_S*R_S)/(alpha*T_Q+(1-alpha)*T_S); current alpha-weighted geometry",
                "aggregation": "arithmetic mean over all supplied seeds, independently for all four R scores; no extra fits",
                "zero_variance": "block R undefined if T=0; balanced undefined if either block is constant; joint ignores zero-weight/zero-variance terms and is undefined only if weighted T=0",
                "display": "R_Q / R_S / R_balanced / R_joint; three decimal places; numeric values not rounded",
            },
            pairwise_ARI_pairs_per_cell=len(random_seeds) * (len(random_seeds) - 1) // 2,
            pairwise_ARI_scope="all unordered distinct seed pairs within each (stage, alpha, k); same fitted languages; no Cluster -1; no extra fits",
            metric="mean overall Silhouette in the full alpha-weighted feature space",
        )
        matrices[stage] = matrix

    # Per stage: Silhouette, seed-pair ARI, then four R scores; no per-fit output.
    if print_results:
        for stage, matrix in matrices.items():
            print(f"\n=== {stage.upper()}: seed-mean overall Silhouette ===")
            print(f"usable_languages={matrix.attrs['usable_languages']}; "
                  f"seeds={random_seeds}; n_init={matrix.attrs['n_init']}; "
                  f"KMeans fits={matrix.attrs['fit_calls']}")
            print(matrix.to_string(float_format=lambda value: f"{value:.6f}"))
            print(f"\n=== {stage.upper()}: pairwise seed ARI (mean / min) ===")
            print(f"seed_pairs_per_cell={matrix.attrs['pairwise_ARI_pairs_per_cell']}; "
                  "same alpha and n_clusters; fitted languages only")
            if matrix.attrs["pairwise_ARI_pairs_per_cell"] == 0:
                print("N/A: at least two distinct seeds are required for pairwise ARI.")
            print(format_pairwise_ari(
                matrix.attrs["pairwise_ARI_mean"], matrix.attrs["pairwise_ARI_min"],
            ).to_string())
            print(f"\n=== {stage.upper()}: seed-mean variance explained "
                  "(R_Q / R_S / R_balanced / R_joint) ===")
            print("Q=standardized log10(quantity); S=sqrt(p/2); "
                  "R_balanced=equal blocks; R_joint=current alpha-weighted space; N/A=undefined")
            print(format_resource_explained_variance(
                matrix.attrs["R_Q_mean"], matrix.attrs["R_S_mean"],
                matrix.attrs["R_balanced_mean"], matrix.attrs["R_joint_mean"],
            ).to_string())
    if print_results and save_results:
        count = len(stages) * len(alphas) * len(n_clusters_list)
        print(f"Saved {count} complete stage reports under {results_root}; "
              f"reference_seed={random_seeds[0]}; multi-seed means/stability in cluster_metrics.json")
    # Render only final seed-mean alpha diagnostics; per-cell reports are already saved.
    if save_alpha_curves:
        # Curves can be saved even when per-parameter reports are disabled.
        results_root.mkdir(parents=True, exist_ok=True)
        for stage, matrix in matrices.items():
            path = results_root / f"{stage}_alpha_curves.pdf"
            save_alpha_curves_pdf(stage, matrix, path)
            if print_results:
                print(f"Saved alpha curves PDF: {path}")
    return matrices


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphas", nargs="+", type=float, required=True,
                        help="Alpha values in [0,1], separated by spaces")
    parser.add_argument("--n-clusters", "--n_clusters", dest="n_clusters_list",
                        nargs="+", type=int, required=True,
                        help="Cluster counts >= 2, separated by spaces")
    parser.add_argument("--seeds", nargs="+", type=int, required=True,
                        help="KMeans random seeds; each contributes once to every cell")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="Project directory containing YAML, registry and reduced JSONs")
    parser.add_argument("--data-type", "--data_type", dest="data_type",
                        choices=("all", "pretrain", "posttrain"), default="all",
                        help="Default: print Silhouette, ARI, and four-score R grids for both stages")
    parser.add_argument("--no-save-alpha-curves", action="store_true",
                        help="Do not create or update alpha PDFs in cluster_grid_search_results")
    parser.add_argument("--no-save-results", action="store_true",
                        help="Do not create per-parameter clustering reports; combine with --no-save-alpha-curves for no writes")
    args = parser.parse_args(argv)
    try:
        run_grid_search(args.alphas, args.n_clusters_list, args.seeds,
                        root=args.root, data_type=args.data_type,
                        save_alpha_curves=not args.no_save_alpha_curves,
                        save_results=not args.no_save_results)
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
