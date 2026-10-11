import argparse
import json
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from numbers import Integral
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from filter_datasets import living_language_codes
from group_datasets import (
    EXCLUDED_TASK_CATEGORY, canonical_task_category, default_source_category,
)
from map_datasets import source_categories
from math_utils import gini


ROOT = Path(__file__).resolve().parent
SOURCE_NAMES = {
    "Expert-Created": "专家构建",
    "Human-Created": "人工构建",
    "Collect-Existed": "已有语料",
    "Human-Audited": "人工审核",
    "Machine-Generated": "机器合成",
}

# Fixed categorical colors for all Source / Task proportion figures.
# The five Source colors come from the ICLR 2025 Data Provenance Initiative
# Figure 2 (the original PDF uses wine-red, ochre, and blue-grey families).
# Unlike a per-panel Matplotlib colormap, these assignments are invariant to
# cluster ordering, the presence of a category, and the training stage.
ICLR_FIG2_ALTAIR_SOURCE_COLORS = {
    "Expert-Created": "#9D354F",
    "Human-Created": "#E04C71",
    "Collect-Existed": "#5B7F91",
    "Human-Audited": "#82B5CF",
    "Machine-Generated": "#9D9066",
}

# Task labels need more than Figure 2's nine swatches. Supplement its muted
# style with Altair/Vega categorical colors (category10 + category20b).
# Keep an explicit category->hex mapping: the same task must not change color
# when it is absent from one stage or from a particular grid-search cell.
# Consistent cluster-identity palette for profile boxplots and PCA.
# The first three colors are the intermediate wine / blue-grey / ochre
# swatches of the SAME ICLR Figure 2 palette as Source proportions, but
# are not assigned to particular provenance or task categories.
# Extra swatches support the K>3 sensitivity/grid-search reports.
ICLR_FIG2_PROFILE_CLUSTER_COLORS = (
    "#C24262", "#719DB3", "#C2B27F",
    "#9D354F", "#5B7F91", "#9D9066",
    "#E04C71", "#82B5CF", "#E0CD92",
)


def build_cluster_profile_colors(n_clusters):
    """Stable cluster colors shared by profiles and PCA across stages."""
    if not isinstance(n_clusters, (int, np.integer)) or n_clusters < 1:
        raise ValueError("n_clusters must be a positive integer")
    return [
        ICLR_FIG2_PROFILE_CLUSTER_COLORS[k % len(ICLR_FIG2_PROFILE_CLUSTER_COLORS)]
        for k in range(n_clusters)
    ]


ALTAIR_VEGA_TASK_COLORS = {
    "pretrain": "#4C78A8",
    "translation": "#F58518",
    "summarization": "#E45756",
    "QA": "#72B7B2",
    "NLU": "#54A24B",
    "coding": "#EECA3B",
    "NLG": "#B279A2",
    "dialogue": "#FF9DA6",
    "classification": "#9D755D",
    "math": "#393B79",
    "evaluation": "#637939",
    "rewriting": "#8C6D31",
    "information_extraction": "#843C39",
    "general_IF": "#7B4173",
    "reasoning": "#6B6ECF",
    "domain_specific": "#B5CF6B",
    "safety_and_ethics": "#E7BA52",
    "linguistic_analysis": "#D6616B",
    "retrieval": "#BAB0AC",
    "Other": "#D1D5DB",
}


def read_data(input_dir, attributes, language_codes):
    # Keep ISO language files without changing the upstream quantity allocation.
    records, totals, excluded = {}, {}, {}
    quantity_key = attributes[0].removesuffix(" log10")
    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".json":
            continue
        code = path.stem
        if code not in language_codes:
            excluded[code] = "not an ISO 639-3 code"
            continue
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        sources = data.get("Text Source Categories Statistics") or {}
        values = []
        for attribute in attributes:
            if "/" in attribute:
                category, field = attribute.split("/", 1)
                value = (sources.get(category) or {}).get(field)
            else:
                value = data.get(attribute)
            # Both absent values and explicit nulls become zero before normalization.
            values.append(0.0 if value is None else float(value))
        records[code] = values
        total = data.get(quantity_key)
        totals[code] = 0.0 if total is None else float(total)

    frame = pd.DataFrame.from_dict(records, orient="index", columns=attributes)
    frame.index.name = "Language"
    totals = pd.Series(totals, dtype=float).reindex(frame.index)
    usable = np.isfinite(totals) & totals.gt(0)
    for code in frame.index[~usable]:
        excluded[code] = "no usable positive quantity"
    return frame, totals, usable, excluded


def read_categorical_profile(input_dir, language_index, statistic_key, fallback="Other"):
    """Read a normalized language x category table from reduced statistics."""
    records = {}
    for code in language_index:
        path = input_dir / f"{code}.json"
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        stats = data.get(statistic_key) or {}
        row = {}
        for category, values in stats.items():
            if not isinstance(category, str) or not category.strip():
                continue
            category = category.strip()
            if statistic_key == "Task Macro Categories Statistics":
                category = canonical_task_category(category)
                if category == EXCLUDED_TASK_CATEGORY:
                    continue
            if not isinstance(values, dict):
                continue
            value = values.get("proportion")
            if value is None:
                continue
            value = float(value)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{code}: invalid {statistic_key} proportion for {category!r}")
            if value > 0:
                row[category] = row.get(category, 0.0) + value
        total = sum(row.values())
        if total > 0:
            row = {key: value / total for key, value in row.items()}
        else:
            row = {fallback: 1.0}
        records[code] = row

    frame = pd.DataFrame.from_dict(records, orient="index").fillna(0.0)
    frame.index.name = "Language"
    if frame.empty:
        frame = pd.DataFrame({fallback: np.ones(len(language_index))}, index=language_index)
        frame.index.name = "Language"
    return frame.reindex(language_index, fill_value=0.0)


def read_source_counts(input_dir, language_index):
    """Read per-language coverage counts without adding clustering features."""
    fields = ("num_collection_sources", "num_dataset_sources")
    records = {}
    for code in language_index:
        data = json.loads((input_dir / f"{code}.json").read_text(encoding="utf-8-sig"))
        counts = [data[field] for field in fields]
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0
               for value in counts):
            raise ValueError(f"{code}: source counts must be nonnegative integers")
        records[code] = counts
    return pd.DataFrame.from_dict(records, orient="index", columns=fields)


def normalize_sources(frame):
    # Form a conditional distribution over the configured source categories.
    normalized = frame.copy()
    proportions = frame.iloc[:, 1:].astype(float)
    if not np.isfinite(proportions.to_numpy()).all():
        raise ValueError("Source proportions must be finite numbers")
    if ((proportions < 0) | (proportions > 1)).to_numpy().any():
        raise ValueError("Source proportions must lie between 0 and 1")
    totals = proportions.sum(axis=1)
    positive = totals.gt(0)
    normalized.loc[positive, frame.columns[1:]] = proportions.loc[positive].div(
        totals.loc[positive], axis=0
    )
    # Last-resort fallback for legacy inputs or rows with no quantity assigned
    # to the modeled categories. Never fabricate a uniform five-source mix.
    if (~positive).any():
        fallback = default_source_category(frame.columns[0])
        column = f"{fallback}/proportion"
        if column not in proportions.columns:
            raise ValueError(f"Required fallback source column is missing: {column}")
        normalized.loc[~positive, frame.columns[1:]] = 0.0
        normalized.loc[~positive, column] = 1.0
    return normalized


def build_features(frame, alpha):
    # The first attribute is already log10(quantity); standardize it only once.
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Clustering attributes must be finite numbers")
    proportions = values[:, 1:]
    if ((proportions < 0) | (proportions > 1)).any():
        raise ValueError("Source proportions must lie between 0 and 1")
    if not np.allclose(proportions.sum(axis=1), 1):
        raise ValueError("Source proportions must be normalized to sum to one")
    standardized = StandardScaler().fit_transform(values[:, :1])
    # Squared source-block distances equal (1 - alpha) times Hellinger distance squared.
    features = np.column_stack([
        np.sqrt(alpha) * standardized,
        np.sqrt((1 - alpha) / 2) * np.sqrt(proportions),
    ])
    return pd.DataFrame(features, index=frame.index, columns=frame.columns)


def validate_alpha(value):
    """A finite quantity/source weight; endpoints 0 and 1 are supported."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("alpha must be a finite number between 0 and 1")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("alpha must be a finite number between 0 and 1") from None
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("alpha must be a finite number between 0 and 1")
    return value


def validate_n_clusters(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 2:
        raise ValueError("n_clusters must be an integer >= 2")
    return int(value)


def validate_random_seed(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            or not 0 <= value <= 2**32 - 1):
        raise ValueError("Each random seed must be an integer in [0, 2**32 - 1]")
    return int(value)


def fit_clusters(features, config, random_seeds=None):
    """Fit explicit seeds, or retain the original ten-seed reporting default.

    Each seed runs one KMeans.fit with the configured n_init restarts. With
    one seed, pairwise ARI is undefined (None), not a fabricated perfect score.
    """
    parameters = {key: config[key] for key in (
        "n_clusters", "init", "n_init", "max_iter", "tol"
    )}
    parameters["tol"] = float(parameters["tol"])
    k = parameters["n_clusters"] = validate_n_clusters(parameters["n_clusters"])
    if not 2 <= k < len(features):
        raise ValueError("Require 2 <= n_clusters < number of usable languages")
    if len(np.unique(features.to_numpy(), axis=0)) < k:
        raise ValueError("Fewer distinct feature vectors than requested clusters")
    if random_seeds is None:
        first = validate_random_seed(config["random_state"])
        random_seeds = range(first, first + 10)
    seeds = [validate_random_seed(seed) for seed in random_seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("random_seeds must be nonempty and contain no duplicates")
    # Avoid the Windows MKL/OpenMP memory leak when chunks are fewer than threads.
    with threadpool_limits(limits=1, user_api="openmp"):
        models = [KMeans(**parameters, random_state=seed).fit(features) for seed in seeds]
    model = models[0]
    labels = model.labels_
    if len(np.unique(labels)) != k:
        raise ValueError("K-means returned fewer occupied clusters than requested")
    aris = [adjusted_rand_score(a.labels_, b.labels_) for a, b in combinations(models, 2)]
    metrics = {
        "inertia": float(model.inertia_),
        "silhouette_score": float(silhouette_score(features, labels)),
        "stability_seeds": seeds,
        "pairwise_ARI_mean": float(np.mean(aris)) if aris else None,
        "pairwise_ARI_min": float(np.min(aris)) if aris else None,
        "iterations_by_seed": [int(item.n_iter_) for item in models],
        "cluster_sizes": {
            str(cluster): int(np.sum(labels == cluster)) for cluster in range(k)
        },
    }
    return model, metrics


def relabel_clusters(model, totals):
    # Order clusters by arithmetic mean raw quantity, not mean log quantity.
    quantities = totals.to_numpy()
    means = [quantities[model.labels_ == cluster].mean()
             for cluster in range(model.n_clusters)]
    order = np.argsort(means, kind="stable")
    old_to_new = np.empty(model.n_clusters, dtype=model.labels_.dtype)
    old_to_new[order] = np.arange(model.n_clusters)
    model.labels_ = old_to_new[model.labels_]
    # Keep internal feature-space centers aligned for prediction and representative selection.
    model.cluster_centers_ = model.cluster_centers_[order].copy()
    return {str(old): int(new) for old, new in enumerate(old_to_new)}


def cluster_summaries(frame, totals, labels, unclustered_count):
    # Recompute interpretable centers directly in raw Num / probability coordinates.
    quantity_key = frame.columns[0].removesuffix(" log10")
    values = frame.iloc[:, 1:].copy()
    values.insert(0, quantity_key, totals.reindex(frame.index))
    values["Cluster"] = labels
    groups = values.groupby("Cluster", sort=True)
    centers = groups.mean()
    counts = groups.size()
    medians = groups[quantity_key].median()
    low, high = centers[quantity_key].quantile([1 / 3, 2 / 3])

    rows = []
    for cluster, center in centers.iterrows():
        mean = center[quantity_key]
        size = "相对低规模" if mean < low else "相对高规模" if mean > high else "中等规模"
        sources = center.iloc[1:].copy()
        sources.index = [column.split("/", 1)[0] for column in sources.index]
        ranked = sources.sort_values(ascending=False, kind="stable")
        # A strict majority avoids assigning a sole dominant source to a 50/50 tie.
        if ranked.iloc[0] > 0.5 + 1e-12:
            dominant = SOURCE_NAMES.get(ranked.index[0], ranked.index[0]) + "主导"
        elif sources.get("Human-Created", 0) + sources.get("Expert-Created", 0) > 0.5 + 1e-12:
            dominant = "人工与专家构建主导"
        else:
            dominant = "混合来源"
        composition = "; ".join(
            f"{SOURCE_NAMES.get(source, source)} {share * 100:.3g}%"
            for source, share in ranked.items() if share > 0
        )
        rows.append({
            "Cluster": int(cluster), "解释性名称": f"{size}、{dominant}",
            "语言数": int(counts.loc[cluster]), "Num均值": mean,
            "Num中位数": medians.loc[cluster], "典型来源构成": composition,
        })
    if unclustered_count:
        rows.append({
            "Cluster": -1, "解释性名称": "无覆盖或数量不可用", "语言数": unclustered_count,
            "Num均值": None, "Num中位数": None, "典型来源构成": "不参与聚类",
        })
    return pd.DataFrame(rows).set_index("Cluster").sort_index(), centers



def extend_cluster_summary(summary, centers, frame, totals, labels, silhouettes, data_type):
    """Combine center columns and diagnostics; all metrics exclude Cluster -1.

    Silhouettes are computed once against ALL fitted clusters in the original
    weighted feature space, not by fitting or scoring each cluster separately.
    The overall row weights languages equally, not cluster centers equally.
    """
    pure_category = {
        "pretrain": "Collect-Existed", "posttrain": "Machine-Generated",
    }[data_type]
    quantity_key = frame.columns[0].removesuffix(" log10")
    labels = np.asarray(labels)
    scores = np.asarray(silhouettes, dtype=float)
    if labels.shape != (len(frame),) or scores.shape != (len(frame),):
        raise ValueError("Cluster labels and silhouette values must align with language rows")
    if not np.isfinite(scores).all():
        raise ValueError("Full-space silhouette values must be finite")
    if not frame.index.is_unique or not totals.index.is_unique:
        raise ValueError("Language indices must be unique")
    quantities = totals.reindex(frame.index)
    if not np.isfinite(quantities.to_numpy()).all() or not quantities.gt(0).all():
        raise ValueError("Summary requires finite positive quantities for fitted languages")
    sources = frame.iloc[:, 1:]
    pure_column = f"{pure_category}/proportion"
    if pure_column not in sources.columns:
        raise ValueError(f"Missing source column: {pure_column}")
    # Strict one-hot membership: do not round near-100% proportions to 100%.
    pure = (sources[pure_column].eq(1.0)
            & sources.drop(columns=[pure_column]).eq(0.0).all(axis=1)).to_numpy()
    count_column = f"100% {pure_category} 语言数"
    silhouette_column = "平均 Silhouette"
    negative_column = "Silhouette为负的语言数"

    # Preserve all old summary fields and all old center fields, including the
    # original-unit quantity column (which equals Num均值 for each fitted row).
    combined = summary.join(centers, how="left", validate="one_to_one")
    combined[count_column] = pd.Series(pd.NA, index=combined.index, dtype="Int64")
    combined[silhouette_column] = np.nan
    combined[negative_column] = pd.Series(pd.NA, index=combined.index, dtype="Int64")
    for cluster in centers.index:
        mask = labels == cluster
        combined.loc[cluster, count_column] = int(pure[mask].sum())
        combined.loc[cluster, silhouette_column] = float(scores[mask].mean())
        combined.loc[cluster, negative_column] = int((scores[mask] < 0).sum())

    source_means = sources.mean()
    ranked = source_means.sort_values(ascending=False, kind="stable")
    composition = "; ".join(
        f"{SOURCE_NAMES.get(column.split('/', 1)[0], column.split('/', 1)[0])} {share * 100:.3g}%"
        for column, share in ranked.items() if share > 0
    )
    overall = {
        "解释性名称": "整体（仅参与聚类语言）",
        "语言数": len(frame),
        "Num均值": float(quantities.mean()),
        "Num中位数": float(quantities.median()),
        "典型来源构成": composition,
        quantity_key: float(quantities.mean()),
        **source_means.to_dict(),
        count_column: int(pure.sum()),
        silhouette_column: float(scores.mean()),
        negative_column: int((scores < 0).sum()),
    }
    # Append after the existing numeric rows; never treat this as a cluster.
    combined.loc["整体"] = overall
    for column in ("语言数", count_column, negative_column):
        combined[column] = combined[column].astype("Int64")
    return combined


def extend_cluster_resource_summary(summary, source_counts, totals, assignments):
    """Append coverage means/SDs/medians and quantity/dataset-count Ginis."""
    result = summary.copy()
    fields = (
        ("num_collection_sources", "平均collection数量", "collection数量标准差",
         "collection数量中位数"),
        ("num_dataset_sources", "平均dataset数量", "dataset数量标准差",
         "dataset数量中位数"),
    )
    columns = [column for _, mean_column, std_column, _ in fields
               for column in (mean_column, std_column)]
    columns += [median_column for _, _, _, median_column in fields]
    columns += ["Gini", "gini-numDataset"]
    # Repeated updates replace these fields and keep them at the far right.
    result = result.drop(columns=columns, errors="ignore")
    result[columns] = np.nan
    # A missing language record means no sources in the current input collection.
    counts = source_counts.reindex(assignments.index, fill_value=0)
    for cluster in result.index:
        overall = cluster == "整体"
        mask = assignments.ge(0) if overall else assignments.eq(int(cluster))
        languages = assignments.index[mask]
        if languages.empty:
            continue
        for field, mean_column, std_column, median_column in fields:
            values = counts.loc[languages, field]
            result.loc[cluster, mean_column] = values.mean()
            # Sample SD is undefined for a singleton and remains blank in CSV.
            result.loc[cluster, std_column] = values.std(ddof=1)
            result.loc[cluster, median_column] = values.median()
        # Coverage counts remain defined for -1; all-zero vectors yield NaN.
        result.loc[cluster, "gini-numDataset"] = gini(
            counts.loc[languages, "num_dataset_sources"].to_numpy()
        )
        # Cluster -1 includes unknown quantities, not observed resource zeros.
        if overall or int(cluster) >= 0:
            result.loc[cluster, "Gini"] = gini(totals.reindex(languages).to_numpy())
    return result


def pearson_or_nan(left, right):
    """Return Pearson r; undefined correlations are blank in CSV, not zero."""
    left = np.asarray(left, dtype=float).reshape(-1)
    right = np.asarray(right, dtype=float).reshape(-1)
    if left.shape != right.shape:
        raise ValueError("Correlation vectors must have the same length")
    if len(left) < 2 or not (np.isfinite(left).all() and np.isfinite(right).all()):
        return np.nan
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return np.nan
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def summarize_pca(frame, coordinates, variance_ratio, labels, data_type):
    """Three numeric columns plus a descriptive row index.

    PC-specific silhouettes evaluate the original labels using only that PC;
    total evaluates the same labels in the joint PC1/PC2 projection. It is not
    a sum or average of the two one-dimensional silhouette scores.
    Correlations use log10(quantity) and pre-square-root source proportions.
    """
    focus = {"pretrain": "Expert-Created", "posttrain": "Machine-Generated"}[data_type]
    source_column = f"{focus}/proportion"
    if source_column not in frame.columns:
        raise ValueError(f"Missing PCA correlation source: {source_column}")
    coordinates = np.asarray(coordinates, dtype=float)
    variance_ratio = np.asarray(variance_ratio, dtype=float)
    labels = np.asarray(labels)
    if coordinates.shape != (len(frame), 2) or variance_ratio.shape != (2,):
        raise ValueError("PCA report requires two coordinates and two variance ratios")
    if labels.shape != (len(frame),):
        raise ValueError("PCA cluster labels must align with language rows")
    log_quantity = frame.iloc[:, 0].to_numpy(dtype=float)
    source = frame[source_column].to_numpy(dtype=float)
    n_labels = len(np.unique(labels))
    projected_silhouettes = [np.nan, np.nan, np.nan]
    if 2 <= n_labels < len(frame):
        projected_silhouettes = [
            float(silhouette_score(coordinates[:, [axis]], labels, metric="euclidean"))
            for axis in range(2)
        ] + [float(silhouette_score(coordinates, labels, metric="euclidean"))]
    report = pd.DataFrame(
        [
            [float(variance_ratio[0]), float(variance_ratio[1]), float(variance_ratio.sum())],
            [pearson_or_nan(coordinates[:, 0], log_quantity),
             pearson_or_nan(coordinates[:, 1], log_quantity), np.nan],
            [pearson_or_nan(coordinates[:, 0], source),
             pearson_or_nan(coordinates[:, 1], source), np.nan],
            projected_silhouettes,
        ],
        index=[
            "解释方差比例",
            "与 log10(数量) 的 Pearson 相关系数",
            f"与 {focus} 比例的 Pearson 相关系数",
            "平均 Silhouette（原簇标签，投影空间）",
        ],
        columns=["PC1", "PC2", "total"],
    )
    report.index.name = "Metric"
    return report


def _profile_cluster_means(profile, labels, clusters):
    """Language-equal mean categorical proportions, ordered by overall prevalence."""
    profile = profile.astype(float)
    if not profile.index.is_unique:
        raise ValueError("Profile language index must be unique")
    values = profile.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Profile proportions must be finite and nonnegative")
    sums = values.sum(axis=1)
    if not np.allclose(sums, 1.0):
        raise ValueError("Each categorical profile row must sum to one")
    means = np.array([
        profile.iloc[labels == cluster].mean().to_numpy()
        for cluster in clusters
    ])
    order = np.argsort(-profile.mean(axis=0).to_numpy(), kind="stable")
    return profile.columns.to_numpy()[order], means[:, order]


def build_task_color_map():
    """Fixed Altair/Vega categorical colors shared across stages and grid cells."""
    categories = list(json.loads(
        (ROOT / "task_macro_category_mapping.json").read_text(encoding="utf-8")
    )["macro_categories"])
    categories = [category for category in categories if category != EXCLUDED_TASK_CATEGORY]
    categories = list(dict.fromkeys([*categories, "Other"]))
    missing = set(categories) - ALTAIR_VEGA_TASK_COLORS.keys()
    if missing:
        raise ValueError(f"Missing Altair task colors for: {sorted(missing)}")
    return {category: ALTAIR_VEGA_TASK_COLORS[category] for category in categories}


def build_profile_legend_order():
    """Use canonical sources and the global top ten ranked text tasks."""
    categories = json.loads(
        (ROOT / "task_macro_category_mapping.json").read_text(encoding="utf-8")
    )["macro_categories"]
    tasks = sorted(
        (category for category in categories if category != EXCLUDED_TASK_CATEGORY),
        key=lambda category: categories[category]["freq_ranking"]["rank"],
    )[:10]
    return {"Source": list(source_categories), "Task": tasks}


def _text_task_profile(profile):
    """Canonicalize legacy labels, hide multimodal, and normalize task shares."""
    profile = profile.copy()
    profile.columns = [
        canonical_task_category(column) if isinstance(column, str) else column
        for column in profile.columns
    ]
    excluded = [column for column in profile.columns
                if column == EXCLUDED_TASK_CATEGORY]
    profile = profile.drop(columns=excluded).astype(float)
    values = profile.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Profile proportions must be finite and nonnegative")
    if not profile.columns.is_unique:
        profile = profile.T.groupby(level=0, sort=False).sum().T
    totals = profile.sum(axis=1)
    empty = totals.eq(0)
    if empty.any():
        if "Other" not in profile.columns:
            profile["Other"] = 0.0
        profile.loc[empty, "Other"] = 1.0
        totals = profile.sum(axis=1)
    return profile.div(totals, axis=0)


def _stacked_profile(ax, profile, labels, clusters, xlabel, cmap_name,
                     *, category_colors=None, power=None):
    categories, means = _profile_cluster_means(profile, labels, clusters)
    if power is not None:
        # Transform cluster means only for display; preserve zero components.
        means = means ** power
        means /= means.sum(axis=1, keepdims=True)
    offsets = np.zeros(len(list(clusters)))
    if category_colors is None:
        cmap = plt.get_cmap(cmap_name)
        colors = cmap(np.linspace(0.05, 0.95, max(len(categories), 2)))[:len(categories)]
    else:
        missing = set(categories) - category_colors.keys()
        if missing:
            raise ValueError(f"Missing profile colors for: {sorted(missing)}")
        colors = [category_colors[category] for category in categories]
    handles, legend_labels = [], []
    for index, category in enumerate(categories):
        bars = ax.barh(
            list(clusters), means[:, index], left=offsets,
            label=str(category), color=colors[index], edgecolor="white", linewidth=0.25,
        )
        offsets += means[:, index]
        handles.append(bars[0])
        legend_labels.append(str(category))
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_xlabel(xlabel)
    return handles, legend_labels


def create_cluster_profile_figure(frame, totals, labels, n_clusters, data_type,
                                  task_profile, format_profile=None, *, task_colors=None,
                                  legend_order=None, quantity_label=None,
                                  cluster_count_label="n", power=None):
    """Build the shared profile figure without fitting, saving, or closing it."""
    if power is not None and not 0 < power < 1:
        raise ValueError("power must satisfy 0 < power < 1")
    task_profile = _text_task_profile(task_profile)
    if task_colors is None:
        task_colors = build_task_color_map()
    if legend_order is None:
        legend_order = build_profile_legend_order()
    k = n_clusters
    clusters = range(k)
    cluster_colors = build_cluster_profile_colors(k)
    quantity_key = frame.columns[0].removesuffix(" log10")

    # Reserve a separate band for five-row, panel-specific legends.
    legend_height = 1.65
    height = max(4, 0.6 * k) + legend_height
    include_format = format_profile is not None
    width_ratios = [1.18, 0.95, 1.05] + ([0.82] if include_format else [])
    fig, axes = plt.subplots(
        1, len(width_ratios), figsize=(18 if include_format else 14, height),
        sharey=True, gridspec_kw={"width_ratios": width_ratios},
    )
    left, source_ax, task_ax = axes[:3]

    boxes = left.boxplot(
        [totals.to_numpy()[labels == cluster] for cluster in clusters],
        orientation="horizontal", positions=list(clusters), patch_artist=True,
    )
    for box, color in zip(boxes["boxes"], cluster_colors):
        box.set_facecolor(color)
        box.set_alpha(0.6)
    left.set_xscale("log")
    left.set_xlabel(quantity_label if quantity_label is not None
                    else f"{quantity_key} (log scale)")
    left.set_yticks(list(clusters), [
        f"Cluster {cluster} ({cluster_count_label}={np.sum(labels == cluster)})"
        for cluster in clusters
    ])
    left.grid(axis="x", alpha=0.25)

    source_profile = frame.iloc[:, 1:].copy()
    source_profile.columns = [
        column.split("/", 1)[0] for column in source_profile.columns
    ]
    task_profile = task_profile.reindex(frame.index)
    panels = [
        (source_ax, source_profile, "Mean normalized source proportion", "Set2", "Source"),
        (task_ax, task_profile, "Mean normalized task proportion", "tab20", "Task"),
    ]
    if include_format:
        panels.append((axes[3], format_profile.reindex(frame.index),
                       "Mean normalized format proportion", "Set3", "Format"))

    legend_sections = []
    for ax, profile, xlabel, cmap_name, prefix in panels:
        panel_power = power if prefix in ("Source", "Task") else None
        if panel_power is not None:
            xlabel = f"Power-transformed {prefix.lower()} weight (power={power:g})"
        handles, labels_text = _stacked_profile(
            ax, profile, labels, clusters, xlabel, cmap_name,
            category_colors=(
                ICLR_FIG2_ALTAIR_SOURCE_COLORS if prefix == "Source"
                else task_colors if prefix == "Task" else None
            ),
            power=panel_power,
        )
        # Reorder/filter only legend entries; every category remains in the bars.
        if prefix in legend_order:
            by_label = dict(zip(labels_text, handles))
            ordered = [label for label in legend_order[prefix] if label in by_label]
            # Pre-training is outside the post-training top ten but must remain
            # visible in its own stage's legend. Do not mutate the cached order.
            if (prefix == "Task" and data_type == "pretrain"
                    and "pretrain" in by_label and "pretrain" not in ordered):
                ordered.append("pretrain")
            if prefix == "Source":
                ordered.extend(label for label in labels_text if label not in ordered)
            handles = [by_label[label] for label in ordered]
            labels_text = ordered
        legend_sections.append((ax, handles, labels_text, prefix))

    # Lay out the panels first, then anchor legends below their x-axis labels.
    fig.tight_layout(rect=(0.0, legend_height / height, 1.0, 1.0), w_pad=1.0)
    legend_y = min(ax.get_position().y0 for ax in axes) - 0.60 / height
    for ax, handles, labels_text, prefix in legend_sections:
        if not handles:
            continue
        ncol = 1 if (prefix == "Source"
                     or (prefix == "Task" and data_type == "pretrain")) else 2
        if prefix == "Format":
            ncol = (len(handles) + 4) // 5
            # Matplotlib fills columns top-to-bottom. Padding gives exactly five
            # slots per format column, with blanks only in the final column.
            padding = ncol * 5 - len(handles)
            handles = handles + [Patch(facecolor="none", edgecolor="none")] * padding
            labels_text = labels_text + [""] * padding
        position = ax.get_position()
        x = position.x0 if prefix == "Format" else (position.x0 + position.x1) / 2
        ax.legend(
            handles, labels_text,
            loc="upper left" if prefix == "Format" else "upper center",
            ncol=ncol, fontsize=10, frameon=False, columnspacing=1.0,
            handlelength=1.4, borderaxespad=0.0,
            bbox_to_anchor=(x, legend_y), bbox_transform=fig.transFigure,
        )
    return fig


def plot_clusters(frame, totals, features, model, output_dir, data_type,
                  task_profile, format_profile, *, filename_prefix="", task_colors=None,
                  legend_order=None):
    fig = create_cluster_profile_figure(
        frame, totals, model.labels_, model.n_clusters, data_type,
        task_profile, format_profile, task_colors=task_colors,
        legend_order=legend_order,
    )
    # Format columns may extend right of the last panel; keep them in the image.
    fig.savefig(output_dir / f"{filename_prefix}cluster_profiles.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)

    labels = model.labels_
    clusters = range(model.n_clusters)
    cluster_colors = build_cluster_profile_colors(model.n_clusters)
    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(features)
    fig, ax = plt.subplots(figsize=(8, 6))
    representatives = {}
    for cluster in clusters:
        positions = np.flatnonzero(labels == cluster)
        ax.scatter(*coordinates[positions].T, s=22, alpha=0.65,
                   color=cluster_colors[cluster], label=f"Cluster {cluster}")
        distances = np.linalg.norm(
            features.iloc[positions].to_numpy() - model.cluster_centers_[cluster], axis=1
        )
        representative = positions[np.argmin(distances)]
        code = str(frame.index[representative])
        representatives[str(cluster)] = code
        ax.annotate(code, coordinates[representative], xytext=(5, 5),
                    textcoords="offset points", fontsize=9)
    variance = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({variance[0]:.1%} explained variance)")
    ax.set_ylabel(f"PC2 ({variance[1]:.1%} explained variance)")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / f"{filename_prefix}cluster_pca.png", dpi=180)
    plt.close(fig)
    pca_summary = summarize_pca(frame, coordinates, variance, labels, data_type)
    return variance.tolist(), representatives, pca_summary


@dataclass
class ClusteringInput:
    """Read/normalized inputs reusable across hyperparameter/seed combinations.

    Treat these objects as read-only. No output directory is opened or created.
    """
    input_dir: Path
    data_type: str
    config: dict
    language_codes: set
    frame: pd.DataFrame
    totals: pd.Series
    usable: pd.Series
    excluded: dict
    zero_sources: pd.Series
    task_profile: pd.DataFrame
    format_profile: pd.DataFrame
    source_counts: pd.DataFrame = None


@dataclass
class ClusteringRun:
    """One computation without plots or file writes; labels are quantity-ordered."""
    prepared: ClusteringInput
    config: dict
    selected: pd.DataFrame
    features: pd.DataFrame
    model: KMeans
    metrics: dict
    label_mapping: dict


def prepare_clustering(input_folder=None, data_type="pretrain", *, root=None):
    """Read the same reduced JSONs and YAML settings as the normal CLI once."""
    if data_type not in ("pretrain", "posttrain"):
        raise ValueError("data_type must be 'pretrain' or 'posttrain'")
    project_root = ROOT if root is None else Path(root).expanduser().resolve()
    folder = input_folder if input_folder is not None else f"meta_data_{data_type}_reduced"
    input_dir = (project_root / folder).resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    config = dict(yaml.safe_load(
        (project_root / "clustering.yaml").read_text(encoding="utf-8")
    )[data_type])
    language_info = json.loads(
        (project_root / "language_639-3_info.json").read_text(encoding="utf-8-sig")
    )
    language_codes = living_language_codes(language_info)
    attributes = config["cluster_attributes"]
    if len(attributes) < 2 or not attributes[0].endswith(" log10"):
        raise ValueError("cluster_attributes must start with a log10 quantity, then sources")
    frame, totals, usable, excluded = read_data(input_dir, attributes, language_codes)
    zero_sources = frame.iloc[:, 1:].sum(axis=1).eq(0)
    frame = normalize_sources(frame)
    task_profile = read_categorical_profile(
        input_dir, frame.index, "Task Macro Categories Statistics"
    )
    format_profile = read_categorical_profile(
        input_dir, frame.index, "Format Statistics"
    )
    source_counts = read_source_counts(input_dir, frame.index)
    if not usable.any():
        raise ValueError("No ISO-Type-L languages with usable positive quantities")
    return ClusteringInput(
        input_dir, data_type, config, language_codes, frame, totals, usable,
        excluded, zero_sources, task_profile, format_profile,
        source_counts,
    )


def run_clustering(alpha, n_clusters, *, data_type, input_folder=None,
                   random_seed=None, evaluate_stability=False, prepared=None):
    """Compute a clustering for (alpha, n_clusters), with no plots or saves.

    The default is exactly ONE seed (YAML random_state unless overridden),
    with YAML n_init restarts. Set evaluate_stability=True only for the normal
    report workflow, which retains the original ten consecutive seeds.
    Pass a ClusteringInput from prepare_clustering() to reuse loaded data.
    The returned silhouette is the overall mean for the main seed in the
    FULL weighted feature space, excluding unclustered (-1) languages.
    """
    alpha = validate_alpha(alpha)
    n_clusters = validate_n_clusters(n_clusters)
    if data_type not in ("pretrain", "posttrain"):
        raise ValueError("data_type must be 'pretrain' or 'posttrain'")
    if not isinstance(evaluate_stability, bool):
        raise ValueError("evaluate_stability must be a boolean")
    if prepared is None:
        prepared = prepare_clustering(input_folder, data_type)
    elif not isinstance(prepared, ClusteringInput) or prepared.data_type != data_type:
        raise ValueError("prepared inputs must match the requested data_type")
    elif input_folder is not None:
        raise ValueError("Pass either prepared inputs or input_folder, not both")

    # Never mutate the prepared configuration or the on-disk YAML.
    config = dict(prepared.config)
    seed = validate_random_seed(
        config["random_state"] if random_seed is None else random_seed
    )
    config.update(alpha=alpha, n_clusters=n_clusters, random_state=seed)
    selected = prepared.frame.loc[prepared.usable]
    features = build_features(selected, alpha)
    model, metrics = fit_clusters(
        features, config, random_seeds=None if evaluate_stability else [seed]
    )
    label_mapping = relabel_clusters(model, prepared.totals.loc[selected.index])
    metrics["cluster_sizes"] = {
        str(cluster): int((model.labels_ == cluster).sum())
        for cluster in range(model.n_clusters)
    }
    return ClusteringRun(prepared, config, selected, features, model, metrics, label_mapping)


def save_clustering_run(run, output_folder, *, print_results=True, prefix_filenames=False,
                        task_colors=None, legend_order=None):
    """Save the normal seven reports from an already fitted ClusteringRun.

    Does NOT fit KMeans or read YAML again. Cluster labels, config and scalar
    silhouette describe this run's reference seed. A grid caller may supply
    explicit multi-seed stability metrics in a copied run.metrics dictionary.
    Ordinary main() and grid search share this reporting implementation.
    The CLI enables stage-prefixed filenames; grid reports retain their names.
    Task colors use the shared canonical mapping, optionally supplied by a grid caller.
    Legend ordering is shared with grid reports and does not filter plotted data.
    """
    if not isinstance(run, ClusteringRun):
        raise TypeError("run must be a ClusteringRun")
    if not isinstance(print_results, bool):
        raise ValueError("print_results must be a boolean")
    if not isinstance(prefix_filenames, bool):
        raise ValueError("prefix_filenames must be a boolean")
    prepared = run.prepared
    data_type = prepared.data_type
    filename_prefix = f"{data_type}_" if prefix_filenames else ""
    output_dir = (ROOT / Path(output_folder).expanduser()).resolve()
    if output_dir == prepared.input_dir.resolve():
        raise ValueError("Input and output folders must be different")
    config, language_codes = run.config, prepared.language_codes
    attributes = config["cluster_attributes"]
    frame, totals, usable = prepared.frame, prepared.totals, prepared.usable
    excluded, zero_sources = prepared.excluded, prepared.zero_sources
    selected, features = run.selected, run.features
    model, metrics, label_mapping = run.model, deepcopy(run.metrics), run.label_mapping
    summary, centers = cluster_summaries(
        selected, totals.loc[selected.index], model.labels_, len(language_codes) - len(selected)
    )

    full_silhouettes = silhouette_samples(features, model.labels_, metric="euclidean")
    summary = extend_cluster_summary(
        summary, centers, selected, totals.loc[selected.index], model.labels_,
        full_silhouettes, data_type,
    )

    # Append the ISO-Type-L registry only after fitting; the manual group never affects
    # the scaler, K-means, silhouette, stability analysis, or PCA.
    result = frame.reindex(sorted(language_codes))
    result["Cluster"] = -1
    result["Cluster Status"] = "Uncovered"
    # Both absent records and unusable quantities share -1; preserve the reason in status.
    result.loc[frame.index, "Cluster Status"] = "Insufficient data"
    result.loc[selected.index, "Cluster"] = model.labels_
    result.loc[selected.index, "Cluster Status"] = "Clustered"
    source_counts = prepared.source_counts
    if source_counts is None:
        source_counts = read_source_counts(prepared.input_dir, frame.index)
    summary = extend_cluster_resource_summary(
        summary, source_counts, totals, result["Cluster"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    variance, representatives, pca_summary = plot_clusters(
        selected, totals.loc[usable], features, model, output_dir, data_type,
        prepared.task_profile.loc[selected.index],
        prepared.format_profile.loc[selected.index],
        filename_prefix=filename_prefix,
        task_colors=task_colors,
        legend_order=legend_order,
    )
    metrics.update({
        "data_type": data_type,
        "config": config,
        "quantity_key": attributes[0].removesuffix(" log10"),
        "cluster_order": "ascending arithmetic mean raw Num; stable old-label order for ties",
        "old_to_new_cluster_labels": label_mapping,
        "cluster_sizes": {
            str(cluster): int((model.labels_ == cluster).sum())
            for cluster in range(model.n_clusters)
        },
        "summary_policy": {
            "centers": "arithmetic means of raw Num and normalized/imputed pre-Hellinger proportions",
            "source_weighting": "equal weight per language, not quantity-weighted",
            "task_format_profiles": "equal weight per language; task and format are descriptive only and do not enter clustering",
            "size_names": "below 1/3 quantile: low; above 2/3 quantile: high; otherwise medium, across cluster mean Num",
            "source_names": "single source >50%; else human+expert >50%; else mixed",
            "manual_group": "Cluster -1 has no numeric center and is excluded from ordering",
            "overall_row": "整体 includes fitted languages only; quantities and source proportions are language-weighted, not cluster-weighted",
            "silhouette": "per-language Euclidean silhouettes in the full weighted feature space, averaged within each cluster or over all fitted languages",
            "pure_source_count": "exact one-hot normalized source vector; includes upstream default assignments, not necessarily independently observed provenance",
            "pure_source_category": "Collect-Existed" if data_type == "pretrain" else "Machine-Generated",
            "coverage_counts": "language-equal means, sample standard deviations (ddof=1), and medians of reduced num_collection_sources and num_dataset_sources; absent language records count as zero; singleton standard deviations are undefined",
            "quantity_gini": "math_utils.gini on raw NCC/Num Dialogs for fitted languages; Cluster -1 is undefined; overall is recomputed over all fitted languages",
            "dataset_count_gini": "math_utils.gini on num_dataset_sources for each cluster, including -1; absent records count as zero; all-zero vectors are undefined; overall includes fitted languages only",
        },
        "registry_languages": len(language_codes),
        "registry_policy": "ISO_639_3_Type == L only",
        "covered_languages": len(frame),
        "uncovered_languages": len(language_codes) - len(frame),
        "insufficient_data_languages": len(frame) - len(selected),
        "clustered_languages": len(selected),
        "zero_source_vector_languages": int(zero_sources.sum()),
        # Keep the old key for readers of existing metric files; no uniform
        # imputation is performed. New counters cover only this final fallback,
        # not record-level defaults already applied by Group/Reduce.
        "uniform_imputed_clustered_languages": 0,
        "default_source_fallback_languages": int(zero_sources.sum()),
        "default_source_fallback_clustered_languages": int((zero_sources & usable).sum()),
        "default_source_category": default_source_category(attributes[0]),
        "source_fallback_is_observed_provenance": False,
        "source_policy": (
            "empty record sources use stage default before aggregation; "
            "null proportions=0; row normalization; zero-sum rows assigned 100% to "
            + default_source_category(attributes[0])
        ),
        "manual_cluster_labels": {"-1": "Uncovered or no usable positive quantity"},
        "output_cluster_sizes": {
            str(cluster): int((result["Cluster"] == cluster).sum())
            for cluster in [-1, *range(model.n_clusters)]
        },
        "excluded": excluded,
        "PCA_explained_variance_ratio": variance,
        "PCA_summary_policy": {
            "variance": "explained variance ratios in [0,1]; total is PC1+PC2",
            "correlation": "Pearson r against log10 quantity and normalized source shares before the square-root transform; constants yield blank cells",
            "silhouette": "fixed full-space cluster labels, scored separately on PC1, PC2, and joint PC1/PC2; total is the joint 2D score",
            "population": "fitted languages only; Cluster -1 excluded",
        },
        "negative_silhouette_languages": int((full_silhouettes < 0).sum()),
        "representative_languages": representatives,
    })
    result.to_csv(output_dir / f"{filename_prefix}cluster_results.csv", encoding="utf-8-sig")
    features.to_csv(output_dir / f"{filename_prefix}clustering_matrix.csv", encoding="utf-8-sig")
    summary.to_csv(output_dir / f"{filename_prefix}cluster_summary.csv", encoding="utf-8-sig")
    pca_summary.to_csv(output_dir / f"{filename_prefix}PCA_summary.csv", encoding="utf-8-sig", na_rep="")
    (output_dir / f"{filename_prefix}cluster_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    # Centers are now part of cluster_summary.csv. Clean only these known
    # legacy generated files, and only after all new reports were saved.
    for legacy_name in ("cluster_centers.csv", "cluster_center.csv"):
        legacy_path = output_dir / f"{filename_prefix}{legacy_name}"
        if legacy_path.is_file():
            legacy_path.unlink()
    if print_results:
        print("Cluster summary (centers included; 整体 excludes Cluster -1):")
        print(summary.to_string())
        print("\nPCA summary (fixed cluster labels; total silhouette uses PC1+PC2):")
        print(pca_summary.to_string())
        print(f"\nSilhouette: {metrics['silhouette_score']:.4f}")
        if metrics['pairwise_ARI_mean'] is None or metrics['pairwise_ARI_min'] is None:
            print("Pairwise ARI: mean=N/A, min=N/A (only one seed)")
        else:
            print(f"Pairwise ARI: mean={metrics['pairwise_ARI_mean']:.4f}, "
                  f"min={metrics['pairwise_ARI_min']:.4f}")
        count = metrics["insufficient_data_languages"]
        print(f"EXCLUDED SUMMARY no usable positive quantity: {count}/{len(frame)} languages "
              f"({count / len(frame):.2%} of valid ISO input languages)")
        print(f"Saved clustering results to {output_dir}")
    return metrics


def main(input_folder, output_folder, data_type):
    input_dir = (ROOT / input_folder).resolve()
    output_dir = (ROOT / output_folder).resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    if input_dir == output_dir:
        raise ValueError("Input and output folders must be different")
    prepared = prepare_clustering(input_folder, data_type)
    run = run_clustering(
        prepared.config["alpha"], prepared.config["n_clusters"],
        data_type=data_type, prepared=prepared, evaluate_stability=True,
    )
    save_clustering_run(run, output_dir, prefix_filenames=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    parser.add_argument("output_folder")
    parser.add_argument("--data_type", required=True, choices=("pretrain", "posttrain"))
    args = parser.parse_args()
    main(args.input_folder, args.output_folder, args.data_type)
