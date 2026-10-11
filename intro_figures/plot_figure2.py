#!/usr/bin/env python3
"""Figure 2A: complete cross-stage language-resource profiles (latest K=3 results).

Data are read from the current pretrain/posttrain *_cluster_results.csv files,
not from the historical six-example figure_data.json snapshot. One marker is
rendered per language with a finite, positive pretraining quantity.

Measured positions are (log10 NCC, log10 post-training sample equivalents).
Languages assigned post-training Cluster -1 have no usable numeric quantity;
they are shown on a separate, explicitly categorical strip (not log10(0)).

Colors encode five semantic cross-stage development regimes, with lighter
and darker variants for paired subgroups. The (1,1) combination is currently
empty and therefore not included in the main semantic legend; if it appears
in future data, it is drawn and reported separately rather than discarded.
Other excluded combinations, if any, are reported.

Run from the project root:
    python intro_figures/plot_figure2.py
or:
    python intro_figures/plot_figure2.py --root /path/to/project

Outputs (PNG, SVG, PDF, JSON audit):
    intro_figures/figure2_resource_profiles.*
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


PAIR_ORDER = (
    (0, -1), (0, 0),
    (1, -1), (1, 0), (1, 1),
    (2, -1), (2, 0), (2, 1), (2, 2),
)

# Five mutually interpretable development regimes; distinct hues separate
# strategies, while light/dark shades distinguish relative pre-/post-coverage.
# The additional (1,1) pair has zero observations in the current snapshot.
PAIR_COLORS = {
    (0, -1): "#EE879D",  # improve pre-training: light rose
    (0, 0): "#C62845",   # improve pre-training: deep crimson
    (1, -1): "#78B8E2",  # build post-training: light blue
    (2, -1): "#2876B7",  # build post-training: deep blue
    (1, 0): "#89C36F",   # post-training lag: light green
    (2, 0): "#4D8339",   # post-training lag: deep green
    (2, 1): "#F28E2B",   # diversification: orange
    (2, 2): "#EDB62C",   # mature: gold
    (1, 1): "#A9B1B9",   # unclassified/rare pair, shown only if present
}

# Legend order is semantic, not alphabetical or by pre-training cluster.
# The second member of two-pair groups receives the deeper shade.
SEMANTIC_GROUPS = (
    ("Strengthen pre-training", ((0, -1), (0, 0))),
    ("Build post-training from pre-training", ((1, -1), (2, -1))),
    ("Scale up lagging post-training", ((1, 0), (2, 0))),
    ("Diversify post-training", ((2, 1),)),
    ("Two-stage rich resources", ((2, 2),)),
)

# Recognizable, real cases chosen only if they belong to the given pair.
# A median-nearest actual language is used if the preferred code disappears.
PREFERRED_CODES = {
    (0, -1): "aar",  # Afar
    (0, 0): "rhg",   # Rohingya
    (1, -1): "ach", # Acoli
    (1, 0): "ain",  # Ainu
    (2, -1): "nav", # Navajo
    (2, 0): "abk",  # Abkhazian
    (2, 1): "zul",  # Zulu
    (2, 2): "eng",  # English
}

DISPLAY_NAMES = {"ain": "Ainu", "ach": "Acoli"}

# Offsets in display points; no alteration of the original data coordinate.
LABEL_OFFSETS = {
    (0, -1): (-14, 19),
    (1, -1): (-36, 24),
    (2, -1): (16, -19),
    (0, 0): (-25, 25),
    (1, 0): (21, -25),
    (2, 0): (22, -18),
    (2, 1): (20, -18),
    (2, 2): (-43, -16),
}

INK = "#253447"
MUTED = "#5F6F7E"
GRID = "#E1E7EC"


def read_stage(path, stage):
    """Read stage labels without interpreting ISO code 'nan' as missing."""
    if not path.is_file():
        raise FileNotFoundError(path)
    data = pd.read_csv(path, keep_default_na=False, dtype={"Language": str})
    if data["Language"].duplicated().any():
        raise ValueError(f"Duplicated language keys: {path}")
    value_col = (
        "Num Unicode Characters log10" if stage == "pretrain"
        else "Num Dialogs log10"
    )
    required = {"Language", "Cluster", value_col}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing CSV columns {required - set(data.columns)}: {path}")
    data = data.set_index("Language")
    clusters = pd.to_numeric(data["Cluster"], errors="raise")
    if clusters.isna().any() or not np.isclose(clusters % 1, 0).all():
        raise ValueError(f"Nonintegral cluster labels: {path}")
    data["Cluster"] = clusters.astype(int)
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    return data[["Cluster", value_col]]


def collect_points(root):
    pre = read_stage(
        root / "meta_data_pretrain_clustered/pretrain_cluster_results.csv", "pretrain"
    ).rename(columns={
        "Cluster": "pre_cluster", "Num Unicode Characters log10": "log_pre"
    })
    post = read_stage(
        root / "meta_data_posttrain_clustered/posttrain_cluster_results.csv", "posttrain"
    ).rename(columns={
        "Cluster": "post_cluster", "Num Dialogs log10": "log_post"
    })
    if not pre.index.equals(post.index):
        if set(pre.index) != set(post.index):
            raise ValueError("Pre- and post-training cluster CSVs have different language keys")
        post = post.reindex(pre.index)

    points = pre.loc[pre.pre_cluster.ge(0)].join(post, how="left")
    if points.empty:
        raise ValueError("No languages with usable pretraining quantity")
    if not np.isfinite(points.log_pre.to_numpy(dtype=float)).all():
        raise ValueError("Pretraining-covered languages have invalid log quantities")
    if points.post_cluster.isna().any():
        raise ValueError("Some pretraining languages lack a post-training cluster label")
    points["post_cluster"] = points.post_cluster.astype(int)
    has_post = points.post_cluster.ge(0)
    if not np.isfinite(points.loc[has_post, "log_post"].to_numpy(dtype=float)).all():
        raise ValueError("Positive post-training cluster labels require positive quantities")
    points["pair"] = list(zip(points.pre_cluster, points.post_cluster))
    keep = points.pair.isin(PAIR_ORDER)
    excluded = points.loc[~keep]
    return points.loc[keep].copy(), excluded


def display_name(code, registry):
    if code in DISPLAY_NAMES:
        return DISPLAY_NAMES[code]
    entries = registry.get(code, {}).get("English_Name", [])
    if entries:
        return str(entries[0]).replace(" (individual language)", "")
    return code


def choose_examples(points):
    result = {}
    for pair in PAIR_ORDER:
        group = points.loc[points.pair.map(lambda current: current == pair)]
        if group.empty:
            continue
        preferred = PREFERRED_CODES.get(pair)
        if preferred in group.index:
            result[pair] = preferred
            continue
        x = group.log_pre.to_numpy(dtype=float)
        y = group.log_post.to_numpy(dtype=float) if pair[1] >= 0 else None
        if y is None:
            distance = abs(x - np.median(x))
        else:
            # Both axes are already in log10-space.
            distance = ((x - np.median(x)) ** 2
                        + (y - np.median(y)) ** 2)
        result[pair] = str(group.index[int(np.argmin(distance))])
    return result


def stable_missing_jitter(code):
    """Small deterministic categorical-strip jitter, never a quantity estimate."""
    value = int.from_bytes(hashlib.blake2s(code.encode("utf-8"), digest_size=4).digest(), "big")
    return 0.28 + (value / (2 ** 32 - 1)) * 0.44


def plot(points, examples, registry, output_dir, dpi):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.1,
        "axes.labelsize": 10.1, "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(11.3, 6.65), facecolor="white")
    grid = GridSpec(
        2, 1, figure=fig, height_ratios=[4.45, 1.15],
        left=0.102, right=0.975,
        top=0.935, bottom=0.13, hspace=0.055,
    )
    ax = fig.add_subplot(grid[0, 0])
    ax_missing = fig.add_subplot(grid[1, 0], sharex=ax)

    # The numeric axes are log10(Q), so 6 means 10^6 units (no log(0)).
    xvals = points.log_pre.to_numpy(dtype=float)
    xp_min = 0.0  # Show the full pre-training lag relative to post-training.
    xp_max = max(15.0, np.ceil(float(xvals.max()) + 0.15))
    ax.set_xlim(xp_min, xp_max)
    main = points.loc[points.post_cluster.ge(0)]
    yvals = main.log_post.to_numpy(dtype=float)
    ymin = min(-0.25, np.floor(float(yvals.min())) - 0.2)
    ymax = max(8.5, np.ceil(float(yvals.max())) + 0.4)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(r"Post-training $\log_{10}(Q)$", labelpad=10, color=INK)
    ax.set_xticks(np.arange(0, 16, 2))
    ax.set_yticks(np.arange(0, 10, 2))
    ax.tick_params(axis="x", labelbottom=False)
    ax.tick_params(axis="both", which="major", direction="out", length=3.0, colors=MUTED)
    ax.grid(which="major", color=GRID, linewidth=0.75, zorder=0)
    ax.spines["left"].set_color("#B7C4CD")
    ax.spines["bottom"].set_color("#B7C4CD")
    ax.set_axisbelow(True)

    counts = Counter(points.pair.tolist())
    for pair in PAIR_ORDER:
        if pair[1] == -1:
            continue
        sub = points.loc[points.pair.map(lambda current: current == pair)]
        if sub.empty:
            continue
        ax.scatter(
            sub.log_pre, sub.log_post,
            s=20 if len(sub) < 50 else 14,
            color=PAIR_COLORS[pair], alpha=0.79,
            edgecolor="white", linewidth=0.24, zorder=3,
        )

    # A separate categorical row avoids assigning missing post-training Q a
    # fabricated numerical value or drawing it on a logarithmic axis.
    ax_missing.set_ylim(0.02, 0.98)
    ax_missing.set_facecolor("#F4F6F8")
    ax_missing.tick_params(axis="y", left=False, labelleft=False)
    ax_missing.tick_params(axis="x", direction="out", length=3, colors=MUTED)
    ax_missing.set_xticks(np.arange(0, 16, 2))
    ax_missing.set_xlabel(r"Pre-training $\log_{10}(Q)$", labelpad=9, color=INK)
    for name in ("left", "right", "bottom", "top"):
        ax_missing.spines[name].set_visible(False)
    ax_missing.axhline(0.96, color="#CFD8DF", linewidth=0.9, zorder=0)
    ax_missing.text(
        0.01, 0.99, "No usable post-training datasets observed",
        transform=ax_missing.transAxes, ha="left", va="top",
        fontsize=8.4, color=MUTED,
    )
    for pair in PAIR_ORDER:
        if pair[1] != -1:
            continue
        sub = points.loc[points.pair.map(lambda current: current == pair)]
        if sub.empty:
            continue
        jitter = np.asarray([stable_missing_jitter(str(lang)) for lang in sub.index])
        ax_missing.scatter(
            sub.log_pre.to_numpy(dtype=float), jitter,
            s=11, color=PAIR_COLORS[pair], alpha=0.50,
            linewidth=0.0, zorder=2,
        )

    # Annotate exactly one real language per nonempty pair.
    selections = {}
    for pair, lang in examples.items():
        row = points.loc[lang]
        axis = ax_missing if pair[1] == -1 else ax
        x = float(row.log_pre)
        y = stable_missing_jitter(lang) if pair[1] == -1 else float(row.log_post)
        color = PAIR_COLORS[pair]
        axis.scatter([x], [y], s=60, c=[color], edgecolor="#243344",
                     linewidth=0.85, zorder=8)
        label = display_name(lang, registry)
        offset = LABEL_OFFSETS.get(pair, (20, 16))
        axis.annotate(
            label, (x, y), xytext=offset,
            textcoords="offset points", fontsize=9.0,
            fontweight="bold", color=INK, ha="left" if offset[0] >= 0 else "right",
            va="center", clip_on=False, zorder=9,
            arrowprops=dict(arrowstyle="-", linewidth=0.9, color=color,
                            shrinkA=3, shrinkB=4),
        )
        selections[f"{pair[0]},{pair[1]}"] = {
            "language_code": lang, "language_name": label,
            "log10_pretraining_NCC": x,
            "log10_posttraining_samples": None if pair[1] == -1 else float(row.log_post),
        }

    # Semantic legend inside the numeric axes, placed in the low-density
    # upper-left area. A paired regime displays both markers in one row.
    # No numeric coordinates or cluster labels are altered by this overlay.
    total = len(points)
    ax.text(
        0.018, 0.975,
        f"Living languages with pre-training corpus plotted (N={total:,})",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=10.0, fontweight="bold", color=INK, zorder=11,
    )
    ax.text(
        0.018, 0.894, "Development regime:",
        transform=ax.transAxes, ha="left", va="center",
        fontsize=9.4, fontweight="bold", color=INK, zorder=11,
    )
    heading_y = (0.836, 0.739, 0.642, 0.545, 0.448)
    marker_y = (0.797, 0.700, 0.603, 0.506, 0.409)
    for (heading, pairs), y_head, y_pair in zip(
        SEMANTIC_GROUPS, heading_y, marker_y
    ):
        ax.text(
            0.018, y_head, heading,
            transform=ax.transAxes, ha="left", va="center",
            fontsize=8.7, fontweight="normal", color=INK, zorder=11,
        )
        for i, pair in enumerate(pairs):
            # Each two-shade regime appears side-by-side on one row.
            x = 0.039 + 0.132 * i
            ax.scatter(
                [x], [y_pair], s=60, color=PAIR_COLORS[pair],
                edgecolor="white", linewidth=0.55,
                transform=ax.transAxes, clip_on=False, zorder=12,
            )
            ax.text(
                x + 0.020, y_pair, f"(N={counts.get(pair, 0):,})",
                transform=ax.transAxes, ha="left", va="center",
                fontsize=8.8, color=INK, zorder=12,
            )

    # A currently empty (1, 1) combination has no marker. If a future
    # snapshot makes it nonempty, show it rather than silently hiding data.
    if counts.get((1, 1), 0):
        ax.text(0.018, 0.359, "Other observed regime", transform=ax.transAxes,
                fontsize=8.2, color=INK, va="center", zorder=11)
        ax.scatter([0.039], [0.330], s=55, color=PAIR_COLORS[(1, 1)],
                   edgecolor="white", linewidth=0.5,
                   transform=ax.transAxes, zorder=12)
        ax.text(0.059, 0.330, f"(N={counts[(1, 1)]:,})",
                transform=ax.transAxes, fontsize=8.5, color=INK,
                va="center", zorder=12)

    for ext in ("png", "svg", "pdf"):
        output = output_dir / f"figure2_resource_profiles.{ext}"
        fig.savefig(output, dpi=dpi, facecolor="white")
        print(f"Saved {output}")
    plt.close(fig)
    return selections, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent.parent,
        help="Project root with the current *_cluster_results.csv files",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Default: <root>/intro_figures",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    out = (args.output_dir or root / "intro_figures").expanduser().resolve()
    if args.dpi < 80:
        parser.error("--dpi must be >= 80")
    points, excluded = collect_points(root)
    examples = choose_examples(points)
    registry_path = root / "language_639-3_info.json"
    registry = (json.loads(registry_path.read_text(encoding="utf-8-sig"))
                if registry_path.is_file() else {})
    out.mkdir(parents=True, exist_ok=True)
    selected, counts = plot(points, examples, registry, out, args.dpi)
    report = {
        "pretrain_languages_plotted": len(points),
        "posttrain_unusable_languages_in_plotted_set": int(points.post_cluster.eq(-1).sum()),
        "posttrain_positive_languages_in_plotted_set": int(points.post_cluster.ge(0).sum()),
        "requested_pair_order": [list(p) for p in PAIR_ORDER],
        "pair_counts": {f"{p[0]},{p[1]}": counts.get(p, 0) for p in PAIR_ORDER},
        "pair_colors": {f"{p[0]},{p[1]}": PAIR_COLORS[p] for p in PAIR_ORDER},
        "semantic_groupings": {
            heading: [f"{p[0]},{p[1]}" for p in pairs]
            for heading, pairs in SEMANTIC_GROUPS
        },
        "excluded_posttrain_richer_pairs": {
            f"{p[0]},{p[1]}": int(n)
            for p, n in Counter(excluded.pair.tolist()).items()
        },
        "labelled_languages": selected,
        "missing_posttrain_policy": (
            "Categorical strip, not a fabricated log10(0) coordinate; "
            "vertical jitter is visual only."
        ),
        "x_coordinate": "log10(pretraining NCC); displayed axis starts at zero",
        "y_coordinate_measured": "log10(posttraining sample equivalents)",
        "legend_position": "upper-left inside measured-data panel",
        "inputs": [
            str(root / "meta_data_pretrain_clustered/pretrain_cluster_results.csv"),
            str(root / "meta_data_posttrain_clustered/posttrain_cluster_results.csv"),
        ],
    }
    audit_path = out / "figure2_resource_profiles_summary.json"
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {audit_path}")
    print("Pair counts:", report["pair_counts"])
    print("Excluded newer-stage-richer pairs:", report["excluded_posttrain_richer_pairs"])
    print("Annotated:", {k: v["language_name"] for k,v in selected.items()})


if __name__ == "__main__":
    main()
