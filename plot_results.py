"""Plot alpha/K sensitivity and both stages' profiles without rerunning clustering."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import pandas as pd
import yaml

# cluster_analysis uses Agg for batch reports; retain this script's display backend.
_display_backend = plt.get_backend()
from cluster_analysis import (
    build_cluster_profile_colors,
    build_profile_legend_order,
    create_cluster_profile_figure,
    read_categorical_profile,
    read_data,
)
plt.switch_backend(_display_backend)


ROOT = Path(__file__).resolve().parent
METRICS = (
    ("R_Q", r"$R_Q$", "o"),
    ("R_S", r"$R_S$", "s"),
    ("R_balanced", r"$R_{\mathrm{balanced}}$", "^"),
)


def load_alpha_results(stage, configured_k):
    """Fix the configured K and collect full-precision seed means across alpha."""
    results = {}
    for folder in (ROOT / "cluster_grid_search_results").glob("result_*_*"):
        saved_alpha, saved_k = folder.name.removeprefix("result_").rsplit("_", 1)
        k = int(saved_k)
        if k != configured_k:
            continue
        alpha = float(saved_alpha)
        path = folder / f"meta_data_{stage}_clustered" / "cluster_metrics.json"
        grid = json.loads(path.read_text(encoding="utf-8"))["grid_search"]
        if grid["alpha"] != alpha or grid["n_clusters"] != k:
            raise ValueError(f"Grid parameters do not match the folder: {path}")
        if alpha in results:
            raise ValueError(f"Duplicate result for {stage}, alpha={alpha}, K={k}")
        results[alpha] = grid["seed_mean_R"]

    if not results:
        raise ValueError(f"No saved results for {stage}, K={configured_k}")
    return dict(sorted(results.items()))


def load_k_results(stage, configured_alpha):
    """Fix alpha and collect seed-mean overall Silhouette across K."""
    results = {}
    for folder in (ROOT / "cluster_grid_search_results").glob("result_*_*"):
        saved_alpha, saved_k = folder.name.removeprefix("result_").rsplit("_", 1)
        alpha = float(saved_alpha)
        if alpha != configured_alpha:
            continue
        k = int(saved_k)
        path = folder / f"meta_data_{stage}_clustered" / "cluster_metrics.json"
        grid = json.loads(path.read_text(encoding="utf-8"))["grid_search"]
        if grid["alpha"] != alpha or grid["n_clusters"] != k:
            raise ValueError(f"Grid parameters do not match the folder: {path}")
        if k in results:
            raise ValueError(f"Duplicate result for {stage}, alpha={alpha}, K={k}")
        results[k] = {"seed_mean_silhouette": grid["seed_mean_silhouette"]}

    if not results:
        raise ValueError(f"No saved results for {stage}, alpha={configured_alpha}")
    return dict(sorted(results.items()))


def load_cluster_profiles(stage, attributes):
    """Reuse saved fitted labels and the same reduced statistics as cluster_analysis."""
    path = ROOT / f"meta_data_{stage}_clustered" / f"{stage}_cluster_results.csv"
    # Preserve literal ISO identifiers such as 'nan'; omit the manual cluster -1.
    result = pd.read_csv(path, keep_default_na=False).set_index("Language")
    fitted = result.loc[result["Cluster"].ge(0)]
    if fitted.empty or not fitted.index.is_unique:
        raise ValueError(f"Expected unique fitted language rows: {path}")
    frame = fitted[attributes].astype(float)
    input_dir = ROOT / f"meta_data_{stage}_reduced"
    _, totals, _, _ = read_data(input_dir, attributes, set(frame.index))
    totals = totals.reindex(frame.index)
    if totals.isna().any() or totals.le(0).any():
        raise ValueError(f"Saved clusters require positive reduced quantities: {path}")
    return {
        "frame": frame,
        "totals": totals,
        "labels": fitted["Cluster"].to_numpy(dtype=int),
        "n_clusters": int(fitted["Cluster"].max()) + 1,
        "data_type": stage,
        "task_profile": read_categorical_profile(
            input_dir, frame.index, "Task Macro Categories Statistics"
        ),
    }


def add_boxplot_legend(figure):
    """Explain boxplot symbols below the quantity panel, aligned with its neighbors."""
    quantity_ax = figure.axes[0]
    whisker = Line2D([], [], color="black", marker="|", markersize=8)
    # Illustrate all cluster-colored boxes, not an unrelated gray swatch.
    box_patches = tuple(
        Patch(facecolor=color, edgecolor="black", linewidth=0.6, alpha=0.6)
        for color in build_cluster_profile_colors(len(quantity_ax.get_yticks()))
    )
    handles = [
        Line2D([], [], linestyle="none", marker="|", markersize=12,
               color=plt.rcParams["boxplot.medianprops.color"]),
        box_patches,
        whisker,
        Line2D([], [], linestyle="none", marker="o", color="black",
               markerfacecolor="none", markersize=6),
    ]
    labels = [
        "Median",
        r"Box: $\in [Q_1, Q_3]$",
        r"Whiskers: $\in [Q_1-1.5\,\mathrm{IQR}, Q_3+1.5\,\mathrm{IQR}]$",
        r"Outliers: $\notin [Q_1-1.5\,\mathrm{IQR}, Q_3+1.5\,\mathrm{IQR}]$",
    ]
    # Reuse the source legend's figure-coordinate anchor to align the legend tops.
    anchor = figure.axes[1].get_legend().get_bbox_to_anchor().transformed(
        figure.transFigure.inverted()
    )
    position = quantity_ax.get_position()
    quantity_ax.legend(
        handles, labels, loc="upper center", ncol=1, fontsize=10, frameon=False,
        handlelength=1.4, borderaxespad=0.0,
        handler_map={whisker: HandlerLine2D(numpoints=2),
                     box_patches: HandlerTuple(ndivide=None, pad=0.1)},
        bbox_to_anchor=((position.x0 + position.x1) / 2, anchor.y0),
        bbox_transform=figure.transFigure,
    )


def main(power=None):
    config = yaml.safe_load((ROOT / "clustering.yaml").read_text(encoding="utf-8"))
    # Load all saved results before creating figures or writing output files.
    stages = ("pretrain", "posttrain")
    alpha_results = {
        stage: load_alpha_results(stage, int(config[stage]["n_clusters"]))
        for stage in stages
    }
    k_results = {
        stage: load_k_results(stage, float(config[stage]["alpha"]))
        for stage in stages
    }
    profiles = {
        stage: load_cluster_profiles(stage, config[stage]["cluster_attributes"])
        for stage in stages
    }

    with plt.rc_context({"pdf.fonttype": 42}):
        output_dir = ROOT / "plot_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        figures = []
        # Figure 1 varies alpha at the selected K; Figure 2 varies K at selected alpha.
        sensitivity_plots = (
            ("alpha", alpha_results, r"$\alpha$", 0.1, "alpha_sensitivity.pdf"),
            ("n_clusters", k_results, r"$K$", 1, "k_sensitivity_curves.pdf"),
        )
        for number, (parameter, results, xlabel, step, filename) in enumerate(
            sensitivity_plots, start=1
        ):
            # Alpha sensitivity uses R scores; K sensitivity uses overall Silhouette.
            metrics = METRICS if parameter == "alpha" else (
                ("seed_mean_silhouette", "Silhouette", "o"),
            )
            shared_k = parameter == "n_clusters" and (
                config["pretrain"]["n_clusters"] == config["posttrain"]["n_clusters"]
            )
            if parameter == "alpha":
                fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
            else:
                fig, ax = plt.subplots(figsize=(6, 4.5))
                axes = (ax, ax)  # Draw both stages on the same K-sensitivity axes.
            for ax, stage, title in zip(
                axes, stages, ("Pre-train", "Post-train")
            ):
                x_values = list(results[stage])
                color = None if parameter == "alpha" else (
                    "tab:blue" if stage == "pretrain" else "tab:orange"
                )
                for key, label, marker in metrics:
                    # Undefined scores remain NaN: do not fill zero or bridge gaps.
                    values = [
                        float("nan") if results[stage][x][key] is None
                        else results[stage][x][key]
                        for x in x_values
                    ]
                    ax.plot(x_values, values, marker=marker, markersize=6,
                            linewidth=1.2, alpha=0.85, color=color,
                            label=label if parameter == "alpha" else title.lower())

                # Mark the YAML-selected parameter; do not recompute an optimum.
                # Identical selected K values share one black marker.
                if not shared_k or stage == "pretrain":
                    ax.axvline(float(config[stage][parameter]),
                               color="gray" if parameter == "alpha" else (
                                   "black" if shared_k else color
                               ), linestyle="--", linewidth=1.2)
                ax.xaxis.set_major_locator(MultipleLocator(step))
                ax.set_xlabel(xlabel)
                if parameter == "alpha":
                    ax.set_title(title)
                    ax.set_ylabel("Explained variance (seed mean)")
                    ax.set_ylim(-0.02, 1.02)
                else:
                    ax.set_ylabel("Silhouette")
                    ax.set_yticks([0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7])
                    ax.set_ylim(0.4, 0.7)
                ax.tick_params(labelleft=True)
                ax.grid(True, linestyle="--", alpha=0.3)
                ax.set_axisbelow(True)

            if parameter == "alpha":
                handles, labels = axes[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="upper center", ncol=len(metrics), frameon=False)
                fig.tight_layout(rect=(0, 0, 1, 0.91))
            else:
                axes[0].legend(loc="upper right", ncol=1, frameon=False)
                fig.tight_layout()
            fig.canvas.manager.set_window_title(f"Figure {number}: {filename.removesuffix('.pdf')}")
            output_path = output_dir / filename
            fig.savefig(output_path, bbox_inches="tight")
            print(f"Saved: {output_path}")
            figures.append(fig)

        for number, stage in enumerate(stages, start=3):
            legend_order = build_profile_legend_order()
            if stage == "pretrain":
                legend_order["Task"] = ["pretrain", "classification", "translation"]
            profile_fig = create_cluster_profile_figure(
                **profiles[stage], quantity_label="log(Q)", cluster_count_label="N",
                power=power, legend_order=legend_order,
            )
            suffix = " (power-transformed)" if power is not None else ""
            for ax, name in zip(profile_fig.axes[1:3], ("Source", "Task")):
                ax.set_xlabel(f"{name} Proportions{suffix}")
            quantity_ax = profile_fig.axes[0]
            quantity_ax.set_yticks(
                quantity_ax.get_yticks(),
                [tick.get_text().replace(" (N=", "\n(N=")
                 for tick in quantity_ax.get_yticklabels()],
                multialignment="center", verticalalignment="center",
            )
            add_boxplot_legend(profile_fig)
            profile_fig.canvas.manager.set_window_title(f"Figure {number}: {stage}")
            for extension in ("pdf", "png"):
                output_path = output_dir / f"{stage}_cluster_profiles.{extension}"
                profile_fig.savefig(output_path, dpi=180, bbox_inches="tight")
                print(f"Saved: {output_path}")
            figures.append(profile_fig)
        plt.show()
        for figure in figures:
            plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--power", type=float, default=None,
        help="Display source/task cluster means as p**power / sum(p**power), with 0 < power < 1.",
    )
    args = parser.parse_args()
    if args.power is not None and not 0 < args.power < 1:
        parser.error("--power must satisfy 0 < power < 1")
    main(power=args.power)
