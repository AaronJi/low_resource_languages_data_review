# low_resource_languages_data_review

Review dataset metadata and analyze language-level resource coverage, quantities, and source composition for pre-training and post-training data. The workflow follows `pipeline.sh`.

## Data and configuration

| Path | Purpose |
| --- | --- |
| `data_summaries_pretrain/` | Pre-training dataset metadata |
| `data_summaries/`, `data_summaries_new/` | Post-training dataset metadata, mapped into the same output directory |
| `constants/`, `task_macro_category_mapping.json` | Source and task category mappings |
| `language_639-3_info.json` | Language registry; analysis retains living languages (`ISO_639_3_Type = L`) |
| `series_works.json` | Ordered dataset series; filtering retains the last listed version |
| `grouped_values.json` | Fields and quantity measures aggregated per language |
| `clustering.yaml` | Features, quantity/source weight (`alpha`), cluster count, and KMeans settings for each stage |

## Workflow

| Step | Script | Result |
| --- | --- | --- |
| Map | `map_datasets.py` | Normalize language names, source categories, task categories, and formats |
| Filter | `filter_datasets.py` | Remove earlier series versions, invalid/non-living language entries, evaluation-only subsets, and subsets with only excluded multimodal tasks |
| Group | `group_datasets.py` | Collect dataset contributions by language, dividing quantities equally across each subset's retained languages |
| Reduce | `reduce_attributes.py` | Aggregate quantities, available confidence intervals, coverage counts, and quantity-weighted category proportions |
| Cluster | `cluster_analysis.py` | Fit KMeans using standardized log10 quantity and source proportions transformed for Hellinger distance; produce summaries, metrics, and plots |
| Join | `join_language_clusters.py` | Attach cluster summaries and centers to language-level results; optionally add linguistic and population data |

Pre-training uses **Num Unicode Characters**; post-training uses **Num Dialogs**. Multi-label category contributions are split equally. Missing source categories default to `Collect-Existed` for pre-training and `Human-Created` for post-training.

Higher `alpha` gives quantity more weight relative to source composition. Task and format profiles describe clusters but are not clustering features. Languages without usable positive quantities, including uncovered registry languages, receive `Cluster = -1` and are excluded from fitting.

## Run the core workflow

Use Python 3.10+ and run commands from the repository root:

```bash
cd /path/to/project
python -m pip install numpy pandas matplotlib scikit-learn pyyaml threadpoolctl openpyxl
```

The following commands run the core stages with the included metadata and configuration:

```bash
python map_datasets.py data_summaries_pretrain meta_data_pretrain
python map_datasets.py data_summaries meta_data_posttrain
python map_datasets.py data_summaries_new meta_data_posttrain

for stage in pretrain posttrain; do
    python filter_datasets.py "meta_data_${stage}" "meta_data_${stage}_filtered"
    python group_datasets.py "meta_data_${stage}_filtered" "meta_data_${stage}_grouped" --data_type "$stage"
    python reduce_attributes.py "meta_data_${stage}_grouped" "meta_data_${stage}_reduced" --data_type "$stage"
    python cluster_analysis.py "meta_data_${stage}_reduced" "meta_data_${stage}_clustered" --data_type "$stage"
done

python -B join_language_clusters.py --skip-linguistic-joins
```

Use clean output directories for a full rebuild: individual stages do not remove stale JSON files. `pipeline.sh` starts by deleting its generated map/filter/group/reduce/cluster directories, then runs the stages and the additional join/plot steps described below.

Scripts resolve project paths from their own location. Grid search and joining also accept `--root` to select another project directory.

## Outputs

For each stage (`pretrain` or `posttrain`):

- `meta_data_<stage>/`, `meta_data_<stage>_filtered/`: mapped and filtered metadata.
- `meta_data_<stage>_grouped/`, `meta_data_<stage>_reduced/`: per-language JSON files.
- `meta_data_<stage>_clustered/`: stage-prefixed `cluster_results.csv`, `clustering_matrix.csv`, `cluster_summary.csv` (including centers), `PCA_summary.csv`, `cluster_metrics.json`, `cluster_profiles.png`, and `cluster_pca.png`.
- `meta_data_<stage>_joined/language_cluster_result.csv`: language results with cluster-level fields prefixed by `cluster-`.

## Optional analysis

After Reduce, compare quantity/source weights and cluster counts across seeds:

```bash
python -B cluster_grid_search.py --alphas 0.05 0.1 0.15 0.2 --n-clusters 2 3 4 5 --seeds 42 43 44
```

This writes reports under `cluster_grid_search_results/` and reports Silhouette, seed stability (ARI), and explained-variance scores. It does not update `clustering.yaml`; set the chosen parameters there before running Cluster again.

The default join command in `pipeline.sh` also requires these external assets, which are not included in this repository:

- `linguistic_analysis/language_cluster_glottolog_wals.xlsx`
- `linguistic_analysis/LinguaMeta_UN_M49_source_tables.xlsx`
- `linguistic_analysis/linguameta/raw/data/` (country speaker JSON files)

With these assets, `python -B join_language_clusters.py` also writes per-stage linguistic, population, regional, and audit outputs. Add `--check-only` to validate joins without writing files, or use `--skip-linguistic-joins` as above for basic cluster joins.

The plotting and website steps in `pipeline.sh` reference `plot_results.py`, `export_web_data.py`, and an external website project; these are not included here. Cluster already generates its own profile and PCA plots.

`cal_language_coverage.py` is a separate coverage-analysis utility, outside the pipeline. It requires `low_resource_datasets_info.xlsx`, `languages_info.xlsx`, `languages101.csv`, and per-dataset CSVs under `dataset_coverage/`, which are also not included.
