#!/usr/bin/env python3
"""Join cluster-level summaries/centers onto language-level results.

Summary non-key columns (now including centers) are prefixed with "cluster-".
The summary's overall row is not a cluster and is excluded from language joins.
Legacy separate-center inputs are supported, but integrated summaries take priority.
By default, write both stage results and a full outer language-code join in
the project root, with "pre-"/"post-" prefixes on all non-language columns.
Uses only the Python standard library. Original CSV files are never modified.
"""

import argparse
import csv
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from tempfile import NamedTemporaryFile


def read_csv(path):
    """Read values as strings to preserve precision, blanks, and language IDs."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        columns = reader.fieldnames
        if not columns or len(columns) != len(set(columns)):
            raise ValueError(f"{path}: missing or duplicate column names")
        keys = [c for c in columns if c.strip().casefold() == "cluster"]
        if len(keys) != 1:
            raise ValueError(f"{path}: expected exactly one Cluster column")
        rows = list(reader)
        for number, row in enumerate(rows, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}: incorrect column count at record {number}")
    return columns, keys[0], rows


def cluster_id(value, path):
    """Normalize integer keys, including CSV spellings such as '1.0'."""
    try:
        number = Decimal(value.strip())
    except (InvalidOperation, AttributeError):
        raise ValueError(f"{path}: invalid Cluster value {value!r}") from None
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError(f"{path}: Cluster must be an integer, got {value!r}")
    return int(number)


def index_clusters(rows, key, path, *, skip_overall=False):
    """Enforce many-to-one keys; optionally omit the descriptive overall row."""
    indexed = {}
    overall_seen = False
    for row in rows:
        if skip_overall and row[key].strip().casefold() in {"整体", "overall"}:
            if overall_seen:
                raise ValueError(f"{path}: duplicate overall summary row")
            overall_seen = True
            continue
        cluster = cluster_id(row[key], path)
        if cluster in indexed:
            raise ValueError(f"{path}: duplicate Cluster key {cluster}")
        indexed[cluster] = row
    return indexed


def prepare_join(folder):
    result_path = folder / "cluster_results.csv"
    summary_path = folder / "cluster_summary.csv"
    result_cols, result_key, results = read_csv(result_path)
    summary_cols, summary_key, summaries = read_csv(summary_path)
    summaries = index_clusters(summaries, summary_key, summary_path, skip_overall=True)

    center_fields = [c.removesuffix(" log10") for c in result_cols if c.endswith(" log10")]
    center_fields += [c for c in result_cols if c.endswith("/proportion")]
    embedded_centers = bool(center_fields) and all(c in summary_cols for c in center_fields)
    summary_extra = [c for c in summary_cols if c != summary_key]
    if embedded_centers:
        # Ignore stale legacy center files, even if still present in the folder.
        center_path = summary_path
        center_extra = []
        centers = {
            cluster: {c: row[c] for c in center_fields}
            for cluster, row in summaries.items()
            if cluster != -1 or any(row[c] != "" for c in center_fields)
        }
    else:
        if any(c in summary_cols for c in center_fields):
            raise ValueError(f"{summary_path}: incomplete integrated center columns")
        center_path = folder / "cluster_centers.csv"
        if not center_path.is_file():
            center_path = folder / "cluster_center.csv"
        center_cols, center_key, center_rows = read_csv(center_path)
        centers = index_clusters(center_rows, center_key, center_path)
        center_extra = [c for c in center_cols if c != center_key]

    # Prefix every non-key cluster field once, preserving the old join naming.
    output_cols = result_cols + [
        f"cluster-{c}" for c in summary_extra + center_extra
    ]
    conflicts = [c for c, count in Counter(output_cols).items() if count > 1]
    if conflicts:
        raise ValueError(f"{folder}: output column name collisions: {conflicts}")

    joined = []
    blank_center_rows = 0
    for row in results:
        cluster = cluster_id(row[result_key], result_path)
        if cluster not in summaries:
            raise ValueError(f"{summary_path}: no summary for Cluster {cluster}")
        if cluster not in centers and cluster != -1:
            raise ValueError(f"{center_path}: no center for Cluster {cluster}")
        summary = summaries[cluster]
        center = centers.get(cluster, {})
        blank_center_rows += cluster not in centers

        # Keep every input row in its original order, including Cluster -1.
        merged = dict(row)
        merged.update({f"cluster-{c}": summary[c] for c in summary_extra})
        merged.update({f"cluster-{c}": center.get(c, "") for c in center_extra})
        joined.append(merged)

    return folder / "language_cluster_result.csv", output_cols, joined, blank_center_rows


def index_languages(columns, rows, path):
    """Validate a unique, nonempty language code for a one-to-one stage join.

    Match codes case-insensitively after trimming surrounding whitespace;
    retain original cell values in the output (pre-training spelling wins).
    """
    keys = [c for c in columns if c.strip().casefold() == "language"]
    if len(keys) != 1:
        raise ValueError(f"{path}: expected exactly one Language column")
    key = keys[0]
    indexed = {}
    for number, row in enumerate(rows, start=2):
        value = row.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path}: empty Language at record {number}")
        code = value.strip().casefold()
        if code in indexed:
            raise ValueError(f"{path}: duplicate Language code {value!r}")
        indexed[code] = row
    return key, indexed


def prepare_cross_stage_join(root, pretrain_plan, posttrain_plan):
    """Full outer join the freshly prepared stage tables, not stale CSV files.

    Keep pre-training row order, then append post-training-only languages
    in their original order. Missing stage values remain blank, never -1.
    """
    pre_path, pre_cols, pre_rows, _ = pretrain_plan
    post_path, post_cols, post_rows, _ = posttrain_plan
    pre_key, pre_index = index_languages(pre_cols, pre_rows, pre_path)
    post_key, post_index = index_languages(post_cols, post_rows, post_path)
    pre_extra = [c for c in pre_cols if c != pre_key]
    post_extra = [c for c in post_cols if c != post_key]
    output_cols = (
        ["Language"]
        + [f"pre-{c}" for c in pre_extra]
        + [f"post-{c}" for c in post_extra]
    )
    conflicts = [c for c, count in Counter(output_cols).items() if count > 1]
    if conflicts:
        raise ValueError(f"{root}: cross-stage column name collisions: {conflicts}")

    # The dictionary insertion order mirrors each stage's language row order.
    ordered_codes = list(pre_index) + [c for c in post_index if c not in pre_index]
    joined = []
    for code in ordered_codes:
        pre = pre_index.get(code, {})
        post = post_index.get(code, {})
        language = pre[pre_key] if code in pre_index else post[post_key]
        row = {"Language": language}
        row.update({f"pre-{c}": pre.get(c, "") for c in pre_extra})
        row.update({f"post-{c}": post.get(c, "") for c in post_extra})
        joined.append(row)

    coverage = {
        "matched_languages": len(pre_index.keys() & post_index.keys()),
        "pretrain_only_languages": len(pre_index.keys() - post_index.keys()),
        "posttrain_only_languages": len(post_index.keys() - pre_index.keys()),
    }
    return root / "language_cluster_result.csv", output_cols, joined, coverage


def write_csv_atomic(path, columns, rows):
    """Finish a temporary file before replacing the output; write Excel-friendly UTF-8."""
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8-sig", newline="", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent,
        help="Project root; default: directory containing this script",
    )
    parser.add_argument(
        "--data-type", choices=("all", "pretrain", "posttrain"), default="all",
        help="Select outputs; all (default): both stages plus the project-root join",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Validate and join in memory without writing output CSV files",
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    stages = ("pretrain", "posttrain") if args.data_type == "all" else (args.data_type,)

    # Validate both stage joins and language keys before writing any output.
    try:
        plans = [prepare_join(root / f"meta_data_{stage}_clustered") for stage in stages]
        combined = (
            prepare_cross_stage_join(root, plans[0], plans[1])
            if args.data_type == "all" else None
        )
        for path, columns, rows, blanks in plans:
            if not args.check_only:
                write_csv_atomic(path, columns, rows)
            action = "CHECKED" if args.check_only else "SAVED"
            print(f"{action} {path} | rows={len(rows)}, columns={len(columns)}, "
                  f"rows_without_center={blanks}")
        if combined is not None:
            path, columns, rows, coverage = combined
            if not args.check_only:
                write_csv_atomic(path, columns, rows)
            action = "CHECKED" if args.check_only else "SAVED"
            details = ", ".join(f"{key}={value}" for key, value in coverage.items())
            print(f"{action} {path} | rows={len(rows)}, columns={len(columns)}, "
                  f"{details}")
    except (OSError, ValueError, csv.Error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
