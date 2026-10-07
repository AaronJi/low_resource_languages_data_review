import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent

# Keep the existing canonical categories: Human-Created represents the
# requested human-written fallback. Do not introduce a sixth source label.
DEFAULT_SOURCE_CATEGORIES = {
    "Num Unicode Characters": "Collect-Existed",
    "Num Dialogs": "Human-Created",
}


def default_source_category(quantity_key):
    key = quantity_key.removesuffix(" log10")
    if key not in DEFAULT_SOURCE_CATEGORIES:
        raise ValueError(f"No default source category for quantity {quantity_key!r}")
    return DEFAULT_SOURCE_CATEGORIES[key]


DEFAULT_LIST_CATEGORY = "Other"


def categories_or_default(categories, fallback, field_name="categorical field"):
    # Missing/null/empty lists and blank placeholders use one explicit fallback.
    # Existing labels (including Other) are preserved; quantities are later
    # split equally across multiple labels by reduce_attributes.item_statistics.
    if categories is None:
        return [fallback]
    if isinstance(categories, str):
        categories = [categories]
    if not isinstance(categories, list):
        raise TypeError(f"{field_name} must be a list, string, or empty")
    labels = []
    for category in categories:
        if category is None:
            continue
        if not isinstance(category, str):
            raise TypeError(f"{field_name} entries must be strings or null")
        category = category.strip()
        if category and category not in labels:
            labels.append(category)
    return labels or [fallback]


def source_categories_or_default(categories, fallback):
    # Backward-compatible name used by reduce_attributes.
    return categories_or_default(
        categories, fallback, field_name="Text Source Categories"
    )


def list_field_fallback(path, source_fallback):
    if path == ("Text Source Categories",):
        return source_fallback
    if path in {("Task Macro Categories",), ("Format",)}:
        return DEFAULT_LIST_CATEGORY
    return None


def field_paths(spec, prefix=()):
    # Expand nested field specifications into complete key paths.
    if isinstance(spec, str):
        yield prefix + (spec,)
    elif isinstance(spec, dict):
        for key, children in spec.items():
            yield from field_paths(children, prefix + (key,))
    else:
        for item in spec:
            yield from field_paths(item, prefix)


def read_value(data, path):
    # Follow the full path; a missing leaf or ancestor is handled as null by the caller.
    for key in path:
        if not isinstance(data, dict) or key not in data:
            raise KeyError(" / ".join(path))
        data = data[key]
    return data


def append_value(group, path, value):
    # Preserve the key hierarchy and append one entry per record.
    # List values remain nested, and nulls retain their positions in the output lists.
    for key in path[:-1]:
        group = group.setdefault(key, {})
    group.setdefault(path[-1], []).append(value)


def read_datasets(input_dir):
    # Yield records across all collections without overwriting repeated IDs across files.
    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".json" or path.name == "_template.json":
            continue
        datasets = json.loads(path.read_text(encoding="utf-8"))
        yield from datasets.items()


def group_datasets(datasets, config):
    # Assign an aggregation method to each leaf field, including the record identifier.
    aggregation = config["aggredated"]
    quantity_key = next(field_paths(aggregation["quanity"]))[-1]
    fallback = default_source_category(quantity_key)
    fields = [("unique_string", path) for path in field_paths(aggregation["unique id"])]
    for method in ("unique_string", "list_string", "quanity"):
        fields.extend((method, path) for path in field_paths(aggregation.get(method, [])))

    # Share one group accumulator across the complete sequence of input records.
    grouped = {}
    null_stats = {path: {"total": 0, "null": 0} for _, path in fields}
    skipped = 0
    records = datasets.items() if isinstance(datasets, dict) else datasets
    for dataset_id, dataset in records:
        group_ids = dataset.get(config["group id list"], [])
        if not isinstance(group_ids, list):
            raise TypeError(f"{dataset_id}: {config['group id list']} must be a list")
        if not group_ids:
            skipped += 1

        for method, path in fields:
            # Missing paths and explicit null values both produce a null contribution.
            try:
                value = read_value(dataset, path)
            except KeyError:
                value = None

            # Count each source record once per field, even when its group list is empty.
            # Multi-language records do not receive extra weight in these statistics.
            null_stats[path]["total"] += 1
            null_stats[path]["null"] += value is None
            if not group_ids:
                continue

            # Allocate missing source/task/format categories before language
            # aggregation so their quantities remain in each categorical
            # denominator. Multi-label records keep all labels and are split
            # equally during Reduce.
            list_fallback = list_field_fallback(path, fallback)
            if list_fallback is not None:
                value = categories_or_default(
                    value, list_fallback, field_name=path[-1]
                )

            # Validate observed values; nulls bypass validation and numerical allocation.
            if value is not None:
                expected_type = {
                    "unique_string": str,
                    "list_string": list,
                    "quanity": (int, float),
                }[method]
                if not isinstance(value, expected_type) or isinstance(value, bool):
                    raise TypeError(f"{dataset_id}: invalid {method} value at {' / '.join(path)}")
                if method == "quanity":
                    # Divide quantities equally among the original list of group IDs.
                    value = value / len(group_ids)

            # Append contributions in record order so all fields remain aligned.
            for group_id in group_ids:
                group = grouped.setdefault(group_id, {})
                append_value(group, path, value)

    # Report null counts and proportions once, after all collections have been grouped.
    for path, counts in null_stats.items():
        total, null = counts["total"], counts["null"]
        ratio = f"{null / total:.2%}" if total else "N/A"
        print(f"{' / '.join(path)}: total={total}, null={null}, null_ratio={ratio}")
    if skipped:
        print(f"SKIP: {skipped} subsets with empty {config['group id list']}")
    return grouped


def main(input_folder, output_folder, scenario):
    # Resolve input and output locations relative to this script.
    input_dir = (ROOT / input_folder).resolve()
    output_dir = (ROOT / output_folder).resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    if input_dir == output_dir:
        raise ValueError("Input and output folders must be different")

    # Apply the selected scenario's rules to the combined stream of collection records.
    config = json.loads((ROOT / "grouped_values.json").read_text(encoding="utf-8"))[scenario]
    grouped = group_datasets(read_datasets(input_dir), config)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save each complete group to its own file, e.g. eng.json or mri.json.
    for group_id, values in grouped.items():
        (output_dir / f"{group_id}.json").write_text(
            json.dumps(values, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(f"Saved {len(grouped)} group files to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    parser.add_argument("output_folder")
    parser.add_argument("--data_type", required=True, choices=("pretrain", "posttrain"))
    args = parser.parse_args()
    main(args.input_folder, args.output_folder, args.data_type)
