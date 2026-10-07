import argparse
import json
from math import log10
from pathlib import Path

from group_datasets import (
    DEFAULT_LIST_CATEGORY, categories_or_default, default_source_category,
    field_paths, read_value,
)
from math_utils import aggregate_size_ci


ROOT = Path(__file__).resolve().parent


def aligned_values(group, path, size):
    # Missing fields retain one null placeholder per source record.
    try:
        values = read_value(group, path)
    except KeyError:
        values = None
    if values is None:
        return [None] * size
    if not isinstance(values, list) or len(values) != size:
        raise ValueError(f"{' / '.join(path)} must contain {size} aligned entries")
    return values


def item_statistics(labels, quantities):
    # Split each record's quantity across its labels before merging repeated labels.
    counts = {}
    for items, quantity in zip(labels, quantities, strict=True):
        if not items:
            continue
        contribution = None if quantity is None else quantity / len(items)
        for item in items:
            if item is None:
                continue
            previous = counts.setdefault(item, None)
            # Ignore unknown contributions; retain null only if none are observed.
            if contribution is not None:
                counts[item] = (0 if previous is None else previous) + contribution

    total = sum(count for count in counts.values() if count is not None)
    return {
        item: {
            "count": count,
            "proportion": count / total if count is not None and total > 0 else None,
        }
        for item, count in counts.items()
    }


def reduce_attributes(group, config):
    aggregation = config["aggredated"]
    id_paths = list(field_paths(aggregation["unique id"]))
    identifiers = read_value(group, id_paths[0])
    if not isinstance(identifiers, list):
        raise ValueError("Unique Dataset Identifier must be a list")
    size = len(identifiers)
    collections = aligned_values(group, ("Collection",), size)
    # Preserve every original field, then add the reduced quantities and statistics.
    result = {
        **group,
        "num_dataset_sources": size,
        "num_collection_sources": len({item for item in collections if item is not None}),
    }

    quantity_paths = list(field_paths(aggregation["quanity"]))
    main_quantity = quantity_paths[0]
    main_values = aligned_values(group, main_quantity, size)
    fallback = default_source_category(main_quantity[-1])
    # Repair missing categorical lists in legacy grouped inputs. Use fresh
    # lists: do not mutate the grouped input or replace observed labels.
    categorical_fallbacks = {
        "Text Source Categories": fallback,
        "Task Macro Categories": DEFAULT_LIST_CATEGORY,
        "Format": DEFAULT_LIST_CATEGORY,
    }
    for field, field_fallback in categorical_fallbacks.items():
        labels = aligned_values(group, (field,), size)
        result[field] = [
            categories_or_default(
                value, field_fallback, field_name=field
            )
            for value in labels
        ]

    for path in quantity_paths:
        key = path[-1]
        if "95%" in key:
            continue
        values = aligned_values(group, path, size)
        # Distinguish an observed zero from an entirely unknown quantity.
        known_values = [value for value in values if value is not None]
        total = sum(known_values) if known_values else None
        result[key] = total
        result[f"{key} log10"] = log10(total) if total is not None and total > 0 else None

        lower_path = path[:-1] + (f"{key} 95% LB",)
        upper_path = path[:-1] + (f"{key} 95% UB",)
        if lower_path in quantity_paths and upper_path in quantity_paths:
            lower_values = aligned_values(group, lower_path, size)
            upper_values = aligned_values(group, upper_path, size)
            sizes, lowers, uppers = [], [], []
            for value, lower, upper in zip(values, lower_values, upper_values, strict=True):
                if value is None or lower is None or upper is None:
                    continue
                sizes.append(value)
                lowers.append(lower)
                uppers.append(upper)
            # Only complete, aligned triples contribute to the confidence interval.
            if sizes:
                _, lower, upper = aggregate_size_ci(sizes, lowers, uppers)
            else:
                lower, upper = None, None
            result[lower_path[-1]] = lower
            result[upper_path[-1]] = upper

    # All statistics use main_quantity as their weight, including scalar identifiers.
    for method in ("unique id", "unique_string", "list_string"):
        for path in field_paths(aggregation.get(method, [])):
            labels = aligned_values(result, path, size)
            if method != "list_string":
                labels = [[item] if item is not None else None for item in labels]
            result[f"{path[-1]} Statistics"] = item_statistics(labels, main_values)
    return result


def main(input_folder, output_folder, data_type):
    input_dir = (ROOT / input_folder).resolve()
    output_dir = (ROOT / output_folder).resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    if input_dir == output_dir:
        raise ValueError("Input and output folders must be different")

    config = json.loads((ROOT / "grouped_values.json").read_text(encoding="utf-8"))[data_type]
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".json" or path.name == "_template.json":
            continue
        group = json.loads(path.read_text(encoding="utf-8"))
        result = reduce_attributes(group, config)
        # Reduced quantities and statistics are stored directly at the root level.
        (output_dir / path.name).write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        saved += 1
    print(f"Saved {saved} reduced files to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    parser.add_argument("output_folder")
    parser.add_argument("--data_type", required=True, choices=("pretrain", "posttrain"))
    args = parser.parse_args()
    main(args.input_folder, args.output_folder, args.data_type)
