import argparse
import json
from pathlib import Path

root = Path(__file__).resolve().parent
language_info = json.loads((root / "language_639-3_info.json").read_text(encoding="utf-8"))
language_name_to_code = {
    name: code
    for code, language in language_info.items()
    for name in language.get("English_Name", [])
}


def read_all_constants(constants_dir):
    return {
        "DOMAIN_GROUPS": json.loads((constants_dir / "domain_groups.json").read_text(encoding="utf-8")),
        "DOMAIN_TYPES": json.loads((constants_dir / "domain_types.json").read_text(encoding="utf-8")),
    }


def invert_dict_of_lists(values):
    return {item: group for group, items in values.items() for item in items}


source_type_to_category = {
    "Social Media & User-Generated Content": "Collect-Existed",
    "News websites": "Human-Created",
    "Multi-Media Entertainment Content": "Human-Created",
    "Synthetic": "Machine-Generated",
    "Other": "Other",
    "General Web Scraped": "Collect-Existed",
    "Books & Academic": "Expert-Created",
    "Government Documents": "Expert-Created",
    "Human / Crowdsourced": "Human-Created",
}


def map_source(source, source_to_group, source_group_to_type):
    mapped = source_to_group.get(source, source)
    mapped = source_group_to_type.get(mapped, mapped)
    return source_type_to_category.get(mapped, mapped)


# Exact, case-insensitive flag matching avoids treating model names such as
# "Nous-Hermes" as a negative flag merely because they contain "no".
NEGATIVE_GENERATION_FLAGS = {
    "no", "n", "false", "0", "0.0", "none", "null", "n/a", "na",
    "not applicable", "not generated", "not used", "否", "无", "没有",
}
YES_ANNOTATION_FLAGS = {
    "yes", "y", "true", "1", "1.0", "是", "有", "是的",
    "已标注", "已审核", "人工标注", "人工审核",
}
EXPERT_ANNOTATION_FLAGS = {
    "expert",
}


def metadata_flag_tokens(value):
    """Normalize scalar/list flags without changing the original metadata."""
    if value is None:
        return set()
    if not isinstance(value, (list, tuple, set, frozenset)):
        value = [value]
    return {
        str(item).strip().casefold()
        for item in value
        if item is not None and str(item).strip()
    }


def map_model_generated(dataset):
    """Override all source categories when generation metadata is present."""
    tokens = metadata_flag_tokens(dataset.get("Model Generated"))
    if tokens - NEGATIVE_GENERATION_FLAGS:
        # Preserve the canonical spelling used by grouping and clustering.
        dataset["Text Source Categories"] = ["Machine-Generated"]


def annotation_intent(annotation):
    """Return the canonical human-annotation intent: no, yes, or expert."""
    tokens = metadata_flag_tokens(annotation)
    # Expert is the more specific state if malformed legacy metadata contains
    # both Expert and Yes-like tokens.
    if tokens & EXPERT_ANNOTATION_FLAGS:
        return "expert"
    if tokens & YES_ANNOTATION_FLAGS:
        return "yes"
    return "no"


def map_human_annotation(dataset):
    annotation = dataset.get("Human Annotation")
    if annotation is None or (isinstance(annotation, str) and not annotation.strip()):
        annotation = dataset["Human Annotation"] = "No"

    intent = annotation_intent(annotation)
    mapped = []
    for category in dataset.get("Text Source Categories", []):
        # Apply after the Model Generated override:
        # 1) machine-generated content with either ordinary or expert human
        #    annotation is Human-Audited;
        # 2) otherwise, ordinary human annotation upgrades every category
        #    except Machine-Generated / Expert-Created to Human-Created;
        # 3) otherwise, expert annotation upgrades the category to
        #    Expert-Created.
        if category == "Machine-Generated" and intent in {"yes", "expert"}:
            category = "Human-Audited"
        elif (
            category not in {"Machine-Generated", "Expert-Created"}
            and intent == "yes"
        ):
            category = "Human-Created"
        elif intent == "expert":
            category = "Expert-Created"

        if category not in mapped:
            mapped.append(category)

    dataset["Text Source Categories"] = mapped


DEFAULT_LIST_CATEGORY = "Other"


def categorical_list_or_default(value, field_name, fallback=DEFAULT_LIST_CATEGORY):
    """Normalize categorical list metadata and preserve one fallback label."""
    if value is None:
        return [fallback]
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list, string, or null")
    labels = []
    for item in value:
        if item is None:
            continue
        if not isinstance(item, str):
            raise TypeError(f"{field_name} entries must be strings or null")
        item = item.strip()
        if item and item not in labels:
            labels.append(item)
    return labels or [fallback]


def map_task(dataset, task_to_macro_category):
    task_categories = categorical_list_or_default(
        dataset.get("Task Categories"), "Task Categories"
    )
    mapped = [
        task_to_macro_category.get(task, task)
        for task in task_categories
        if task != DEFAULT_LIST_CATEGORY
    ]
    dataset["Task Macro Categories"] = mapped or [DEFAULT_LIST_CATEGORY]


def map_format(dataset):
    dataset["Format"] = categorical_list_or_default(dataset.get("Format"), "Format")


def map_language(language, language_name_to_code):
    return language_name_to_code.get(language, language)


def main(input_folder_name, output_folder_name=None):

    input_dir = root / input_folder_name
    output_dir = input_dir if output_folder_name in (None, input_folder_name) else root / output_folder_name
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    constants = read_all_constants(root / "constants")
    source_to_group = invert_dict_of_lists(constants["DOMAIN_GROUPS"])
    source_group_to_type = invert_dict_of_lists(constants["DOMAIN_TYPES"])
    valid_categories = set(source_type_to_category.values())
    task_to_macro_category = json.loads(
        (root / "task_macro_category_mapping.json").read_text(encoding="utf-8")
    )["task_to_macro_category"]


    for json_path in sorted(input_dir.iterdir()):
        if not json_path.is_file() or json_path.suffix != ".json" or json_path.name == "_template.json":
            continue
        datasets = json.loads(json_path.read_text(encoding="utf-8"))
        for dataset_id, dataset in datasets.items():
            sources = dataset.get("Text Sources", [])
            if not isinstance(sources, list):
                print(f"ERROR {json_path.name} | {dataset_id} | Text Sources is not a list")
                exit(1)
                continue

            categories = [map_source(source, source_to_group, source_group_to_type) for source in sources]
            dataset["Text Source Categories"] = list(dict.fromkeys(categories))
            for source, category in zip(sources, categories):
                if category not in valid_categories:
                    print(f"ERROR {json_path.name} | {dataset_id} | {source!r} -> {category!r}")
                    exit(1)

            map_model_generated(dataset)
            map_human_annotation(dataset)
            map_task(dataset, task_to_macro_category)
            map_format(dataset)

            languages = dataset.get("Languages", [])
            if isinstance(languages, list):
                dataset["Languages"] = [map_language(language, language_name_to_code) for language in languages]

        (output_dir / json_path.name).write_text(
            json.dumps(datasets, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder_name", help="Input metadata folder name under this script's directory")
    parser.add_argument("output_folder_name", nargs="?", help="Optional output metadata folder name")
    args = parser.parse_args()
    main(args.input_folder_name, args.output_folder_name)
