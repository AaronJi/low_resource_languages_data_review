import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def filter_same_series(filename, series_works):
    for series, filenames in series_works.items():
        if filename in filenames[:-1]:
            return series
    return None


def living_language_codes(language_info):
    # Only ISO 639-3 Type=L entries participate in the downstream analysis.
    # Missing, unknown, historical, constructed, special, etc. fail closed.
    return {
        code for code, info in language_info.items()
        if isinstance(info, dict)
        and str(info.get("ISO_639_3_Type", "")).strip().upper() == "L"
    }


def filter_languages(datasets, language_codes):
    # language_codes is the living-only whitelist selected from the registry.
    removed = []
    total_language_entries = 0
    excluded_language_entries = 0
    for name, dataset in list(datasets.items()):
        languages = dataset.get("Languages", [])
        if isinstance(languages, str):
            languages = [languages]
        elif not isinstance(languages, list):
            languages = []

        total_language_entries += len(languages)
        valid_languages = [
            language for language in languages
            if isinstance(language, str) and language in language_codes
        ]
        excluded_language_entries += len(languages) - len(valid_languages)

        if not valid_languages:
            del datasets[name]
            removed.append(name)
            continue

        dataset["Languages"] = valid_languages
    return removed, total_language_entries, excluded_language_entries


def filter_format(datasets):
    removed = []
    for name, dataset in list(datasets.items()):
        formats = dataset.get("Format", [])
        if isinstance(formats, str):
            formats = [formats]
        if formats and all(value == "Evaluation" for value in formats):
            del datasets[name]
            removed.append(name)
    return removed


def main(input_folder, output_folder=None):
    input_dir = (ROOT / input_folder).resolve()
    output_dir = input_dir if output_folder is None else (ROOT / output_folder).resolve()
    series_works = json.loads((ROOT / "series_works.json").read_text(encoding="utf-8"))
    language_codes = living_language_codes(
        json.loads((ROOT / "language_639-3_info.json").read_text(encoding="utf-8-sig"))
    )

    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    if output_dir != input_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    total_language_entries = 0
    excluded_language_entries = 0
    removed_language_subsets = 0

    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".json":
            continue

        series = filter_same_series(path.name, series_works)
        if series is not None:
            print(f"DROP {path.name} | series: {series}")
            if output_folder is None:
                path.unlink()
            continue

        datasets = json.loads(path.read_text(encoding="utf-8"))
        removed_languages, checked, excluded = filter_languages(datasets, language_codes)
        total_language_entries += checked
        excluded_language_entries += excluded
        removed_language_subsets += len(removed_languages)
        removed_formats = filter_format(datasets)

        if not datasets:
            reasons = []
            if removed_formats:
                reasons.append("format: " + ", ".join(removed_formats))
            if not removed_languages and not reasons:
                reasons.append("empty")
            if reasons:
                print(f"DROP {path.name} | " + "; ".join(reasons))
            if output_folder is None:
                path.unlink()
            continue

        destination = output_dir / path.name
        destination.write_text(
            json.dumps(datasets, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    ratio = (
        excluded_language_entries / total_language_entries
        if total_language_entries else 0.0
    )
    print(
        f"NON-L/INVALID LANGUAGE EXCLUSION SUMMARY: "
        f"{excluded_language_entries}/{total_language_entries} language entries "
        f"({ratio:.2%}); {removed_language_subsets} subsets removed because "
        f"no ISO-Type-L language entries remained"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter dataset metadata JSON files.")
    parser.add_argument("input_folder", help="Folder containing input JSON files")
    parser.add_argument(
        "output_folder",
        nargs="?",
        help="Optional output folder; omit to update input files in place",
    )
    args = parser.parse_args()
    main(args.input_folder, args.output_folder)
