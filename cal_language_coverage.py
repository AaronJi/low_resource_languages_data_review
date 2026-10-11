from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

#root = Path(".")

project_path = r"D:\Projects\low-resource-language-review"
root_path = Path(project_path)

level_order = {"S": 0, "A": 1, "B": 2, "C": 3, "D": 4, "uncovered": 5} # , "E": 5

datasets_info = pd.read_excel(root_path / "low_resource_datasets_info.xlsx", usecols=["data", "format", "source_level"],).dropna(subset=["data", "format", "source_level"])
languages = pd.read_excel(root_path / "languages_info.xlsx", usecols=["ISO_639_3"],)["ISO_639_3"].dropna().astype(str).tolist()
#print(datasets_info)

languages101 = pd.read_csv(root_path / "languages101.csv", encoding="utf-8-sig",)


def build_matrix(datasets):
    matrix = pd.DataFrame(0, index=languages, columns=datasets, dtype="uint8")
    matrix.index.name = "ISO_639_3"

    for dataset in datasets:
        codes = pd.read_csv(root_path / "dataset_coverage" / f"{dataset}.csv", header=None).iloc[:, 0].astype(str)
        matrix.loc[matrix.index.intersection(codes), dataset] = 1
    return matrix

def cal_lang_coverage(datasets_info, dataset_format=None):
    datasets = datasets_info["data"] if dataset_format is None else datasets_info.loc[datasets_info["format"] == dataset_format, "data"]
    matrix = build_matrix(datasets.dropna().astype(str).tolist())

    language_coverage = matrix.apply(lambda row: row.index[row.eq(1)].tolist(), axis=1).to_frame(name="covered_datasets")
    language_coverage["covered_dataset_count"] = matrix.sum(axis=1)
    language_coverage.index.name = "ISO_639_3"

    dataset_levels = datasets_info.set_index("data")["source_level"]
    language_coverage["covered_dataset_level"] = language_coverage["covered_datasets"].apply(lambda names: dataset_levels.loc[names].tolist())

    language_coverage["best_covered_level"] = language_coverage["covered_dataset_level"].apply(lambda levels: min(levels, key=level_order.get) if levels else None)

    return language_coverage

def filter_with_101_languages(language_coverage):
    language_coverage_101 = language_coverage.reindex(languages101["ISO_639_3"]).copy()
    language_coverage_101.insert(0, "chinese_name", languages101["chinese_name"].to_numpy(),)
    return language_coverage_101


def from_language_to_level_show(language_coverage):
    languages_best_level = (language_coverage.groupby("best_covered_level")["covered_datasets"].apply(lambda series: series.index.tolist()).reindex(level_order.keys(), fill_value=[]).to_frame(name="languages"))
    languages_best_level["languages_in_Chinese"] = (language_coverage.groupby("best_covered_level")["chinese_name"].apply(list).reindex(level_order.keys(), fill_value=[]))
    languages_best_level["language_count"] = languages_best_level["languages"].apply(len)
    languages_best_level.index.name = "best_covered_level"
    return languages_best_level

def search_datasets(source_level, dataset_format=None):
    if dataset_format is None:
        datasets = datasets_info.loc[datasets_info["source_level"] == source_level, "data",].dropna().astype(str).tolist()
    else:
        datasets = datasets_info.loc[(datasets_info["source_level"] == source_level) & (datasets_info["format"] == dataset_format), "data",].dropna().astype(str).tolist()
    return datasets

level_datasets = {}
for format in ['text', 'QA']:
    level_datasets[format] = {}
    for level in level_order:
        level_datasets[format][level] = search_datasets(level, dataset_format=format)


#language_coverage_all = cal_lang_coverage(datasets_info)
language_coverage_text = cal_lang_coverage(datasets_info, dataset_format="text")
language_coverage_QA = cal_lang_coverage(datasets_info, dataset_format="QA")

#language_coverage_all_101 = filter_with_101_languages(language_coverage_all)
language_coverage_text_101 = filter_with_101_languages(language_coverage_text)
language_coverage_QA_101 = filter_with_101_languages(language_coverage_QA)

#check_lang = "eng"
#check_lang = "zho"
check_lang = "spa"

#print(language_coverage_all_101.loc[check_lang])
#print(language_coverage_text_101.loc[check_lang])
#print(language_coverage_QA_101.loc[check_lang])

languages_text_best_level_101 = from_language_to_level_show(language_coverage_text_101)
languages_QA_best_level_101 = from_language_to_level_show(language_coverage_QA_101)

def print_level_language_dataset(languages_best_level, format, level, ):
    print("### %s %s" % (
        format,
        level
    ))

    print("%d languages:" % (
        languages_best_level.loc[level, "language_count"],
    ))
    print(languages_best_level.loc[level]["languages_in_Chinese"])

    print("from %d datasets:" % len(level_datasets[format][level]))
    print(level_datasets[format][level])
    return

print('======================================')

print_level_language_dataset(languages_text_best_level_101, format="text", level="B")
print('--------------------------------------')
print_level_language_dataset(languages_text_best_level_101, format="text", level="C")

print('======================================')
print_level_language_dataset(languages_QA_best_level_101, format="QA", level="S")
print('--------------------------------------')
print_level_language_dataset(languages_QA_best_level_101, format="QA", level="A")
print('--------------------------------------')
print_level_language_dataset(languages_QA_best_level_101, format="QA", level="D")
print('--------------------------------------')
uncovered_languages_QA_101 = language_coverage_QA_101.loc[language_coverage_QA_101["covered_dataset_count"] == 0, "chinese_name",]
print("### %s uncovered" % "QA")
print("%d languages" % len(uncovered_languages_QA_101))
print(uncovered_languages_QA_101.tolist())
print('======================================')

def find_lang_QA_worse_than_text(language_coverage_text, language_coverage_QA):
    if not "chinese_name" in language_coverage_text.columns:
        language_coverage_text["chinese_name"] = ''
    level_comparison = pd.DataFrame({
        "code": language_coverage_text.index,
        "chinese_name": language_coverage_text["chinese_name"],
        "text_level": language_coverage_text["best_covered_level"],
        "QA_level": language_coverage_QA["best_covered_level"],
    }).fillna("uncovered")
    level_comparison["level_compare"] = level_comparison["text_level"] + '->' + level_comparison["QA_level"]

    QA_lower_than_text = level_comparison.loc[level_comparison["QA_level"].map(level_order) > level_comparison["text_level"].map(level_order)]
    return level_comparison, QA_lower_than_text


text_level_counts = language_coverage_text['best_covered_level'].value_counts()
print(text_level_counts)
QA_level_counts = language_coverage_QA['best_covered_level'].value_counts()
print(QA_level_counts)


level_comparison, QA_lower_than_text = find_lang_QA_worse_than_text(language_coverage_text, language_coverage_QA)
print(QA_lower_than_text)
print("count:", len(QA_lower_than_text))
QA_worse_level_compare_counts = QA_lower_than_text['level_compare'].value_counts()
print(QA_worse_level_compare_counts)

level_comparison_101, QA_lower_than_text_101 = find_lang_QA_worse_than_text(language_coverage_text_101, language_coverage_QA_101)
print(QA_lower_than_text_101)
print("count:", len(QA_lower_than_text_101))
QA_worse_level_compare_counts_101 = QA_lower_than_text_101['level_compare'].value_counts()
print(QA_worse_level_compare_counts_101)

level_compare_counts_101 = level_comparison_101['level_compare'].value_counts()
print(level_compare_counts_101)

for level_compare_101, names_101 in (
    level_comparison_101.groupby("level_compare")["chinese_name"]
    .apply(list)
    .items()
):
    print(level_compare_101, names_101)

plt.figure(1)
plt.subplot(221)
text_level_counts.plot(kind='bar', color='skyblue')
plt.title('Frequency Histogram of text level')
plt.xlabel('Level')
plt.ylabel('Frequency')
#plt.show()

plt.subplot(222)
QA_level_counts.plot(kind='bar', color='skyblue')
plt.title('Frequency Histogram of QA level')
plt.xlabel('Level')
plt.ylabel('Frequency')
#plt.show()

plt.subplot(223)
QA_worse_level_compare_counts.plot(kind='bar', color='skyblue')
plt.title('Frequency Histogram of text level to QA level')
plt.xlabel('Level')
plt.ylabel('Frequency')

plt.subplot(224)
QA_worse_level_compare_counts_101.plot(kind='bar', color='skyblue')
plt.title('Frequency Histogram of text level to QA level (101)')
plt.xlabel('Level')
plt.ylabel('Frequency')

#plt.show()