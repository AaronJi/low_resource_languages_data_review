from pathlib import Path
import pandas as pd

root = Path(r"D:\Projects\low-resource-language-review")
info = pd.read_excel(root / "low_resource_datasets_info.xlsx", usecols=["data", "format", "source_level"])
candidates = info.loc[(info["format"] == "QA") & (info["source_level"] == "A"), "data"].dropna()
covered = []
for dataset in candidates:
    codes = pd.read_csv(root / "dataset_coverage" / f"{dataset}.csv", header=None).iloc[:, 0].astype(str)
    if "mri" in set(codes):
        covered.append(dataset)
print("QA + A datasets:", candidates.tolist())
print("mri covered by:", covered)
