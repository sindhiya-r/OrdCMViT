import pandas as pd
from pathlib import Path

data_root = Path("./BCMID")
folders = set([p.name.strip() for p in data_root.iterdir() if p.is_dir()])

df = pd.read_csv("./BCMID/BCMID_labels.csv", header=None)
df.columns = ["patient_id", "birads", "binary_label"]

csv_ids = set(df["patient_id"].astype(str).str.strip())

print("Folders not in CSV:", len(folders - csv_ids))
print("CSV not in Folders:", len(csv_ids - folders))

from pathlib import Path

data_root = Path("./BCMID")

complete = 0
wrong_us = 0
wrong_mm = 0

for p in data_root.iterdir():
    if not p.is_dir():
        continue

    subdirs = [d.name for d in p.iterdir() if d.is_dir()]

    has_us = any("ultra" in s.lower() or "us" in s.lower() for s in subdirs)
    has_mm = any("mammo" in s.lower() or "mg" in s.lower() for s in subdirs)

    if has_us and has_mm:
        complete += 1
    else:
        if not has_us:
            wrong_us += 1
        if not has_mm:
            wrong_mm += 1

print("Flexible complete:", complete)
print("Missing US:", wrong_us)
print("Missing MM:", wrong_mm)