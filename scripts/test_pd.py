from pathlib import Path
import glob

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

print(f"Pandas version: {pd.__version__}")
pattern = str(ROOT / "data" / "things-meg-ds004212" / "**" / "*events.tsv")
files = glob.glob(pattern, recursive=True)
for path in files:
    try:
        pd.read_csv(path, sep="\t", encoding="latin-1", engine="c", on_bad_lines="skip")
    except Exception as exc:
        print(f"Error on {path}: {exc}")
        break
