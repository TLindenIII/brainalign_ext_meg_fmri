from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
notebook_path = ROOT / "notebooks" / "02_retrieval_evaluation.ipynb"

with open(notebook_path, "r") as handle:
    notebook = json.load(handle)

for cell in notebook["cells"]:
    if cell["cell_type"] != "code":
        continue

    for index, line in enumerate(cell["source"]):
        if "trial['image_id'] for trial in test_dataset.trials" not in line:
            continue
        cell["source"][index] = line.replace(
            "trial['image_id'] for trial in test_dataset.trials",
            "Path(f).stem for f in test_dataset.files",
        )

with open(notebook_path, "w") as handle:
    json.dump(notebook, handle, indent=1)
