import json

with open('notebooks/02_retrieval_evaluation.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for i, line in enumerate(cell['source']):
            if "trial['image_id'] for trial in test_dataset.trials" in line:
                cell['source'][i] = line.replace(
                    "trial['image_id'] for trial in test_dataset.trials",
                    "Path(f).stem for f in test_dataset.files"
                )

with open('notebooks/02_retrieval_evaluation.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

