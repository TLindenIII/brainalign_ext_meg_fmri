# Results Summary

## Retrieval By Subject

Note: Retrieval rows come from the per-subject retrieval evaluator for all available modalities.

| Modality | Subject | Split | Scope | Group | Shared | Candidates | M->I Top-1 | M->I Top-5 | M->I 2-way | I->M Top-1 | I->M Top-5 | I->M 2-way |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eeg | 1 | test | pair | shared-eeg-meg | True | 200 | 1.50 | 10.50 | 74.40 | 1.00 | 5.50 | 62.55 |
| eeg | 2 | test | pair | shared-eeg-meg | True | 200 | 6.00 | 21.50 | 85.27 | 3.00 | 14.00 | 77.85 |
| eeg | 3 | test | pair | shared-eeg-meg | True | 200 | 3.50 | 13.50 | 79.20 | 2.50 | 10.50 | 71.00 |
| eeg | 4 | test | pair | shared-eeg-meg | True | 200 | 7.50 | 27.00 | 88.48 | 8.00 | 19.50 | 84.78 |
| meg | 1 | test | pair | shared-eeg-meg | True | 200 | 4.50 | 16.00 | 70.98 | 4.00 | 14.00 | 66.96 |
| meg | 2 | test | pair | shared-eeg-meg | True | 200 | 10.00 | 32.00 | 86.23 | 5.00 | 23.50 | 82.11 |
| meg | 3 | test | pair | shared-eeg-meg | True | 200 | 8.00 | 24.00 | 83.53 | 3.00 | 19.50 | 79.07 |
| meg | 4 | test | pair | shared-eeg-meg | True | 200 | 3.00 | 11.00 | 69.79 | 2.00 | 7.50 | 66.19 |

## Retrieval Summary

| Modality | Split | Scope | Group | Shared | N | M->I Top-1 | M->I Top-5 | M->I 2-way | Base Top-1 | Base Top-5 | Retrieval Size | Classes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eeg | test | pair | shared-eeg-meg | True | 4 | 4.62 | 18.12 | 81.84 | 0.50 | 2.50 | 200 | 200 |
| meg | test | pair | shared-eeg-meg | True | 4 | 6.38 | 20.75 | 77.63 | 0.50 | 2.50 | 200 | 200 |

## Conversion By Pair

| Source | Src Sub | Target | Tgt Sub | Split | Scope | Group | Shared | Candidates | Forward Top-1 | Forward Top-5 | Forward 2-way | Forward Norm | Reverse Top-1 | Reverse Top-5 | Reverse 2-way | Reverse Norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eeg | 1 | meg | 1 | test | pair | shared-eeg-meg | True | 200 | 1.50 | 7.00 | 65.19 | 0.92 | 1.50 | 6.00 | 65.10 | 0.87 |
| eeg | 1 | meg | 2 | test | pair | shared-eeg-meg | True | 200 | 2.00 | 7.50 | 69.70 | 0.81 | 3.50 | 9.50 | 74.92 | 1.01 |
| eeg | 1 | meg | 3 | test | pair | shared-eeg-meg | True | 200 | 1.00 | 5.00 | 69.53 | 0.83 | 2.00 | 6.50 | 72.45 | 0.97 |
| eeg | 1 | meg | 4 | test | pair | shared-eeg-meg | True | 200 | 1.50 | 4.00 | 65.61 | 0.94 | 1.50 | 7.00 | 66.36 | 0.89 |
| eeg | 2 | meg | 1 | test | pair | shared-eeg-meg | True | 200 | 1.50 | 7.50 | 69.42 | 0.98 | 1.00 | 7.50 | 71.26 | 0.84 |
| eeg | 2 | meg | 2 | test | pair | shared-eeg-meg | True | 200 | 6.00 | 19.50 | 82.36 | 0.96 | 6.00 | 18.00 | 82.70 | 0.97 |
| eeg | 2 | meg | 3 | test | pair | shared-eeg-meg | True | 200 | 2.50 | 16.00 | 78.89 | 0.94 | 5.00 | 14.00 | 78.51 | 0.92 |
| eeg | 2 | meg | 4 | test | pair | shared-eeg-meg | True | 200 | 1.00 | 5.50 | 67.71 | 0.97 | 1.50 | 6.50 | 68.98 | 0.81 |
| eeg | 3 | meg | 1 | test | pair | shared-eeg-meg | True | 200 | 2.00 | 8.50 | 65.95 | 0.93 | 1.00 | 9.00 | 68.49 | 0.86 |
| eeg | 3 | meg | 2 | test | pair | shared-eeg-meg | True | 200 | 2.00 | 12.50 | 70.99 | 0.82 | 3.50 | 13.50 | 77.45 | 0.98 |
| eeg | 3 | meg | 3 | test | pair | shared-eeg-meg | True | 200 | 4.00 | 13.00 | 76.54 | 0.92 | 3.00 | 17.00 | 78.11 | 0.99 |
| eeg | 3 | meg | 4 | test | pair | shared-eeg-meg | True | 200 | 3.50 | 7.50 | 67.76 | 0.97 | 1.00 | 7.00 | 68.05 | 0.86 |
| eeg | 4 | meg | 1 | test | pair | shared-eeg-meg | True | 200 | 2.50 | 8.50 | 66.91 | 0.94 | 2.00 | 10.50 | 70.12 | 0.79 |
| eeg | 4 | meg | 2 | test | pair | shared-eeg-meg | True | 200 | 1.00 | 10.00 | 71.56 | 0.83 | 4.50 | 16.00 | 80.04 | 0.90 |
| eeg | 4 | meg | 3 | test | pair | shared-eeg-meg | True | 200 | 2.50 | 12.50 | 77.40 | 0.93 | 3.50 | 13.00 | 79.72 | 0.90 |
| eeg | 4 | meg | 4 | test | pair | shared-eeg-meg | True | 200 | 2.50 | 8.50 | 69.77 | 1.00 | 1.50 | 7.50 | 68.51 | 0.77 |

## Conversion Summary

Note: Conversion normalization uses the matching retrieval scope/group when available (pairwise shared conversions are normalized against pairwise shared retrieval, 3-way conversions against 3-way retrieval). Full-retrieval normalization is still exported in the CSV outputs as a secondary reference.

| Source | Target | Split | Scope | Group | Shared | N | Forward 2-way | Forward Norm | Reverse 2-way | Reverse Norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eeg | meg | test | pair | shared-eeg-meg | True | 16 | 70.96 | 0.92 | 73.17 | 0.90 |
