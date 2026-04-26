# Results Summary

## Retrieval By Subject

Note: Retrieval rows come from the per-subject retrieval evaluator for all available modalities.

| Modality | Subject | Split | Scope | Group | Shared | Candidates | M->I Top-1 | M->I Top-5 | M->I 2-way | I->M Top-1 | I->M Top-5 | I->M 2-way |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eeg | 1 | test | pair | shared-eeg-meg | True | 200 | 11.00 | 22.00 | 83.90 | 8.00 | 20.00 | 75.66 |
| eeg | 2 | test | pair | shared-eeg-meg | True | 200 | 8.50 | 28.50 | 85.41 | 10.50 | 27.50 | 83.92 |
| eeg | 3 | test | pair | shared-eeg-meg | True | 200 | 11.00 | 29.50 | 87.79 | 8.00 | 21.00 | 79.76 |
| eeg | 4 | test | pair | shared-eeg-meg | True | 200 | 12.50 | 29.00 | 87.87 | 9.00 | 22.50 | 84.11 |
| eeg | 5 | test | pair | shared-eeg-meg | True | 200 | 7.00 | 24.00 | 79.07 | 2.50 | 12.50 | 73.93 |
| eeg | 6 | test | pair | shared-eeg-meg | True | 200 | 9.50 | 33.50 | 87.95 | 6.50 | 20.50 | 81.06 |
| eeg | 7 | test | pair | shared-eeg-meg | True | 200 | 9.00 | 25.50 | 86.03 | 8.00 | 21.50 | 83.58 |
| eeg | 8 | test | pair | shared-eeg-meg | True | 200 | 14.50 | 41.00 | 89.51 | 5.00 | 21.00 | 81.68 |
| eeg | 9 | test | pair | shared-eeg-meg | True | 200 | 8.00 | 25.50 | 82.37 | 6.00 | 19.50 | 74.72 |
| eeg | 10 | test | pair | shared-eeg-meg | True | 200 | 15.00 | 39.50 | 91.66 | 10.50 | 34.00 | 88.51 |
| meg | 1 | test | pair | shared-eeg-meg | True | 200 | 3.50 | 13.50 | 75.68 | 2.00 | 10.50 | 70.92 |
| meg | 2 | test | pair | shared-eeg-meg | True | 200 | 12.00 | 31.00 | 84.49 | 8.50 | 26.00 | 79.37 |
| meg | 3 | test | pair | shared-eeg-meg | True | 200 | 4.00 | 19.00 | 81.08 | 5.00 | 14.00 | 78.50 |
| meg | 4 | test | pair | shared-eeg-meg | True | 200 | 2.00 | 9.50 | 67.30 | 3.50 | 10.00 | 63.41 |

## Retrieval Summary

| Modality | Split | Scope | Group | Shared | N | M->I Top-1 | M->I Top-5 | M->I 2-way | Base Top-1 | Base Top-5 | Retrieval Size | Classes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eeg | test | pair | shared-eeg-meg | True | 10 | 10.60 | 29.80 | 86.16 | 0.50 | 2.50 | 200 | 200 |
| meg | test | pair | shared-eeg-meg | True | 4 | 5.38 | 18.25 | 77.14 | 0.50 | 2.50 | 200 | 200 |

## Conversion By Pair

| Source | Src Sub | Target | Tgt Sub | Split | Scope | Group | Shared | Candidates | Forward Top-1 | Forward Top-5 | Forward 2-way | Forward Norm | Reverse Top-1 | Reverse Top-5 | Reverse 2-way | Reverse Norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eeg | 1 | meg | 1 | test | pair | shared-eeg-meg | True | 200 | 1.00 | 7.00 | 69.01 | 0.91 | 1.50 | 9.00 | 69.59 | 0.83 |
| eeg | 1 | meg | 2 | test | pair | shared-eeg-meg | True | 200 | 4.00 | 17.00 | 79.04 | 0.94 | 3.00 | 16.50 | 79.28 | 0.94 |
| eeg | 1 | meg | 3 | test | pair | shared-eeg-meg | True | 200 | 0.50 | 10.50 | 78.14 | 0.96 | 2.00 | 11.50 | 76.93 | 0.92 |
| eeg | 1 | meg | 4 | test | pair | shared-eeg-meg | True | 200 | 1.50 | 5.00 | 66.78 | 0.99 | 2.00 | 5.50 | 67.01 | 0.80 |
| eeg | 2 | meg | 1 | test | pair | shared-eeg-meg | True | 200 | 2.00 | 7.00 | 68.20 | 0.90 | 3.00 | 9.50 | 69.44 | 0.81 |
| eeg | 2 | meg | 2 | test | pair | shared-eeg-meg | True | 200 | 2.00 | 14.50 | 77.48 | 0.92 | 2.00 | 15.50 | 77.18 | 0.90 |
| eeg | 2 | meg | 3 | test | pair | shared-eeg-meg | True | 200 | 3.00 | 14.00 | 76.60 | 0.94 | 2.00 | 15.00 | 76.24 | 0.89 |
| eeg | 2 | meg | 4 | test | pair | shared-eeg-meg | True | 200 | 2.00 | 4.50 | 65.25 | 0.97 | 2.50 | 9.00 | 65.22 | 0.76 |
| eeg | 3 | meg | 1 | test | pair | shared-eeg-meg | True | 200 | 1.00 | 9.00 | 67.96 | 0.90 | 2.50 | 8.50 | 68.51 | 0.78 |
| eeg | 3 | meg | 2 | test | pair | shared-eeg-meg | True | 200 | 3.50 | 12.50 | 80.29 | 0.95 | 5.00 | 15.50 | 80.83 | 0.92 |
| eeg | 3 | meg | 3 | test | pair | shared-eeg-meg | True | 200 | 1.50 | 9.00 | 74.99 | 0.92 | 6.00 | 12.00 | 75.28 | 0.86 |
| eeg | 3 | meg | 4 | test | pair | shared-eeg-meg | True | 200 | 2.00 | 6.50 | 66.83 | 0.99 | 1.00 | 6.00 | 66.40 | 0.76 |
| eeg | 4 | meg | 1 | test | pair | shared-eeg-meg | True | 200 | 2.00 | 6.00 | 69.07 | 0.91 | 2.00 | 10.50 | 68.89 | 0.78 |
| eeg | 4 | meg | 2 | test | pair | shared-eeg-meg | True | 200 | 4.00 | 15.50 | 78.95 | 0.93 | 5.00 | 12.50 | 79.26 | 0.90 |
| eeg | 4 | meg | 3 | test | pair | shared-eeg-meg | True | 200 | 1.50 | 14.00 | 76.36 | 0.94 | 2.50 | 11.50 | 76.75 | 0.87 |
| eeg | 4 | meg | 4 | test | pair | shared-eeg-meg | True | 200 | 2.50 | 6.00 | 66.26 | 0.98 | 0.50 | 5.00 | 66.13 | 0.75 |
| eeg | 5 | meg | 1 | test | pair | shared-eeg-meg | True | 200 | 1.00 | 7.00 | 66.39 | 0.88 | 1.50 | 6.00 | 66.22 | 0.84 |
| eeg | 5 | meg | 2 | test | pair | shared-eeg-meg | True | 200 | 2.00 | 15.00 | 76.03 | 0.90 | 4.00 | 17.50 | 76.44 | 0.97 |
| eeg | 5 | meg | 3 | test | pair | shared-eeg-meg | True | 200 | 2.00 | 9.50 | 73.44 | 0.91 | 3.00 | 10.00 | 73.10 | 0.92 |
| eeg | 5 | meg | 4 | test | pair | shared-eeg-meg | True | 200 | 0.50 | 5.00 | 64.43 | 0.96 | 0.50 | 4.50 | 63.34 | 0.80 |
| eeg | 6 | meg | 1 | test | pair | shared-eeg-meg | True | 200 | 1.50 | 9.50 | 65.79 | 0.87 | 2.00 | 9.50 | 66.81 | 0.76 |
| eeg | 6 | meg | 2 | test | pair | shared-eeg-meg | True | 200 | 4.00 | 17.00 | 80.21 | 0.95 | 5.00 | 15.00 | 79.55 | 0.90 |
| eeg | 6 | meg | 3 | test | pair | shared-eeg-meg | True | 200 | 3.50 | 15.00 | 77.11 | 0.95 | 4.00 | 12.50 | 76.97 | 0.88 |
| eeg | 6 | meg | 4 | test | pair | shared-eeg-meg | True | 200 | 1.50 | 4.50 | 64.50 | 0.96 | 1.50 | 6.50 | 64.23 | 0.73 |
| eeg | 7 | meg | 1 | test | pair | shared-eeg-meg | True | 200 | 2.00 | 10.00 | 71.34 | 0.94 | 3.50 | 7.00 | 71.61 | 0.83 |
| eeg | 7 | meg | 2 | test | pair | shared-eeg-meg | True | 200 | 3.00 | 15.50 | 78.89 | 0.93 | 6.00 | 17.50 | 78.42 | 0.91 |
| eeg | 7 | meg | 3 | test | pair | shared-eeg-meg | True | 200 | 3.00 | 12.50 | 78.73 | 0.97 | 2.50 | 13.50 | 79.70 | 0.93 |
| eeg | 7 | meg | 4 | test | pair | shared-eeg-meg | True | 200 | 1.50 | 5.50 | 65.49 | 0.97 | 1.00 | 5.50 | 65.44 | 0.76 |
| eeg | 8 | meg | 1 | test | pair | shared-eeg-meg | True | 200 | 4.00 | 11.50 | 70.63 | 0.93 | 2.50 | 8.00 | 70.94 | 0.79 |
| eeg | 8 | meg | 2 | test | pair | shared-eeg-meg | True | 200 | 6.00 | 23.00 | 83.17 | 0.98 | 6.00 | 20.50 | 83.24 | 0.93 |
| eeg | 8 | meg | 3 | test | pair | shared-eeg-meg | True | 200 | 4.50 | 14.50 | 78.99 | 0.97 | 5.00 | 16.00 | 78.62 | 0.88 |
| eeg | 8 | meg | 4 | test | pair | shared-eeg-meg | True | 200 | 3.00 | 6.50 | 66.41 | 0.99 | 1.00 | 6.50 | 65.66 | 0.73 |
| eeg | 9 | meg | 1 | test | pair | shared-eeg-meg | True | 200 | 0.50 | 7.00 | 66.62 | 0.88 | 1.50 | 5.00 | 67.38 | 0.82 |
| eeg | 9 | meg | 2 | test | pair | shared-eeg-meg | True | 200 | 3.00 | 13.50 | 77.94 | 0.92 | 3.50 | 15.50 | 77.81 | 0.94 |
| eeg | 9 | meg | 3 | test | pair | shared-eeg-meg | True | 200 | 1.50 | 10.50 | 72.92 | 0.90 | 2.50 | 11.50 | 72.84 | 0.88 |
| eeg | 9 | meg | 4 | test | pair | shared-eeg-meg | True | 200 | 1.00 | 8.00 | 64.81 | 0.96 | 1.00 | 7.50 | 64.42 | 0.78 |
| eeg | 10 | meg | 1 | test | pair | shared-eeg-meg | True | 200 | 2.00 | 13.50 | 71.93 | 0.95 | 2.50 | 11.50 | 72.47 | 0.79 |
| eeg | 10 | meg | 2 | test | pair | shared-eeg-meg | True | 200 | 7.50 | 21.50 | 81.49 | 0.96 | 7.50 | 22.00 | 82.18 | 0.90 |
| eeg | 10 | meg | 3 | test | pair | shared-eeg-meg | True | 200 | 5.00 | 18.00 | 82.28 | 1.01 | 4.50 | 16.50 | 83.17 | 0.91 |
| eeg | 10 | meg | 4 | test | pair | shared-eeg-meg | True | 200 | 1.00 | 5.00 | 66.05 | 0.98 | 1.00 | 7.00 | 65.82 | 0.72 |

## Conversion Summary

Note: Conversion normalization uses the matching retrieval scope/group when available (pairwise shared conversions are normalized against pairwise shared retrieval, 3-way conversions against 3-way retrieval). Full-retrieval normalization is still exported in the CSV outputs as a secondary reference.

| Source | Target | Split | Scope | Group | Shared | N | Forward 2-way | Forward Norm | Reverse 2-way | Reverse Norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eeg | meg | test | pair | shared-eeg-meg | True | 40 | 72.67 | 0.94 | 72.73 | 0.84 |
