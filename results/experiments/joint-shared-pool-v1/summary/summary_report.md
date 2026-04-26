# Results Summary

## Retrieval By Subject

Note: Retrieval rows come from the per-subject retrieval evaluator for all available modalities.

| Modality | Subject | Split | Scope | Group | Shared | Candidates | M->I Top-1 | M->I Top-5 | M->I 2-way | I->M Top-1 | I->M Top-5 | I->M 2-way |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eeg | 1 | test | three_way | shared-eeg-fmri-meg | True | 200 | 1.50 | 7.50 | 69.32 | 1.00 | 5.00 | 57.19 |
| eeg | 2 | test | three_way | shared-eeg-fmri-meg | True | 200 | 3.00 | 9.00 | 73.77 | 2.50 | 6.50 | 66.23 |
| eeg | 3 | test | three_way | shared-eeg-fmri-meg | True | 200 | 2.00 | 6.50 | 66.84 | 1.00 | 4.00 | 60.73 |
| fmri | 1 | test | three_way | shared-eeg-fmri-meg | True | 200 | 5.00 | 12.00 | 76.00 | 1.00 | 9.00 | 71.85 |
| fmri | 2 | test | three_way | shared-eeg-fmri-meg | True | 200 | 3.00 | 11.50 | 72.03 | 2.00 | 10.50 | 69.64 |
| fmri | 3 | test | three_way | shared-eeg-fmri-meg | True | 200 | 3.50 | 10.50 | 68.25 | 2.00 | 8.00 | 65.90 |
| meg | 1 | test | three_way | shared-eeg-fmri-meg | True | 200 | 1.00 | 6.00 | 60.80 | 0.50 | 6.00 | 58.63 |
| meg | 2 | test | three_way | shared-eeg-fmri-meg | True | 200 | 4.00 | 12.50 | 70.57 | 1.00 | 6.00 | 66.90 |
| meg | 3 | test | three_way | shared-eeg-fmri-meg | True | 200 | 2.00 | 8.00 | 64.33 | 1.50 | 4.00 | 62.31 |

## Retrieval Summary

| Modality | Split | Scope | Group | Shared | N | M->I Top-1 | M->I Top-5 | M->I 2-way | Base Top-1 | Base Top-5 | Retrieval Size | Classes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eeg | test | three_way | shared-eeg-fmri-meg | True | 3 | 2.17 | 7.67 | 69.98 | 0.50 | 2.50 | 200 | 200 |
| fmri | test | three_way | shared-eeg-fmri-meg | True | 3 | 3.83 | 11.33 | 72.09 | 0.50 | 2.50 | 200 | 200 |
| meg | test | three_way | shared-eeg-fmri-meg | True | 3 | 2.33 | 8.83 | 65.23 | 0.50 | 2.50 | 200 | 200 |

## Conversion By Pair

| Source | Src Sub | Target | Tgt Sub | Split | Scope | Group | Shared | Candidates | Forward Top-1 | Forward Top-5 | Forward 2-way | Forward Norm | Reverse Top-1 | Reverse Top-5 | Reverse 2-way | Reverse Norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eeg | 1 | fmri | 1 | test | three_way | shared-eeg-fmri-meg | True | 200 | 1.50 | 4.50 | 59.92 | 0.79 | 2.00 | 4.00 | 59.80 | 0.86 |
| eeg | 1 | fmri | 2 | test | three_way | shared-eeg-fmri-meg | True | 200 | 2.00 | 6.50 | 65.24 | 0.91 | 0.50 | 5.50 | 67.51 | 0.97 |
| eeg | 1 | fmri | 3 | test | three_way | shared-eeg-fmri-meg | True | 200 | 1.00 | 3.50 | 58.07 | 0.85 | 1.50 | 6.00 | 58.86 | 0.85 |
| eeg | 1 | meg | 1 | test | three_way | shared-eeg-fmri-meg | True | 200 | 0.00 | 3.50 | 51.25 | 0.84 | 0.00 | 2.00 | 49.75 | 0.72 |
| eeg | 1 | meg | 2 | test | three_way | shared-eeg-fmri-meg | True | 200 | 1.50 | 8.00 | 69.86 | 0.99 | 1.50 | 6.50 | 71.93 | 1.04 |
| eeg | 1 | meg | 3 | test | three_way | shared-eeg-fmri-meg | True | 200 | 1.50 | 4.50 | 58.75 | 0.91 | 1.00 | 6.00 | 60.88 | 0.88 |
| eeg | 2 | fmri | 1 | test | three_way | shared-eeg-fmri-meg | True | 200 | 2.50 | 5.00 | 60.74 | 0.80 | 0.50 | 3.00 | 63.93 | 0.87 |
| eeg | 2 | fmri | 2 | test | three_way | shared-eeg-fmri-meg | True | 200 | 0.50 | 3.50 | 66.30 | 0.92 | 1.50 | 7.50 | 67.26 | 0.91 |
| eeg | 2 | fmri | 3 | test | three_way | shared-eeg-fmri-meg | True | 200 | 2.00 | 5.50 | 60.06 | 0.88 | 0.50 | 6.00 | 59.76 | 0.81 |
| eeg | 2 | meg | 1 | test | three_way | shared-eeg-fmri-meg | True | 200 | 0.50 | 5.00 | 53.84 | 0.89 | 1.50 | 8.00 | 54.93 | 0.74 |
| eeg | 2 | meg | 2 | test | three_way | shared-eeg-fmri-meg | True | 200 | 1.00 | 6.00 | 71.01 | 1.01 | 2.50 | 6.00 | 71.60 | 0.97 |
| eeg | 2 | meg | 3 | test | three_way | shared-eeg-fmri-meg | True | 200 | 2.00 | 5.00 | 62.01 | 0.96 | 2.50 | 8.50 | 64.78 | 0.88 |
| eeg | 3 | fmri | 1 | test | three_way | shared-eeg-fmri-meg | True | 200 | 1.00 | 3.50 | 59.85 | 0.79 | 1.00 | 5.00 | 59.16 | 0.89 |
| eeg | 3 | fmri | 2 | test | three_way | shared-eeg-fmri-meg | True | 200 | 1.00 | 4.00 | 63.86 | 0.89 | 0.50 | 8.50 | 66.10 | 0.99 |
| eeg | 3 | fmri | 3 | test | three_way | shared-eeg-fmri-meg | True | 200 | 1.50 | 5.00 | 55.80 | 0.82 | 1.50 | 6.00 | 55.45 | 0.83 |
| eeg | 3 | meg | 1 | test | three_way | shared-eeg-fmri-meg | True | 200 | 0.00 | 4.50 | 50.06 | 0.82 | 0.00 | 2.00 | 51.98 | 0.78 |
| eeg | 3 | meg | 2 | test | three_way | shared-eeg-fmri-meg | True | 200 | 1.50 | 4.50 | 65.48 | 0.93 | 1.50 | 6.50 | 70.14 | 1.05 |
| eeg | 3 | meg | 3 | test | three_way | shared-eeg-fmri-meg | True | 200 | 1.00 | 5.50 | 60.09 | 0.93 | 1.00 | 5.00 | 61.74 | 0.92 |
| meg | 1 | fmri | 1 | test | three_way | shared-eeg-fmri-meg | True | 200 | 1.00 | 5.00 | 60.79 | 0.80 | 0.00 | 4.00 | 60.90 | 1.00 |
| meg | 1 | fmri | 2 | test | three_way | shared-eeg-fmri-meg | True | 200 | 0.50 | 3.50 | 56.46 | 0.78 | 1.00 | 5.00 | 56.33 | 0.93 |
| meg | 1 | fmri | 3 | test | three_way | shared-eeg-fmri-meg | True | 200 | 1.00 | 5.00 | 56.85 | 0.83 | 1.50 | 5.00 | 56.73 | 0.93 |
| meg | 2 | fmri | 1 | test | three_way | shared-eeg-fmri-meg | True | 200 | 2.50 | 8.00 | 61.85 | 0.81 | 3.00 | 6.50 | 61.23 | 0.87 |
| meg | 2 | fmri | 2 | test | three_way | shared-eeg-fmri-meg | True | 200 | 1.50 | 8.50 | 66.66 | 0.93 | 1.00 | 7.00 | 67.09 | 0.95 |
| meg | 2 | fmri | 3 | test | three_way | shared-eeg-fmri-meg | True | 200 | 1.50 | 4.50 | 59.39 | 0.87 | 2.00 | 5.00 | 58.74 | 0.83 |
| meg | 3 | fmri | 1 | test | three_way | shared-eeg-fmri-meg | True | 200 | 1.00 | 6.00 | 59.90 | 0.79 | 2.50 | 5.50 | 60.09 | 0.93 |
| meg | 3 | fmri | 2 | test | three_way | shared-eeg-fmri-meg | True | 200 | 1.00 | 5.00 | 61.51 | 0.85 | 1.00 | 5.00 | 61.37 | 0.95 |
| meg | 3 | fmri | 3 | test | three_way | shared-eeg-fmri-meg | True | 200 | 0.50 | 2.50 | 56.28 | 0.82 | 1.00 | 4.50 | 56.57 | 0.88 |

## Conversion Summary

Note: Conversion normalization uses the matching retrieval scope/group when available (pairwise shared conversions are normalized against pairwise shared retrieval, 3-way conversions against 3-way retrieval). Full-retrieval normalization is still exported in the CSV outputs as a secondary reference.

| Source | Target | Split | Scope | Group | Shared | N | Forward 2-way | Forward Norm | Reverse 2-way | Reverse Norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eeg | fmri | test | three_way | shared-eeg-fmri-meg | True | 9 | 61.09 | 0.85 | 61.98 | 0.89 |
| eeg | meg | test | three_way | shared-eeg-fmri-meg | True | 9 | 60.26 | 0.92 | 61.97 | 0.89 |
| meg | fmri | test | three_way | shared-eeg-fmri-meg | True | 9 | 59.97 | 0.83 | 59.89 | 0.92 |
