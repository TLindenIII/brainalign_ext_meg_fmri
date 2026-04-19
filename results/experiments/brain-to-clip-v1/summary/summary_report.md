# Results Summary

## Retrieval By Subject

Note: Retrieval rows come from the per-subject retrieval evaluator for all available modalities.

| Modality | Subject | Split | Scope | Group | Shared | Candidates | M->I Top-1 | M->I Top-5 | M->I 2-way | I->M Top-1 | I->M Top-5 | I->M 2-way |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eeg | 1 | test | pair | shared-eeg-meg | True | 200 | 11.00 | 22.00 | 83.90 | 8.00 | 20.00 | 75.66 |
| meg | 1 | test | pair | shared-eeg-meg | True | 200 | 3.50 | 13.50 | 75.68 | 2.00 | 10.50 | 70.92 |

## Retrieval Summary

| Modality | Split | Scope | Group | Shared | N | M->I Top-1 | M->I Top-5 | M->I 2-way | Base Top-1 | Base Top-5 | Retrieval Size | Classes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eeg | test | pair | shared-eeg-meg | True | 1 | 11.00 | 22.00 | 83.90 | 0.50 | 2.50 | 200 | 200 |
| meg | test | pair | shared-eeg-meg | True | 1 | 3.50 | 13.50 | 75.68 | 0.50 | 2.50 | 200 | 200 |

## Conversion By Pair

| Source | Src Sub | Target | Tgt Sub | Split | Scope | Group | Shared | Candidates | Forward Top-1 | Forward Top-5 | Forward 2-way | Forward Norm | Reverse Top-1 | Reverse Top-5 | Reverse 2-way | Reverse Norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eeg | 1 | meg | 1 | test | pair | shared-eeg-meg | True | 200 | 1.00 | 7.00 | 69.01 | 0.91 | 1.50 | 9.00 | 69.59 | 0.83 |

## Conversion Summary

Note: Conversion normalization uses the matching retrieval scope/group when available (pairwise shared conversions are normalized against pairwise shared retrieval, 3-way conversions against 3-way retrieval). Full-retrieval normalization is still exported in the CSV outputs as a secondary reference.

| Source | Target | Split | Scope | Group | Shared | N | Forward 2-way | Forward Norm | Reverse 2-way | Reverse Norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eeg | meg | test | pair | shared-eeg-meg | True | 1 | 69.01 | 0.91 | 69.59 | 0.83 |
