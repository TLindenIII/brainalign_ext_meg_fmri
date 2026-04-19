# Results Summary

## Retrieval By Subject

Note: Retrieval rows come from the per-subject retrieval evaluator for all available modalities.

| Modality | Subject | Split | Scope | Group | Shared | Candidates | M->I Top-1 | M->I Top-5 | M->I 2-way | I->M Top-1 | I->M Top-5 | I->M 2-way |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eeg | 1 | test | pair | shared-eeg-meg | True | 200 | 1.50 | 10.50 | 74.40 | 1.00 | 5.50 | 62.55 |
| meg | 1 | test | pair | shared-eeg-meg | True | 200 | 4.50 | 16.00 | 70.98 | 4.00 | 14.00 | 66.96 |

## Retrieval Summary

| Modality | Split | Scope | Group | Shared | N | M->I Top-1 | M->I Top-5 | M->I 2-way | Base Top-1 | Base Top-5 | Retrieval Size | Classes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eeg | test | pair | shared-eeg-meg | True | 1 | 1.50 | 10.50 | 74.40 | 0.50 | 2.50 | 200 | 200 |
| meg | test | pair | shared-eeg-meg | True | 1 | 4.50 | 16.00 | 70.98 | 0.50 | 2.50 | 200 | 200 |

## Conversion By Pair

| Source | Src Sub | Target | Tgt Sub | Split | Scope | Group | Shared | Candidates | Forward Top-1 | Forward Top-5 | Forward 2-way | Forward Norm | Reverse Top-1 | Reverse Top-5 | Reverse 2-way | Reverse Norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eeg | 1 | meg | 1 | test | pair | shared-eeg-meg | True | 200 | 1.50 | 7.00 | 65.19 | 0.92 | 1.50 | 6.00 | 65.10 | 0.87 |

## Conversion Summary

Note: Conversion normalization uses the matching retrieval scope/group when available (pairwise shared conversions are normalized against pairwise shared retrieval, 3-way conversions against 3-way retrieval). Full-retrieval normalization is still exported in the CSV outputs as a secondary reference.

| Source | Target | Split | Scope | Group | Shared | N | Forward 2-way | Forward Norm | Reverse 2-way | Reverse Norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eeg | meg | test | pair | shared-eeg-meg | True | 1 | 65.19 | 0.92 | 65.10 | 0.88 |
