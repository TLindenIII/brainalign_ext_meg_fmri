# Results Summary

## Retrieval By Subject

Note: Retrieval rows come from the per-subject retrieval evaluator for all available modalities.

| Modality | Subject | Split | Scope | Group | Shared | Candidates | M->I Top-1 | M->I Top-5 | M->I 2-way | I->M Top-1 | I->M Top-5 | I->M 2-way |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eeg | 1 | test | three_way | shared-eeg-fmri-meg | True | 200 | 1.50 | 7.50 | 69.32 | 1.00 | 5.00 | 57.19 |
| fmri | 1 | test | three_way | shared-eeg-fmri-meg | True | 200 | 5.00 | 12.00 | 76.00 | 1.00 | 9.00 | 71.85 |
| meg | 1 | test | three_way | shared-eeg-fmri-meg | True | 200 | 1.00 | 6.00 | 60.80 | 0.50 | 6.00 | 58.63 |

## Retrieval Summary

| Modality | Split | Scope | Group | Shared | N | M->I Top-1 | M->I Top-5 | M->I 2-way | Base Top-1 | Base Top-5 | Retrieval Size | Classes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eeg | test | three_way | shared-eeg-fmri-meg | True | 1 | 1.50 | 7.50 | 69.32 | 0.50 | 2.50 | 200 | 200 |
| fmri | test | three_way | shared-eeg-fmri-meg | True | 1 | 5.00 | 12.00 | 76.00 | 0.50 | 2.50 | 200 | 200 |
| meg | test | three_way | shared-eeg-fmri-meg | True | 1 | 1.00 | 6.00 | 60.80 | 0.50 | 2.50 | 200 | 200 |

## Conversion By Pair

| Source | Src Sub | Target | Tgt Sub | Split | Scope | Group | Shared | Candidates | Forward Top-1 | Forward Top-5 | Forward 2-way | Forward Norm | Reverse Top-1 | Reverse Top-5 | Reverse 2-way | Reverse Norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eeg | 1 | fmri | 1 | test | three_way | shared-eeg-fmri-meg | True | 200 | 1.50 | 4.50 | 59.92 | 0.79 | 2.00 | 4.00 | 59.80 | 0.86 |
| eeg | 1 | meg | 1 | test | three_way | shared-eeg-fmri-meg | True | 200 | 0.00 | 3.50 | 51.25 | 0.84 | 0.00 | 2.00 | 49.75 | 0.72 |
| meg | 1 | fmri | 1 | test | three_way | shared-eeg-fmri-meg | True | 200 | 1.00 | 5.00 | 60.79 | 0.80 | 0.00 | 4.00 | 60.90 | 1.00 |

## Conversion Summary

Note: Conversion normalization uses the matching retrieval scope/group when available (pairwise shared conversions are normalized against pairwise shared retrieval, 3-way conversions against 3-way retrieval). Full-retrieval normalization is still exported in the CSV outputs as a secondary reference.

| Source | Target | Split | Scope | Group | Shared | N | Forward 2-way | Forward Norm | Reverse 2-way | Reverse Norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eeg | fmri | test | three_way | shared-eeg-fmri-meg | True | 1 | 59.92 | 0.79 | 59.80 | 0.86 |
| eeg | meg | test | three_way | shared-eeg-fmri-meg | True | 1 | 51.25 | 0.84 | 49.75 | 0.72 |
| meg | fmri | test | three_way | shared-eeg-fmri-meg | True | 1 | 60.79 | 0.80 | 60.90 | 1.00 |
