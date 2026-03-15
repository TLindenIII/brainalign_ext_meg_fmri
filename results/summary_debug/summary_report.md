# Results Summary

## Retrieval By Subject

Note: EEG rows come from the existing 200-way EEG summary table, while MEG/fMRI rows come from the full-image retrieval evaluator.

| Modality | Subject | Split | Shared | Candidates | M->I Top-1 | M->I Top-5 | M->I 2-way | I->M Top-1 | I->M Top-5 | I->M 2-way |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fmri | 1 | test | False | 874 | 1.60 | 5.26 | 79.08 | 1.49 | 5.84 | 79.84 |
| fmri | 2 | test | False | 874 | 1.95 | 6.86 | 78.51 | 2.52 | 8.47 | 78.91 |
| fmri | 3 | test | False | 874 | 0.69 | 3.78 | 75.65 | 1.60 | 5.03 | 75.96 |
| meg | 1 | test | False | 2245 | 0.18 | 0.76 | 70.59 | 0.18 | 1.16 | 71.21 |
| meg | 2 | test | False | 2245 | 0.76 | 4.32 | 84.71 | 1.20 | 4.63 | 85.53 |
| meg | 3 | test | False | 2245 | 0.76 | 2.76 | 74.94 | 0.62 | 1.96 | 75.41 |
| meg | 4 | test | False | 2245 | 0.22 | 1.16 | 68.10 | 0.09 | 0.67 | 68.17 |
| eeg | 1 | test_200way | False | 200 | 93.50 | 93.50 | 93.50 | 93.50 | 93.50 | 93.50 |
| eeg | 2 | test_200way | False | 200 | 94.50 | 94.50 | 94.50 | 94.50 | 94.50 | 94.50 |
| eeg | 3 | test_200way | False | 200 | 91.50 | 91.50 | 91.50 | 91.50 | 91.50 | 91.50 |
| eeg | 4 | test_200way | False | 200 | 96.50 | 96.50 | 96.50 | 96.50 | 96.50 | 96.50 |
| eeg | 5 | test_200way | False | 200 | 94.00 | 94.00 | 94.00 | 94.00 | 94.00 | 94.00 |
| eeg | 6 | test_200way | False | 200 | 96.00 | 96.00 | 96.00 | 96.00 | 96.00 | 96.00 |
| eeg | 7 | test_200way | False | 200 | 94.00 | 94.00 | 94.00 | 94.00 | 94.00 | 94.00 |
| eeg | 8 | test_200way | False | 200 | 97.50 | 97.50 | 97.50 | 97.50 | 97.50 | 97.50 |
| eeg | 9 | test_200way | False | 200 | 91.50 | 91.50 | 91.50 | 91.50 | 91.50 | 91.50 |
| eeg | 10 | test_200way | False | 200 | 98.50 | 98.50 | 98.50 | 98.50 | 98.50 | 98.50 |

## Retrieval Summary

| Modality | Split | Shared | N | M->I Top-1 | M->I Top-5 | M->I 2-way | Base Top-1 | Base Top-5 | Classes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eeg | test_200way | False | 10 | 94.75 | 94.75 | 94.75 | 0.50 | 2.50 | 200 |
| fmri | test | False | 3 | 1.41 | 5.30 | 77.75 | 0.11 | 0.57 | 874 |
| meg | test | False | 4 | 0.48 | 2.25 | 74.58 | 0.04 | 0.22 | 2245 |

## Conversion By Pair

| Source | Src Sub | Target | Tgt Sub | Split | Shared | Candidates | Forward Top-1 | Forward Top-5 | Forward 2-way | Forward Norm | Reverse Top-1 | Reverse Top-5 | Reverse 2-way | Reverse Norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| meg | 1 | fmri | 1 | test | True | 874 | 0.80 | 2.40 | 67.26 | 0.85 | 0.46 | 1.49 | 67.66 | 0.96 |
| meg | 1 | fmri | 2 | test | True | 874 | 0.57 | 2.52 | 67.65 | 0.86 | 0.23 | 1.72 | 68.25 | 0.97 |
| meg | 1 | fmri | 3 | test | True | 874 | 0.57 | 2.40 | 65.00 | 0.86 | 0.34 | 1.95 | 65.83 | 0.93 |
| meg | 2 | fmri | 1 | test | True | 874 | 0.34 | 3.43 | 71.59 | 0.91 | 0.57 | 3.43 | 71.77 | 0.85 |
| meg | 2 | fmri | 2 | test | True | 874 | 1.60 | 3.55 | 73.59 | 0.94 | 1.14 | 4.00 | 73.99 | 0.87 |
| meg | 2 | fmri | 3 | test | True | 874 | 0.57 | 2.75 | 68.48 | 0.91 | 0.34 | 2.63 | 68.59 | 0.81 |
| meg | 3 | fmri | 1 | test | True | 874 | 1.03 | 2.17 | 70.82 | 0.90 | 0.80 | 2.75 | 70.83 | 0.95 |
| meg | 3 | fmri | 2 | test | True | 874 | 1.03 | 3.66 | 71.42 | 0.91 | 1.60 | 3.89 | 71.56 | 0.95 |
| meg | 3 | fmri | 3 | test | True | 874 | 0.92 | 2.29 | 68.46 | 0.90 | 0.69 | 2.17 | 68.25 | 0.91 |
| meg | 4 | fmri | 1 | test | True | 874 | 0.11 | 1.49 | 66.06 | 0.84 | 0.34 | 1.95 | 66.54 | 0.98 |
| meg | 4 | fmri | 2 | test | True | 874 | 0.23 | 1.37 | 67.15 | 0.86 | 0.23 | 1.83 | 66.99 | 0.98 |
| meg | 4 | fmri | 3 | test | True | 874 | 0.34 | 2.17 | 63.92 | 0.84 | 0.00 | 1.14 | 63.85 | 0.94 |

## Conversion Summary

| Source | Target | Split | Shared | N | Forward 2-way | Forward Norm | Reverse 2-way | Reverse Norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| meg | fmri | test | True | 12 | 68.45 | 0.88 | 68.68 | 0.92 |
