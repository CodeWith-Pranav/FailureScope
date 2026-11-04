# FailureScope specification

Purpose: demonstrate reproducible classical ML for contemporaneous machine-state failure classification using the synthetic UCI AI4I 2020 benchmark. This is a portfolio research demo, not a future-failure forecast or industrial safety system.

## Completed behavior

- Frozen logistic regression and random forest; Python training and full-precision numeric export.
- Training-only preprocessing, engineered sensor features, seeded disjoint 60/20/20 split, validation-only model/threshold selection and held-out evaluation.
- Single observation and 1–100 row CSV inference, per-row errors, missing-value imputation warnings and training-range warnings.
- Threshold exploration, confusion matrix, precision/recall curves, explicitly chosen cost weights, model comparison and ordered-row stress results.
- Global validation permutation importance, provenance, dataset attribution and model card.
- Up to 20 browser-local named policy snapshots; JSON report/result/snapshot exports.

## Runtime contract

Node 22, no paid model API or database. Frontend and inference API deploy together on Vercel. Training stays offline. Data submissions are not persisted by application code; named policy snapshots use localStorage and require export for portability.

## Completion criteria

Preserve the frozen training evidence. Existing 13 automated checks passed; only rerun if related implementation changes. Remaining milestones are publication, browser validation, source/CI synchronization and final deployment provenance. No claims of calibrated real-world probabilities, remaining useful life or production traffic.
