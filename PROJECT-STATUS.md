# FailureScope checkpoint

## Current state

Source: public CodeWith-Pranav/FailureScope, main commit 227c12114e90e836a6a03fce237ca7526bdbb8a1. Published code was preserved; no model retraining or application rebuild was needed to resume.

Live: https://failurescope.vercel.app/ . Production deployment dpl_6meCV2TH5TbQReatJtHXiPGcW8as is READY in the existing Vercel project prj_AXr74syR4PDI7UCNlpbjGLpL8xnq. Deployment uses the unchanged application/runtime files from that commit, uploaded directly. Vercel is now Git-linked for subsequent pushes to main.

## Completed

Offline Python training, frozen logistic and random-forest artifacts, disjoint split manifest, model parity tests, Node API, single/CSV batch predictions, threshold/cost evaluation, global permutation importance, ordered-row stress results, local policy snapshots and JSON exports. Existing 13 automated tests passed before publication, including 80 Python-to-JavaScript comparisons. They were not rerun during this resumption because implementation did not change.

Live browser QA: high-score benchmark returned a 99.7% failure-class score and a review flag; three-row CSV returned two valid rows and one invalid type, including a missing-air imputation warning; frozen evaluation reproduced TP58/FP9/FN10/TN1923; threshold 0.01 policy saved with recall98.5%/precision7.5%; policy restore and snapshot download worked. QA snapshot was removed. Desktop layout inspected; dedicated mobile/assistive-technology checks have not been performed. Export download completed; its bytes were not independently inspected.

## Publication follow-up

GitHub sign-in was restored. The CI workflow is now published and the existing Vercel project is connected to CodeWith-Pranav/FailureScope. The final documentation push will exercise the Git deployment path. Inspect that commit's Actions run and Vercel deployment before claiming automated publication verification. Do not recreate any resource or retrain the frozen models.

Standalone SPEC.md and ARCHITECTURE.md document the preserved implementation. Keep the synthetic-data/contemporaneous-classification scope explicit. Saved experiments are browser-local policy snapshots.
