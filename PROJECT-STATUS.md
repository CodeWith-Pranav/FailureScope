# FailureScope checkpoint

## Current state

Runtime source baseline: public CodeWith-Pranav/FailureScope, commit 227c12114e90e836a6a03fce237ca7526bdbb8a1. Published code was preserved; no model retraining or application rebuild was needed to resume.

Live: https://failurescope.vercel.app/ . Production deployment dpl_6meCV2TH5TbQReatJtHXiPGcW8as is READY in the existing Vercel project prj_AXr74syR4PDI7UCNlpbjGLpL8xnq. Deployment uses the unchanged application/runtime files from that commit, uploaded directly. Vercel is now Git-linked for subsequent pushes to main.

## Completed

Offline Python training, frozen logistic and random-forest artifacts, disjoint split manifest, model parity tests, Node API, single/CSV batch predictions, threshold/cost evaluation, global permutation importance, ordered-row stress results, local policy snapshots and JSON exports. Existing 13 automated tests passed before publication, including 80 Python-to-JavaScript comparisons. They were not rerun during this resumption because implementation did not change.

Live browser QA: high-score benchmark returned a 99.7% failure-class score and a review flag; three-row CSV returned two valid rows and one invalid type, including a missing-air imputation warning; frozen evaluation reproduced TP58/FP9/FN10/TN1923; threshold 0.01 policy saved with recall98.5%/precision7.5%; policy restore and snapshot download worked. QA snapshot was removed. Desktop layout inspected; dedicated mobile/assistive-technology checks have not been performed. Export download completed; its bytes were not independently inspected.

## Publication completed

The public repository includes the CI workflow, SPEC.md and ARCHITECTURE.md. GitHub Actions Tests passed on commit 304d41aa7a9bbab0cf8954350e62453f8eff25cb (run 41881668324). The existing Vercel project is connected to the repository. The first Git-driven production deployment, dpl_M1hmyTu5ZGkrAecyTrm4bx1fjN33 from documentation commit 39e63f75b0d302f06ca0cb144da02df6a0b3c04f, reached READY and assigned https://failurescope.vercel.app/.

No application or model changes were made during publication. The source/runtime baseline remains 227c121; later commits add CI and documentation. Future pushes to main trigger CI and Vercel builds. CI and deployment run independently; a passing test is not configured as a blocking promotion gate.

No required implementation milestone remains. Optional future work includes dedicated mobile/accessibility QA and new independently evaluated model experiments. Keep synthetic-data/contemporaneous-classification limitations explicit. Saved experiments remain browser-local policy snapshots.
