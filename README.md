# FailureScope

A reproducible machine-failure classification dashboard with real trained logistic-regression and random-forest models. Inspect an observation, score a CSV batch, explore threshold tradeoffs, compare frozen models and save policy experiments in your browser.

## Run

Node 22+: `npm ci`, `npm test`, `npm start`. Open http://localhost:3000. No environment variables, paid APIs or database are required for the deployed application.

## Train from source

Use Python 3.12 and an isolated virtual environment. Install `requirements.txt`, then run `python train.py`. The included original `ai4i2020.csv` is the UCI AI4I 2020 dataset. The script records its SHA-256, explicit train/validation/test row indices, environment versions, trained numeric artifacts, metrics, permutation importance and Python inference fixtures. Run `npm test` after retraining to check the exported serving implementation.

Training runs offline. The website does not accept training jobs. Exported model parameters are in `model-bundle.json`; the Vercel Node API uses them directly. The forest reproduces sklearn's float32 input conversion and leaf probabilities. Tests compare 80 exported-model predictions against Python to within 1e-12.

## Measured results

Fixed stratified 60/20/20 split, seed 42. Training has 6,000 rows, validation 2,000, test 2,000. The model is chosen by validation average precision (AP), then its threshold by validation F2. No test-based model selection is performed. Frozen selected policy: random forest rf-v1, threshold 0.46.

| Model | Test AP | Test precision | Test recall |
| --- | --- | --- | --- |
| Logistic regression lr-v1 | 0.412 | 16.8% | 82.4% |
| Random forest rf-v1 | 0.866 | 86.6% | 85.3% |

Both rows use the same frozen threshold. Random forest confusion matrix: TP 58, FP 9, FN 10, TN 1,923. An always-negative classifier has 96.6% test accuracy but zero failure recall. AP is sklearn average_precision_score, not trapezoidal PR area.

A separately retrained ordered-row stress experiment gives forest AP 0.779 and recall 71.8% at its own validation-selected threshold 0.48. This illustrates a limitation of random splitting; row order is not a verified timeline or independent-machine split.

## Features and limitations

- Single and batch inference through a real same-origin API, not fabricated output.
- Training-only numeric medians and logistic scaling; fixed category encoding.
- Six observed inputs plus temperature gap, mechanical power and wear × torque.
- Class weighting; no SMOTE, calibration or hyperparameter tuning claimed.
- Threshold-driven confusion matrix, recall/precision curves and illustrative costs.
- Validation permutation importance; global association, not causal/local explanations.
- Versioned frozen artifacts and downloadable model reports.
- Up to 20 named policy snapshots in localStorage; JSON export for backup.
- Strict six-column CSV format, 100-row limit, per-row validation, missing-value and training-range warnings.

This uses **synthetic benchmark data**. It classifies contemporaneous recorded machine states; it does not forecast future failures, estimate remaining useful life, certify machine safety or demonstrate field accuracy. Scores are not calibrated real-world probabilities. Exploring test thresholds in the UI does not create an unbiased new evaluation. Retraining or threshold selection after seeing the test results requires a new independent test set.

## Architecture

`train.py` → `model-bundle.json` → `lib/inference.mjs` → `api/index.mjs` → `public/index.html`.

The frontend is buildless HTML/CSS/JavaScript. Vercel hosts it and the Node API. The model bundle is included in the function and is not exposed as a static asset. `server.mjs` is the local equivalent. Application code does not persist submitted readings; hosting providers can retain request metadata. Experiments are local to the browser and have no cross-device sync.

## API

- `GET /api/health`: readiness/version.
- `GET /api/report`: model cards, aggregated evaluation curves, benchmark examples and provenance.
- `POST /api/predict`: `{ "row": { "type":"L", "air":300, "process":310, "speed":1500, "torque":40, "wear":100 }, "model":"rf-v1", "threshold":0.46 }`.
- `POST /api/batch`: `{ "csv":"type,air,process,speed,torque,wear\nL,300,310,1500,40,100", "model":"rf-v1", "threshold":0.46 }`.
- `POST /api/evaluate`: model, threshold and optional integer `falseAlarmCost`/`missCost` in 1–1,000. Defaults 1 and 20.

Input temperatures are Kelvin, speed rpm, torque Nm and wear minutes. Missing numeric values are imputed with training medians; missing type uses training mode L. At least one observed numeric measurement is required. Hard input bounds protect the demo; they are not certified operating envelopes. Requests are bounded to 100 KB. The origin check is not authentication or a general abuse-prevention mechanism.

## Deployment

Vercel only: no framework preset, output directory `public`, Node 22. The API is `api/index.mjs`. `vercel.json` rewrites API paths and includes the model bundle. GitHub Actions runs the Node tests on pushes and pull requests. Training is intentionally not repeated on every deploy.

## Dataset attribution

AI4I 2020 Predictive Maintenance Dataset [Dataset]. (2020). UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5HS5C. Original source: https://archive.ics.uci.edu/dataset/601/ai4i. Dataset license: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/. The original CSV is redistributed unchanged; model parameters, features, splits, metrics and interface are derived work. Dataset contributors do not endorse this project.

See `BEGINNER-GUIDE.md` for a demonstration and interview preparation, and `PROJECT-STATUS.md` for the latest implementation checkpoint.
