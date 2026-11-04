# FailureScope architecture

## Training

`train.py` reads `ai4i2020.csv`. A seeded stratified split records explicit disjoint indices in `split-indices.json`. Numeric medians and logistic scaling are fitted only on training observations. The script trains two fixed class-weighted candidates, selects the model by validation average precision and its threshold by validation F2, then reports held-out metrics. A separately trained ordered-row experiment exposes split sensitivity.

`model-bundle.json` contains exported logistic parameters, forest tree arrays, preprocessing, version/provenance, aggregated metrics, test scores for interactive threshold evaluation and Python golden inference examples. The original dataset is synthetic and attributed under CC BY 4.0.

## Serving

`lib/inference.mjs` loads the frozen bundle. It validates six sensor/product inputs, imputes missing values, computes derived features and reproduces logistic/forest inference. Forest inputs use float32 to match sklearn. It also validates CSV rows and computes policy metrics using frozen test scores.

`api/index.mjs` exposes health/report GET routes and prediction/batch/evaluation POST routes. Requests are bounded to 100 KB and unsupported origins are rejected. This is not authentication or a full abuse-prevention system. Vercel bundles the model artifact with the function; only the public directory is served statically.

## Interface and storage

`public/index.html` contains the responsive interface and client logic. It uses same-origin fetch for inference and evaluation, renders evidence and exports JSON through browser downloads. Named policy snapshots are browser-local; there is no server experiment database or cross-device sync. `server.mjs` serves the same application locally.

## Verification

`test.mjs` has 13 checks, including 80 Python-to-JavaScript golden predictions, frozen metric reproduction, split disjointness, threshold behavior, missing/invalid input, CSV validation and HTTP errors. These checks passed before publication. Browser QA is a separate unfinished milestone and is not implied by unit-test success.
