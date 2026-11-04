# Understand FailureScope

## What is the model doing?

It takes measured air/process temperatures, rotational speed, torque, tool wear and product type. It returns a score for the failure class. If that score crosses your selected threshold, the UI flags the observation for review. The score comes from learned weights/trees, not an LLM API or a hand-written failure rule.

The data is synthetic: its labels were generated using physical-style rules and randomness. That makes it useful for learning, but a good benchmark result does not prove industrial usefulness.

## The three parts

1. **Training:** Python reads the dataset and divides it into train, validation and test rows. It learns preprocessing and two classifiers using training rows. Validation chooses the model and threshold. Test reports the final performance.
2. **Backend:** the trained weights and trees are exported to JSON. Node reads them and calculates a score for each API request. No Python installation is required on Vercel.
3. **Frontend:** the browser gathers inputs, calls the API, renders results and offers JSON exports. Named experiments are stored in this browser, not a server database.

## Demonstrate it in five minutes

Open Prediction lab. Choose a low-score benchmark example and analyze it, then choose the high-score example. Explain that these are selected benchmark rows with known labels, not fresh real-world validation. Switch to CSV batch analysis, download the example CSV, paste or upload it and analyze. One row is missing air temperature; show the imputation warning. Change a type to X to demonstrate a row-level error.

Open Model evaluation. At the frozen forest threshold 0.46, 58 out of 68 failures are caught, nine healthy rows are flagged and ten failures are missed. Move the slider and explain the tradeoff. Compare logistic regression and the forest using AP, not just accuracy. Show the model card's lower ordered-row results. Finally save and restore a policy experiment.

## Concepts to defend

- **Classification vs forecasting:** this labels a recorded state. There are no future-horizon labels or remaining-life estimates.
- **Imbalance:** failure labels are rare. Predicting healthy every time looks accurate but misses all failures. Class weighting changes training costs; it does not manufacture more evidence.
- **Precision:** of flagged observations, what fraction were failures? **Recall:** of failures, what fraction were caught?
- **Average precision:** a ranking summary across score thresholds. It differs from ROC AUC and from trapezoidal PR area.
- **F2:** weights recall more strongly than precision. The validation F2 optimum selected 0.46; the test set did not select it.
- **Data leakage:** IDs and failure-mode target columns are excluded. Imputation and scaling use training rows only. The feature-engineering formulas use observed sensors, never target values.
- **Random vs ordered splits:** neighboring operating regimes may be similar. A random holdout is optimistic evidence for deployment under drift. Ordered rows give a stress check but cannot prove time-generalization without real timestamps.
- **Interpretability:** permutation importance measures how shuffling a feature changes validation AP. Correlation can hide importance. It is neither a causal effect nor a per-case explanation.
- **Calibration:** a score of 0.8 is not established to mean an 80% real-world failure rate. We report Brier score without claiming calibration.
- **Model serving:** Python and JavaScript inference must agree. Golden tests check 40 rows on each model, including sklearn tree float32 behavior.

## Running locally

Install Node 22 or newer. Open the project folder in a terminal. Run `npm ci`, `npm test`, then `npm start`. Open http://localhost:3000. Stop with Ctrl+C. This needs no API keys or accounts. Training requires Python 3.12 and the pinned packages; create a virtual environment, install requirements and run `python train.py`.

## Make an actual learning contribution

Read `train.py` and reproduce the model bundle first. Implement a new training-only feature or compare a new classifier using validation data, keeping the original held-out test untouched until final comparison. Once you have already explored that test, treat it as development data for future iterations and reserve new independent data for a fresh claim. Keep a real note of your reasoning and failed experiments.

For UI changes, edit `public/index.html`. For inference and CSV validation, edit `lib/inference.mjs`. For HTTP handling, edit `api/index.mjs`. Retraining changes `model-bundle.json` and possibly split indices; preserve version names or increment them deliberately so saved snapshots remain interpretable.

## Interview honesty

Explain the assistance used and the parts you understand and modified. Do not invent users, production equipment, downtime savings or measured safety benefits. A clear explanation of limitations is stronger than a claim you cannot reproduce.
