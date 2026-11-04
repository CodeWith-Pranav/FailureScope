import { readFileSync } from "node:fs";
export const bundle = JSON.parse(
  readFileSync(new URL("../model-bundle.json", import.meta.url), "utf8"),
);
export const numeric = ["air", "process", "speed", "torque", "wear"];
const bounds = {
  air: [250, 400],
  process: [250, 450],
  speed: [100, 10000],
  torque: [0, 200],
  wear: [0, 1000],
};
export function settings(input = {}) {
  const model = input.model ?? bundle.selected,
    threshold = input.threshold ?? bundle.threshold;
  if (!Object.hasOwn(bundle.models, model))
    throw Error("Unknown model version");
  if (
    typeof threshold !== "number" ||
    !Number.isFinite(threshold) ||
    threshold < 0.01 ||
    threshold > 0.99
  )
    throw Error("Threshold must be between 0.01 and 0.99");
  return { model, threshold };
}
export function prepare(row) {
  if (!row || typeof row !== "object" || Array.isArray(row))
    throw Error("Each row must be an object");
  for (const k of Object.keys(row))
    if (!["type", ...numeric].includes(k)) throw Error("Unknown column: " + k);
  const type = row.type ?? "L";
  if (!["H", "L", "M"].includes(type)) throw Error("Type must be H, L or M");
  const warnings = [],
    values = [];
  let observed = 0;
  if (row.type == null)
    warnings.push("Missing type imputed with training mode L");
  numeric.forEach((k, i) => {
    let v = row[k];
    if (v == null || v === "") {
      v = bundle.medians[i];
      warnings.push(k + " missing: training median used");
    } else {
      observed++;
      if (
        typeof v !== "number" ||
        !Number.isFinite(v) ||
        v < bounds[k][0] ||
        v > bounds[k][1]
      )
        throw Error(
          k + " must be a finite number between " + bounds[k].join(" and "),
        );
    }
    const r = bundle.ranges[k];
    if (v < r.min || v > r.max)
      warnings.push(k + " outside training range; score may be unreliable");
    values.push(v);
  });
  if (!observed) throw Error("Supply at least one measured numeric value");
  const [air, process, speed, torque, wear] = values;
  if (process < air)
    warnings.push(
      "Process temperature is below air temperature; verify units and sensor values",
    );
  return {
    values: [
      ...values,
      process - air,
      speed * torque * ((2 * Math.PI) / 60),
      wear * torque,
      ...["H", "L", "M"].map((t) => +(t === type)),
    ],
    warnings,
    input: {
      type,
      ...Object.fromEntries(numeric.map((k, i) => [k, values[i]])),
    },
  };
}
export function probability(values, model) {
  const m = bundle.models[model];
  if (!m) throw Error("Unknown model version");
  if (model === "lr-v1") {
    const z =
      m.intercept +
      values.reduce(
        (s, v, i) => s + ((v - m.mean[i]) / m.scale[i]) * m.weights[i],
        0,
      );
    return 1 / (1 + Math.exp(-z));
  }
  // sklearn's tree prediction converts input features to float32.
  const v = values.map(Math.fround);
  return (
    m.trees.reduce((sum, t) => {
      let i = 0;
      while (t.left[i] !== -1)
        i = v[t.feature[i]] <= t.threshold[i] ? t.left[i] : t.right[i];
      return sum + t.value[i];
    }, 0) / m.trees.length
  );
}
export function predict(row, config = {}) {
  const s = settings(config),
    p = prepare(row),
    score = probability(p.values, s.model);
  return {
    ...s,
    score,
    flagged: score >= s.threshold,
    input: p.input,
    warnings: p.warnings,
    modelVersion: s.model,
  };
}
export function evaluate(config = {}) {
  const s = settings(config);
  let tp = 0,
    tn = 0,
    fp = 0,
    fn = 0;
  for (const r of bundle.holdout) {
    const pred = r.scores[s.model] >= s.threshold;
    if (pred && r.actual) tp++;
    else if (pred) fp++;
    else if (r.actual) fn++;
    else tn++;
  }
  const precision = tp / (tp + fp) || 0,
    recall = tp / (tp + fn) || 0;
  const falseAlarmCost = config.falseAlarmCost ?? 1,
    missCost = config.missCost ?? 20;
  for (const c of [falseAlarmCost, missCost])
    if (!Number.isInteger(c) || c < 1 || c > 1000)
      throw Error("Costs must be integers from 1 to 1000");
  return {
    ...s,
    tp,
    tn,
    fp,
    fn,
    precision,
    recall,
    f2: (5 * precision * recall) / (4 * precision + recall) || 0,
    accuracy: (tp + tn) / bundle.holdout.length,
    flagged: tp + fp,
    cost: fp * falseAlarmCost + fn * missCost,
    falseAlarmCost,
    missCost,
    ...Object.fromEntries(
      ["averagePrecision", "rocAuc", "brier", "n", "positives"].map((k) => [
        k,
        bundle.report[s.model].test[k],
      ]),
    ),
  };
}
export function csvRows(text) {
  if (typeof text !== "string" || !text.trim()) throw Error("CSV is empty");
  let rows = [],
    row = [],
    field = "",
    quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '"') {
      if (quoted && text[i + 1] === '"') {
        field += '"';
        i++;
      } else if (!field || quoted) quoted = !quoted;
      else throw Error("Invalid CSV quote");
    } else if (c === "," && !quoted) {
      row.push(field);
      field = "";
    } else if ((c === "\n" || c === "\r") && !quoted) {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field);
      if (row.some((x) => x.trim())) rows.push(row);
      row = [];
      field = "";
    } else field += c;
  }
  if (quoted) throw Error("CSV contains an unclosed quote");
  row.push(field);
  if (row.some((x) => x.trim())) rows.push(row);
  const headers = rows.shift()?.map((s) => s.trim().replace(/^\uFEFF/, ""));
  if (
    !headers ||
    headers.length !== 6 ||
    new Set(headers).size !== 6 ||
    !["type", ...numeric].every((k) => headers.includes(k))
  )
    throw Error(
      "CSV headers must be exactly type,air,process,speed,torque,wear",
    );
  if (rows.length < 1 || rows.length > 100)
    throw Error("CSV must contain 1–100 data rows");
  return rows.map((values, i) => {
    if (values.length !== 6)
      return { rowNumber: i + 2, error: "Expected six columns" };
    const data = {};
    headers.forEach((k, j) => {
      const v = values[j].trim();
      data[k] = k === "type" ? v || null : v === "" ? null : Number(v);
    });
    return { rowNumber: i + 2, data };
  });
}
export function batch(text, config = {}) {
  const s = settings(config);
  const rows = csvRows(text).map((r) => {
    if (r.error) return r;
    try {
      return { rowNumber: r.rowNumber, ...predict(r.data, s) };
    } catch (e) {
      return { rowNumber: r.rowNumber, error: e.message };
    }
  });
  return {
    ...s,
    rows,
    summary: {
      total: rows.length,
      valid: rows.filter((r) => !r.error).length,
      invalid: rows.filter((r) => r.error).length,
      flagged: rows.filter((r) => r.flagged).length,
    },
  };
}
export function report() {
  return {
    version: bundle.version,
    dataset: bundle.dataset,
    selected: bundle.selected,
    threshold: bundle.threshold,
    models: bundle.report,
    chronological: bundle.chronological,
    ranges: bundle.ranges,
    samples: bundle.samples,
    runtime: bundle.runtime,
    features: bundle.featureNames,
    curves: Object.fromEntries(
      Object.keys(bundle.models).map((model) => [
        model,
        Array.from({ length: 99 }, (_, i) => {
          const m = evaluate({ model, threshold: (i + 1) / 100 });
          return {
            threshold: m.threshold,
            precision: m.precision,
            recall: m.recall,
          };
        }),
      ]),
    ),
  };
}
