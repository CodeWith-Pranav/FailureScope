import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  bundle,
  prepare,
  predict,
  evaluate,
  batch,
  report,
  settings,
} from "./lib/inference.mjs";
import api from "./api/index.mjs";
const valid = {
  type: "L",
  air: 300,
  process: 310,
  speed: 1500,
  torque: 40,
  wear: 100,
};
test("exported models match 40 Python predictions each", () => {
  for (const g of bundle.parity)
    for (const model of Object.keys(bundle.models))
      assert.ok(
        Math.abs(predict(g.row, { model }).score - g.scores[model]) < 1e-12,
        model,
      );
});
test("frozen policy reproduces Python test metrics", () => {
  for (const model of Object.keys(bundle.models)) {
    const r = evaluate({ model, threshold: bundle.threshold }),
      e = bundle.report[model].test;
    for (const k of [
      "tp",
      "tn",
      "fp",
      "fn",
      "precision",
      "recall",
      "f2",
      "accuracy",
    ])
      assert.ok(Math.abs(r[k] - e[k]) < 1e-12, k);
    assert.equal(r.tp + r.fn, 68);
    assert.equal(r.tp + r.tn + r.fp + r.fn, 2000);
  }
});
test("higher threshold never increases recall or workload", () => {
  let last = { flagged: Infinity, recall: Infinity };
  for (let i = 1; i <= 99; i++) {
    const m = evaluate({ threshold: i / 100 });
    assert.ok(m.flagged <= last.flagged && m.recall <= last.recall);
    last = m;
  }
});
test("missing values use training medians with warning", () => {
  const r = predict({ ...valid, air: null });
  assert.equal(r.input.air, bundle.medians[0]);
  assert.ok(r.warnings.some((x) => x.includes("median")));
});
test("derived features and outside-training warnings are explicit", () => {
  const r = prepare({ ...valid, air: 350 });
  assert.equal(r.values[5], -40);
  assert.ok(r.warnings.some((x) => x.includes("outside training range")));
  assert.ok(r.warnings.some((x) => x.includes("below air")));
  assert.ok(Math.abs(r.values[6] - (1500 * 40 * 2 * Math.PI) / 60) < 1e-10);
});
test("invalid inputs, label leakage and unknown models rejected", () => {
  for (const row of [
    { ...valid, speed: NaN },
    { ...valid, wear: -1 },
    { ...valid, type: "X" },
    { ...valid, MachineFailure: 1 },
    { type: "L" },
    null,
  ])
    assert.throws(() => predict(row));
  for (const c of [
    { model: "fake" },
    { threshold: 1 },
    { threshold: "0.5" },
    { threshold: NaN },
  ])
    assert.throws(() => settings(c));
});
test("batch handles row errors and quoted CRLF CSV", () => {
  const r = batch(
    '"type",air,process,speed,torque,wear\r\nL,300,310,1500,40,100\r\nX,300,310,1500,40,100\r\nM,,310,1500,40,100',
  );
  assert.equal(r.summary.valid, 2);
  assert.equal(r.summary.invalid, 1);
  assert.equal(r.rows[1].rowNumber, 3);
  assert.ok(r.rows[1].error);
  assert.ok(r.rows[2].warnings.length);
});
test("malformed CSV, unexpected headers and too many rows rejected", () => {
  for (const csv of [
    "",
    "type,air\nL,300",
    'type,air,process,speed,torque,wear\n"L,300,310,1500,40,100',
    "type,air,process,speed,torque,wear\n" +
      Array(101).fill("L,300,310,1500,40,100").join("\n"),
  ])
    assert.throws(() => batch(csv));
});
test("cost uses explicit weights and validates them", () => {
  const r = evaluate({ falseAlarmCost: 3, missCost: 10 });
  assert.equal(r.cost, r.fp * 3 + r.fn * 10);
  assert.throws(() => evaluate({ missCost: -1 }));
});
test("split manifests are disjoint and cover 10000 rows", () => {
  const s = JSON.parse(
    readFileSync(new URL("./split-indices.json", import.meta.url)),
  );
  assert.equal(new Set([...s.train, ...s.validation, ...s.test]).size, 10000);
  assert.equal(s.train.length, 6000);
  assert.equal(s.validation.length, 2000);
  assert.equal(s.test.length, 2000);
});
test("public report excludes internal trees and held-out row labels", () => {
  const r = report();
  assert.equal(r.dataset.synthetic, true);
  assert.equal(r.curves["rf-v1"].length, 99);
  assert.equal(r.holdout, undefined);
  assert.equal(r.models["rf-v1"].trees, undefined);
});
test("API health, report and real model inference", async () => {
  const r = await api.fetch(
    new Request("https://example.com/api/predict", {
      method: "POST",
      body: JSON.stringify({ row: valid }),
    }),
  );
  assert.equal(r.status, 200);
  assert.equal((await r.json()).score, predict(valid).score);
  for (const path of ["report", "health"])
    assert.equal(
      (await api.fetch(new Request("https://example.com/api/" + path))).status,
      200,
    );
  assert.equal(
    (await api.fetch(new Request("https://example.com/api/unknown"))).status,
    404,
  );
});
test("API rejects bad JSON, origin, oversized body and unsupported methods", async () => {
  const request = (body, headers = {}) =>
    new Request("https://example.com/api/predict", {
      method: "POST",
      body,
      headers,
    });
  assert.equal((await api.fetch(request("{"))).status, 400);
  assert.equal(
    (await api.fetch(request("{}", { origin: "https://bad.com" }))).status,
    403,
  );
  assert.equal((await api.fetch(request(" ".repeat(100001)))).status, 413);
  assert.equal(
    (
      await api.fetch(
        new Request("https://example.com/api/report", { method: "DELETE" }),
      )
    ).status,
    405,
  );
});
