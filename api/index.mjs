import { predict, batch, evaluate, report } from "../lib/inference.mjs";
export default {
  async fetch(req) {
    const headers = {
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff",
    };
    const json = (body, status = 200) =>
      Response.json(body, { status, headers });
    const path = new URL(req.url).pathname;
    if (req.method === "GET") {
      if (path === "/api/health")
        return json({ status: "ok", app: "FailureScope", version: "1.0.0" });
      if (path === "/api/report") return json(report());
      return json({ error: "Not found" }, 404);
    }
    if (req.method !== "POST")
      return json({ error: "Method not allowed" }, 405);
    if (
      req.headers.get("origin") &&
      req.headers.get("origin") !== new URL(req.url).origin
    )
      return json({ error: "Foreign origin rejected" }, 403);
    if (!["/api/predict", "/api/batch", "/api/evaluate"].includes(path))
      return json({ error: "Not found" }, 404);
    try {
      const reader = req.body?.getReader();
      if (!reader) throw Error("JSON body required");
      let size = 0,
        text = "";
      const decoder = new TextDecoder();
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        size += value.length;
        if (size > 100000) {
          await reader.cancel();
          return json({ error: "Request exceeds 100 KB limit" }, 413);
        }
        text += decoder.decode(value, { stream: true });
      }
      text += decoder.decode();
      const input = JSON.parse(text);
      if (!input || Array.isArray(input) || typeof input !== "object")
        throw Error("JSON object required");
      return json(
        path === "/api/predict"
          ? predict(input.row, input)
          : path === "/api/batch"
            ? batch(input.csv, input)
            : evaluate(input),
      );
    } catch (e) {
      return json({ error: e.message }, 400);
    }
  },
};
