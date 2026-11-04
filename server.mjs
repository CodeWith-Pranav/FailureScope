import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import api from "./api/index.mjs";
const assets = {
  "/": ["index.html", "text/html"],
  "/app.js": ["app.js", "text/javascript"],
  "/style.css": ["style.css", "text/css"],
};
createServer(async (req, res) => {
  try {
    const url = new URL(req.url, "http://" + req.headers.host);
    if (url.pathname.startsWith("/api/")) {
      let body = "",
        size = 0;
      for await (const c of req) {
        size += c.length;
        if (size > 8000) {
          res.writeHead(413);
          res.end();
          return;
        }
        body += c;
      }
      const response = await api.fetch(
        new Request(url, {
          method: req.method,
          headers: req.headers,
          ...(req.method === "GET" ? {} : { body }),
        }),
      );
      res.writeHead(response.status, Object.fromEntries(response.headers));
      res.end(await response.text());
      return;
    }
    const asset = assets[url.pathname];
    if (!asset) {
      res.writeHead(404);
      res.end();
      return;
    }
    res.setHeader("Content-Type", asset[1]);
    res.end(await readFile(new URL("./public/" + asset[0], import.meta.url)));
  } catch {
    res.writeHead(500);
    res.end("Server error");
  }
}).listen(Number(process.env.PORT || 3000), "127.0.0.1", () =>
  console.log("FailureScope on http://localhost:" + (process.env.PORT || 3000)),
);
