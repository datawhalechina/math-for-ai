#!/usr/bin/env node
// Validate every math block in docs/ with the same KaTeX the site loads.
// Reads a JSON array of {id, tex, display} on stdin, writes JSON array of
// {id, error} for the ones that fail to render.
const katex = require("katex");

let raw = "";
process.stdin.on("data", (c) => (raw += c));
process.stdin.on("end", () => {
  const out = [];
  for (const { id, tex, display } of JSON.parse(raw)) {
    try {
      katex.renderToString(tex, {
        displayMode: display,
        throwOnError: true,
        strict: false,
        macros: { "\\R": "\\mathbb{R}" }, // matches docs/index.html
      });
    } catch (e) {
      out.push({ id, error: e.message.split("\n")[0] });
    }
  }
  process.stdout.write(JSON.stringify(out));
});
