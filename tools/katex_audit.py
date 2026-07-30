#!/usr/bin/env python3
"""Report every math block in docs/ that KaTeX cannot render.

The site renders math with docsify-katex, so anything KaTeX rejects shows up as
a red error box for readers. This extracts each $$...$$ and $...$ block and
checks it against the real KaTeX (via tools/katex_check.js).

Usage: katex_audit.py [--inline] [--quiet]
"""
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import mdio

DOCS = Path("docs")
CHECKER = Path(__file__).parent / "katex_check.js"
NODE_MODULES = "/tmp/katextest/node_modules"

DISPLAY = re.compile(r"\$\$(.+?)\$\$", re.S)
# inline $...$ that is not part of a $$ delimiter
INLINE = re.compile(r"(?<![$\\])\$(?!\$)((?:[^$\\\n]|\\.)+?)\$(?!\$)")


def blocks(want_inline: bool):
    for md in sorted(DOCS.rglob("*.md")):
        text = mdio.read(md)
        spans = []
        for m in DISPLAY.finditer(text):
            spans.append(m.span())
            yield md, text, m, True
        if not want_inline:
            continue
        for m in INLINE.finditer(text):
            if any(a <= m.start() < b for a, b in spans):
                continue
            yield md, text, m, False


def main(argv: list[str]) -> int:
    want_inline = "--inline" in argv
    items, meta = [], []
    for md, text, m, display in blocks(want_inline):
        items.append({"id": len(items), "tex": m.group(1), "display": display})
        meta.append((md, text[: m.start()].count("\n") + 1, display, m.group(1)))

    proc = subprocess.run(
        ["node", str(CHECKER)],
        input=json.dumps(items),
        capture_output=True,
        text=True,
        env={"NODE_PATH": NODE_MODULES, "PATH": "/usr/bin:/bin:/opt/homebrew/bin"},
    )
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return 2
    failures = json.loads(proc.stdout)

    kinds = Counter()
    by_file = Counter()
    for f in failures:
        md, line, display, tex = meta[f["id"]]
        kind = f["error"].replace("KaTeX parse error: ", "")
        kind = re.sub(r"\s*at position \d+.*", "", kind)
        kind = re.sub(r"'[^']*'", "'...'", kind)
        kinds[kind] += 1
        by_file[str(md)] += 1
        if "--quiet" not in argv:
            print(f"  {md}:{line} [{'display' if display else 'inline'}] {kind}")
            print(f"      {tex.strip()[:150]}")

    scope = "display+inline" if want_inline else "display"
    print(f"\n{len(items)} {scope} math blocks checked, {len(failures)} fail to render")
    for kind, n in kinds.most_common():
        print(f"  {n:4}  {kind}")
    if by_file:
        print("\nfiles affected:", len(by_file))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
