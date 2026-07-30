#!/usr/bin/env python3
"""Give each numbered equation row its own \\tag{} so KaTeX can render it.

An upstream commit moved standalone "(10.24)" equation numbers into \\tag{}, but
when an equation had several numbered rows every tag was appended to the end of
the one block:

    $$\\begin{aligned}a&=1\\\\b&=2\\end{aligned} \\tag{4.74} \\tag{4.75}$$

KaTeX rejects more than one \\tag per block ("Multiple \\tag"), so these render
as a red error box instead of an equation. KaTeX also only honours \\tag inside
`align`, not `aligned`. This rewrites such blocks to one tag per row:

    $$\\begin{align}a&=1\\tag{4.74}\\\\b&=2\\tag{4.75}\\end{align}$$

Only blocks whose row count matches the tag count are touched, and every
rewrite is verified against the real KaTeX before it is written. Anything else
is reported for a human to resolve.

Usage: fix_equation_tags.py [--apply]
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import mdio

DOCS = Path("docs")
CHECKER = Path(__file__).parent / "katex_check.js"
NODE_ENV = {"NODE_PATH": "/tmp/katextest/node_modules", "PATH": "/usr/bin:/bin:/opt/homebrew/bin"}

DISPLAY = re.compile(r"\$\$(.+?)\$\$", re.S)
TAG = re.compile(r"\\tag\{([^}]*)\}")
ENV = re.compile(r"\\(begin|end)\s*\{([^}]*)\}")
ROW_ENVS = {"aligned", "align", "align*", "gather", "gathered", "array", "cases", "split"}


def split_rows(s: str) -> list[str]:
    """Split on \\\\ that is at brace depth 0 and not inside a nested environment."""
    rows, cur, i, depth, envs = [], [], 0, 0, 0
    while i < len(s):
        if s[i] == "\\":
            if s[i : i + 2] == "\\\\":
                if depth == 0 and envs == 0:
                    rows.append("".join(cur))
                    cur, i = [], i + 2
                    continue
                cur.append("\\\\")
                i += 2
                continue
            m = ENV.match(s, i) or re.compile(r"\\[a-zA-Z]+|\\.", re.S).match(s, i)
            if m and m.re is ENV:
                envs += 1 if m.group(1) == "begin" else -1
            cur.append(m.group(0))
            i = m.end()
            continue
        depth += (s[i] == "{") - (s[i] == "}")
        cur.append(s[i])
        i += 1
    rows.append("".join(cur))
    return rows


def unwrap(core: str) -> tuple[str, str | None]:
    """If `core` is exactly one row environment, return its inner content."""
    s = core.strip()
    m = re.fullmatch(r"\\begin\s*\{([^}]*)\}(?:\{[^}]*\})?(.*)\\end\s*\{\1\}", s, re.S)
    if m and m.group(1) in ROW_ENVS:
        return m.group(2), m.group(1)
    return core, None


def katex_ok(*tex: str) -> list[bool]:
    """Which of these render under the real KaTeX?"""
    proc = subprocess.run(
        ["node", str(CHECKER)],
        input=json.dumps([{"id": i, "tex": t, "display": True} for i, t in enumerate(tex)]),
        capture_output=True, text=True, env=NODE_ENV,
    )
    proc.check_returncode()
    bad = {f["id"] for f in json.loads(proc.stdout)}
    return [i not in bad for i in range(len(tex))]


def rebuild(body: str, eol: str = "\n") -> str | None:
    """Rewrite a multi-tag block as an `align` with one tag per row, or None."""
    tags = TAG.findall(body)
    if len(tags) < 2:
        return None
    inner, _ = unwrap(TAG.sub("", body))
    rows = [r for r in split_rows(inner) if r.strip()]
    if len(rows) != len(tags):
        return None
    body = f"\\\\{eol}".join(f"{r.strip()} \\tag{{{t}}}" for r, t in zip(rows, tags))
    return f"\\begin{{align}}{eol}{body}{eol}\\end{{align}}"


def main(argv: list[str]) -> int:
    apply = "--apply" in argv
    fixed, skipped, already = 0, [], 0
    for md in sorted(DOCS.rglob("*.md")):
        text = mdio.read(md)
        eol = mdio.newline(text)
        multi = [m for m in DISPLAY.finditer(text) if len(TAG.findall(m.group(1))) >= 2]
        if not multi:
            continue
        # only touch blocks readers actually see break; many are already correct
        renders = katex_ok(*(m.group(1) for m in multi))
        out, end = [], 0
        for m, fine in zip(multi, renders):
            if fine:
                already += 1
                continue
            line = text[: m.start()].count("\n") + 1
            new = rebuild(m.group(1), eol)
            if new is None or not katex_ok(new)[0]:
                why = "row/tag count mismatch" if new is None else "still fails KaTeX"
                skipped.append(f"  SKIP {md}:{line} ({why}): {','.join(TAG.findall(m.group(1)))}")
                continue
            out.append(text[end : m.start(1)] + new)
            end = m.end(1)
            fixed += 1
            print(f"  {md}:{line} -> align with {len(TAG.findall(new))} tags")
        out.append(text[end:])
        if end and apply:
            mdio.write(md, "".join(out))
    print(f"\n{fixed} blocks rewritten, {already} already rendering, {len(skipped)} left for review")
    print(*skipped, sep="\n") if skipped else None
    if not apply:
        print("\n(dry run -- pass --apply to write)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
