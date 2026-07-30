#!/usr/bin/env python3
"""Rewrite docsify image paths so they resolve, and report the ones that cannot.

docs/index.html does not set `relativePath`, so docsify resolves every relative
image URL against the site root (docs/), not against the directory of the
markdown file that contains it. Paths written as if they were file-relative
("../attachments/x.png" from docs/ch8/, or "./attachments/x.png" for an image
that really lives in docs/ch9/attachments/) therefore 404.

For each reference this looks for the image under the locations the repo actually
uses -- docs/<chapter>/attachments/ and docs/attachments/ -- and rewrites the
path to the root-relative form that exists on disk. References already resolving
are left alone; references whose target is missing everywhere are reported.

Usage: fix_image_paths.py [--apply]
"""
import re
import sys
import urllib.parse
from pathlib import Path

import mdio

DOCS = Path("docs")
# group 1 = <img src="...">, group 2 = ![alt](...)
REF = re.compile(r'(?:<img[^>]*?src=["\']([^"\']+)["\']|!\[[^\]]*\]\(([^)\s]+)(?:\s+"[^"]*")?\))')


def resolves(path: str) -> bool:
    """Would docsify find this root-relative path on disk?"""
    return (DOCS / urllib.parse.unquote(path)).is_file()


def candidates(md: Path, src: str) -> list[str]:
    """Root-relative paths to try for `src`, most specific first."""
    name = src.rsplit("/", 1)[-1]
    chapter = md.parent.relative_to(DOCS).as_posix()
    out = [] if chapter == "." else [f"{chapter}/attachments/{name}"]
    return [*out, f"attachments/{name}", name]


def rewrite(md: Path) -> tuple[str, int, int, list[str]]:
    text, fixed, kept, missing = mdio.read(md), 0, 0, []
    out, end = [], 0
    for m in REF.finditer(text):
        group = 1 if m.group(1) else 2
        src = m.group(group)
        if src.startswith(("http", "data:", "#")) or resolves(src.removeprefix("/")):
            kept += not src.startswith(("http", "data:", "#"))
            continue
        target = next((c for c in candidates(md, src) if resolves(c)), None)
        if target is None:
            missing.append(f"  MISSING  {md}: {src}")
            continue
        start, stop = m.span(group)
        out.append(text[end:start] + target)
        end = stop
        fixed += 1
        print(f"  {md}: {src} -> {target}")
    out.append(text[end:])
    return "".join(out), fixed, kept, missing


def main(argv: list[str]) -> int:
    apply = "--apply" in argv
    fixed = kept = 0
    problems: list[str] = []
    for md in sorted(DOCS.rglob("*.md")):
        text, f, k, miss = rewrite(md)
        fixed, kept = fixed + f, kept + k
        problems += miss
        if f and apply:
            mdio.write(md, text)
    print(f"\n{fixed} rewritten, {kept} already correct, {len(problems)} target missing")
    if problems:
        print(*problems, sep="\n")
    if not apply:
        print("\n(dry run -- pass --apply to write)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
