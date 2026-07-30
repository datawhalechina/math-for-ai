#!/usr/bin/env python3
"""Repair the encoding damage introduced by the 2026-07-17 formatting commits.

Damage taxonomy:
  A) whole file re-encoded as GB18030 by a Windows tool, which silently dropped
     (or best-fit substituted) every character GBK cannot represent: ˜ ⊆ − ö ∼
  B) UTF-8 processed as GBK pair-by-pair: pairs that map to a GBK character were
     passed through unchanged, unmappable pairs collapsed into a single '?'.
     Each site therefore loses one CJK char (and possibly the ASCII char that
     shared its byte pair), while the surviving lead bytes still identify it.
  C) same as B, already re-encoded to UTF-8, so each site reads U+FFFD
     (optionally followed by '?')                        -> no surviving prefix

Lost characters are restored from the last known-good UTF-8 revision of the same
file: the formatting commits only rewrote ASCII structure (<center><img>,
\\tag{...}, [第x.y节](...) links), so a character-level alignment attributes every
CJK difference to the corruption. Holes inside text those commits *created* are
resolved by a second pass that looks the following context up in the reference.
Every restored character is validated against the surviving byte prefix.
"""
import difflib
import subprocess
import sys
from pathlib import Path

PUA_BASE = 0xE000  # one private-use char per hole, so each keeps its own prefix

# Holes whose surroundings the formatting commits rewrote beyond automatic
# recognition. Each replacement was read off the reference revision by hand:
#   ch10_5 @ f3e7658: "...\mathbb{R}^{N\times D}$，得到" / "使它们的范数为1。"
#   ch9_4  @ 0b42408: "...最大似然解被解释为一个投影。*"
MANUAL: dict[tuple[str, str], str] = {
    ("docs/ch10/ch10_5.md", "\\times D}$，得"): "到",
    ("docs/ch10/ch10_5.md", "使它们的范数为"): "。",
    ("docs/ch9/ch9_4.md", "被解释为一个投影"): "。",
}


def git_show(rev: str, path: str) -> bytes:
    r = subprocess.run(["git", "show", f"{rev}:{path}"], capture_output=True)
    if r.returncode:
        raise RuntimeError(f"git show {rev}:{path} failed: {r.stderr.decode()}")
    return r.stdout


def classify(data: bytes) -> str:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    else:
        return "C" if "\ufffd" in text else "clean"
    try:
        data.decode("gb18030")
    except UnicodeDecodeError:
        return "B"
    return "A"  # valid GB18030 but not UTF-8 -> genuinely GB18030 encoded


class Holes:
    """Allocates a unique placeholder char per lost character."""

    def __init__(self) -> None:
        self.prefix: dict[str, bytes | None] = {}

    def new(self, prefix: bytes | None) -> str:
        ch = chr(PUA_BASE + len(self.prefix))
        self.prefix[ch] = prefix
        return ch

    def __contains__(self, ch: str) -> bool:
        return ch in self.prefix

    def any_in(self, chunk: str) -> bool:
        return any(c in self.prefix for c in chunk)

    def count_in(self, chunk: str) -> int:
        return sum(c in self.prefix for c in chunk)

    def accepts(self, ch: str, candidate: str) -> bool:
        """Could `candidate` be the first character lost at hole `ch`?

        A hole always starts with the multi-byte character whose byte pair GBK
        could not map; any ASCII that followed it in that pair comes after.
        """
        if candidate.isascii():
            return False
        prefix = self.prefix[ch]
        return prefix is None or candidate.encode("utf-8").startswith(prefix)


def decode_type_b(data: bytes, holes: Holes) -> str:
    out, i, n = [], 0, len(data)
    while i < n:
        c = data[i]
        if c < 0x80:
            out.append(chr(c))
            i += 1
            continue
        length = 2 if 0xC2 <= c <= 0xDF else 3 if 0xE0 <= c <= 0xEF else 4 if 0xF0 <= c <= 0xF4 else 0
        if length:
            seq = data[i : i + length]
            if len(seq) == length:
                try:
                    out.append(seq.decode("utf-8"))
                    i += length
                    continue
                except UnicodeDecodeError:
                    pass
            body = data[i : i + length - 1]
            if len(body) == length - 1 and all(0x80 <= x <= 0xBF for x in body[1:]):
                nxt = data[i + length - 1 : i + length]
                if nxt == b"?":  # trailing byte replaced by '?'
                    out.append(holes.new(body))
                    i += length
                    continue
                if not nxt or not 0x80 <= nxt[0] <= 0xBF:  # trailing byte dropped
                    out.append(holes.new(body))
                    i += length - 1
                    continue
        out.append(holes.new(None))
        i += 1
    return "".join(out)


def decode_type_c(data: bytes, holes: Holes) -> str:
    out, text, i = [], data.decode("utf-8"), 0
    while i < len(text):
        if text[i] == "\ufffd":
            out.append(holes.new(None))
            i += 2 if text[i + 1 : i + 2] == "?" else 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def repair(damaged: str, clean: str, holes: Holes, path: str) -> tuple[str, int, list[str]]:
    """Fill every hole in place from `clean`.

    Only the lost characters are taken from the reference; all surrounding text
    comes from the damaged file, so the formatting commits' ASCII edits (<center>,
    </center>, \\tag{...}, links) survive untouched.
    """
    text, filled = fill_from_alignment(damaged, clean, holes, 0)
    return fill_from_context(text, clean, holes, filled, path)


def fill_from_alignment(text: str, clean: str, holes: Holes, filled: int) -> tuple[str, int]:
    """Fill holes that a diff pins between two exactly adjacent matching blocks.

    If the reference matches the damaged text right up to the hole and again right
    after it, whatever sits in between in the reference is exactly what was lost.
    """
    progress = True
    while progress:
        progress = False
        blocks = difflib.SequenceMatcher(None, text, clean, autojunk=False).get_matching_blocks()
        ends = {b.a + b.size: b.b + b.size for b in blocks if b.size}
        starts = {b.a: b.b for b in blocks if b.size}
        fixes = []
        for i, ch in enumerate(text):
            if ch not in holes or i not in ends or i + 1 not in starts:
                continue
            seg = clean[ends[i] : starts[i + 1]]
            if 0 < len(seg) <= 3 and holes.accepts(ch, seg[0]) and seg[1:].isascii():
                fixes.append((i, seg))
        for i, seg in reversed(fixes):
            text = text[:i] + seg + text[i + 1 :]
        filled += len(fixes)
        progress = bool(fixes)
    return text, filled


def locate(text: str, clean: str, i: int, holes: Holes) -> int:
    """Index in `clean` of the character right after the hole at text[i], or -1."""
    for width in range(30, 5, -1):  # prefer the trailing context
        suffix = text[i + 1 : i + 1 + width]
        if len(suffix) == width and not holes.any_in(suffix) and clean.count(suffix) == 1:
            return clean.index(suffix)
    for width in range(30, 5, -1):  # fall back to the leading context
        prefix = text[max(0, i - width) : i]
        if len(prefix) == width and not holes.any_in(prefix) and clean.count(prefix) == 1:
            return clean.index(prefix) + width + 1
    return -1


def fill_from_context(text: str, clean: str, holes: Holes, filled: int, path: str = "") -> tuple[str, int, list[str]]:
    """Replace each hole with the character(s) it swallowed, found via its context.

    A hole may stand for the CJK character *plus* trailing ASCII characters that
    shared its corrupted byte pair (e.g. "图 9.1" and "图9" both collapse to one).
    Repeated passes let a resolved hole unblock the context of its neighbours.
    """
    progress = True
    while progress:
        progress = False
        i = 0
        while i < len(text):
            if text[i] not in holes:
                i += 1
                continue
            j, fix = locate(text, clean, i, holes), None
            for k in range(1, 4):
                if 0 <= j - k and holes.accepts(text[i], clean[j - k]) and clean[j - k + 1 : j].isascii():
                    fix = clean[j - k : j]
                    break
            if fix is None:
                fix = next(
                    (v for (p, ctx), v in MANUAL.items() if p == path and text[:i].endswith(ctx)),
                    None,
                )
            if fix is None or not holes.accepts(text[i], fix[0]):
                i += 1
                continue
            text = text[:i] + fix + text[i + 1 :]
            filled, progress = filled + 1, True
            i += len(fix)
    unresolved = [f"{holes.prefix[c]!r} at ...{text[max(0, k - 20) : k]!r}" for k, c in enumerate(text) if c in holes]
    return text, filled, unresolved


def gbk_lossy(ch: str) -> bool:
    """True if GBK cannot represent `ch`, i.e. re-encoding would corrupt it."""
    try:
        ch.encode("gbk")
    except UnicodeEncodeError:
        return True
    return False


def restore_gbk_dropouts(text: str, clean: str) -> tuple[str, int]:
    """Put back characters that the GBK round-trip dropped, '?'-ed or best-fit replaced."""
    out, restored = [], 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, text, clean, autojunk=False).get_opcodes():
        nchunk, cchunk = text[i1:i2], clean[j1:j2]
        lost = [c for c in cchunk if c != "\ufeff"]  # never restore a BOM
        if (
            tag != "equal"
            and lost
            and len(cchunk) <= 4
            and all(gbk_lossy(c) for c in lost)
            and all(c == "?" or not c.isascii() for c in nchunk)
        ):
            out.append("".join(lost))
            restored += len(lost)
        else:
            out.append(nchunk)
    return "".join(out), restored


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: fix_encoding.py <path>=<reference-rev> ...", file=sys.stderr)
        return 2
    exit_code = 0
    for spec in argv[1:]:
        path_s, _, ref = spec.partition("=")
        path = Path(path_s)
        data = path.read_bytes()
        kind = classify(data)
        if kind == "clean":
            print(f"  skip     {path_s} (already clean UTF-8)")
            continue
        if kind == "A":
            text, restored = restore_gbk_dropouts(data.decode("gb18030"), git_show(ref, path_s).decode("utf-8"))
            path.write_text(text, encoding="utf-8")
            print(f"  GB18030  {path_s} -> UTF-8 ({restored} non-GBK chars restored from {ref})")
            continue
        holes = Holes()
        damaged = decode_type_b(data, holes) if kind == "B" else decode_type_c(data, holes)
        fixed, filled, unresolved = repair(damaged, git_show(ref, path_s).decode("utf-8"), holes, path_s)
        if unresolved:
            print(f"  FAILED   {path_s}: {len(unresolved)} unresolved: {unresolved[:3]}")
            exit_code = 1
            continue
        path.write_text(fixed, encoding="utf-8")
        print(f"  type-{kind}   {path_s} -> UTF-8 ({filled}/{len(holes.prefix)} chars restored from {ref})")
    return exit_code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
