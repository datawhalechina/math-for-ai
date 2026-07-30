"""Read/write markdown without disturbing existing line endings.

Path.read_text() translates CRLF to LF and write_text() writes LF back, which
turns a one-line edit into a whole-file diff for the CRLF files in docs/.
"""
from pathlib import Path


def read(path: Path) -> str:
    """Read text with line endings left exactly as they are on disk."""
    with open(path, encoding="utf-8", newline="") as fh:
        return fh.read()


def write(path: Path, text: str) -> None:
    """Write text verbatim, performing no newline translation."""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def newline(text: str) -> str:
    """The line ending this text uses, for any content we add to it."""
    return "\r\n" if "\r\n" in text else "\n"
