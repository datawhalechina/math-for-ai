# AGENTS.md

## Project overview

Chinese translation of the [MML book](https://mml-book.github.io/), published by Datawhale. Content is served as a static Docsify site.

## Commands

```bash
docsify serve docs    # local dev server (requires docsify-cli)
```

No build step, no linting, no tests, no CI.

## Content structure

- `docs/chN/chN.md` — chapter overview (may contain translator notes)
- `docs/chN/chN_X.md` — section X of chapter N
- `docs/chN/chN_习题.md` — chapter exercises
- `docs/chN/attachments/` — chapter-local images (plus some in `docs/attachments/`)
- `docs/_sidebar.md` — full table of contents; keep in sync when adding/moving sections
- `docs/index.html` — Docsify + KaTeX + plugin config; math delimiters are `$...$` (inline) and `$$...$$` (display)

## Writing conventions

- All content is in Chinese (Simplified)
- Math uses LaTeX inside `$...$` (inline) or `$$...$$` (display), rendered by KaTeX via docsify-katex
- Images reference relative paths: `./attachments/` or `../attachments/`

## Translation status (from README)

- Chapter 11 is in progress
- Images, equations, and callout formatting pending across all chapters
- Cross-reference hyperlinks pending

## License

CC BY-NC-SA 4.0
