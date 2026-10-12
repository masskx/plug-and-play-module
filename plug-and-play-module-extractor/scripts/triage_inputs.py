#!/usr/bin/env python3
"""Classify what the user supplied and choose the workflow mode.

Modes (see references/input_modes.md):

* ``pdf-plus-code``  paper (PDF/text/arXiv/URL) and code are both available
* ``pdf-only``       paper available, no code
* ``code-only``      code available, no paper
* ``title-only``     only a paper title / arXiv id / bare name

Inputs may be paths, URLs, arXiv ids, or a quoted title.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

ARXIV_ID = re.compile(r"^(?:arxiv:)?(\d{4}\.\d{4,5})(?:v\d+)?$", re.I)
ARXIV_URL = re.compile(r"arxiv\.org/(?:abs|pdf|html)/([\w.\-/]+?)(?:v\d+)?(?:\.pdf)?/?$", re.I)
GITHUB_URL = re.compile(r"^(?:https?://)?(?:www\.)?(github\.com|gitlab\.com|gitee\.com|huggingface\.co/(?!papers))/", re.I)
PAPER_URL = re.compile(r"(openreview\.net|aclanthology\.org|doi\.org|ieeexplore|dl\.acm\.org|proceedings\.|openaccess\.thecvf|huggingface\.co/papers|semanticscholar\.org|\.pdf$)", re.I)
CODE_SUFFIXES = {".py", ".ipynb"}
TEXT_SUFFIXES = {".txt", ".md", ".tex"}


@dataclass
class InputItem:
    raw: str
    kind: str  # pdf | paper-text | arxiv | paper-url | code-local | code-remote | title
    detail: str = ""


@dataclass
class Triage:
    source_mode: str
    items: list[InputItem] = field(default_factory=list)
    paper: str = "none"  # pdf | paper-text | arxiv | paper-url | title | none
    code: str = "none"  # local | remote | none
    title: str | None = None
    needs: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)


def looks_like_paper_text(path: Path) -> bool:
    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:6000]
    except OSError:
        return False
    return len(head) > 1500 or bool(re.search(r"\babstract\b", head, re.I))


def classify(raw: str) -> InputItem:
    value = raw.strip().strip("\"'")
    path = Path(value).expanduser()
    if path.exists():
        if path.is_dir():
            has_code = any(path.rglob("*.py")) or any(path.rglob("*.ipynb"))
            pdfs = sorted(path.glob("*.pdf"))
            if has_code:
                return InputItem(raw, "code-local", str(path))
            if pdfs:
                return InputItem(raw, "pdf", str(pdfs[0]))
            return InputItem(raw, "title", value)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return InputItem(raw, "pdf", str(path))
        if suffix in CODE_SUFFIXES:
            return InputItem(raw, "code-local", str(path))
        if suffix in TEXT_SUFFIXES:
            if looks_like_paper_text(path):
                return InputItem(raw, "paper-text", str(path))
            first = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
            return InputItem(raw, "title", first[0].strip() if first else value)
    match = ARXIV_ID.match(value)
    if match:
        return InputItem(raw, "arxiv", match.group(1))
    match = ARXIV_URL.search(value)
    if match:
        return InputItem(raw, "arxiv", match.group(1))
    if GITHUB_URL.match(value):
        return InputItem(raw, "code-remote", value)
    if re.match(r"^https?://", value, re.I):
        return InputItem(raw, "paper-url" if PAPER_URL.search(value) else "paper-url", value)
    return InputItem(raw, "title", value)


def triage(inputs: list[str]) -> Triage:
    items = [classify(raw) for raw in inputs if raw.strip()]
    papers = [item for item in items if item.kind in {"pdf", "paper-text", "arxiv", "paper-url"}]
    codes = [item for item in items if item.kind in {"code-local", "code-remote"}]
    titles = [item for item in items if item.kind == "title"]

    paper_kind = "none"
    for preferred in ("paper-text", "pdf", "arxiv", "paper-url"):
        if any(item.kind == preferred for item in papers):
            paper_kind = preferred
            break
    code_kind = "none"
    if any(item.kind == "code-local" for item in codes):
        code_kind = "local"
    elif codes:
        code_kind = "remote"
    title = titles[0].detail if titles else None

    needs: list[str] = []
    steps: list[str] = []
    if paper_kind != "none" and code_kind != "none":
        mode = "pdf-plus-code"
    elif paper_kind != "none":
        mode = "pdf-only"
    elif code_kind != "none":
        mode = "code-only"
    elif title:
        mode = "title-only"
        paper_kind = "title"
    else:
        mode = "none"

    first = lambda kinds: next((item for item in items if item.kind in kinds), None)  # noqa: E731
    if paper_kind == "pdf":
        item = first({"pdf"})
        steps.append(f"python scripts/extract_pdf_text.py {item.detail} --output paper.txt")
    elif paper_kind == "arxiv":
        item = first({"arxiv"})
        needs.append("fetch_arxiv_pdf")
        steps.append(f"python scripts/resolve_paper_by_title.py --arxiv-id {item.detail} --download-dir ./paper")
    elif paper_kind == "paper-url":
        needs.append("fetch_paper_url")
        steps.append("Fetch the paper PDF/HTML with WebFetch, save it, then extract the text.")
    if code_kind == "remote":
        needs.append("clone_repo")
        item = first({"code-remote"})
        steps.append(f"git clone --depth 1 {item.detail} ./code   # read-only: never run its install scripts or tests")
    if mode == "title-only":
        needs.append("resolve_title")
        steps.append(f'python scripts/resolve_paper_by_title.py --title "{title}" --download-dir ./paper --github')
        steps.append("If the script reports network_unavailable, use WebSearch/WebFetch to find the arXiv page and official repo.")
    if mode in {"pdf-plus-code", "pdf-only", "code-only"}:
        flags = []
        if paper_kind != "none":
            flags.append("--paper-text paper.txt")
        if code_kind != "none":
            code_item = first({"code-local", "code-remote"})
            flags.append(f"--code-root {code_item.detail if code_kind == 'local' else './code'}")
        steps.append("python scripts/review_gate.py " + " ".join(flags) + " --json")
    if mode == "none":
        steps.append("Ask the user for a paper (PDF/arXiv link/title) and/or a code path.")
    return Triage(source_mode=mode, items=items, paper=paper_kind, code=code_kind, title=title, needs=needs, next_steps=steps)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*", help="Paths, URLs, arXiv ids, or a quoted paper title.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    result = triage(args.inputs)
    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        print(f"source_mode: {result.source_mode}")
        print(f"paper: {result.paper}")
        print(f"code: {result.code}")
        if result.title:
            print(f"title: {result.title}")
        for item in result.items:
            print(f"  - {item.kind}: {item.detail or item.raw}")
        if result.needs:
            print("needs: " + ", ".join(result.needs))
        print("next_steps:")
        for step in result.next_steps:
            print(f"  - {step}")
    return 0 if result.source_mode != "none" else 1


if __name__ == "__main__":
    raise SystemExit(main())
