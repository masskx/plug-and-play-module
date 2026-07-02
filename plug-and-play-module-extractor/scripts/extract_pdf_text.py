#!/usr/bin/env python3
"""Extract readable text from a PDF into a page-delimited text file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def ensure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _extract_with_pymupdf(pdf_path: Path, max_pages: int | None) -> list[str]:
    import fitz  # type: ignore

    pages: list[str] = []
    with fitz.open(pdf_path) as doc:
        limit = len(doc) if max_pages is None else min(max_pages, len(doc))
        for index in range(limit):
            pages.append(doc[index].get_text("text"))
    return pages


def _extract_with_pypdf(pdf_path: Path, max_pages: int | None) -> list[str]:
    from pypdf import PdfReader  # type: ignore

    reader = PdfReader(str(pdf_path))
    limit = len(reader.pages) if max_pages is None else min(max_pages, len(reader.pages))
    return [reader.pages[index].extract_text() or "" for index in range(limit)]


def _extract_with_pdfplumber(pdf_path: Path, max_pages: int | None) -> list[str]:
    import pdfplumber  # type: ignore

    pages: list[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        limit = len(pdf.pages) if max_pages is None else min(max_pages, len(pdf.pages))
        for index in range(limit):
            pages.append(pdf.pages[index].extract_text() or "")
    return pages


def extract_pages(pdf_path: Path, max_pages: int | None = None) -> tuple[str, list[str]]:
    errors: list[str] = []
    for name, extractor in (
        ("pymupdf", _extract_with_pymupdf),
        ("pypdf", _extract_with_pypdf),
        ("pdfplumber", _extract_with_pdfplumber),
    ):
        try:
            return name, extractor(pdf_path, max_pages)
        except ModuleNotFoundError as exc:
            errors.append(f"{name}: missing dependency {exc.name}")
        except Exception as exc:  # Keep trying other PDF backends.
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    message = "\n".join(errors)
    raise RuntimeError(
        "No PDF extraction backend succeeded. Install one of: pymupdf, pypdf, pdfplumber.\n"
        + message
    )


def render_pages(pdf_path: Path, backend: str, pages: list[str]) -> str:
    chunks = [
        f"# Extracted PDF Text",
        f"",
        f"source: {pdf_path}",
        f"backend: {backend}",
        f"pages: {len(pages)}",
        f"",
    ]
    for index, text in enumerate(pages, start=1):
        chunks.append(f"## Page {index}")
        chunks.append(text.strip())
        chunks.append("")
    return "\n".join(chunks)


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Path to the paper PDF.")
    parser.add_argument("--output", "-o", type=Path, help="Write extracted text to this file.")
    parser.add_argument("--max-pages", type=int, default=None, help="Only extract the first N pages.")
    args = parser.parse_args(argv)

    pdf_path = args.pdf.resolve()
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 2
    if pdf_path.suffix.lower() != ".pdf":
        print(f"Input is not a PDF: {pdf_path}", file=sys.stderr)
        return 2

    try:
        backend, pages = extract_pages(pdf_path, args.max_pages)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    output = render_pages(pdf_path, backend, pages)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
        print(f"Wrote {len(pages)} pages using {backend}: {args.output}")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
