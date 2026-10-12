#!/usr/bin/env python3
"""Resolve a paper title (or arXiv id) to a verified paper record and candidate code repos.

This is the entry point for the *title-only* situation. It never guesses: the result is one
of ``resolved``, ``ambiguous``, ``unresolved`` or ``network_unavailable``. Only a
``resolved`` record may be used as the basis for extraction; the PDF it downloads is
untrusted data and is only read, never executed.

Sources: the public arXiv API (title / id lookup, PDF download) and, with ``--github``,
the public GitHub repository search. When the network is blocked, fall back to the
agent's own WebSearch/WebFetch tools and feed the resulting PDF into the normal workflow.
"""

from __future__ import annotations

import argparse
import copy
import difflib
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
ARXIV_API = "https://export.arxiv.org/api/query"
GITHUB_SEARCH = "https://api.github.com/search/repositories"
GITHUB_URL = re.compile(r"https?://(?:www\.)?github\.com/[\w.\-]+/[\w.\-]+", re.I)
USER_AGENT = "plug-and-play-module-extractor/1.0 (research tooling)"
RESOLVE_THRESHOLD = 0.9
AMBIGUOUS_THRESHOLD = 0.7


@dataclass
class PaperRecord:
    title: str
    arxiv_id: str
    authors: list[str]
    published: str
    summary: str
    pdf_url: str
    abs_url: str
    comment: str = ""
    similarity: float = 0.0
    code_links: list[str] = field(default_factory=list)


@dataclass
class Resolution:
    status: str  # resolved | ambiguous | unresolved | network_unavailable
    query: str
    paper: PaperRecord | None = None
    alternatives: list[PaperRecord] = field(default_factory=list)
    repo_candidates: list[dict] = field(default_factory=list)
    downloaded_pdf: str | None = None
    notes: list[str] = field(default_factory=list)


class NetworkUnavailable(RuntimeError):
    pass


def http_get(url: str, timeout: int = 20, binary: bool = False) -> bytes | str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https endpoints
            data = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise NetworkUnavailable(f"{type(exc).__name__}: {exc}") from exc
    return data if binary else data.decode("utf-8", errors="replace")


def normalize_title(value: str) -> str:
    value = re.sub(r"\s+", " ", value.lower())
    return re.sub(r"[^a-z0-9 ]", "", value).strip()


def title_similarity(query: str, candidate: str) -> float:
    """Similarity of a user-supplied title to an arXiv title (1.0 = identical after normalization)."""
    right = normalize_title(candidate)
    variants = [query]
    if ":" in query:  # "CBAM: Convolutional Block Attention Module" -> also try the part after the acronym
        variants.append(query.split(":", 1)[1])
    best = 0.0
    for variant in variants:
        left = normalize_title(variant)
        if not left or not right:
            continue
        ratio = difflib.SequenceMatcher(None, left, right).ratio()
        if len(left) >= 12 and left in right:
            ratio = max(ratio, 0.9)
        best = max(best, ratio)
    return best


def parse_arxiv_feed(xml_text: str) -> list[PaperRecord]:
    root = ET.fromstring(xml_text)
    records: list[PaperRecord] = []
    for entry in root.findall("a:entry", ATOM):
        title = " ".join((entry.findtext("a:title", "", ATOM) or "").split())
        if not title or title.lower() == "error":
            continue
        abs_url = (entry.findtext("a:id", "", ATOM) or "").strip()
        arxiv_id = re.sub(r"^https?://arxiv.org/abs/", "", abs_url)
        summary = " ".join((entry.findtext("a:summary", "", ATOM) or "").split())
        comment = " ".join((entry.findtext("arxiv:comment", "", ATOM) or "").split())
        pdf_url = ""
        for link in entry.findall("a:link", ATOM):
            if link.get("title") == "pdf" or (link.get("type") == "application/pdf"):
                pdf_url = link.get("href", "")
        pdf_url = (pdf_url or f"https://arxiv.org/pdf/{arxiv_id}").replace("http://", "https://")
        authors = [" ".join((a.findtext("a:name", "", ATOM) or "").split()) for a in entry.findall("a:author", ATOM)]
        code_links = sorted({link.rstrip(".,;:)]}'\"").removesuffix(".git") for link in GITHUB_URL.findall(f"{summary} {comment}")})
        records.append(
            PaperRecord(
                title=title,
                arxiv_id=arxiv_id,
                authors=authors,
                published=(entry.findtext("a:published", "", ATOM) or "")[:10],
                summary=summary,
                pdf_url=pdf_url,
                abs_url=abs_url.replace("http://", "https://"),
                comment=comment,
                code_links=code_links,
            )
        )
    return records


def arxiv_query(search_query: str | None = None, id_list: str | None = None, max_results: int = 5) -> list[PaperRecord]:
    params = {"max_results": str(max_results)}
    if search_query:
        params["search_query"] = search_query
    if id_list:
        params["id_list"] = id_list
    xml_text = http_get(f"{ARXIV_API}?{urllib.parse.urlencode(params)}")
    return parse_arxiv_feed(str(xml_text))


def resolve_title(title: str, fetch=arxiv_query) -> Resolution:
    cleaned = title.strip().strip("\"'")
    queries = [f'ti:"{cleaned}"']
    if ":" in cleaned:
        queries.append(f'ti:"{cleaned.split(":", 1)[1].strip()}"')
    queries.append("all:" + " AND all:".join(re.findall(r"[A-Za-z0-9]+", cleaned)[:8]))

    pool: dict[str, PaperRecord] = {}
    try:
        for query in queries:
            for record in fetch(search_query=query):
                record.similarity = round(title_similarity(cleaned, record.title), 3)
                pool.setdefault(record.arxiv_id, record)
            if any(item.similarity >= RESOLVE_THRESHOLD for item in pool.values()):
                break
    except NetworkUnavailable as exc:
        return Resolution(
            status="network_unavailable",
            query=cleaned,
            notes=[
                str(exc),
                "Use WebSearch/WebFetch to locate the arXiv page and official repository, or ask the user for the PDF.",
            ],
        )

    ordered = sorted(pool.values(), key=lambda item: -item.similarity)
    if not ordered or ordered[0].similarity < AMBIGUOUS_THRESHOLD:
        return Resolution(
            status="unresolved",
            query=cleaned,
            alternatives=ordered[:3],
            notes=["No arXiv record matches this title closely enough; do not guess. Ask for the PDF or a link."],
        )
    best = ordered[0]
    runner_up = ordered[1].similarity if len(ordered) > 1 else 0.0
    if best.similarity >= RESOLVE_THRESHOLD and best.similarity - runner_up >= 0.05:
        return Resolution(status="resolved", query=cleaned, paper=best, alternatives=ordered[1:3])
    return Resolution(
        status="ambiguous",
        query=cleaned,
        alternatives=ordered[:3],
        notes=["Several records match; confirm the intended paper (authors/year) before extracting."],
    )


def resolve_arxiv_id(arxiv_id: str, fetch=arxiv_query) -> Resolution:
    clean = re.sub(r"(?i)^arxiv:", "", arxiv_id.strip())
    try:
        records = fetch(id_list=clean)
    except NetworkUnavailable as exc:
        return Resolution(status="network_unavailable", query=clean, notes=[str(exc)])
    if not records:
        return Resolution(status="unresolved", query=clean, notes=["arXiv returned no record for this id."])
    records[0].similarity = 1.0
    return Resolution(status="resolved", query=clean, paper=records[0])


def github_repo_candidates(paper: PaperRecord, max_results: int = 5) -> list[dict]:
    """Public repository search. Candidates are *unverified* until the README cites the paper."""
    queries = [paper.arxiv_id, paper.title]
    seen: dict[str, dict] = {}
    for query in queries:
        url = f"{GITHUB_SEARCH}?{urllib.parse.urlencode({'q': query, 'sort': 'stars', 'per_page': max_results})}"
        try:
            payload = json.loads(str(http_get(url)))
        except (NetworkUnavailable, json.JSONDecodeError):
            continue
        for item in payload.get("items", []):
            seen.setdefault(
                item["html_url"],
                {
                    "url": item["html_url"],
                    "stars": item.get("stargazers_count", 0),
                    "description": item.get("description") or "",
                    "verified": False,
                    "source": f"github search: {query}",
                },
            )
    for link in paper.code_links:
        seen[link] = {"url": link, "stars": None, "description": "linked from the arXiv abstract/comment", "verified": True, "source": "arxiv"}
    return sorted(seen.values(), key=lambda item: (not item["verified"], -(item["stars"] or 0)))


def download_pdf(paper: PaperRecord, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", paper.arxiv_id)
    target = directory / f"{safe}.pdf"
    data = http_get(paper.pdf_url, timeout=60, binary=True)
    if not isinstance(data, bytes) or not data.startswith(b"%PDF"):
        raise NetworkUnavailable("downloaded content is not a PDF")
    target.write_bytes(data)
    return target


def render_text(result: Resolution) -> str:
    lines = [f"resolution: {result.status}", f"query: {result.query}"]
    if result.paper:
        paper = result.paper
        lines += [
            f"title: {paper.title}",
            f"arxiv_id: {paper.arxiv_id}",
            f"authors: {', '.join(paper.authors[:6])}",
            f"published: {paper.published}",
            f"similarity: {paper.similarity}",
            f"pdf_url: {paper.pdf_url}",
        ]
    if result.alternatives:
        lines.append("alternatives:")
        for item in result.alternatives:
            lines.append(f"  - {item.title} ({item.arxiv_id}, {item.published}, similarity={item.similarity})")
    if result.repo_candidates:
        lines.append("repo_candidates:")
        for repo in result.repo_candidates:
            flag = "verified" if repo["verified"] else "unverified"
            lines.append(f"  - {repo['url']} [{flag}] stars={repo['stars']} {repo['description'][:80]}")
    if result.downloaded_pdf:
        lines.append(f"downloaded_pdf: {result.downloaded_pdf}")
    for note in result.notes:
        lines.append(f"note: {note}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--title", help="Paper title (or a distinctive part of it).")
    group.add_argument("--arxiv-id", help="arXiv id such as 1807.06521.")
    parser.add_argument(
        "--from-feed",
        type=Path,
        help="With --title: match against a saved arXiv Atom feed instead of using the network (offline/testing).",
    )
    parser.add_argument("--github", action="store_true", help="Also search GitHub for candidate repositories.")
    parser.add_argument("--download-dir", type=Path, help="Download the resolved paper's PDF here.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.from_feed:
        if not args.title:
            parser.error("--from-feed requires --title")
        records = parse_arxiv_feed(args.from_feed.read_text(encoding="utf-8"))
        result = resolve_title(args.title, fetch=lambda **_: [copy.copy(record) for record in records])
    elif args.arxiv_id:
        result = resolve_arxiv_id(args.arxiv_id)
    else:
        result = resolve_title(args.title)

    if result.paper and result.status == "resolved":
        if args.github:
            result.repo_candidates = github_repo_candidates(result.paper)
            if not result.repo_candidates:
                result.notes.append("No repository candidates found; the paper may have no public code.")
        if args.download_dir:
            try:
                result.downloaded_pdf = str(download_pdf(result.paper, args.download_dir))
            except NetworkUnavailable as exc:
                result.notes.append(f"PDF download failed: {exc}")

    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        print(render_text(result))
    return {"resolved": 0, "ambiguous": 3, "unresolved": 4, "network_unavailable": 5}[result.status]


if __name__ == "__main__":
    raise SystemExit(main())
