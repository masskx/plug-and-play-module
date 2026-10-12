#!/usr/bin/env python3
"""Check whether a repository actually implements the module a paper describes.

Compares paper candidate names (from ``analyze_paper_modules.py``) with the classes,
functions, and file names found by ``find_module_candidates.py``. Handles the usual
naming drift: abbreviations (``CA`` -> ``CoordAtt``), truncated words
(``Coordinate`` -> ``Coord``), snake/camel case, and generic suffixes (``Block``).
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_paper_modules import analyze  # noqa: E402
from find_module_candidates import iter_python_files, ranked, scan_file  # noqa: E402

GENERIC_TOKENS = {"module", "block", "layer", "network", "net", "the", "of", "for", "and", "unit", "model", "plug", "play"}


@dataclass
class Match:
    paper_name: str
    code_symbol: str
    code_path: str
    code_kind: str
    code_tier: str
    score: float
    why: str


def split_tokens(value: str) -> list[str]:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", value)
    return [token.lower() for token in re.split(r"[^A-Za-z0-9]+", value) if token]


def compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def token_match(left: str, right: str) -> bool:
    if left == right:
        return True
    short, long_ = sorted((left, right), key=len)
    return len(short) >= 3 and long_.startswith(short)


def similarity(paper_name: str, code_name: str) -> tuple[float, str]:
    paper_compact, code_compact = compact(paper_name), compact(code_name)
    if not paper_compact or not code_compact:
        return 0.0, "empty"
    if paper_compact == code_compact:
        return 1.0, "same name ignoring case/separators"

    paper_tokens = [token for token in split_tokens(paper_name) if token not in GENERIC_TOKENS] or split_tokens(paper_name)
    code_tokens = [token for token in split_tokens(code_name) if token not in GENERIC_TOKENS] or split_tokens(code_name)

    best = 0.0
    why = "weak string similarity"
    ratio = difflib.SequenceMatcher(None, paper_compact, code_compact).ratio()
    best, why = ratio * 0.8, f"string similarity {ratio:.2f}"

    if len(paper_compact) >= 4 and len(code_compact) >= 4 and (paper_compact in code_compact or code_compact in paper_compact):
        best, why = max(best, 0.9), "one name contains the other"

    if paper_tokens and code_tokens:
        matched = sum(1 for p in paper_tokens if any(token_match(p, c) for c in code_tokens))
        coverage = matched / len(paper_tokens)
        reverse = sum(1 for c in code_tokens if any(token_match(p, c) for p in paper_tokens)) / len(code_tokens)
        token_score = 0.5 * coverage + 0.5 * reverse
        if matched == len(paper_tokens) and matched >= 1:
            token_score = max(token_score, 0.85)
        if token_score > best:
            best, why = token_score, f"{matched}/{len(paper_tokens)} paper words match code words"

    # Acronym: "Coordinate Attention" ~ "CoordAtt"; or paper "CA" ~ code initials.
    paper_is_acronym = paper_name.replace(" ", "").isupper() and 2 <= len(paper_compact) <= 8
    if paper_is_acronym and len(code_tokens) >= 2:
        initials = "".join(token[0] for token in code_tokens)
        if initials == paper_compact or initials.startswith(paper_compact):
            best, why = max(best, 0.8), "paper acronym matches code word initials"
    if len(paper_tokens) >= 2 and len(code_compact) <= 8:
        initials = "".join(token[0] for token in paper_tokens)
        if initials == code_compact:
            best, why = max(best, 0.8), "code abbreviation matches paper word initials"
    return min(best, 1.0), why


def match(paper_names: list[str], code_candidates: list, limit: int = 10) -> list[Match]:
    results: list[Match] = []
    for candidate in code_candidates:
        stem = Path(candidate.path).stem
        best: Match | None = None
        for paper_name in paper_names:
            for label, text in (("symbol", candidate.name), ("file", stem)):
                score, why = similarity(paper_name, text)
                if label == "file":
                    score *= 0.9
                if best is None or score > best.score:
                    best = Match(
                        paper_name=paper_name,
                        code_symbol=candidate.name,
                        code_path=candidate.path,
                        code_kind=candidate.kind,
                        code_tier=candidate.tier,
                        score=round(score, 3),
                        why=f"{label}: {why}",
                    )
        if best:
            results.append(best)
    tier_rank = {"strong": 0, "possible": 1, "weak": 2}
    results.sort(key=lambda m: (-m.score, tier_rank.get(m.code_tier, 3)))
    return results[:limit]


def alignment_label(matches: list[Match]) -> str:
    if not matches:
        return "none"
    top = matches[0].score
    if top >= 0.75:
        return "matched"
    if top >= 0.6:
        return "weak"
    return "none"


def align_paper_text_to_code(paper_text: str, code_root: Path) -> dict:
    analysis = analyze(paper_text)
    names = [candidate.name for candidate in analysis.candidates]
    if analysis.title:
        names.append(analysis.title.split(":")[0].strip())
    summaries = [scan_file(path, code_root if code_root.is_dir() else code_root.parent) for path in sorted(iter_python_files(code_root))]
    code_candidates = ranked(summaries)
    matches = match(names, code_candidates)
    return {
        "alignment": alignment_label(matches),
        "paper_candidates": names,
        "matches": [asdict(item) for item in matches],
    }


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper_text", type=Path, help="Extracted paper text file.")
    parser.add_argument("code_root", type=Path, help="Repository root or Python file.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    for path in (args.paper_text, args.code_root):
        if not path.exists():
            parser.error(f"not found: {path}")

    result = align_paper_text_to_code(args.paper_text.read_text(encoding="utf-8", errors="replace"), args.code_root.resolve())
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"paper_code_alignment: {result['alignment']}")
        print(f"paper_candidates: {', '.join(result['paper_candidates'][:8]) or 'none'}")
        for item in result["matches"][:8]:
            print(
                f"  {item['score']:.2f} {item['paper_name']} -> {item['code_symbol']} "
                f"({item['code_path']}, {item['code_kind']}, tier={item['code_tier']}; {item['why']})"
            )
    return 0 if result["alignment"] != "none" else 1


if __name__ == "__main__":
    raise SystemExit(main())
