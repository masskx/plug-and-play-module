#!/usr/bin/env python3
"""Analyze extracted paper text for plug-and-play module candidates."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


POSITIVE_TERMS = {
    "module",
    "block",
    "layer",
    "attention",
    "adapter",
    "filter",
    "loss",
    "fusion",
    "selector",
    "selection",
    "token",
    "normalization",
    "decomposition",
    "mixing",
    "mixer",
    "plug-and-play",
    "plug and play",
    "can be inserted",
    "inserted into",
    "lightweight",
    "ablation",
}

NEGATIVE_TERMS = {
    "dataset",
    "benchmark",
    "survey",
    "annotation",
    "challenge",
    "leaderboard",
    "evaluation protocol",
}

MODULE_NOUNS = (
    "module",
    "block",
    "layer",
    "attention",
    "adapter",
    "filter",
    "loss",
    "fusion",
    "selector",
    "selection",
    "mixing",
    "mixer",
    "decomposition",
    "network",
)

STOP_CANDIDATES = {
    "ABSTRACT",
    "ADAM",
    "ASPP",
    "CNN",
    "CV",
    "DETAILS",
    "ELU",
    "EMD",
    "FFT",
    "FMCW",
    "FOR",
    "GPT",
    "ICLR",
    "IEEE",
    "IFFT",
    "IMPLEMENTATION",
    "LINEAR",
    "LSTM",
    "MLP",
    "MSE",
    "PDF",
    "RNN",
    "SOTA",
    "TRANSFORMERS",
    "VISION",
    "A",
    "AN",
    "AND",
    "AS",
    "BY",
    "CAN",
    "DURING",
    "FOR",
    "FUNCTION",
    "IN",
    "IS",
    "IT",
    "MECHANISM",
    "MECHANISMS",
    "MODULE",
    "OF",
    "ON",
    "THE",
    "TO",
    "TWO",
    "UNDER",
    "WHICH",
}


@dataclass
class ModuleCandidate:
    name: str
    score: int
    evidence: list[str]


@dataclass
class PaperAnalysis:
    found_module: str
    title: str
    score: int
    positive_hits: list[str]
    negative_hits: list[str]
    candidates: list[ModuleCandidate]
    evidence: list[str]


def ensure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def normalize_space(text: str) -> str:
    text = text.replace("-\n", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_extractor_metadata(text: str) -> str:
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if not stripped:
            kept.append(line)
            continue
        if lowered in {"# extracted pdf text"}:
            continue
        if lowered.startswith(("source:", "backend:", "pages:", "## page")):
            continue
        kept.append(line)
    return "\n".join(kept)


def split_sentences(text: str) -> list[str]:
    normalized = normalize_space(text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", normalized)
    return [part.strip() for part in parts if len(part.strip()) > 20]


def extract_title(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    lines = [
        line
        for line in lines
        if not line.lower().startswith(("source:", "backend:", "pages:", "## page", "# extracted"))
    ]
    if not lines:
        return ""
    title_parts: list[str] = []
    for line in lines[:8]:
        if re.search(r"\b(abstract|introduction)\b", line, flags=re.I):
            break
        if re.search(r"[@{}]|\b(university|institute|school|department)\b", line, flags=re.I):
            break
        title_parts.append(line)
        if len(" ".join(title_parts)) > 140:
            break
    return normalize_space(" ".join(title_parts))


def candidate_names_from_sentence(sentence: str) -> list[str]:
    names: list[str] = []

    propose_pattern = re.compile(
        r"\b(?:we\s+)?(?:propose|present|introduce|develop)\s+"
        r"(?:(?:a|an|the|one|our|novel|simple|effective|lightweight|new|deep|learning|method|framework|"
        r"model|network|module|architecture|plug-and-play|called|namely|,)\s+)*"
        r"([A-Z][A-Za-z0-9+_.-]{1,30}(?:\s+[A-Z][A-Za-z0-9+_.-]{1,30}){0,5})"
    )
    for match in propose_pattern.finditer(sentence):
        phrase = match.group(1)
        phrase = re.split(r"\s+(?:to|for|that|which|with|by|and|as|built|based)\b", phrase, maxsplit=1)[0]
        names.append(phrase)

    named_pattern = re.compile(
        r"\b(?:namely|called|termed)\s+([A-Z][A-Za-z0-9+_.-]{1,30}(?:\s+[A-Z][A-Za-z0-9+_.-]{1,30}){0,5})"
    )
    for match in named_pattern.finditer(sentence):
        names.append(match.group(1))

    module_after_pattern = re.compile(
        r"\b(?:module|block|layer|loss|attention|filter|network|mechanism)\s*,?\s+([A-Z][A-Za-z0-9+_.-]{1,30})\b"
    )
    for match in module_after_pattern.finditer(sentence):
        names.append(match.group(1))

    name_before_module_pattern = re.compile(
        r"\b([A-Z][A-Za-z0-9+_.-]{1,30}(?:\s+[A-Z][A-Za-z0-9+_.-]{1,30}){0,4})\s+"
        r"(?:module|block|layer|loss|attention|filter|network|mechanism)\b"
    )
    for match in name_before_module_pattern.finditer(sentence):
        names.append(match.group(1))

    paren_pattern = re.compile(
        r"\b([A-Z][A-Za-z0-9+_.-]*(?:\s+[A-Z][A-Za-z0-9+_.-]*){0,4})\s*\(([A-Z][A-Z0-9+_.-]{1,12})\)"
    )
    for match in paren_pattern.finditer(sentence):
        long_name, acronym = match.groups()
        if any(noun in long_name.lower() for noun in MODULE_NOUNS):
            names.append(acronym)
            names.append(long_name)

    noun_pattern = re.compile(
        r"\b([A-Z][A-Za-z0-9+_.-]*(?:\s+[A-Z][A-Za-z0-9+_.-]*){0,5}\s+"
        r"(?:Module|Block|Layer|Attention|Adapter|Filter|Loss|Fusion|Selector|Selection|Mixing|Mixer|Decomposition|Network))\b"
    )
    for match in noun_pattern.finditer(sentence):
        names.append(match.group(1))

    acronym_pattern = re.compile(r"\b([A-Z][A-Z0-9+_.-]{2,15})\b")
    for match in acronym_pattern.finditer(sentence):
        token = match.group(1)
        if any(noun in sentence.lower() for noun in MODULE_NOUNS):
            names.append(token)

    cleaned: list[str] = []
    for name in names:
        value = normalize_space(name).strip(" ,.;:()[]")
        value = re.sub(r"^(a|an|the|our|novel|simple|effective|new)\s+", "", value, flags=re.I)
        value = value.strip(" ,.;:()[]")
        compact = value.replace(" ", "").upper()
        if (
            len(value) >= 2
            and (len(value) > 2 or value.isupper())
            and not value.islower()
            and value.lower() not in {"the", "this", "our", "figure", "table"}
            and compact not in STOP_CANDIDATES
            and not re.fullmatch(r"[A-Z]\.?\d+", compact)
        ):
            cleaned.append(value)
    return cleaned


def candidate_names_from_title(title: str) -> list[str]:
    names: list[str] = []
    if ":" in title:
        before_colon, after_colon = title.split(":", 1)
        method = normalize_space(before_colon).strip("0123456789 .,-")
        if 2 <= len(method) <= 80:
            names.append(method)
        names.extend(candidate_names_from_sentence(after_colon))
    names.extend(candidate_names_from_sentence(title))
    return names


def analyze(text: str, max_evidence: int = 8) -> PaperAnalysis:
    text = strip_extractor_metadata(text)
    normalized = normalize_space(text)
    lowered = normalized.lower()
    title = extract_title(text)
    positive_hits = sorted(term for term in POSITIVE_TERMS if term in lowered)
    negative_hits = sorted(term for term in NEGATIVE_TERMS if term in lowered)

    sentences = split_sentences(text)
    evidence_sentences: list[str] = []
    candidate_map: dict[str, ModuleCandidate] = {}

    for sentence in sentences:
        sentence_lower = sentence.lower()
        hits = [term for term in POSITIVE_TERMS if term in sentence_lower]
        if not hits:
            continue
        if len(evidence_sentences) < max_evidence:
            evidence_sentences.append(sentence[:500])
        for name in candidate_names_from_sentence(sentence):
            candidate = candidate_map.setdefault(name, ModuleCandidate(name=name, score=0, evidence=[]))
            candidate.score += 2 + len(hits)
            if len(candidate.evidence) < 3:
                candidate.evidence.append(sentence[:350])

    if title:
        for name in candidate_names_from_title(title):
            candidate = candidate_map.setdefault(name, ModuleCandidate(name=name, score=0, evidence=[]))
            candidate.score += 8
            if title not in candidate.evidence:
                candidate.evidence.append(title)

    candidates = sorted(candidate_map.values(), key=lambda item: (-item.score, item.name.lower()))
    score = len(positive_hits) * 2 + min(len(evidence_sentences), 6) + sum(1 for item in candidates if item.score >= 3)
    score -= min(len(negative_hits), 3)

    if candidates and score >= 8:
        found_module = "yes"
    elif score >= 6 and positive_hits:
        found_module = "uncertain"
    else:
        found_module = "no"

    return PaperAnalysis(
        found_module=found_module,
        title=title,
        score=score,
        positive_hits=positive_hits,
        negative_hits=negative_hits,
        candidates=candidates[:12],
        evidence=evidence_sentences,
    )


def render_text(result: PaperAnalysis) -> str:
    lines = [
        f"found_module: {result.found_module}",
        f"title: {result.title}",
        f"score: {result.score}",
        f"positive_hits: {', '.join(result.positive_hits) if result.positive_hits else 'none'}",
        f"negative_hits: {', '.join(result.negative_hits) if result.negative_hits else 'none'}",
        "candidates:",
    ]
    if result.candidates:
        for candidate in result.candidates:
            lines.append(f"  - {candidate.name} (score={candidate.score})")
    else:
        lines.append("  none")
    lines.append("evidence:")
    if result.evidence:
        for sentence in result.evidence:
            lines.append(f"  - {sentence}")
    else:
        lines.append("  none")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", type=Path, help="Extracted paper text file.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    args = parser.parse_args(argv)

    if not args.text.exists():
        parser.error(f"text file not found: {args.text}")

    text = args.text.read_text(encoding="utf-8", errors="replace")
    result = analyze(text)
    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        print(render_text(result))
    return 0 if result.found_module != "no" else 1


if __name__ == "__main__":
    raise SystemExit(main())
