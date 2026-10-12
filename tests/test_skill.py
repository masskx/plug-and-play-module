"""Self-contained regression tests for the plug-and-play-module-extractor skill.

Run from the repository root:

    python -m unittest discover -s tests -v

Everything uses synthetic fixtures under tests/fixtures, so no private samples are needed.
Tests that execute extracted modules need PyTorch and are skipped when it is missing.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "plug-and-play-module-extractor"
SCRIPTS = SKILL / "scripts"
FIXTURES = ROOT / "tests" / "fixtures"
PAPERS = FIXTURES / "papers"
sys.path.insert(0, str(SCRIPTS))

import analyze_paper_modules as analyzer  # noqa: E402
import check_module_contract as contract  # noqa: E402
import extract_python_module as extractor  # noqa: E402
import find_module_candidates as finder  # noqa: E402
import match_paper_to_code as matcher  # noqa: E402
import resolve_paper_by_title as resolver  # noqa: E402
import review_gate  # noqa: E402
import triage_inputs  # noqa: E402

HAS_TORCH = importlib.util.find_spec("torch") is not None


def paper(name: str) -> str:
    return (PAPERS / f"{name}.txt").read_text(encoding="utf-8")


def run_script(name: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args],
        text=True,
        capture_output=True,
        encoding="utf-8",
    )


class PaperAnalysisTests(unittest.TestCase):
    def verdict(self, name: str) -> str:
        return analyzer.analyze(paper(name)).found_module

    def test_real_module_paper_is_yes(self) -> None:
        result = analyzer.analyze(paper("module_positive"))
        self.assertEqual(result.found_module, "yes")
        self.assertEqual(result.contribution_type, "module")
        self.assertIn("Ripple Attention", [c.name for c in result.candidates])
        self.assertTrue(result.structural.plug_claims)
        self.assertGreaterEqual(result.structural.equation_lines, 2)

    def test_non_module_contributions_are_no(self) -> None:
        for name, kind in [
            ("dataset_only", "dataset"),
            ("survey_only", "survey"),
            ("theory_only", "theory"),
            ("system_only", "system"),
        ]:
            with self.subTest(name=name):
                result = analyzer.analyze(paper(name))
                self.assertEqual(result.found_module, "no")
                self.assertEqual(result.contribution_type, kind)

    def test_named_module_without_detail_is_uncertain_not_no(self) -> None:
        self.assertEqual(self.verdict("vague_module"), "uncertain")

    def test_generic_vocabulary_alone_never_gives_yes(self) -> None:
        text = (
            "A Study of Training Dynamics\n\nAbstract\n"
            + "We study how the layer and block configuration and the attention module affect results. " * 40
        )
        self.assertNotEqual(analyzer.analyze(text).found_module, "yes")

    def test_short_extraction_cannot_prove_absence(self) -> None:
        result = analyzer.analyze("Some Title\n\nThe network has layers and a loss. " * 3)
        self.assertEqual(result.found_module, "uncertain")

    def test_title_excludes_author_line(self) -> None:
        self.assertEqual(
            analyzer.analyze(paper("module_positive")).title,
            "Ripple Attention: A Lightweight Plug-and-Play Channel Attention Block",
        )

    def test_candidate_noise_is_filtered(self) -> None:
        names = [c.name for c in analyzer.analyze(paper("module_positive")).candidates]
        self.assertFalse([n for n in names if "Abstract" in n or "University" in n or "Science" in n])


class CodeScanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.summaries = [
            finder.scan_file(path, FIXTURES / "repo_a") for path in sorted(finder.iter_python_files(FIXTURES / "repo_a"))
        ]
        self.by_name = {c.name: c for s in self.summaries for c in s.candidates}

    def test_clean_modules_are_strong(self) -> None:
        for name in ("ChannelGate", "SpatialAttention", "ChannelSpatialBlock"):
            self.assertEqual(self.by_name[name].tier, "strong", name)

    def test_inherited_module_base_is_detected(self) -> None:
        self.assertIn("ChannelGate", self.by_name["ChannelSpatialBlock"].bases)

    def test_scaffolding_and_whole_models_are_not_strong(self) -> None:
        self.assertNotIn("SegTrainer", self.by_name.keys() - {n for n, c in self.by_name.items() if c.tier == "weak"})
        self.assertNotEqual(self.by_name["TinyResNet"].tier, "strong")
        self.assertTrue(any("class count" in risk for risk in self.by_name["TinyResNet"].risks))

    def test_coupled_module_reports_why_it_is_not_standalone(self) -> None:
        coupled = self.by_name["CoupledBlock"]
        self.assertEqual(coupled.tier, "weak")
        joined = " ".join(coupled.risks)
        for needle in ("mmcv", "project-local", "registry", "config"):
            self.assertIn(needle, joined)

    def test_repo_without_modules_has_no_candidates(self) -> None:
        summaries = [finder.scan_file(p, FIXTURES / "repo_none") for p in finder.iter_python_files(FIXTURES / "repo_none")]
        self.assertEqual(finder.ranked(summaries), [])

    def test_json_output_keeps_legacy_keys(self) -> None:
        proc = run_script("find_module_candidates.py", str(FIXTURES / "repo_a"), "--json")
        data = json.loads(proc.stdout)
        candidate = data[0]["candidates"][0]
        for key in ("path", "kind", "name", "line", "reason", "bases"):
            self.assertIn(key, candidate)
        self.assertIn("has_main", data[0])


class MatchTests(unittest.TestCase):
    def test_naming_drift_is_matched(self) -> None:
        for paper_name, code_name in [
            ("Coordinate Attention", "CoordAtt"),
            ("CA", "CoordAtt"),
            ("Efficient Channel Attention", "efficient_channel_attention"),
            ("Ripple Attention", "RippleAttention"),
        ]:
            with self.subTest(paper_name=paper_name):
                self.assertGreaterEqual(matcher.similarity(paper_name, code_name)[0], 0.75)

    def test_unrelated_names_do_not_match(self) -> None:
        self.assertLess(matcher.similarity("Ripple Attention", "SegTrainer")[0], 0.5)
        self.assertLess(matcher.similarity("Harmony Fusion", "focal_loss")[0], 0.5)

    def test_alignment_with_matching_and_unrelated_repo(self) -> None:
        text = paper("module_positive")
        self.assertEqual(matcher.align_paper_text_to_code(text, FIXTURES / "repo_match")["alignment"], "matched")
        self.assertNotEqual(matcher.align_paper_text_to_code(text, FIXTURES / "repo_a")["alignment"], "matched")


class TriageTests(unittest.TestCase):
    def test_four_modes(self) -> None:
        paper_file = str(PAPERS / "module_positive.txt")
        repo = str(FIXTURES / "repo_a")
        self.assertEqual(triage_inputs.triage([paper_file, repo]).source_mode, "pdf-plus-code")
        self.assertEqual(triage_inputs.triage([paper_file]).source_mode, "pdf-only")
        self.assertEqual(triage_inputs.triage([repo]).source_mode, "code-only")
        self.assertEqual(triage_inputs.triage(["CBAM: Convolutional Block Attention Module"]).source_mode, "title-only")
        self.assertEqual(triage_inputs.triage([]).source_mode, "none")

    def test_links_and_ids(self) -> None:
        result = triage_inputs.triage(["1807.06521", "https://github.com/example/repo"])
        self.assertEqual(result.source_mode, "pdf-plus-code")
        self.assertIn("fetch_arxiv_pdf", result.needs)
        self.assertIn("clone_repo", result.needs)
        self.assertEqual(triage_inputs.triage(["https://arxiv.org/abs/1807.06521v2"]).paper, "arxiv")

    def test_title_plus_code_is_code_only(self) -> None:
        result = triage_inputs.triage(["Some Paper Title", str(FIXTURES / "repo_a")])
        self.assertEqual(result.source_mode, "code-only")
        self.assertEqual(result.title, "Some Paper Title")


class TitleResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = resolver.parse_arxiv_feed((FIXTURES / "arxiv_feed.xml").read_text(encoding="utf-8"))

    def fetch(self, **_: object):
        return [r for r in self.records]

    def test_feed_parsing(self) -> None:
        first = self.records[0]
        self.assertEqual(first.title, "Ripple Attention: A Lightweight Plug-and-Play Channel Attention Block")
        self.assertEqual(first.code_links, ["https://github.com/example-lab/ripple-attention"])
        self.assertTrue(first.pdf_url.startswith("https://arxiv.org/pdf/"))

    def test_exact_title_resolves(self) -> None:
        result = resolver.resolve_title(
            "Ripple Attention: A Lightweight Plug-and-Play Channel Attention Block", fetch=self.fetch
        )
        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.paper.arxiv_id, "2401.00001v2")

    def test_short_title_is_ambiguous(self) -> None:
        self.assertEqual(resolver.resolve_title("Ripple Attention", fetch=self.fetch).status, "ambiguous")

    def test_unknown_title_is_unresolved_not_guessed(self) -> None:
        result = resolver.resolve_title("Quantum chromodynamics on the lattice", fetch=self.fetch)
        self.assertEqual(result.status, "unresolved")
        self.assertIsNone(result.paper)

    def test_blocked_network_is_reported(self) -> None:
        def boom(**_: object):
            raise resolver.NetworkUnavailable("offline")

        result = resolver.resolve_title("Anything", fetch=boom)
        self.assertEqual(result.status, "network_unavailable")
        self.assertIn("WebSearch", " ".join(result.notes))


class ExtractionTests(unittest.TestCase):
    def extract(self, source: Path, root: Path, *names: str, **kwargs):
        return extractor.extract(source, list(names), root=root, **kwargs)

    def test_single_file_extraction_keeps_dependencies_and_main(self) -> None:
        text, notes = self.extract(FIXTURES / "repo_a" / "blocks.py", FIXTURES / "repo_a", "ChannelSpatialBlock", include_main=True)
        for needle in ("class ChannelGate", "class SpatialAttention", "class ChannelSpatialBlock", '__name__ == "__main__"'):
            self.assertIn(needle, text)
        self.assertNotIn("SegTrainer", text)
        self.assertEqual(notes, [])

    def test_cross_file_helpers_are_inlined(self) -> None:
        root = FIXTURES / "repo_b"
        text, notes = self.extract(root / "pkg" / "blocks.py", root, "DWBlock", include_main=True)
        self.assertIn("class HSwish", text)
        self.assertIn("def make_divisible", text)
        self.assertIn("DEFAULT_DIVISOR = 8", text)
        self.assertIn("md = make_divisible", text)
        self.assertNotIn("from .ops", text)
        self.assertNotIn("class Unrelated", text)
        self.assertEqual(notes, [])
        compile(text, "extracted", "exec")

    def test_registry_decorator_is_dropped_and_unresolved_import_is_reported(self) -> None:
        text, notes = self.extract(FIXTURES / "repo_a" / "coupled.py", FIXTURES / "repo_a", "CoupledBlock", prune_imports=True)
        self.assertNotIn("register_module", text)
        self.assertNotIn("NECKS", text)
        self.assertNotIn("from .utils", text)
        self.assertTrue(any("make_divisible" in note for note in notes))
        report = contract.check(self.write_temp(text))
        self.assertTrue(any(f.code == "undefined-name" and "make_divisible" in f.message for f in report.findings))

    def test_non_registry_decorators_are_kept(self) -> None:
        source = (
            "import dataclasses\nimport torch.nn as nn\nfrom reg import MODELS\n\n\n"
            "@dataclasses.dataclass\nclass Opts:\n    k: int = 3\n\n\n"
            "@MODELS.register_module()\n@dataclasses.dataclass(frozen=True)\nclass Block(nn.Module):\n    k: int = Opts.k\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.py"
            path.write_text(source, encoding="utf-8")
            text, _ = extractor.extract(path, ["Block"], root=Path(tmp), prune_imports=True)
        self.assertIn("@dataclasses.dataclass\nclass Opts", text)
        self.assertIn("@dataclasses.dataclass(frozen=True)\nclass Block", text)
        self.assertNotIn("register_module", text)
        self.assertNotIn("from reg", text)
        compile(text, "extracted", "exec")

    def test_no_resolve_local_keeps_original_imports(self) -> None:
        root = FIXTURES / "repo_b"
        text, _ = self.extract(root / "pkg" / "blocks.py", root, "DWBlock", resolve_local=False)
        self.assertIn("from .ops.act import HSwish", text)

    def test_fidelity_check_flags_edits_but_not_subsets_or_inlining(self) -> None:
        import compare_module_interface as compare

        root = FIXTURES / "repo_b"
        text, _ = self.extract(root / "pkg" / "blocks.py", root, "DWBlock", include_main=True)
        exported = self.write_temp(text)
        report = compare.fidelity(root / "pkg" / "blocks.py", exported)
        self.assertEqual(report["modified"], [])
        self.assertIn("DWBlock", report["identical"])
        self.assertIn("HSwish", report["added"])  # inlined from a sibling file
        edited = self.write_temp(text.replace("x + self.project", "x - self.project"))
        self.assertEqual(compare.fidelity(root / "pkg" / "blocks.py", edited)["modified"], ["DWBlock"])
        # Inlined project-local imports are not reported as lost.
        self.assertEqual(compare.compare(root / "pkg" / "blocks.py", exported).missing_imports, [])

    def test_missing_symbol_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.extract(FIXTURES / "repo_a" / "blocks.py", FIXTURES / "repo_a", "DoesNotExist")

    def test_cli_list_and_output(self) -> None:
        proc = run_script("extract_python_module.py", str(FIXTURES / "repo_a" / "blocks.py"), "--list")
        self.assertIn("ChannelGate", proc.stdout.split())
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "m.py"
            proc = run_script(
                "extract_python_module.py",
                str(FIXTURES / "repo_b" / "pkg" / "blocks.py"),
                "--name", "DWBlock", "--include-main", "--root", str(FIXTURES / "repo_b"), "-o", str(out),
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(out.exists())

    @unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
    def test_extracted_module_runs_standalone(self) -> None:
        root = FIXTURES / "repo_b"
        text, _ = self.extract(root / "pkg" / "blocks.py", root, "DWBlock", include_main=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dw_block.py"
            path.write_text(text, encoding="utf-8")
            proc = subprocess.run([sys.executable, str(path)], text=True, capture_output=True, cwd=tmp)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("torch.Size([1, 16, 8, 8])", proc.stdout)

    def write_temp(self, text: str) -> Path:
        handle = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8")
        handle.write(text)
        handle.close()
        self.addCleanup(Path(handle.name).unlink)
        return Path(handle.name)


class ContractTests(unittest.TestCase):
    GOOD = '''"""Demo. Implementation basis: paper-derived. Input/output shape: (B, C) -> (B, C)."""
import torch
import torch.nn as nn


class Demo(nn.Module):
    def forward(self, x):
        return x


if __name__ == "__main__":
    x = torch.randn(2, 3, requires_grad=True)
    y = Demo()(x)
    y.sum().backward()
    print("shape", x.shape, y.shape)
'''

    def check(self, source: str):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.py"
            path.write_text(source, encoding="utf-8")
            return contract.check(path)

    def codes(self, source: str) -> set[str]:
        return {f.code for f in self.check(source).findings if f.severity == "error"}

    def test_clean_module_passes(self) -> None:
        report = self.check(self.GOOD)
        self.assertEqual(report.status, "passed")
        self.assertEqual(report.third_party_dependencies, ["torch"])
        self.assertEqual(report.findings, [])

    def test_each_blocking_problem_is_detected(self) -> None:
        cases = {
            "hard-cuda": self.GOOD.replace("Demo()(x)", "Demo().cuda()(x.cuda())"),
            "network": self.GOOD.replace("import torch\n", "import torch\nimport requests\nrequests.get('http://x')\n"),
            "local-import": self.GOOD.replace("import torch\n", "import torch\nfrom .utils import f\n"),
            "undefined-name": self.GOOD.replace("return x", "return helper(x)"),
            "main": self.GOOD.split("if __name__")[0],
        }
        for code, source in cases.items():
            with self.subTest(code=code):
                self.assertIn(code, self.codes(source))

    def test_guarded_cuda_selection_is_allowed(self) -> None:
        source = self.GOOD.replace(
            "    x = torch.randn(2, 3, requires_grad=True)",
            '    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n    x = torch.randn(2, 3, requires_grad=True, device=device)',
        )
        self.assertEqual(self.check(source).status, "passed")

    def test_pretrained_download_is_blocked(self) -> None:
        self.assertIn("network", self.codes(self.GOOD + "\nm = make(pretrained=True)\n"))

    @unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
    def test_run_flag_executes_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.py"
            path.write_text(self.GOOD, encoding="utf-8")
            proc = run_script("check_module_contract.py", str(path), "--run", "--json")
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertEqual(json.loads(proc.stdout)["run"]["classification"], "passed")


class ReviewGateTests(unittest.TestCase):
    def gate(self, paper_name: str | None = None, repo: str | None = None, title: str | None = None, lang: str = "en"):
        return review_gate.run_gate(
            paper(paper_name) if paper_name else None,
            FIXTURES / repo if repo else None,
            title,
            lang,
        )

    def test_pdf_only_module(self) -> None:
        gate = self.gate("module_positive")
        self.assertEqual((gate.source_mode, gate.found_module, gate.implementation_basis), ("pdf-only", "yes", "paper-derived"))

    def test_pdf_only_without_module_outputs_standard_report_only(self) -> None:
        gate = self.gate("dataset_only")
        self.assertEqual(gate.found_module, "no")
        self.assertEqual(gate.implementation_basis, "none")
        self.assertTrue(gate.report.startswith("After review, no usable plug-and-play modules were found in"))
        self.assertEqual(gate.targets, [])
        self.assertIn("create no code", " ".join(gate.next_steps))

    def test_chinese_report(self) -> None:
        self.assertTrue(self.gate("survey_only", lang="zh").report.startswith("经审查，"))

    def test_pdf_plus_matching_code_uses_source(self) -> None:
        gate = self.gate("module_positive", "repo_match")
        self.assertEqual((gate.found_module, gate.implementation_basis), ("yes", "source-code-derived"))
        self.assertEqual(gate.targets[0]["name"], "RippleAttention")
        self.assertTrue(any("extract_python_module.py ripple.py --name RippleAttention" in s for s in gate.next_steps))

    def test_pdf_plus_unrelated_code_falls_back_to_mixed(self) -> None:
        gate = self.gate("module_positive", "repo_a")
        self.assertEqual((gate.found_module, gate.implementation_basis), ("yes", "mixed"))
        self.assertFalse(any("extract_python_module.py" in s for s in gate.next_steps))

    def test_paper_without_module_and_code_without_modules_is_no(self) -> None:
        self.assertEqual(self.gate("dataset_only", "repo_none").found_module, "no")

    def test_paper_without_module_but_strong_code_is_flagged_for_review(self) -> None:
        gate = self.gate("dataset_only", "repo_a")
        self.assertEqual(gate.found_module, "uncertain")

    def test_code_only(self) -> None:
        gate = self.gate(repo="repo_a")
        self.assertEqual((gate.source_mode, gate.found_module, gate.implementation_basis), ("code-only", "yes", "source-code-derived"))
        none = self.gate(repo="repo_none")
        self.assertEqual(none.found_module, "no")
        self.assertIn("no usable plug-and-play modules", none.report)

    def test_title_only_never_claims_no_module(self) -> None:
        gate = self.gate(title="Some Paper Title")
        self.assertEqual((gate.source_mode, gate.found_module), ("title-only", "uncertain"))
        self.assertEqual(gate.report, "")
        self.assertTrue(any("resolve_paper_by_title.py" in s for s in gate.next_steps))

    def test_exit_codes(self) -> None:
        self.assertEqual(run_script("review_gate.py", "--paper-text", str(PAPERS / "dataset_only.txt")).returncode, 1)
        self.assertEqual(run_script("review_gate.py", "--paper-text", str(PAPERS / "module_positive.txt")).returncode, 0)
        self.assertEqual(run_script("review_gate.py", "--title", "X Y Z").returncode, 2)

    def test_json_is_valid(self) -> None:
        proc = run_script("review_gate.py", "--code-root", str(FIXTURES / "repo_a"), "--json")
        data = json.loads(proc.stdout)
        self.assertEqual(data["found_module"], "yes")


class EndToEndTests(unittest.TestCase):
    @unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
    def test_gate_extract_contract_pipeline(self) -> None:
        gate = review_gate.run_gate(paper("module_positive"), FIXTURES / "repo_match", None)
        target = gate.targets[0]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ripple_attention.py"
            text, _ = extractor.extract(
                FIXTURES / "repo_match" / target["path"], [target["name"]], root=FIXTURES / "repo_match", include_main=True, prune_imports=True
            )
            out.write_text(text, encoding="utf-8")
            proc = run_script("check_module_contract.py", str(out), "--run", "--python", sys.executable)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("smoke_test: passed", proc.stdout)


class ScriptHygieneTests(unittest.TestCase):
    def test_every_script_compiles_and_has_help(self) -> None:
        for script in sorted(SCRIPTS.glob("*.py")):
            with self.subTest(script=script.name):
                compile(script.read_text(encoding="utf-8"), str(script), "exec")
                proc = run_script(script.name, "--help")
                self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_skill_md_mentions_every_script(self) -> None:
        skill_md = (SKILL / "SKILL.md").read_text(encoding="utf-8")
        for script in sorted(SCRIPTS.glob("*.py")):
            with self.subTest(script=script.name):
                self.assertIn(script.name, skill_md)

    def test_skill_has_no_machine_specific_paths(self) -> None:
        for path in [SKILL / "SKILL.md", *sorted((SKILL / "references").glob("*.md"))]:
            self.assertNotIn("D:\\", path.read_text(encoding="utf-8"), path.name)


if __name__ == "__main__":
    unittest.main()
