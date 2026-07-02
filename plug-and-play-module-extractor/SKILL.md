---
name: plug-and-play-module-extractor
description: Extract, reconstruct, and validate plug-and-play deep-learning modules from research paper PDFs, or from paired paper PDFs and source code repositories. Use when the user uploads or points to a machine-learning paper, official/unofficial code, or examples and asks whether the paper contains a reusable module, attention block, loss, normalization/filter layer, adapter, feature-fusion unit, point-cloud block, time-series block, or other independently runnable component, and wants a clean runnable Python module plus dependency and smoke-test guidance.
---

# Plug-and-Play Module Extractor

## Overview

Use this skill to decide whether a research paper contains an independently reusable module and, when it does, produce a clean runnable implementation. Prefer the sample style in `D:\codex\plug-and-play-module\样例`: a single Python file with a short paper/module note, imports, core classes/functions, and a `__main__` smoke test using random tensors.

## Workflow

1. Inventory the inputs.
   - Locate PDFs, code roots, existing extracted modules, and any user-specified target paper or module.
   - For PDF text, run `scripts/extract_pdf_text.py <paper.pdf> --output <paper.txt>` when direct reading is not enough.
   - For paper-only module detection, run `scripts/analyze_paper_modules.py <paper.txt> --json` and use its `found_module`, candidates, and evidence before writing code.
   - For PDF-only reconstruction, run `scripts/build_reconstruction_brief.py <paper.txt> --output <brief.md>` and use the brief as the implementation checklist.
   - For code roots, run `scripts/find_module_candidates.py <code-root>` to identify candidate classes, functions, imports, and existing test entrypoints.

2. Decide whether a plug-and-play module exists before generating code.
   - Read `references/module_detection.md` when the paper's module status is unclear.
   - Report one of `found_module: yes`, `found_module: no`, or `found_module: uncertain`.
   - If `found_module: no`, stop without generating code. Explain the evidence briefly.
   - If `found_module: uncertain`, continue only with conservative reconstruction or extraction and label uncertainty in the final report.

3. Choose the implementation path.
   - **PDF-only**: read `references/paper_reconstruction.md`, then reconstruct from the paper's architecture text, equations, algorithms, module diagrams, tensor shapes, and ablations. State that the code is paper-derived, not official source code.
   - **PDF + code**: locate the implementation matching the paper module and preserve source behavior, parameter names, tensor conventions, and class/function boundaries where practical. Prefer `scripts/extract_python_module.py <source.py> --output <module.py> --name <ClassOrFunction>`; it automatically includes same-file top-level dependencies referenced by the selected symbol and `__main__`. Use `--include-all-helpers` only when the module boundary is uncertain or several cooperating modules should stay together. Only decouple training pipelines, datasets, logging, checkpoints, and unrelated model scaffolding.

4. Produce the module.
   - Default to one standalone `.py` file. Use a small package only when multiple files are required for the module to run.
   - Include only necessary imports and helper classes/functions.
   - Add a `__main__` smoke test with synthetic inputs that exercises the main forward path and prints input/output shapes.
   - Read `references/output_contract.md` before finalizing the output format.

5. Validate and report.
   - Run `python -m py_compile <module.py>`.
   - Run `python <module.py>` or `scripts/smoke_test_module.py <module.py>` when feasible.
   - For PDF + code tasks, run `scripts/compare_module_interface.py <source.py> <module.py>` when a source file was extracted, then compare the extracted module's smoke-test classification with the source module's classification when the source has a runnable `__main__` block.
   - For sample-based PDF-only forward tests, run `scripts/evaluate_generated_module.py <sample-case-dir> <generated.py> --python <python-with-deps>` to compare generated symbols with paper candidates and the sample reference module. The generated class name may differ from the paper name if it aligns with the reference symbols and the paper aligns with the reference file/symbols.
   - For batch PDF-only forward tests, place generated files in a directory as `case1.py`, `case_1.py`, `1.py`, or `1/module.py`, then run `scripts/evaluate_generated_batch.py <samples-root> <generated-root> --python <python-with-deps>`. Use `--allow-missing` only for partial/local debugging; omit it for full regression.
   - For full skill regression, run `scripts/run_regression_suite.py <samples-root> --python <python-with-deps> --validator <quick_validate.py>`. Add `--report-json <report.json>` when a structured run record is needed.
   - Treat missing optional dependencies as an environment issue, not as proof that extraction failed.
   - Final response must include the module path, dependency notes, run command, smoke-test result, and input/output shape summary.

## Regression Loop

When improving this skill against the project examples, run:

```bash
python scripts/evaluate_sample_cases.py D:\codex\plug-and-play-module\样例 --python <python-with-torch>
```

Use the summary to decide what to optimize next:

- `paper_found=no` for a known module paper: improve `analyze_paper_modules.py` or module detection guidance.
- `paper_alignment=failed`: paper candidates do not resemble the sample code symbols or file name; improve candidate extraction/ranking before trusting PDF-only reconstruction.
- `brief=failed` or `brief=incomplete`: improve `build_reconstruction_brief.py` so PDF-only work has candidates, evidence, implementation hints, and implementation notes before coding.
- `needs_attention` with `no module candidates found`: improve candidate detection or module-boundary guidance.
- `interface=failed`: the extraction dropped imports, classes, functions, constants, or `__main__`; fix extraction before trusting smoke results.
- `extracted_smoke` worse than `smoke`: improve `extract_python_module.py` selection, helper inclusion, or generated smoke-test repair.
- `named_smoke` worse than `smoke`: improve dependency closure for `--name` extraction or target symbol selection.
- `missing_dependency`: keep extraction but report dependency installation clearly.
- `accelerator_unavailable`: make generated smoke tests CPU-compatible by guarding CUDA usage.
- `runtime_error`: inspect whether the extracted code or smoke-test inputs need repair.

## Extraction Rules

- Prefer code-faithful extraction over rewriting when source code is provided.
- Preserve paper terminology for class names and comments when it improves traceability.
- Do not claim official implementation status for PDF-only reconstructions.
- Do not fabricate unsupported training results, benchmark numbers, or compatibility claims.
- Keep generated modules focused on reusable behavior, not full training or dataset pipelines.
- If the paper describes several cooperating plug-and-play units, output them together in one module unless the user asks for a specific submodule.

## Resources

- `references/module_detection.md`: use for deciding whether the paper really contains an extractable module.
- `references/output_contract.md`: use for the required structure and reporting contract of generated modules.
- `references/paper_reconstruction.md`: use for PDF-only reconstruction when no source code is available.
- `scripts/analyze_paper_modules.py`: analyze extracted paper text and produce `found_module` plus candidate evidence.
- `scripts/build_reconstruction_brief.py`: convert paper analysis into a Markdown implementation brief for PDF-only reconstruction.
- `scripts/compare_module_interface.py`: compare source and extracted Python modules for top-level interface preservation.
- `scripts/extract_pdf_text.py`: extract text from PDFs using available local PDF libraries.
- `scripts/find_module_candidates.py`: scan Python source trees for module candidates without importing the project.
- `scripts/extract_python_module.py`: copy source-code-derived top-level classes/functions and helpers into a standalone Python module.
- `scripts/smoke_test_module.py`: compile, import, and run generated Python modules with clear dependency diagnostics.
- `scripts/evaluate_sample_cases.py`: run repeatable regression checks over numbered PDF/code sample folders.
- `scripts/evaluate_generated_module.py`: evaluate one generated PDF-only module against a sample case's paper and reference code.
- `scripts/evaluate_generated_batch.py`: evaluate a directory of generated PDF-only modules against numbered sample cases.
- `scripts/run_regression_suite.py`: run compile checks, skill validation, sample regression, and simulated generated-batch regression in one command.
