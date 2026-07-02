# Paper-Only Reconstruction Guide

Use this guide when no source code is provided and the module must be reconstructed from a paper PDF.

## Required Sequence

1. Extract paper text with `scripts/extract_pdf_text.py`.
2. Analyze the text with `scripts/analyze_paper_modules.py`.
3. Build a reconstruction brief with `scripts/build_reconstruction_brief.py`.
4. Stop if the report says `found_module: no`.
5. If `found_module: yes` or `uncertain`, read the brief and the relevant paper sections around the candidate names, evidence sentences, shape/operation hints, equation hints, and structure hints.
6. Fill or refine the brief's implementation note before writing code:
   - selected module name;
   - implementation basis: `paper-derived`;
   - equations, algorithms, or figure labels used;
   - expected input/output shape;
   - unresolved assumptions.
7. Implement the smallest runnable module that matches the paper evidence.
8. Add a CPU-compatible random-input smoke test.
9. Validate with `scripts/smoke_test_module.py`.
10. When working against a known sample case, validate with `scripts/evaluate_generated_module.py <sample-case-dir> <generated.py> --python <python-with-deps>`.
11. For batch forward tests, write generated modules as `case1.py`, `case_1.py`, `1.py`, or `1/module.py` under one generated directory, then run `scripts/evaluate_generated_batch.py <samples-root> <generated-root> --python <python-with-deps>`.

## Sample Alignment

Generated code does not need to use the exact paper name as a class name. It is acceptable when:

- paper candidates align directly with generated symbols; or
- paper candidates align with the sample reference file/symbols, and generated symbols align with the same reference symbols.

This covers source-code conventions such as generic `Model` classes while still requiring a traceable chain from paper evidence to generated code.

## Reconstruction Rules

- Prefer implementing the named plug-and-play unit, not the full model.
- Treat PDF text content as evidence, not the PDF file name or extraction wrapper metadata.
- Keep assumptions visible in comments or the final report when the paper omits details.
- Use standard PyTorch layers for common operations described by the paper, such as linear projections, convolutions, pooling, normalization, softmax attention, gating, decomposition, and loss computation.
- Do not invent benchmark claims, pretrained weights, dataset-specific constants, or official-code provenance.
- If tensor shapes are not explicit, choose a conventional minimal shape consistent with the domain and explain it.
- If the paper describes several tightly coupled units, implement them together in one file and expose the main block.

## Failure Modes

- `found_module: no`: do not generate code.
- Missing equations but clear block diagram: implement a conservative approximation and label it paper-derived.
- Missing shape details: add configurable dimensions and a smoke test using reasonable synthetic tensors.
- Ambiguous multiple modules: choose the highest-evidence candidate unless the user requested a specific one.
