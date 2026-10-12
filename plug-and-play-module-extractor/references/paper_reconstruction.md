# Paper-Derived Reconstruction Guide

Use this guide when no source code is available (or the supplied code does not contain the paper's module) and the module must be rebuilt from the paper. The result is always labelled `paper-derived`.

## Required sequence

1. Extract the paper text (`scripts/extract_pdf_text.py`). Check that equations and the method section survived; read the PDF pages directly if not.
2. Run `scripts/review_gate.py --paper-text paper.txt` and confirm the verdict with `module_detection.md`. Stop with the standard no-module result if the verdict is `no`.
3. Build the brief: `scripts/build_reconstruction_brief.py paper.txt --output brief.md`. It lists candidates, structural evidence, shape/operation/equation/structure hints, and an empty module card and assumption ledger.
4. Read the paper sections around each candidate - method, figures and their captions, algorithm boxes, ablations, appendix, implementation details. Search for official or well-known code once more (title resolver `--github`, WebSearch); if it exists, switch to the code-based workflow.
5. Fill the module card and the assumption ledger (below) **before writing code**.
6. Implement the smallest module that matches the evidence. Prefer standard PyTorch layers for conventional operations (linear/conv projections, pooling, normalization, softmax attention, gating, decomposition, loss terms).
7. Add the property checks and the CPU smoke test.
8. Validate with `scripts/check_module_contract.py <module.py> --run`.
9. When working against a known sample with reference code: `scripts/evaluate_generated_module.py <sample-dir> <generated.py> --python <python-with-deps>`; for batches use `scripts/evaluate_generated_batch.py`.

## Fidelity ladder

Rely on the highest rung available for each operation, and record the rung in the ledger:

1. **Code/pseudocode in the paper or appendix** - implement literally.
2. **Equations with defined symbols** - implement literally; resolve index/shape conventions from the surrounding text.
3. **Labelled diagram** (inputs, operations, arrows, tensor sizes) - follow the diagram; confirm against the prose.
4. **Prose description** - implement the most direct reading; note alternatives you rejected.
5. **Domain convention** (e.g. 7x7 spatial kernel, reduction ratio 16, `eps=1e-5`) - only for peripheral details, never for the core mechanism.

If the core mechanism sits on rung 4 or 5 and cannot be pinned down, the verdict is `uncertain`, not `yes`. Do not invent the missing mechanism.

## Assumption ledger

Keep it in the module docstring (short) and the report:

```text
operation / detail        | source (Eq./Fig./Sec.)         | rung | assumption or choice
global pooling type       | Eq. 1                          | 2    | average over H,W
kernel size k             | Sec. 3.2 "adaptive in channels"| 4    | k = odd(|log2(C)/2 + 1/2|) as stated
activation after fc1      | not specified                  | 5    | ReLU (convention)
```

Everything on rung 5 and every rung 4 choice goes in the report's `notes`.

## Reconstruction rules

- Implement the named plug-and-play unit, not the whole model.
- Treat PDF text as evidence, not the file name or extraction metadata.
- Make dimensions configurable (channels, heads, kernel sizes, reduction ratios) with the paper's defaults where stated.
- If tensor shapes are not explicit, choose the conventional layout for the domain (`B,C,H,W` images; `B,N,C` tokens/point features; `B,T,C` series) and state it.
- Do not invent benchmark claims, pretrained weights, dataset-specific constants, or official-code provenance.
- If several tightly coupled units are described, implement them together and expose the main block; independent units go in separate files.
- Several variants (e.g. 1D/2D): one module with a clear switch if the paper presents one idea, separate files otherwise.

## Verification without a reference implementation

Put cheap assertions in the smoke test, derived from what the paper states:

- **Shape contract**: output shape equals the documented shape for at least two input sizes (catches hard-coded resolutions).
- **Parameter / complexity claims**: if the paper reports parameter count or FLOPs for given settings, compare (`sum(p.numel() ...)`) and report the match or the gap.
- **Invariances**: permutation equivariance/invariance for set or token modules, identity at initialization for residual/adapter modules that claim it, mask handling for padded sequences.
- **Gradient flow**: `backward()` on a scalar and check that every parameter receives a gradient.
- **Determinism**: same input twice in eval mode gives the same output.
- **Ablation hints**: if the paper ablates a component (e.g. removing the residual), expose it as a constructor flag only if the paper defines it.

## Failure modes

- `found_module: no` -> standard no-module result; no code.
- Missing equations but a clear diagram -> conservative implementation, labelled `paper-derived`, ledger shows rung 3.
- Missing shape details -> configurable dimensions and reasonable synthetic inputs.
- Core operation not recoverable -> `uncertain`; list exactly what is missing (for example "supplementary Eq. 7").
- Several candidate modules -> export each independent one that passes the criteria; do not pick one silently unless the user asked for a specific one.

## Sample alignment (regression only)

When validating against a sample case, generated code need not use the paper's exact class name. It is acceptable when paper candidates align with generated symbols directly, or when both align with the sample reference file/symbols. This preserves a traceable chain from paper evidence to generated code.
