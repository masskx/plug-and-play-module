# Module Detection Guide

Use this guide to decide whether a paper and/or repository contains a **usable** plug-and-play module. Make the decision before generating code, and make it per candidate - one paper can hold several modules, or none.

## Decision labels

- `found_module: yes`: at least one candidate passes every usability criterion below.
- `found_module: no`: after review, no candidate passes. This is a normal result, not a failure.
- `found_module: uncertain`: a component probably exists but the available material cannot confirm it (thin method description, lossy extraction, missing appendix, compiled-op dependency, paper not retrievable). Say what is missing.

`no` requires that the material was actually reviewed. A scanned PDF with empty text, a title with no retrievable paper, or a method section you did not read supports `uncertain` at most.

## Usability criteria

A candidate is a usable plug-and-play module only if all hold:

1. **Bounded interface.** You can state its input tensor(s) (with layout, e.g. `B,C,H,W`) and output tensor(s) or scalar loss.
2. **Fully specified.** Every operation between input and output is given by code, equations, an algorithm, or an unambiguous figure. Reconstruction needs no invented core mechanism - only conventional details (padding, initialization, a standard activation) that you can list as assumptions.
3. **Separable.** It runs without the rest of the paper's pipeline: no dataset objects, no hard-coded task heads, no training-loop state, no pretrained checkpoint, no special data format baked in.
4. **Reusable.** It can be inserted into models other than the one in the paper, with configurable sizes. Domain-specific modules are fine (point-cloud, time-series, graph, audio) if reusable within that domain.
5. **Non-trivial.** It is the paper's contribution or at least a distinct operation, not a standard layer (`Conv-BN-ReLU`, plain MLP, vanilla multi-head attention, `LayerNorm`) re-labelled.
6. **Runnable.** It executes on CPU with standard dependencies, or its extra dependency is clearly pip-installable. A module that only works with a compiled CUDA extension that cannot be replaced is `uncertain`, with the dependency named.

Losses, regularizers, augmentations/samplers implemented as independent transforms, positional encodings, normalization/filter layers, adapters (LoRA-style), fusion units, token mixers/selectors, and pooling layers qualify under the same tests.

## Disqualifiers (strong `no` signals)

- The contribution is a dataset, benchmark, metric, survey, position paper, theory, annotation protocol, or deployment system.
- The method is an end-to-end architecture with no separable component, or the only "module" is a standard layer arrangement.
- A training recipe (curriculum, data mixture, schedule) without an independent operation.
- Reproducing it would require inventing the central computation (the paper says "details in supplementary" and the supplement is unavailable -> `uncertain`, not `yes`).
- The only code is trainers, dataset classes, scripts, configs, or whole models.

## Positive signals

- Named block/layer/loss/attention/adapter/filter/selector/fusion unit with a diagram where inputs and outputs are labelled.
- Equations or pseudocode for a bounded operation; stated tensor shapes; parameter/FLOPs analysis of the unit.
- Ablations that add or remove exactly this unit; claims such as "plug-and-play", "can be inserted into existing networks", "drop-in replacement", "architecture-agnostic".
- Compact code class with `forward(x) -> y`, explicit constructor arguments, and a name that matches the paper.

Keyword counts (`module`, `layer`, `ablation`, `lightweight`) appear in nearly every ML paper and are **not** evidence by themselves. `analyze_paper_modules.py` therefore reports separate structural evidence (plug-in claims, equation lines, shape mentions, algorithm/figure references, module ablations) and the paper's contribution type; trust those over the old keyword score.

## Procedure

1. Run `scripts/review_gate.py` (it runs the paper analysis, code scan, and alignment). Treat its output as a lead list.
2. Read the abstract/introduction contribution list, the method section, the ablation tables, and the appendix. Skim the repo README and the top code candidates.
3. For each candidate fill this card (keep it for the report):

   ```text
   name / aliases:
   source: paper section / figure / equation, or file:line
   input:      (layout, dtype, optional masks)
   output:     (layout) or scalar
   operations: ordered list
   hyperparameters / defaults:
   dependencies: torch only | extra pip package | compiled op
   independent of pipeline: yes/no (why)
   criteria 1-6: pass/fail
   decision: export | reject (reason) | uncertain (what is missing)
   ```
4. Export the candidates marked `export`. Record the rejected ones with a one-line reason.
5. If none are marked `export` and none are `uncertain` -> `found_module: no`.

## Boundary rules

- A full model can contain extractable modules: extract the separable component, not the training system.
- A loss is a module if it has clear inputs and works in other training loops. A sampler/augmentation qualifies when implemented as an independent transform.
- If only the code reveals the boundary, define the unit from the code and say so.
- Multiple cooperating units presented as one contribution stay together unless each is independently usable; independent ones are exported separately.
- Variants of the same module (e.g. 1D/2D/3D versions) become one file with a dimension argument if the paper treats them as one idea; otherwise separate files.

## The standard "no module" result

When the verdict is `no`, produce exactly this and nothing else - no code, no files, no speculative alternatives:

```text
After review, no usable plug-and-play modules were found in "<title or repository>".

Basis:
- <1-3 concrete reasons, e.g. "the paper's contribution is a dataset and evaluation protocol">

Candidates considered and rejected:
- <name>: <reason>
```

Chinese form: `经审查，“<题目/仓库>”中未发现可用的即插即用模块。` followed by `依据：` and `已排除的候选：` lists. Use the language the user wrote in. Set `found_module: no`, `module_path: none`, `implementation_basis: none` in any structured report.

## Evidence checklist

Record concise evidence for: the paper section/figure/algorithm/equation or code path behind the decision; module name and aliases; expected input and output; key dependencies; and whether the export is paper-derived or source-code-derived.
