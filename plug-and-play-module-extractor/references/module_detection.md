# Module Detection Guide

Use this guide when deciding whether a paper contains a plug-and-play module.

## Decision Labels

- `found_module: yes`: the paper describes a reusable component that can be inserted into another model, loss, feature pipeline, or training objective with bounded inputs and outputs.
- `found_module: no`: the paper mainly contributes a dataset, benchmark, full system, training recipe, evaluation protocol, or end-to-end architecture without a separable reusable component.
- `found_module: uncertain`: evidence suggests a component may exist, but the paper does not provide enough detail to reconstruct or isolate it confidently.

Always make this decision before generating code.

## Strong Positive Signals

- Named block, layer, loss, attention, adapter, filter, selector, fusion unit, normalization layer, prompt unit, point-cloud block, time-series block, or feature refinement module.
- Architecture diagram where a component has clear inputs, transformations, and outputs.
- Equations or pseudocode for a bounded operation.
- Ablations that add/remove a specific module.
- Paper language such as "module", "block", "plug-and-play", "can be inserted", "lightweight", "replacement", "adapter", or "loss".
- Official code contains a compact class/function matching the paper name or abbreviation.

## Strong Negative Signals

- Contribution is only a dataset, metric, survey, benchmark, annotation protocol, or training corpus.
- Method is inseparable from a full pipeline and lacks clear module boundaries.
- The paper provides only high-level system design with no operations, shapes, algorithms, or code.
- Reusable behavior would require inventing major missing architecture details.

## Boundary Rules

- A full model can still contain extractable modules. Extract the separable components, not the entire training system.
- A loss function can be a plug-and-play module if it has clear inputs and can be used in other training loops.
- A data augmentation or sampling strategy can qualify if it is implemented as an independent transform.
- If only the official code reveals the boundary, use the code to define the extracted unit and cite that the boundary came from source inspection.
- If multiple cooperating modules are presented as one contribution, keep them together unless they are independently usable.

## Evidence Checklist

Record concise evidence for:

- paper section, figure, algorithm, equation, or code path used for the decision;
- module name and aliases;
- expected input tensors or objects;
- expected outputs;
- key dependencies;
- whether the output is paper-derived or source-code-derived.
