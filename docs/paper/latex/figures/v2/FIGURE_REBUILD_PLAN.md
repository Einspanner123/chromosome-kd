# Paper figure reconstruction plan

## Narrative order

The paper uses a generation--interaction--decision narrative. Rectified Flow
and DPM-Solver++ are adopted foundations, not claimed inventions. The retained
independent contribution is LQCR at the final decision stage. Stochastic
Coupling is a controlled negative result, and OCGR remains outside the final
model unless a predeclared paired multi-seed gate is passed.

1. **System overview (`fig01`)**: define the three stages and clearly separate
   adopted components from LQCR. No provisional module may appear as final.
2. **LQCR principle (`fig02`)**: connect the exact posterior factorization to
   strict final-only causal isolation and the fixed-checkpoint Dataset 2 gain.
3. **Clean paired replication (`fig03`, pending)**: show every Dataset 1 seed,
   paired LQCR effect, mean effect, and uncertainty. This replaces legacy
   cross-lineage significance plots.
4. **Precision bottleneck (`fig04`)**: show oracle headroom, strict-IoU AP, and
   scale/overlap recall. Ground-truth-IoU ranking is labeled as an oracle.
5. **Stratified LQCR effect (`fig05`, pending)**: determine whether gains are
   concentrated in small, near, or overlapping targets using identical boxes,
   classes, and checkpoints before ranking.
6. **Efficiency (`fig06`)**: report same-A6000 latency decomposition and the
   speed--accuracy frontier. DPM-Solver++ is credited only for efficiency.
7. **Qualitative cases (`fig07`)**: use editable vector boxes and labels over
   raster microscopy crops; selected images are illustrative only.

## Evidence gates

- A metric enters a figure only from a checkpoint-linked JSON export.
- LQCR causal claims require the same checkpoint, coordinates, classes,
  renewal decisions, and solver trajectory; only final scores may differ.
- Multi-seed plots must expose seed-level points and cannot pool incompatible
  experimental lineages.
- Hardware comparisons require the same RTX A6000 protocol at 512 x 512,
  batch size one, 10 warmups, and 500 timed iterations.
- OCGR receives a main-paper figure only if the corrected paired experiment
  improves the predeclared primary metric across seeds; otherwise it remains a
  documented candidate or negative result outside the final method.

## Current state

Figures 1, 2, 4, 6, and 7 are generated as PDF/SVG/PNG and compiled in the
paper. Figures 3 and 5 are deliberately blocked on the running clean Dataset 1
baselines and their paired final-only LQCR exports.
