# Narrative Framework — Finalized (2026-05-12)

## 论文定位 (Three-Sentence Thesis)

1. **主张**: Noise-target coupling is a first-order design dimension in diffusion-based detection.
2. **机制结论**: In dense multi-class settings, deterministic decoding suppresses supervisory diversity and can *reverse* the intended effect of entropic OT — argmax turns a smoother plan into **winner-takes-more** dynamics, where realized assignment support shrinks as ε grows.
3. **设计结论**: The right coupling is task-dependent; stochastic and structured decoders restore *controllable* diversity rather than universally outperforming random coupling.

## 三条贡献 (Fixed — no more, no less)

1. **Diagnostic framework** — H(V|Z), ρ(ε), η(ε), D_eff, and variance decomposition form a quantitative toolkit for analyzing noise-target coupling quality in diffusion detection.
2. **Mechanism discovery** — In dense multi-class chromosome detection, we identify:
   - Diversity collapse under deterministic OT (velocity entropy 3.841→0 nats, between-group variance −70%)
   - **Argmax reverse-concentration**: as ε increases, the Sinkhorn plan becomes smoother, but argmax-decoded realized support *shrinks* (D_eff declines monotonically), converting entropic smoothing into winner-takes-more dynamics
   - The optimal coupling depends on task structure (density, class confusion): random ≥ OT in dense multi-class; OT ≥ random in sparse single-class
3. **Design instances** — Sinkhorn sampling (restores ε as a controllable diversity knob) and GHSS (deploys diversity within domain-structured groups). Both are *framework-guided design examples*, not standalone method contributions.

## 关键修正 (5 Corrections from Earlier Draft)

| #   | Issue             | Old (Wrong)                    | New (Correct)                                                                                                                                                            |
| --- | ----------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | D_eff behavior    | "Argmax freezes D_eff at ~2.8" | Argmax D_eff *declines* from ~44→14 as ε grows; stochastic D_eff stable at ~42. The argmax trend is more damning than "frozen" — it's actively *anti-smoothing*.         |
| 2   | Random optimality | "Random coupling is optimal"   | Random is a surprisingly strong baseline in dense multi-class; OT wins in single-class. Coupling optimality is *task-dependent*.                                         |
| 3   | Density ratio     | δ = N/M, smaller = denser      | Use **target density = M/N** (targets per proposal). Chromosome: 46/500 = 0.092; COCO: ~7/500 = 0.014.                                                                   |
| 4   | Argmax framing    | "Argmax is a strawman"         | Argmax is a natural but incorrect decoder that reveals a general pitfall of entropic OT in discrete assignment.                                                          |
| 5   | CAM formalization | "Theorem: ∂D_eff/∂ε = 0 a.e."  | **Proposition**: π_argmax(ε) is piecewise constant in ε (changes only at ranking-change thresholds). D_eff trends are the *empirical consequence*, not the proof target. |

## 改稿顺序 (Revision Roadmap)

1. **Unify D_eff** — Reconcile definition, JSON, figure, body text, and appendix. Compute D_eff both per-target and per-group; pick one and be explicit.
2. **Rewrite contributions list** — Downgrade method claims, upgrade framework claims per the 3-item list above.
3. **Rewrite Section 3** — From "Why OT Fails" to "A Diagnostic Framework for Coupling Analysis." Framework first, failure as discovered application.
4. **Rewrite Section 4** — From "Method" to "Framework-Guided Design Instances." Sinkhorn sampling and GHSS as two instances of the same principle.
5. **Revise title, abstract, introduction** — Promise only what can be proven. Title direction:
   > *Diversity Over Efficiency: Analyzing Noise-Target Coupling in Diffusion Detection*
   > *— A Diagnostic Framework and Case Study in Dense Chromosome Detection*

## 理论原则 (Theoretical Guidelines)

- **Prove assignment-level properties first, then discuss metric-level phenomena.**
- The core provable claim: π_argmax(ε) is piecewise constant (changes only at ranking-change thresholds of the transport matrix).
- D_eff trends, ρ(ε) saturation, η(ε) growth are empirical consequences — not the proof target.
- ε three-regime phase diagram (OT-dominated / sweet spot / bias-dominated) is a *descriptive* framework validated by data, not a theorem.
