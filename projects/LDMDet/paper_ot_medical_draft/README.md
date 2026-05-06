# OT Medical Paper Draft

This directory contains the first Markdown draft of the medical-imaging paper on OT failure and stochastic coupling in chromosome detection.

## Structure
- `draft.md`: main paper draft
- `figures/`: generated placeholder figures and future final figures
- `scripts/`: scripts for figure generation
- `tables/`: table templates and result placeholders

## Usage
Generate placeholder figures with:

```bash
conda run -n chromo python scripts/generate_placeholder_figures.py
```

Then open `draft.md` in the editor or render it with your preferred Markdown tool.

## Notes
- All quantitative fields remain `TBD` until the final experiments are frozen.
- The current environment check shows `matplotlib==3.7.5` is available in the `chromo` environment, while the default shell environment does not include it.
- Use the `chromo` environment for any figure regeneration or future conversion to publication-style plots.
- Replace placeholder figures with final publication-quality plots later.
- Keep all paper-specific assets in this directory to avoid polluting the main project tree.
