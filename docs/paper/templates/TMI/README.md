# IEEE Transactions on Medical Imaging (TMI) LaTeX Template

## Source
All `IEEEtran.*` and `bare_*.tex` files were downloaded from CTAN,
the official IEEE LaTeX template mirror:
- Package page: https://ctan.org/pkg/ieeetran
- Direct zip:   https://mirrors.ctan.org/macros/latex/contrib/IEEEtran.zip
- IEEE Template Selector: https://template-selector.ieee.org/

IEEE TMI does **not** ship a journal-specific .cls file. The journal
instructs authors to use the standard `IEEEtran.cls` in `journal` mode.
Source of truth for TMI-specific rules:
- TMI Author Instructions (Rev 10.2, Oct 9 2025):
  https://ieeetmi.org/authors-instructions/
- Submission checklist: https://ieeetmi.org/submission-checklist/
- Scope: https://ieeetmi.org/scope/

## File Manifest

### Core class and bibliography style (required for any submission)
- `IEEEtran.cls`        — IEEE LaTeX class, v1.8b (2015/08/26), the latest release.
- `IEEEtran.bst`        — default IEEE numeric bibliography style (recommended for TMI).
- `IEEEtranS.bst`       — sorted variant.
- `IEEEtranN.bst`       — named-reference variant (NOT used by TMI).
- `IEEEtranSA.bst`      — sorted + annotation variant.
- `IEEEtranSN.bst`      — sorted + named variant.

### IEEE abbreviation bibliographies (optional helpers)
- `IEEEabrv.bib`, `IEEEfull.bib`, `IEEEexample.bib` — sample bibliography databases.

### Bare-bones skeleton templates from CTAN
- `bare_jrnl.tex`           — generic IEEE Transactions journal article skeleton.
- `bare_jrnl_compsoc.tex`   — Computer Society journal variant (NOT for TMI).
- `bare_conf.tex`           — IEEE conference paper skeleton (NOT for TMI).

### TMI-specific example (created for this project)
- `TMI_template.tex`        — chromosome-detection example pre-filled with
                               the project's title/abstract skeleton, TMI
                               editorial comments inline, and a thebibliography
                               block so it compiles standalone.
- `TMI_template.pdf`        — proof-of-compile output (1 page).

### Documentation
- `IEEEtran_HOWTO.pdf`      — user manual for IEEEtran.cls.
- `IEEEtran_bst_HOWTO.pdf`  — user manual for IEEEtran.bst.

## How to Compile
```bash
cd /home/linkst/workspace/projects/chromosome-kd/docs/paper/templates/TMI
pdflatex TMI_template.tex
# (For a real bibliography instead of thebibliography:)
# bibtex TMI_template && pdflatex TMI_template.tex && pdflatex TMI_template.tex
```

## Key TMI Formatting Rules (verify against the live author-instructions page before submitting)
1. `documentclass[journal,10pt]{IEEEtran}` — TMI uses journal mode.
2. Initial submission: **<= 10 pages including references**. Hard cap;
   over-limit manuscripts are returned without review.
3. Double-column, single-spaced, justified, 10pt font.
4. Abstract < 250 words; include `IEEEkeywords`.
5. Do NOT include author biographies in initial submission.
6. Figures and tables must appear inline in the main text.
7. References: IEEEtran.bst with author-name format (e.g., `J. Smith`).
8. Cover letter strongly discouraged unless extending a conference paper
   or disclosing prior journal review.
9. Graphical abstract is optional; if provided, upload as a supporting
   document (660x295 px, >=300 dpi, <45 KB).
10. Open-source code strongly encouraged; provide GitHub URL.
11. Final accepted papers exceeding 8 printed pages incur mandatory
    overlength page charges: $250/page for pp. 9-10, $350/page from p. 11.

## Submission Portal
- IEEE Author Portal (primary): linked from https://ieeetmi.org/
- ScholarOne (legacy revisions): https://mc.manuscriptcentral.com/tmi
- Editorial office: tmi@computer.ieee.org
