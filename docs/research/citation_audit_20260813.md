# KaryoFlow TMI citation audit (2026-08-13)

## Scope and result

- Manuscript citation occurrences: 54 after correction.
- Unique cited works: 47.
- Missing BibTeX keys: 0.
- Final numbered references: 47.
- Cited works verified through persistent primary records:
  - 23 Crossref-registered DOI records;
  - 2 DataCite/arXiv DOI records;
  - 6 official arXiv records;
  - 1 official CVF record;
  - 1 official dataset page;
  - 14 established conference papers checked against their official proceedings or official preprints.
- The BibTeX library contains 13 uncited entries; BibTeX excludes them from the submitted reference list.

All 47 works in the compiled manuscript were found to be real. No fabricated
paper, invalid cited title, or unresolved citation key remains.

## Corrections made

1. **YOLOX taxonomy.** The introduction incorrectly grouped YOLOX with
   anchor-based detectors. YOLOX explicitly adopts an anchor-free design. The
   sentence now distinguishes region-based Cascade R-CNN from anchor-free
   YOLOX. Source: <https://arxiv.org/abs/2107.08430>.

2. **AdaLN-Zero source.** AdaLN-Zero was incorrectly attributed to Dhariwal and
   Nichol. It is now cited to Peebles and Xie's DiT paper, which introduces the
   adaLN-Zero block. Source:
   <https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html>.

3. **Unsupported transfer-learning claim.** The chromosome transfer-learning
   paper does not report overlap-related cropping artifacts. The manuscript now
   states its actual finding: transfer gains depend on staining-domain
   similarity and target-data quality. Source:
   <https://doi.org/10.1038/s41598-026-38662-w>.

4. **Overlap claim source.** The overlap-related workflow statement is now
   supported by the integrated karyotyping paper, whose upstream quality-control
   gate excludes morphologically unreliable overlapped metaphases. Source:
   <https://doi.org/10.1038/s41598-026-52728-9>.

5. **Chromosome-work citation placement.** Joint segmentation-classification,
   chromosome classification, enumeration, straightening, and metaphase
   detection are now cited separately instead of assigning all tasks to a
   combined citation group.

6. **Overbroad negative claim.** The assertion that existing flow detectors do
   not analyze solver-history/renewal interaction was replaced with a narrower
   statement of KaryoFlow's complementary focus.

7. **Calibration boundary.** The text now explicitly separates LQCR ranking
   alignment from posterior-probability calibration; the medical-segmentation
   calibration paper is cited only for the latter.

8. **Bootstrap interval typography.** The malformed interval
   `[-0.0033,,0.0022]` was corrected to `[-0.0033, 0.0022]`.

## Bibliographic metadata corrected

- **ChromosomeNet:** publication year corrected from 2024 to 2025 and DOI
  `10.1109/OJEMB.2024.3512932` added. The registered record is volume 6,
  pages 227-236. Source: <https://doi.org/10.1109/OJEMB.2024.3512932>.
- **DPM-Solver++:** volume 22, issue 4, pages 730-751 added. Source:
  <https://doi.org/10.1007/s11633-025-1562-4>.
- **Hungarian method:** DOI `10.1002/nav.3800020109` added. Source:
  <https://doi.org/10.1002/nav.3800020109>.
- **DiffuBox:** full author list and official preprint URL added. Source:
  <https://arxiv.org/abs/2405.16034>.
- **Cell Image Library dataset:** title aligned with the DOI registry record.
  Source: <https://doi.org/10.7295/W9CIL54816>.

## Recent-work authenticity checks

The recent references most vulnerable to authenticity or date errors were
individually checked:

- FlowDet: <https://arxiv.org/abs/2512.16771>
- DeFloMat: <https://arxiv.org/abs/2512.22406>
- Cellular-microscopy flow matching:
  <https://arxiv.org/abs/2603.26790>
- Optimized subclass-prior flow matching:
  <https://arxiv.org/abs/2605.16469>
- Sister-chromatid-cohesion detection:
  <https://doi.org/10.1038/s41598-026-43009-6>
- Integrated automated karyotyping:
  <https://doi.org/10.1038/s41598-026-52728-9>
- Two-step chromosome transfer learning:
  <https://doi.org/10.1038/s41598-026-38662-w>
- CHROMA foundation model:
  <https://doi.org/10.1038/s41698-026-01383-4>
- Fourier diffusion in TMI:
  <https://doi.org/10.1109/TMI.2025.3553805>
- Domain-generalized discrete diffusion in TMI:
  <https://doi.org/10.1109/TMI.2025.3564474>
- Latent-guided diffusion and nested ensembles in TMI:
  <https://doi.org/10.1109/TMI.2025.3583974>

## Remaining caveat

The RST/Chromosome20240904 source is a Roboflow Universe dataset page rather
than a peer-reviewed data paper. The link is retained because it is the public
release source, but the manuscript correctly treats it as a dataset citation,
not as peer-reviewed methodological evidence.
