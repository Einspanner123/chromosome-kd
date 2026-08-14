# MMDetection dependency boundary

KaryoFlow depends on the public `mmdet==3.3.0` package. MMDetection is not
vendored in this repository.

## Audit result

Before cleanup, the repository contained 1,777 tracked files under `mmdet/`
(approximately 20 MB). The tree originated from MMDetection 3.3.0 and also
contained 1,174 copied `.mim` configuration files plus 92 copied upstream
configuration files. Active KaryoFlow configurations do not inherit from those
copies; their required model, data, and schedule definitions are self-contained
under `experiments/configs/`.

The only project-authored runtime change after the 3.3.0 import added extra
per-class values to `CocoMetric.eval_results` for visualization backends. It did
not alter aggregate COCO AP computation or the prediction/evidence export path.
The cleaned project therefore uses the unmodified public evaluator and keeps
paper metrics reproducible through saved predictions and independent
`pycocotools` recomputation.

## Pinned identity

- Package: `mmdet==3.3.0`
- PyPI wheel: `mmdet-3.3.0-py3-none-any.whl`
- Wheel SHA-256:
  `2e23e291281ac57e7dccf8678e957da45fbe560ce78a1f5ded6afeccd3730f17`
- Tested with `mmcv==2.1.0` and `mmengine==0.10.5`

Project-specific integration belongs in `experiments/mmdet_bridge/`; changes to
upstream MMDetection must not be copied back into the repository. If a future
upstream patch becomes unavoidable, maintain it as an explicit, minimal patch
with its own compatibility test and dependency pin.

## Verification

With the vendored tree absent and MMDetection imported from `site-packages`:

- all 92 project tests pass;
- all 38 canonical model/configuration combinations build;
- four generation stages, 76 inference variants, and eight deployment/analysis
  protocols pass their audits;
- the evidence database and 261-run paper route matrix remain unchanged;
- the project builds a standalone `karyoflow-0.1.0` wheel.

The original vendored source and its local evaluator patch remain recoverable
from `refs/tags/archive/pre-cleanup-20260815`.
