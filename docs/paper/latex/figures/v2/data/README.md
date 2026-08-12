# Figure data exports

This directory contains machine-readable inputs for the four figures used in
the manuscript. Plotting scripts read these files directly; values are not
duplicated in plotting code.

- `source_speed_accuracy_cross_dataset.json`: accuracy and same-hardware
  latency operating points for Fig. 1.
- `source_lqcr_test_ap_curve.json`: fixed-model strict-IoU analysis for Fig. 3.
- `source_small_object_cross_dataset.json`: held-out-test scale analysis for
  Fig. 4. KaryoFlow and LQCR use paired means from three independently trained
  detectors; AP-S is interpreted only within each dataset.

The model-overview figure uses representative detections and contains no
additional performance claim. Formal result provenance is maintained in the
experiment database outside the submission package.
