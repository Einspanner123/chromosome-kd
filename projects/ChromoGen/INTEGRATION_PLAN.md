# ChromoGen UNet → LDMDet Feature Injection (Future Work)

## Concept

Leverage chromosome-generation UNet encoder features (trained in ChromoGen Phase1) 
as external knowledge to enhance LDMDet detection backbone.

## Architecture Sketch

```
Input Image (H×W×3)
    ├── LDMDet ResNet-50 → FPN (P2-P5)         [detection path]
    └── ChromoGen UNet Encoder (frozen)          [generative path]
            │
            DownBlock1 (H/8)  ──proj──→ + P2
            DownBlock2 (H/16) ──proj──→ + P3
            DownBlock3 (H/32) ──proj──→ + P4
            MidBlock   (H/64) ──proj──→ + P5
```

Each UNet feature map projected to 256ch via 1×1 conv, then element-wise
added to corresponding FPN level before ROI extraction.

## Why UNet, not VAE

- VAE: pretrained on LAION-5B natural images, frozen during ChromoGen — does NOT
  have chromosome-specific knowledge
- UNet: trained on chromosome synthesis during ChromoGen Phase1 — DOES have
  chromosome structural understanding (banding patterns, centromere position,
  arm ratios)

## Requirements

- ChromoGen checkpoint: `work_dirs/chromogen_phase1/checkpoint_epoch_100.pt`
- No internet needed — all weights are in the checkpoint
- Need to implement: UNet encoder extraction, 1×1 projection layers, FPN fusion

## Risks / Open Questions

1. Generative features may not align with discriminative features needed for detection
2. UNet ~400M params (frozen), adds ~50ms inference overhead
3. May not improve over ResNet-only baseline — redundancy risk
4. Story mismatch with core paper (coupling strategies)

## Decision: Deferred to Future Work

Do not include in current paper. Keep as independent follow-up study.
Core paper narrative: coupling strategies in diffusion detectors.
