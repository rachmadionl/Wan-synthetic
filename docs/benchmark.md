# Wan-Animate-2 benchmark: v2 baseline vs v3 (8 steps)

19 avatars × 16 s driving video (`motion_25fps.mp4`, 400 frames), native 1080p
(1904×1072 generated → 1920×1080 Lanczos), Distillation model, per-person captions,
`regions` color anchoring, seed 42. Modal, 8× H100 per clip, FA3 attention.

| | v2 baseline (`results_v2/`) | v3 (`results_v3/steps8/`) |
|---|---|---|
| Changes | — | `WAN_PRECOMPUTE=1` (driving-window encodes, ref/CLIP/T5 cached) + 8 denoise steps (vs 10) |
| Clips ok | 19/19 | 19/19 |
| Processing time / clip (mean, min–max) | 627 s (607–666) | **532 s (523–554), −15%** |
| Per window (windows 2–5) | ~114 s | ~94 s (prep 8.8 / denoise 70 / decode 15.5) |
| Cost / output second | ~$0.38 | **~$0.32** |
| Cost / 16 s clip | ~$6.10 | ~$5.20 |
| Cost, 19 clips | ~$115 | ~$98 |

## Quality (v3 vs v2, full-clip, all 400 frames)

- Mean SSIM **0.987**, mean PSNR **41.7 dB** over 19 clips (range 0.967–0.994 / 33.6–47.5 dB).
- Lowest: 4d45b0c9 (33.6 dB) and dcbfcccf (36.4 dB). Their worst frames (321, 162)
  differ only in micro-pose/expression (eyelid opening, slight head angle); sharpness,
  skin texture, color, identity and clothing are equivalent.
- Earlier single-clip check (d3c60dae): detail energy 3.53 vs 3.57 — no softening.

Verdict: no visible quality loss at 8 steps; v3 is the new default.

## Remaining overhead (future work)

Per window, every GPU redundantly runs prep (~8.8 s, y_reft encode) and decode
(~15.5 s): ~25% of window time. Splitting decode across ranks could cut another ~10–15%.

## Reproduce

```
export PATH=/opt/conda/envs/wan_animate_2/bin:$PATH
WAN_PRECOMPUTE=1 WAN_ATTN=fa3 WAN_GPU='H100:8' modal run modal_app.py::batch \
  --frames-dir first_frames --driving motion_25fps.mp4 --out-dir results_v3/steps8 \
  --width 1080 --height 1920 --short-side 0 --final-size 1920x1080 --steps 8 --max-containers 3
```

## v4: de-duplicated prep/decode (`WAN_SPLIT_VAE=1`), tested on d3c60dae

- VAE decode and continuity-frame encode split into 8 column strips (one per GPU,
  12-latent-column halo) instead of repeated on every GPU; driving-video resize split
  across GPUs; per-window 4 GB float64 batch no longer built under precompute.
- Per window: prep 8.8 → 2.7 s, decode/post 15.5 → 6.2 s, denoise unchanged (69.6 s).
- Clip: 545 → **441 s (−19%)**, est. **~$0.26/output s** (~$4.20/clip).
- Output vs v3 (same seed/steps): PSNR 49.5 dB mean, 47.7 dB min, SSIM 0.992 —
  bf16 rounding-level differences (~1/255 on a few % of pixels), visually identical.

### v4 full run (`results_v4/steps8/`, 19/19 ok)

| | v2 | v3 | v4 |
|---|---|---|---|
| Time / clip (mean) | 627 s | 532 s | **420 s** (413–441) |
| $ / output s | ~$0.38 | ~$0.32 | **~$0.25** |
| 19 clips | ~$115 | ~$98 | **~$77** |

v4 vs v3 over all 19 clips: SSIM 0.993, PSNR 48.6 dB mean. Lowest 4d45b0c9
(43.6 dB mean, 33.5 dB at frame 324): the rounding difference compounds over
windows into a slightly different micro-pose; side-by-side the frames are
indistinguishable in detail, color and identity.

## Denoise-step profile and v5 experiments (d3c60dae)

One step on 8×H100 (GPU time 7.6 s): FA3 attention 63% (at ~660 TFLOPS, near bf16
peak), elementwise/copies ~21%, linear layers 9.5%, NCCL 4.7%.

- **FP8 attention (`WAN_ATTN=fa3fp8`)**: kernel 1.48× faster, clip −10% (396 s), but
  output drifts (27 dB vs v4) and looks overexposed/harsher, with identity drift. Rejected.
- **Exact cleanups** (rotated reference keys cached per window; bf16 attention inputs):
  bit-identical to v4, but only ~0.5% faster. Kept (harmless).
- Conclusion: exact speedups of the denoise step are exhausted; the remaining levers
  (fewer steps, step caching) trade quality.
