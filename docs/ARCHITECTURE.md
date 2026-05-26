# FlashStereo — Architecture Notes

## The bottleneck FlashStereo removes

FoundationStereo's two-stage TensorRT deployment splits inference into:

1. **`feature_runner`** (TRT engine): RGB stereo pair → multi-scale feature maps.
2. **GWC volume builder** (Python + CPU + custom PyCUDA kernel): builds the group-wise correlation cost volume from the quarter-resolution feat maps.
3. **`post_runner`** (TRT engine): feat maps + GWC volume → disparity.

The reference Python middle stage looks roughly like this:

```python
def build_gwc_volume_optimized(refimg_fea, targetimg_fea, normalize=True):
    # Step 1: D2H both feat maps (≈ 16 MiB each on a 480×640 480-channel input)
    refimg_fea_cuda = cuda.mem_alloc(refimg_fea.nbytes)
    targetimg_fea_cuda = cuda.mem_alloc(targetimg_fea.nbytes)
    cuda.memcpy_htod(refimg_fea_cuda, refimg_fea.astype(np.float32))
    cuda.memcpy_htod(targetimg_fea_cuda, targetimg_fea.astype(np.float32))

    # Step 2: per-group L2 norm on the CPU (35M reductions in numpy)
    ref_norm = np.linalg.norm(refimg_fea.reshape(B, G, K, H, W), axis=2)
    tar_norm = np.linalg.norm(targetimg_fea.reshape(B, G, K, H, W), axis=2)
    ref_norm = np.maximum(ref_norm, 1e-5)
    tar_norm = np.maximum(tar_norm, 1e-5)
    # H2D the norms ...
    cuda.memcpy_htod(ref_norm_cuda, ref_norm)
    cuda.memcpy_htod(tar_norm_cuda, tar_norm)

    # Step 3: PyCUDA gwc kernel
    gwc_kernel(...)

    # Step 4: D2H the GWC volume (≈ 28 MiB)
    cuda.memcpy_dtoh(cost_volume_np, cost_volume_cuda)
    return cost_volume_np
```

And then `post_runner.run_trt(...)` H2Ds every input again. Net traffic per inference: **≈ 100 MiB of avoidable host↔device copies + one CPU-bound reduction over 35 M floats**.

On a Jetson AGX Orin we measured 42.6 ms p50 inside `build_gwc_volume_optimized` alone — ~17% of the full pipeline.

## What FlashStereo changes

Three things, in order of impact:

### 1. GPU-resident chaining (the headline 1.92× speed-up)

Persistent device buffers are allocated once for every feat / post I/O tensor. After binding, the post_runner's input tensor addresses point directly at the feat_runner's output buffers. The CUDA execution graph for one inference becomes:

```
H2D(left, right)                  ─ once per frame
feat_runner.execute_async_v3      ─ writes outputs in place
GPU norm kernel ×2                 ─ all on device
GPU GWC kernel                     ─ all on device, writes gwc_volume in place
post_runner.execute_async_v3       ─ reads feat + gwc directly via device ptrs
D2H(disp)                          ─ once per frame
stream.synchronize()
```

Every op runs on the same CUDA stream — ordering is guaranteed by stream semantics. No explicit events / barriers needed.

### 2. CUDA norm kernel (replaces CPU `np.linalg.norm`)

Reduces 35M FP32 elements per inference on the GPU instead of bouncing them through the CPU. The kernel mirrors numpy's semantics (per-(b, g, h, w) L2 norm over the K channels of one group, clamped to >= 1e-5).

The kernel is intentionally simple — the win isn't its compute speed (the workload is small) but the absence of two host↔device round trips for the inputs and outputs.

### 3. Optional INT8 swap (additional 1.41×)

The two TensorRT engines can be rebuilt as INT8 (with FP16 fallback) using:

- `scripts/build_int8.py`: feat_runner INT8, calibrated on images.
- `scripts/gen_post_calib_data.py` + `scripts/build_int8_post.py`: post_runner INT8, calibrated on the **intermediate tensors** produced by the FP16 feat path, since post_runner does not take images as input.

INT8 quality on the Middlebury 2014 calibration set, evaluated against the FP16 reference on the same input:

| Config | cosine | L1 mean | rel L1 |
|---|---|---|---|
| INT8 feat + FP16 post | 0.999998 | 0.04 px | 0.06% |
| FP16 feat + INT8 post | 0.999989 | 0.12 px | 0.19% |
| INT8 feat + INT8 post | 0.999990 | 0.12 px | 0.19% |

The largest single-pixel disagreement we observed in 8 reps was about 35 disparity pixels on a 640-wide image — concentrated at fine occlusion boundaries. Mean disagreement is well under one pixel.

## What FlashStereo does NOT change

- **No model surgery.** The ONNX exports of feat_runner and post_runner are used as-is. The CUDA kernels for GWC and norm are byte-identical replacements for what the reference pipeline already does (we wrote the norm; the GWC kernel was already a custom PyCUDA kernel — we kept the same source).
- **No retraining.** Engine weights are unchanged. INT8 build uses standard TRT entropy calibration with public data.

## Why TensorRT alone leaves performance on the table

We profiled the *full* (single-engine) FoundationStereo variant separately. There, TensorRT's `execute_async_v3` is a single large async call: GPU compute dominates 99.96% of the latency, with no measurable Python overhead. **CUDA Graph wrapping or kernel rewriting buys < 1 ms.**

The two-stage variant is different because the Python middle stage is the one introducing the overhead. By keeping the Python boundary minimal (just one D2H at the end), FlashStereo reclaims those ≈ 100 ms. This is the same insight FlashRT exploits for VLA models — the kernels differ but the principle is shared.

## Numerical correctness

We verified FlashStereo's FP16 path is **bit-identical** to the reference pipeline by:

1. Capturing each intermediate tensor (feat outputs, ref/tar norm, GWC volume, final disp) at every stage of both pipelines.
2. Comparing pairwise: L1 = 0, L_inf = 0, cosine = 1.000000000 across all stages.

There was a single subtle bug along the way: the GWC reference kernel does **not** divide the inner sum by K, whereas an earlier draft of FlashStereo's GWC kernel did. The signature symptom: cosine = 1.0 (structure preserved) but rel L1 ≈ (K−1)/K ≈ 96.4% — an offset that exactly identifies an unintended divide-by-K. Both kernels match exactly now; the fix is preserved as a code comment.

## Where the time goes (Jetson AGX Orin, 480×640, p50)

| Stage | FP16 baseline (stock) | FlashStereo FP16 | FlashStereo all-INT8 |
|---|---:|---:|---:|
| H2D inputs | 0.3 ms | 0.3 ms | 0.3 ms |
| feat_runner | 32.9 ms | 32.9 ms | 31.7 ms |
| GWC volume builder | 42.6 ms (Python+CPU+GPU mix) | < 1 ms (GPU only) | < 1 ms (GPU only) |
| post_runner | 144.8 ms | 89.6 ms | 55.0 ms |
| D2H disp | 0.04 ms | 0.04 ms | 0.04 ms |
| Python overhead (binds, sync, etc.) | ~ 18 ms | < 1 ms | < 1 ms |
| **End-to-end p50** | **238 ms** | **124 ms** | **88 ms** |

The "Python overhead" line in the stock pipeline is the cost of re-binding tensors, allocating per-call buffers in `run_trt`, and the sync barriers between sub-stages. FlashStereo's persistent-buffer approach eliminates almost all of it.
