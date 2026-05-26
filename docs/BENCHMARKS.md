# FlashStereo — Benchmarks

All numbers are measured on:

- **Jetson AGX Orin 64 GB** (Ampere SM87)
- **JetPack 6.0 / L4T R36.3**
- **TensorRT 10.3 / CUDA 12.6** (with `NVIDIA_DISABLE_REQUIRE=1` bypassing the 12.2 driver gate)
- **Input resolution**: 480×640 stereo pair (grayscale, replicated to 3 channels)
- **Methodology**: 3 warmup iterations + 8-rep p50, single-stream async execution

## End-to-end latency

| Configuration | p50 (ms) | Throughput (Hz) | Speedup |
|---|---:|---:|---:|
| Stock TensorRT pipeline (FP16, Python GWC builder) | 238.32 | 4.20 | 1.00× |
| FlashStereo FP16 (GPU-resident) | 124.78 | 8.01 | 1.91× |
| FlashStereo + INT8 feat + FP16 post | 122.46 | 8.17 | 1.95× |
| FlashStereo + FP16 feat + INT8 post | 89.51 | 11.17 | 2.66× |
| **FlashStereo + INT8 feat + INT8 post** | **87.93** | **11.37** | **2.71×** |

## Disparity quality (INT8 vs. FP16 reference)

Measured on the same input pair, against the FP16 FlashStereo output (which is itself bit-identical to the stock pipeline).

| Configuration | cosine | L1 mean (px) | L_∞ (px) | rel L1 |
|---|---|---:|---:|---:|
| INT8 feat + FP16 post | 0.999998 | 0.04 | 18.5 | 0.06% |
| FP16 feat + INT8 post | 0.999989 | 0.12 | 35.1 | 0.19% |
| INT8 feat + INT8 post | 0.999990 | 0.12 | 31.8 | 0.19% |

The L_∞ values are concentrated at thin occlusion boundaries; mean disagreement is well under one disparity pixel.

## Per-stage breakdown (FP16 path)

| Stage | Stock pipeline | FlashStereo FP16 |
|---|---:|---:|
| `feat_runner` (TRT) | 32.9 ms | 32.9 ms |
| GWC volume builder | 42.6 ms (CPU+GPU bouncing) | < 1 ms (GPU only) |
| `post_runner` (TRT) | 144.8 ms | 89.6 ms (sees pre-bound device pointers) |
| H2D + D2H + Python overhead | ~ 18 ms | < 1 ms |
| **Total p50** | **238 ms** | **124 ms** |

The 55 ms drop in `post_runner` wall time isn't from a faster engine — the engine binary is byte-identical. It comes from removing the per-call `set_input_shape` + `mem_alloc` + `memcpy_htod` work that the stock `run_trt` helper does on every inference.

## Reproducing

```bash
# 1) Download the public Middlebury 2014 stereo dataset
python scripts/download_calib_data.py --out-dir assets/calib_pairs --n 16

# 2) (Optional) build INT8 engines
python scripts/build_int8.py \
    --onnx /path/to/feature_runner.onnx \
    --engine-out artifacts/feature_runner_int8.engine \
    --calib-dir assets/calib_pairs

python scripts/gen_post_calib_data.py \
    --feat-engine /path/to/feature_runner.engine \
    --post-engine /path/to/post_runner.engine \
    --calib-dir assets/calib_pairs \
    --out-dir artifacts/post_calib

python scripts/build_int8_post.py \
    --onnx /path/to/post_runner.onnx \
    --engine-out artifacts/post_runner_int8.engine \
    --npz-dir artifacts/post_calib

# 3) Run the precision sweep
python scripts/bench.py \
    --feat-fp16 /path/to/feature_runner.engine \
    --feat-int8 artifacts/feature_runner_int8.engine \
    --post-fp16 /path/to/post_runner.engine \
    --post-int8 artifacts/post_runner_int8.engine \
    --left  assets/calib_pairs/Motorcycle_left.png \
    --right assets/calib_pairs/Motorcycle_right.png \
    --sweep
```
