<div align="center">

# ⚡ FlashStereo

**Realtime stereo depth on Jetson Orin — 2.7× faster than the stock TensorRT pipeline, with bit-identical FP16 output.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![TensorRT 10](https://img.shields.io/badge/TensorRT-10.3+-76B900.svg)](https://developer.nvidia.com/tensorrt)
[![Jetson Orin](https://img.shields.io/badge/Jetson-AGX_Orin-76B900.svg)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)

</div>

---

FlashStereo is a drop-in replacement for the two-stage TensorRT inference pipeline of [FoundationStereo](https://github.com/NVlabs/FoundationStereo). It keeps every intermediate tensor on the GPU between stages, lifts the per-group L2 norm and group-wise correlation (GWC) volume into hand-written CUDA kernels, and ships a calibration toolchain so you can swap each stage to **INT8** at <0.2% disparity error.

The headline result on Jetson AGX Orin (TensorRT 10.3, 480×640 input):

| Configuration | Latency (p50) | Throughput | vs. stock |
|---|---:|---:|---:|
| **Stock pipeline** (FP16, Python GWC, D2H/H2D bouncing) | 238 ms | 4.20 Hz | 1.00× |
| **FlashStereo FP16** (GPU-resident, **bit-identical** ✓) | **124 ms** | **8.07 Hz** | **1.92×** |
| **FlashStereo + INT8 post** | 90 ms | 11.17 Hz | 2.66× |
| **FlashStereo + INT8 feat + INT8 post** | **88 ms** | **11.37 Hz** | **2.71×** |

INT8 disparity quality vs. the FP16 reference: cosine **0.99999**, relative L1 **0.19%** — visually indistinguishable, single-pixel-scale worst case.

---

## ✨ Why it's fast

The stock FoundationStereo two-stage pipeline pays a ~40 ms tax per frame on a Python-mediated GWC volume builder that does the following dance:

```
  feat_runner (TRT) → device buffers
       ↓
       D2H to numpy            (≈ 16 MiB × 2)
  CPU np.linalg.norm           (35 M FP32 reductions on the CPU)
       H2D back to GPU
  PyCUDA gwc kernel
       D2H to numpy            (≈ 28 MiB)
       ↓
  post_runner (TRT)
       H2D every input         (≈ 70 MiB total)
       ↓
       D2H disparity
```

That's roughly **100 MiB of needless host↔device traffic and a CPU-bound reduction every frame**. FlashStereo rewrites the data flow as GPU-resident:

```
  feat_runner (TRT)  ── writes outputs to persistent device buffers
  GPU norm kernel    ── reads / writes on device
  GPU GWC kernel     ── reads / writes on device
  post_runner (TRT)  ── reads feat outputs + GWC volume directly via the
                        same device pointers (zero-copy chaining)
       ↓
       D2H disparity   ← the only mandatory copy per inference
```

Same engines. Same kernels. Same numerics. Just **one host↔device round-trip per frame instead of half a dozen**. The FP16 path is bit-identical to the reference (verified by `tests/` style stage-by-stage diff: L1 = 0, cosine = 1.000000000 across feat outputs, GWC volume, and final disparity).

INT8 wins are stacked on top: post_runner INT8 alone shaves another 34 ms (the RAFT iterative refinement is dominated by quantizable convolutions); feat_runner INT8 is a smaller ~1 ms gain since it's a much smaller engine.

## 📦 Install

Requires:

| | |
|---|---|
| GPU | Jetson AGX Orin (SM87) or any CUDA device with TensorRT 10.x |
| CUDA | 12.4+ |
| TensorRT | 10.3+ |
| Python | 3.10 / 3.11 / 3.12 |
| Other | `numpy`, `opencv-python`, `pycuda`, `tensorrt`, `torch` (only for ONNX export, optional at runtime) |

```bash
git clone https://github.com/saofund/flashstereo
cd flashstereo
pip install -e .
```

On Jetson, install PyCUDA + TensorRT Python bindings from the JetPack apt repo (the `nvidia-tensorrt` and `python3-libnvinfer*` packages); the rest is plain `pip install`.

## 🏃 Quickstart

You need two FoundationStereo TensorRT engines: the feature_runner and the post_runner. If you don't have them yet, see [the original FoundationStereo deployment guide](https://github.com/NVlabs/FoundationStereo) for the ONNX export steps, or grab a pre-built pair from your own checkpoint hub.

```python
import cv2
from flashstereo import FlashStereoPipeline

pipe = FlashStereoPipeline(
    feat_engine_path="path/to/feature_runner.engine",
    post_engine_path="path/to/post_runner.engine",
    input_h=480, input_w=640,
)

left  = cv2.imread("left.png",  cv2.IMREAD_GRAYSCALE)
right = cv2.imread("right.png", cv2.IMREAD_GRAYSCALE)
left_rgb  = cv2.cvtColor(left,  cv2.COLOR_GRAY2RGB)
right_rgb = cv2.cvtColor(right, cv2.COLOR_GRAY2RGB)

disp = pipe.infer(left_rgb, right_rgb)   # (H, W) float32 disparity in pixels
```

Or as a one-shot demo:

```bash
python examples/single_pair_demo.py \
    --feat-engine path/to/feature_runner.engine \
    --post-engine path/to/post_runner.engine \
    --left  assets/example_left.png \
    --right assets/example_right.png \
    --out   disp_color.png
```

## 📦 Pre-built INT8 weights

We host a Jetson AGX Orin / TensorRT 10.3 build of the INT8
`feature_runner` engine on HuggingFace, calibrated on 16 public
[Middlebury 2014 Stereo](https://vision.middlebury.edu/stereo/data/scenes2014/) pairs:

> 🤗 **`https://huggingface.co/saofund/flashstereo-int8-orin`** *(replace with actual URL once published)*

```bash
pip install huggingface_hub
huggingface-cli download saofund/flashstereo-int8-orin \
    engines/feature_runner_int8.engine \
    --local-dir artifacts/
```

⚠️ **TensorRT engines are not portable** across GPU arch / TRT version /
CUDA version. If your target differs from Orin SM87 + TRT 10.3, rebuild
locally using the recipe below.

The matching `post_runner_int8.engine` is **not** pre-published — it
takes ~45 min of GPU time to build but is trivial to regenerate from
the same Middlebury pairs (see step 3-4 below). See
[`docs/HUGGINGFACE.md`](docs/HUGGINGFACE.md) for the full release /
upload guide.

## 🔧 Rebuilding INT8 engines from scratch

INT8 calibration is reproducible from any public stereo dataset. The reference recipe uses the [Middlebury 2014 Stereo](https://vision.middlebury.edu/stereo/data/scenes2014/) dataset — 23 high-quality lab scenes, free, no auth required.

```bash
# 1) Pull 16 stereo pairs from Middlebury 2014 and resize them to 480x640
python scripts/download_calib_data.py \
    --out-dir assets/calib_pairs --n 16

# 2) Build INT8 feature_runner engine (~5 min on Orin)
python scripts/build_int8.py \
    --onnx path/to/feature_runner.onnx \
    --engine-out artifacts/feature_runner_int8.engine \
    --calib-dir assets/calib_pairs

# 3) Generate post_runner calibration tensors via the FP16 feat pipeline
python scripts/gen_post_calib_data.py \
    --feat-engine path/to/feature_runner.engine \
    --post-engine path/to/post_runner.engine \
    --calib-dir assets/calib_pairs \
    --out-dir artifacts/post_calib

# 4) Build INT8 post_runner engine (~45 min on Orin)
python scripts/build_int8_post.py \
    --onnx path/to/post_runner.onnx \
    --engine-out artifacts/post_runner_int8.engine \
    --npz-dir artifacts/post_calib
```

## 📊 Bench it yourself

Single config:

```bash
python scripts/bench.py \
    --feat  artifacts/feature_runner_int8.engine \
    --post  artifacts/post_runner_int8.engine \
    --left  assets/calib_pairs/Motorcycle_left.png \
    --right assets/calib_pairs/Motorcycle_right.png \
    --reps  20
```

Full precision sweep (4 combinations of FP16/INT8 feat × FP16/INT8 post) with disparity diff against the FP16 baseline:

```bash
python scripts/bench.py \
    --feat-fp16 path/to/feature_runner.engine \
    --feat-int8 artifacts/feature_runner_int8.engine \
    --post-fp16 path/to/post_runner.engine \
    --post-int8 artifacts/post_runner_int8.engine \
    --left  assets/calib_pairs/Motorcycle_left.png \
    --right assets/calib_pairs/Motorcycle_right.png \
    --sweep
```

## 🧠 What it does NOT change

- **No retraining.** The model weights are identical. FlashStereo only rewires the data flow between TensorRT engines.
- **No new ONNX.** It uses the standard FoundationStereo two-stage ONNX exports as-is.
- **No accuracy tradeoff in the FP16 path.** Bit-identical disparity output, verified end-to-end.

## 🗺️ When *not* to use FlashStereo

This optimization targets the **two-stage variant** of FoundationStereo (separate feature_runner + post_runner + Python-mediated GWC volume). If your deployment uses the single-engine "full" variant, this codebase will not help you — profiling shows the single-engine variant spends 99.96% of its time inside the TensorRT engine itself, with no Python overhead left to remove.

## 🙏 Credits & inspiration

FlashStereo stands on the shoulders of two pieces of work:

- **[FoundationStereo](https://github.com/NVlabs/FoundationStereo)** (NVIDIA, 2025) — the stereo depth model and the original two-stage TensorRT deployment recipe.
- **[FlashRT](https://github.com/gugudeshubao/FlashRT)** (`feat/orin-pipelined-streaming` branch) — a CUDA-native realtime inference engine for VLA models. FlashStereo borrows its core design principle: *small-batch latency-sensitive inference is best served by hand-written, GPU-resident pipelines, not by general-purpose tactic-search compilers.* The specific kernels are different (FlashRT optimizes transformer attention; FlashStereo optimizes a stereo cost volume) but the philosophy is shared.

If you use FlashStereo in academic work, please cite the original FoundationStereo paper:

```bibtex
@inproceedings{wen2025foundationstereo,
  title     = {FoundationStereo: Zero-Shot Stereo Matching},
  author    = {Wen, Bowen and Trepte, Matthew and Aribido, Joseph and Kautz, Jan and Gallo, Orazio and Birchfield, Stan},
  booktitle = {CVPR},
  year      = {2025},
}
```

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE). The bundled GWC and norm CUDA kernels are released under the same license. Model weights and ONNX files are governed by the FoundationStereo license; this repo does not redistribute them.

---

<div align="center">
<sub>Benchmarks measured on Jetson AGX Orin 64 GB / JetPack 6.0 / TensorRT 10.3 / 480×640 input / 8-rep p50.</sub>
</div>
