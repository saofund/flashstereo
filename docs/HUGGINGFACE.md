# Hosting pre-built engines on HuggingFace

This repo intentionally keeps the source tree small (~150 KB) and
free of binary artifacts. Pre-built TensorRT INT8 engines, calibration
caches, and the Middlebury calibration pairs are released separately
on HuggingFace so users can grab them with one `huggingface-cli download`.

## Suggested repo layout on HF

```
Dexmal/flashstereo-int8-orin/      # (or your namespace)
├── README.md
├── engines/
│   ├── feature_runner_int8.engine        # 19 MB, Middlebury-calibrated
│   └── post_runner_int8.engine           # 14 MB, built on Orin (~45 min)
├── calib_cache/
│   ├── feature_runner_int8.engine.calib  # 50 KB
│   └── post_runner_int8.engine.calib     # 118 KB
└── calib_pairs/                          # optional: the 16 PNG pairs
    ├── Adirondack_left.png  Adirondack_right.png
    ├── Backpack_left.png    Backpack_right.png
    └── ... (14 more)
```

## Upload workflow

```bash
# install hf cli once
pip install huggingface_hub

# log in
huggingface-cli login

# create the repo (one-shot)
huggingface-cli repo create flashstereo-int8-orin --type model

# clone locally and copy the artifacts in
git lfs install
git clone https://huggingface.co/saofund/flashstereo-int8-orin
cd flashstereo-int8-orin
mkdir -p engines calib_cache
cp /path/to/feature_runner_int8.engine        engines/
cp /path/to/feature_runner_int8.engine.calib  calib_cache/
# (optional) cp /path/to/post_runner_int8.engine       engines/
# (optional) cp /path/to/post_runner_int8.engine.calib calib_cache/
git lfs track "*.engine" "*.calib"
git add .
git commit -m "FlashStereo INT8 engines built on Middlebury 2014"
git push
```

## Download (for end users)

```bash
pip install huggingface_hub

# fetch only what you need
huggingface-cli download saofund/flashstereo-int8-orin \
    engines/feature_runner_int8.engine \
    --local-dir artifacts/

# or grab the whole repo
huggingface-cli download saofund/flashstereo-int8-orin \
    --local-dir hf_weights/
```

Then point the FlashStereo pipeline at the downloaded engine:

```python
from flashstereo import FlashStereoPipeline

pipe = FlashStereoPipeline(
    feat_engine_path="hf_weights/engines/feature_runner_int8.engine",
    post_engine_path="path/to/your/post_runner.engine",   # FP16 or INT8
    input_h=480, input_w=640,
)
```

## Reminders

- **TensorRT engines are NOT portable** across GPU arch, driver, or
  TensorRT version. The pre-built engines on HF target Jetson AGX Orin
  (SM87) / TensorRT 10.3 / CUDA 12.6. For other targets, use the build
  scripts in this repo to regenerate.
- **Don't upload calibration npz files** to HF (~1.2 GB and easily
  regenerable from the published `feature_runner.engine` + Middlebury
  pairs via `scripts/gen_post_calib_data.py`).
- **License**: keep Apache 2.0 on the HF repo to match this codebase.
