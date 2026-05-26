"""Build an INT8 TensorRT engine for the post_runner stage.

The post_runner takes 7 intermediate tensors as input (not images), so
calibration requires real activations from the upstream FP16 pipeline.
Generate those first with gen_post_calib_data.py, then run this script.

Example
-------
    # 1) generate calibration tensors (~5 min on Orin)
    python scripts/gen_post_calib_data.py \\
        --feat-engine path/to/feature_runner.engine \\
        --post-engine path/to/post_runner.engine \\
        --calib-dir assets/calib_pairs \\
        --out-dir artifacts/post_calib

    # 2) build INT8 post engine (~45 min on Orin)
    python scripts/build_int8_post.py \\
        --onnx path/to/post_runner.onnx \\
        --engine-out artifacts/post_runner_int8.engine \\
        --npz-dir artifacts/post_calib
"""
import argparse
import glob
import os
import time

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401


# Default post_runner I/O shapes at 480x640. Must match what feat_engine produces.
POST_INPUTS = [
    ("features_left_04",  (1, 224, 120, 160)),
    ("features_left_08",  (1, 192,  60,  80)),
    ("features_left_16",  (1, 320,  30,  40)),
    ("features_left_32",  (1, 304,  15,  20)),
    ("features_right_04", (1, 224, 120, 160)),
    ("stem_2x",           (1,  16, 240, 320)),
    ("gwc_volume",        (1,   8,  48, 120, 160)),
]


class PostCascadeCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, npz_paths, cache_file):
        super().__init__()
        self.paths = npz_paths
        self.cache_file = cache_file
        self.cur = 0
        self.dmem = {name: cuda.mem_alloc(int(np.prod(shape)) * 4)
                     for name, shape in POST_INPUTS}
        print(f"[calibrator] {len(npz_paths)} pairs, {len(POST_INPUTS)} input tensors")

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.cur >= len(self.paths):
            return None
        data = np.load(self.paths[self.cur])
        self.cur += 1
        for name, _shape in POST_INPUTS:
            arr = np.ascontiguousarray(data[name], dtype=np.float32)
            cuda.memcpy_htod(self.dmem[name], arr)
        out = []
        for n in names:
            if n in self.dmem:
                out.append(int(self.dmem[n]))
            else:
                print(f"[calibrator] WARN: unknown input '{n}'")
                return None
        if self.cur % 4 == 0:
            print(f"[calibrator] fed {self.cur}/{len(self.paths)}")
        return out

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"[calibrator] reading cache {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        print(f"[calibrator] writing cache {self.cache_file} ({len(cache)} bytes)")
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def build(onnx_path, engine_out, npz_paths, cache_file, workspace_gb=4, verbose=False):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    trt.init_libnvinfer_plugins(None, "")
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"[parser err {i}] {parser.get_error(i)}")
            raise RuntimeError("ONNX parse failed")
    print(f"[build] network: {network.num_layers} layers, {network.num_inputs} inputs, {network.num_outputs} outputs")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.int8_calibrator = PostCascadeCalibrator(npz_paths, cache_file)

    profile = builder.create_optimization_profile()
    has_dyn = False
    shape_map = dict(POST_INPUTS)
    for i in range(network.num_inputs):
        t = network.get_input(i)
        if any(d == -1 for d in t.shape):
            has_dyn = True
            shape = shape_map.get(t.name)
            if shape is None:
                raise RuntimeError(f"No default shape for dynamic input {t.name}")
            profile.set_shape(t.name, shape, shape, shape)
    if has_dyn:
        config.add_optimization_profile(profile)
        config.set_calibration_profile(profile)

    print(f"[build] INT8 + FP16-fallback, workspace={workspace_gb} GB")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None")
    try:
        size_mb = serialized.nbytes / 1024 / 1024
    except AttributeError:
        size_mb = len(serialized) / 1024 / 1024
    print(f"[build] done in {time.time()-t0:.1f}s, engine size = {size_mb:.1f} MB")
    os.makedirs(os.path.dirname(engine_out) or ".", exist_ok=True)
    with open(engine_out, "wb") as f:
        f.write(bytes(serialized))
    print(f"[build] saved to {engine_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--engine-out", required=True)
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--workspace-gb", type=int, default=4)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cache_file = args.cache or args.engine_out + ".calib"
    paths = sorted(glob.glob(os.path.join(args.npz_dir, "calib_*.npz")))
    if not paths:
        raise RuntimeError(f"No calib_*.npz found in {args.npz_dir}")
    print(f"[main] {len(paths)} calibration npz files")
    build(args.onnx, args.engine_out, paths, cache_file,
          workspace_gb=args.workspace_gb, verbose=args.verbose)


if __name__ == "__main__":
    main()
