"""Build an INT8 TensorRT engine for the feature_runner stage.

Uses a Python IInt8EntropyCalibrator2 fed with image pairs from a public
stereo dataset (see download_calib_data.py).

Example
-------
    # 1) download public calibration data
    python scripts/download_calib_data.py --out-dir assets/calib_pairs --n 16

    # 2) build INT8 feat engine
    python scripts/build_int8.py \\
        --onnx /path/to/feature_runner.onnx \\
        --engine-out artifacts/feature_runner_int8.engine \\
        --calib-dir assets/calib_pairs
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401


def load_pair_chw(path_left: str, path_right: str, h: int, w: int):
    """Load a grayscale stereo pair from PNG, resize to h×w, return as
    (1,3,h,w) float32 NCHW with the grayscale channel replicated 3x."""
    l = cv2.imread(path_left, cv2.IMREAD_GRAYSCALE)
    r = cv2.imread(path_right, cv2.IMREAD_GRAYSCALE)
    if l is None or r is None:
        raise FileNotFoundError(f"missing {path_left} or {path_right}")
    if l.shape != (h, w):
        l = cv2.resize(l, (w, h))
        r = cv2.resize(r, (w, h))
    l_rgb = cv2.cvtColor(l, cv2.COLOR_GRAY2RGB)
    r_rgb = cv2.cvtColor(r, cv2.COLOR_GRAY2RGB)
    l_chw = np.ascontiguousarray(l_rgb.transpose(2, 0, 1)[None], dtype=np.float32)
    r_chw = np.ascontiguousarray(r_rgb.transpose(2, 0, 1)[None], dtype=np.float32)
    return l_chw, r_chw


class StereoCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, pairs, cache_file, h, w, in_left="left", in_right="right"):
        super().__init__()
        self.pairs = pairs
        self.cache_file = cache_file
        self.cur = 0
        self.in_left = in_left
        self.in_right = in_right
        nbytes = 1 * 3 * h * w * 4
        self.d_left = cuda.mem_alloc(nbytes)
        self.d_right = cuda.mem_alloc(nbytes)
        print(f"[calibrator] {len(pairs)} pairs at {h}x{w} (inputs: {in_left}, {in_right})")

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.cur >= len(self.pairs):
            return None
        l, r = self.pairs[self.cur]
        self.cur += 1
        cuda.memcpy_htod(self.d_left, np.ascontiguousarray(l, dtype=np.float32))
        cuda.memcpy_htod(self.d_right, np.ascontiguousarray(r, dtype=np.float32))
        out = []
        for n in names:
            if n == self.in_left:
                out.append(int(self.d_left))
            elif n == self.in_right:
                out.append(int(self.d_right))
            else:
                print(f"[calibrator] WARN: unknown input '{n}'")
                return None
        if self.cur % 4 == 0:
            print(f"[calibrator] fed {self.cur}/{len(self.pairs)}")
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


def build(onnx_path, engine_out, h, w, calibrator, workspace_gb=4, verbose=False):
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
    for i in range(network.num_inputs):
        t = network.get_input(i)
        print(f"  IN  {t.name}  shape={t.shape}  dtype={t.dtype}")
    for i in range(network.num_outputs):
        t = network.get_output(i)
        print(f"  OUT {t.name}  shape={t.shape}  dtype={t.dtype}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.int8_calibrator = calibrator

    profile = builder.create_optimization_profile()
    has_dynamic = False
    fixed_shape = (1, 3, h, w)
    for i in range(network.num_inputs):
        t = network.get_input(i)
        if any(d == -1 for d in t.shape):
            has_dynamic = True
            profile.set_shape(t.name, fixed_shape, fixed_shape, fixed_shape)
    if has_dynamic:
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
    ap.add_argument("--onnx", required=True,
                    help="path to feature_runner.onnx")
    ap.add_argument("--engine-out", required=True,
                    help="output path for the INT8 engine")
    ap.add_argument("--calib-dir", default="./assets/calib_pairs",
                    help="directory with stereo pairs (from download_calib_data.py)")
    ap.add_argument("--cache", default=None,
                    help="path to calibration cache (defaults to <engine-out>.calib)")
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--in-left", default="left")
    ap.add_argument("--in-right", default="right")
    ap.add_argument("--workspace-gb", type=int, default=4)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cache_file = args.cache or args.engine_out + ".calib"

    # Load all stereo pairs from calib_dir (pattern: <scene>_left.png + <scene>_right.png)
    left_paths = sorted(glob.glob(os.path.join(args.calib_dir, "*_left.png")))
    if not left_paths:
        raise RuntimeError(f"No *_left.png found in {args.calib_dir}; run download_calib_data.py first")
    pairs = []
    for lp in left_paths:
        rp = lp.replace("_left.png", "_right.png")
        if not os.path.exists(rp):
            print(f"  skip {lp}: missing matching {rp}")
            continue
        l, r = load_pair_chw(lp, rp, args.h, args.w)
        pairs.append((l, r))
    print(f"Loaded {len(pairs)} calibration pairs from {args.calib_dir}")

    calibrator = StereoCalibrator(pairs, cache_file, args.h, args.w,
                                   in_left=args.in_left, in_right=args.in_right)
    build(args.onnx, args.engine_out, args.h, args.w, calibrator,
          workspace_gb=args.workspace_gb, verbose=args.verbose)


if __name__ == "__main__":
    main()
