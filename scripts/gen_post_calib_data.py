"""Generate post_runner calibration tensors.

The post_runner stage takes 7 intermediate tensors (4 feat maps from
the left view + 1 from the right + stem_2x + gwc_volume) as input, not
images. To calibrate it for INT8 we need real activations from the
upstream FP16 pipeline.

This script runs the FlashStereoPipeline on each public stereo pair
from the calib directory and dumps the 7 intermediate tensors as a
single .npz file per pair, ready to be consumed by build_int8_post.py.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import pycuda.driver as cuda

# allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flashstereo import FlashStereoPipeline


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feat-engine", required=True,
                    help="FP16 feature_runner.engine (used to produce realistic calib activations)")
    ap.add_argument("--post-engine", required=True,
                    help="FP16 post_runner.engine (only used so the pipeline class can bind)")
    ap.add_argument("--calib-dir", default="./assets/calib_pairs")
    ap.add_argument("--out-dir", required=True, help="where to save the .npz files")
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--w", type=int, default=640)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    left_paths = sorted(glob.glob(os.path.join(args.calib_dir, "*_left.png")))
    if not left_paths:
        raise RuntimeError(f"No *_left.png found in {args.calib_dir}")

    pipe = FlashStereoPipeline(
        feat_engine_path=args.feat_engine,
        post_engine_path=args.post_engine,
        input_h=args.h, input_w=args.w,
    )

    saved = 0
    for i, lp in enumerate(left_paths):
        rp = lp.replace("_left.png", "_right.png")
        if not os.path.exists(rp):
            print(f"  skip {lp}: no matching right")
            continue
        left = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
        if left.shape != (args.h, args.w):
            left = cv2.resize(left, (args.w, args.h))
            right = cv2.resize(right, (args.w, args.h))
        left_rgb = cv2.cvtColor(left, cv2.COLOR_GRAY2RGB)
        right_rgb = cv2.cvtColor(right, cv2.COLOR_GRAY2RGB)

        # Run a forward pass to populate feat + gwc buffers on the GPU
        _ = pipe.infer(left_rgb, right_rgb)

        # D2H every persistent post-input buffer for this pair
        record = {}
        for name in ["features_left_04", "features_left_08", "features_left_16",
                     "features_left_32", "features_right_04", "stem_2x"]:
            if name in pipe._feat_buffers:
                shape = pipe._feat_shapes[name]
                h_mem = cuda.pagelocked_empty(int(np.prod(shape)), np.float32)
                cuda.memcpy_dtoh(h_mem, pipe._feat_buffers[name])
                record[name] = np.asarray(h_mem).reshape(shape).copy()
        h_gwc = cuda.pagelocked_empty(int(np.prod(pipe._gwc_shape)), np.float32)
        cuda.memcpy_dtoh(h_gwc, pipe._d_gwc_volume)
        record["gwc_volume"] = np.asarray(h_gwc).reshape(pipe._gwc_shape).copy()

        scene = os.path.basename(lp).replace("_left.png", "")
        out = os.path.join(args.out_dir, f"calib_{i:03d}_{scene}.npz")
        np.savez(out, **record)
        saved += 1
        if (i + 1) % 4 == 0:
            print(f"  saved {saved} pairs so far")

    print(f"\nDone. {saved} npz files in {args.out_dir}")
    pipe.release()


if __name__ == "__main__":
    main()
