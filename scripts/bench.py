"""Benchmark FlashStereoPipeline under any precision combination.

Measures p50 / mean / min / max latency over `--reps` iterations, plus
the disparity diff (cosine, L1, L_inf) against the FP16 baseline so you
can spot precision regressions when sweeping INT8.

Example
-------
    # Single config
    python scripts/bench.py \\
        --feat path/to/feature_runner.engine \\
        --post path/to/post_runner.engine \\
        --left assets/calib_pairs/Motorcycle_left.png \\
        --right assets/calib_pairs/Motorcycle_right.png

    # Precision sweep (4 combos)
    python scripts/bench.py \\
        --feat-fp16 ... --feat-int8 ... \\
        --post-fp16 ... --post-int8 ... \\
        --sweep
"""
import argparse
import os
import sys
import time
import gc

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flashstereo import FlashStereoPipeline


def run_one(feat_path, post_path, left_rgb, right_rgb, reps):
    pipe = FlashStereoPipeline(feat_engine_path=feat_path,
                                post_engine_path=post_path,
                                input_h=left_rgb.shape[0],
                                input_w=left_rgb.shape[1])
    for _ in range(3):
        _ = pipe.infer(left_rgb, right_rgb)
    e2e = []
    disp_out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        d = pipe.infer(left_rgb, right_rgb)
        e2e.append((time.perf_counter() - t0) * 1000)
        disp_out = d
    pipe.release()
    del pipe
    gc.collect()
    return np.array(e2e), disp_out.copy()


def cmp_disp(label, base, other):
    a, b = base.astype(np.float64), other.astype(np.float64)
    diff = a - b
    l1 = float(np.abs(diff).mean())
    l_inf = float(np.abs(diff).max())
    rel = l1 / (np.abs(a).mean() + 1e-12)
    cos = float(np.dot(a.flatten(), b.flatten()) /
                (np.linalg.norm(a.flatten()) * np.linalg.norm(b.flatten()) + 1e-12))
    print(f"    vs baseline -> cos={cos:.6f}  L1={l1:.4f}  L_inf={l_inf:.4f}  relL1={rel:.4%}")


def report(label, e2e, disp, base_disp=None, base_p50=None):
    p50 = float(np.median(e2e))
    print(f"\n  [{label}] p50={p50:.2f} ms ({1000/p50:.2f} Hz)  "
          f"mean={e2e.mean():.2f}  min={e2e.min():.2f}  max={e2e.max():.2f}")
    print(f"      disp: mean(|d|)={np.abs(disp).mean():.4f}  "
          f"max={disp.max():.4f}  min={disp.min():.4f}")
    if base_disp is not None:
        cmp_disp(label, base_disp, disp)
    if base_p50 is not None:
        s = base_p50 / p50
        print(f"      speedup vs baseline: {s:.3f}x  (saved {base_p50 - p50:+.2f} ms)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default=None, help="feat engine for single-config bench")
    ap.add_argument("--post", default=None, help="post engine for single-config bench")
    ap.add_argument("--feat-fp16", default=None)
    ap.add_argument("--feat-int8", default=None)
    ap.add_argument("--post-fp16", default=None)
    ap.add_argument("--post-int8", default=None)
    ap.add_argument("--sweep", action="store_true",
                    help="run all 4 combinations of (FP16, INT8) feat x post")
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--w", type=int, default=640)
    args = ap.parse_args()

    left = cv2.imread(args.left, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(args.right, cv2.IMREAD_GRAYSCALE)
    if left.shape != (args.h, args.w):
        left = cv2.resize(left, (args.w, args.h))
        right = cv2.resize(right, (args.w, args.h))
    left_rgb = cv2.cvtColor(left, cv2.COLOR_GRAY2RGB)
    right_rgb = cv2.cvtColor(right, cv2.COLOR_GRAY2RGB)

    if args.sweep:
        if not all([args.feat_fp16, args.feat_int8, args.post_fp16, args.post_int8]):
            raise SystemExit("--sweep requires --feat-fp16/--feat-int8/--post-fp16/--post-int8")
        configs = [
            ("FP16 feat + FP16 post (baseline)", args.feat_fp16, args.post_fp16),
            ("INT8 feat + FP16 post",            args.feat_int8, args.post_fp16),
            ("FP16 feat + INT8 post",            args.feat_fp16, args.post_int8),
            ("INT8 feat + INT8 post (all INT8)", args.feat_int8, args.post_int8),
        ]
        print("=" * 70)
        print("FlashStereo precision sweep")
        print("=" * 70)
        base_disp, base_p50 = None, None
        for i, (label, fp, pp) in enumerate(configs):
            e2e, disp = run_one(fp, pp, left_rgb, right_rgb, args.reps)
            if i == 0:
                base_disp = disp
                base_p50 = float(np.median(e2e))
                report(label, e2e, disp)
            else:
                report(label, e2e, disp, base_disp=base_disp, base_p50=base_p50)
        print("\n" + "=" * 70)
    else:
        if not (args.feat and args.post):
            raise SystemExit("provide --feat and --post (or use --sweep)")
        e2e, disp = run_one(args.feat, args.post, left_rgb, right_rgb, args.reps)
        report("single", e2e, disp)


if __name__ == "__main__":
    main()
