"""Minimal end-to-end demo: load a stereo pair, run inference, save the
disparity map as a colorized PNG.
"""
import argparse
import cv2
import numpy as np

from flashstereo import FlashStereoPipeline


def colorize_disp(disp: np.ndarray, max_disp: float = None) -> np.ndarray:
    if max_disp is None:
        max_disp = float(disp[disp < np.inf].max())
    vis = (disp / max(max_disp, 1e-6) * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-engine", required=True)
    ap.add_argument("--post-engine", required=True)
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--out", default="disp_color.png")
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--w", type=int, default=640)
    args = ap.parse_args()

    pipe = FlashStereoPipeline(
        feat_engine_path=args.feat_engine,
        post_engine_path=args.post_engine,
        input_h=args.h, input_w=args.w,
    )

    left = cv2.imread(args.left, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(args.right, cv2.IMREAD_GRAYSCALE)
    if left.shape != (args.h, args.w):
        left = cv2.resize(left, (args.w, args.h))
        right = cv2.resize(right, (args.w, args.h))
    left_rgb = cv2.cvtColor(left, cv2.COLOR_GRAY2RGB)
    right_rgb = cv2.cvtColor(right, cv2.COLOR_GRAY2RGB)

    disp = pipe.infer(left_rgb, right_rgb)
    print(f"disp: shape={disp.shape}  min={disp.min():.2f}  max={disp.max():.2f}  mean={disp.mean():.2f}")

    vis = colorize_disp(disp)
    cv2.imwrite(args.out, vis)
    print(f"saved colorized disparity to {args.out}")
    pipe.release()


if __name__ == "__main__":
    main()
