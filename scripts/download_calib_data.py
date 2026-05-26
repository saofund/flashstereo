"""Download a public stereo dataset for INT8 calibration.

Default: Middlebury 2014 Stereo dataset (high-quality lab scenes, no auth
required, ~23 stereo pairs in the "perfect" subset). Each pair is
resized to the target engine resolution (default 480x640) and saved as
PNG for use by gen_post_calib_data.py and build_int8.py.

Reference: D. Scharstein et al., "High-Resolution Stereo Datasets with
Subpixel-Accurate Ground Truth", GCPR 2014.
https://vision.middlebury.edu/stereo/data/scenes2014/
"""
import argparse
import io
import os
import urllib.request
import zipfile

import cv2
import numpy as np


# Middlebury 2014 "perfect" subset — high-quality calibration scenes
MIDDLEBURY_2014_SCENES = [
    "Adirondack", "Backpack", "Bicycle1", "Cable", "Classroom1", "Couch",
    "Flowers", "Jadeplant", "Mask", "Motorcycle", "Piano", "Pipes",
    "Playroom", "Playtable", "Recycle", "Shelves", "Shopvac", "Sticks",
    "Storage", "Sword1", "Sword2", "Umbrella", "Vintage",
]


def download_scene(scene: str, out_dir: str, h: int, w: int) -> bool:
    url = f"https://vision.middlebury.edu/stereo/data/scenes2014/zip/{scene}-perfect.zip"
    target_left = os.path.join(out_dir, f"{scene}_left.png")
    target_right = os.path.join(out_dir, f"{scene}_right.png")
    if os.path.exists(target_left) and os.path.exists(target_right):
        return True
    try:
        print(f"  fetching {scene}-perfect.zip ...", end="", flush=True)
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            members = zf.namelist()
            im0 = next((m for m in members if m.endswith("/im0.png")), None)
            im1 = next((m for m in members if m.endswith("/im1.png")), None)
            if im0 is None or im1 is None:
                print(" missing im0/im1.png")
                return False
            im0_bytes = zf.read(im0)
            im1_bytes = zf.read(im1)
        left = cv2.imdecode(np.frombuffer(im0_bytes, np.uint8), cv2.IMREAD_COLOR)
        right = cv2.imdecode(np.frombuffer(im1_bytes, np.uint8), cv2.IMREAD_COLOR)
        if left is None or right is None:
            print(" decode failed")
            return False
        # Convert to grayscale-style 3-channel (matches typical stereo IR input
        # to FoundationStereo). The model accepts color or replicated grayscale.
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        left_gray = cv2.resize(left_gray, (w, h), interpolation=cv2.INTER_AREA)
        right_gray = cv2.resize(right_gray, (w, h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(target_left, left_gray)
        cv2.imwrite(target_right, right_gray)
        print(f" -> {target_left} + {target_right}")
        return True
    except Exception as e:
        print(f" FAILED: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="./assets/calib_pairs",
                    help="where to save the downloaded stereo pairs")
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--n", type=int, default=16,
                    help="how many scenes to download (max 23)")
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="explicit scene names (override --n)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    scenes = args.scenes if args.scenes else MIDDLEBURY_2014_SCENES[: args.n]
    print(f"Downloading {len(scenes)} Middlebury 2014 stereo pairs -> {args.out_dir}")
    print(f"Target resolution: {args.h}x{args.w}, single-channel saved as grayscale PNG.")
    ok = 0
    for s in scenes:
        if download_scene(s, args.out_dir, args.h, args.w):
            ok += 1
    print(f"\nDone: {ok}/{len(scenes)} scenes downloaded.")


if __name__ == "__main__":
    main()
