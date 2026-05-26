"""FlashStereo — GPU-resident realtime stereo depth on edge GPUs.

A 2.7x speed-up over the stock TensorRT pipeline for FoundationStereo's
two-stage variant, achieved by keeping all intermediate tensors on the
GPU and optionally swapping the post-runner to INT8.

Reference: see README.md.
"""
from .pipeline import FlashStereoPipeline, __version__

__all__ = ["FlashStereoPipeline", "__version__"]
