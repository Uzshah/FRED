"""
FRED - Fast Refinement for Equirectangular Depth
=================================================

A Python package for high-resolution depth estimation on 360° panoramic images
using patch-based PromptDA inference with optimized GPU batching.
"""

__version__ = "0.1.0"

from .pipeline import run_pipeline
from .io import (
    load_equirect_rgb_as_tensor,
    load_prompt_depth_as_tensor,
    save_depth_u16_qmm_png,
    save_depth_colormap,
)
from .geometry import build_icosahedron, subdivide_geodesic, tri_face_centers
from .fusion import fuse_depth_patches

__all__ = [
    "run_pipeline",
    "load_equirect_rgb_as_tensor",
    "load_prompt_depth_as_tensor",
    "save_depth_u16_qmm_png",
    "save_depth_colormap",
    "build_icosahedron",
    "subdivide_geodesic",
    "tri_face_centers",
    "fuse_depth_patches",
]
