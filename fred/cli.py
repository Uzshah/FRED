"""
Command-line interface for FRED.
"""

import argparse
from .pipeline import run_pipeline


def parse_args():
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        description="FRED: Fast Refinement for Equirectangular Depth - "
        "Batched PromptDA inference with spherical tiling."
    )
    ap.add_argument("--image", required=True, help="Path to input equirectangular RGB image")
    ap.add_argument("--prompt-depth", required=True, help="Path to low-res prompt depth map")
    ap.add_argument(
        "--prompt-units",
        default="qmm",
        choices=["m", "mm", "qmm"],
        help="Prompt depth units: 'm'=meters, 'mm'=millimeters, 'qmm'=quarter-millimeters",
    )
    ap.add_argument("--out", required=True, help="Output path for 16-bit depth PNG (qmm)")
    ap.add_argument("--ply", default="", help="Optional PLY point cloud output path")
    ap.add_argument("--ply-binary", action="store_true", help="Write binary little-endian PLY")
    ap.add_argument(
        "--ply-stride", type=int, default=1, help="Subsample stride for PLY export (>=1)"
    )
    ap.add_argument("--depth-vis", default="", help="Optional plasma colormap visualization path")
    ap.add_argument("--depth-vis-width", type=int, default=4096, help="Visualization width")
    ap.add_argument("--depth-vis-height", type=int, default=2048, help="Visualization height")
    ap.add_argument(
        "--subdiv", type=int, default=1, help="Icosahedron subdivision level (0-3 typical)"
    )
    ap.add_argument(
        "--patch", type=int, default=1008, help="Patch resolution (must be multiple of 14)"
    )
    ap.add_argument(
        "--margin-deg", type=float, default=6.0, help="Angular margin in degrees for patch overlap"
    )
    ap.add_argument("--device", default="cuda:0", help="PyTorch device (cuda:0, cpu, etc.)")
    ap.add_argument(
        "--model",
        default="depth-anything/prompt-depth-anything-vitl",
        help="HuggingFace model name",
    )
    ap.add_argument(
        "--consistency-tau-mm",
        type=float,
        default=300.0,
        help="Decay constant for consistency weighting (mm)",
    )
    ap.add_argument(
        "--window-radius",
        type=float,
        default=0.35,
        help="Tukey window taper parameter [0, 1]",
    )
    ap.add_argument(
        "--fusion-mode",
        default="invdepth",
        choices=["depth", "invdepth", "logdepth"],
        help="Depth fusion mode",
    )
    ap.add_argument("--lat-weight", action="store_true", help="Apply latitude-based weighting")
    ap.add_argument("--no-wrap-u", action="store_true", help="Disable horizontal wrapping")

    # Batching & profiling
    ap.add_argument(
        "--batch-size", type=int, default=4, help="Number of patches to process together on GPU"
    )
    ap.add_argument(
        "--no-batch", action="store_true", help="Disable batching (same as --batch-size 1)"
    )
    ap.add_argument(
        "--profile", action="store_true", help="Print detailed timing breakdown & throughput"
    )

    # Iterative refinement outputs
    ap.add_argument(
        "--save-weights",
        action="store_true",
        help="Save fusion weight map as {out}_weight.png (for iterative refinement)",
    )
    ap.add_argument(
        "--save-consistency",
        action="store_true",
        help="Save consistency map as {out}_consistency.png (for debugging)",
    )

    return ap.parse_args()


def main():
    """Main entry point for FRED CLI."""
    args = parse_args()
    run_pipeline(
        args.image,
        args.prompt_depth,
        args.prompt_units,
        args.out,
        args.ply,
        args.ply_binary,
        args.ply_stride,
        args.depth_vis,
        args.depth_vis_width,
        args.depth_vis_height,
        args.subdiv,
        args.patch,
        args.margin_deg,
        args.device,
        args.model,
        args.consistency_tau_mm,
        args.window_radius,
        fusion_mode=args.fusion_mode,
        lat_weight=args.lat_weight,
        wrap_u=(not args.no_wrap_u),
        batch_size=max(1, args.batch_size),
        no_batch=args.no_batch,
        profile=args.profile,
        save_weights=args.save_weights,
        save_consistency=args.save_consistency,
    )


if __name__ == "__main__":
    main()
