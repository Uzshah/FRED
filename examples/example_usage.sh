#!/bin/bash
# Example usage of FRED for depth estimation on 360° panoramas

# Example 1: Basic usage with default parameters
echo "Example 1: Basic usage"
fred \
  --image my_panorama.jpg \
  --prompt-depth lowres_depth.png \
  --prompt-units qmm \
  --out highres_depth.png

# Example 2: High quality with visualization and point cloud
echo "Example 2: High quality with full outputs"
fred \
  --image my_panorama.jpg \
  --prompt-depth lowres_depth.png \
  --prompt-units qmm \
  --out highres_depth.png \
  --depth-vis depth_visualization.png \
  --depth-vis-width 4096 \
  --depth-vis-height 2048 \
  --ply pointcloud.ply \
  --ply-binary \
  --ply-stride 2 \
  --subdiv 2 \
  --patch 1008 \
  --batch-size 8 \
  --lat-weight \
  --profile

# Example 3: Fast processing with lower quality
echo "Example 3: Fast processing"
fred \
  --image my_panorama.jpg \
  --prompt-depth lowres_depth.png \
  --prompt-units qmm \
  --out highres_depth.png \
  --subdiv 0 \
  --patch 700 \
  --batch-size 16 \
  --profile

# Example 4: Custom fusion parameters
echo "Example 4: Custom fusion settings"
fred \
  --image my_panorama.jpg \
  --prompt-depth lowres_depth.png \
  --prompt-units qmm \
  --out highres_depth.png \
  --fusion-mode logdepth \
  --window-radius 0.5 \
  --consistency-tau-mm 500.0 \
  --lat-weight \
  --batch-size 4

# Example 5: Save intermediate outputs for iterative refinement
echo "Example 5: Iterative refinement mode"
fred \
  --image my_panorama.jpg \
  --prompt-depth lowres_depth.png \
  --prompt-units qmm \
  --out highres_depth.png \
  --save-weights \
  --save-consistency \
  --profile

# Example 6: Process with millimeter depth units
echo "Example 6: Different depth units"
fred \
  --image my_panorama.jpg \
  --prompt-depth depth_in_mm.png \
  --prompt-units mm \
  --out highres_depth.png \
  --batch-size 4

echo "All examples completed!"
