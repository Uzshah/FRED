<<<<<<< HEAD
# FRED - Fast Refinement for Full Resolution Equirectangular Depth Estimation

**FRED** is a Python package for high-resolution depth estimation on 360° equirectangular panoramic images using batched PromptDA inference with optimized GPU processing.

## Features

- **Spherical Patch Sampling**: Uses icosahedral subdivision for uniform coverage of the sphere
- **GPU Batch Processing**: Process multiple patches simultaneously for maximum throughput
- **Intelligent Fusion**: Combines overlapping patches using:
  - Tukey windowing for smooth blending
  - Latitude-based weighting to reduce polar distortion
  - Consistency-based weighting (prompt-prediction agreement)
  - Multiple fusion modes (depth, inverse-depth, log-depth)
- **Flexible I/O**:
  - Support for multiple depth formats (PNG 16-bit, NPZ)
  - Point cloud export (PLY format)
  - Depth visualization colormaps
- **Performance Profiling**: Detailed timing breakdowns for optimization

## Installation

### From Source

```bash
git clone https://github.com/magus74/FRED.git
cd FRED
pip install -e .
```

### Requirements

- Python >= 3.8
- PyTorch >= 1.10.0
- CUDA (recommended for GPU acceleration)
- PromptDA model (from HuggingFace)

See `requirements.txt` for complete dependencies. PromptDA is installed from its official GitHub repository. For tests, install `pip install -e .[dev]` and run `pytest`.

## Quick Start

### Basic Usage

```bash
fred \
  --image input_panorama.jpg \
  --prompt-depth lowres_depth.png \
  --prompt-units qmm \
  --out output_depth.png \
  --subdiv 1 \
  --patch 1008 \
  --batch-size 4 \
  --profile
```

### With Visualization and Point Cloud

```bash
fred \
  --image input_panorama.jpg \
  --prompt-depth lowres_depth.png \
  --prompt-units qmm \
  --out output_depth.png \
  --depth-vis depth_colormap.png \
  --depth-vis-width 4096 \
  --depth-vis-height 2048 \
  --ply output_pointcloud.ply \
  --ply-binary \
  --ply-stride 2 \
  --subdiv 1 \
  --patch 1008 \
  --batch-size 8 \
  --lat-weight \
  --profile
```

## Command-Line Arguments

### Required Arguments

- `--image`: Path to input equirectangular RGB image
- `--prompt-depth`: Path to low-resolution prompt depth map
- `--out`: Output path for high-resolution depth (16-bit PNG in qmm)

### Depth Units

- `--prompt-units`: Units of prompt depth
  - `m` - meters
  - `mm` - millimeters
  - `qmm` - quarter-millimeters (default for indoor precision)

### Processing Parameters

- `--subdiv`: Icosahedron subdivision level (0-3, default: 1)
  - Level 0: 20 faces
  - Level 1: 80 faces
  - Level 2: 320 faces
  - Level 3: 1280 faces
- `--patch`: Patch resolution (must be multiple of 14, default: 1008)
- `--margin-deg`: Angular margin in degrees for overlap (default: 6.0)
- `--batch-size`: Number of patches per GPU batch (default: 4)
- `--no-batch`: Disable batching (sequential processing)

### Fusion Options

- `--fusion-mode`: Depth fusion method
  - `invdepth` - Inverse depth (default, better for accuracy)
  - `depth` - Linear depth
  - `logdepth` - Log depth
- `--lat-weight`: Apply latitude-based weighting (reduces polar distortion)
- `--window-radius`: Tukey window taper parameter 0-1 (default: 0.35)
- `--consistency-tau-mm`: Consistency decay constant in mm (default: 300.0)

### Output Options

- `--ply`: Output PLY point cloud path (optional)
- `--ply-binary`: Write binary PLY (faster, smaller)
- `--ply-stride`: Subsample stride for PLY (default: 1)
- `--depth-vis`: Depth visualization colormap path (optional)
- `--depth-vis-width`: Visualization width (default: 4096)
- `--depth-vis-height`: Visualization height (default: 2048)
- `--save-weights`: Save fusion weight map (for iterative refinement)
- `--save-consistency`: Save consistency map (for debugging)

### Other Options

- `--device`: PyTorch device (default: cuda:0)
- `--model`: HuggingFace model name (default: depth-anything/prompt-depth-anything-vitl)
- `--profile`: Print detailed timing breakdown

## Python API

You can also use FRED as a Python library:

```python
from fred import run_pipeline

run_pipeline(
    image_path="panorama.jpg",
    prompt_depth_path="depth_lowres.png",
    prompt_units="qmm",
    out_path="depth_highres.png",
    ply_path="pointcloud.ply",
    ply_binary=True,
    ply_stride=2,
    depth_vis_path="depth_vis.png",
    depth_vis_w=4096,
    depth_vis_h=2048,
    subdiv=1,
    patch=1008,
    margin_deg=6.0,
    device_str="cuda:0",
    batch_size=4,
    lat_weight=True,
    profile=True
)
```

## Module Organization

```
fred/
├── __init__.py      # Package initialization and exports
├── io.py            # Image and depth map loading/saving
├── geometry.py      # Icosahedral mesh generation
├── projections.py   # Gnomonic and ERP transformations
├── fusion.py        # Depth patch fusion with windowing
├── export.py        # Point cloud PLY export
├── pipeline.py      # Main processing pipeline
└── cli.py           # Command-line interface
```

## Performance Tips

1. **Batch Size**: Tune based on GPU memory
   - 4-8 for 8GB VRAM
   - 16-32 for 24GB VRAM

2. **Patch Resolution**: Balance quality vs speed
   - 1008x1008: Good quality, moderate speed
   - 1400x1400: Higher quality, slower
   - 700x700: Faster, lower quality

3. **Subdivision Level**: More faces = better coverage
   - Level 1 (80 faces): Good for most cases
   - Level 2 (320 faces): Better quality, 4x slower

4. **Profiling**: Use `--profile` to identify bottlenecks

## Citation

If you use FRED in your research, please cite:

```bibtex
@software{fred2026agus,
  title={FRED: Fast Refinement for Equirectangular Depth},
  author={Marco Agus},
  year={2026},
  url={https://github.com/magus74/FRED}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built on top of [PromptDA](https://github.com/DepthAnything/PromptDA)

=======
# FRED
Fred
>>>>>>> 8f1a7d008248052f04f21eeac1d13016e50421f0
