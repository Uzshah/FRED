"""
IO utilities for loading and saving equirectangular images and depth maps.
"""

import os
import numpy as np
import torch
from PIL import Image


def _ensure_parent_directory(path):
    parent = os.path.dirname(os.path.abspath(os.fspath(path)))
    os.makedirs(parent, exist_ok=True)


def load_equirect_rgb_as_tensor(
    path: str, device: torch.device, target_width: int = None, target_height: int = None
) -> torch.Tensor:
    """
    Load an equirectangular RGB image as a PyTorch tensor.

    Args:
        path: Path to the image file
        device: Target PyTorch device
        target_width: Optional target width for resizing
        target_height: Optional target height for resizing

    Returns:
        Tensor of shape (1, 3, H, W) with values in [0, 1]
    """
    img = Image.open(path).convert("RGB")

    # Resize if target dimensions are specified
    if target_width is not None and target_height is not None:
        original_size = img.size
        if img.size != (target_width, target_height):
            print(
                f"[INFO] Resizing image from {original_size[0]}x{original_size[1]} to {target_width}x{target_height}"
            )
            img = img.resize((target_width, target_height), resample=Image.LANCZOS)

    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def load_prompt_depth_as_tensor(path: str, units: str, device: torch.device) -> torch.Tensor:
    """
    Load a prompt depth map as a PyTorch tensor.

    Args:
        path: Path to the depth file (.png or .npz)
        units: Units of the depth values ('m', 'mm', or 'qmm')
        device: Target PyTorch device

    Returns:
        Tensor of shape (1, 1, H, W) with depth in meters
    """
    assert units in ("m", "mm", "qmm"), f"Invalid units: {units}. Must be 'm', 'mm', or 'qmm'."

    if path.endswith(".png"):
        img = Image.open(path)
        if img.mode not in ("I;16", "I", "F"):
            img = img.convert("I;16")
        arr = np.array(img, dtype=np.uint32).astype(np.float32)
        if units == "mm":
            arr = arr / 1000.0
        elif units == "qmm":
            arr = arr / 4000.0  # quarter-millimeters to meters
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
    elif path.endswith(".npz"):
        arr = np.load(path)["depth"].astype(np.float32)
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
    else:
        raise ValueError(f"Unsupported prompt depth format: {path}")


def save_depth_u16_qmm_png(depth_m: np.ndarray, out_path: str, invalid_val_qmm: int = 0):
    """
    Save depth map as 16-bit PNG in quarter-millimeters (qmm) for indoor precision.

    Args:
        depth_m: Depth map in meters
        out_path: Output file path
        invalid_val_qmm: Value to use for invalid/NaN depths (default: 0)
    """
    d = np.asarray(depth_m, dtype=np.float32)
    d = np.where(np.isfinite(d), d, np.nan)
    if np.isnan(d).any():
        d = d.copy()
        d[np.isnan(d)] = float(invalid_val_qmm) / 4000.0
    d_qmm = np.clip(d * 4000.0, 0.0, 65535.0).astype(np.uint16)
    _ensure_parent_directory(out_path)
    Image.fromarray(d_qmm).save(out_path)


def save_depth_colormap(
    depth_m: np.ndarray, out_path: str, width: int = 2048, height: int = 1024
):
    """
    Save depth map as a color-coded visualization using the plasma colormap.

    Args:
        depth_m: Depth map in meters
        out_path: Output file path
        width: Output image width
        height: Output image height
    """
    import matplotlib

    d = np.asarray(depth_m, dtype=np.float32)
    mask = np.isfinite(d) & (d > 0)
    if not np.any(mask):
        raise ValueError("No valid depth values to visualize.")
    vmin = np.percentile(d[mask], 1.0)
    vmax = np.percentile(d[mask], 99.0)
    vmax = max(vmax, vmin + 1e-6)
    d_clipped = np.clip(d, vmin, vmax)
    norm = (d_clipped - vmin) / (vmax - vmin)
    norm[~mask] = 0.0
    cmap = matplotlib.colormaps["plasma"]
    rgb = (cmap(norm)[..., :3] * 255.0).astype(np.uint8)
    im = Image.fromarray(rgb).resize((width, height), resample=Image.BILINEAR)
    _ensure_parent_directory(out_path)
    im.save(out_path)
