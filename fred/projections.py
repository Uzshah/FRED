"""
Projection utilities for gnomonic and equirectangular transformations.
"""

import math
import torch


def build_gnomonic_grid(P, fov_rad, device):
    """
    Build a gnomonic (rectilinear) projection grid.

    Args:
        P: Grid resolution (P x P)
        fov_rad: Field of view in radians
        device: PyTorch device

    Returns:
        Normalized ray directions (P, P, 3)
    """
    half = math.tan(0.5 * fov_rad)
    xs = torch.linspace(-half, half, P, device=device)
    ys = torch.linspace(-half, half, P, device=device)
    yv, xv = torch.meshgrid(ys, xs, indexing="ij")
    dirs = torch.stack([xv, yv, torch.ones_like(xv)], dim=-1)
    return dirs / torch.linalg.norm(dirs, dim=-1, keepdims=True)


def rotate_dirs(dirs_local, R):
    """
    Rotate local ray directions to world space using rotation matrix.

    Args:
        dirs_local: Local ray directions (..., 3)
        R: Rotation matrix (3, 3)

    Returns:
        World-space ray directions (..., 3)
    """
    return torch.einsum("...j,ij->...i", dirs_local, R)


def rays_to_equirect_uv(rays, W, H):
    """
    Convert 3D ray directions to equirectangular UV coordinates.

    Args:
        rays: Ray directions (..., 3) in (x, y, z) format
        W: Equirectangular image width
        H: Equirectangular image height

    Returns:
        grid: Normalized grid coordinates for grid_sample (..., 2) in [-1, 1]
        u: Pixel u-coordinates (...) in [0, W-1]
        v: Pixel v-coordinates (...) in [0, H-1]
    """
    x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
    lam = torch.atan2(z, x)
    phi = torch.asin(torch.clamp(y, -1.0, 1.0))
    u = (lam + math.pi) / (2.0 * math.pi) * (W - 1)
    v = (phi + math.pi / 2.0) / math.pi * (H - 1)
    u_norm = 2.0 * (u / (W - 1)) - 1.0
    v_norm = 2.0 * (v / (H - 1)) - 1.0
    grid = torch.stack([u_norm, v_norm], dim=-1)
    return grid, u, v


def sample_bilinear_grid(tensor, grid, mode="bilinear"):
    """
    Sample from a tensor using bilinear interpolation.

    Args:
        tensor: Input tensor (B, C, H, W)
        grid: Sampling coordinates (..., 2) in [-1, 1]
        mode: Interpolation mode

    Returns:
        Sampled values (B, C, ...)
    """
    grid_in = grid.unsqueeze(0).to(tensor.dtype)
    return torch.nan_to_num(
        torch.nn.functional.grid_sample(
            tensor, grid_in, mode=mode, padding_mode="border", align_corners=True
        ),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
