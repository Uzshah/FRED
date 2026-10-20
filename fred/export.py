"""
Point cloud export utilities for PLY format.
"""

import os
import numpy as np


def _ensure_parent_directory(path):
    parent = os.path.dirname(os.path.abspath(os.fspath(path)))
    os.makedirs(parent, exist_ok=True)


def save_point_cloud_ply_equirect(
    depth_m: np.ndarray,
    rgb01: np.ndarray,
    out_path: str,
    stride: int = 4,
    binary: bool = True,
):
    """
    Export equirectangular depth map to colored point cloud in PLY format.

    Args:
        depth_m: Depth map in meters (H, W)
        rgb01: RGB image with values in [0, 1] (H, W, 3)
        out_path: Output PLY file path
        stride: Subsampling stride (use every Nth pixel)
        binary: Write binary PLY (faster) vs ASCII
    """
    H, W = depth_m.shape
    uu = np.arange(0, W, stride, dtype=np.int32)
    vv = np.arange(0, H, stride, dtype=np.int32)
    U, V = np.meshgrid(uu, vv)
    D = depth_m[V, U]
    C = rgb01[V, U]

    mask = np.isfinite(D) & (D > 0)
    U = U[mask].astype(np.float32)
    V = V[mask].astype(np.float32)
    D = D[mask].astype(np.float32)
    C = (np.clip(C[mask], 0.0, 1.0) * 255.0).astype(np.uint8)
    if U.size == 0:
        raise ValueError("No valid depth to export as point cloud.")

    # Convert equirectangular coordinates to 3D points
    lam = (U / (W - 1)) * (2.0 * np.pi) - np.pi
    phi = (V / (H - 1)) * np.pi - (np.pi / 2.0)
    cosphi = np.cos(phi)
    x = cosphi * np.cos(lam) * D
    y = np.sin(phi) * D
    z = cosphi * np.sin(lam) * D

    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = (
        x[good].astype(np.float32),
        y[good].astype(np.float32),
        z[good].astype(np.float32),
    )
    C = C[good]
    N = x.shape[0]
    _ensure_parent_directory(out_path)

    if binary:
        header = f"""ply
format binary_little_endian 1.0
element vertex {N}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        with open(out_path, "wb") as f:
            f.write(header.encode("ascii"))
            vert_dtype = np.dtype(
                [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
                align=False,
            )
            chunk = 2_000_000
            for start in range(0, N, chunk):
                end = min(start + chunk, N)
                rec = np.empty(end - start, dtype=vert_dtype)
                rec["x"] = x[start:end]
                rec["y"] = y[start:end]
                rec["z"] = z[start:end]
                rec["red"] = C[start:end, 0]
                rec["green"] = C[start:end, 1]
                rec["blue"] = C[start:end, 2]
                f.write(rec.tobytes())
    else:
        header = f"""ply
format ascii 1.0
element vertex {N}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        with open(out_path, "w") as f:
            f.write(header)
            for i in range(N):
                f.write(f"{x[i]:.6f} {y[i]:.6f} {z[i]:.6f} {C[i,0]} {C[i,1]} {C[i,2]}\n")
