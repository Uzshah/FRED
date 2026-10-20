"""
Depth patch fusion with bilinear splatting and windowing.
"""

import math
import torch


def tukey_window_2d(P: int, rho: float, device, dtype=torch.float32):
    """
    Generate a 2D Tukey (tapered cosine) window for smooth edge blending.

    Args:
        P: Window size (P x P)
        rho: Taper parameter in [0, 1]. 0 = rectangular, 1 = Hann window
        device: PyTorch device
        dtype: Data type

    Returns:
        2D window (P, P)
    """
    t = torch.linspace(-1.0, 1.0, P, device=device, dtype=dtype)

    def tukey_1d(tt, rho):
        w = torch.ones_like(tt)
        if rho <= 0:
            return w
        a = 1 - rho
        lead = (torch.abs(tt) >= a) & (torch.abs(tt) <= 1)
        w[lead] = 0.5 * (1 + torch.cos(math.pi * (torch.abs(tt[lead]) - a) / rho))
        w[torch.abs(tt) > 1] = 0.0
        return w

    wx = tukey_1d(t, rho)
    wy = tukey_1d(t, rho)
    return torch.ger(wy, wx)


@torch.no_grad()
def accumulate_bilinear(
    acc_sum: torch.Tensor,
    acc_w: torch.Tensor,
    D_patch: torch.Tensor,
    W_patch: torch.Tensor,
    u_px: torch.Tensor,
    v_px: torch.Tensor,
    wrap_u: bool = True,
):
    """
    Accumulate a patch into output buffers using bilinear splatting.

    Args:
        acc_sum: Accumulated weighted sum (H, W)
        acc_w: Accumulated weights (H, W)
        D_patch: Depth values to splat (P, P)
        W_patch: Weights for splatting (P, P)
        u_px: Horizontal pixel coordinates (P, P)
        v_px: Vertical pixel coordinates (P, P)
        wrap_u: Whether to wrap horizontally for equirectangular images
    """
    H, W = acc_sum.shape
    u0 = torch.floor(u_px).long()
    v0 = torch.floor(v_px).long()
    fu = (u_px - u0.float()).clamp_(0, 1)
    fv = (v_px - v0.float()).clamp_(0, 1)
    u1 = u0 + 1
    v1 = v0 + 1
    v0 = v0.clamp_(0, H - 1)
    v1 = v1.clamp_(0, H - 1)
    if wrap_u:
        u0 = torch.remainder(u0, W)
        u1 = torch.remainder(u1, W)
    else:
        u0 = u0.clamp_(0, W - 1)
        u1 = u1.clamp_(0, W - 1)

    w00 = (1 - fu) * (1 - fv) * W_patch
    w01 = (1 - fu) * fv * W_patch
    w10 = fu * (1 - fv) * W_patch
    w11 = fu * fv * W_patch
    v00 = D_patch * w00
    v01 = D_patch * w01
    v10 = D_patch * w10
    v11 = D_patch * w11

    def scatter_add(dst, xs, ys, vals):
        idx = ys * W + xs
        dst.view(-1).index_add_(0, idx.view(-1), vals.view(-1))

    scatter_add(acc_sum, u0, v0, v00)
    scatter_add(acc_w, u0, v0, w00)
    scatter_add(acc_sum, u0, v1, v01)
    scatter_add(acc_w, u0, v1, w01)
    scatter_add(acc_sum, u1, v0, v10)
    scatter_add(acc_w, u1, v0, w10)
    scatter_add(acc_sum, u1, v1, v11)
    scatter_add(acc_w, u1, v1, w11)


@torch.no_grad()
def fuse_depth_patches(
    H: int,
    W: int,
    patches: list,
    window_radius: float = 0.35,
    mode: str = "invdepth",
    min_depth: float = 1e-3,
    lat_weight: bool = True,
    wrap_u: bool = True,
    eps: float = 1e-6,
    device: torch.device = torch.device("cuda"),
    return_dict: bool = False,
):
    """
    Fuse depth patches with optional consistency tracking.

    Args:
        H: Output height
        W: Output width
        patches: List of patch dictionaries with keys:
            - 'D': Depth values (P, P)
            - 'u': Pixel u-coordinates (P, P)
            - 'v': Pixel v-coordinates (P, P)
            - 'w_extra': Optional extra weights (P, P)
        window_radius: Tukey window taper parameter [0, 1]
        mode: Fusion mode - 'depth', 'invdepth', or 'logdepth'
        min_depth: Minimum valid depth threshold
        lat_weight: Apply latitude-based weighting (cos(latitude))
        wrap_u: Wrap horizontally for equirectangular images
        eps: Small value for numerical stability
        device: PyTorch device
        return_dict: If True, return dict with depth, weight, and consistency

    Returns:
        If return_dict=True:
            {
                'depth': fused depth map (H, W),
                'weight': normalized fusion weights (H, W),
                'consistency': mean consistency weights per pixel (H, W)
            }
        If return_dict=False:
            (fused_depth, acc_w) - backward compatible
    """
    acc_sum = torch.zeros((H, W), device=device, dtype=torch.float32)
    acc_w = torch.zeros((H, W), device=device, dtype=torch.float32)
    acc_consistency = torch.zeros((H, W), device=device, dtype=torch.float32)
    acc_patch_count = torch.zeros((H, W), device=device, dtype=torch.float32)

    if not patches:
        if return_dict:
            return {"depth": acc_sum, "weight": acc_w, "consistency": acc_consistency}
        else:
            return acc_sum, acc_w

    P = patches[0]["D"].shape[-1]
    Wwin = tukey_window_2d(P, window_radius, device)

    def lat_w_from_v(v_patch):
        phi = (v_patch / (H - 1)) * math.pi - (math.pi / 2.0)
        return torch.cos(phi).clamp_(0.0, 1.0)

    for p in patches:
        Dp = p["D"].to(device).float().clone()
        up = p["u"].to(device).float()
        vp = p["v"].to(device).float()
        Wp = Wwin
        if lat_weight:
            Wp = Wp * lat_w_from_v(vp)

        # Extract consistency weight (w_pc) if available
        w_consistency = None
        if "w_extra" in p and p["w_extra"] is not None:
            w_extra = p["w_extra"].to(device).float()
            w_consistency = w_extra.clone()  # Save for consistency tracking
            Wp = Wp * w_extra

        valid = torch.isfinite(Dp) & (Dp > min_depth)
        if mode == "invdepth":
            Dm = torch.zeros_like(Dp)
            Dm[valid] = 1.0 / Dp[valid]
        elif mode == "logdepth":
            Dm = torch.zeros_like(Dp)
            Dm[valid] = torch.log(Dp[valid].clamp(min=min_depth))
        else:
            Dm = Dp
        Wmask = valid.float()
        Dm = Dm * Wmask
        Wp = Wp * Wmask

        accumulate_bilinear(acc_sum, acc_w, Dm, Wp, up, vp, wrap_u=wrap_u)

        # Accumulate consistency weights
        if w_consistency is not None:
            w_cons_masked = w_consistency * Wmask
            accumulate_bilinear(
                acc_consistency, acc_patch_count, w_cons_masked, Wmask, up, vp, wrap_u=wrap_u
            )

    fused = acc_sum / (acc_w + eps)
    if mode == "invdepth":
        fused = torch.where(fused > eps, 1.0 / fused, torch.zeros_like(fused))
    elif mode == "logdepth":
        fused = torch.exp(fused)

    # Compute mean consistency per pixel
    consistency_mean = torch.where(
        acc_patch_count > eps,
        acc_consistency / acc_patch_count,
        torch.zeros_like(acc_consistency),
    )

    if return_dict:
        return {"depth": fused, "weight": acc_w, "consistency": consistency_mean}
    else:
        # Backward compatible return
        return fused, acc_w
