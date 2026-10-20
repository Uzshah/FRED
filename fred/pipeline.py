"""
Main FRED pipeline for batched depth estimation on equirectangular images.
"""

import math
import time
from pathlib import Path
import torch
from PIL import Image

from .io import (
    load_equirect_rgb_as_tensor,
    load_prompt_depth_as_tensor,
    save_depth_u16_qmm_png,
    save_depth_colormap,
)
from .geometry import (
    build_icosahedron,
    subdivide_geodesic,
    tri_face_centers,
    face_rotation_from_center,
    max_angular_radius,
)
from .projections import (
    build_gnomonic_grid,
    rotate_dirs,
    rays_to_equirect_uv,
    sample_bilinear_grid,
)
from .fusion import fuse_depth_patches
from .export import save_point_cloud_ply_equirect


def run_pipeline(
    image_path,
    prompt_depth_path,
    prompt_units,
    out_path,
    ply_path,
    ply_binary,
    ply_stride,
    depth_vis_path,
    depth_vis_w,
    depth_vis_h,
    subdiv,
    patch,
    margin_deg,
    device_str,
    model_name="depth-anything/prompt-depth-anything-vitl",
    consistency_tau_mm=300.0,
    window_radius=0.35,
    fusion_mode="invdepth",
    lat_weight=True,
    wrap_u=True,
    batch_size=4,
    no_batch=False,
    profile=False,
    save_weights=False,
    save_consistency=False,
):
    """
    Run the full FRED pipeline: batched per-patch inference + depth fusion.

    Args:
        image_path: Path to input equirectangular RGB image
        prompt_depth_path: Path to low-resolution prompt depth map
        prompt_units: Units of prompt depth ('m', 'mm', or 'qmm')
        out_path: Output path for high-res depth (16-bit PNG in qmm)
        ply_path: Optional output path for point cloud
        ply_binary: Write binary PLY format
        ply_stride: Subsampling stride for PLY export
        depth_vis_path: Optional output path for depth visualization
        depth_vis_w: Width for depth visualization
        depth_vis_h: Height for depth visualization
        subdiv: Icosahedron subdivision level
        patch: Patch resolution (must be multiple of 14)
        margin_deg: Angular margin in degrees for patch overlap
        device_str: PyTorch device string ('cuda:0', 'cpu', etc.)
        model_name: HuggingFace model name for PromptDA
        consistency_tau_mm: Decay constant for consistency weighting (mm)
        window_radius: Tukey window taper parameter [0, 1]
        fusion_mode: Depth fusion mode ('depth', 'invdepth', 'logdepth')
        lat_weight: Apply latitude-based weighting
        wrap_u: Wrap horizontally for equirectangular
        batch_size: Number of patches to process together
        no_batch: Force sequential processing
        profile: Print detailed timing breakdown
        save_weights: Save fusion weight map
        save_consistency: Save consistency map
    """
    from promptda.promptda import PromptDA
    device = torch.device(device_str)
    if patch % 14 != 0:
        raise ValueError(f"--patch must be a multiple of 14. Got {patch}.")
    if no_batch:
        batch_size = 1

    t0 = time.perf_counter()
    # Load and resize image to target resolution
    image = load_equirect_rgb_as_tensor(
        image_path, device, target_width=depth_vis_w, target_height=depth_vis_h
    )
    H, W = image.shape[-2], image.shape[-1]
    prompt_low = load_prompt_depth_as_tensor(prompt_depth_path, prompt_units, device)
    t1 = time.perf_counter()

    # Load model
    model = PromptDA.from_pretrained(model_name).to(device).eval()

    # Generate icosahedron faces
    V, F = build_icosahedron()
    V, F = subdivide_geodesic(V, F, level=subdiv)
    C = tri_face_centers(V, F)
    faces = []
    for fi in range(F.shape[0]):
        tri = V[F[fi]]
        c = C[fi]
        R = face_rotation_from_center(c)
        theta = max_angular_radius(c, tri)
        fov = 2.0 * theta + math.radians(margin_deg)
        faces.append((R, fov))
    print(
        f"[INFO] Image: {W}x{H} | Prompt LR: {prompt_low.shape[-1]}x{prompt_low.shape[-2]} | "
        f"Faces: {len(faces)} | Patch: {patch}x{patch} | batch={batch_size}"
    )

    tau_m = consistency_tau_mm / 1000.0
    patches = []
    prep_start = time.perf_counter()

    def to_tensor_R(R):
        return torch.from_numpy(R).to(device=device, dtype=torch.float32)

    R_list = [to_tensor_R(R) for (R, fov) in faces]
    fov_list = [fov for (R, fov) in faces]
    prep_mid = time.perf_counter()

    inf_time = 0.0
    fuse_prep_time = 0.0
    h_lr, w_lr = prompt_low.shape[-2], prompt_low.shape[-1]
    with torch.inference_mode():
        for start in range(0, len(faces), batch_size):
            end = min(start + batch_size, len(faces))
            bsz = end - start
            dirs_local_list = [
                build_gnomonic_grid(patch, fov_list[i], device) for i in range(start, end)
            ]
            rays_world_list = [
                rotate_dirs(dirs_local_list[i - start], R_list[i]) for i in range(start, end)
            ]
            grid_rgb_list, u_px_list, v_px_list = zip(
                *[
                    rays_to_equirect_uv(rays_world_list[i - start], W, H)
                    for i in range(start, end)
                ]
            )

            rgb_batch = torch.cat(
                [
                    sample_bilinear_grid(image, grid_rgb_list[i - start], mode="bilinear")
                    for i in range(start, end)
                ],
                dim=0,
            )
            u_lr_list = [u_px_list[i - start] * (w_lr - 1) / (W - 1) for i in range(start, end)]
            v_lr_list = [v_px_list[i - start] * (h_lr - 1) / (H - 1) for i in range(start, end)]
            grid_lr_list = [
                torch.stack([2.0 * (u_lr / (w_lr - 1)) - 1.0, 2.0 * (v_lr / (h_lr - 1)) - 1.0], dim=-1)
                for (u_lr, v_lr) in zip(u_lr_list, v_lr_list)
            ]
            prompt_batch = torch.cat(
                [sample_bilinear_grid(prompt_low, glr, mode="bilinear") for glr in grid_lr_list],
                dim=0,
            )

            if device.type == "cuda":
                torch.cuda.synchronize()
            t_inf0 = time.perf_counter()
            depth_batch = model.predict(rgb_batch, prompt_batch)  # [B,1,P,P] meters
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_inf1 = time.perf_counter()
            inf_time += t_inf1 - t_inf0

            t_fp0 = time.perf_counter()
            for bi in range(bsz):
                D = depth_batch[bi, 0]
                Pk = prompt_batch[bi, 0]
                w_pc = torch.exp(-torch.abs(D - Pk) / tau_m).clamp_min(1e-6)
                patches.append({"D": D, "u": u_px_list[bi], "v": v_px_list[bi], "w_extra": w_pc})
            t_fp1 = time.perf_counter()
            fuse_prep_time += t_fp1 - t_fp0

    prep_end = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_fuse0 = time.perf_counter()

    # Fuse depth patches
    return_dict = save_weights or save_consistency
    fusion_result = fuse_depth_patches(
        H,
        W,
        patches,
        window_radius=window_radius,
        mode=fusion_mode,
        min_depth=1e-3,
        lat_weight=lat_weight,
        wrap_u=wrap_u,
        device=device,
        return_dict=return_dict,
    )

    # Extract outputs
    if return_dict:
        depth_spherical_m = fusion_result["depth"]
        acc_w = fusion_result["weight"]
        consistency = fusion_result["consistency"]
    else:
        depth_spherical_m, acc_w = fusion_result
        consistency = None

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_fuse1 = time.perf_counter()

    depth_np = depth_spherical_m.detach().cpu().numpy().astype("float32")

    t_io0 = time.perf_counter()
    save_depth_u16_qmm_png(depth_np, out_path)

    # Save weight map if requested
    if save_weights:
        weight_np = acc_w.detach().cpu().numpy().astype("float32")
        weight_max = weight_np.max()
        if weight_max > 0:
            weight_norm = weight_np / weight_max
        else:
            weight_norm = weight_np
        weight_u16 = (weight_norm * 65535).astype("uint16")
        weight_path = str(Path(out_path).with_name(Path(out_path).stem + "_weight.png"))
        Image.fromarray(weight_u16).save(weight_path)
        if profile:
            print(f"[INFO] Saved weight map: {weight_path}")

    # Save consistency map if requested
    if save_consistency and consistency is not None:
        consistency_np = consistency.detach().cpu().numpy().astype("float32")
        consistency_max = consistency_np.max()
        if consistency_max > 0:
            consistency_norm = consistency_np / consistency_max
        else:
            consistency_norm = consistency_np
        consistency_u16 = (consistency_norm * 65535).astype("uint16")
        consistency_path = str(Path(out_path).with_name(Path(out_path).stem + "_consistency.png"))
        Image.fromarray(consistency_u16).save(consistency_path)
        if profile:
            print(f"[INFO] Saved consistency map: {consistency_path}")

    if ply_path:
        rgb01 = image[0].permute(1, 2, 0).detach().cpu().numpy()
        save_point_cloud_ply_equirect(
            depth_np, rgb01, ply_path, stride=max(1, ply_stride), binary=ply_binary
        )
    if depth_vis_path:
        save_depth_colormap(depth_np, depth_vis_path, width=depth_vis_w, height=depth_vis_h)
    t_io1 = time.perf_counter()

    t2 = time.perf_counter()
    if profile:

        def fmt(s):
            return f"{s*1000.0:7.1f} ms"

        total = t2 - t0
        load = t1 - t0
        prep = prep_end - prep_start
        proj = prep_mid - prep_start
        grids_batches = prep - proj
        fuse = t_fuse1 - t_fuse0
        io = t_io1 - t_io0
        K = len(faces)
        pixels = H * W
        print("[TIMING]")
        print(f"  load:        {fmt(load)}")
        print(f"  prep total:  {fmt(prep)}  (proj={fmt(proj)}, batched_sampling={fmt(grids_batches)})")
        print(f"  infer:       {fmt(inf_time)}  ({K} faces, batch={batch_size})")
        print(f"  fuse:        {fmt(fuse)}")
        print(f"  io:          {fmt(io)}")
        print(f"  TOTAL:       {fmt(total)}")
        faces_per_sec = K / max(total, 1e-6)
        mpix_per_sec = (pixels / 1e6) / max(total, 1e-6)
        print(f"[THROUGHPUT] faces/s={faces_per_sec:.2f}, MPix/s@ERP={mpix_per_sec:.2f}")

    # Always print output paths
    print(f"[INFO] Output depth: {out_path}")
    if ply_path:
        print(f"[INFO] Output PLY:   {ply_path}")
    if depth_vis_path:
        print(f"[INFO] Output vis:   {depth_vis_path}")
