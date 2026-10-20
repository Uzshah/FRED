import torch
from fred.fusion import fuse_depth_patches

def test_single_patch_depth_is_preserved():
    n = 4
    patch = {'D': torch.ones(n, n), 'u': torch.arange(n).repeat(n, 1).float(), 'v': torch.arange(n).view(-1, 1).repeat(1, n).float()}
    depth, weight = fuse_depth_patches(n, n, [patch], window_radius=0, lat_weight=False, device=torch.device('cpu'))
    torch.testing.assert_close(depth, torch.ones_like(depth), atol=1e-5, rtol=0)
    assert torch.all(weight > 0)
