import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):  
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # TODO (1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        z_vals = torch.linspace(self.min_depth, self.max_depth, self.n_pts_per_ray).to(device)

        # TODO (1.4): Sample points from z values
        ndirs = ray_bundle.origins.shape[0]
        # num_pts = z_vals.shape[0]
        sample_points = torch.zeros(ndirs, self.n_pts_per_ray, 3).to(device)
        for i in range(self.n_pts_per_ray):
            sample_points[:,i,:] = ray_bundle.origins + (z_vals[i] * ray_bundle.directions)
        # sample_points = (ray_bundle.origins[:, None, :] + z_vals[:, None, None] * ray_bundle.directions[:, None, :]).repeat(1, self.n_pts_per_ray, 1).to(device)

        origins = ray_bundle.origins.reshape(-1, 1, 3).repeat(1, self.n_pts_per_ray, 1)
        directions = ray_bundle.directions.reshape(-1, 1, 3).repeat(1, self.n_pts_per_ray, 1)
        z_vals = z_vals.reshape(1, -1, 1).repeat(ray_bundle.directions.shape[0], 1, 1)
        sample_points = origins + directions * z_vals
        
        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]), #.unsqueeze(2)
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}