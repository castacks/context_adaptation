import numpy as np
import matplotlib.pyplot as plt
import heapq
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import yaml
import os
import cv2
from scipy.spatial.transform import Rotation as R
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torchdiffeq
from physics_atv_visual_mapping.pointcloud_colorization.torch_color_pcl_utils import *

# import torchsde
# from torchdyn.core import NeuralODE
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from torchvision.transforms.functional import hflip
from tqdm import tqdm
# from efficientnet_pytorch import EfficientNet
from typing import List, Dict, Optional, Tuple, Callable
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
# from torchcfm.models.unet import UNetModel
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
import time
from diffusers import DDPMScheduler, DDIMScheduler
from torch.nn import MSELoss, HuberLoss
import rasterio
from torch.utils.data import ConcatDataset
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingWarmRestarts
import torch
import torch.nn as nn
import torch.nn.functional as F
# from warmup_scheduler import GradualWarmupScheduler
from torch.nn.functional import grid_sample

from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

import wandb

CMAP = cm.magma
CMAP_JET = cm.jet

def emd_loss_fn(pred_probs, target_probs):
    cdf_pred = torch.cumsum(pred_probs, dim=1)
    cdf_target = torch.cumsum(target_probs, dim=1)
    return torch.mean((cdf_pred - cdf_target)**2)

import torch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def overlay_heatmap_on_image(image, heatmap, max_val = 3., threshold=0.01, alpha=0.6, cmap='jet'):

    H, W = image.shape[:2]
    heatmap = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)

    # Normalize heatmap to [0, 1]
    # hm_norm = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    hm_norm = np.clip(heatmap/max_val, 0, 1.)
    # Apply matplotlib colormap (returns RGBA)
    # cmap_fn = cm.get_cmap(cmap)
    hm_color = CMAP_JET(hm_norm)[..., :3]  # drop alpha channel

    # Create binary mask where heatmap exceeds threshold
    mask = (hm_norm > threshold)[..., None].astype(float)

    # Convert image to float in [0,1] if needed
    if image.dtype == np.uint8:
        img_float = image.astype(np.float32) / 255.0
    else:
        img_float = image.copy()

    # Blend where mask is active
    overlay = img_float * (1 - alpha * mask) + hm_color * (alpha * mask)

    # Convert back to uint8
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    return overlay


def compute_pixel_headings(K, H, W, H_orig=None, W_orig=None, device='cpu'):
    """
    Compute per-pixel headings for a camera image of arbitrary size.

    Args:
        K: camera intrinsic matrix [3,3] (numpy or torch) for original resolution
        H, W: output image size
        H_orig, W_orig: original image size that K was calibrated for (if None, assume H_orig=H, W_orig=W)
        device: 'cpu' or 'cuda'

    Returns:
        pixel_headings: [H, W] tensor with heading angles in radians (-pi to pi)
    """
    if isinstance(K, np.ndarray):
        K = torch.from_numpy(K).to(torch.float32).to(device)
    else:
        K = K.to(torch.float32).to(device)

    if H_orig is None:
        H_orig = H
    if W_orig is None:
        W_orig = W

    # Rescale intrinsics
    scale_x = W / W_orig
    scale_y = H / H_orig
    fx = K[0, 0] * scale_x
    fy = K[1, 1] * scale_y
    cx = K[0, 2] * scale_x
    cy = K[1, 2] * scale_y

    # Create pixel grid
    u = torch.arange(W, device=device)
    v = torch.arange(H, device=device)
    uu, vv = torch.meshgrid(u, v, indexing='xy')  # [H, W]

    # Compute horizontal heading only
    pixel_headings = torch.atan2(uu - cx, fx)

    # plt.imshow(pixel_headings.cpu().numpy())
    # plt.show()

    return pixel_headings  # [H, W]


def project_points_rectified(
    xyz_robot,                # (N,3) points in robot frame
    image,           # optional (H, W) to mask to image bounds
    color = (0,0,255),
    thickness=2,
    alpha = .4
):

    extrinsics = torch.Tensor([[ 0.0033, -0.9997,  0.0236,  0.1726],
        [-0.2247, -0.0237, -0.9741, -0.1523],
        [ 0.9744, -0.0021, -0.2247,  0.0571],
        [ 0.0000,  0.0000,  0.0000,  1.0000]])
    
    intrinsics = torch.Tensor([[455.7750,   0.0000, 497.1180,   0.0000],
        [  0.0000, 456.3191, 251.8580,   0.0000],
        [  0.0000,   0.0000,   1.0000,   0.0000],
        [  0.0000,   0.0000,   0.0000,   1.0000]])
    
    P = get_projection_matrix(intrinsics, extrinsics).to('cpu')

    img = torch.from_numpy(image)
    xyz_robot = torch.from_numpy(xyz_robot)
    footprint_pcl_px_in_frame, ind_in_frame = get_pixel_projection(xyz_robot, P.unsqueeze(0),img.unsqueeze(0))
    footprint_pcl_px_in_frame = footprint_pcl_px_in_frame[0]
    ind_in_frame = ind_in_frame[0]

    footprint_pcl_px_in_frame = footprint_pcl_px_in_frame[ind_in_frame]
    # print(footprint_pcl_px_in_frame.shape)
    # traj_mask_px = torch.unique(footprint_pcl_px_in_frame.long(), dim=0)
    # print(traj_mask_px.shape)
    overlay = image.copy()
    traj_mask_px = footprint_pcl_px_in_frame.long().numpy()
    plot_color = tuple(int(x) for x in color)
    for i in range(len(traj_mask_px)-1):
        pt1 = tuple(traj_mask_px[i])
        pt2 = tuple(traj_mask_px[i+1])
        cv2.line(overlay, pt1, pt2, color=plot_color, thickness=thickness)

    image = cv2.addWeighted(overlay, alpha, image, 1-alpha, 0.)
    return image


def heading_cost_field(cost_map, start, threshold, K=64, min_dist=15, max_dist=50, step=1.0):
    """
    Vectorized computation of cost-to-go for each heading from a starting point.
    """
    M, N = cost_map.shape
    sy, sx = start
    headings = np.linspace(0, 2*np.pi, K, endpoint=False)

    steps = np.arange(min_dist, max_dist + step, step)  # shape (S,)
    S = steps.size

    dx = np.cos(headings)[:, None] * steps[None, :]
    dy = np.sin(headings)[:, None] * steps[None, :]
    xs = sx + dx
    ys = sy + dy

    xi = np.clip(xs.astype(int), 0, N - 1)
    yi = np.clip(ys.astype(int), 0, M - 1)

    ray_costs = cost_map[yi, xi]  # shape (K, S)

    mask = (ray_costs > threshold) | (xs < 0) | (xs >= N) | (ys < 0) | (ys >= M)

    mask_cumsum = np.cumsum(mask, axis=1)
    ray_costs[mask_cumsum > 0] = 0.0  # zero cost after obstacle
    distances = np.where(mask_cumsum > 0, 0.0, step)  # distance steps

    total_costs = ray_costs.sum(axis=1)
    total_dists = distances.sum(axis=1) + 1e-6

    costs = total_costs / total_dists
    scores = total_dists / (1 + 10*costs)
    scores = scores/max_dist

    # scores = total_dists/max_dist

    # print(scores.max(), scores.min())

    return headings, scores




def plot_cost_map_and_heading(cost_map, start, threshold, headings, costs):
    """
    Plot the cost map (with lethal contour) and polar heading heatmap side by side.
    """
    # Normalize heading costs
    normed = (costs - np.nanmin(costs)) / (np.nanmax(costs) - np.nanmin(costs) + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: cost map with lethal contour
    ax = axes[0]
    ax.imshow(cost_map, cmap='viridis')
    ax.contour(cost_map > threshold, levels=[0.5], colors='red', linewidths=1)
    ax.plot(start[1], start[0], 'wo', markersize=8, markeredgecolor='k')
    ax.set_title("Cost Map with Lethal Regions (red)")

    # Right: polar heading heatmap
    ax = axes[1] = plt.subplot(1, 2, 2, polar=True)
    ax.bar(headings, np.ones_like(costs), width=2*np.pi/len(costs),
           color=plt.cm.inferno(normed), edgecolor='none')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(1)
    ax.set_title("Heading Cost Heatmap (lower = better)")

    plt.tight_layout()
    plt.show()


def plot_cost_map_and_heading_image(cost_map, start, threshold, headings, costs, img, max_val=5., sname=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec

    # Normalize heading costs
    # normed = (costs - np.nanmin(costs)) / (np.nanmax(costs) - np.nanmin(costs) + 1e-9)

    forward_mask = (np.cos(headings) >= 0)  # same as abs(heading) <= π/2
    forward_costs = costs[forward_mask]

    if np.any(np.isfinite(forward_costs)):
        min_c = np.nanmin(forward_costs)
        max_c = np.nanmax(forward_costs)
    else:
        # fallback if all NaNs or inf
        min_c, max_c = np.nanmin(costs), np.nanmax(costs)

    # Normalize all headings using forward range
    # normed = (costs - min_c) / (max_c - min_c + 1e-9)
    normed = costs/max_val
    normed = np.clip(normed, 0, 1)

    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 3, figure=fig)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2], projection='polar')

    # Cost map
    ax0.imshow(cost_map, cmap='viridis')
    ax0.contour(cost_map > threshold, levels=[0.5], colors='red', linewidths=1)
    ax0.plot(start[1], start[0], 'wo', markersize=8, markeredgecolor='k')
    ax0.set_title("Cost Map with Lethal Regions (red)")
    viz_range = 250
    ax0.set_xlim(start[1] - viz_range, start[1] + viz_range)
    ax0.set_ylim(start[0] + viz_range, start[0] - viz_range)
    # arrow_length = 55.0  # meters, or scale by score if you want
    # points = []
    # for h, s in zip(headings, costs):
    #     dx = np.cos(h) * arrow_length
    #     dy = np.sin(h) * arrow_length
    #     dz = 0.0  # assume points are at ground level
    #     points.append([dx, dy, dz])
    # points = np.array(points)  # shape (K,3)
    # points[:, 2] -= 1.7  # camera height
    # # print(points)
    

    # origin = np.array([[0,0,-1.7]])  # camera center in local frame
    # drawn = img.copy()
    # for p, score in zip(points, normed):
    #     path_color = CMAP(score)
    #     drawn = project_points_rectified(np.vstack([origin, p]).astype(np.float32), drawn, np.array(path_color[:3])*255, thickness=4, alpha=1.)

    num_pts = 80  # how smooth the projected line should look
    origin = np.array([0, 0, -1.7])  # camera origin in local frame
    drawn = img.copy()
    for h, s in zip(headings, normed):
        # Length of arrow can be fixed or proportional to score
        arrow_length = 10.0 * (s / np.max(normed))  # meters
        end = np.array([np.cos(h)*arrow_length, np.sin(h)*arrow_length, -1.7])
        
        # Interpolate between origin and end
        t = np.linspace(0, 1, num_pts)
        line_pts = origin + np.outer(t, end - origin)  # (num_pts, 3)
        line_pts = line_pts.astype(np.float32)

        path_color = CMAP(s)
        # Project and draw
        drawn = project_points_rectified(line_pts, drawn, np.array(path_color[:3])*255, thickness=4, alpha=1.)

    # Image
    ax1.imshow(drawn)
    ax1.set_title("Original Image")
    ax1.axis('off')

    # Polar heatmap
    ax2.bar(headings, np.ones_like(costs), width=2*np.pi/len(costs),
            color=plt.cm.inferno(normed), edgecolor='none')
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(1)
    ax2.set_title("Heading Cost Heatmap (higher = better)")

    plt.tight_layout()
    if sname is not None:
        plt.savefig(sname + ".png", dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close("all")
    else:
        plt.show()

def plot_heatmap_and_heading_image(heatmap, start, threshold, headings, costs, img, sname=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec

    heatmap_viz = heatmap.copy()
    heatmap_viz[heatmap< -1.0] = 0
    overlay = overlay_heatmap_on_image(img, heatmap_viz)
    # plt.imshow(overlay)
    # plt.show()

    # Normalize heading costs
    # normed = (costs - np.nanmin(costs)) / (np.nanmax(costs) - np.nanmin(costs) + 1e-9)

    forward_mask = (np.cos(headings) >= 0)  # same as abs(heading) <= π/2
    forward_costs = costs[forward_mask]

    if np.any(np.isfinite(forward_costs)):
        min_c = np.nanmin(forward_costs)
        max_c = np.nanmax(forward_costs)
    else:
        # fallback if all NaNs or inf
        min_c, max_c = np.nanmin(costs), np.nanmax(costs)

    # Normalize all headings using forward range
    normed = (costs - min_c) / (max_c - min_c + 1e-9)
    normed = np.clip(normed, 0, 1)

    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 3, figure=fig)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2], projection='polar')

    # Cost map
    ax0.imshow(overlay, cmap='viridis')
    ax0.set_title("Affordance Heatmap")
    
    num_pts = 80  # how smooth the projected line should look
    origin = np.array([0, 0, -1.7])  # camera origin in local frame
    drawn = img.copy()
    for h, s in zip(headings, normed):
        # Length of arrow can be fixed or proportional to score
        arrow_length = 10.0 * (s / np.max(normed))  # meters
        end = np.array([np.cos(h)*arrow_length, np.sin(h)*arrow_length, -1.7])
        
        # Interpolate between origin and end
        t = np.linspace(0, 1, num_pts)
        line_pts = origin + np.outer(t, end - origin)  # (num_pts, 3)
        line_pts = line_pts.astype(np.float32)

        path_color = CMAP(s)
        # Project and draw
        drawn = project_points_rectified(line_pts, drawn, np.array(path_color[:3])*255, thickness=4, alpha=1.)

    # Image
    ax1.imshow(drawn)
    ax1.set_title("Original Image")
    ax1.axis('off')

    # Polar heatmap
    ax2.bar(headings, np.ones_like(costs), width=2*np.pi/len(costs),
            color=plt.cm.inferno(normed), edgecolor='none')
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(1)
    ax2.set_title("Heading Cost Heatmap (higher = better)")

    plt.tight_layout()
    if sname is not None:
        # plt.savefig(sname + ".png", dpi=300, bbox_inches='tight')
        plt.savefig(sname + ".jpg", dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close("all")
    else:
        plt.show()

class ConvNeXtStage1(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            self.model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        else:
            self.model = convnext_tiny(weights=None)

        # Modify input to 1 channel
        old_conv = self.model.features[0][0]
        self.model.features[0][0] = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )

        if pretrained:
            with torch.no_grad():
                self.model.features[0][0].weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

        # Only keep stem + stage 1
        self.backbone = nn.Sequential(
            self.model.features[0],  # stem (conv+norm+activation)
            self.model.features[1]   # stage 1
        )

    def forward(self, x):
        return self.backbone(x)  # (B, 96, 56, 56)

# class DinoHeadingCostHead(nn.Module):
#     def __init__(self, in_channels=1152, hidden_dim=256, num_bins=128, directional=True):
#         super().__init__()
#         self.directional = directional


#         # self.fpv_encoder = ConvNeXtStage1()


#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
#             nn.ReLU(inplace=True)
#         )

#         # Pooling layer
#         if directional:
#             # Keep width equal to heading bins, collapse height
#             self.pool = nn.AdaptiveAvgPool2d((1, num_bins))
#             self.final = nn.Conv2d(hidden_dim, 1, kernel_size=1)  # [B,1,1,num_bins] -> [B,num_bins]
#         else:
#             # Collapse both spatial dimensions, produce global feature
#             self.pool = nn.AdaptiveAvgPool2d(1)
#             self.final = nn.Linear(hidden_dim, num_bins)

#     def forward(self, x):
#         """
#         x: [B, C, H, W]
#         returns: [B, num_bins]
#         """
#         # print('1, ', x.shape)
#         # x = self.fpv_encoder(x)
#         # print('2, ', x.shape)
#         f = self.conv(x)
#         # print('3, ', f.shape)

#         if self.directional:
#             f = self.pool(f)            # [B, hidden_dim, 1, num_bins]
#             out = self.final(f)         # [B,1,1,num_bins]
#             return out.squeeze(1).squeeze(1)  # [B,num_bins]
#         else:
#             f = self.pool(f)            # [B, hidden_dim, 1, 1]
#             f = f.flatten(1)            # [B, hidden_dim]
#             return self.final(f)        # [B, num_bins]
    
class DinoHeadingCostHeadV2(nn.Module):
    def __init__(self, in_channels=1152, hidden_dim=128, num_bins=128, upsample_scale=8):
        super().__init__()
        self.num_bins = num_bins
        self.upsample_scale = upsample_scale
        self.in_channels = in_channels

        # Decide number of doubling layers to reach upsample_scale
        num_upsample_layers = int(np.ceil(np.log2(upsample_scale)))
        current_scale = 1

        layers = [nn.Conv2d(in_channels, hidden_dim, kernel_size=1), nn.ReLU()]
        for _ in range(num_upsample_layers):
            layers.append(nn.ConvTranspose2d(
                hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1
            ))
            layers.append(nn.ReLU())
            current_scale *= 2

        # If current_scale overshoots upsample_scale, use final Conv2d to reduce spatial size
        final_kernel = 1
        layers.append(nn.Conv2d(hidden_dim, 1, kernel_size=final_kernel))

        self.decoder = nn.Sequential(*layers)

        

    def register_headings(self, K, H_orig, W_orig):
        dummy_input = torch.zeros(1, self.in_channels,28, 54).cuda()
        with torch.no_grad():
            heatmap = self.decoder(dummy_input)
        H, W = heatmap.shape[2], heatmap.shape[3]


        pixel_headings = compute_pixel_headings(K, H, W, H_orig=H_orig, W_orig=W_orig)
        #TODO is this ok to do
        pixel_headings *= -1


        bin_edges = torch.linspace(-np.pi/2, np.pi/2, self.num_bins + 1)
        bin_indices = torch.bucketize(pixel_headings, bin_edges) - 1
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)

        bin_indices_flat = bin_indices.flatten().cuda()
        self.register_buffer('bin_indices', bin_indices_flat)
        pixel_counts = torch.bincount(bin_indices_flat, minlength=self.num_bins).float()
        self.register_buffer('pixel_counts', pixel_counts)

        self.valid_idxs = torch.where(pixel_counts != 0)[0]


    def forward(self, x):
        B, _, H_small, W_small = x.shape
        device = x.device

        #TODO this should be done before precomputing pixel headings
        h, w = x.shape[-2:]
        pad_h = 1 if h % 2 != 0 else 0
        pad_w = 1 if w % 2 != 0 else 0
        if pad_h > 0 or pad_w > 0:
            # F.pad takes (left, right, top, bottom)
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # Decode to per-pixel heatmap
        heatmap = self.decoder(x).squeeze(1)  # [B, H_up, W_up]
        # heatmap = torch.sigmoid(heatmap)
        heatmap = F.softplus(heatmap)

        # print(heatmap.max())

        # Optionally crop/trim to match pixel_headings if minor mismatch
        # print(heatmap.shape, '----------')
        # H_target, W_target = self.pixel_headings.shape[0:2]
        # H_up, W_up = heatmap.shape[1:3]
        # if H_up != H_target or W_up != W_target:
        #     heatmap = heatmap[:, :H_target, :W_target]

        B = heatmap.shape[0]
        binned_scores = torch.zeros(B, self.num_bins, device=device)
        heatmap_flat = heatmap.flatten(1)  # [B, H*W]
        binned_scores = binned_scores.scatter_add(1, self.bin_indices.unsqueeze(0).expand(B, -1), heatmap_flat)

        # Normalize
        binned_scores = binned_scores / (self.pixel_counts.unsqueeze(0) + 1e-6)

        # binned_scores = torch.sigmoid(binned_scores)

        return heatmap, binned_scores  # [B, num_bins]

class DinoHeadingCostHeadV2_highres(nn.Module):
    def __init__(self, in_channels=1152, hidden_dim=128, num_bins=128, upsample_scale=8):
        super().__init__()
        self.num_bins = num_bins
        self.upsample_scale = upsample_scale
        self.in_channels = in_channels

        # Decide number of doubling layers to reach upsample_scale
        num_upsample_layers = int(np.ceil(np.log2(upsample_scale)))
        current_scale = 1

        # layers = [nn.Conv2d(in_channels, hidden_dim, kernel_size=1), nn.ReLU()]
        layers = [nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(hidden_dim), nn.ReLU()]

        for _ in range(num_upsample_layers):
            layers.append(nn.ConvTranspose2d(
                hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1
            ))
            layers.append(nn.ReLU())
            current_scale *= 2

        # If current_scale overshoots upsample_scale, use final Conv2d to reduce spatial size
        final_kernel = 1
        layers.append(nn.Conv2d(hidden_dim, 1, kernel_size=final_kernel))

        self.decoder = nn.Sequential(*layers)

        

    def register_headings(self, K, H_orig, W_orig):
        # dummy_input = torch.zeros(1, self.in_channels,28, 54).cuda()
        dummy_input = torch.zeros(1, self.in_channels,224, 224).cuda()
        with torch.no_grad():
            heatmap = self.decoder(dummy_input)
        H, W = heatmap.shape[2], heatmap.shape[3]


        pixel_headings = compute_pixel_headings(K, H, W, H_orig=H_orig, W_orig=W_orig)
        #TODO is this ok to do
        pixel_headings *= -1


        bin_edges = torch.linspace(-np.pi/2, np.pi/2, self.num_bins + 1)
        bin_indices = torch.bucketize(pixel_headings, bin_edges) - 1
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)

        bin_indices_flat = bin_indices.flatten().cuda()
        self.register_buffer('bin_indices', bin_indices_flat)
        pixel_counts = torch.bincount(bin_indices_flat, minlength=self.num_bins).float()
        self.register_buffer('pixel_counts', pixel_counts)

        self.valid_idxs = torch.where(pixel_counts != 0)[0]

    def forward(self, x):
        B, _, H_small, W_small = x.shape
        device = x.device

        # print(x.shape)

        # print(x.max())

        #TODO this should be done before precomputing pixel headings
        h, w = x.shape[-2:]
        pad_h = 1 if h % 2 != 0 else 0
        pad_w = 1 if w % 2 != 0 else 0
        if pad_h > 0 or pad_w > 0:
            # F.pad takes (left, right, top, bottom)
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # Decode to per-pixel heatmap
        heatmap = self.decoder(x).squeeze(1)  # [B, H_up, W_up]
        # heatmap = torch.sigmoid(heatmap)
        heatmap = F.softplus(heatmap)

        # print(heatmap.max())

        # Optionally crop/trim to match pixel_headings if minor mismatch
        # print(heatmap.shape, '----------')
        # H_target, W_target = self.pixel_headings.shape[0:2]
        # H_up, W_up = heatmap.shape[1:3]
        # if H_up != H_target or W_up != W_target:
        #     heatmap = heatmap[:, :H_target, :W_target]

        B = heatmap.shape[0]
        binned_scores = torch.zeros(B, self.num_bins, device=device)
        heatmap_flat = heatmap.flatten(1)  # [B, H*W]
        binned_scores = binned_scores.scatter_add(1, self.bin_indices.unsqueeze(0).expand(B, -1), heatmap_flat)

        # Normalize
        binned_scores = binned_scores / (self.pixel_counts.unsqueeze(0) + 1e-6)

        # binned_scores = torch.sigmoid(binned_scores)

        return heatmap, binned_scores  # [B, num_bins]

class FrontierDataset(Dataset):
    def __init__(self, dir, transform=None, augment = False):
        """
        Args:
            grid_dir (str): Path to the directory containing occupancy grids (100x100 .npy files).
            traj_dir (str): Path to the directory containing trajectories (N x 1000 x 2 .npy files).
            transform (callable, optional): Optional transform to apply to each grid.
        """
        self.dir = dir
        self.image_dir = os.path.join(dir, 'image')
        self.dino_dir = os.path.join(dir, 'dino_img')
        # self.dino_dir = os.path.join(dir, 'cost_img')
        self.cost_dir = os.path.join(dir, 'cost_img')
        self.odom_dir = os.path.join(dir, 'odometry')
        self.gps_dir = os.path.join(dir, 'gps_odometry')
        self.map_dir = os.path.join(dir, 'bev_map_reduce')
        self.synth_dir = os.path.join(dir, 'synth_demo')
        # self.costmap = np.load('global_map.npy')
        # plt.imshow(self.costmap)
        # plt.show()
        # self.costmap = np.clip(self.costmap, .02,1.0) #TODO hacky
        costmap = np.load('../LRDP/global_costmap_gp.npy')
        # self.costmap = np.load('global_costmap_gp_pre.npy')
        # self.costmap = np.load('global_costmap_gp_new.npy')
        # self.costmap = np.load('global_costmap_gp_24.npy')

        gascola_tif_path = "/home/tartandriver/tartandriver_ws/src/core/mission_manager/gps_maps/gascola.tif"
        rr_tif_path = "/home/tartandriver/tartandriver_ws/src/core/mission_manager/gps_maps/renegade.tif"

        if 'rr' in dir:
            tif_path = rr_tif_path
        else:
            tif_path = gascola_tif_path

        self.tif = rasterio.open(tif_path)
        self.TT = torch.from_numpy(np.array(self.tif.transform).reshape(3,3))
        self.TT_inv = torch.inverse(self.TT)
        gps_odom = np.loadtxt(os.path.join(self.gps_dir, 'data.txt'))
        self.gps = gps_odom

        demo_x, demo_y = -self.gps[:,1], self.gps[:,0]

        costmap -= .18
        costmap = np.clip(costmap, 0, 1.)
        costmap /= costmap.max()

        #TODO get rid of this check when we have costmap
        if 'rr' not in dir:
            # print(demo_x)
            idx, idy = self.tif.index(demo_x, demo_y)

            from scipy.ndimage import binary_dilation

            radius = 2
            # Initialize mask for all points
            mask = np.zeros_like(costmap, dtype=bool)
            mask[idx, idy] = True

            # Create circular structuring element
            y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
            footprint = x**2 + y**2 <= radius**2

            # Dilate mask
            inflated_mask = binary_dilation(mask, structure=footprint)

            # Apply inflation
            costmap[inflated_mask] = 0.0

            # plt.imshow(costmap, cmap='magma')
            # plt.show()

        self.costmap = costmap

        self.horizon = 100

        self.augment = augment
        self.goal_cond = True

        start_stop_map = {
            '/home/tartandriver/workspace/datasets/lrdp/04_garage_to_turnpike_afternoon': [170,900],
            '/home/tartandriver/workspace/datasets/lrdp/2023-11-14-15-02-21_figure_8': [135,4955],
            # '/home/tartandriver/workspace/datasets/lrdp/2023-11-14-15-02-21_figure_8': [475,650],
            '/home/tartandriver/workspace/datasets/lrdp/turnpike_2023-09-12-12-53-32': [130,6520], #moved 100 from both sides
            '/home/tartandriver/workspace/datasets/lrdp/20251009_3_red_course': [353,2200],
            '/home/tartandriver/workspace/datasets/lrdp/20251009_4_fig8_to_horseshoe': [464,3840],
            '/home/tartandriver/workspace/datasets/lrdp/20251009_5_turnpike': [920,3395],
            '/home/tartandriver/workspace/datasets/lrdp/2025-10-16-18-00-49_snow_bag': [160,2820],
            '/home/tartandriver/workspace/datasets/lrdp/rr_demo_lap_6': [280,2750]
        }
        idx_start, idx_stop = start_stop_map[dir]



        # metadata_dir = os.path.join(dir, 'bev_map_reduce', '00000000_metadata.yaml')
        # metadata = yaml.safe_load(open(metadata_dir, 'r'))
        # resolution = metadata['resolution'][0]
        # height = int(metadata['length'][0] // resolution) + 1 #TODO
        # width = int(metadata['length'][1] // resolution) + 1
        # self.resolution, self.height, self.width = resolution, height, width

        odom = np.loadtxt(os.path.join(self.odom_dir, 'data.txt'))
        all_min = odom.min(axis=0)
        all_max = odom.max(axis=0)
        min_x, min_y = all_min[:2]
        max_x, max_y = all_max[:2]
        self.odom = odom

        

        self.context_size = 0
        self.context_stride = 5

        # tif_path = "/home/matthew/Downloads/gascola.tif"
        

        # RR temp
        # rgb_map = self.tif.read([1,2,3])
        # rgb_map = np.transpose(rgb_map, [1,2,0])
        # self.costmap = np.zeros_like(rgb_map[:,:,0]).astype(np.float32) + .3
        # print(self.TT)
        # min_x, min_y, max_x, max_y = int(min_x//resolution), int(min_y//resolution), int(max_x//resolution), int(max_y//resolution)
        # range_x = int(max_x - min_x)
        # range_y = int(max_y - min_y)
        # print("min_x: {}, min_y: {}, max_x: {}, max_y: {}".format(min_x, min_y, max_x, max_y))
        # self.min_x, self.min_y, self.max_x, self.max_y = min_x, min_y, max_x, max_y

        self.indices = sorted([
            int(fname.split('.')[0])
            for fname in os.listdir(self.image_dir)
            if fname.endswith('.png') and fname.split('.')[0].isdigit()
        ])

        # self.indices = self.indices[170:900]
        self.indices = self.indices[idx_start:idx_stop]

        self.transform = transforms.Compose([
        # transforms.Resize(256),
        transforms.ToTensor(),
        # transforms.Normalize(  # Normalize with ImageNet mean/std
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        #     )
        ])

        # base_lib = np.load('base_traj_lib.npy')
        # # print(base_lib.shape)
        # # base_lib = base_lib[::100]
        # ids = np.linspace(0,len(base_lib)-1,16).astype(int)
        # base_lib = base_lib[ids]
        # self.base_lib = base_lib[:,::4]
        # # for i,traj in enumerate(base_lib):
        # #     # color = 'g' if valid[i] else 'r'
        # #     plt.plot(traj[:, 0], traj[:, 1],c=CMAP(i/len(base_lib)))
        
        # plt.show()

        if not os.path.isdir(self.dino_dir):
            print("GENERATING FEAT IMAGES FOR " + dir)
            os.makedirs(self.dino_dir)
            self.gen_dino()
        # self.gen_cost()
        # self.gen_examples()

        self.indices = self.indices[self.context_size*self.context_stride:]

        # print(self.indices)

        img_mask = cv2.imread('/home/tartandriver/tartandriver_ws/src/perception/physics_atv_visual_mapping/data/masks/yamaha/image_left_color_mask.png')
        img_mask = img_mask[:,:,0]

        idx_str = f"{self.indices[0]:08d}"
        dino_path = os.path.join(self.dino_dir, f"{idx_str}_data.npy")
        if not os.path.exists(dino_path):
            dino_path = dino_path.replace("_data", "")
        feat_img = np.load(dino_path)
        img_mask = cv2.resize(img_mask, (feat_img.shape[-1], feat_img.shape[-2]),interpolation=cv2.INTER_NEAREST)
        self.img_mask = img_mask == 255

        self.max_dist = 300

    def to_local_frame(self, points_xy, origin_x, origin_y, origin_yaw):
        """
        Convert global (x, y) points to a local frame.

        Parameters:
            points_xy : np.ndarray of shape (N, 2)
                Global coordinates of the points.
            origin_x : float
                X coordinate of the local frame origin.
            origin_y : float
                Y coordinate of the local frame origin.
            origin_yaw : float (radians)
                Yaw of the local frame relative to the global frame.

        Returns:
            np.ndarray of shape (N, 2) with local frame coordinates.
        """
        # Translate points so origin is at (0, 0)
        translated = points_xy - np.array([origin_x, origin_y])

        # Rotation matrix for -yaw (to align global to local)
        c, s = np.cos(-origin_yaw), np.sin(-origin_yaw)
        R = np.array([[c, -s],
                      [s,  c]])

        # Rotate translated points
        local_points = translated @ R.T
        return local_points

    def to_global_frame(self, points_local, origin_x, origin_y, origin_yaw):
        """
        Convert local (x, y) points to global frame.

        Parameters:
            points_local : np.ndarray of shape (N, 2)
                Local coordinates of the points.
            origin_x : float
                X coordinate of the local frame origin in global frame.
            origin_y : float
                Y coordinate of the local frame origin in global frame.
            origin_yaw : float (radians)
                Yaw of the local frame relative to the global frame.

        Returns:
            np.ndarray of shape (N, 2) with global coordinates.
        """
        # Rotation matrix for +yaw
        c, s = np.cos(origin_yaw), np.sin(origin_yaw)
        R = np.array([[c, -s],
                      [s,  c]])

        # Rotate then translate
        rotated = points_local @ R.T
        global_points = rotated + np.array([origin_x, origin_y])
        return global_points

    def to_global_frame_torch(self, points_local, origin):
        """
        Convert local (x,y) points to global frame (batched, differentiable).

        Args:
            points_local : (B, N, 2) tensor
            origin_x : (B,) tensor
            origin_y : (B,) tensor
            origin_yaw : (B,) tensor (radians)

        Returns:
            global_points : (B, N, 2) tensor
        """
        B, N, _ = points_local.shape

        c = torch.cos(origin[:,2])  # (B,)
        s = torch.sin(origin[:,2])  # (B,)

        # Build rotation matrices (B, 2, 2)
        R = torch.stack([torch.stack([c, -s], dim=-1),
                         torch.stack([s,  c], dim=-1)], dim=-2)

        # Rotate: (B, N, 2) @ (B, 2, 2)^T → (B, N, 2)
        rotated = torch.bmm(points_local, R.transpose(1, 2))

        # Translate: broadcast (B, N, 2) + (B, 1, 2)
        global_points = rotated + origin[:,:2].unsqueeze(1)

        return global_points


    def __len__(self):
        return len(self.indices)

    def transform_img(self, img):
        return self.transform(img)

    def load_feat_image(self, idx_str):
        dino_path = os.path.join(self.dino_dir, f"{idx_str}_data.npy")
        if not os.path.exists(dino_path):
            dino_path = dino_path.replace("_data", "")
        feat_img = np.load(dino_path)

        feat_img = torch.from_numpy(feat_img)
        if len(feat_img.shape) == 2:
            feat_img = feat_img.unsqueeze(0)

        feat_img[:,self.img_mask] = 0

        return feat_img

    def __getitem__(self, idx):
        index = self.indices[idx]
        idx_str = f"{index:08d}"

        img_path = os.path.join(self.image_dir, f"{idx_str}.png")
        # print(img_path)
        # img = cv2.imread(img_path)
        og_img = Image.open(img_path).convert("RGB")
        img = self.transform_img(og_img)
        og_img = np.array(og_img)
        # print(og_img.shape)

        # feat_img = self.load_feat_image(idx_str)
        feat_img = img

        img_plot= img.permute(1,2,0).cpu().numpy()

        cur_odom = self.gps[index]
        rpy = R.from_quat(cur_odom[3:7]).as_euler('xyz')
        yaw = rpy[2] + np.pi/2
        roll, pitch = rpy[:2]
        # cur_x, cur_y = cur_odom[:2]
        cur_x, cur_y = -cur_odom[1], cur_odom[0]

        cur_row, cur_col = self.tif.index(cur_x, cur_y)
        start = (cur_row, cur_col)
        thresh = .55
        headings, costs = heading_cost_field(self.costmap, start, thresh, K=256, max_dist=self.max_dist)

        # costs /= self.max_dist

        # print(headings)

        # print(yaw)
        # headings -= (yaw - np.pi/2)
        headings = -(yaw + headings)
        headings = torch.from_numpy(headings)

       
        forward_mask = (torch.cos(headings) >= 0)  # same as abs(heading) <= π/2
        forward_costs = costs[forward_mask]
        forward_headings = headings[forward_mask]
        forward_costs = torch.from_numpy(forward_costs).float()
        # print(forward_headings)

        num_bins = 128
        bins = torch.linspace(-torch.pi/2, torch.pi/2, num_bins+1)

        # Wrap all headings into [-pi, pi]
        wrapped = (forward_headings + torch.pi) % (2*torch.pi) - torch.pi

        binned_headings = 0.5 * (bins[:-1] + bins[1:])

        do_augment = np.random.choice([True, False])
        do_augment = do_augment and self.augment
        if do_augment:
            feat_img = hflip(feat_img)
            wrapped *= -1.
            binned_headings *= -1
            og_img = np.fliplr(og_img).copy()
            

        bin_indices = torch.bucketize(wrapped, bins) - 1
        bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)

        # Initialize cost vector
        binned_costs = torch.full((num_bins,), float('nan'))

        # Fill the bins
        binned_costs[bin_indices] = forward_costs
        


        if torch.isnan(binned_costs).any():
            print("SOMETHING IS WRONG")
            s=r

        return {
            'img': feat_img,
            # 'feat_img': feat_img,
            'img_raw': og_img,
            'origin': torch.Tensor([cur_x, cur_y, yaw]).float(),
            'origin_idx': torch.Tensor(start).float(),
            'costs_all': torch.Tensor(costs).float(),
            'headings': torch.Tensor(headings).float(),
            'costs': torch.Tensor(binned_costs).float(),
            'headings_binned': torch.Tensor(binned_headings).float(),
            'flipped': do_augment
        }



# trainset = GridTrajectoryDataset('/home/tartandriver/workspace/datasets/lrdp/20251009_4_fig8_to_horseshoe')

# # trainset[1]
# for i in range(0,len(trainset),5):
#     trainset[i]

#decreased weight decay
#mess with augment?
#make sure cost of ray at hit is not accounted for
#check score scaling
#need to add traversability back into computation
#is learning through that summing operation valid? or should we learn that also?

def main():

    batch_size = 8
    n_epochs = 60
    kl_weight = .5

    LR = .001
    weight_decay = .1

    AUGMENT = True

    mse_loss_fn = MSELoss(reduction='mean')

    K = np.array([455.7750, 0., 497.1180, 0., 456.3191, 251.8580, 0., 0., 1.]).reshape(3,3)

    # plot_cost_map_and_heading(occ, start, thresh, headings, costs)
    trainset6 = FrontierDataset('/home/tartandriver/workspace/datasets/lrdp/04_garage_to_turnpike_afternoon', augment=AUGMENT)
    trainset7 = FrontierDataset('/home/tartandriver/workspace/datasets/lrdp/2025-10-16-18-00-49_snow_bag', augment=AUGMENT)
    trainset = FrontierDataset('/home/tartandriver/workspace/datasets/lrdp/2023-11-14-15-02-21_figure_8', augment=AUGMENT)
    trainset2= FrontierDataset('/home/tartandriver/workspace/datasets/lrdp/turnpike_2023-09-12-12-53-32', augment=AUGMENT)
    trainset3 = FrontierDataset('/home/tartandriver/workspace/datasets/lrdp/20251009_3_red_course', augment=AUGMENT)
    trainset4 = FrontierDataset('/home/tartandriver/workspace/datasets/lrdp/20251009_4_fig8_to_horseshoe', augment=AUGMENT)
    trainset5 = FrontierDataset('/home/tartandriver/workspace/datasets/lrdp/20251009_5_turnpike', augment=AUGMENT)
   
    # # trainset_all = ConcatDataset([trainset, trainset2, trainset3])
    datasets = [trainset, trainset2, trainset3, trainset4, trainset5, trainset6, trainset7]
    # datasets = [trainset]
    trainset_all = ConcatDataset(datasets)
    # trainset_all = trainset
    print("------------", len(trainset_all))

    # model = DinoHeadingCostHead(in_channels=1152, num_bins=128, directional=True)
    # model = DinoHeadingCostHead(in_channels=96, num_bins=128, directional=True)
    # model = DinoHeadingCostHeadV2(in_channels=1152, num_bins=128)
    model = DinoHeadingCostHeadV2_highres(in_channels=384, num_bins=128, upsample_scale=1)


    model.train()
    model.cuda()

    model.register_headings(K, 544, 1024)

    params = list(model.parameters())
    trainable_params = sum(p.numel() for p in params if p.requires_grad)
    print("TRAINABLE PARAMS - ", trainable_params)

    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay = weight_decay)

    train_loader = torch.utils.data.DataLoader(
        trainset_all, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=5
    )

    from physics_atv_visual_mapping.image_processing.image_pipeline import setup_image_pipeline

    config = {"models_dir": '/home/tartandriver/tartandriver_ws/models',

        "image_processing": [{

            "type": 'jafar',

            "args": {'image_insize': [224, 224]}},

            ],

        "device": 'cuda'

    }

    # -
#     type: loftup
#     args:
#         image_insize: [224, 224]
# - 
#     type: vlad
#     args:
#         n_clusters: 8
#         cache_dir: physics_atv_visual_mapping/dino_clusters/8_clusters_loftup_dinov2s_224x224

    config = {"models_dir": '/home/tartandriver/tartandriver_ws/models',

        "image_processing": [{

            "type": 'loftup',

            "args": {'image_insize': [224, 224]}},
            # {
            #     "type": 'vlad',
            #     "args": {'n_clusters': 8, 'cache_dir': 'physics_atv_visual_mapping/dino_clusters/8_clusters_loftup_dinov2s_224x224'}
        
            # }

            ],

        "device": 'cuda'

    }


    image_pipeline = setup_image_pipeline(config)

    for epoch in range(n_epochs):
        running_emd = 0
        running_kl = 0 
        running_loss = 0
        count = 0
        scount = 0
        for i, data in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()

            img = data['img'].cuda() #Bx1x100x100
            img_plot= data['img_raw'][0].cpu().numpy()

            start = data['origin_idx']

            headings = data['headings_binned'].cuda()
            costs = data['costs'].cuda()

            costs = costs[:,model.valid_idxs]

            # print(costs.min(), costs.max())

            gt_probs = torch.softmax(costs, dim=1)

            image_intrinsics = torch.eye(4).unsqueeze(0)

            with torch.no_grad():

                # feat_img = dino(og_img).cpu().numpy()[0]

                img, feature_intrinsics = image_pipeline.run(

                    img, image_intrinsics

                )

                img = F.normalize(img)
            

            # print(costs.mean(), costs.max())
            heatmap, pred_costs = model(img)
            pred_costs = pred_costs[:,model.valid_idxs]
            pred_probs = torch.softmax(pred_costs, dim=1)

            # print(pred_probs.shape, gt_probs.shape)
            # print(costs.max(), pred_costs.max())

            emd_loss = emd_loss_fn(pred_probs, gt_probs)

            kl_loss = F.kl_div(pred_probs.log(), gt_probs, reduction='batchmean')

            

            # entropy = -(gt_probs * gt_probs.log()).sum(dim=1)
            # lambda_adaptive = (1.0 - entropy / np.log(gt_probs.size(1))).mean()

            # loss = emd_loss + kl_weight*kl_loss
            # loss = emd_loss
            # loss = kl_loss
            loss = mse_loss_fn(pred_costs, costs)
            # loss = emd_loss

            running_emd += emd_loss.item()
            running_kl += kl_loss.item()
            running_loss += loss.item()
            count += 1

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(params, max_norm=10.0) # Clip gradients by norm
            optimizer.step()

            # plt.plot(gt_probs[0].detach().cpu().numpy())
            # plt.plot(pred_probs[0].detach().cpu().numpy())
            # plt.show()

            # if i % 100 == 0:
            #     sname = "debug/" + str(epoch) + "_" + str(count)
            #     plot_cost_map_and_heading_image(trainset.costmap, 
            #                                 start[0].cpu().numpy(),
            #                                   .6, 
            #                                   headings[0].detach().cpu().numpy(), 
            #                                   pred_costs[0].detach().cpu().numpy(), 
            #                                   img_plot,
            #                                   sname=sname)
            # print(loss.item())
        # lr_scheduler.step()
        avg_loss = running_loss/count
        avg_kl = running_kl/count
        avg_emd = running_emd/count

        print(f"epoch: {epoch}, steps: {i}, loss: {avg_loss:.4}, emd: {avg_emd:.4}, kl: {avg_kl:.4}")


        state_dict = { 'head': model.state_dict()}
        torch.save(state_dict,'model.pt')

        for ds in datasets:
                ds.augment = False
        test_loader = torch.utils.data.DataLoader(
            trainset_all, batch_size=1, shuffle=True)

        model.eval()
        with torch.no_grad():
            count = 0
            for i, data in enumerate(test_loader):
                if i > 10:
                    break

                img = data['img'].cuda() #Bx1x100x100
                img_plot= data['img_raw'][0].cpu().numpy()

                start = data['origin_idx']

                headings = data['headings_binned'].cuda()
                costs = data['costs'].cuda()

                gt_probs = torch.softmax(costs, dim=1)

                image_intrinsics = torch.eye(4).unsqueeze(0)

                with torch.no_grad():

                    # feat_img = dino(og_img).cpu().numpy()[0]

                    img, feature_intrinsics = image_pipeline.run(

                        img, image_intrinsics

                    )

                    img = F.normalize(img)

                # print(costs.mean(), costs.max())
                heatmap, pred_costs = model(img)
                pred_probs = torch.softmax(pred_costs, dim=1)

                sname = "debug/" + str(epoch) + "_" + str(count)
                # plot_cost_map_and_heading_image(trainset.costmap, 
                #                             start[0].cpu().numpy(),
                #                               .6, 
                #                               headings[0].detach().cpu().numpy(), 
                #                               pred_costs[0].detach().cpu().numpy(), 
                #                               img_plot,
                #                               sname=sname)
                plot_heatmap_and_heading_image(heatmap[0].cpu().numpy(), 
                                            start[0].cpu().numpy(),
                                              .55, 
                                              headings[0].detach().cpu().numpy(), 
                                              pred_costs[0].detach().cpu().numpy(), 
                                              img_plot,
                                              sname=sname)
                count += 1

        model.train()
        # fpv_goal_attention.train()
        for ds in datasets:
            ds.augment = AUGMENT



if __name__ == "__main__":
    main()
