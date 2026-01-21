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

import torch.nn as nn
import torch.nn.functional as F


CMAP = cm.magma
CMAP_JET = cm.jet


def overlay_heatmap_on_image(
    image,
    heatmap,
    max_val=3.0,
    threshold=0.01,
    alpha=0.6,
):
    H, W = image.shape[:2]

    # Resize heatmap
    heatmap = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_NEAREST)

    # Normalize to [0, 255]
    hm = np.clip(heatmap / max_val, 0, 1)
    hm_u8 = (hm * 255).astype(np.uint8)

    # Apply OpenCV colormap (BGR!)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)

    # Threshold mask
    mask = hm > threshold

    # Ensure image uint8
    if image.dtype != np.uint8:
        img_u8 = np.clip(image * 255, 0, 255).astype(np.uint8)
    else:
        img_u8 = image

    # Alpha blend only where mask is true
    overlay = img_u8.copy()
    overlay[mask] = (
        (1 - alpha) * img_u8[mask] +
        alpha * hm_color[mask]
    ).astype(np.uint8)

    return overlay
# def overlay_heatmap_on_image(image, heatmap, max_val = 3., threshold=0.01, alpha=0.6, cmap='jet'):

#     H, W = image.shape[:2]
#     heatmap = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)

#     # Normalize heatmap to [0, 1]
#     # hm_norm = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
#     hm_norm = np.clip(heatmap/max_val, 0, 1.)
#     # Apply matplotlib colormap (returns RGBA)
#     # cmap_fn = cm.get_cmap(cmap)
#     hm_color = CMAP_JET(hm_norm)[..., :3]  # drop alpha channel

#     # Create binary mask where heatmap exceeds threshold
#     mask = (hm_norm > threshold)[..., None].astype(float)

#     # Convert image to float in [0,1] if needed
#     if image.dtype == np.uint8:
#         img_float = image.astype(np.float32) / 255.0
#     else:
#         img_float = image.copy()

#     # Blend where mask is active
#     overlay = img_float * (1 - alpha * mask) + hm_color * (alpha * mask)

#     # Convert back to uint8
#     overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
#     return overlay

def compute_pixel_headings(K, H, W, H_orig=None, W_orig=None, device='cpu'):
    """
    Compute per-pixel horizontal headings for a camera image of arbitrary size.
    Works for both odd and even image dimensions.

    Args:
        K: camera intrinsic matrix [3,3] (numpy or torch) for original resolution
        H, W: output image size
        H_orig, W_orig: original image size K was calibrated for (if None, assume H_orig=H, W_orig=W)
        device: 'cpu' or 'cuda'

    Returns:
        pixel_headings: [H, W] tensor with heading angles in radians (-pi/2 to pi/2)
    """
    if isinstance(K, np.ndarray):
        K = torch.from_numpy(K).float().to(device)
    else:
        K = K.float().to(device)

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

    # Create centered floating-point pixel grid
    # Coordinates of pixel centers range from 0 to W-1 and 0 to H-1
    xs = torch.linspace(0, W-1, W, device=device)
    ys = torch.linspace(0, H-1, H, device=device)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')  # [H, W]

    # Compute horizontal heading relative to optical center
    pixel_headings = torch.atan2(grid_x - cx, fx)

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


    
class DinoHeadingCostHeadV2(nn.Module):
    def __init__(self, in_channels=1152, hidden_dim=128, num_bins=128, upsample_scale=8):
        super().__init__()
        self.num_bins = num_bins
        self.upsample_scale = upsample_scale
        self.in_channels = in_channels

        # Decide number of doubling layers to reach upsample_scale
        num_upsample_layers = int(np.ceil(np.log2(upsample_scale)))
        current_scale = 1

        print("NUM UPSAMPLE ", num_upsample_layers)

        # layers = [nn.Conv2d(in_channels, hidden_dim, kernel_size=1), nn.ReLU()]
        # for _ in range(num_upsample_layers):
        #     layers += [
        #         nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #         nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        #         nn.ReLU()
        #     ]
        #     current_scale *= 2

        # # If current_scale overshoots upsample_scale, use final Conv2d to reduce spatial size
        # final_kernel = 1
        # layers.append(nn.Conv2d(hidden_dim, 1, kernel_size=final_kernel))

        # self.decoder = nn.Sequential(*layers)

        layers = []
        layers += [
            nn.Conv2d(in_channels, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        ]

        for k, out_ch in zip([7, 5, 3], [256, 128, 64]):
            layers += [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(512 if out_ch == 256 else prev_ch,
                        out_ch,
                        kernel_size=k,
                        padding=k // 2,
                        bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            prev_ch = out_ch

        layers.append(nn.Conv2d(prev_ch, 1, kernel_size=1))
        self.decoder = nn.Sequential(*layers)

        # layers = [
        #     nn.Conv2d(in_channels, 256, kernel_size=1),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #     nn.Conv2d(256, 1, kernel_size=7)
        # ]

        # self.decoder = nn.Sequential(*layers)

    def register_headings(self, K, H_orig, W_orig):
        dummy_input = torch.zeros(1, self.in_channels,28, 53).cuda()
        with torch.no_grad():
            heatmap = self.decoder(dummy_input)
        H, W = heatmap.shape[2], heatmap.shape[3]

        ys = torch.linspace(-(H-1)/2, (H-1)/2, H)
        xs = torch.linspace(-(W-1)/2, (W-1)/2, W)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # [H_up, W_up]

        # pixel_headings = compute_pixel_headings(K, grid_y, grid_x, H_orig=H_orig, W_orig=W_orig)
        pixel_headings = compute_pixel_headings(K, H, W, H_orig=H_orig, W_orig=W_orig)
        #TODO is this ok to do
        pixel_headings *= -1
        
        # import pdb;pdb.set_trace()

        bin_edges = torch.linspace(-np.pi/2, np.pi/2, self.num_bins + 1)
        bin_indices = torch.bucketize(pixel_headings, bin_edges) - 1
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)

        bin_indices_flat = bin_indices.flatten().cuda()
        self.register_buffer('bin_indices', bin_indices_flat)
        pixel_counts = torch.bincount(bin_indices_flat, minlength=self.num_bins).float()
        self.register_buffer('pixel_counts', pixel_counts)

        self.valid_idxs = torch.where(pixel_counts != 0)[0]


        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        heading_flat = pixel_headings.flatten()[..., None]  # [H*W, 1]
        prior_mask = torch.exp(-0.5 * ((heading_flat - bin_centers[None, :]) / 0.05)**2)
        prior_mask = prior_mask / (prior_mask.sum(dim=1, keepdim=True) + 1e-6)  # normalize across bins
        prior_mask = prior_mask.T.view(self.num_bins, H, W)  # [num_bins, H, W]
        prior_mask = prior_mask.cuda()

        # import pdb;pdb.set_trace()

        self.register_buffer('prior_mask', prior_mask)

    def forward(self, x, tau=2.):
        """
        Args:
            x: [B, C, H, W] input features
            tau: temperature for softmax-like aggregation
        Returns:
            heatmap: [B, H, W] per-pixel scores
            alpha: [B, num_bins] Dirichlet parameters
            probs: [B, num_bins] expected probability per heading
        """
        B, _, H_small, W_small = x.shape
        device = x.device

        # --- Per-pixel heatmap ---
        heatmap = self.decoder(x).squeeze(1)  # [B, H, W]
        heatmap = F.elu(heatmap)  # preserves some negative values; could use ReLU if preferred

        # --- Flatten pixels ---
        heatmap_flat = heatmap.flatten(1)  # [B, H*W]
        B, N = heatmap_flat.shape

        # --- Compute softmax-style per-heading evidence ---
        z_scaled = tau * heatmap_flat  # [B, N]
        # z_scaled = torch.clamp(tau * heatmap_flat, max=20.0)
        exp_z = torch.exp(z_scaled)

        # print(heatmap_flat.max(), z_scaled.max(), exp_z.max())

        # binned sum of exponentiated logits
        evidence = torch.zeros(B, self.num_bins, device=device)
        evidence.scatter_add_(1, self.bin_indices.unsqueeze(0).expand(B, -1), exp_z)

        # prior_flat = self.prior_mask.view(self.num_bins, -1)
        # evidence = exp_z @ prior_flat.T

        # --- logsumexp approximation per heading ---
        e_h = torch.log(evidence + 1e-8)  # [B, num_bins]

        # --- Dirichlet parameters ---
        alpha = 1.0 + F.softplus(e_h)  # [B, num_bins], alpha > 1 ensures minimal prior

        # --- Expected probabilities per heading ---
        probs = alpha / alpha.sum(dim=1, keepdim=True)  # E[pi_h]

        return heatmap, alpha

