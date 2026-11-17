"""
gradcam3d.py - 3D Grad-CAM Activation Mapping, Axial Slice Projection, and Base64 Encoding.

Features:
- Hooks gradients and feature maps from the final 3D convolutional layer.
- Computes channel weights via global average pooling of gradients.
- Interpolates 3D CAM to original volume dimensions.
- Max-projects or slices along axial z-axis, applies colormap, and blends with CT slice.
- Encodes output as base64 PNG for direct web display without disk writes.
"""

import os
import io
import base64
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Headless rendering: never try to open a GUI window from the API process
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.cm as cm


def _resolve_colormap(colormap_name: str):
    """Returns a colormap across Matplotlib versions (cm.get_cmap was removed in 3.9)."""
    try:
        return matplotlib.colormaps[colormap_name]
    except (AttributeError, KeyError):
        pass
    get_cmap = getattr(cm, "get_cmap", None)
    if get_cmap is not None:
        return get_cmap(colormap_name)
    return matplotlib.colormaps["jet"]


class GradCAM3D:
    """
    3D Gradient-weighted Class Activation Mapping (Grad-CAM).
    """

    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or self._find_last_conv_layer(model)
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        if self.target_layer is not None:
            self._register_hooks()

    def _find_last_conv_layer(self, model: torch.nn.Module) -> Optional[torch.nn.Module]:
        """
        Find the last Conv3d layer in the model.

        TorchScript modules expose their graph rather than ``modules()``, so the
        layer cannot be located by inspection. In that case the CAM falls back to
        the attenuation/gaussian heatmap instead of raising.
        """
        last_conv = None
        try:
            for module in model.modules():
                if isinstance(module, torch.nn.Conv3d):
                    last_conv = module
        except Exception:
            return None
        return last_conv

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_score: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Generates 3D CAM volume of shape (D, H, W).
        """
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        if input_tensor.grad is not None:
            input_tensor.grad.zero_()

        # Gradients are required even when the caller runs inside torch.no_grad()
        with torch.enable_grad():
            work_input = input_tensor.detach().clone().requires_grad_(True)
            det_logits, _grade_logits, _size_pred = self.model(work_input)

            if target_score is None:
                target_score = det_logits.sum()
            elif not target_score.requires_grad:
                target_score = det_logits.sum()

            target_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback to Gaussian centered heatmap
            d, h, w = input_tensor.shape[2:]
            return generate_synthetic_cam_3d((d, h, w))

        # Channel-wise global average pooling of gradients: [B, C, 1, 1, 1]
        weights = torch.mean(self.gradients, dim=(2, 3, 4), keepdim=True)
        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample to input resolution
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="trilinear",
            align_corners=False,
        )

        cam_np = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np


def generate_synthetic_cam_3d(shape: Tuple[int, int, int] = (64, 64, 64)) -> np.ndarray:
    """Generates a realistic spherical 3D Gaussian heat response for mock/offline demo."""
    d, h, w = shape
    z, y, x = np.ogrid[:d, :h, :w]
    cz, cy, cx = d // 2, h // 2, w // 2
    sigma = min(d, h, w) / 6.0
    dist_sq = (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2
    cam = np.exp(-dist_sq / (2.0 * (sigma ** 2)))
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def blend_axial_slice_gradcam(
    ct_volume_slice: np.ndarray,
    cam_slice: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = "jet",
) -> str:
    """
    Blends a 2D axial CT slice with a 2D Grad-CAM heatmap and returns a base64 PNG data URL.

    Args:
        ct_volume_slice: 2D array (H, W) normalized to [0, 1] or raw HU.
        cam_slice: 2D array (H, W) normalized to [0, 1].
        alpha: Blend weight for heatmap over CT slice.
        colormap_name: Matplotlib colormap name.

    Returns:
        Base64-encoded PNG string ("data:image/png;base64,...").
    """
    # Normalize CT slice to uint8 grayscale
    ct_norm = np.clip(ct_volume_slice, 0.0, 1.0)
    ct_uint8 = (ct_norm * 255.0).astype(np.uint8)
    ct_rgb = np.stack([ct_uint8] * 3, axis=-1).astype(np.float32)

    # Colormap heatmap
    cam_slice = np.clip(np.nan_to_num(cam_slice.astype(np.float32)), 0.0, 1.0)
    cmap = _resolve_colormap(colormap_name)
    cam_colored = cmap(cam_slice)[:, :, :3]  # Drop alpha, [H, W, 3] in [0, 1]
    cam_rgb = (cam_colored * 255.0).astype(np.float32)

    # Mask low activation areas so pure CT shows through
    cam_mask = np.clip((cam_slice - 0.15) / 0.85, 0.0, 1.0)[:, :, np.newaxis]

    # Alpha blend
    blended = (1.0 - alpha * cam_mask) * ct_rgb + (alpha * cam_mask) * cam_rgb
    blended = np.clip(blended, 0.0, 255.0).astype(np.uint8)

    # Encode to PNG in memory
    img = Image.fromarray(blended)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def generate_mock_ct_gradcam_slice(
    diameter_mm: float = 12.0,
    has_nodule: bool = True,
) -> str:
    """
    Generates an anatomically plausible synthetic axial CT lung slice with Grad-CAM overlay.
    Used for instant demo and mock predictor without needing huge training weights.
    """
    h, w = 256, 256
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2

    # Thorax boundary
    dist_chest = ((y - cy) / (0.42 * h)) ** 2 + ((x - cx) / (0.45 * w)) ** 2
    chest_mask = dist_chest <= 1.0

    # Lungs (left and right ellipsoids)
    left_lung_dist = ((y - cy) / (0.32 * h)) ** 2 + ((x - (cx - 0.22 * w)) / (0.18 * w)) ** 2
    right_lung_dist = ((y - cy) / (0.32 * h)) ** 2 + ((x - (cx + 0.22 * w)) / (0.18 * w)) ** 2
    lungs_mask = (left_lung_dist <= 1.0) | (right_lung_dist <= 1.0)

    # Mediastinum spine & heart
    spine_dist = ((y - (cy + 0.3 * h)) / (0.08 * h)) ** 2 + ((x - cx) / (0.06 * w)) ** 2
    spine_mask = spine_dist <= 1.0

    # Background tissue intensities
    ct_slice = np.full((h, w), fill_value=0.05, dtype=np.float32)  # Ambient air
    ct_slice[chest_mask] = 0.55  # Chest wall / soft tissue
    ct_slice[lungs_mask] = 0.15  # Low-density lung parenchyma
    ct_slice[spine_mask] = 0.95  # Dense bone

    # Nodule placement
    cam_slice = np.zeros((h, w), dtype=np.float32)

    if has_nodule and diameter_mm > 0.0:
        nodule_cy = int(cy - 0.08 * h)
        nodule_cx = int(cx + 0.20 * w)
        radius_voxels = max(3.0, diameter_mm * 1.2)

        nod_dist = np.sqrt((y - nodule_cy) ** 2 + (x - nodule_cx) ** 2)
        nodule_mask = nod_dist <= radius_voxels
        ct_slice[nodule_mask] = 0.70  # Soft tissue nodule attenuation

        # Grad-CAM Gaussian hot spot centered on the nodule
        sigma = radius_voxels * 2.2
        cam_slice = np.exp(-(nod_dist ** 2) / (2.0 * (sigma ** 2)))
        cam_slice = (cam_slice - cam_slice.min()) / (cam_slice.max() - cam_slice.min() + 1e-8)

    return blend_axial_slice_gradcam(ct_slice, cam_slice)
