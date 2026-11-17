"""
losses.py - Multi-Task Loss Functions for 3D Lung Nodule Detection and Grading.

Includes:
- CORAL (Consistent Rank Logits) Ordinal Loss for cumulative-link malignancy grading.
- Smooth L1 Loss for nodule diameter regression.
- Weighted BCE With Logits Loss for 3D nodule detection.
- Unified MultiTaskLoss module.
"""

from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def ordinal_label_to_binary(labels: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """
    Converts 1-indexed ordinal labels (e.g. 1..5) into (num_classes - 1) binary targets.

    For label y in {1..num_classes}:
        binary_targets[k] = 1 if y > (k + 1) else 0, for k in 0..(num_classes - 2).

    Args:
        labels: 1D Tensor of shape [B] containing integer ranks from 1 to num_classes.
        num_classes: Total number of ordinal levels (default: 5).

    Returns:
        Tensor of shape [B, num_classes - 1] with binary float targets.
    """
    batch_size = labels.shape[0]
    num_cutoffs = num_classes - 1
    # Expand labels: [B, 1]
    y = labels.view(-1, 1).float()
    # Threshold cutoffs: [1, num_cutoffs] -> e.g. [1, 2, 3, 4]
    cutoffs = torch.arange(1, num_classes, device=labels.device).view(1, -1).float()
    binary_targets = (y > cutoffs).float()
    return binary_targets


def binary_to_ordinal_prediction(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decodes CORAL logits into predicted ordinal rank (1..num_classes) and class probabilities.

    Args:
        logits: Tensor of shape [B, num_classes - 1] (or [B, num_classes] where first K-1 are used).

    Returns:
        predicted_ranks: Tensor of shape [B] with integer ranks {1..num_classes}.
        class_probs: Tensor of shape [B, num_classes] with normalized probability distribution.
    """
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)

    # Use first K-1 cutoffs if K logits provided
    cutoffs = logits[:, :4] if logits.shape[1] >= 4 else logits
    num_cutoffs = cutoffs.shape[1]
    num_classes = num_cutoffs + 1

    # Cumulative probabilities P(Y > k)
    cum_probs = torch.sigmoid(cutoffs)

    # Predict rank: 1 + number of cutoffs where prob > 0.5 (or logit > 0)
    predicted_ranks = 1 + torch.sum((cum_probs > 0.5).long(), dim=1)

    # Convert cumulative probabilities to individual class probabilities P(Y = k)
    batch_size = logits.shape[0]
    probs = torch.zeros((batch_size, num_classes), device=logits.device, dtype=logits.dtype)

    # P(Y = 1) = 1 - P(Y > 1)
    probs[:, 0] = 1.0 - cum_probs[:, 0]

    # P(Y = k) = P(Y > k-1) - P(Y > k) for k in 2..(K-1)
    for k in range(1, num_cutoffs):
        probs[:, k] = torch.clamp(cum_probs[:, k - 1] - cum_probs[:, k], min=0.0)

    # P(Y = K) = P(Y > K-1)
    probs[:, num_classes - 1] = cum_probs[:, num_cutoffs - 1]

    # Renormalize to ensure sum to 1.0
    probs_sum = torch.sum(probs, dim=1, keepdim=True).clamp(min=1e-8)
    probs = probs / probs_sum

    return predicted_ranks, probs


class CoralOrdinalLoss(nn.Module):
    """
    Consistent Rank Logits (CORAL) cumulative-link ordinal cross-entropy loss.
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Shape [B, num_classes - 1] or [B, num_classes].
            targets: Shape [B], integer ranks in 1..num_classes.

        Returns:
            Scalar loss tensor.
        """
        if logits.numel() == 0 or targets.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        num_cutoffs = self.num_classes - 1
        relevant_logits = logits[:, :num_cutoffs]
        binary_targets = ordinal_label_to_binary(targets, num_classes=self.num_classes)

        loss = F.binary_cross_entropy_with_logits(
            relevant_logits, binary_targets, reduction="mean"
        )
        return loss


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining:
      1. BCEWithLogitsLoss for detection (with positive class weighting).
      2. CORAL ordinal loss for malignancy grading (applied to positive nodules).
      3. Smooth L1 loss for diameter regression (applied to positive nodules).
    """

    def __init__(
        self,
        num_classes: int = 5,
        detection_weight: float = 1.0,
        grading_weight: float = 1.0,
        size_weight: float = 0.5,
        pos_weight: Optional[float] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.det_weight = detection_weight
        self.grade_weight = grading_weight
        self.size_weight = size_weight
        self.coral_loss = CoralOrdinalLoss(num_classes=num_classes)
        self.pos_weight = torch.tensor([pos_weight]) if pos_weight is not None else None

    def forward(
        self,
        det_logits: torch.Tensor,
        grade_logits: torch.Tensor,
        size_pred: torch.Tensor,
        det_targets: torch.Tensor,
        grade_targets: torch.Tensor,
        size_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes total loss and sub-task loss metrics.
        """
        device = det_logits.device
        pw = self.pos_weight.to(device) if self.pos_weight is not None else None

        # 1. Detection loss
        det_flat = det_logits.view(-1)
        det_targets_flat = det_targets.view(-1).float()
        loss_det = F.binary_cross_entropy_with_logits(
            det_flat, det_targets_flat, pos_weight=pw
        )

        # Mask for true positive candidates
        pos_mask = (det_targets_flat > 0.5)

        # 2. Grading loss on positives
        if pos_mask.sum() > 0:
            loss_grade = self.coral_loss(grade_logits[pos_mask], grade_targets.view(-1)[pos_mask])
            loss_size = F.smooth_l1_loss(
                size_pred.view(-1)[pos_mask], size_targets.view(-1)[pos_mask].float()
            )
        else:
            loss_grade = torch.tensor(0.0, device=device, requires_grad=True)
            loss_size = torch.tensor(0.0, device=device, requires_grad=True)

        total_loss = (
            self.det_weight * loss_det
            + self.grade_weight * loss_grade
            + self.size_weight * loss_size
        )

        metrics = {
            "loss_total": float(total_loss.detach().item()),
            "loss_det": float(loss_det.detach().item()),
            "loss_grade": float(loss_grade.detach().item()),
            "loss_size": float(loss_size.detach().item()),
        }

        return total_loss, metrics
