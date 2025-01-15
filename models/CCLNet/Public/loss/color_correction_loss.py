# File: models/CCLNet/loss/color_correction_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class ColorCorrectionLoss(nn.Module):
    """
    A simple color correction loss that penalizes the L1 difference
    between predicted and reference images in LAB color space.
    
    Steps:
      1) Convert pred, ref from [-1,1] -> [0,1].
      2) Convert from RGB -> LAB using OpenCV.
      3) Compute L1 difference in LAB space.
      4) Multiply by loss_weight and return.
    """
    def __init__(self, loss_weight=1.0):
        super(ColorCorrectionLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, ref):
        """
        pred, ref: (B, C, H, W), typically in [-1,1] from Tanh output.
        
        Returns: scalar tensor on pred.device
        """
        # 1) Shift to [0,1]
        pred_01 = (pred + 1.0) * 0.5
        ref_01  = (ref  + 1.0) * 0.5

        # 2) Clamp to [0,1] just in case
        pred_01 = torch.clamp(pred_01, 0.0, 1.0)
        ref_01  = torch.clamp(ref_01,  0.0, 1.0)
        # Print ranges after normalization
        # print(f"range of pred_01: {pred_01.min()}, {pred_01.max()}")
        # print(f"range of ref_01: {ref_01.min()}, {ref_01.max()}")

        # We'll do color space conversion sample by sample on CPU
        # For each sample in the batch, convert (C,H,W)->(H,W,C) -> OpenCV LAB -> L1 difference
        B = pred_01.shape[0]
        total_lab_loss = 0.0

        for i in range(B):
            # Move to CPU for OpenCV
            pred_np = pred_01[i].permute(1,2,0).detach().cpu().numpy()  # shape [H,W,C], in [0,1]
            ref_np  = ref_01[i].permute(1,2,0).detach().cpu().numpy()

            # Convert float [0,1]->[0,255]
            pred_np_255 = (pred_np * 255.0).astype(np.uint8)
            ref_np_255  = (ref_np  * 255.0).astype(np.uint8)

            # Convert from RGB to LAB
            pred_lab = cv2.cvtColor(pred_np_255, cv2.COLOR_RGB2Lab)
            ref_lab  = cv2.cvtColor(ref_np_255,  cv2.COLOR_RGB2Lab)

            # Convert to float for difference
            pred_lab_f = pred_lab.astype(np.float32)
            ref_lab_f  = ref_lab.astype(np.float32)

            # L1 difference
            diff = np.abs(pred_lab_f - ref_lab_f)
            mean_diff = diff.mean()  # scalar

            total_lab_loss += mean_diff
        
        # Average over batch
        if B > 0:
            total_lab_loss = total_lab_loss / float(B)

        # Multiply by self.loss_weight
        # Return as a tensor on same device as pred
        return self.loss_weight * torch.tensor(total_lab_loss, requires_grad=True, device=pred.device)
