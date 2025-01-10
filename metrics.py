# File: metrics.py

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
import math
import numpy as np
import cv2

# ====================================================
# Utility: Range Checking
# ====================================================

def _check_and_clip_range(img, min_val=0.0, max_val=1.0, clip=True):
    """
    Ensures 'img' is within [min_val, max_val].
    If clip=True, we clamp any out-of-range values.
    Otherwise, we raise an Exception if out-of-range is detected.
    """
    img_min = img.min().item()
    img_max = img.max().item()
    if img_min < min_val - 1e-8 or img_max > max_val + 1e-8:
        if clip:
            img = torch.clamp(img, min_val, max_val)
        else:
            raise ValueError(f"Image values out of [{min_val}, {max_val}] range: min={img_min}, max={img_max}")
    return img


# ====================================================
# Standard Image Quality Metrics
# ====================================================

def compute_psnr_batch(pred, gt, data_range=1.0, reduction='none'):
    """
    Compute PSNR for each sample in a batch.
    Arguments:
      pred, gt: Tensors of shape [B, C, H, W].
      data_range: the max value if your data is in [0, data_range].
      reduction: 'none' -> return list of PSNRs, 'mean' -> return average PSNR.

    Returns: list of floats or a single float (if reduction=='mean').
    """
    # 1) Shape checks
    if pred.shape != gt.shape:
        raise ValueError(f"Shapes of pred{pred.shape} and gt{gt.shape} do not match.")

    B = pred.shape[0]

    # 2) Range check
    pred = _check_and_clip_range(pred, 0.0, data_range)
    gt   = _check_and_clip_range(gt,   0.0, data_range)

    psnr_vals = []
    for i in range(B):
        mse = F.mse_loss(pred[i], gt[i], reduction='mean')
        if mse == 0:
            psnr_vals.append(float('inf'))
        else:
            val = 10.0 * math.log10((data_range**2) / mse.item())
            psnr_vals.append(val)

    if reduction == 'mean':
        return float(np.mean(psnr_vals))
    else:
        return psnr_vals


def compute_ssim_batch(pred, gt, data_range=1.0, reduction='none'):
    """
    Compute SSIM for each sample in a batch.
    Arguments:
      pred, gt: Tensors of shape [B, C, H, W].
      data_range: the max value if your data is in [0, data_range].
      reduction: 'none' -> return list of SSIMs, 'mean' -> return average SSIM.

    Returns: list of floats or a single float (if reduction=='mean').
    """
    if pred.shape != gt.shape:
        raise ValueError(f"Shapes of pred{pred.shape} and gt{gt.shape} do not match.")

    B = pred.shape[0]

    # Range check
    pred = _check_and_clip_range(pred, 0.0, data_range)
    gt   = _check_and_clip_range(gt,   0.0, data_range)

    ssim_vals = []
    for i in range(B):
        val = ssim(pred[i].unsqueeze(0), gt[i].unsqueeze(0), data_range=data_range)
        ssim_vals.append(val.item())

    if reduction == 'mean':
        return float(np.mean(ssim_vals))
    else:
        return ssim_vals


# ====================================================
# Underwater-Specific Metrics: UIQM & UCIQE
# (Now handle multiple images in a batch)
# ====================================================

def _uicm(im):
    """Internal: colorfulness measure for UIQM."""
    r = im[:,:,0]
    g = im[:,:,1]
    b = im[:,:,2]

    rg = r - g
    yb = 0.5*(r + g) - b

    mu_rg = np.mean(rg)
    mu_yb = np.mean(yb)

    sigma_rg = np.std(rg)
    sigma_yb = np.std(yb)

    UICM = np.sqrt((mu_rg**2 + mu_yb**2)) + np.sqrt((sigma_rg**2 + sigma_yb**2))
    return UICM

def _uism(im):
    """Internal: sharpness measure for UIQM."""
    gray = cv2.cvtColor((im*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    UISM = np.std(lap)
    return UISM

def _uiconm(im):
    """Internal: contrast measure for UIQM."""
    hsv = cv2.cvtColor((im*255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    v = hsv[:,:,2]/255.0
    UIConM = np.std(v)
    return UIConM


def compute_uiqm_batch(pred, reduction='none'):
    """
    Compute UIQM for each sample in a batch.
    pred: Tensor [B, C, H, W], in [0,1].
    reduction: 'none' -> list of UIQMs, 'mean' -> average.

    Returns: list of floats or single float.
    """
    B = pred.shape[0]
    # We can check range [0,1].
    pred = _check_and_clip_range(pred, 0.0, 1.0)

    uiqm_vals = []
    for i in range(B):
        # Convert to CPU numpy [H,W,C]
        img = pred[i].permute(1,2,0).cpu().numpy()
        # Check type
        if img.dtype != np.uint8:
            # Already float in [0,1], so we do not need to rescale
            pass

        UICM = _uicm(img)
        UISM = _uism(img)
        UIConM = _uiconm(img)
        # Weighted sum
        UIQM = (0.0282 * UICM) + (0.2953 * UISM) + (3.5753 * UIConM)
        uiqm_vals.append(UIQM)

    if reduction == 'mean':
        return float(np.mean(uiqm_vals))
    else:
        return uiqm_vals


def compute_uciqe_batch(pred, reduction='none'):
    """
    Compute UCIQE for each sample in the batch.
    pred: Tensor [B, C, H, W], in [0,1].
    reduction: 'none' -> list, 'mean' -> average.

    Returns: list of floats or single float.
    """
    B = pred.shape[0]
    pred = _check_and_clip_range(pred, 0.0, 1.0)

    uciqe_vals = []
    for i in range(B):
        img = pred[i].permute(1,2,0).cpu().numpy()

        # Convert to LAB
        lab = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        L = lab[:,:,0].astype(np.float32)
        a = lab[:,:,1].astype(np.float32) - 128.0
        b = lab[:,:,2].astype(np.float32) - 128.0

        # Chroma
        chroma = np.sqrt(a**2 + b**2)
        std_chroma = np.std(chroma)

        # L contrast
        L_norm = L/255.0
        Lmin = L_norm.min()
        Lmax = L_norm.max()
        con_l = (Lmax - Lmin) / (Lmax + Lmin + 1e-8)

        # Saturation
        hsv = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        S = hsv[:,:,1].astype(np.float32)/255.0
        mean_sat = np.mean(S)

        # Weighted sum
        UCIQE = 0.4680*std_chroma + 0.2745*con_l + 0.2576*mean_sat
        uciqe_vals.append(UCIQE)

    if reduction == 'mean':
        return float(np.mean(uciqe_vals))
    else:
        return uciqe_vals
