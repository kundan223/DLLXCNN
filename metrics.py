# # File: metrics.py

# import torch
# import torch.nn.functional as F
from pytorch_msssim import ssim
# import math
# import numpy as np
# import cv2

# # ====================================================
# # Utility: Range Checking
# # ====================================================

# def _check_and_clip_range(img, min_val=0.0, max_val=1.0, clip=True):
#     """
#     Ensures 'img' is within [min_val, max_val].
#     If clip=True, we clamp any out-of-range values.
#     Otherwise, we raise an Exception if out-of-range is detected.
#     """
#     img_min = img.min().item()
#     img_max = img.max().item()
#     if img_min < min_val - 1e-8 or img_max > max_val + 1e-8:
#         if clip:
#             img = torch.clamp(img, min_val, max_val)
#         else:
#             raise ValueError(f"Image values out of [{min_val}, {max_val}] range: min={img_min}, max={img_max}")
#     return img


# # ====================================================
# # Standard Image Quality Metrics
# # ====================================================

# def compute_psnr_batch(pred, gt, data_range=1.0, reduction='none'):
#     """
#     Compute PSNR for each sample in a batch.
#     Arguments:
#       pred, gt: Tensors of shape [B, C, H, W].
#       data_range: the max value if your data is in [0, data_range].
#       reduction: 'none' -> return list of PSNRs, 'mean' -> return average PSNR.

#     Returns: list of floats or a single float (if reduction=='mean').
#     """
#     # 1) Shape checks
#     if pred.shape != gt.shape:
#         raise ValueError(f"Shapes of pred{pred.shape} and gt{gt.shape} do not match.")

#     B = pred.shape[0]

#     # 2) Range check
#     pred = _check_and_clip_range(pred, 0.0, data_range)
#     gt   = _check_and_clip_range(gt,   0.0, data_range)

#     psnr_vals = []
#     for i in range(B):
#         mse = F.mse_loss(pred[i], gt[i], reduction='mean')
#         if mse == 0:
#             psnr_vals.append(float('inf'))
#         else:
#             val = 10.0 * math.log10((data_range**2) / mse.item())
#             psnr_vals.append(val)

#     if reduction == 'mean':
#         return float(np.mean(psnr_vals))
#     else:
#         return psnr_vals


# def compute_ssim_batch(pred, gt, data_range=1.0, reduction='none'):
#     """
#     Compute SSIM for each sample in a batch.
#     Arguments:
#       pred, gt: Tensors of shape [B, C, H, W].
#       data_range: the max value if your data is in [0, data_range].
#       reduction: 'none' -> return list of SSIMs, 'mean' -> return average SSIM.

#     Returns: list of floats or a single float (if reduction=='mean').
#     """
#     if pred.shape != gt.shape:
#         raise ValueError(f"Shapes of pred{pred.shape} and gt{gt.shape} do not match.")

#     B = pred.shape[0]

#     # Range check
#     pred = _check_and_clip_range(pred, 0.0, data_range)
#     gt   = _check_and_clip_range(gt,   0.0, data_range)

#     ssim_vals = []
#     for i in range(B):
#         val = ssim(pred[i].unsqueeze(0), gt[i].unsqueeze(0), data_range=data_range)
#         ssim_vals.append(val.item())

#     if reduction == 'mean':
#         return float(np.mean(ssim_vals))
#     else:
#         return ssim_vals


# # ====================================================
# # Underwater-Specific Metrics: UIQM & UCIQE
# # (Now handle multiple images in a batch)
# # ====================================================

# def _uicm(im):
#     """Internal: colorfulness measure for UIQM."""
#     r = im[:,:,0]
#     g = im[:,:,1]
#     b = im[:,:,2]

#     rg = r - g
#     yb = 0.5*(r + g) - b

#     mu_rg = np.mean(rg)
#     mu_yb = np.mean(yb)

#     sigma_rg = np.std(rg)
#     sigma_yb = np.std(yb)

#     UICM = np.sqrt((mu_rg**2 + mu_yb**2)) + np.sqrt((sigma_rg**2 + sigma_yb**2))
#     return UICM

# def _uism(im):
#     """Internal: sharpness measure for UIQM."""
#     gray = cv2.cvtColor((im*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
#     lap = cv2.Laplacian(gray, cv2.CV_64F)
#     UISM = np.std(lap)
#     return UISM

# def _uiconm(im):
#     """Internal: contrast measure for UIQM."""
#     hsv = cv2.cvtColor((im*255).astype(np.uint8), cv2.COLOR_RGB2HSV)
#     v = hsv[:,:,2]/255.0
#     UIConM = np.std(v)
#     return UIConM


# def compute_uiqm_batch(pred, reduction='none'):
#     """
#     Compute UIQM for each sample in a batch.
#     pred: Tensor [B, C, H, W], in [0,1].
#     reduction: 'none' -> list of UIQMs, 'mean' -> average.

#     Returns: list of floats or single float.
#     """
#     B = pred.shape[0]
#     # We can check range [0,1].
#     pred = _check_and_clip_range(pred, 0.0, 1.0)

#     uiqm_vals = []
#     for i in range(B):
#         # Convert to CPU numpy [H,W,C]
#         img = pred[i].permute(1,2,0).cpu().numpy()
#         # Check type
#         if img.dtype != np.uint8:
#             # Already float in [0,1], so we do not need to rescale
#             pass

#         UICM = _uicm(img)
#         UISM = _uism(img)
#         UIConM = _uiconm(img)
#         # Weighted sum
#         UIQM = (0.0282 * UICM) + (0.2953 * UISM) + (3.5753 * UIConM)
#         uiqm_vals.append(UIQM)

#     if reduction == 'mean':
#         return float(np.mean(uiqm_vals))
#     else:
#         return uiqm_vals


# def compute_uciqe_batch(pred, reduction='none'):
#     """
#     Compute UCIQE for each sample in the batch.
#     pred: Tensor [B, C, H, W], in [0,1].
#     reduction: 'none' -> list, 'mean' -> average.

#     Returns: list of floats or single float.
#     """
#     B = pred.shape[0]
#     pred = _check_and_clip_range(pred, 0.0, 1.0)

#     uciqe_vals = []
#     for i in range(B):
#         img = pred[i].permute(1,2,0).cpu().numpy()

#         # Convert to LAB
#         lab = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2LAB)
#         L = lab[:,:,0].astype(np.float32)
#         a = lab[:,:,1].astype(np.float32) - 128.0
#         b = lab[:,:,2].astype(np.float32) - 128.0

#         # Chroma
#         chroma = np.sqrt(a**2 + b**2)
#         std_chroma = np.std(chroma)

#         # L contrast
#         L_norm = L/255.0
#         Lmin = L_norm.min()
#         Lmax = L_norm.max()
#         con_l = (Lmax - Lmin) / (Lmax + Lmin + 1e-8)

#         # Saturation
#         hsv = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2HSV)
#         S = hsv[:,:,1].astype(np.float32)/255.0
#         mean_sat = np.mean(S)

#         # Weighted sum
#         UCIQE = 0.4680*std_chroma + 0.2745*con_l + 0.2576*mean_sat
#         uciqe_vals.append(UCIQE)

#     if reduction == 'mean':
#         return float(np.mean(uciqe_vals))
#     else:
#         return uciqe_vals


import torch
import torch.nn.functional as F
import math
import numpy as np
import cv2

###############################################
# Utility: Range Checking
###############################################
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

###############################################
# Standard Reference-based Metrics (PSNR/SSIM)
###############################################
def compute_psnr_batch(pred, gt, data_range=1.0, reduction='none'):
    """
    pred, gt: [B, C, H, W], in [0, data_range].
    Returns: list of floats or single float if 'mean'.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"Shapes of pred{pred.shape} and gt{gt.shape} do not match.")
    B = pred.shape[0]
    # clamp
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
    return psnr_vals  # you can do np.mean(...) afterwards

def compute_ssim_batch(pred, gt, data_range=1.0, reduction='none'):
    """
    pred, gt: [B, C, H, W], in [0,data_range].
    Using pytorch_msssim or your own ssim code is also possible.
    This function placeholder: uses some library or custom approach.
    For now, let's keep your existing approach if you have it.
    """
    # You can plug in your existing ssim from `pytorch_msssim` or another custom function
    # For example:
    from pytorch_msssim import ssim as ssim_lib
    if pred.shape != gt.shape:
        raise ValueError(f"Shapes of pred{pred.shape} and gt{gt.shape} do not match.")
    B = pred.shape[0]
    # clamp
    pred = _check_and_clip_range(pred, 0.0, data_range)
    gt   = _check_and_clip_range(gt,   0.0, data_range)
    ssim_vals = []
    for i in range(B):
        val = ssim_lib(pred[i].unsqueeze(0), gt[i].unsqueeze(0), data_range=data_range)
        ssim_vals.append(val.item())
    return ssim_vals

###############################################
# Underwater-Specific: UIQM & UCIQE
# (Adopting alpha-trim + PLIP from new snippet)
###############################################

#### 1) alpha-trimmed mean helpers
def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    """
      alpha-trimmed mean.
    """
    x_sorted = np.sort(x)
    K = len(x_sorted)
    T_a_L = math.ceil(alpha_L*K)
    T_a_R = math.floor(alpha_R*K)
    # handle edge cases
    valid_length = (K - T_a_L - T_a_R)
    if valid_length <= 0:
        return np.mean(x_sorted)  # fallback
    s = int(T_a_L)
    e = int(K - T_a_R)
    val = np.sum(x_sorted[s:e])
    return val / valid_length

def s_a(x, mu):
    """
      alpha-trimmed variance measure
    """
    val = 0
    for pixel in x:
        val += (pixel - mu)**2
    return val / len(x)

def _uicm_alpha_trim(img):
    """
    alpha-trim approach for UICM from snippet
    """
    # split channels
    R = img[:,:,0].flatten()
    G = img[:,:,1].flatten()
    B = img[:,:,2].flatten()
    RG = R - G
    YB = 0.5*(R+G) - B

    mu_RG = mu_a(RG)
    mu_YB = mu_a(YB)
    var_RG= s_a(RG, mu_RG)
    var_YB= s_a(YB, mu_YB)

    l = math.sqrt( mu_RG**2 + mu_YB**2 )
    r = math.sqrt( var_RG + var_YB )
    # from snippet:
    val = (-0.0268*l)+(0.1586*r)
    return val


#### 2) Sobel-based sharpness + EME for _uism
import numpy as np
from scipy import ndimage

def sobel_255(x):
    dx = ndimage.sobel(x, 0)
    dy = ndimage.sobel(x, 1)
    mag = np.hypot(dx, dy)
    if np.max(mag) > 1e-8:
        mag *= 255.0 / np.max(mag)
    return mag

def eme(block, eps=1e-8):
    # block is [h,w], we find max/min, then log(max/min)
    max_ = np.max(block)
    min_ = np.min(block)
    if min_ < eps or max_<eps:
        return 0
    return math.log(max_/min_)

def EME_channel(img_gray, window_size=10):
    H, W = img_gray.shape
    k_h = H // window_size
    k_w = W // window_size
    val_sum = 0
    for i in range(k_h):
        for j in range(k_w):
            r0 = i*window_size
            c0 = j*window_size
            block = img_gray[r0:r0+window_size, c0:c0+window_size]
            val_sum += eme(block)
    # weight
    total_blocks = k_h*k_w
    if total_blocks<1:
        return 0
    return (2.0/(k_h*k_w)) * val_sum

def _uism(img):
    """
      Underwater Image Sharpness Measure (alpha snippet style).
      1) sobel each channel
      2) multiply with channel
      3) EME
      4) weighted sum
    """
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    Rs = sobel_255(R)
    Gs = sobel_255(G)
    Bs = sobel_255(B)
    R_edge = (Rs/255.)*R
    G_edge = (Gs/255.)*G
    B_edge = (Bs/255.)*B
    r_eme = EME_channel(R_edge, 10)
    g_eme = EME_channel(G_edge, 10)
    b_eme = EME_channel(B_edge, 10)

    # typical weighting for r,g,b
    lambda_r = 0.299
    lambda_g = 0.587
    lambda_b = 0.144
    uism_val = (lambda_r*r_eme) + (lambda_g*g_eme) + (lambda_b*b_eme)
    return uism_val

#### 3) PLIP-based local contrast for _uiconm
def _uiconm_plip(img, window_size=10):
    """
    block-based local contrast measure with logs, from snippet.
    We can do a simpler version or replicate the code exactly.
    We'll do the simpler replicate:
    """
    H, W, _ = img.shape
    k_w = W // window_size
    k_h = H // window_size
    val_sum = 0
    alpha = 1.0  # snippet uses alpha exponent
    block_count = k_w*k_h
    if block_count<1:
        return 0

    for i in range(k_h):
        for j in range(k_w):
            # block
            r0 = i*window_size
            c0 = j*window_size
            block = img[r0:r0+window_size, c0:c0+window_size, :]
            block_max = np.max(block)
            block_min = np.min(block)
            top = block_max - block_min
            bot = block_max + block_min
            if bot<1e-8 or top<1e-8:
                continue
            ratio = top / (bot+1e-8)
            # snippet does ratio^alpha * log(ratio)
            val_sum += (ratio**alpha)*math.log(ratio+1e-8)

    # weighting
    w = -1.0/(block_count+1e-8)
    return w*val_sum

#### 4) final UIQM function
def getUIQM_snippet(img):
    """
    replicate the snippet's final UIQM:
      c1=0.282, c2=0.2953, c3=3.5753
    """
    c1 = 0.282
    c2 = 0.2953
    c3 = 3.5753

    # alpha trimmed UICM
    uicm_val   = _uicm_alpha_trim(img)
    # sobel EME
    uism_val   = _uism(img)
    # plip local contrast
    uiconm_val = _uiconm_plip(img, 10)

    uiqm_val = (c1*uicm_val) + (c2*uism_val) + (c3*uiconm_val)
    return uiqm_val

###############################################
# Now we define compute_uiqm_batch() that uses above approach
###############################################
def compute_uiqm_batch(pred, reduction='none'):
    """
    pred: [B, C, H, W], in [-1,1] or [0,1].
    We'll clamp to [0,1].
    We'll do the snippet approach for final UIQM.
    """
    B = pred.shape[0]
    # 1) clamp in [0,1]
    pred = (pred+1)/2.0  # if in [-1,1], that is.
    pred = torch.clamp(pred, 0,1)
    pred = _check_and_clip_range(pred, 0.0, 1.0)

    vals = []
    for i in range(B):
        # [C,H,W] -> [H,W,C] on CPU
        img_np = pred[i].permute(1,2,0).cpu().numpy()
        # compute snippet-based UIQM
        val = getUIQM_snippet(img_np)
        vals.append(val)
    return vals

###############################################
# UCIQE remains the same or a snippet-based approach
###############################################

def getUCIQE_snippet(img):
    """
    Compute standard UCIQE measure on an image in [H,W,C], float in [0,1], RGB space.
    Weighted combination of:
      - stdChroma (in LAB)
      - L channel contrast
      - mean saturation
    """
    # Convert float[0,1] -> [0,255] for OpenCV
    lab = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    # Extract L,a,b
    L = lab[:,:,0].astype(np.float32)        # [0..255]
    a = lab[:,:,1].astype(np.float32) - 128. # [-128..127]
    b = lab[:,:,2].astype(np.float32) - 128. # [-128..127]
    # Chroma
    chroma = np.sqrt(a**2 + b**2)
    std_chroma = float(np.std(chroma))

    # L contrast: (Lmax - Lmin)/(Lmax + Lmin)
    L_norm = L/255.0
    Lmin = float(np.min(L_norm))
    Lmax = float(np.max(L_norm))
    # Avoid division by 0 if Lmin+Lmax==0
    if (Lmin + Lmax) < 1e-8:
        con_l = 0.0
    else:
        con_l = (Lmax - Lmin)/(Lmax + Lmin)

    # Saturation from HSV
    hsv = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    S = hsv[:,:,1].astype(np.float32)/255.0
    mean_sat = float(np.mean(S))

    # Weighted sum
    UCIQE = 0.4680*std_chroma + 0.2745*con_l + 0.2576*mean_sat
    return UCIQE


def compute_uciqe_batch(pred, reduction='none'):
    """
    Evaluate UCIQE on a batch of images 'pred' with shape [B, C, H, W], possibly in [-1,1].
    We clamp to [0,1], then compute a real UCIQE.
    Returns a list (length B) of floats if reduction='none'.
    """
    # from . import _check_and_clip_range  # or wherever you defined it

    B = pred.shape[0]
    # If your pred is in [-1,1], shift/clamp to [0,1]
    pred = (pred + 1.0)/2.0
    pred = torch.clamp(pred, 0.0, 1.0)
    pred = _check_and_clip_range(pred, 0.0, 1.0)

    out_vals = []
    for i in range(B):
        # Convert [C,H,W]->[H,W,C] then to CPU numpy
        img_np = pred[i].permute(1,2,0).cpu().numpy()
        val = getUCIQE_snippet(img_np)
        out_vals.append(val)

    return out_vals


