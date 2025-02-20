
# File: models/CCLNet/ScaterringBranch/ScaterringBranch.py

import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

########################################################################
# 1. SCEdgeDetectionModule
########################################################################
class SCEdgeDetectionModule(nn.Module):
    """
    Edge Detection Module with improved numerical stability and weight normalization.
    """
    def __init__(self, channels):
        super(SCEdgeDetectionModule, self).__init__()

        # Depthwise conv with weight normalization
        self.cdc = nn.utils.weight_norm(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        )
        self.hdc = nn.utils.weight_norm(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        )
        self.vdc = nn.utils.weight_norm(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        )

        # Fuse the 3 edge maps
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1),
            nn.InstanceNorm2d(channels),
            nn.GELU()
        )

        # Init custom kernels
        self._init_edge_kernels()

    def _init_edge_kernels(self):
        # Center Difference Kernel
        cdc_kernel = torch.zeros(1, 1, 3, 3)
        cdc_kernel[0, 0, 1, 1] = 1
        cdc_kernel[0, 0, :, :] -= 1/8
        epsilon = 1e-5
        cdc_kernel = cdc_kernel / (cdc_kernel.abs().sum() + epsilon)

        # Sobel-like horizontal
        hdc_kernel = torch.tensor([
            [-1,  0, 1],
            [-2,  0, 2],
            [-1,  0, 1]
        ]).float().view(1, 1, 3, 3) / 8.0

        # Sobel-like vertical
        vdc_kernel = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ]).float().view(1, 1, 3, 3) / 8.0

        self.register_buffer('cdc_kernel', cdc_kernel)
        self.register_buffer('hdc_kernel', hdc_kernel)
        self.register_buffer('vdc_kernel', vdc_kernel)

        # Initialize weights
        self.cdc.weight.data = cdc_kernel.repeat(self.cdc.weight.shape[0], 1, 1, 1)
        self.hdc.weight.data = hdc_kernel.repeat(self.hdc.weight.shape[0], 1, 1, 1)
        self.vdc.weight.data = vdc_kernel.repeat(self.vdc.weight.shape[0], 1, 1, 1)

    def forward(self, x):
        # Gradient checkpoint for memory saving
        cdc_out = torch.utils.checkpoint.checkpoint(self.cdc, x)
        hdc_out = torch.utils.checkpoint.checkpoint(self.hdc, x)
        vdc_out = torch.utils.checkpoint.checkpoint(self.vdc, x)

        # Fuse
        edge_feats = torch.cat([cdc_out, hdc_out, vdc_out], dim=1)
        return self.fusion(edge_feats)


########################################################################
# 2. SCAttention
########################################################################
class SCAttention(nn.Module):
    """
    Attention with channel + spatial + edge enhancement.
    """
    def __init__(self, channels, reduction=8):
        super(SCAttention, self).__init__()

        # Channel attention
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.LayerNorm([channels, 1, 1]),
            nn.Conv2d(channels, channels//reduction, 1),
            nn.GELU(),
            nn.Conv2d(channels//reduction, channels, 1)
        )

        # For normalizing channel gate output
        self.layer_norm = nn.LayerNorm([channels, 1, 1])

        # Spatial attention
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.InstanceNorm2d(1),
            nn.Sigmoid()
        )

        # Edge enhancement
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.InstanceNorm2d(channels),
            nn.GELU()
        )

        # Fuse (dropout for regularization)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.InstanceNorm2d(channels),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Channel attention
        c_att = self.channel_gate(x)           # shape (B, C, 1, 1)
        c_att = self.layer_norm(c_att)         # layernorm
        channel_enhanced = x * c_att.sigmoid() # broadcast along H,W

        # Spatial attention (avg + max across channel dim)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.amax(x, dim=1, keepdim=True)
        s_in = torch.cat([avg_out, max_out], dim=1)
        s_att = self.spatial_gate(s_in)
        spatial_enhanced = x * s_att

        # Edge
        edge_enhanced = self.edge_conv(x)

        # Combine channel & spatial
        combined = torch.cat([channel_enhanced, spatial_enhanced], dim=1)
        fused = self.fusion(combined) + edge_enhanced
        return fused


########################################################################
# 3. SCDecoderBlock
########################################################################
class SCDecoderBlock(nn.Module):
    """
    Decoder block with upsampling, skip-connection, attention, and final conv.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super(SCDecoderBlock, self).__init__()

        self.skip_norm = nn.InstanceNorm2d(skip_channels)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.GELU()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, in_channels + skip_channels, kernel_size=1),
            nn.InstanceNorm2d(in_channels + skip_channels),
            nn.GELU()
        )

        self.attention = SCAttention(in_channels + skip_channels)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x, skip):
        skip = self.skip_norm(skip)
        x = torch.utils.checkpoint.checkpoint(self.upsample, x)

        # Merge skip
        x = torch.cat([x, skip], dim=1)

        # Refine
        x = self.refine(x)

        # Attention
        x = self.attention(x)

        # Final conv
        return self.conv(x)


########################################################################
# 4. SCEncoderBlock
########################################################################
class SCEncoderBlock(nn.Module):
    """
    Encoder block with edge detection + attention + conv.
    """
    def __init__(self, in_channels, out_channels):
        super(SCEncoderBlock, self).__init__()

        # First conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.GELU()
        )

        # Edge detection + attention
        self.edge_detect = SCEdgeDetectionModule(out_channels)
        self.attention   = SCAttention(out_channels)

        # Second conv
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.GELU()
        )

        # Skip connection if channels differ
        if in_channels != out_channels:
            self.skip_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.InstanceNorm2d(out_channels)
            )
        else:
            self.skip_conv = nn.Identity()

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = self.conv1(x)

        # Edge + attention (checkpointing)
        e_feats = torch.utils.checkpoint.checkpoint(self.edge_detect, out)
        a_feats = torch.utils.checkpoint.checkpoint(self.attention, out)
        out = out + e_feats + a_feats

        out = self.dropout(out)

        # Second conv
        out = self.conv2(out)

        # Residual
        skip = self.skip_conv(x)
        out = out + skip

        return out


########################################################################
# 5. SCBackbone
########################################################################
class SCBackbone(nn.Module):
    """
    Full Enhanced HRNet-like backbone with 3 encoders, bottleneck, 3 decoders.
    We produce a final image with Tanh, plus an intermediate feature (featrueHR).
    """
    def __init__(self, in_ch=3, base_ch=64):
        super(SCBackbone, self).__init__()

        self.init_conv = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=7, padding=3),
            nn.InstanceNorm2d(base_ch),
            nn.GELU(),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_ch),
            nn.GELU()
        )

        # Encoders
        self.encoder1 = SCEncoderBlock(base_ch, base_ch)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), nn.Dropout(0.1))

        self.encoder2 = SCEncoderBlock(base_ch, base_ch * 2)
        self.down2 = nn.Sequential(nn.MaxPool2d(2), nn.Dropout(0.1))

        self.encoder3 = SCEncoderBlock(base_ch * 2, base_ch * 4)
        self.down3 = nn.Sequential(nn.MaxPool2d(2), nn.Dropout(0.1))

        # Bottleneck
        self.bottleneck = nn.Sequential(
            SCEncoderBlock(base_ch * 4, base_ch * 8),
            SCAttention(base_ch * 8),
            nn.Dropout(0.2)
        )

        # Decoders
        self.decoder3 = SCDecoderBlock(base_ch * 8, base_ch * 4, base_ch * 4)
        self.decoder2 = SCDecoderBlock(base_ch * 4, base_ch * 2, base_ch * 2)
        self.decoder1 = SCDecoderBlock(base_ch * 2, base_ch, base_ch)

        # Final
        self.final = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_ch),
            nn.GELU(),
            nn.Conv2d(base_ch, in_ch, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Return: (featrueHR, finalImage)
           - featrueHR = final features (we choose the last decoder's feature)
           - finalImage = Tanh-ed image
        """
        # ============ Encoder path ============
        x0 = self.init_conv(x)

        s1 = self.encoder1(x0)
        x1 = self.down1(s1)

        s2 = self.encoder2(x1)
        x2 = self.down2(s2)

        s3 = self.encoder3(x2)
        x3 = self.down3(s3)

        # bottleneck
        b = torch.utils.checkpoint.checkpoint(self.bottleneck, x3)

        # ============ Decoder path ============
        d3 = self.decoder3(b, s3)
        d2 = self.decoder2(d3, s2)
        d1 = self.decoder1(d2, s1)

        # final output
        out = self.final(d1)
        # print(f"############ out and d1#########: {out.dtype}, {d1.dtype}")
        # We define featrueHR = d1, the last decoder's feature map prior to final conv
        return d1, out


########################################################################
# 6. HRBranch
########################################################################
class HRBranch(nn.Module):
    """
    Upgraded 'ScaterringBranch' that uses our new SCBackbone
    for advanced haze removal + scattering correction.

    The forward(...) MUST return: (featrueHR, hazeRemoval)
    to maintain compatibility with HRNet.py usage:
        featrueHR, hazeRemoval = self.hfBranch(input)
    """
    def __init__(self, in_channels=3, base_channels=64):
        super(HRBranch, self).__init__()
        # Instantiating our new backbone
        self.backbone = SCBackbone(in_ch=in_channels, base_ch=base_channels)

    def forward(self, input):
        """
        input: (B, 3, H, W)  # if you're using RGB
        returns: (featrueHR, hazeRemoval)
        """
        featrueHR, hazeRemoval = self.backbone(input)
        return featrueHR, hazeRemoval