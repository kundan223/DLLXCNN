#file: models/CCLNet/ScaterringLoss/HazeRemovalLossV1_0.py

from collections import OrderedDict

import torch.nn as nn
from models.CCLNet.Public.loss.ssim_loss import SSIMLoss
from models.CCLNet.Public.loss.vgg19cr_loss import ContrastLoss
from models.CCLNet.Public.loss.edge_aware_loss import EdgeAwareLoss


class HRLossV1_0(nn.Module):
    def __init__(self, edge_weight=0.1):
        super(HRLossV1_0, self).__init__()
        self.loss_ssim = SSIMLoss()
        self.loss_cr = ContrastLoss(loss_weight=0.5)
        self.loss_edge = EdgeAwareLoss(loss_weight=edge_weight)
        self.losses = OrderedDict()

        pass

    def forward(self, raw, enc, ref):

        self.lss_ssim = self.loss_ssim(enc, ref)
        self.lss_cr = self.loss_cr(enc, ref, raw)
        self.lss_edge = self.loss_edge(enc, ref)
        self.losses["hr_ssim"] = self.lss_ssim
        self.losses["hr_cr"]   = self.lss_cr

        self.losses["hr_edge"] = self.lss_edge

        total_loss = self.lss_ssim + self.lss_cr + self.lss_edge
        return total_loss

    def get_losses(self):
        return self.losses

