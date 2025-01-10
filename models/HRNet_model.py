#File: models/HRNet_model.py
import torch
from collections import OrderedDict

from .base_model import BaseModel

from .CCLNet.HRNet import HRNet
from .CCLNet.HRLoss import HRLoss

from .CCLNet.CCNet import CCNet
from .CCLNet.Public.util.LAB2RGB_v2 import Lab2RGB

class HRNetModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.isTrain = opt.isTrain

        self.netG = HRNet(opt)

        self.netG_CC = CCNet(opt)
        self.lab2rgb = Lab2RGB()

        if self.isTrain:
            self.load_networks1(opt.epoch_cc, ['G_CC'])
            self.netG_CC.disable_grad()

            self.loss_G = HRLoss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G) # for adjust lr rate in BaseModel

            self.loss_names = ["all"]

            self.model_names = ['G']
            self.visual_names = ['raw_rgb', 'ref_rgb', 'pred_hr', 'rgb_fcc']
        else:  # during test time, load netG
            self.model_names = ['G', 'G_CC']
            self.visual_names = ['pred_hr']

        pass

    def set_input(self, input):
        self.input = input
        self.raw = input['raw'].to(self.device)
        if 'ref' in input:
            self.ref = input['ref'].to(self.device)
        else:
            self.ref = None

        self.image_paths = input['raw_paths']
        pass

    def optimize_parameters(self):
        if self.isTrain:
            self.forward()
            # Removed direct backward and step calls
            # self.optimizer_G.zero_grad()
            # self.__backward()
            # self.optimizer_G.step()
        else:
            pass

    def compute_loss(self):
        """
        Compute the total loss and store it in self.loss_all without calling backward().
        """
        if self.isTrain:
            self.loss_all = self.loss_G(self.raw_rgb, self.ref_rgb, self.pred_hr, self.pred_cc, self.pred_enc)
        else:
            self.loss_all = None

    def forward(self):
        """
        Forward pass for the model.
        """
        _, self.pred_cc_fcc, _ = self.netG_CC(self.raw)
        self.rgb_fcc = self.lab2rgb.labn12p1_to_rgbn12p1(self.pred_cc_fcc)

        self.pred_hr, self.pred_cc, self.pred_enc = self.netG(self.rgb_fcc)

        # Expose pred_hr for metric computations
        self.pred_hr = self.pred_hr  # Ensure this attribute exists

        if self.isTrain:
            self.raw_rgb = self.lab2rgb.labn12p1_to_rgbn12p1(self.raw)
            self.ref_rgb = self.lab2rgb.labn12p1_to_rgbn12p1(self.ref)

        pass

    # Remove the self.loss_all.backward() from __backward(), or remove __backward() completely
    def __backward(self):
        if self.isTrain:
            self.loss_all = self.loss_G(self.raw_rgb, self.ref_rgb, self.pred_hr, self.pred_cc, self.pred_enc)

            loss_hr = self.loss_G.gethrloss()
            _ = self.loss_G.getccloss()
            _ = self.loss_G.getencloss()

            if isinstance(loss_hr, OrderedDict):
                for k, v in loss_hr.items():
                    self.loss_names.append(k)
                    setattr(self, "loss_" + k, v)

        else:
            pass

    def get_current_learning_rate(self):
        return self.optimizer_G.param_groups[0]['lr']

