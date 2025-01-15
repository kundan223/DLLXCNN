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
        ########################################
        # A) Pretrained checkpoint loading
        ########################################
        # This part is only relevant if we want to use CCNet's pretrained weights
        # and if we do not skip CCNet. We assume you already have a flag 
        # --use_ccnet_pretrain for that. We'll show a quick check:
        if self.isTrain:
            if not opt.skip_ccnet and opt.use_ccnet_pretrain:
                # 1) Load the CCNet checkpoint 
                self.load_networks1(opt.epoch_cc, ['G_CC'])
                # 2) Freeze CCNet so it doesn't train
                self.netG_CC.disable_grad()
                print(">> Loaded and disabled grad for pretrained CCNet.")
            elif not opt.skip_ccnet:
                # Means we do NOT use pretrained => CCNet is random init
                # (We'll train it unless we call disable_grad() manually)
                print(">> CCNet is random-init; will be trained with HRNet.")
            else:
                print(">> Skipping CCNet entirely (no color correction).")

            # Setup loss, optimizer, etc.
            self.loss_G = HRLoss(opt)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), 
                                                lr=opt.lr, 
                                                betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.model_names = ['G']  # we only save HRNet
            self.loss_names = ["all"]
            self.visual_names = ['raw_rgb', 'ref_rgb', 'pred_hr', 'rgb_fcc']
        else:
            # For test mode:
            # If skip_ccnet => we won't load or run CCNet
            # If using ccnet_pretrain => we can load it for inference
            self.model_names = ['G', 'G_CC']
            self.visual_names = ['pred_hr']
            if not opt.skip_ccnet and opt.use_ccnet_pretrain:
                self.load_networks1(opt.epoch_cc, ['G_CC'])
                print(">> Loaded pretrained CCNet for inference (test mode).")

    def set_input(self, input):
        self.input = input
        self.raw = input['raw'].to(self.device)

        if 'ref' in input:
            self.ref = input['ref'].to(self.device)
        else:
            self.ref = None

        self.image_paths = input['raw_paths']

    def forward(self):
        """
        Forward pass:
         1) Possibly skip CCNet (color correction).
         2) If not skipped, pass raw Lab -> netG_CC -> corrected Lab, then Lab->RGB
         3) If skipped, just convert raw from Lab->RGB
         4) Finally pass the resulting RGB to HRNet for haze removal
        """
        # --------------------------------
        # A) Convert raw Lab -> RGB directly if skipping CCNet
        # --------------------------------
        if self.opt.skip_ccnet:
            # We do not call netG_CC. 
            # Instead, do a direct Lab->RGB for the raw
            self.rgb_fcc = self.lab2rgb.labn12p1_to_rgbn12p1(self.raw)
        else:
            # --------------------------------
            # B) Use CCNet (pretrained or random init)
            # --------------------------------
            with torch.no_grad():
                # If not pretrained, ccnet is random => it does some color correction
                # If pretrained and disabled grad, we just do a forward pass
                _, pred_cc_lab, _ = self.netG_CC(self.raw)
            self.rgb_fcc = self.lab2rgb.labn12p1_to_rgbn12p1(pred_cc_lab)

        # Now self.rgb_fcc is our "color corrected" (or raw) RGB
        # Pass that into HRNet
        self.pred_hr, self.pred_cc, self.pred_enc = self.netG(self.rgb_fcc)

        # For training visuals
        if self.isTrain:
            self.raw_rgb = self.lab2rgb.labn12p1_to_rgbn12p1(self.raw)
            if self.ref is not None:
                self.ref_rgb = self.lab2rgb.labn12p1_to_rgbn12p1(self.ref)
            else:
                self.ref_rgb = None

    def compute_loss(self):
        """
        Compute the total loss. This is called in the training loop (train.py).
        Usually HRLoss is something like self.loss_G(...) that depends on
         - raw_rgb
         - ref_rgb
         - pred_hr
         - pred_cc
         - pred_enc
        """
        if self.isTrain:
            self.loss_all = self.loss_G(self.raw_rgb,
                                        self.ref_rgb,
                                        self.pred_hr,
                                        self.pred_cc,
                                        self.pred_enc)
        else:
            self.loss_all = None

    def optimize_parameters(self):
        """
        Typical training step:
         1) forward pass
         2) zero grad
         3) compute loss
         4) backward
         5) optimizer step
        """
        if self.isTrain:
            self.forward()
            self.optimizer_G.zero_grad()
            self.compute_loss()
            self.loss_all.backward()
            self.optimizer_G.step()
            # If we also wanted to train CCNet (when skip_ccnet=False & no pretrain),
            # we could add netG_CC parameters to the same or a separate optimizer.
        else:
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

