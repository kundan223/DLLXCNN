#File: models/CCLNet/HRNet.py


from models.CCLNet.BaseNet import BaseNet


from models.CCLNet.Public.net.empty_branch import EmptyBranch
from models.CCLNet.Public.net.empty_fm import EmptyFM
# from models.CCLNet.ScaterringBranch.ScatteringBranch import HRBranch
from models.CCLNet.ScaterringBranch.uw_edge_model import HRBranch


class HRNet(BaseNet):
    def __init__(self, opt):
        super(HRNet, self).__init__()

        self.ccBranch = EmptyBranch()
        self.hfBranch = HRBranch()
        self.mfaModule = EmptyFM()
        # self.ccBranch = self.init_net(self.ccBranch, opt.init_type, opt.init_gain, opt.gpu_ids)
        # self.hfBranch = self.init_net(self.hfBranch, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.init_net(self, opt.init_type, opt.init_gain, opt.gpu_ids)
        pass

    def forward(self, input):

        featrueCC, colorCorrect = self.ccBranch(input)

        featrueHR, hazeRemoval = self.hfBranch(input)

        mixedFeature = {"input": input,
                        "featrueCC": featrueCC,
                        "featrueHR": featrueHR}
        enc = self.mfaModule(mixedFeature)

        return hazeRemoval, colorCorrect, enc
