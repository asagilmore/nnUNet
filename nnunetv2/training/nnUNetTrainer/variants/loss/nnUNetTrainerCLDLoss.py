from nnunetv2.training.nnUNetTrainer.nnUNetTrainer.network_architechture.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
from nnunetv2.training.loss.clDice import clDice
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from torch import nn

class combinedLossCLD(nn.Module):
    def __init__(self, dc_ce_weight=1, cl_weight=1):
        super(combinedLossCLD, self).__init__()
        self.dc_ce = DC_and_CE_loss({'batch_dice': False, 'smooth': 1e-5, 'do_bg': False})
        self.cl = clDice()

        # norm so weights sum to 1
        self.dc_ce_weight = dc_ce_weight / (dc_ce_weight + cl_weight)
        self.cl_weight = cl_weight / (dc_ce_weight + cl_weight)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        dc_ce_loss = self.dc_ce(net_output, target)
        cl_loss = self.cl(net_output, target)

        return self.dc_ce_weight * dc_ce_loss + self.cl_weight * cl_loss


class nnUNetTrainerCLDLoss(nnUNetTrainerNoDeepSupervision):
    def _build_loss_clDice(self):
        if self.label_manager.has_regions:
            raise NotImplementedError("clDice loss is not implemented for sparse segmentation")
        else:
            self.loss = combinedLossCLD(dc_ce_weight=1, cl_weight=1)

    def __init__(self, *args, **kwargs):
        self._build_loss = self._build_loss_clDice
        super(nnUNetTrainerCLDLoss, self).__init__(*args, **kwargs)
