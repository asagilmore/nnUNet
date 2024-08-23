from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.clDice import clDice
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
import torch


class combinedLoss(torch.nn.Module):
    def __init__(self, loss1: torch.nn.Module, loss2: torch.nn.Module, weight1=1, weight2=1):
        super(combinedLoss, self).__init__()
        self.loss1 = loss1
        self.loss2 = loss2

        # norm so weights sum to 1
        self.weight1 = weight1 / (weight1 + weight2)
        self.weight2 = weight2 / (weight1 + weight2)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        loss1 = self.loss1(net_output, target)
        loss2 = self.loss2(net_output, target)

        return self.weight1 * loss1 + self.weight2 * loss2


class nnUNetTrainerCLDLoss(nnUNetTrainer):

    def _build_loss(self):
        if self.label_manager.has_regions:
            raise NotImplementedError("clDice loss is not implemented for sparse segmentation")
        else:
            dc_ce = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)
            cl = clDice()

            loss = combinedLoss(dc_ce, cl, weight1=1, weight2=1)

        # not gonna try to compile this rn might do later if i feel like it

        return loss

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = False
