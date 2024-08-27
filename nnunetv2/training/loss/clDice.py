from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.utilities.helpers import softmax_helper_dim1
from nnunetv2.training.loss.skeletonize import Skeletonize


'''
This code is modified from the clDice repository: https://github.com/dmitrysarov/clDice/blob/master/dice_helpers.py

the original clDice paper can be found here: https://arxiv.org/abs/2003.07311  https://doi.org/10.48550/arXiv.2003.07311
'''


def norm_intersection(center_line, vessel):
    '''
    inputs shape  (batch, channel, depth, height, width)
    intersection formalized by first ares
    x - suppose to be centerline of vessel (pred or gt) and y - is vessel (pred or gt)
    '''
    smooth = 1.
    if center_line.is_contiguous():
        clf = center_line.view(*center_line.shape[:2], -1)
    else:
        clf = center_line.reshape(*center_line.shape[:2], -1)
    if vessel.is_contiguous():
        vf = vessel.view(*vessel.shape[:2], -1)
    else:
        vf = vessel.reshape(*vessel.shape[:2], -1)
    intersection = (clf * vf).sum(-1)
    return (intersection + smooth) / (clf.sum(-1) + smooth)


# probably use from nnunetv2.utilities.helpers import softmax_helper_dim1 for nonlinearity
class clDice(torch.nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False,
                 do_bg: bool = True, smooth: float = 1e-5,
                 ddp: bool = False, clip_tp: float = None, **kwargs):

        super(clDice, self).__init__()
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.skeletonize = Skeletonize()
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.clip_tp = clip_tp
        if ddp:
            raise NotImplementedError("ddp is not implemented for centerline dice")

    def skeletonize_all_channels(self, x):
        if x.shape[1] != 1:
            channels = torch.split(x, 1, dim=1)
            skeletons = [self.skeletonize(c) for c in channels]
            return torch.cat(skeletons, dim=1)
        else:
            return self.skeletonize(x)

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        dc = self._get_centerline_dice_coefficent(x, y, axes, loss_mask, square=False)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc

    def _get_centerline_dice_coefficent(self, net_output, gt, axes=None, mask=None,
                                        square=False):
        if axes is None:
            axes = list(range(2, net_output.ndim))

        with torch.no_grad():
            if net_output.ndim != gt.ndim:
                gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

            if net_output.shape == gt.shape:
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                y_onehot = torch.zeros(net_output.shape,
                                       device=net_output.device,
                                       dtype=torch.float32)
                y_onehot.scatter_(1, gt.long(), 1)

            target_skeleton = self.skeletonize_all_channels(y_onehot)

        pred_skeleton = self.skeletonize_all_channels(net_output)

        iflat = norm_intersection(pred_skeleton, y_onehot)
        tflat = norm_intersection(target_skeleton, net_output)
        intersection = iflat * tflat + self.smooth
        return (2. * intersection) / (iflat + tflat + self.smooth)