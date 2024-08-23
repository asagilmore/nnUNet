import numpy as np
import torch
from differentiable_skeletonize import Skeletonize

'''
This code is modified from the clDice repository: https://github.com/dmitrysarov/clDice/blob/master/dice_helpers.py

the original clDice paper can be found here: https://arxiv.org/abs/2003.07311  https://doi.org/10.48550/arXiv.2003.07311
'''


def dice_loss(pred, target):
    '''
    inputs shape  (batch, channel, depth(optional), height, width).
    calculate dice loss per batch and channel of sample.
    E.g. if batch shape is [64, 1, 128, 128] -> [64, 1]
    '''
    smooth = 1.
    if pred.is_contiguous():
        iflat = pred.view(*pred.shape[:2], -1) #batch, channel, -1
    else:
        iflat = pred.reshape(*pred.shape[:2], -1)

    if target.is_contiguous():
        tflat = target.view(*target.shape[:2], -1)
    else:
        tflat = target.reshape(*target.shape[:2], -1)
    intersection = (iflat * tflat).sum(-1)
    return -((2. * intersection + smooth) /
              (iflat.sum(-1) + tflat.sum(-1) + smooth))


def soft_skeletonize(x, thresh_width=10):
    '''
    Differenciable aproximation of morphological skelitonization operaton
    thresh_width - maximal expected width of vessel
    '''
    for i in range(thresh_width):
        min_pool_x = torch.nn.functional.max_pool2d(x*-1, (3, 3), 1, 1)*-1
        contour = torch.nn.functional.relu(torch.nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1) - min_pool_x)
        x = torch.nn.functional.relu(x - contour)
    return x


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


def soft_cldice_loss(pred, target, target_skeleton=None):
    '''
    inputs shape  (batch, channel, x, y, z).
    calculate clDice loss
    Because pred and target at moment of loss calculation will be a torch tensors
    it is preferable to calculate target_skeleton on the step of batch forming,
    when it will be in numpy array format by means of opencv
    '''
    cl_pred = soft_skeletonize(pred)
    if target_skeleton is None:
        target_skeleton = soft_skeletonize(target)
    iflat = norm_intersection(cl_pred, target)
    tflat = norm_intersection(target_skeleton, pred)
    intersection = iflat * tflat
    return -((2. * intersection) /
              (iflat + tflat))


class clDice(torch.nn.Module):
    def __init__(self, **kwargs):
        super(clDice, self).__init__()
        self.skeletonize = Skeletonize(**kwargs)

    def forward(self, pred, target):
        cl_pred = self.skeletonize(pred)
        target_skeleton = self.skeletonize(target)
        iflat = norm_intersection(cl_pred, target)
        tflat = norm_intersection(target_skeleton, pred)
        intersection = iflat * tflat
        return -((2. * intersection) /
                 (iflat + tflat))
