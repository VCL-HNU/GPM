
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.functional import adaptive_avg_pool2d, interpolate
try:
    from torch.nn.functional import gaussian_blur
except ImportError:
    from torchvision.transforms.functional import gaussian_blur

from mmengine.registry import MODELS
from mmengine import mm_transfer


def init():
    cfg = mm_transfer.cfg
    cfg.setdefault('sdloss_cfg', dict())
    sdloss_cfg = cfg.sdloss_cfg
    sdloss_cfg.setdefault('after_iter', 0)
    sdloss_cfg.setdefault('warmup_iter', 0)
    sdloss_cfg.setdefault('grad_modify', dict())

    # clip_grad_ratio
    sdloss_cfg.grad_modify.setdefault('clip_grad_ratio', None)
    sdloss_cfg.grad_modify.setdefault('clip_to_zero', False)
    clip_grad_ratio = sdloss_cfg.grad_modify.clip_grad_ratio
    if clip_grad_ratio is not None:
        assert isinstance(clip_grad_ratio, list) and len(clip_grad_ratio) == 2, clip_grad_ratio
        assert 0 <= clip_grad_ratio[0] <= clip_grad_ratio[1] <= 1, clip_grad_ratio

    # optimizer and gradient accumulation
    sdloss_cfg.setdefault('sdl_accumulative_counts', 1)
    mm_transfer.sdl_inner_count = 0
    mm_transfer.accumulated_fd_grads = []
    mm_transfer.accumulated_sd_grads = []

    # invalid
    sdloss_cfg.grad_modify.setdefault('invalid', None)
    invalid = sdloss_cfg.grad_modify.invalid
    if invalid is not None:
        assert isinstance(invalid, list) and len(invalid) == 3, invalid
        value_use = []
        for value in invalid:
            if value == 'inf':
                value_use.append(torch.inf)
            elif value == '-inf':
                value_use.append(-torch.inf)
            elif value == 'nan':
                value_use.append(torch.nan)
            elif value is None or value == 'None' or value == 'none':
                value_use.append(None)
            else:
                assert isinstance(value, (int, float))
                value_use.append(value)
        sdloss_cfg.grad_modify.invalid = value_use


def norm_loss(predict, loss_weight=1., order=1., use_ratio=[0, 1.], reduction=['sum', 'mean']):
    # this loss is to reduce ‘predict’ towards zero.
    # predict: (B, C, ...)
    if loss_weight == 0:
        return torch.tensor(0., requires_grad=True)
    loss = predict.abs().pow(order)
    if use_ratio:
        loss = loss.mean(1)
        loss = loss.flatten(1)
        use_dim = [int(ur * loss.size()[1]) for ur in use_ratio]
        loss = torch.sort(loss, 1)[0]
        loss = loss[:, use_dim[0]:use_dim[1]]
    loss = loss.flatten(1)
    if reduction[0] == 'mean':
        loss = loss.mean(1)
    elif reduction[0] == 'sum':
        loss = loss.sum(1)
    loss = loss.pow(1/order)
    if reduction[1] == 'mean':
        loss = loss.mean(0)
    elif reduction[1] == 'sum':
        loss = loss.sum(0)
    return loss * loss_weight


def get_loss_weight(loss_weight_fun, loss_weight_based):
    warmup_iter = mm_transfer.cfg.sdloss_cfg.warmup_iter
    after_iter = mm_transfer.cfg.sdloss_cfg.after_iter
    epoch_now = mm_transfer.runner.epoch
    iter_now = mm_transfer.runner.iter

    loss_weight = loss_weight_fun(epoch_now) \
        if loss_weight_based == 'epoch' \
        else loss_weight_fun(iter_now)

    if iter_now < after_iter:
        if isinstance(loss_weight, (tuple, list)):
            weight_use = (0 for lw in loss_weight)
        else:
            weight_use = 0
    elif iter_now < after_iter + warmup_iter:
        if isinstance(loss_weight, (tuple, list)):
            weight_use = (lw * (iter_now - after_iter) / warmup_iter for lw in loss_weight)
        else:
            weight_use = loss_weight * (iter_now - after_iter) / warmup_iter
    else:
        weight_use = loss_weight

    if isinstance(weight_use, (tuple, list)):
        return weight_use
    else:
        assert weight_use >= 0
        return weight_use


@MODELS.register_module()
class NormLoss(nn.Module):
    def __init__(self,
                 order=1,
                 reduction=['mean', 'mean'],
                 loss_weight=1.,
                 use_ratio=None,
                 loss_target=0):
        super(NormLoss, self).__init__()
        self.loss_type = 'sd_loss'
        self.loss_name = f'norm_loss({order})'
        self.order = order
        self.loss_target = loss_target

        if isinstance(loss_weight, (int, float)): # single loss_weight
            self.loss_weight_fun = lambda x: loss_weight
            self.loss_weight_based = 'epoch'
        elif isinstance(loss_weight, str): # loss_weight function
            assert 'iter' in loss_weight or 'epoch' in loss_weight, loss_weight
            if 'iter' in loss_weight:
                self.loss_weight_based = 'iter'
            elif 'epoch' in loss_weight:
                self.loss_weight_based = 'epoch'
            self.loss_weight_fun = lambda x: eval(loss_weight.replace(self.loss_weight_based, '').format(x))
        else:
            raise TypeError(loss_weight)

        for _reduction in reduction:
            assert _reduction in ['mean', 'sum'], reduction
        self.reduction = reduction
        if use_ratio is not None:
            assert isinstance(use_ratio, list) and len(use_ratio) == 2, use_ratio
            assert 0 <= use_ratio[0] < use_ratio[1] <= 1, use_ratio
        self.use_ratio = use_ratio
        init()

    def forward(self, input_grad):
        input_grad = input_grad - self.loss_target
        losses = OrderedDict()
        loss_weight = get_loss_weight(self.loss_weight_fun, self.loss_weight_based)
        losses[self.loss_name] = norm_loss(input_grad, loss_weight, self.order, self.use_ratio, self.reduction)
        return losses
