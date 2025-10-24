import copy
from collections import OrderedDict

import quadprog
import numpy as np
import torch
import torch.nn as nn

from mmengine import dist
from mmengine import mm_transfer
from mmengine.registry import MODELS

from .utils import is_model_wrapper


def get_sdloss_fun():
    if 'sdloss_fun' in mm_transfer:
        return mm_transfer.sdloss_fun
    else:
        losses = mm_transfer.cfg.sdloss_cfg.losses
        sdloss_fun = []
        for loss in losses:
            if not isinstance(loss, nn.Module):
                loss = MODELS.build(loss)
            sdloss_fun.append(loss)
        mm_transfer.sdloss_fun = sdloss_fun
        return sdloss_fun

def calculate_sdloss():
    # check sd_loss after_iter for warm-up
    after_iter = mm_transfer.cfg.sdloss_cfg.after_iter
    iter_now = mm_transfer.runner.iter
    return iter_now >= after_iter

def set_grads(model, grads, grads_unused, mode):
    # assign grads to model parameters that are leaf-params and requires_grad.
    assert mode in ['sum', 'new']
    grads = list(grads)
    all_grads = []
    for gu in grads_unused:
        if gu == 1:
            all_grads.append(None)
        else:
            all_grads.append(grads.pop(0))
    grads = all_grads
    params = model_parameters(model)
    assert len(grads) == len(params), f'when set grads to models, len(grads) ' \
                                      f'should be equal to len(params). But got ' \
                                      f'len(grads) = {len(grads)}, ' \
                                      f'len(params) = {len(params)}.'
    if mode == 'new':
        for param, grad in zip(params, grads):
            if grad is None:
                param.grad = None
                continue
            if param.grad is not None:
                param.grad.data.copy_(grad.detach().data)
            else:
                param.grad = grad.detach()
    elif mode == 'sum':
        for param, grad in zip(params, grads):
            if grad is None:
                continue
            if param.grad is not None:
                param.grad.data.add_(grad.detach().data)
            else:
                param.grad = grad.detach()

def model_parameters(model):
    # get model parameters that are leaf-params and requires_grad.
    if is_model_wrapper(model):
        model = model.module
    return [p for p in model.parameters() if p.requires_grad and p.is_leaf]

def scale_sdloss(sd_loss):
    mixed_precision = mm_transfer.cfg.mixed_precision
    scale_value = mm_transfer.cfg.sdloss_cfg.get('scale_loss', 1) if mixed_precision else 1
    sd_loss = sd_loss * scale_value
    return sd_loss, scale_value

def should_sync_1():
    return mm_transfer.optim_wrappers[0].should_sync()

def should_sync_2():
    should_sync = False
    if should_sync_1():
        # accumulative_counts of sd_loss is based on first accumulative_counts.
        sdloss_cfg = mm_transfer.cfg.sdloss_cfg
        if (mm_transfer.sdl_inner_count + 1) % sdloss_cfg.sdl_accumulative_counts == 0:
            should_sync = True
        if (mm_transfer.optim_wrappers[0]._inner_count + 1) == mm_transfer.optim_wrappers[0]._max_counts:
            should_sync = True
    return should_sync

def sdloss_backward(fd_loss_used, sd_loss, fd_grads, fd_grads_unused, model, scale):
    """
    invalid: None, list. replace invalid values (nan, inf, -inf)
        in grads to the value. Defaults to None.
    """
    grad_modify = mm_transfer.cfg.sdloss_cfg.grad_modify
    invalid = grad_modify.invalid
    clip_grad_ratio = grad_modify.clip_grad_ratio
    clip_to_zero = grad_modify.clip_to_zero
    # get sdloss grads
    sd_loss, scale_value = scale_sdloss(sd_loss)
    params = model_parameters(model)
    grads = torch.autograd.grad(
        outputs=sd_loss,
        inputs=params,
        allow_unused=True,
    )
    sd_grads_unused = [0 for grad in grads]
    grads = [
        grad if grad is not None else torch.zeros_like(param)
        for grad, param in zip(grads, params)
    ]
    sd_grads = []
    nan_masks = []
    for sd_grad in grads:
        sd_grad = sd_grad / scale_value
        nan_mask = torch.isnan(sd_grad)
        nan_masks.append(nan_mask)
        sd_grads.append(sd_grad.detach())
    if not calculate_sdloss():
        sd_grads = [torch.zeros_like(sd_grad) for sd_grad in sd_grads]
    # replace invalid values before gradient_accumulation
    if invalid is not None:
        max_value = torch.finfo(torch.float16).max if mm_transfer.cfg.mixed_precision else torch.finfo(torch.float32).max
        invalid = [value * scale for value in invalid if value is not None]
        if len(invalid) == 1:
            invalid = invalid + [max_value, -max_value]
        for sd_grad in sd_grads:
            torch.nan_to_num(sd_grad, *invalid, out=sd_grad)
    # gradient_accumulation
    if mm_transfer.accumulated_fd_grads == []:
        mm_transfer.accumulated_fd_grads = fd_grads
    else:
        assert len(mm_transfer.accumulated_fd_grads) == len(fd_grads)
        fd_grads = [(g1 + g2).detach() for g1, g2 in zip(mm_transfer.accumulated_fd_grads, fd_grads)]
        mm_transfer.accumulated_fd_grads = fd_grads
    if mm_transfer.accumulated_sd_grads == []:
        mm_transfer.accumulated_sd_grads = sd_grads
    else:
        assert len(mm_transfer.accumulated_sd_grads) == len(sd_grads)
        sd_grads = [(g1 + g2).detach() for g1, g2 in zip(mm_transfer.accumulated_sd_grads, sd_grads)]
        mm_transfer.accumulated_sd_grads = sd_grads
    # gradient update
    if not should_sync_1():
        mm_transfer.optim_wrappers[0].zero_grad()
    # update gradients for fdloss only
    elif not should_sync_2():
        # gather grads from all gpus
        dist.all_reduce_params(fd_grads, op='mean')
        # put the modified grads back
        set_grads(model, fd_grads, fd_grads_unused, 'new')
        # empty grads buffer
        mm_transfer.accumulated_fd_grads = []
    # update all gradients
    else:
        # gather grads from all gpus
        dist.all_reduce_params(fd_grads, op='mean')
        dist.all_reduce_params(sd_grads, op='mean')
        # clip gradients of second derivative losses by ratio.
        if clip_grad_ratio is not None and clip_grad_ratio != [0, 1]:
            all_grads = []
            for sd_grad, nan_mask in zip(sd_grads, nan_masks):
                all_grads.append(sd_grad[~nan_mask])
            all_grads = torch.cat(tuple(all_grads)).abs()
            lenth = len(all_grads)
            if lenth > 0:
                lenth_used = min(100000, lenth)
                all_grads = all_grads[::lenth//lenth_used]
                nan_clip = grad_modify.get('nan_clip', False)
                if nan_clip:
                    clip_grad_ratio[1] = 1 - max(1 - clip_grad_ratio[1] - nan_ratio, 0)
                clip_grad_ratio = torch.tensor(clip_grad_ratio).to(
                    device=all_grads.device,
                    dtype=all_grads.dtype)
                thresholds = torch.quantile(all_grads, clip_grad_ratio)
                for sd_grad in sd_grads:
                    sd_grad[sd_grad.abs()<thresholds[0]] = 0
                    if clip_to_zero:
                        sd_grad[sd_grad<-thresholds[1]] = 0
                        sd_grad[sd_grad>thresholds[1]] = 0
                    else:
                        torch.clamp(sd_grad, -thresholds[1], thresholds[1], out=sd_grad)
        set_grads(model, fd_grads, fd_grads_unused, 'new')
        set_grads(model, sd_grads, sd_grads_unused, 'sum')
        # empty grads buffer
        mm_transfer.accumulated_fd_grads = []
        mm_transfer.accumulated_sd_grads = []
    if should_sync_1():
        mm_transfer.sdl_inner_count += 1


def sdl_calculate_and_backward(fd_loss, inputs, loss_scaler=None):
    # backward of first derivative losses. (二阶loss单独反向传播)
    # loss_scaler is for 'AmpOptimWrapper'.

    scale = 1.
    if mm_transfer.cfg.has_sd_loss:
        assert inputs.requires_grad
    model = mm_transfer.runner.model
    params = model_parameters(model)
    if mm_transfer.temp['fd_loss_used']:
        fd_loss_used = mm_transfer.temp['fd_loss_used']
        assert isinstance(fd_loss_used, torch.Tensor) and fd_loss_used > 0
        if loss_scaler is not None:
            scale = loss_scaler.get_scale()
            fd_loss = loss_scaler.scale(fd_loss)
            fd_loss_used = loss_scaler.scale(fd_loss_used)
        inputs_grad = torch.autograd.grad(
            outputs=fd_loss_used,
            inputs=[inputs],
            create_graph=True,
            allow_unused=True,
        )[0]
        assert inputs_grad is not None
        fd_grads = torch.autograd.grad(
            outputs=fd_loss,
            inputs=params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        fd_grads = list(fd_grads)
    else:
        if loss_scaler is not None:
            scale = loss_scaler.get_scale()
            fd_loss = loss_scaler.scale(fd_loss)
        fd_grads = torch.autograd.grad(
            outputs=fd_loss,
            inputs=[inputs] + params,
            create_graph=True,
            allow_unused=True,
        )
        fd_grads = list(fd_grads)
        assert fd_grads[0] is not None
        inputs_grad = fd_grads.pop(0)
        fd_loss_used = fd_loss

    fd_grads_unused = [0 for fd_grad in fd_grads]
    fd_grads = [
        fd_grad if fd_grad is not None else torch.zeros_like(param)
        for fd_grad, param in zip(fd_grads, params)
    ]

    # calculation of second derivative losses
    sd_losses = []
    sdloss_fun = get_sdloss_fun()
    for loss_fun in sdloss_fun:
        loss = loss_fun(inputs_grad)
        sd_losses.append(loss)
    sd_loss = sum(value for loss in sd_losses for value in loss.values() if isinstance(value, torch.Tensor))

    # backward of first and second derivative losses w.r.t. all model parameters
    sdloss_backward(fd_loss_used, sd_loss, fd_grads, fd_grads_unused, model, scale)

    mm_transfer.sd_loss = sd_loss.detach() / scale
    mm_transfer.sd_losses = OrderedDict(
        (name, value.detach() / scale if isinstance(value, torch.Tensor) else value / scale)
        for loss_fun, loss in zip(sdloss_fun, sd_losses)
        for name, value in loss.items()
    )
