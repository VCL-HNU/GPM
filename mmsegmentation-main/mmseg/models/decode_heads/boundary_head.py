
# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from mmseg.registry import MODELS
from mmseg.utils import ConfigType, SampleList
from .fcn_head import FCNHead


@MODELS.register_module()
class BGHead(FCNHead):

    def __init__(self, boundary_threshold=0.1, **kwargs):
        super(BGHead, self).__init__(**kwargs)
        self.boundary_threshold = boundary_threshold
        self.register_buffer(
            'laplacian_kernel',
            torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1],
                         dtype=torch.float32,
                         requires_grad=False).reshape((1, 1, 3, 3)))
        self.fusion_kernel = torch.nn.Parameter(
            torch.tensor([[6. / 10], [3. / 10], [1. / 10]],
                         dtype=torch.float32).reshape(1, 3, 1, 1),
            requires_grad=False)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        """Compute Boundary Loss."""
        seg_label = self._stack_batch_gt(batch_data_samples)
        seg_label = seg_label.to(self.laplacian_kernel)

        boundary_targets = F.conv2d(
            seg_label, self.laplacian_kernel, padding=1)
        boundary_targets = boundary_targets.clamp(min=0)
        boundary_targets[boundary_targets > self.boundary_threshold] = 1
        boundary_targets[boundary_targets <= self.boundary_threshold] = 0

        boundary_targets_x2 = F.conv2d(
            seg_label, self.laplacian_kernel, stride=2, padding=1)
        boundary_targets_x2 = boundary_targets_x2.clamp(min=0)
        boundary_targets_x2_up = F.interpolate(
            boundary_targets_x2, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x2_up[
            boundary_targets_x2_up > self.boundary_threshold] = 1
        boundary_targets_x2_up[
            boundary_targets_x2_up <= self.boundary_threshold] = 0

        boundary_targets_x4 = F.conv2d(
            seg_label, self.laplacian_kernel, stride=4, padding=1)
        boundary_targets_x4 = boundary_targets_x4.clamp(min=0)
        boundary_targets_x4_up = F.interpolate(
            boundary_targets_x4, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x4_up[
            boundary_targets_x4_up > self.boundary_threshold] = 1
        boundary_targets_x4_up[
            boundary_targets_x4_up <= self.boundary_threshold] = 0

        boundary_targets_pyramids = torch.stack(
            (boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up), dim=1)
        boundary_targets_pyramids = boundary_targets_pyramids.squeeze(2)
        boundary_targets_pyramid = F.conv2d(boundary_targets_pyramids, self.fusion_kernel)
        boundary_targets_pyramid[
            boundary_targets_pyramid > self.boundary_threshold] = 1
        boundary_targets_pyramid[
            boundary_targets_pyramid <= self.boundary_threshold] = 0

        seg_label = boundary_targets_pyramid.long()
        batch_data_samples = copy.deepcopy(batch_data_samples)
        for label, data_sample in zip(seg_label, batch_data_samples):
            data_sample.gt_sem_seg.data = label.squeeze(0)

        return super().loss_by_feat(seg_logits, batch_data_samples)
