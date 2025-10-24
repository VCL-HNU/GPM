# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch.nn as nn
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import ConfigType, SampleList


@MODELS.register_module()
class MultiDatasetHead(nn.Module):
    """
    This head creates one head for each dataset.
    """

    def __init__(self, num_classes, head_cfg):
        super().__init__()
        assert isinstance(num_classes, list)
        self.num_classes = num_classes
        assert 'loss_decode' in head_cfg
        head_cfg['loss_decode']['reduction'] = 'none'
        self.head_cfg = head_cfg

        dataset_heads = []
        for nc in num_classes:
            head_cfg['num_classes'] = nc
            dataset_heads.append(MODELS.build(head_cfg))
        self.dataset_heads = nn.ModuleList(dataset_heads)
        self.align_corners = self.dataset_heads[0].align_corners
        self.out_channels = self.dataset_heads[0].out_channels

    def forward(self, inputs):
        """Forward function."""
        outputs = [head(inputs) for head in self.dataset_heads]
        return outputs

    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        losses = []
        for head in self.dataset_heads:
            losses.append(head.loss(inputs, batch_data_samples, train_cfg))
        return losses

    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tensor:
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        predicts = []
        for head in self.dataset_heads:
            predicts.append(head.predict(inputs, batch_img_metas, test_cfg))

        return predicts
