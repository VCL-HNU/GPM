# Copyright (c) OpenMMLab. All rights reserved.
import os

import numpy as np

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class CaDISDataset(BaseSegDataset):
    """CaDIS dataset.
    """

    def __init__(self,
                 data_root,
                 img_suffix='_leftImg8bit.png',
                 seg_map_suffix='_gtFine_labelIds.png',
                 **kwargs) -> None:
        dataset_info = np.load(os.path.join(data_root, 'data.npy'), allow_pickle=True).item()
        metainfo = dict(classes=dataset_info['CLASSES'], palette=dataset_info['PALETTE'])
        super().__init__(
            data_root=data_root,
            metainfo=metainfo,
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs)
