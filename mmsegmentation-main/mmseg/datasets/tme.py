# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class TMEDataset(BaseSegDataset):
    """CaDIS dataset.
    """
    METAINFO = dict(
        classes=("Background", "Tissue Space", "Ultrasonic Scalpel", "Bipolar Forceps",
               "Intestinal Forceps", "Clip Applier", "Cutting and Closing Instrument", "Suction Instrument"),
        palette=[(0, 0, 0),
                (128, 128, 128),
                (128, 0, 128),
                (0, 128, 128),
                (128, 0, 0),
                (0, 0, 128),
                (128, 128, 0),
                (0, 128, 0)])

    def __init__(self,
                 img_suffix='_leftImg8bit.png',
                 seg_map_suffix='_gtFine_labelIds.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
