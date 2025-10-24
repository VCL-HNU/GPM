# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class Endovis2018Dataset(BaseSegDataset):
    """Endovis2018 dataset.
    """
    METAINFO = dict(
        classes=("background-tissue", "instrument-shaft", "instrument-clasper",
               "instrument-wrist", "kidney-parenchyma", "covered-kidney", "thread",
               "clamps", "suturing-needle", "suction-instrument", "small-intestine", "US probe"),
        palette=[(0, 0, 0), (0, 255, 0), (0, 255, 255), (125, 255, 12), (255, 55, 0), (24, 55, 125),
               (187, 155, 25), (0, 255, 125), (255, 255, 125), (123, 15, 175), (124, 155, 5), (12, 255, 141)])

    def __init__(self,
                 img_suffix='_leftImg8bit.png',
                 seg_map_suffix='_gtFine_labelIds.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
