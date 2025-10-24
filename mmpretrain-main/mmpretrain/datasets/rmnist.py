# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from mmengine.fileio import (BaseStorageBackend, get_file_backend,
                             list_from_file)
from mmengine.logging import MMLogger

from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset
from .custom import CustomDataset


@DATASETS.register_module()
class RMnist(CustomDataset):
    """
    Args:
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        data_prefix (str | dict): Prefix for the data. Defaults to ''.
        lazy_init (bool): Whether to load annotation during instantiation.
            In some cases, such as visualization, only the meta information of
            the dataset is needed, which is not necessary to load annotation
            file. ``Basedataset`` can skip load annotations to save time by set
            ``lazy_init=False``. Defaults to False.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

    def __init__(self,
                 data_root: str = '',
                 data_prefix: Union[str, dict] = '',
                 lazy_init: bool = False,
                 **kwargs):

        self.exclusions = ('_mask.png', '_masked_gray.png',)
        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            ann_file='',
            with_label=True,
            extensions=('.png',),
            metainfo={'classes': tuple(str(c) for c in range(9))},
            lazy_init=lazy_init,
            **kwargs)

    def load_data_list(self):
        """Load image paths and gt_labels."""
        samples = self._find_samples()

        # Pre-build file backend to prevent verbose file backend inference.
        backend = get_file_backend(self.img_prefix, enable_singleton=True)
        data_list = []
        for sample in samples:
            filename, gt_label = sample
            img_path = backend.join_path(self.img_prefix, filename)
            seg_map_path = img_path[:-4] + '_mask.png'
            info = {'img_path': img_path, 'seg_map_path': seg_map_path, 'gt_label': int(gt_label)}
            data_list.append(info)
        return data_list

    def is_valid_file(self, filename: str) -> bool:
        """Check if a file is a valid sample."""
        return filename.lower().endswith(self.extensions) and not filename.lower().endswith(self.exclusions)
